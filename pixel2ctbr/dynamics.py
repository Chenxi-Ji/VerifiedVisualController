"""Batched, differentiable quadrotor rigid-body dynamics with a CTBR interface.

Models the plant a CTBR policy actually talks to: PX4's rate loop + motors are
abstracted as first-order tracking of the commanded body rates / collective
thrust (the standard learned-CTBR sim2real abstraction), plus transport delay,
rotor drag, and per-episode parameter randomization.

Frames & units
--------------
- World = gate-centered frame from the splat twin, in METERS. z points DOWN
  (NED-like), +y through the gate toward the deploy side. Gravity = +G ẑ.
- Body = FRD (x forward, y right, z down). Collective thrust acts along -z_body.
- Scene units (1 u = 0.85 m) exist only at the render boundary, not here.

State (all torch, batch B):
  p (B,3) position [m], v (B,3) velocity [m/s], q (B,4) unit quaternion
  body->world (w,x,y,z), w (B,3) body rates [rad/s], plus actuator internals:
  thrust_state (B,) mass-normalized thrust actually produced [m/s^2],
  cmd_fifo (B, D, 4) delayed commands.

Action (B,4): [c, wx_cmd, wy_cmd, wz_cmd]
  c: mass-normalized collective thrust command in m/s^2, clamped [0, twr*G].
  w_cmd: body-rate setpoints [rad/s], clamped to rate_limit.

All parameters are (B,)-broadcastable tensors so domain randomization is just
sampling them per episode. Values here are engineering placeholders; see
docs/pixel2ctbr/05_implementation_log.md and 04_design.md for provenance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

G = 9.81


# ---------------------------------------------------------------- quaternions
def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product, (…,4)x(…,4)->(…,4), (w,x,y,z)."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        dim=-1,
    )


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate world<-body: R(q) @ v for body vector v. (…,4),(…,3)->(…,3)."""
    qw = q[..., :1]
    qv = q[..., 1:]
    t = 2.0 * torch.linalg.cross(qv, v, dim=-1)
    return v + qw * t + torch.linalg.cross(qv, t, dim=-1)


def quat_rotate_inv(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate body<-world: R(q)^T @ v for world vector v."""
    qw = q[..., :1]
    qv = -q[..., 1:]
    t = 2.0 * torch.linalg.cross(qv, v, dim=-1)
    return v + qw * t + torch.linalg.cross(qv, t, dim=-1)


def quat_exp_map(w: torch.Tensor, dt) -> torch.Tensor:
    """Quaternion increment exp(0.5*w*dt) for body rates w (…,3)."""
    theta = torch.linalg.norm(w, dim=-1, keepdim=True) * (
        dt if not torch.is_tensor(dt) else dt.unsqueeze(-1)
    ) * 0.5
    # sinc for numerical stability at theta -> 0
    half = w * (dt if not torch.is_tensor(dt) else dt.unsqueeze(-1)) * 0.5
    small = theta < 1e-8
    k = torch.where(small, 1.0 - theta * theta / 6.0, torch.sin(theta) / theta.clamp_min(1e-12))
    return torch.cat((torch.cos(theta), k * half), dim=-1)


def quat_normalize(q: torch.Tensor) -> torch.Tensor:
    return q / torch.linalg.norm(q, dim=-1, keepdim=True).clamp_min(1e-12)


def quat_from_euler_zyx(yaw, pitch, roll) -> torch.Tensor:
    """Intrinsic Z-Y-X (yaw, pitch, roll) -> quaternion (w,x,y,z). Matches
    scipy Rotation.from_euler('ZYX', (yaw, pitch, roll)) used by the renderer."""
    cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)
    cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
    cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)
    return torch.stack(
        (
            cy * cp * cr + sy * sp * sr,
            cy * cp * sr - sy * sp * cr,
            cy * sp * cr + sy * cp * sr,
            sy * cp * cr - cy * sp * sr,
        ),
        dim=-1,
    )


def euler_zyx_from_quat(q: torch.Tensor):
    """Quaternion -> (yaw, pitch, roll), intrinsic ZYX. Inverse of the above."""
    w, x, y, z = q.unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    sinp = (2 * (w * y - z * x)).clamp(-1.0, 1.0)
    pitch = torch.asin(sinp)
    roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    return yaw, pitch, roll


# ------------------------------------------------------------------- params
@dataclass
class DynParams:
    """Per-episode physical parameters, each (B,) tensor. Placeholders ⚠ per
    05_implementation_log.md until Starling 2 numbers / system ID land."""

    twr: torch.Tensor          # thrust-to-weight ratio; max thrust = twr*G [m/s^2]
    tau_w: torch.Tensor        # rate-loop closed-loop time constant [s]
    tau_c: torch.Tensor        # thrust (motor) time constant [s]
    kd_lin: torch.Tensor       # linear rotor-drag coefficient [1/s]
    delay_steps: torch.Tensor  # transport delay in CONTROL steps (B,) long
    thrust_gain: torch.Tensor  # multiplicative thrust-map error (battery etc.)

    @staticmethod
    def nominal(batch: int, device="cpu") -> "DynParams":
        t = lambda v: torch.full((batch,), float(v), device=device)
        return DynParams(
            twr=t(2.0), tau_w=t(0.05), tau_c=t(0.03), kd_lin=t(0.1),
            delay_steps=torch.full((batch,), 3, device=device, dtype=torch.long),
            thrust_gain=t(1.0),
        )

    @staticmethod
    def randomized(batch: int, device="cpu", g: torch.Generator | None = None) -> "DynParams":
        """DR ranges ⚠ initial guesses; tighten/widen after system ID."""
        u = lambda lo, hi: lo + (hi - lo) * torch.rand((batch,), device=device, generator=g)
        return DynParams(
            twr=u(1.6, 2.6),
            tau_w=u(0.02, 0.10),
            tau_c=u(0.015, 0.06),
            kd_lin=u(0.0, 0.30),
            delay_steps=torch.randint(2, 5, (batch,), device=device, generator=g),
            thrust_gain=u(0.85, 1.15),
        )


@dataclass
class QuadState:
    p: torch.Tensor            # (B,3) m
    v: torch.Tensor            # (B,3) m/s
    q: torch.Tensor            # (B,4) body->world
    w: torch.Tensor            # (B,3) rad/s actual body rates
    thrust: torch.Tensor       # (B,)  produced mass-normalized thrust m/s^2
    cmd_fifo: torch.Tensor     # (B,D,4) pending delayed commands
    a_world: torch.Tensor = field(default=None)  # (B,3) last linear accel (for IMU)

    def detach(self) -> "QuadState":
        return QuadState(*(x.detach() if torch.is_tensor(x) else x for x in
                           (self.p, self.v, self.q, self.w, self.thrust,
                            self.cmd_fifo, self.a_world)))


class QuadCTBRDynamics:
    """dt_ctrl-stepped plant; each control step integrates n_sub substeps of
    dt_sim. Differentiable (BPTT-safe): no in-place ops on gradient paths."""

    RATE_LIMIT = (4.0, 4.0, 2.0)  # ⚠ rad/s clamp on commanded body rates (x,y,z)

    def __init__(self, dt_ctrl: float = 0.025, n_sub: int = 5, max_delay_steps: int = 5):
        self.dt_ctrl = dt_ctrl
        self.n_sub = n_sub
        self.dt_sim = dt_ctrl / n_sub
        self.max_delay = max_delay_steps

    # ------------------------------------------------------------- lifecycle
    def make_state(self, p, v, q, w, params: DynParams) -> QuadState:
        B = p.shape[0]
        dev = p.device
        hover = torch.full((B,), G, device=dev) / params.thrust_gain
        fifo = torch.zeros((B, self.max_delay, 4), device=dev)
        # pre-fill FIFO with hover commands so t=0 isn't a free-fall artifact
        fifo[..., 0] = hover.unsqueeze(-1)
        return QuadState(p=p, v=v, q=quat_normalize(q), w=w,
                         thrust=hover * params.thrust_gain,
                         cmd_fifo=fifo, a_world=torch.zeros((B, 3), device=dev))

    # ----------------------------------------------------------------- step
    def step(self, s: QuadState, action: torch.Tensor, params: DynParams) -> QuadState:
        """action (B,4) = [c m/s^2, wx, wy, wz rad/s] commanded NOW; the plant
        applies the FIFO-delayed command. Returns new state."""
        B = action.shape[0]
        dev = action.device
        rl = torch.tensor(self.RATE_LIMIT, device=dev)
        c_cmd = action[:, 0].clamp(0.0, 1.0e9)  # upper bound applied via twr below
        c_cmd = torch.minimum(c_cmd, params.twr * G)
        w_cmd = torch.max(torch.min(action[:, 1:], rl), -rl)
        cmd = torch.cat((c_cmd.unsqueeze(-1), w_cmd), dim=-1)

        # FIFO delay (per-sample depth): shift-in cmd, read at delay_steps-1
        fifo = torch.cat((cmd.unsqueeze(1), s.cmd_fifo[:, :-1]), dim=1)
        idx = (params.delay_steps - 1).clamp(0, self.max_delay - 1)
        applied = fifo[torch.arange(B, device=dev), idx]  # (B,4)
        c_app, w_app = applied[:, 0], applied[:, 1:]

        p, v, q, w, thrust = s.p, s.v, s.q, s.w, s.thrust
        a_world = s.a_world
        alpha_w = 1.0 - torch.exp(-self.dt_sim / params.tau_w)
        alpha_c = 1.0 - torch.exp(-self.dt_sim / params.tau_c)

        for _ in range(self.n_sub):
            # actuator lags: rates track command; thrust tracks command*gain
            w = w + alpha_w.unsqueeze(-1) * (w_app - w)
            thrust = thrust + alpha_c * (c_app * params.thrust_gain - thrust)
            # kinematics/dynamics
            q = quat_normalize(quat_mul(q, quat_exp_map(w, self.dt_sim)))
            thrust_world = quat_rotate(q, torch.stack(
                (torch.zeros_like(thrust), torch.zeros_like(thrust), -thrust), dim=-1))
            a_world = thrust_world + torch.tensor([0.0, 0.0, G], device=dev) \
                - params.kd_lin.unsqueeze(-1) * v
            v = v + a_world * self.dt_sim
            p = p + v * self.dt_sim

        return QuadState(p=p, v=v, q=q, w=w, thrust=thrust, cmd_fifo=fifo,
                         a_world=a_world)

    # ------------------------------------------------------------ rendering
    @staticmethod
    def render_pose(s: QuadState, meters_per_unit: float = 0.85) -> torch.Tensor:
        """(B,6) [x,y,z,yaw,pitch,roll] in SCENE UNITS for render_batch()."""
        yaw, pitch, roll = euler_zyx_from_quat(s.q)
        return torch.cat((s.p / meters_per_unit,
                          torch.stack((yaw, pitch, roll), dim=-1)), dim=-1)
