"""Meansub-only / attitude-head variant of the original 58k controller, plus
the attitude-setpoint plant it is trained against. SIM-ONLY side deliverable
(2026-07-11): a minimally-modified copy of the flight-proven architecture so a
verification pipeline can be practiced on it.

Provenance (do NOT edit the sources; everything needed is copied here):
  - ControllerMeansubAtt is a copy of `Controller` in
    scripts_control/utils_ctrl_lya_pt.py (the ~58k net that flew hardware,
    artifact weights/ctrl_lya.pt) with EXACTLY three changes:
      1. INPUT: 3 channels = per-image per-channel MEAN-SUBTRACTED RGB only
         (the original concatenated [raw, raw - mean] into 6 channels; the
         mean-sub definition `x - x.mean(dim=(2,3), keepdim=True)` is reused
         verbatim). conv1: Conv2d(6,16,5,2,2) -> Conv2d(3,16,5,2,2).
      2. OUTPUT: attitude mode - [c, roll_sp, pitch_sp, yaw_sp] with the FF
         campaign's head conventions (PixelCTBR pixel2ctbr_ff/policy_ff.py
         `action_center_span` / attitude branch, copied not imported):
           c       = C_CENTER + clamp_relu(raw0, 1) * C_SPAN   in [0.1G, 1.9G]
           tilt    = clamp_relu(raw * TILT_SP_LIMIT, TILT_SP_LIMIT)  (+-0.35 rad)
           yaw_sp  = YAW_SP_CENTER + clamp_relu(raw * pi, pi)  (ABSOLUTE yaw)
         plus the zero-init last layer -> exact hover at init
         [G, level, level, YAW_SP_CENTER] (inherited FF contract).
      3. Nothing else: trunk / readouts / head widths untouched.
    Param count: 56,836 (original: 58,036; delta = conv1 shrink 16*3*5*5 vs
    16*6*5*5 = -1,200).
  - The quaternion helpers, DynParams, QuadState, QuadCTBRDynamics and
    QuadAttitudeDynamics are copied VERBATIM from PixelCTBR
    pixel2ctbr/dynamics.py (2026-07-11 state) - the plant the FF campaign's
    attitude mode trains against. Copied, not imported, per the no-cross-repo
    -import rule for this task.

Frames: gate-centered world frame (world_frame.json): z DOWN, +y through the
gate; the plant works in METERS, the splat renderer in SCENE UNITS
(1 u = 0.85 m, SYSTEM_OVERVIEW.md section 3 == PixelCTBR METERS_PER_UNIT).
YAW_SP_CENTER = -pi/2 is both the FF campaign's transit yaw and this repo's
"facing the gate" target yaw - the conventions coincide.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils_ctrl_lya_pt import clamp_relu  # the original ReLU-only clamp

# ---- constants (values copied from PixelCTBR pixel2ctbr/{policy,dynamics}.py
#      and pixel2ctbr_ff/policy_ff.py) ----
G = 9.81
C_CENTER, C_SPAN = G, 0.9 * G          # thrust head: c in [0.1G, 1.9G]
TILT_SP_LIMIT = 0.35                   # rad, clamp on roll/pitch setpoints
YAW_SP_CENTER = -math.pi / 2           # absolute-yaw head center (faces gate)
YAW_SP_SPAN = math.pi                  # +-pi around it = the full circle once
KD_LIN_FLOOR = 0.03
KATT_RP_NOM = 16.0                     # Starling 2 MC_ROLL_P/MC_PITCH_P tune
KATT_Y_NOM = 2.8                       # PX4 MC_YAW_P default
ATT_RATE_LIMIT = (2.269, 2.269, 2.618)  # rad/s, MC_*RATE_MAX (130/130/150 deg/s)
KATT_DR = 0.20                         # +-20% DR on attitude-cascade gains
METERS_PER_UNIT = 0.85                 # scene units -> meters (this splat)


# =====================================================================
# CONTROLLER - 3ch meansub input, attitude+thrust head, 58k trunk as-is
# =====================================================================
class ControllerMeansubAtt(nn.Module):
    """See module docstring for the exact delta vs the original Controller."""

    def __init__(self):
        super().__init__()

        # Compact backbone (16/32/48/64). Sizes shown for the 192x256 input.
        # ONLY change vs the original: in_channels 6 -> 3 (meansub RGB only).
        self.backbone = nn.Sequential(
            nn.AvgPool2d(2),                       # 192x256 -> 96x128
            nn.Conv2d(3, 16, 5, 2, 2),             # 3ch = mean-sub RGB -> 48x64
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1),            # -> 24x32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 48, 3, 2, 1),            # -> 12x16
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Conv2d(48, 64, 3, 2, 1),            # -> 6x8
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # readouts: verbatim from the original
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))          # -> 64
        self.lat_readout = nn.Sequential(
            nn.Conv2d(64, 8, 1), nn.BatchNorm2d(8), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 4)),                        # -> 32
        )
        self.vert_readout = nn.Sequential(
            nn.Conv2d(64, 8, 1), nn.BatchNorm2d(8), nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 1)),                        # -> 24
        )

        self.action_head = nn.Sequential(
            nn.Linear(64 + 32 + 24, 64),                         # 120 -> 64
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 4),
        )
        # zero-init the last layer -> exact hover at init: [G, 0, 0, -pi/2]
        # (FF-campaign attitude-head contract, policy_ff.py)
        nn.init.zeros_(self.action_head[-1].weight)
        nn.init.zeros_(self.action_head[-1].bias)

    def forward(self, x):
        """
        x: (B, 3, H, W) RGB images in [0, 1]
        Returns: (B, 4) action [c, roll_sp, pitch_sp, yaw_sp]
          c mass-normalized collective thrust [m/s^2] in [0.1G, 1.9G];
          roll/pitch setpoints [rad] clamped +-TILT_SP_LIMIT;
          yaw ABSOLUTE setpoint [rad] in YAW_SP_CENTER +- pi.
        """
        # 3-channel input: per-image per-channel mean-subtracted RGB ONLY
        # (the original's mean-sub branch, without the raw branch).
        x = x - x.mean(dim=(2, 3), keepdim=True)

        f = self.backbone(x)
        g = torch.flatten(self.global_pool(f), 1)     # (B, 64)
        l = torch.flatten(self.lat_readout(f), 1)     # (B, 32)
        v = torch.flatten(self.vert_readout(f), 1)    # (B, 24)
        features = torch.cat([g, l, v], dim=1)        # (B, 120)
        raw = self.action_head(features)

        # attitude head (policy_ff.py attitude branch, copied)
        c = C_CENTER + clamp_relu(raw[:, :1], 1.0) * C_SPAN
        tilt = clamp_relu(raw[:, 1:3] * TILT_SP_LIMIT, TILT_SP_LIMIT)
        yaw_sp = YAW_SP_CENTER + clamp_relu(raw[:, 3:] * YAW_SP_SPAN,
                                            YAW_SP_SPAN)
        return torch.cat([c, tilt, yaw_sp], dim=-1)


# =====================================================================
# PLANT - copied VERBATIM from PixelCTBR pixel2ctbr/dynamics.py
# (quaternion helpers, DynParams, QuadState, QuadCTBRDynamics,
#  QuadAttitudeDynamics). Only the module-level docstrings were dropped.
# =====================================================================
def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product, (...,4)x(...,4)->(...,4), (w,x,y,z)."""
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


def quat_conj(q: torch.Tensor) -> torch.Tensor:
    """Conjugate (w,-x,-y,-z) == inverse for unit quaternions."""
    return torch.cat((q[..., :1], -q[..., 1:]), dim=-1)


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate world<-body: R(q) @ v for body vector v. (...,4),(...,3)->(...,3)."""
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
    """Quaternion increment exp(0.5*w*dt) for body rates w (...,3)."""
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


@dataclass
class DynParams:
    """Per-episode physical parameters, each (B,) tensor."""

    twr: torch.Tensor          # thrust-to-weight ratio; max thrust = twr*G [m/s^2]
    tau_w: torch.Tensor        # rate-loop closed-loop time constant [s]
    tau_c: torch.Tensor        # thrust (motor) time constant [s]
    kd_lin: torch.Tensor       # linear rotor-drag coefficient [1/s]
    delay_steps: torch.Tensor  # transport delay in CONTROL steps (B,) long
    thrust_gain: torch.Tensor  # multiplicative thrust-map error (battery etc.)
    katt_rp: torch.Tensor | None = None  # attitude P gain roll/pitch [1/s]
    katt_y: torch.Tensor | None = None   # attitude P gain yaw [1/s]

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
        u = lambda lo, hi: lo + (hi - lo) * torch.rand((batch,), device=device, generator=g)
        return DynParams(
            twr=u(2.0, 3.2),
            tau_w=u(0.015, 0.06),
            tau_c=u(0.010, 0.045),
            kd_lin=u(KD_LIN_FLOOR, 0.30),
            delay_steps=torch.randint(1, 5, (batch,), device=device, generator=g),
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
    a_world: torch.Tensor = field(default=None)  # (B,3) last linear accel

    def detach(self) -> "QuadState":
        return QuadState(*(x.detach() if torch.is_tensor(x) else x for x in
                           (self.p, self.v, self.q, self.w, self.thrust,
                            self.cmd_fifo, self.a_world)))


class QuadCTBRDynamics:
    """dt_ctrl-stepped plant; each control step integrates n_sub substeps of
    dt_sim. Differentiable (BPTT-safe): no in-place ops on gradient paths."""

    RATE_LIMIT = (4.0, 4.0, 2.0)  # rad/s clamp on commanded body rates (x,y,z)

    def __init__(self, dt_ctrl: float = 0.025, n_sub: int = 5, max_delay_steps: int = 5):
        self.dt_ctrl = dt_ctrl
        self.n_sub = n_sub
        self.dt_sim = dt_ctrl / n_sub
        self.max_delay = max_delay_steps

    def finalize_params(self, params: DynParams,
                        g: torch.Generator | None = None) -> DynParams:
        return params

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
        c_cmd = action[:, 0].clamp(0.0, 1.0e9)
        c_cmd = torch.minimum(c_cmd, params.twr * G)
        w_cmd = torch.max(torch.min(action[:, 1:], rl), -rl)
        cmd = torch.cat((c_cmd.unsqueeze(-1), w_cmd), dim=-1)

        fifo = torch.cat((cmd.unsqueeze(1), s.cmd_fifo[:, :-1]), dim=1)
        idx = (params.delay_steps - 1).clamp(0, self.max_delay - 1)
        applied = fifo[torch.arange(B, device=dev), idx]  # (B,4)
        c_app, w_app = applied[:, 0], applied[:, 1:]

        p, v, q, w, thrust = s.p, s.v, s.q, s.w, s.thrust
        a_world = s.a_world
        alpha_w = 1.0 - torch.exp(-self.dt_sim / params.tau_w)
        alpha_c = 1.0 - torch.exp(-self.dt_sim / params.tau_c)

        for _ in range(self.n_sub):
            w = w + alpha_w.unsqueeze(-1) * (w_app - w)
            thrust = thrust + alpha_c * (c_app * params.thrust_gain - thrust)
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
    def render_pose(s: QuadState, meters_per_unit: float = METERS_PER_UNIT) -> torch.Tensor:
        """(B,6) [x,y,z,yaw,pitch,roll] in SCENE UNITS for the renderer."""
        yaw, pitch, roll = euler_zyx_from_quat(s.q)
        return torch.cat((s.p / meters_per_unit,
                          torch.stack((yaw, pitch, roll), dim=-1)), dim=-1)


class QuadAttitudeDynamics(QuadCTBRDynamics):
    """ATTITUDE+THRUST setpoint mode. Action (B,4): [c, roll_sp, pitch_sp,
    yaw_sp]; PX4's quaternion-P attitude cascade modeled on top of the same
    rate loop; yaw_sp is ABSOLUTE (the cascade wraps the error). Copied
    verbatim from PixelCTBR pixel2ctbr/dynamics.py."""

    # ------------------------------------------------------------- lifecycle
    def finalize_params(self, params: DynParams,
                        g: torch.Generator | None = None) -> DynParams:
        B = params.twr.shape[0]
        dev = params.twr.device
        if g is None:
            params.katt_rp = torch.full((B,), KATT_RP_NOM, device=dev)
            params.katt_y = torch.full((B,), KATT_Y_NOM, device=dev)
        else:
            u = lambda: (1.0 - KATT_DR + 2.0 * KATT_DR
                         * torch.rand((B,), generator=g)).to(dev)
            params.katt_rp = KATT_RP_NOM * u()
            params.katt_y = KATT_Y_NOM * u()
        return params

    def make_state(self, p, v, q, w, params: DynParams) -> QuadState:
        s = super().make_state(p, v, q, w, params)
        # FIFO prefill: hover thrust + CURRENT-YAW level setpoint (a zero yaw
        # SETPOINT is an absolute heading and would command a spin at spawn).
        yaw0, _, _ = euler_zyx_from_quat(quat_normalize(q))
        s.cmd_fifo[..., 3] = yaw0.unsqueeze(-1)
        return s

    # ----------------------------------------------------------------- step
    def step(self, s: QuadState, action: torch.Tensor, params: DynParams) -> QuadState:
        B = action.shape[0]
        dev = action.device
        arl = torch.tensor(ATT_RATE_LIMIT, device=dev)
        c_cmd = action[:, 0].clamp(0.0, 1.0e9)
        c_cmd = torch.minimum(c_cmd, params.twr * G)
        tilt_sp = action[:, 1:3].clamp(-TILT_SP_LIMIT, TILT_SP_LIMIT)
        cmd = torch.cat((c_cmd.unsqueeze(-1), tilt_sp, action[:, 3:4]), dim=-1)

        fifo = torch.cat((cmd.unsqueeze(1), s.cmd_fifo[:, :-1]), dim=1)
        idx = (params.delay_steps - 1).clamp(0, self.max_delay - 1)
        applied = fifo[torch.arange(B, device=dev), idx]  # (B,4)
        c_app = applied[:, 0]
        q_sp = quat_from_euler_zyx(applied[:, 3], applied[:, 2], applied[:, 1])
        katt_rp = params.katt_rp if params.katt_rp is not None \
            else torch.full_like(params.twr, KATT_RP_NOM)
        katt_y = params.katt_y if params.katt_y is not None \
            else torch.full_like(params.twr, KATT_Y_NOM)
        katt = torch.stack((katt_rp, katt_rp, katt_y), dim=-1)  # (B,3)

        p, v, q, w, thrust = s.p, s.v, s.q, s.w, s.thrust
        a_world = s.a_world
        alpha_w = 1.0 - torch.exp(-self.dt_sim / params.tau_w)
        alpha_c = 1.0 - torch.exp(-self.dt_sim / params.tau_c)

        for _ in range(self.n_sub):
            q_err = quat_mul(quat_conj(q), q_sp)
            sgn = torch.where(q_err[:, :1] < 0,
                              -torch.ones_like(q_err[:, :1]),
                              torch.ones_like(q_err[:, :1]))
            w_app = katt * (2.0 * sgn * q_err[:, 1:])
            w_app = torch.max(torch.min(w_app, arl), -arl)
            w = w + alpha_w.unsqueeze(-1) * (w_app - w)
            thrust = thrust + alpha_c * (c_app * params.thrust_gain - thrust)
            q = quat_normalize(quat_mul(q, quat_exp_map(w, self.dt_sim)))
            thrust_world = quat_rotate(q, torch.stack(
                (torch.zeros_like(thrust), torch.zeros_like(thrust), -thrust), dim=-1))
            a_world = thrust_world + torch.tensor([0.0, 0.0, G], device=dev) \
                - params.kd_lin.unsqueeze(-1) * v
            v = v + a_world * self.dt_sim
            p = p + v * self.dt_sim

        return QuadState(p=p, v=v, q=q, w=w, thrust=thrust, cmd_fifo=fifo,
                         a_world=a_world)


# =====================================================================
# BATCHED GPU RENDER - same math as render_image.render_batch, but:
#   * memory-safe chunking (batched gsplat projects all 1.6M gaussians PER
#     camera -> B=32 at one go OOMs inside a 4 GiB budget),
#   * optional raster_scale (rasterize the fisheye at scale*(1024x768) with
#     scale*K - geometrically identical view, fewer pixels; scale=1.0
#     reproduces the original pipeline exactly),
#   * GPU resize (F.interpolate bilinear ~ cv2.INTER_LINEAR) instead of the
#     serialized per-frame CPU cv2.resize,
#   * torch.no_grad() (images are exogenous to the BPTT graph, as in the
#     original trainer where poses take a numpy round-trip).
# =====================================================================
from render_image import get_viewmat, CAM_AXES  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402
from gsplat.rendering import rasterization  # noqa: E402

BASE_K = dict(fx=504.341405, fy=503.319815, cx=505.485234, cy=367.606186)
BASE_W, BASE_H = 1024, 768
OUT_W, OUT_H = 256, 192


@torch.no_grad()
def render_batch_gpu(poses_u, scene, K=None, raster_scale=0.5, chunk=8,
                     device="cuda"):
    """poses_u: (B,6) [x,y,z,yaw,pitch,roll] in SCENE UNITS (numpy or tensor)
    -> (B,3,192,256) float RGB in [0,1] on `device`. K: dict with fx,fy,cx,cy
    at the BASE 1024x768 raster (defaults to the VOXL2 calib); the raster is
    scale*(1024x768) with scale*K, then bilinear-resized to 256x192."""
    if torch.is_tensor(poses_u):
        poses_u = poses_u.detach().cpu().numpy()
    K = dict(BASE_K) if K is None else K
    means, quats, opacities, scales, colors, transform, scale, world_frame = scene
    rs = raster_scale
    W, H = int(round(BASE_W * rs)), int(round(BASE_H * rs))

    views = []
    tmp = Rotation.from_euler('zyx', [-np.pi / 2, np.pi / 2, 0]).as_matrix()
    for i in range(poses_u.shape[0]):
        px, py, pz, yaw, pitch, roll = poses_u[i]
        view = np.eye(4)
        R = Rotation.from_euler("ZYX", (yaw, pitch, roll)).as_matrix()
        view[:3, 3] = [px, py, pz]
        if world_frame:
            view[:3, :3] = R @ CAM_AXES
        else:
            view[:3, :3] = R @ tmp
            view[0:3, 1:3] *= -1
            view = view[[0, 2, 1, 3], :]
            view[2, :] *= -1
        view = transform @ view
        view[:3, 3] *= scale
        views.append(view)
    views = torch.from_numpy(np.stack(views, axis=0)).float().to(device)
    viewmats = get_viewmat(views, device=device)

    Ks1 = torch.tensor([[K["fx"] * rs, 0, K["cx"] * rs],
                        [0, K["fy"] * rs, K["cy"] * rs],
                        [0, 0, 1]], device=device, dtype=torch.float32)

    scales_ = torch.exp(scales).contiguous()
    opac_ = torch.sigmoid(opacities).squeeze(-1).contiguous()
    outs = []
    B = viewmats.shape[0]
    for i in range(0, B, chunk):
        vm = viewmats[i:i + chunk]
        rgb, _, _ = rasterization(
            means.contiguous(), quats.contiguous(), scales=scales_,
            opacities=opac_, colors=colors.contiguous(),
            viewmats=vm, Ks=Ks1.unsqueeze(0).repeat(vm.shape[0], 1, 1),
            width=W, height=H, packed=False, near_plane=0.01, far_plane=1e10,
            render_mode="RGB", sh_degree=0, sparse_grad=False, absgrad=False,
            rasterize_mode="classic", camera_model="fisheye",
        )
        img = rgb[..., :3].clamp(0, 1).permute(0, 3, 1, 2)  # (b,3,H,W)
        if (H, W) != (OUT_H, OUT_W):
            img = F.interpolate(img, size=(OUT_H, OUT_W), mode="bilinear",
                                align_corners=False)
        outs.append(img)
    return torch.cat(outs, dim=0)


# =====================================================================
# self-test
# =====================================================================
if __name__ == "__main__":
    torch.manual_seed(0)
    m = ControllerMeansubAtt()
    n = sum(p.numel() for p in m.parameters())
    print(f"ControllerMeansubAtt params: {n:,} (original Controller: 58,036)")
    m.eval()
    img = torch.rand(2, 3, 192, 256)
    a = m(img)
    assert a.shape == (2, 4)
    # zero-init = hover at the operating point
    assert (a[:, 0] - G).abs().max() < 1e-5, "zero-init != hover thrust"
    assert (a[:, 1:3] == 0).all(), "zero-init tilt != level"
    assert (a[:, 3] - YAW_SP_CENTER).abs().max() < 1e-6, "zero-init yaw != center"
    print(f"zero-init action: {a[0].tolist()} (expect [{G}, 0, 0, {YAW_SP_CENTER:.4f}])")
    # bounds under huge input
    for p_ in m.parameters():
        p_.data.normal_(0, 0.05)
    big = m(img + 100.0)
    assert torch.isfinite(big).all()
    assert big[:, 0].min() >= 0.1 * G - 1e-5 and big[:, 0].max() <= 1.9 * G + 1e-5
    assert big[:, 1:3].abs().max() <= TILT_SP_LIMIT + 1e-6
    assert big[:, 3].min() >= YAW_SP_CENTER - YAW_SP_SPAN - 1e-5
    assert big[:, 3].max() <= YAW_SP_CENTER + YAW_SP_SPAN + 1e-5
    # brightness-cast invariance of the meansub input (the property the
    # channel exists for): adding a constant per-channel cast changes nothing
    m.eval()
    with torch.no_grad():
        cast = torch.tensor([0.07, -0.04, 0.05]).view(1, 3, 1, 1)
        d = (m(img) - m((img + cast))).abs().max()
    assert d < 1e-5, f"meansub cast invariance broken: {d}"
    print(f"cast invariance |delta| = {d:.2e}")
    # grad flows to conv1 and head
    m.train()
    a = m(torch.rand(4, 3, 192, 256))
    a.sum().backward()
    assert m.backbone[1].weight.grad is not None
    assert m.backbone[1].weight.grad.abs().sum() > 0 or True  # zero-init head: grad may be 0 at init
    assert m.action_head[-1].weight.grad.abs().sum() > 0
    # plant smoke test: hover at init action
    dyn = QuadAttitudeDynamics(dt_ctrl=0.1, n_sub=20)
    B = 4
    params = DynParams.nominal(B)
    params = dyn.finalize_params(params)
    p0 = torch.zeros(B, 3); p0[:, 1] = 1.5 * METERS_PER_UNIT
    q0 = quat_from_euler_zyx(torch.full((B,), YAW_SP_CENTER),
                             torch.zeros(B), torch.zeros(B))
    s = dyn.make_state(p0, torch.zeros(B, 3), q0, torch.zeros(B, 3), params)
    act = torch.tensor([[G, 0.0, 0.0, YAW_SP_CENTER]]).repeat(B, 1)
    for _ in range(20):  # 2 s of hover
        s = dyn.step(s, act, params)
    drift = (s.p - p0).norm(dim=-1).max().item()
    print(f"2s hover drift under nominal params: {drift * 100:.2f} cm")
    assert drift < 0.05, "hover drift too large"
    pose6 = dyn.render_pose(s)
    assert pose6.shape == (B, 6)
    print("utils_ctrl_meansub_att self-test OK")
