"""Milestone-2 v0: gate TRANSIT task (03_strategy §path-to-gates).

Task: from the hover start box (+y side), fly to and THROUGH the gate
opening, brake to hover at an exit point 0.8 m past the plane. Yaw stays
~-pi/2 (facing along the direction of travel throughout).

Splat-validity note (legacy finding: the gate's -y FACE is washed out and
unlearnable): during approach the camera sees +y-side content (fine); after
passage it faces AWAY from the gate, seeing far-side background captured
from +y viewpoints — mild view extrapolation, acceptable for a short 0.8 m
overshoot, degrading with distance. Hence the exit point is close.

Phase machine (per drone):
  A approach: target = pre-gate waypoint [0, +PRE_Y, 0] (centers the drone
    on the opening before committing)
  B transit : once centered near the pre-gate wp -> target = exit hover
    point [0, -EXIT_Y, 0]; phase latches (no regression on overshoot).
The same GeometricHoverExpert tracks the phase target (teacher + oracle);
HoverEnv's observation/DR machinery is inherited unchanged.

Success: crossed the plane INSIDE the opening (|x|,|z| < 0.30 m at y=0
crossing) AND final within 0.15 m of the exit point, |v|<0.25, no crash.
"""

from __future__ import annotations

import torch

from dynamics import G, euler_zyx_from_quat
from env import EnvConfig, HoverEnv
from render_bridge import METERS_PER_UNIT


class GateTransitEnv(HoverEnv):
    PRE_Y = 0.7          # m: pre-gate alignment waypoint (short runway =
                         # less drift accumulation before the plane)
    EXIT_Y = 0.8         # m: exit hover point past the plane
    CENTER_TOL = 0.12    # m: lateral centering required to commit
    VLAT_TOL = 0.20      # m/s: lateral velocity gate at commit
    YAW_TOL = 0.12       # rad: yaw alignment gate at commit
    NEAR_TOL = 0.40      # m: distance to pre-gate wp to commit
    OPENING = 0.30       # m: half-extent counted as "through the opening"

    def __init__(self, cfg: EnvConfig, **kw):
        super().__init__(cfg, **kw)
        B, dev = cfg.B, cfg.device
        self.wp_pre = torch.tensor([0.0, self.PRE_Y, 0.0], device=dev).expand(B, 3)
        self.wp_exit = torch.tensor([0.0, -self.EXIT_Y, 0.0], device=dev).expand(B, 3)

    # ------------------------------------------------------------- reset
    def reset(self, vel_range=0.5):
        s = super().reset(vel_range)
        B, dev = self.cfg.B, self.cfg.device
        self.phase_b = torch.zeros(B, dtype=torch.bool, device=dev)
        self.crossed_ok = torch.zeros(B, dtype=torch.bool, device=dev)
        self.crossed_bad = torch.zeros(B, dtype=torch.bool, device=dev)
        self.prev_y = self.state.p[:, 1].detach().clone()
        return s

    # -------------------------------------------------- phase-aware target
    @property
    def tgt_p(self):
        # HoverEnv methods (expert call, window losses) read env.tgt_p —
        # making it phase-aware retargets everything consistently
        return torch.where(self.phase_b.unsqueeze(-1), self.wp_exit, self.wp_pre)

    @tgt_p.setter
    def tgt_p(self, v):
        pass  # base-class __init__ assigns a static target; ignored here

    # ------------------------------------------------------------- update
    @torch.no_grad()
    def update_phases(self):
        """Call once per control step (after dyn.step). Latches phase B and
        records plane crossings (through the opening vs outside it)."""
        p = self.state.p.detach()
        v = self.state.v.detach()
        near = (p - self.wp_pre).norm(dim=-1) < self.NEAR_TOL
        centered = (p[:, 0].abs() < self.CENTER_TOL) & \
                   (p[:, 2].abs() < self.CENTER_TOL)
        slow_lat = (v[:, 0].abs() < self.VLAT_TOL) & \
                   (v[:, 2].abs() < self.VLAT_TOL)
        yaw, _, _ = euler_zyx_from_quat(self.state.q)
        aligned = torch.atan2(torch.sin(yaw - self.tgt_yaw),
                              torch.cos(yaw - self.tgt_yaw)).abs() < self.YAW_TOL
        self.phase_b |= near & centered & slow_lat & aligned
        crossing = (self.prev_y > 0) & (p[:, 1] <= 0)
        inside = (p[:, 0].abs() < self.OPENING) & (p[:, 2].abs() < self.OPENING)
        self.crossed_ok |= crossing & inside
        self.crossed_bad |= crossing & ~inside     # rang the frame
        self.prev_y = p[:, 1].clone()

    # ---------------------------------------------------------------- hooks
    def observe(self):
        # called once per control step by every rollout path -> per-step
        # phase/crossing updates without overriding the rollout loops
        self.update_phases()
        return super().observe()

    # ------------------------------------------------------------ rollouts
    def expert_rollout(self, T: int):
        raise NotImplementedError("use transit_expert_rollout")

    @torch.no_grad()
    def transit_expert_rollout(self, T: int):
        """Expert flies the phase machine (state-only; feasibility oracle)."""
        for _ in range(T):
            self.update_phases()
            a = self.expert(self.state, self.tgt_p, self.tgt_yaw)
            self.state = self.dyn.step(self.state, a, self.params)
            self.last_action = a
        self.update_phases()
        return self.transit_metrics()

    # ------------------------------------------------------------- metrics
    @torch.no_grad()
    def transit_metrics(self, err_pos=0.15, err_v=0.25):
        p, v = self.state.p, self.state.v
        e = (p - self.wp_exit).norm(dim=-1)
        crash = (p[:, 2] > 1.2) | ~torch.isfinite(p).all(-1) | self.crossed_bad
        ok = self.crossed_ok & (~crash) & (e < err_pos) & (v.norm(dim=-1) < err_v)
        return {
            "success": ok.float().mean().item(),
            "crossed": self.crossed_ok.float().mean().item(),
            "rang_frame": self.crossed_bad.float().mean().item(),
            "exit_err_med": e.median().item(),
            "phaseB": self.phase_b.float().mean().item(),
            "crash": (crash & ~self.crossed_bad).float().mean().item(),
        }


if __name__ == "__main__":
    # feasibility oracle: can the phase-machine EXPERT transit the gate on
    # the randomized plant? (gate for everything downstream)
    import sys
    torch.manual_seed(0)
    env = GateTransitEnv(EnvConfig(B=256, tilt_dropout=0.0, device="cpu"),
                         renderer=False, image_dr=False)
    env.seed(5)
    env.reset()
    m = env.transit_expert_rollout(int(13.0 / env.cfg.dt_ctrl))
    print("expert transit feasibility (B=256, 13 s):", m)
    sys.exit(0 if m["success"] > 0.97 else 1)
