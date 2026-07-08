"""Milestone-2 v1: N-gate tracks in the FalconGym-edited splat twin.

Generalizes env_transit.py's phase machine to a WAYPOINT LIST over N gates
(gate REAL_GATE — in traversal order — is the real splat gate at the
origin; every other entry is a scene_edit duplicate composited into the
renderer's scene). Per gate i:

  c_i  center [m]; phi_i yaw of the gate about z (0 = original);
  n_i = Rz(phi_i) @ (0,1,0)  approach-side plane normal;
  u_i = Rz(phi_i) @ (1,0,0)  in-plane horizontal axis (z is in-plane too);
  psi_i = -pi/2 + phi_i      drone transit yaw (faces along -n_i).

Phase k in 0..n: k<n targets the pre-gate waypoint wp_k = c_k + PRE_Y*n_k
with yaw psi_k; the COMMIT GATE from env_transit (tuned there 85.9%->100%:
centered <0.12 m, lateral |v|<0.2, yaw <0.12) is evaluated in gate k's own
plane coordinates and latches phase k+1. Phase n targets the exit hover
point EXIT_Y past the last gate.

Crossing detection per gate, per step: signed offset s_k = n_k.(p-c_k)
flipping + -> - counts as a crossing; in-plane offset < OPENING = clean,
< FRAME_R = frame strike, farther = wide miss (no strike flag — but success
requires ALL gates crossed clean). FRAME_R is new vs env_transit: with
several oblique gate planes, a distant crossing of an (infinite) plane is
not a physical frame hit.

Success: every gate crossed inside the opening, zero frame strikes, final
hover within 0.15 m of the exit point, |v|<0.25, no crash.
"""

from __future__ import annotations

import math

import torch

from dynamics import euler_zyx_from_quat, quat_from_euler_zyx
from env import EnvConfig, HoverEnv


class MultiGateEnv(HoverEnv):
    PRE_Y = 0.7          # m: pre-gate commit waypoint distance (env_transit)
    EXIT_Y = 0.8         # m: exit hover point past the LAST gate (splat
                         # validity: content degrades deep past the capture)
    CENTER_TOL = 0.12    # m: in-plane centering required to commit
    VLAT_TOL = 0.20      # m/s: in-plane velocity gate at commit
    YAW_TOL = 0.12       # rad: yaw alignment gate at commit
    NEAR_TOL = 0.40      # m: distance to the pre-gate wp to commit
    OPENING = 0.30       # m: half-extent counted as "through the opening"
    FRAME_R = 0.75       # m: half-extent counted as "hit the frame"
    EVAL_T = 640         # control steps (16 s) for policy transit evals

    # [(center_m, gate_yaw_rad)] in traversal order; entry REAL_GATE MUST be
    # ((0,0,0), 0.0) (the real splat gate — not necessarily passed first)
    GATES: list = []
    REAL_GATE = 0        # index of the real gate within GATES

    # start box in gate-1's APPROACH frame, meters (x across the approach,
    # y along +n_1 measured from the gate-1 plane, z about gate height).
    # Defaults = HoverEnv's native box; envs whose first gate is a duplicate
    # (REAL_GATE > 0) may shrink it to stay on the arena mat.
    START_X = 1.275
    START_Y = (0.425, 2.55)
    START_Z = (-0.425, 0.34)

    def __init__(self, cfg: EnvConfig, renderer=None, image_dr: bool = True):
        rg = self.GATES[self.REAL_GATE]
        assert tuple(rg[0]) == (0.0, 0.0, 0.0) and rg[1] == 0.0
        if renderer is None:
            # composite the duplicated gates into the splat scene once
            import scene_edit as se
            from render_bridge import SplatRenderer
            scene = se.multi_gate_scene(
                [se.gate_pose(c, y) for i, (c, y) in enumerate(self.GATES)
                 if i != self.REAL_GATE])
            renderer = SplatRenderer(
                width=cfg.width, height=cfg.height, device=cfg.device,
                mount_jitter_rad=0.5 * torch.pi / 180, intrinsics_jitter=1.0,
                gray=True, supersample=2, scene=scene)
        super().__init__(cfg, renderer=renderer, image_dr=image_dr)

        dev = cfg.device
        self.n_gates = n = len(self.GATES)
        c = torch.tensor([g[0] for g in self.GATES], dtype=torch.float32)
        phi = torch.tensor([g[1] for g in self.GATES], dtype=torch.float32)
        self.gate_c = c.to(dev)                                    # (n,3)
        self.gate_n = torch.stack(                                 # (n,3)
            (-torch.sin(phi), torch.cos(phi), torch.zeros(n)), -1).to(dev)
        self.gate_u = torch.stack(                                 # (n,3)
            (torch.cos(phi), torch.sin(phi), torch.zeros(n)), -1).to(dev)
        psi = -torch.pi / 2 + phi                                  # transit yaws
        self.wps = torch.cat((                                     # (n+1,3)
            self.gate_c + self.PRE_Y * self.gate_n,
            (self.gate_c[-1] - self.EXIT_Y * self.gate_n[-1]).unsqueeze(0)))
        self.tgt_yaws = torch.cat((psi, psi[-1:])).to(dev)         # (n+1,)
        self.wp_exit = self.wps[-1]

    # -------------------------------------------------- phase-aware targets
    @property
    def tgt_p(self):
        return self.wps[self.phase]                                # (B,3)

    @tgt_p.setter
    def tgt_p(self, v):
        pass  # base-class __init__ assigns a static target; ignored here

    @property
    def tgt_yaw(self):
        return self.tgt_yaws[self.phase]                           # (B,)

    @tgt_yaw.setter
    def tgt_yaw(self, v):
        pass

    # ------------------------------------------------------------- reset
    def reset(self, vel_range=0.5):
        s = super().reset(vel_range)
        B, dev, n = self.cfg.B, self.cfg.device, self.n_gates
        (c1x, c1y, c1z), phi1 = self.GATES[0]
        if phi1 != 0.0 or c1x != 0.0 or c1y != 0.0 or c1z != 0.0:
            # HoverEnv's start box serves a gate at the origin facing +y;
            # when the FIRST gate of the track is elsewhere (REAL_GATE > 0),
            # resample the pose in gate 1's approach frame (START_* box,
            # rotated by phi1 about z, shifted to c_1) so every episode
            # still begins on gate 1's +n side. Legacy tracks (gate 1 = the
            # origin gate) skip this branch and stay bit-exact.
            g = self.g
            u = lambda lo, hi: (lo + (hi - lo) * torch.rand(B, generator=g)).to(dev)
            p = torch.stack((u(-self.START_X, self.START_X),
                             u(*self.START_Y), u(*self.START_Z)), dim=-1)
            cp, sp = math.cos(phi1), math.sin(phi1)
            R = torch.tensor([[cp, -sp, 0.0], [sp, cp, 0.0], [0.0, 0.0, 1.0]],
                             device=dev)
            p = p @ R.T + torch.tensor([c1x, c1y, c1z], device=dev)
            q = quat_from_euler_zyx(self.tgt_yaws[0] + u(-0.6, 0.6),
                                    u(-0.1, 0.1), u(-0.1, 0.1))
            v = torch.stack([u(-vel_range, vel_range) for _ in range(3)], -1)
            self.state = self.dyn.make_state(p, v, q,
                                             torch.zeros(B, 3, device=dev),
                                             self.params)
            s = self.state
        self.phase = torch.zeros(B, dtype=torch.long, device=dev)
        self.crossed_ok = torch.zeros(B, n, dtype=torch.bool, device=dev)
        self.crossed_bad = torch.zeros(B, n, dtype=torch.bool, device=dev)
        rel = self.state.p.detach().unsqueeze(1) - self.gate_c
        self.prev_s = (rel * self.gate_n).sum(-1)                  # (B,n)
        return s

    # ------------------------------------------------------------- update
    @torch.no_grad()
    def update_phases(self):
        """Once per control step: latch plane crossings + phase commits."""
        p = self.state.p.detach()
        v = self.state.v.detach()
        B, n = p.shape[0], self.n_gates
        rel = p.unsqueeze(1) - self.gate_c                         # (B,n,3)
        s = (rel * self.gate_n).sum(-1)                            # (B,n)
        lat_u = (rel * self.gate_u).sum(-1)                        # (B,n)
        lat_z = rel[..., 2]
        crossing = (self.prev_s > 0) & (s <= 0)
        inside = (lat_u.abs() < self.OPENING) & (lat_z.abs() < self.OPENING)
        near = (lat_u.abs() < self.FRAME_R) & (lat_z.abs() < self.FRAME_R)
        self.crossed_ok |= crossing & inside
        self.crossed_bad |= crossing & near & ~inside              # rang frame
        self.prev_s = s

        k = self.phase.clamp(max=n - 1)                            # (B,)
        ar = torch.arange(B, device=p.device)
        near_wp = (p - self.wps[k]).norm(dim=-1) < self.NEAR_TOL
        centered = (lat_u[ar, k].abs() < self.CENTER_TOL) & \
                   (lat_z[ar, k].abs() < self.CENTER_TOL)
        vu = (v * self.gate_u[k]).sum(-1)
        slow_lat = (vu.abs() < self.VLAT_TOL) & (v[:, 2].abs() < self.VLAT_TOL)
        yaw, _, _ = euler_zyx_from_quat(self.state.q)
        dyaw = yaw - self.tgt_yaws[k]
        aligned = torch.atan2(torch.sin(dyaw), torch.cos(dyaw)).abs() < self.YAW_TOL
        commit = (self.phase < n) & near_wp & centered & slow_lat & aligned
        self.phase = self.phase + commit.long()

    # ---------------------------------------------------------------- hooks
    def observe(self):
        # per-step phase/crossing updates without overriding rollout loops
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
        all_ok = self.crossed_ok.all(-1)
        any_bad = self.crossed_bad.any(-1)
        crash = (p[:, 2] > 1.2) | ~torch.isfinite(p).all(-1) | any_bad
        ok = all_ok & (~crash) & (e < err_pos) & (v.norm(dim=-1) < err_v)
        m = {
            "success": ok.float().mean().item(),
            "crossed_all": all_ok.float().mean().item(),
            "rang_frame": any_bad.float().mean().item(),
            "exit_err_med": e.median().item(),
            "phase_mean": self.phase.float().mean().item(),
            "crash": (crash & ~any_bad).float().mean().item(),
        }
        for i in range(self.n_gates):
            m[f"gate{i + 1}"] = self.crossed_ok[:, i].float().mean().item()
        return m


def verification_poses(env_cls):
    """Drone-view poses [x,y,z(m),yaw,pitch,roll] along the waypoint chain
    (start, mid-approach, just before / just after each gate, exit) for the
    visual gate (spike_multigate.py): the next gate must be in-frame at each
    pass, duplicated rings must render clean."""
    env = env_cls(EnvConfig(B=1, device="cpu"), renderer=False, image_dr=False)
    c, nrm, wps, yaws = env.gate_c, env.gate_n, env.wps, env.tgt_yaws
    poses, names = [], []

    def add(name, p, yaw):
        poses.append([*p.tolist(), float(yaw), 0.0, 0.0])
        names.append(name)

    add("start", c[0] + 2.1 * nrm[0], yaws[0])
    add("mid", c[0] + 1.3 * nrm[0], yaws[0])
    for k in range(env.n_gates):
        add(f"pre{k + 1}", wps[k], yaws[k])
        add(f"post{k + 1}", c[k] - 0.35 * nrm[k], yaws[k + 1])
    add("exit", env.wp_exit, yaws[-1])
    return poses, names


def oracle_main(env_cls, T_s: float, seed=5, B=256):
    """Shared __main__ for the concrete envs: state-only expert feasibility
    oracle (the gate for everything downstream, as in env_transit)."""
    import sys
    torch.manual_seed(0)
    env = env_cls(EnvConfig(B=B, tilt_dropout=0.0, device="cpu"),
                  renderer=False, image_dr=False)
    env.seed(seed)
    env.reset()
    m = env.transit_expert_rollout(int(T_s / env.cfg.dt_ctrl))
    print(f"expert feasibility {env_cls.__name__} (B={B}, {T_s} s):", m)
    sys.exit(0 if m["success"] > 0.97 else 1)
