"""Rollout environment: dynamics + IMU + splat renderer + DR + obs assembly.

Serves both training phases (04_design.md §1):
  expert_rollout  — no_grad, expert actions, collects (image, vec, action)
                    sequences for BC (Phase A)
  policy_rollout  — differentiable through dynamics (images/IMU detached),
                    returns state trajectories for BPTT losses (Phase B)

Obs vector (12-D, normalize_vec in policy.py): gyro(3) accel(3) tilt(2)
last_action(4). Tilt dropout per episode makes the PX4 tilt estimate optional
at deployment. Images grayscale (B,1,96,128), DR'd per frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from dynamics import (DynParams, G, QuadCTBRDynamics, QuadState,
                      quat_from_euler_zyx, euler_zyx_from_quat)
from expert import GeometricHoverExpert
from imu import ImuParams, ImuSim
from policy import normalize_vec
from render_bridge import SplatRenderer, METERS_PER_UNIT


# --------------------------------------------------------------- gray image DR
class GrayDomainRandomizer:
    """Legacy DomainRandomizer's grayscale-meaningful subset (02 audit §5):
    small affine, gamma, contrast, exposure+clamp (AE clip survives mean-sub),
    blur, noise, cutout. Per-sample, no_grad, GPU."""

    def __init__(self, geo_p=0.8, rot=1.0, scale=0.02, trans=0.01,
                 gamma=0.30, contrast=0.25, exposure=0.08,
                 blur_p=0.30, noise=0.03, cutout_p=0.3):
        self.geo_p, self.rot, self.scale, self.trans = geo_p, rot, scale, trans
        self.gamma, self.contrast, self.exposure = gamma, contrast, exposure
        self.blur_p, self.noise, self.cutout_p = blur_p, noise, cutout_p

    @torch.no_grad()
    def __call__(self, imgs: torch.Tensor) -> torch.Tensor:
        B, _, H, W = imgs.shape
        dev = imgs.device
        u = lambda lo, hi, *s: lo + (hi - lo) * torch.rand(B, *s, device=dev)

        # geometric affine
        do_geo = torch.rand(B, device=dev) < self.geo_p
        ang = u(-self.rot, self.rot) * torch.pi / 180 * do_geo
        sc = 1 + u(-self.scale, self.scale) * do_geo
        tx = u(-self.trans, self.trans) * do_geo
        ty = u(-self.trans, self.trans) * do_geo
        cos, sin = torch.cos(ang) * sc, torch.sin(ang) * sc
        theta = torch.stack((
            torch.stack((cos, -sin, tx), -1),
            torch.stack((sin, cos, ty), -1)), dim=1)
        grid = torch.nn.functional.affine_grid(theta, imgs.shape, align_corners=False)
        x = torch.nn.functional.grid_sample(imgs, grid, padding_mode="border",
                                            align_corners=False)

        x = x.clamp(1e-4, 1).pow(u(1 - self.gamma, 1 + self.gamma).view(B, 1, 1, 1))
        mean = x.mean(dim=(2, 3), keepdim=True)
        x = mean + (x - mean) * u(1 - self.contrast, 1 + self.contrast).view(B, 1, 1, 1)
        x = (x + u(-self.exposure, self.exposure).view(B, 1, 1, 1)).clamp(0, 1)

        do_blur = torch.rand(B, device=dev) < self.blur_p
        if do_blur.any():
            k = torch.tensor([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]],
                             device=dev).view(1, 1, 3, 3) / 16
            xb = torch.nn.functional.conv2d(x, k, padding=1)
            x = torch.where(do_blur.view(B, 1, 1, 1), xb, x)

        x = (x + torch.randn_like(x) * self.noise).clamp(0, 1)

        do_cut = torch.rand(B, device=dev) < self.cutout_p
        if do_cut.any():
            for b in torch.nonzero(do_cut).flatten().tolist():
                ch = int(H * float(u(0.07, 0.18)[0]))
                cw = int(W * float(u(0.07, 0.18)[0]))
                cy = int(float(u(0, 1)[0]) * (H - ch))
                cx = int(float(u(0, 1)[0]) * (W - cw))
                x[b, :, cy:cy + ch, cx:cx + cw] = float(u(0, 1)[0])
        return x


# ------------------------------------------------------------------- env
@dataclass
class EnvConfig:
    B: int = 64
    dt_ctrl: float = 0.025
    n_sub: int = 5
    width: int = 128
    height: int = 96
    tilt_dropout: float = 0.2
    frame_gap: int = 6      # pair current frame with frame(t - gap): 150 ms
    # baseline makes hover-speed visual motion 1.5-4 px (consecutive frames
    # at 40 Hz differ sub-pixel — unlearnable; 05 log run-8 finding)
    device: str = "cuda"
    # start box (scene units where noted; converted to meters inside)
    target_u = (0.0, 1.5, 0.0)
    yaw_target: float = -torch.pi / 2


class HoverEnv:
    def __init__(self, cfg: EnvConfig, renderer: SplatRenderer | None = None,
                 image_dr: bool = True):
        self.cfg = cfg
        self.dyn = QuadCTBRDynamics(dt_ctrl=cfg.dt_ctrl, n_sub=cfg.n_sub)
        self.renderer = renderer or SplatRenderer(
            width=cfg.width, height=cfg.height, device=cfg.device,
            mount_jitter_rad=0.5 * torch.pi / 180, intrinsics_jitter=1.0,
            gray=True, supersample=2)
        self.dr = GrayDomainRandomizer() if image_dr else None
        self.expert = GeometricHoverExpert(dt=cfg.dt_ctrl)
        B, dev = cfg.B, cfg.device
        self.tgt_p = (torch.tensor(cfg.target_u) * METERS_PER_UNIT).expand(B, 3).to(dev)
        self.tgt_yaw = torch.full((B,), cfg.yaw_target, device=dev)
        self.g = torch.Generator().manual_seed(0)  # CPU generator for reproducibility

    def seed(self, seed: int):
        self.g = torch.Generator().manual_seed(seed)

    # ------------------------------------------------------------- reset
    def reset(self, vel_range=0.5):
        cfg, B, g, dev = self.cfg, self.cfg.B, self.g, self.cfg.device
        u = lambda lo, hi: (lo + (hi - lo) * torch.rand(B, generator=g)).to(dev)
        pr = DynParams.randomized(B, g=g)
        self.params = DynParams(**{k: getattr(pr, k).to(dev)
                                   for k in pr.__dataclass_fields__})
        ip = ImuParams.randomized(B, g=g)
        ip.gyro_bias = ip.gyro_bias.to(dev)
        ip.accel_bias = ip.accel_bias.to(dev)
        self.imu = ImuSim(ip, cfg.dt_ctrl)
        p0 = torch.stack((u(-1.5, 1.5) * METERS_PER_UNIT,
                          (1.5 + u(-1.0, 1.5)) * METERS_PER_UNIT,
                          u(-0.5, 0.4) * METERS_PER_UNIT), dim=-1)
        q0 = quat_from_euler_zyx(cfg.yaw_target + u(-0.6, 0.6),
                                 u(-0.1, 0.1), u(-0.1, 0.1))
        v0 = torch.stack([u(-vel_range, vel_range) for _ in range(3)], dim=-1)
        self.state = self.dyn.make_state(p0, v0, q0,
                                         torch.zeros(B, 3, device=dev), self.params)
        self.last_action = torch.zeros(B, 4, device=dev)
        self.last_action[:, 0] = G
        self.tilt_mask = (torch.rand(B, generator=g) > cfg.tilt_dropout).float().to(dev)
        self.expert.reset()
        self.renderer.sample_episode_dr(
            B, g=torch.Generator(device=dev).manual_seed(
                int(torch.randint(1 << 30, (1,), generator=g))))
        self.frame_ring = []     # primed on first observe()
        return self.state

    # ------------------------------------------------------------- obs
    @torch.no_grad()
    def observe(self):
        """-> image (B,2,H,W) [current, previous] on device, vec (B,12).
        Each frame is DR'd once when current and reused as previous — exactly
        how deployment behaves (each camera frame preprocessed once, model
        helper caches the last one)."""
        img = self.renderer.render_state(self.state.detach())
        if self.dr is not None:
            img = self.dr(img)
        if not self.frame_ring:
            self.frame_ring = [img] * self.cfg.frame_gap
        obs = torch.cat((img, self.frame_ring[0]), dim=1)
        self.frame_ring = self.frame_ring[1:] + [img]
        r = self.imu.read(self.state)
        tilt = r["tilt"] * self.tilt_mask.unsqueeze(-1)
        vec = normalize_vec(r["gyro"], r["accel"], tilt, self.last_action)
        return obs, vec

    # ------------------------------------------------------- expert data
    @torch.no_grad()
    def expert_rollout(self, T: int):
        """Run expert for T steps. Returns uint8 images (T,B,1,H,W) on CPU,
        vecs (T,B,12), expert actions (T,B,4), states for diagnostics."""
        imgs, vecs, acts = [], [], []
        for _ in range(T):
            img, vec = self.observe()
            a = self.expert(self.state, self.tgt_p, self.tgt_yaw)
            imgs.append((img * 255).to(torch.uint8).cpu())
            vecs.append(vec)
            acts.append(a)
            self.state = self.dyn.step(self.state, a, self.params)
            self.last_action = a
        return torch.stack(imgs), torch.stack(vecs), torch.stack(acts)

    # ------------------------------------------------- student closed loop
    def policy_rollout(self, policy, H: int, h0=None, no_grad=False,
                       with_expert=False, with_feat=False):
        """Closed-loop rollout of the policy for H steps. Images/IMU always
        detached; dynamics differentiable unless no_grad. Everything lives on
        cfg.device (policy included). Returns (states, actions, final hidden)
        [+ expert_labels if with_expert] [+ feats if with_feat].
        Expert labels are per-step expert actions at the *student-visited*
        states (DAgger-style; expert integrator runs along the student
        trajectory). feats are [h2, z] per step for auxiliary heads."""
        h = policy.init_hidden(self.cfg.B, self.cfg.device) if h0 is None else h0
        states, actions, labels, feats = [], [], [], []
        ctx = torch.no_grad() if no_grad else torch.enable_grad()
        with ctx:
            for _ in range(H):
                img, vec = self.observe()
                if with_expert:
                    with torch.no_grad():
                        labels.append(self.expert(self.state.detach(),
                                                  self.tgt_p, self.tgt_yaw))
                if with_feat:
                    a, h, f = policy(img, vec, h, return_feat=True)
                    feats.append(f)
                else:
                    a, h = policy(img, vec, h)
                self.state = self.dyn.step(self.state, a, self.params)
                self.last_action = a.detach()
                states.append(self.state)
                actions.append(a)
        out = [states, actions, h]
        if with_expert:
            out.append(labels)
        if with_feat:
            out.append(feats)
        return tuple(out)

    # ------------------------------------------------------------- metrics
    @torch.no_grad()
    def success_metrics(self, err_pos=0.15, err_v=0.2):
        e = (self.state.p - self.tgt_p).norm(dim=-1)
        v = self.state.v.norm(dim=-1)
        yaw, pitch, roll = euler_zyx_from_quat(self.state.q)
        yerr = torch.atan2(torch.sin(yaw - self.tgt_yaw),
                           torch.cos(yaw - self.tgt_yaw)).abs()
        crash = (self.state.p[:, 2] > 1.2) | ~torch.isfinite(self.state.p).all(-1)
        ok = (~crash) & (e < err_pos) & (v < err_v) & (yerr < 0.15)
        return {"success": ok.float().mean().item(),
                "err_med": e.median().item(), "err_p95": e.quantile(0.95).item(),
                "crash": crash.float().mean().item()}
