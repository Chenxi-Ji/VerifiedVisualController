"""Meansub-only / CTBR-shaped-head variant of the original 58k controller,
trained against the ORIGINAL kinematic velocity plant with the OLD PINHOLE
camera. SIM-ONLY side deliverable (2026-07-12) for verification practice.

Provenance (do NOT edit the sources; everything needed is copied here):
  - ControllerMeansubCtbrVel is a copy of `Controller` in
    scripts_control/utils_ctrl_lya_pt.py (the ~58k net that flew hardware,
    artifact weights/ctrl_lya.pt) with EXACTLY two architecture changes:
      1. INPUT: 3 channels = per-image per-channel MEAN-SUBTRACTED RGB only
         (the original concatenated [raw, raw - mean] into 6 channels; the
         mean-sub expression `x - x.mean(dim=(2,3), keepdim=True)` is reused
         verbatim). conv1: Conv2d(6,16,5,2,2) -> Conv2d(3,16,5,2,2).
      2. OUTPUT: the original velocity squashing is replaced by the CTBR
         head of the PixelCTBR campaign (pixel2ctbr/policy.py constants +
         pixel2ctbr_ff/policy_ff.py `ctbr` branch, COPIED not imported):
           c = C_CENTER + clamp_relu(raw0, 1) * C_SPAN    in [0.1G, 1.9G] m/s^2
           w = clamp_relu(raw_{1:} * RATE_LIM, RATE_LIM)  +-(4,4,2) rad/s
         plus the campaign's zero-init last layer -> exact hover CTBR
         [G, 0, 0, 0] at init. The head LAYERS (Linear 120->64, ReLU,
         Dropout 0.1, Linear 64->4) are the original's; only the squashing
         differs. The net genuinely outputs CTBR-format numbers.
    Trunk / readouts / everything else: untouched.
    Param count: 56,836 (original 58,036; delta = conv1 16*3*5*5 vs
    16*6*5*5 = -1,200). Adaptive pools make the trunk resolution-agnostic,
    so the same weights consume the old 200x300 pinhole frames.

FIXED CTBR->VELOCITY INTERFACE (the load-bearing convention of this variant)
  The 4 CTBR-slot numbers are NOT flown through a rate loop here: the sim
  CONSUMES THEM AS THE BODY-FRAME VELOCITY COMMAND of the original plant
  (utils_ctrl_lya_pt.body_to_world_velocity + pose += v*dt, exactly as
  train_ctrl_lya_pt.py / test_ctrl_lya_pt.py integrate it). The mapping
  `ctbr_to_velocity` below is FIXED, affine (verification-friendly), and
  maps each slot's full CTBR range exactly onto the original velocity head's
  clamp range, with physically sensible signs (FRD body frame, z DOWN):

    CTBR slot (net output)          -> original velocity command slot
    --------------------------------------------------------------------
    c   thrust  [0.1G, 1.9G] m/s^2  -> vz  = -(c - G) / (0.9*G)  in [-1,1] u/s
                                       (above-hover thrust = climb = -z)
    wx  roll rate   +-4 rad/s       -> vy  = +wx / 4.0           in [-1,1] u/s
                                       (roll right = translate right)
    wy  pitch rate  +-4 rad/s       -> vx  = -wy / 4.0           in [-1,1] u/s
                                       (nose-down pitch = forward)
    wz  yaw rate    +-2 rad/s       -> yaw_rate = 0.3 * wz / 2.0 in [-0.3,0.3] rad/s
                                       (same axis; rescaled onto the
                                        original +-0.3 rad/s authority)

  So the reachable command set is IDENTICAL to the original Controller's
  ([-1,1] u/s per translation axis, +-0.3 rad/s yaw), the verification story
  is "this net outputs CTBR-format numbers", and the plant story is the
  original kinematic integrator, verbatim.

OLD PINHOLE CAMERA (the second delta; switch mechanics documented here)
  The repo migrated camera models at commit 27cd577 ("updates to camera and
  domain randomisation", 2026-06-21): the migration ADDED
  `camera_model="fisheye"` to both gsplat `rasterization()` calls in
  scripts_control/render_image.py and moved to a 1024x768 raster
  (fx=504.34...) downscaled to 256x192. BEFORE that commit (70c113e era,
  weights/old/ctrl_lya_20260611_035813.pt) `render()` passed NO
  camera_model kwarg -- and gsplat's default is camera_model='pinhole'
  (verified against the installed gsplat signature) -- rasterizing directly
  at width=300, height=200 with fx=113.258171, fy=113.347599,
  cx=158.868074, cy=98.837772 and returning the frame UNRESIZED.
  `render_pinhole` below replicates that old code path exactly (same view
  math, same rasterization kwargs) and passes camera_model="pinhole"
  EXPLICITLY where the old code relied on the default. The net therefore
  consumes (3, 200, 300) frames, exactly as the original pinhole-era
  training did. figures/camera_model_old_vs_new.png illustrates the
  projection difference (pinhole/rectilinear vs fisheye/equidistant).

Frames: gate-centered world frame (world_frame.json): z DOWN, +y through the
gate; everything (plant, losses, renderer) in SCENE UNITS as in the original
trainer; 1 u = 0.85 m only for REPORTING eval errors in meters.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils_ctrl_lya_pt import clamp_relu  # the original ReLU-only clamp

# ---- CTBR head constants (values copied from PixelCTBR
#      pixel2ctbr/policy.py lines 17-18; copied, not imported) ----
G = 9.81
C_CENTER, C_SPAN = G, 0.9 * G          # c = G + clamp_relu(.,1)*0.9G in [0.1G, 1.9G]
RATE_LIM = (4.0, 4.0, 2.0)             # rad/s, matches QuadCTBRDynamics.RATE_LIMIT

# ---- original velocity-command authority (utils_ctrl_lya_pt.Controller) ----
VEL_LIMIT = 1.0                        # u/s, the original clamp_relu(.,1.0)
YAWRATE_LIMIT = 0.3                    # rad/s, the original clamp_relu(.,0.3)

# ---- OLD pinhole camera (pre-27cd577 render() defaults) ----
OLD_PINHOLE_K = dict(fx=113.258171, fy=113.347599, cx=158.868074, cy=98.837772)
OLD_W, OLD_H = 300, 200

METERS_PER_UNIT = 0.85                 # scene units -> meters (this splat)


# =====================================================================
# CONTROLLER - 3ch meansub input, CTBR-shaped head, 58k trunk as-is
# =====================================================================
class ControllerMeansubCtbrVel(nn.Module):
    """See module docstring for the exact delta vs the original Controller."""

    def __init__(self):
        super().__init__()

        # Compact backbone (16/32/48/64). The original showed sizes for
        # 192x256; with the OLD pinhole frames the input is 200x300 ->
        # 100x150 -> 50x75 -> 25x38 -> 13x19 -> 7x10 (adaptive pools below
        # make the head input 120-d at any resolution).
        # ONLY change vs the original: in_channels 6 -> 3 (meansub RGB only).
        self.backbone = nn.Sequential(
            nn.AvgPool2d(2),                       # 200x300 -> 100x150
            nn.Conv2d(3, 16, 5, 2, 2),             # 3ch = mean-sub RGB -> 50x75
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1),            # -> 25x38
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 48, 3, 2, 1),            # -> 13x19
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Conv2d(48, 64, 3, 2, 1),            # -> 7x10
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
        # zero-init the last layer -> exact hover CTBR [G, 0, 0, 0] at init
        # (campaign head contract, policy_ff.py; ctbr_to_velocity maps it to
        # the zero velocity command, i.e. "stay put" on the original plant)
        nn.init.zeros_(self.action_head[-1].weight)
        nn.init.zeros_(self.action_head[-1].bias)

    def forward(self, x):
        """
        x: (B, 3, H, W) RGB images in [0, 1]
        Returns: (B, 4) CTBR-format action [c, wx, wy, wz]
          c mass-normalized collective thrust [m/s^2] in [0.1G, 1.9G];
          wx, wy, wz body rates [rad/s] clamped +-(4, 4, 2).
        (The sim consumes this through ctbr_to_velocity -- module docstring.)
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

        # CTBR head (policy_ff.py `ctbr` branch, copied)
        c = C_CENTER + clamp_relu(raw[:, :1], 1.0) * C_SPAN
        rl = torch.tensor(RATE_LIM, device=raw.device)
        w = clamp_relu(raw[:, 1:] * rl, rl)
        return torch.cat([c, w], dim=-1)


def ctbr_to_velocity(ctbr: torch.Tensor) -> torch.Tensor:
    """THE fixed CTBR->velocity interface (module docstring table).

    ctbr: (..., 4) [c m/s^2, wx, wy, wz rad/s]  (net output)
    Returns: (..., 4) [vx, vy, vz, yaw_rate] -- the ORIGINAL controller's
    body-frame velocity command semantics ([-1,1] u/s translation,
    [-0.3,0.3] rad/s yaw), ready for body_to_world_velocity + pose += v*dt.
    Affine and differentiable (BPTT flows through it during training).
    """
    c, wx, wy, wz = ctbr[..., 0], ctbr[..., 1], ctbr[..., 2], ctbr[..., 3]
    vx = -wy / RATE_LIM[1] * VEL_LIMIT           # nose-down pitch = forward
    vy = wx / RATE_LIM[0] * VEL_LIMIT            # roll right = right
    vz = -(c - C_CENTER) / C_SPAN * VEL_LIMIT    # above-hover thrust = climb (z down)
    yr = wz / RATE_LIM[2] * YAWRATE_LIMIT        # same axis, rescaled
    return torch.stack([vx, vy, vz, yr], dim=-1)


class CtbrAsVelocity(nn.Module):
    """ctrl-compatible wrapper: forward(img) = ctbr_to_velocity(net(img)).
    Lets the ORIGINAL test_ctrl_lya_pt.run_test drive this variant verbatim
    (it expects a module returning the velocity command)."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x):
        return ctbr_to_velocity(self.net(x))


# =====================================================================
# OLD PINHOLE RENDER - replica of scripts_control/render_image.py::render
# BEFORE commit 27cd577 (the pinhole era): 300x200 raster, fx~113 calib,
# camera_model pinhole (the old code omitted the kwarg; 'pinhole' is the
# gsplat default), NO resize. Same view math as the current render().
# =====================================================================
from render_image import get_viewmat, CAM_AXES  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402
from gsplat.rendering import rasterization  # noqa: E402


def render_pinhole(pose, scene, width=OLD_W, height=OLD_H,
                   fx=OLD_PINHOLE_K["fx"], fy=OLD_PINHOLE_K["fy"],
                   cx=OLD_PINHOLE_K["cx"], cy=OLD_PINHOLE_K["cy"],
                   device=torch.device("cuda" if torch.cuda.is_available()
                                       else "cpu")):
    """OLD-era single-image render: returns (3, 200, 300) float RGB in [0,1].
    Signature-compatible with render_image.render for test_ctrl_lya_pt's
    render_fn(pose, scene, device=...) call sites."""
    means, quats, opacities, scales, colors, transform, scale, world_frame = scene

    px, py, pz, yaw, pitch, roll = pose

    view = np.eye(4)
    R = Rotation.from_euler("ZYX", (yaw, pitch, roll)).as_matrix()
    view[:3, 3] = [px, py, pz]

    if world_frame:
        view[:3, :3] = R @ CAM_AXES
    else:
        view[:3, :3] = R
        tmp = Rotation.from_euler('zyx', [-np.pi / 2, np.pi / 2, 0]).as_matrix()
        view[:3, :3] = view[:3, :3] @ tmp
        view[0:3, 1:3] *= -1
        view = view[np.array([0, 2, 1, 3]), :]
        view[2, :] *= -1

    view = transform @ view
    view[:3, 3] *= scale

    view = torch.FloatTensor(view).unsqueeze(0).to(device)
    view = get_viewmat(view)

    Ks = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                      device=device).unsqueeze(0)

    rgb, alpha, _ = rasterization(
        means, quats,
        scales=torch.exp(scales),
        opacities=torch.sigmoid(opacities).squeeze(-1),
        colors=colors,
        viewmats=view,
        Ks=Ks,
        width=width,
        height=height,
        packed=False,
        near_plane=0.01,
        far_plane=1e10,
        render_mode="RGB+ED",
        sh_degree=0,
        sparse_grad=False,
        absgrad=True,
        rasterize_mode="classic",
        camera_model="pinhole",   # explicit; the old code relied on the default
    )

    img = rgb[0, ..., :3].clamp(0, 1)
    return img.permute(2, 0, 1).to(device)   # old era: NO resize


@torch.no_grad()
def render_batch_pinhole_gpu(poses_u, scene, K=None, chunk=8, device="cuda"):
    """Batched old-pinhole render: (B,6) scene-unit poses -> (B,3,200,300)
    float RGB in [0,1]. Same view math and rasterizer settings as
    render_pinhole (parity asserted by check_pinhole_parity), chunked so a
    batch fits the 25% GPU-memory budget. K: dict fx,fy,cx,cy (defaults to
    the OLD calib) -- the per-epoch intrinsics-jitter hook."""
    if torch.is_tensor(poses_u):
        poses_u = poses_u.detach().cpu().numpy()
    K = dict(OLD_PINHOLE_K) if K is None else K
    means, quats, opacities, scales, colors, transform, scale, world_frame = scene

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

    Ks1 = torch.tensor([[K["fx"], 0, K["cx"]],
                        [0, K["fy"], K["cy"]],
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
            width=OLD_W, height=OLD_H, packed=False,
            near_plane=0.01, far_plane=1e10,
            render_mode="RGB", sh_degree=0, sparse_grad=False, absgrad=False,
            rasterize_mode="classic", camera_model="pinhole",
        )
        outs.append(rgb[..., :3].clamp(0, 1).permute(0, 3, 1, 2))
    return torch.cat(outs, dim=0)


def check_pinhole_parity(scene, device, n=6, seed=0, atol=2e-3):
    """Assert the batched path reproduces the old single-image path."""
    rng = np.random.RandomState(seed)
    poses = np.array([0.0, 1.5, 0.0, -np.pi / 2, 0.0, 0.0]) + rng.uniform(
        low=[-1.2, -0.8, -0.5, -0.5, -0.2, -0.2],
        high=[1.2, 1.2, 0.4, 0.5, 0.2, 0.2], size=(n, 6))
    batched = render_batch_pinhole_gpu(poses, scene, chunk=3, device=device)
    worst = 0.0
    for i in range(n):
        single = render_pinhole(poses[i], scene, device=device)
        worst = max(worst, (batched[i] - single).abs().max().item())
    assert worst <= atol, f"pinhole batched/single parity broke: {worst}"
    return worst


# =====================================================================
# self-test (no scene needed):  python scripts_control/utils_ctrl_meansub_ctbrvel.py
# =====================================================================
if __name__ == "__main__":
    torch.manual_seed(0)
    m = ControllerMeansubCtbrVel()
    n_params = sum(p.numel() for p in m.parameters())
    print(f"ControllerMeansubCtbrVel params: {n_params:,} "
          "(original Controller: 58,036; meansub_att: 56,836)")
    assert n_params == 56836
    m.eval()
    img = torch.rand(2, 3, OLD_H, OLD_W)          # the old 200x300 frames
    a = m(img)
    assert a.shape == (2, 4)
    # zero-init = hover CTBR -> zero velocity command
    assert (a[:, 0] - G).abs().max() < 1e-5, "zero-init != hover thrust"
    assert (a[:, 1:] == 0).all(), "zero-init rates != 0"
    v = ctbr_to_velocity(a)
    assert v.abs().max() < 1e-6, "hover CTBR must map to zero velocity"
    print(f"zero-init CTBR: {a[0].tolist()} -> velocity {v[0].tolist()}")
    # CTBR bounds under huge inputs / random weights
    for p_ in m.parameters():
        p_.data.normal_(0, 0.05)
    m.eval()
    big = m(img + 100.0)
    assert torch.isfinite(big).all()
    assert big[:, 0].min() >= 0.1 * G - 1e-5 and big[:, 0].max() <= 1.9 * G + 1e-5
    for j, lim in enumerate(RATE_LIM):
        assert big[:, 1 + j].abs().max() <= lim + 1e-5
    # velocity image of the CTBR box == the original head's command box
    vb = ctbr_to_velocity(big)
    assert vb[:, :3].abs().max() <= VEL_LIMIT + 1e-5
    assert vb[:, 3].abs().max() <= YAWRATE_LIMIT + 1e-5
    # exact corner mapping: full thrust -> full descent-limit climb etc.
    corners = torch.tensor([[1.9 * G, 4.0, 4.0, 2.0],
                            [0.1 * G, -4.0, -4.0, -2.0],
                            [G, 0.0, 0.0, 0.0]])
    vc = ctbr_to_velocity(corners)
    expect = torch.tensor([[-1.0, 1.0, -1.0, 0.3],
                           [1.0, -1.0, 1.0, -0.3],
                           [0.0, 0.0, 0.0, 0.0]])
    assert (vc - expect).abs().max() < 1e-6, f"slot mapping wrong:\n{vc}"
    print("slot-mapping corners OK:\n", torch.cat([corners, vc], dim=-1))
    # brightness-cast invariance of the meansub input
    with torch.no_grad():
        cast = torch.tensor([0.07, -0.04, 0.05]).view(1, 3, 1, 1)
        d = (m(img) - m(img + cast)).abs().max()
    assert d < 1e-5, f"meansub cast invariance broken: {d}"
    print(f"cast invariance |delta| = {d:.2e}")
    # grad flows through the shim to conv1 and head
    m.train()
    out = ctbr_to_velocity(m(torch.rand(4, 3, OLD_H, OLD_W)))
    out.sum().backward()
    assert m.backbone[1].weight.grad is not None
    assert m.action_head[-1].weight.grad.abs().sum() > 0
    # wrapper equivalence
    m.eval()
    wrap = CtbrAsVelocity(m).eval()
    x = torch.rand(2, 3, OLD_H, OLD_W)
    assert torch.equal(wrap(x), ctbr_to_velocity(m(x)))
    print("utils_ctrl_meansub_ctbrvel self-test OK")
