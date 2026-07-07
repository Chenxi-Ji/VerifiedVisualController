"""Bridge: QuadState -> low-res fisheye images from the splat twin.

Renders DIRECTLY at policy resolution with intrinsics scaled from the measured
1024x768 calibration (same fisheye projection as render-big-then-resize, minus
the box-filter AA; see 02_repo_audit.md §2). packed=True for memory; batch is
chunked to stay under the projection-buffer OOM ceiling measured on this GPU.

Also owns the camera-model DR (per-call intrinsics/mount jitter, replacing the
old per-epoch jitter_camera_model + ImageCache dance: CTBR rollouts revisit
poses too rarely for caching to pay, so every frame is rendered fresh and DR
can be per-episode instead of per-epoch).
"""

from __future__ import annotations

import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts_control"))
from render_image import Config, load_gsplat_scene, get_viewmat  # noqa: E402
from gsplat.rendering import rasterization  # noqa: E402

from dynamics import QuadState, quat_mul, quat_from_euler_zyx  # noqa: E402

# measured hires calib at 1024x768 (reproj 0.37 px)
FX, FY, CX, CY = 504.341405, 503.319815, 505.485234, 367.606186
FULL_W, FULL_H = 1024, 768
METERS_PER_UNIT = 0.85

# CAM_AXES (render_image.py) as a quaternion (w,x,y,z): camera looks along +x
# body, image up = -z. R_cam = R_body @ CAM_AXES. Value from
# Rotation.from_matrix(CAM_AXES); sign convention verified in test_render_bridge.
_CAM_AXES_Q = torch.tensor([-0.5, 0.5, 0.5, -0.5])


class SplatRenderer:
    def __init__(self, width=128, height=96, device="cuda", chunk=64,
                 mount_jitter_rad=0.0, intrinsics_jitter=0.0, gray=True,
                 supersample=2, scene=None):
        """supersample: render at N x target res and average-pool NxN down.
        Approximates the box-filter of the deployed downscale (spec: onboard
        preprocessing uses cv2 INTER_AREA); nearly free because throughput is
        gaussian-projection-bound, not pixel-bound (02_repo_audit.md §2).
        scene: prebuilt load_gsplat_scene tuple (e.g. edited by scene_edit.py
        — duplicated gates); None loads the pristine checkpoint."""
        self.gray = gray
        self.ss = int(supersample)
        cfg = Config()
        self.scene = load_gsplat_scene(cfg) if scene is None else scene
        (self.means, self.quats, opac, scl, self.colors,
         transform, self.scale, world_frame) = self.scene
        assert world_frame, "world_frame.json required (gate-centered frame)"
        self.opac = torch.sigmoid(opac).squeeze(-1).contiguous()
        self.scl = torch.exp(scl).contiguous()
        self.means = self.means.contiguous()
        self.quats = self.quats.contiguous()
        self.colors = self.colors.contiguous()
        self.transform = torch.tensor(transform, dtype=torch.float32, device=device)
        self.w, self.h = width, height          # policy resolution
        self.rw, self.rh = width * self.ss, height * self.ss  # render resolution
        self.device = device
        self.chunk = chunk
        self.mount_jitter = mount_jitter_rad
        self.intr_jitter = intrinsics_jitter
        sx, sy = self.rw / FULL_W, self.rh / FULL_H
        self.K0 = torch.tensor([[FX * sx, 0, CX * sx],
                                [0, FY * sy, CY * sy],
                                [0, 0, 1]], device=device)
        # per-episode DR state (B,3,3) K and (B,4) mount quaternion
        self._K = None
        self._mount_q = None

    def sample_episode_dr(self, B: int, g: torch.Generator | None = None):
        """Draw per-episode camera DR (call at env reset)."""
        dev = self.device
        jf = 1.0 + self.intr_jitter * 0.004 * torch.randn(B, 2, device=dev, generator=g)
        jc = self.intr_jitter * 0.5 * torch.randn(B, 2, device=dev, generator=g) \
            * (self.rw / FULL_W)
        K = self.K0.expand(B, 3, 3).clone()
        K[:, 0, 0] = K[:, 0, 0] * jf[:, 0]
        K[:, 1, 1] = K[:, 1, 1] * jf[:, 1]
        K[:, 0, 2] = K[:, 0, 2] + jc[:, 0]
        K[:, 1, 2] = K[:, 1, 2] + jc[:, 1]
        self._K = K
        ang = self.mount_jitter * torch.randn(B, 3, device=dev, generator=g)
        self._mount_q = quat_from_euler_zyx(ang[:, 0], ang[:, 1], ang[:, 2])

    @torch.no_grad()
    def render_state(self, s: QuadState) -> torch.Tensor:
        """(B,1,H,W) grayscale in [0,1] at the drone's camera pose."""
        return self.render_pose_quat(s.p, s.q)

    @torch.no_grad()
    def render_pose_quat(self, p_m: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        B = p_m.shape[0]
        dev = self.device
        if self._K is None or self._K.shape[0] != B:
            self.sample_episode_dr(B)
        q_cam = quat_mul(q.to(dev), self._mount_q)
        q_cam = quat_mul(q_cam, _CAM_AXES_Q.to(dev).expand(B, 4))
        # quaternion -> rotation matrix (body->world with camera axes)
        w_, x_, y_, z_ = q_cam.unbind(-1)
        R = torch.stack((
            torch.stack((1 - 2 * (y_ * y_ + z_ * z_), 2 * (x_ * y_ - w_ * z_), 2 * (x_ * z_ + w_ * y_)), -1),
            torch.stack((2 * (x_ * y_ + w_ * z_), 1 - 2 * (x_ * x_ + z_ * z_), 2 * (y_ * z_ - w_ * x_)), -1),
            torch.stack((2 * (x_ * z_ - w_ * y_), 2 * (y_ * z_ + w_ * x_), 1 - 2 * (x_ * x_ + y_ * y_)), -1),
        ), dim=-2)
        c2w = torch.zeros(B, 4, 4, device=dev)
        c2w[:, 3, 3] = 1.0
        c2w[:, :3, :3] = R
        c2w[:, :3, 3] = p_m.to(dev) / METERS_PER_UNIT
        c2w = self.transform @ c2w
        c2w[:, :3, 3] *= self.scale
        viewmats = get_viewmat(c2w)

        outs = []
        for i in range(0, B, self.chunk):
            rgb, _, _ = rasterization(
                self.means, self.quats, scales=self.scl, opacities=self.opac,
                colors=self.colors, viewmats=viewmats[i:i + self.chunk],
                Ks=self._K[i:i + self.chunk], width=self.rw, height=self.rh,
                packed=True, near_plane=0.01, far_plane=1e10, render_mode="RGB",
                sh_degree=0, rasterize_mode="classic", camera_model="fisheye",
            )
            outs.append(rgb[..., :3].clamp(0, 1))
        img = torch.cat(outs, 0)                      # (B,rh,rw,3)
        if self.ss > 1:
            img = torch.nn.functional.avg_pool2d(
                img.permute(0, 3, 1, 2), self.ss).permute(0, 2, 3, 1)
        if not self.gray:
            return img.permute(0, 3, 1, 2)            # (B,3,H,W)
        g = (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2])
        return g.unsqueeze(1)                         # (B,1,H,W)
