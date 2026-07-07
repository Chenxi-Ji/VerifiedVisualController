"""Benchmark gsplat rendering throughput for pixel2ctbr training.

Decides render-in-the-loop RL feasibility: how many images/sec can the twin
deliver at policy resolution on this GPU, batched, with the fisheye camera
model? Two paths are measured:

  legacy : render full 1024x768 fisheye (RGB+ED, as train_ctrl_lya_pt.py does)
           then cv2 INTER_LINEAR downscale on CPU  -> matches deployed pipeline
  direct : render at target resolution with intrinsics scaled by out/full
           (geometrically identical fisheye projection; cheaper; RGB only)

Run from repo root:  python pixel2ctbr/bench_render.py
"""

import os
import sys
import time

import numpy as np
import torch
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts_control"))
from render_image import Config, load_gsplat_scene, get_viewmat, CAM_AXES  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402
from gsplat.rendering import rasterization  # noqa: E402

FX, FY, CX, CY = 504.341405, 503.319815, 505.485234, 367.606186  # 1024x768 calib
FULL_W, FULL_H = 1024, 768


def make_viewmats(poses, scene, device):
    means, quats, opacities, scales, colors, transform, scale, world_frame = scene
    assert world_frame, "expected gate-centered world_frame.json convention"
    views = []
    for p in poses:
        px, py, pz, yaw, pitch, roll = p
        view = np.eye(4)
        R = Rotation.from_euler("ZYX", (yaw, pitch, roll)).as_matrix()
        view[:3, 3] = [px, py, pz]
        view[:3, :3] = R @ CAM_AXES
        view = transform @ view
        view[:3, 3] *= scale
        views.append(view)
    views = torch.from_numpy(np.stack(views)).float().to(device)
    return get_viewmat(views)


def bench(scene, batch, w, h, mode, device, iters=10, warmup=3, cpu_resize_to=None):
    means, quats, opacities, scales, colors, *_ = scene
    sx, sy = w / FULL_W, h / FULL_H
    Ks = torch.tensor(
        [[FX * sx, 0, CX * sx], [0, FY * sy, CY * sy], [0, 0, 1]],
        device=device, dtype=torch.float32,
    ).unsqueeze(0).repeat(batch, 1, 1)

    rng = np.random.default_rng(0)
    poses = np.stack([
        rng.uniform([-1.5, 0.5, -0.5, -np.pi / 2 - 0.6, -0.2, -0.2],
                    [+1.5, 3.0, +0.4, -np.pi / 2 + 0.6, +0.2, +0.2])
        for _ in range(batch)
    ])
    viewmats = make_viewmats(poses, scene, device)

    scales_e = torch.exp(scales).contiguous()
    opac = torch.sigmoid(opacities).squeeze(-1).contiguous()

    def one():
        rgb, _, _ = rasterization(
            means.contiguous(), quats.contiguous(), scales=scales_e,
            opacities=opac, colors=colors.contiguous(),
            viewmats=viewmats, Ks=Ks, width=w, height=h,
            packed=False, near_plane=0.01, far_plane=1e10,
            render_mode=mode, sh_degree=0, sparse_grad=False,
            absgrad=False, rasterize_mode="classic", camera_model="fisheye",
        )
        img = rgb[..., :3].clamp(0, 1)
        if cpu_resize_to is not None:
            ow, oh = cpu_resize_to
            arr = img.detach().cpu().numpy()
            img = np.stack([cv2.resize(a, (ow, oh), interpolation=cv2.INTER_LINEAR)
                            for a in arr])
        else:
            torch.cuda.synchronize()
        return img

    with torch.no_grad():
        for _ in range(warmup):
            one()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            one()
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / iters
    return batch / dt, dt * 1000


def main():
    cfg = Config()
    device = cfg.device
    scene = load_gsplat_scene(cfg)
    n_gauss = scene[0].shape[0]
    print(f"scene: {n_gauss:,} gaussians | GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'path':<28}{'res':<12}{'batch':<7}{'img/s':>10}{'ms/call':>10}")

    rows = [
        ("legacy 1024x768+cpu-resize", FULL_W, FULL_H, "RGB+ED", 1, (256, 192)),
        ("legacy 1024x768+cpu-resize", FULL_W, FULL_H, "RGB+ED", 8, (256, 192)),
        ("legacy 1024x768+cpu-resize", FULL_W, FULL_H, "RGB+ED", 32, (256, 192)),
        ("direct RGB", 256, 192, "RGB", 1, None),
        ("direct RGB", 256, 192, "RGB", 32, None),
        ("direct RGB", 256, 192, "RGB", 128, None),
        ("direct RGB", 256, 192, "RGB", 256, None),
        ("direct RGB", 128, 96, "RGB", 32, None),
        ("direct RGB", 128, 96, "RGB", 256, None),
        ("direct RGB", 128, 96, "RGB", 512, None),
        ("direct RGB", 64, 48, "RGB", 256, None),
        ("direct RGB", 64, 48, "RGB", 512, None),
    ]
    for name, w, h, mode, b, resize in rows:
        try:
            ips, ms = bench(scene, b, w, h, mode, device, cpu_resize_to=resize)
            print(f"{name:<28}{f'{w}x{h}':<12}{b:<7}{ips:>10.0f}{ms:>10.1f}")
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"{name:<28}{f'{w}x{h}':<12}{b:<7}{'OOM':>10}")


if __name__ == "__main__":
    main()
