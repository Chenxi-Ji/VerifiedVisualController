"""Visual verification for scene_edit + the multi-gate envs (07 doc).

Renders drone-view color frames of edited scenes to spike_out/multigate/:
  crop  — pristine / gate-deleted / gate-only / two-gate views (crop-box QA)
  two   — frames along the two-gate trajectory (env_two_gate waypoints)
  three — frames along the three-gate-turn trajectory

Run from repo root:  python pixel2ctbr/spike_multigate.py [crop|two|three]
GPU-frugal: chunk=8, <=4 poses per call (a training run owns the GPU).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, "pixel2ctbr")
sys.path.insert(0, "scripts_control")

from dynamics import quat_from_euler_zyx  # noqa: E402
from render_bridge import SplatRenderer, METERS_PER_UNIT  # noqa: E402
from render_image import Config, load_gsplat_scene  # noqa: E402
import scene_edit as se  # noqa: E402

OUT = "pixel2ctbr/spike_out/multigate"
W, H = 512, 384          # inspection res (fisheye intrinsics scale with W,H)


def save_views(scene, poses, names, prefix):
    """poses: list of [x,y,z(m), yaw,pitch,roll]; renders <=4 at a time."""
    r = SplatRenderer(width=W, height=H, gray=False, supersample=1,
                      chunk=8, scene=scene)
    os.makedirs(OUT, exist_ok=True)
    for i in range(0, len(poses), 4):
        chunk = poses[i:i + 4]
        p = torch.tensor([c[:3] for c in chunk], dtype=torch.float32)
        e = torch.tensor([c[3:] for c in chunk], dtype=torch.float32)
        q = quat_from_euler_zyx(e[:, 0], e[:, 1], e[:, 2])
        img = r.render_pose_quat(p, q)          # (b,3,H,W)
        for j, name in enumerate(names[i:i + 4]):
            arr = (img[j].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            import cv2
            cv2.imwrite(f"{OUT}/{prefix}_{name}.png", arr[..., ::-1])
            print(f"saved {OUT}/{prefix}_{name}.png")
    del r
    torch.cuda.empty_cache()


def crop_qa():
    scene = load_gsplat_scene(Config())
    mask = se.extract_gate_gaussians(scene)
    print(f"gate mask: {int(mask.sum()):,} gaussians")
    front = [0.0, 1.4, 0.0, -np.pi / 2, 0.0, 0.0]
    side = [1.6, 1.1, -0.15, -np.pi / 2 - 0.6, 0.0, 0.0]
    save_views(scene, [front, side], ["front", "side"], "00_pristine")
    save_views(se.delete_gaussians(scene, mask), [front, side],
               ["front", "side"], "01_deleted")
    gate_only = tuple(t[mask] for t in scene[:5]) + tuple(scene[5:])
    save_views(gate_only, [front, side], ["front", "side"], "02_gateonly")
    two = se.compose(scene, se.duplicate_gate(
        scene, se.gate_pose((0, -2.2, 0)), mask=mask))
    save_views(two, [
        [0.0, 2.0, 0.0, -np.pi / 2, 0, 0],     # start: both gates line up
        [0.0, 0.7, 0.0, -np.pi / 2, 0, 0],     # pre-gate-1 waypoint
        [0.0, -0.4, 0.0, -np.pi / 2, 0, 0],    # past gate 1 -> gate 2 ahead
        [0.9, -1.0, -0.1, -np.pi / 2 - 0.35, 0, 0],  # oblique on the copy
    ], ["start", "pre1", "past1", "oblique"], "03_twogate")


def traj_qa(env_mod, prefix):
    """Frames along an env's waypoint chain: start, mid-approach, just
    before / just after each gate, exit."""
    mod = __import__(env_mod)
    scene = se.multi_gate_scene(mod.DUP_GATE_POSES)
    poses, names = mod.verification_poses()
    save_views(scene, poses, names, prefix)


def bench():
    """Throughput sanity at training settings (128x96 ss=2 gray) for the
    composited scenes vs pristine. chunk=8/B=8 keeps the footprint tiny
    while a training run owns the GPU — relative slowdown is the number."""
    import time
    import env_two_gate, env_three_gate_turn  # noqa: F401
    scenes = [("pristine", load_gsplat_scene(Config()))]
    scenes.append(("two-gate", se.multi_gate_scene(env_two_gate.DUP_GATE_POSES)))
    scenes.append(("three-gate",
                   se.multi_gate_scene(env_three_gate_turn.DUP_GATE_POSES)))
    B = 8
    p = torch.zeros(B, 3)
    p[:, 1] = torch.linspace(0.2, 2.2, B)
    q = quat_from_euler_zyx(torch.full((B,), -np.pi / 2),
                            torch.zeros(B), torch.zeros(B))
    for name, scene in scenes:
        r = SplatRenderer(width=128, height=96, gray=True, supersample=2,
                          chunk=8, scene=scene)
        for _ in range(3):
            r.render_pose_quat(p, q)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        iters = 40
        for _ in range(iters):
            r.render_pose_quat(p, q)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / iters
        print(f"{name:<11} {scene[0].shape[0]:>9,} gaussians  "
              f"{B / dt:7.1f} img/s  ({dt * 1e3 / B:.2f} ms/img)")
        del r
        torch.cuda.empty_cache()


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "crop"
    torch.manual_seed(0)
    if what == "crop":
        crop_qa()
    elif what == "two":
        traj_qa("env_two_gate", "10_two")
    elif what == "three":
        traj_qa("env_three_gate_turn", "20_three")
    elif what == "bench":
        bench()
