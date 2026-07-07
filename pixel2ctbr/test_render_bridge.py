"""Verify render_bridge against the battle-tested render_image.render() path.
Run: python pixel2ctbr/test_render_bridge.py"""

import sys

import numpy as np
import torch

sys.path.insert(0, "pixel2ctbr")
sys.path.insert(0, "scripts_control")

from dynamics import quat_from_euler_zyx
from render_bridge import SplatRenderer, _CAM_AXES_Q, METERS_PER_UNIT
from render_image import CAM_AXES

FAIL = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAIL.append(name)


# 1. CAM_AXES quaternion equals CAM_AXES matrix ------------------------------
q = _CAM_AXES_Q
w, x, y, z = q
R = np.array([
    [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
    [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
    [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
])
check("CAM_AXES quaternion == matrix", np.abs(R - CAM_AXES).max() < 1e-6,
      f"maxdiff {np.abs(R - CAM_AXES).max():.2e}")

# 2. image parity vs legacy render() at several poses ------------------------
from render_image import Config, load_gsplat_scene, render  # noqa: E402

cfg = Config()
scene = load_gsplat_scene(cfg)
br = SplatRenderer(width=256, height=192, gray=False,
                   mount_jitter_rad=0.0, intrinsics_jitter=0.0)

poses = [
    [0.0, 1.5, 0.0, -np.pi / 2, 0.0, 0.0],
    [0.8, 2.3, -0.3, -np.pi / 2 + 0.4, 0.12, -0.08],   # nonzero pitch/roll
    [-1.2, 1.0, 0.3, -np.pi / 2 - 0.5, -0.15, 0.1],
]
worst = 0.0
for pose in poses:
    legacy = render(np.array(pose), scene)            # (3,192,256) via 1024 + resize
    p_m = torch.tensor([pose[:3]], dtype=torch.float32) * METERS_PER_UNIT
    qq = quat_from_euler_zyx(torch.tensor([pose[3]]), torch.tensor([pose[4]]),
                             torch.tensor([pose[5]]))
    mine = br.render_pose_quat(p_m, qq)[0]            # (3,192,256) direct
    d = (legacy.cpu() - mine.cpu()).abs()
    worst = max(worst, d.mean().item())
check("bridge ≈ legacy renders (AA differences only)", worst < 0.02,
      f"worst mean|diff| {worst:.4f}")

# 3. DR jitter actually changes the image, zero-DR is deterministic ----------
br_dr = SplatRenderer(width=128, height=96, gray=True,
                      mount_jitter_rad=0.6 * np.pi / 180, intrinsics_jitter=1.0)
p_m = torch.zeros(4, 3); p_m[:, 1] = 1.5 * METERS_PER_UNIT
qq = quat_from_euler_zyx(torch.full((4,), -np.pi / 2), torch.zeros(4), torch.zeros(4))
br_dr.sample_episode_dr(4, g=torch.Generator(device="cuda").manual_seed(1))
a = br_dr.render_pose_quat(p_m, qq)
br_dr.sample_episode_dr(4, g=torch.Generator(device="cuda").manual_seed(2))
b = br_dr.render_pose_quat(p_m, qq)
check("camera DR changes image", (a - b).abs().mean() > 1e-4,
      f"mean|diff| {(a-b).abs().mean():.5f}")
check("same-pose batch differs across batch (per-episode DR)",
      (a[0] - a[1]).abs().mean() > 1e-4)
check("gray output shape", list(a.shape) == [4, 1, 96, 128], str(list(a.shape)))

print("\n" + ("ALL PASS" if not FAIL else f"FAILURES: {FAIL}"))
sys.exit(1 if FAIL else 0)
