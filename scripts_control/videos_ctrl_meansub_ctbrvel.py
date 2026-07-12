"""Rollout videos for the meansub/CTBR-as-velocity 58k variant, produced by
the ORIGINAL test_ctrl_lya_pt.run_test — called VERBATIM (imported, not
copied): because this variant's CTBR output is consumed as the original
velocity command through the fixed `ctbr_to_velocity` interface, wrapping
the net in `CtbrAsVelocity` (forward = ctbr_to_velocity(net(img))) makes it
a drop-in velocity-head controller for run_test's native integration
(body_to_world_velocity + pose += v*dt) and its 3-panel figure (3D
trajectory + Lyapunov curve + rendered view), H=30, 5 episodes, fps 3.

The ONLY divergences from the native invocation:
  - render_fn = render_pinhole (the OLD pinhole camera this variant is
    trained on) instead of the current fisheye render();
  - run_test writes its hardcoded rollout_pt_{idx}.mp4 names into a private
    subdirectory, which we then move to videos/rollout_meansub_ctbrvel_XX.mp4
    (so the original videos/rollout_pt_*.mp4 are never touched).

Run:  SIDE_MEM_FRAC=0.25 python scripts_control/videos_ctrl_meansub_ctbrvel.py
Out:  videos/rollout_meansub_ctbrvel_{00..04}.mp4
"""

import os
import shutil
import sys
import warnings

import matplotlib
matplotlib.use("Agg")

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from render_image import load_gsplat_scene  # noqa: E402
from test_ctrl_lya_pt import Config, run_test  # noqa: E402  (the ORIGINAL)
from utils_ctrl_lya_pt import Lyapunov  # noqa: E402
from utils_ctrl_meansub_ctbrvel import (  # noqa: E402
    ControllerMeansubCtbrVel, CtbrAsVelocity, render_pinhole)

WEIGHTS = "weights/ctrl_lya_meansub_ctbrvel.pt"
OUT_DIR = "videos"
TMP_DIR = os.path.join(OUT_DIR, "_tmp_meansub_ctbrvel")

if __name__ == "__main__":
    # plt.pause inside the (verbatim) run_test warns under the Agg backend
    warnings.filterwarnings(
        "ignore", message=".*non-GUI backend.*|.*non-interactive.*")

    cfg = Config()
    device = cfg.device
    if device == "cuda":
        torch.cuda.set_per_process_memory_fraction(
            float(os.environ.get("SIDE_MEM_FRAC", "0.25")), 0)

    scene = load_gsplat_scene(cfg)

    net = ControllerMeansubCtbrVel().to(device)
    Vnet = Lyapunov().to(device)

    ckpt = torch.load(WEIGHTS, map_location=device, weights_only=False)
    net.load_state_dict(ckpt["controller"])
    Vnet.load_state_dict(ckpt["lyapunov"])
    print(f"loaded {WEIGHTS} (epoch {ckpt.get('epoch')})")

    ctrl = CtbrAsVelocity(net)          # velocity-command view of the net

    np.random.seed(7)                    # reproducible episode set

    os.makedirs(TMP_DIR, exist_ok=True)
    run_test(
        ctrl=ctrl,
        Vnet=Vnet,
        scene=scene,
        target=cfg.target_pose,
        gate=cfg.gate_pose,
        render_fn=render_pinhole,        # OLD pinhole view panel
        device=device,
        dt=cfg.dt,
        H=cfg.H,                         # 30
        sample_num=cfg.sample_num,       # 5
        video_dir=TMP_DIR,
    )

    for idx in range(cfg.sample_num):
        src = os.path.join(TMP_DIR, f"rollout_pt_{idx:02d}.mp4")
        dst = os.path.join(OUT_DIR, f"rollout_meansub_ctbrvel_{idx:02d}.mp4")
        shutil.move(src, dst)
        print(f"[OUT] {dst}")
    shutil.rmtree(TMP_DIR, ignore_errors=True)
