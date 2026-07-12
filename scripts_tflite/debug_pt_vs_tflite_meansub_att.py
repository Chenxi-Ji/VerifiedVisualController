"""Numerical parity: ctrl_lya_meansub_att.pt vs ctrl_lya_meansub_att*.tflite.

Thin variant of debug_pt_vs_tflite.py for the meansub/attitude model
(reuses its TFLiteFusedModel wrapper). Reports PER-CHANNEL abs deltas of the
action [c m/s^2, roll_sp, pitch_sp, yaw_sp] on
  * N random uniform-[0,1] images (default 256), and
  * the repo's real frames (real_t*.png + frames sampled from
    starling_video.mp4), preprocessed exactly like the drone
    (BGR->RGB, cv2 INTER_LINEAR -> 256x192, /255),
plus the Lyapunov-V delta on random poses (same ranges as the original).

Run on CPU:
    conda run -n certified_visual_controller python \
        scripts_tflite/debug_pt_vs_tflite_meansub_att.py \
        [--tflite weights/ctrl_lya_meansub_att.tflite] [--n 256]
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import glob
import sys

import numpy as np
import torch
import cv2

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "scripts_control"))

from scripts_control.utils_ctrl_lya_pt import Lyapunov                 # noqa: E402
from utils_ctrl_meansub_att import ControllerMeansubAtt, C_SPAN, TILT_SP_LIMIT  # noqa: E402
from debug_pt_vs_tflite import TFLiteFusedModel, _POSE_LOW, _POSE_HIGH  # noqa: E402

CH_NAMES = ["c[m/s^2]", "roll_sp", "pitch_sp", "yaw_sp"]
# per-channel span used for the "normalized units" view of the deltas
CH_SPANS = np.array([C_SPAN, TILT_SP_LIMIT, TILT_SP_LIMIT, np.pi])


def load_pt(weights):
    ck = torch.load(weights, map_location="cpu", weights_only=False)
    ctrl = ControllerMeansubAtt().eval()
    ctrl.load_state_dict(ck["controller"])
    lya = Lyapunov().eval()
    lya.load_state_dict(ck["lyapunov"])
    return ctrl, lya


def action_pair(ctrl, fused, img_hw3):
    """img_hw3: (192,256,3) float32 [0,1] -> (pt_action, tfl_action), (4,) each."""
    nchw = torch.from_numpy(np.ascontiguousarray(img_hw3.transpose(2, 0, 1))[None])
    with torch.no_grad():
        pt = ctrl(nchw).numpy().ravel()
    zeros = np.zeros((1, 6), np.float32)
    tfl, _ = fused(nchw, zeros, zeros)
    return pt, tfl.ravel()


def report(tag, diffs):
    diffs = np.asarray(diffs)  # (n, 4)
    print(f"\n=== {tag}  (n={len(diffs)}) ===")
    print(f"  {'channel':10s} {'max abs':>12s} {'mean abs':>12s} "
          f"{'max/span':>12s}")
    for j, name in enumerate(CH_NAMES):
        print(f"  {name:10s} {diffs[:, j].max():12.3e} "
              f"{diffs[:, j].mean():12.3e} "
              f"{diffs[:, j].max() / CH_SPANS[j]:12.3e}")
    print(f"  overall max abs: {diffs.max():.3e}   "
          f"overall max normalized: {(diffs / CH_SPANS).max():.3e}")
    return diffs


def real_frames(max_video_frames=16):
    """Yield (name, (192,256,3) float32) with the exact drone preprocessing."""
    for png in sorted(glob.glob(os.path.join(_PROJECT_ROOT, "real_t*.png"))):
        bgr = cv2.imread(png)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(rgb, (256, 192),
                         interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        yield os.path.basename(png), img
    vid = os.path.join(_PROJECT_ROOT, "starling_video.mp4")
    cap = cv2.VideoCapture(vid)
    if cap.isOpened():
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        idxs = np.linspace(0, max(n - 1, 0), max_video_frames).astype(int)
        for k in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(k))
            ok, bgr = cap.read()
            if not ok:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            img = cv2.resize(rgb, (256, 192),
                             interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
            yield f"starling_video.mp4[{k}]", img
        cap.release()


def compare_lyapunov(lya, fused, n=100, seed=0):
    rng = np.random.default_rng(seed)
    diffs = []
    img0 = torch.zeros((1, 3, 192, 256))
    for _ in range(n):
        pose = rng.uniform(_POSE_LOW, _POSE_HIGH, (1, 6)).astype(np.float32)
        target = rng.uniform(_POSE_LOW, _POSE_HIGH, (1, 6)).astype(np.float32)
        with torch.no_grad():
            V_pt, _ = lya(torch.tensor(pose), torch.tensor(target))
        _, V_tfl = fused(img0, pose, target)
        diffs.append(abs(V_pt.item() - V_tfl))
    diffs = np.array(diffs)
    print(f"\n=== Lyapunov V  (n={n}) ===")
    print(f"  max abs diff : {diffs.max():.3e}")
    print(f"  mean abs diff: {diffs.mean():.3e}")
    return diffs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights/ctrl_lya_meansub_att.pt")
    ap.add_argument("--tflite", default="weights/ctrl_lya_meansub_att.tflite")
    ap.add_argument("--n", type=int, default=256, help="random images")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"pt: {args.weights}\ntflite: {args.tflite}")
    ctrl, lya = load_pt(os.path.join(_PROJECT_ROOT, args.weights))
    fused = TFLiteFusedModel(os.path.join(_PROJECT_ROOT, args.tflite))

    # ---- random images ----
    rng = np.random.default_rng(args.seed)
    diffs, last = [], None
    for _ in range(args.n):
        img = rng.uniform(0, 1, (192, 256, 3)).astype(np.float32)
        pt, tfl = action_pair(ctrl, fused, img)
        diffs.append(np.abs(pt - tfl))
        last = (pt, tfl)
    report(f"controller, random images", diffs)
    print(f"  PT  last: {np.array2string(last[0], precision=4)}")
    print(f"  TFL last: {np.array2string(last[1], precision=4)}")

    # ---- real frames ----
    diffs, names = [], []
    for name, img in real_frames():
        pt, tfl = action_pair(ctrl, fused, img)
        diffs.append(np.abs(pt - tfl))
        names.append(name)
        print(f"  {name:28s} pt={np.array2string(pt, precision=3, suppress_small=True)} "
              f"tfl={np.array2string(tfl, precision=3, suppress_small=True)}")
    report(f"controller, real frames", diffs)

    # ---- lyapunov ----
    compare_lyapunov(lya, fused, n=100)
