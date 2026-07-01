#!/usr/bin/env python3
"""
Run BOTH ctrl_lya.tflite (deployed) and ctrl_lya.pt (PyTorch source) over a
recorded video and plot the action sequences with the hand-motion phases marked.

Preprocessing matches the drone EXACTLY: BGR->RGB, cv2 INTER_LINEAR -> 256x192, /255.
(tflite gets NHWC, the PyTorch Controller gets NCHW — same pixels.)

Usage:
  python scripts_tflite/replay_real_video.py <video.mp4> [ctrl_lya.tflite] [ctrl_lya.pt]

Outputs:
  - <video>_actions.csv   (frame, t_s, tfl_vx..yaw, pt_vx..yaw)
  - <video>_actions.png   (two graphs: tflite + pt, motion phases shaded)
  - per-phase mean actions printed (lateral-tracking sanity)
"""
import sys
import os
import csv
import numpy as np
import cv2
import tensorflow as tf
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
from scripts_control.utils_ctrl_lya_pt import Controller   # noqa: E402

video  = sys.argv[1] if len(sys.argv) > 1 else "gate_test.mp4"
tflite = sys.argv[2] if len(sys.argv) > 2 else "weights/ctrl_lya.tflite"
pt     = sys.argv[3] if len(sys.argv) > 3 else "weights/ctrl_lya.pt"

# Motion phases (seconds) from the recorded hand-move — EDIT to match your clip.
# (drone LEFT of center -> should command vy>0 right; drone RIGHT -> vy<0 left)
PHASES = [
    (0.0,  3.5,  "LEFT of center"),
    (3.5,  9.5,  "RIGHT (past mid)"),
    (9.5,  11.5, "back to MID"),
    (11.5, 15.5, "MID + right yaw"),
    (15.5, 1e9,  "fwd/close -> origin"),
]

# --- load both models ---
it = tf.lite.Interpreter(model_path=tflite)
it.allocate_tensors()
ins = {d['name']: d['index'] for d in it.get_input_details()}
def set_in(key, arr):
    name = next(n for n in ins if key in n)
    it.set_tensor(ins[name], arr.astype(np.float32))
zero_pose = np.zeros((1, 6), np.float32)
target    = np.array([[0.0, 1.5, 0.0, -np.pi / 2, 0.0, 0.0]], np.float32)

ctrl = Controller().eval()
ckpt = torch.load(pt, map_location="cpu")
ctrl.load_state_dict(ckpt["controller"])

cap = cv2.VideoCapture(video)
if not cap.isOpened():
    sys.exit(f"could not open {video}")
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

rows = []
i = 0
while True:
    ok, bgr = cap.read()
    if not ok:
        break
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(rgb, (256, 192), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0

    set_in("image", img[np.newaxis])      # tflite NHWC
    set_in("pose", zero_pose)
    set_in("target", target)
    it.invoke()
    tfl = None
    for d in it.get_output_details():
        v = it.get_tensor(d['index']).flatten()
        if v.size == 4:
            tfl = v

    nchw = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))[None])  # NCHW
    with torch.no_grad():
        pta = ctrl(nchw).numpy().flatten()

    rows.append([i, i / fps, *tfl, *pta])
    if i % 25 == 0:
        print(f"frame {i:5d} t={i/fps:5.2f}s  "
              f"tfl=({tfl[0]:+.2f},{tfl[1]:+.2f},{tfl[2]:+.2f},{tfl[3]:+.2f})  "
              f"pt=({pta[0]:+.2f},{pta[1]:+.2f},{pta[2]:+.2f},{pta[3]:+.2f})")
    i += 1
cap.release()

arr = np.array(rows)
t   = arr[:, 1]
tfl = arr[:, 2:6]
pta = arr[:, 6:10]

out_csv = os.path.splitext(video)[0] + "_actions.csv"
with open(out_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["frame", "t_s", "tfl_vx", "tfl_vy", "tfl_vz", "tfl_yaw",
                "pt_vx", "pt_vy", "pt_vz", "pt_yaw"])
    w.writerows(rows)
print(f"\n{i} frames -> {out_csv}")
print(f"tflite vs pt  max action diff = {np.abs(tfl - pta).max():.4f}  (parity on real frames)")

labels = ["vx(fwd+)", "vy(right+)", "vz(down+)", "yaw(CW+)"]
print("\nper-phase mean tflite action  [expect vy>0 when LEFT, vy<0 when RIGHT]:")
for (a, b, name) in PHASES:
    m = (t >= a) & (t < b)
    if m.any():
        mean = tfl[m].mean(0)
        print(f"  {a:4.1f}-{min(b, t.max()):5.1f}s  {name:20s} "
              f"vx={mean[0]:+.2f} vy={mean[1]:+.2f} vz={mean[2]:+.2f} yaw={mean[3]:+.2f}")

# --- plot: two graphs (tflite, pt) with phases shaded ---
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for ax, data, title in [(axes[0], tfl, "TFLite (deployed)"),
                            (axes[1], pta, "PyTorch .pt (source)")]:
        ax.axhline(0, color="0.5", lw=0.8)
        for j, lab in enumerate(labels):
            ax.plot(t, data[:, j], label=lab, lw=1.2)
        for k, (a, b, name) in enumerate(PHASES):
            if a < t.max():
                ax.axvspan(a, min(b, t.max()), color=("0.85" if k % 2 else "0.95"), alpha=0.6, zorder=0)
                ax.text(a + 0.1, 1.04, name, fontsize=8,
                        transform=ax.get_xaxis_transform(), va="bottom")
        ax.set_ylabel("action")
        ax.set_title(title)
        ax.legend(loc="lower right", ncol=4, fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_ylim(-1.15, 1.15)
    axes[1].set_xlabel("time (s)")
    fig.suptitle(os.path.basename(video))
    fig.tight_layout()
    out_png = os.path.splitext(video)[0] + "_actions.png"
    fig.savefig(out_png, dpi=120)
    print(f"plot -> {out_png}")
except Exception as e:
    print(f"(plot skipped: {e})")
