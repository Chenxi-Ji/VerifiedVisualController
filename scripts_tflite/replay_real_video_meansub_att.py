#!/usr/bin/env python3
"""
Thin variant of replay_real_video.py for the meansub/attitude model: run BOTH
ctrl_lya_meansub_att.tflite and ctrl_lya_meansub_att.pt over a recorded video
and plot the action sequences with the hand-motion phases marked.

Only the model-specific parts differ from the original script:
  - PyTorch class Controller -> ControllerMeansubAtt (3ch meansub input)
  - action semantics [vx,vy,vz,yaw_rate] -> [c m/s^2, roll_sp, pitch_sp,
    yaw_sp(absolute)]; CSV stores RAW actions; the plot shows c as c/G-1 so
    the thrust trace shares an axis with the rad-valued setpoints
  - outputs get a _meansub_att suffix (the originals from ctrl_lya stay put)

Preprocessing matches the drone EXACTLY: BGR->RGB, cv2 INTER_LINEAR -> 256x192, /255.
(tflite gets NHWC, the PyTorch controller gets NCHW — same pixels.)

Usage:
  python scripts_tflite/replay_real_video_meansub_att.py <video.mp4> \
      [weights/ctrl_lya_meansub_att.tflite] [weights/ctrl_lya_meansub_att.pt]

Outputs:
  - <video>_meansub_att_actions.csv   (frame, t_s, tfl_c..yaw, pt_c..yaw)
  - <video>_meansub_att_actions.png   (two graphs: tflite + pt, phases shaded)
  - per-phase mean actions printed (lateral-tracking sanity via roll_sp sign)
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
sys.path.insert(0, os.path.join(_ROOT, "scripts_control"))
from utils_ctrl_meansub_att import ControllerMeansubAtt, G   # noqa: E402

video  = sys.argv[1] if len(sys.argv) > 1 else "starling_video.mp4"
tflite = sys.argv[2] if len(sys.argv) > 2 else "weights/ctrl_lya_meansub_att.tflite"
pt     = sys.argv[3] if len(sys.argv) > 3 else "weights/ctrl_lya_meansub_att.pt"

# Motion phases (seconds) from the recorded hand-move — same clip as the
# original replay (starling_video.mp4).
# (drone LEFT of center -> expect roll_sp>0 rightward; RIGHT -> roll_sp<0)
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

ctrl = ControllerMeansubAtt().eval()
ckpt = torch.load(pt, map_location="cpu", weights_only=False)
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

out_csv = os.path.splitext(video)[0] + "_meansub_att_actions.csv"
with open(out_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["frame", "t_s", "tfl_c", "tfl_roll_sp", "tfl_pitch_sp", "tfl_yaw_sp",
                "pt_c", "pt_roll_sp", "pt_pitch_sp", "pt_yaw_sp"])
    w.writerows(rows)
print(f"\n{i} frames -> {out_csv}")
print(f"tflite vs pt  max action diff = {np.abs(tfl - pta).max():.4f}  (parity on real frames)")
print(f"per-channel [c, roll, pitch, yaw] max diff = "
      f"{np.abs(tfl - pta).max(0).round(5)}")

labels = ["c/G-1 (thrust)", "roll_sp (rad)", "pitch_sp (rad)", "yaw_sp (rad, abs)"]
print("\nper-phase mean tflite action [c m/s^2, roll_sp, pitch_sp, yaw_sp] "
      "(expect roll_sp>0 when LEFT, <0 when RIGHT; yaw_sp ~ -1.571):")
for (a, b, name) in PHASES:
    m = (t >= a) & (t < b)
    if m.any():
        mean = tfl[m].mean(0)
        print(f"  {a:4.1f}-{min(b, t.max()):5.1f}s  {name:20s} "
              f"c={mean[0]:+.2f} roll={mean[1]:+.3f} pitch={mean[2]:+.3f} yaw={mean[3]:+.3f}")

# --- plot: two graphs (tflite, pt) with phases shaded ---
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def plot_cols(data):
        # thrust as c/G-1 so all traces share the axis; rest raw (rad)
        return np.column_stack([data[:, 0] / G - 1.0, data[:, 1:4]])

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for ax, data, title in [(axes[0], plot_cols(tfl), "TFLite (exported)"),
                            (axes[1], plot_cols(pta), "PyTorch .pt (source)")]:
        ax.axhline(0, color="0.5", lw=0.8)
        ax.axhline(-np.pi / 2, color="0.7", lw=0.8, ls="--")  # yaw center
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
        ax.set_ylim(-2.0, 1.15)
    axes[1].set_xlabel("time (s)")
    fig.suptitle(os.path.basename(video) + "  (meansub/attitude model)")
    fig.tight_layout()
    out_png = os.path.splitext(video)[0] + "_meansub_att_actions.png"
    fig.savefig(out_png, dpi=120)
    print(f"plot -> {out_png}")
except Exception as e:
    print(f"(plot skipped: {e})")
