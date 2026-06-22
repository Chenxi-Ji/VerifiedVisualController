#!/usr/bin/env python3
"""
Replay a real hires_small_color frame through ctrl_lya.tflite to check that the
action points toward the gate BEFORE flight.

Replicates the on-drone preprocessing EXACTLY (model_helper.cpp):
  NV12->RGB (here: PNG is RGB, cv2 loads BGR -> convert), cv2.INTER_LINEAR -> 256x192, /255.

Usage:
  python scripts_tflite/replay_real_frame.py <frame.png> [weights/ctrl_lya.tflite]

NOTE: run this AFTER retrain + export (weights/ctrl_lya.tflite must be the new model).
Use a frame from inside the training box (drone ~0.5-3u in front of the gate, facing
it) for a meaningful result — far / off-axis frames are out of distribution.
"""
import sys
import numpy as np
import cv2
import tensorflow as tf

png    = sys.argv[1] if len(sys.argv) > 1 else "hires_small_color.png"
tflite = sys.argv[2] if len(sys.argv) > 2 else "weights/ctrl_lya.tflite"

# --- preprocess identically to the drone ---
bgr = cv2.imread(png)
if bgr is None:
    sys.exit(f"could not read {png}")
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
img = cv2.resize(rgb, (256, 192), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
img = img[np.newaxis]                                   # (1, 192, 256, 3)
print(f"frame {png}: {bgr.shape[1]}x{bgr.shape[0]} -> 256x192 RGB [0,1]")

# --- run the tflite (action depends only on the image; pose/target feed V only) ---
it = tf.lite.Interpreter(model_path=tflite)
it.allocate_tensors()
ins = {d['name']: d['index'] for d in it.get_input_details()}
def set_in(key, arr):
    name = next(n for n in ins if key in n)
    it.set_tensor(ins[name], arr.astype(np.float32))
set_in("image",  img)
set_in("pose",   np.zeros((1, 6), np.float32))
set_in("target", np.array([[0.0, -1.5, 0.0, np.pi / 2, 0.0, 0.0]], np.float32))
it.invoke()

action = V = None
for d in it.get_output_details():
    t = it.get_tensor(d['index']).flatten()
    if t.shape[-1] == 4 or t.size == 4:
        action = t
    else:
        V = float(t[0])

vx, vy, vz, yaw = action
print(f"\naction [vx, vy, vz, yaw_rate] = [{vx:+.3f} {vy:+.3f} {vz:+.3f} {yaw:+.3f}]   V = {V:.3f}")
print(f"  forward : {vx:+.3f}  ({'toward gate' if vx > 0 else 'BACKWARD'})")
print(f"  lateral : {vy:+.3f}  ({'right' if vy > 0 else 'left'})")
print(f"  vert    : {vz:+.3f}  ({'down' if vz > 0 else 'up'})")
print(f"  yaw_rate: {yaw:+.3f}  ({'right/CW' if yaw > 0 else 'left/CCW'})")
print("\n(values are scene-units/s & rad/s, body FRD; deploy scales by meters_per_unit*safety)")
