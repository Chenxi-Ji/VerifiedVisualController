#!/usr/bin/env python3
"""
Isolation test: render the scene at lateral & forward offsets and run the tflite.
Tells us whether the model TRACKS in sim (action varies with offset) or just
outputs a constant bias. If it tracks here but not on the real video, the
lateral failure is sim-to-real (camera aim / appearance); if it's flat here too,
it's the model/training.

Usage: python scripts_tflite/sim_lateral_test.py [weights/ctrl_lya.tflite]
Saves sim_lat_*.png / sim_fwd_*.png so you can eyeball the gate moving.
"""
import sys
import os
import numpy as np
import cv2
import tensorflow as tf
import torch
from dataclasses import dataclass

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
from scripts_control.render_image import render, load_gsplat_scene   # noqa: E402


@dataclass
class Cfg:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gsplat_path = "nerfstudio/outputs/Gate_Long_hloc_seq/splatfacto/2026-06-11_015308_cleaned"
    checkpoint  = "nerfstudio_models/step-000129999.ckpt"


cfg = Cfg()
tflite = sys.argv[1] if len(sys.argv) > 1 else "weights/ctrl_lya.tflite"

print(f"loading scene {cfg.gsplat_path} ({cfg.device}) ...")
scene = load_gsplat_scene(cfg)

it = tf.lite.Interpreter(model_path=tflite)
it.allocate_tensors()
ins = {d['name']: d['index'] for d in it.get_input_details()}
def set_in(key, arr):
    name = next(n for n in ins if key in n)
    it.set_tensor(ins[name], arr.astype(np.float32))
zero_pose = np.zeros((1, 6), np.float32)
target    = np.array([[0.0, 1.5, 0.0, -np.pi / 2, 0.0, 0.0]], np.float32)

def model_action(pose, tag):
    img = render(pose, scene, device=cfg.device)            # (3,192,256) RGB [0,1]
    nhwc = img.permute(1, 2, 0).detach().cpu().numpy()[None].astype(np.float32)
    bgr = cv2.cvtColor((nhwc[0] * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"sim_{tag}.png", bgr)
    set_in("image", nhwc)
    set_in("pose", zero_pose)
    set_in("target", target)
    it.invoke()
    for d in it.get_output_details():
        v = it.get_tensor(d['index']).flatten()
        if v.size == 4:
            return v
    return None

print("\n=== LATERAL sweep (y=1.5 in front, facing gate). a tracker's vy should FLIP sign with x ===")
print(" x_off    vx      vy      vz      yaw")
xs = [-1.0, -0.5, 0.0, 0.5, 1.0]
lat = {}
for x in xs:
    a = model_action([x, 1.5, 0.0, -np.pi / 2, 0.0, 0.0], f"lat_{x:+.1f}")
    lat[x] = a
    print(f" {x:+.1f}    {a[0]:+.3f}  {a[1]:+.3f}  {a[2]:+.3f}  {a[3]:+.3f}")

vy = np.array([lat[x][1] for x in xs])
print(f"\n vy across x[-1..+1] = {vy.round(2)}   range = {vy.max()-vy.min():.2f}")
if vy.max() - vy.min() > 0.6:
    print(" -> vy VARIES strongly with lateral offset: the model TRACKS in sim.")
    print("    => the flat/biased real-world vy is SIM-TO-REAL (camera aim / appearance).")
else:
    print(" -> vy is ~CONSTANT across lateral offset: the model does NOT track laterally even in sim.")
    print("    => the lateral failure is in the MODEL/TRAINING (retrain / arch).")

print("\n=== FORWARD sweep (x=0, centered). expect vx>0 when far, vx<0 when too close ===")
print(" y_off    vx      vy      vz      yaw   (y=1.5 = target)")
for y in [2.5, 2.0, 1.5, 1.0, 0.5]:
    a = model_action([0.0, y, 0.0, -np.pi / 2, 0.0, 0.0], f"fwd_{y:+.1f}")
    print(f" {y:+.1f}    {a[0]:+.3f}  {a[1]:+.3f}  {a[2]:+.3f}  {a[3]:+.3f}")

print("\nrenders saved: sim_lat_*.png / sim_fwd_*.png — eyeball that the gate shifts left/right & grows.")
