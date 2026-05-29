"""Numerical comparison of PyTorch vs fused TFLite model."""

import os
import sys
import numpy as np
import torch
import cv2

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

from scripts_control.utils_ctrl_lya_pt import Controller, Lyapunov

try:
    from ai_edge_litert.interpreter import Interpreter as TFLiteInterp
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter as TFLiteInterp
    except ImportError:
        import tensorflow as tf
        TFLiteInterp = tf.lite.Interpreter


# =============================
# POSE RANGE (x,y,z,yaw)
# =============================
_POSE_LOW  = np.array([-1.5, -4.2, -0.7, -3.14, 0.0, 0.0], dtype=np.float32)
_POSE_HIGH = np.array([ 1.5, -1.8,  0.7,  3.14, 0.0, 0.0], dtype=np.float32)


# =============================
# FUSED TFLITE MODEL
# =============================
class TFLiteFusedModel:
    """Runs controller + Lyapunov in one TFLite call."""

    def __init__(self, model_path):
        self.interp = TFLiteInterp(model_path=model_path)
        self.interp.allocate_tensors()

        inp = self.interp.get_input_details()
        out = self.interp.get_output_details()

        # -------- input parsing --------
        self._img = None
        self._pose = None
        self._target = None
        fallback = []

        for d in inp:
            if len(d["shape"]) == 4:
                self._img = d
            else:
                n = d["name"].lower()
                if "pose" in n:
                    self._pose = d
                elif "target" in n:
                    self._target = d
                else:
                    fallback.append(d)

        if self._pose is None:
            self._pose = fallback.pop(0)
        if self._target is None:
            self._target = fallback.pop(0)

        # -------- image format --------
        s = self._img["shape"]
        if s[1] == 3:  # NCHW
            self.h, self.w = int(s[2]), int(s[3])
            self.nhwc = False
        else:
            self.h, self.w = int(s[1]), int(s[2])
            self.nhwc = True

        # -------- outputs --------
        outs = self.interp.get_output_details()
        self._action = None
        self._V = None

        for d in outs:
            if len(d["shape"]) >= 2 and d["shape"][-1] == 4:
                self._action = d
            else:
                self._V = d

        if self._action is None:
            self._action, self._V = outs[0], outs[1]

    def __call__(self, img_tensor, pose_np, target_np):
        img = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        img = cv2.resize(img, (self.w, self.h))

        x = (img[np.newaxis] if self.nhwc
             else img.transpose(2, 0, 1)[np.newaxis]).astype(np.float32)

        self.interp.set_tensor(self._img["index"], x)
        self.interp.set_tensor(self._pose["index"], pose_np.astype(np.float32))
        self.interp.set_tensor(self._target["index"], target_np.astype(np.float32))
        self.interp.invoke()

        action = self.interp.get_tensor(self._action["index"]).astype(np.float32)
        V = float(self.interp.get_tensor(self._V["index"]).flat[0])

        return action, V


# =============================
# LOAD PYTORCH MODELS
# =============================
def load_pt_models():
    ckpt = torch.load(
        os.path.join(_PROJECT_ROOT, "weights/ctrl_lya.pt"),
        map_location="cpu",
        weights_only=False,
    )

    ctrl = Controller().eval()
    ctrl.load_state_dict(ckpt["controller"])

    lya = Lyapunov().eval()
    lya.load_state_dict(ckpt["lyapunov"])

    return ctrl, lya


# =============================
# LOAD TFLITE FUSED MODEL
# =============================
def load_tflite_fused():
    return TFLiteFusedModel(
        os.path.join(_PROJECT_ROOT, "weights/ctrl_lya.tflite")
    )


# =============================
# CONTROLLER COMPARISON
# =============================
def compare_controller(ctrl_pt, fused, n=50, seed=42):
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        img_nchw = rng.uniform(0, 1, (1, 3, 200, 300)).astype(np.float32)
        img_nhwc = img_nchw[0].transpose(1, 2, 0)[np.newaxis]   # (1,200,300,3)

        with torch.no_grad():
            pt_out = ctrl_pt(torch.tensor(img_nchw)).numpy()    # (1,6)

        pose = np.zeros((1, 6), dtype=np.float32)
        target = np.zeros((1, 6), dtype=np.float32)

        tfl_out, _ = fused(
            torch.tensor(img_nchw),
            pose,
            target
        )

        diffs.append(np.abs(pt_out - tfl_out).max())

    diffs = np.array(diffs)
    print(f"\n=== Controller  (n={n}) ===")
    print(f"  max  abs diff : {diffs.max():.6f}")
    print(f"  mean abs diff : {diffs.mean():.6f}")
    print(f"  p95  abs diff : {np.percentile(diffs, 95):.6f}")

    # spot-check last sample
    print(f"  PT  last: {pt_out.flatten()}")
    print(f"  TFL last: {tfl_out.flatten()}")
    return diffs

# =============================
# LYA COMPARISON
# =============================
def compare_lyapunov(lya_pt, fused, n=100, seed=0):
    rng = np.random.default_rng(seed)
    diffs = []

    for _ in range(n):
        pose = rng.uniform(_POSE_LOW, _POSE_HIGH, (1, 6)).astype(np.float32)
        target = rng.uniform(_POSE_LOW, _POSE_HIGH, (1, 6)).astype(np.float32)

        with torch.no_grad():
            V_pt, _ = lya_pt(
                torch.tensor(pose),
                torch.tensor(target)
            )

        _, V_tfl = fused(
            torch.zeros((1, 3, 200, 300)),
            pose,
            target
        )

        diffs.append(abs(V_pt.item() - V_tfl))

    diffs = np.array(diffs)

    print(f"\n=== Lyapunov (Fused) n={n} ===")
    print(f"  max  abs diff : {diffs.max():.6f}")
    print(f"  mean abs diff : {diffs.mean():.6f}")
    print(f"  p95  abs diff : {np.percentile(diffs, 95):.6f}")

    print(f"  PT  last: {V_pt.item():.6f}")
    print(f"  TFL last: {V_tfl:.6f}")

    return diffs


# =============================
# MAIN
# =============================
if __name__ == "__main__":

    print("Loading PyTorch models ...")
    ctrl_pt, lya_pt = load_pt_models()

    print("Loading TFLite fused model ...")
    fused = load_tflite_fused()

    ctrl_diffs = compare_controller(ctrl_pt, fused, n=50)
    lya_diffs = compare_lyapunov(lya_pt, fused, n=100)

    print("\n=== Summary ===")
    print(f"  Controller max diff: {ctrl_diffs.max():.4f}")
    print(f"  Lyapunov   max diff: {lya_diffs.max():.4f}")