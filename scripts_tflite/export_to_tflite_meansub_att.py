"""Export ControllerMeansubAtt (+ its co-trained Lyapunov V) -> ONNX -> TFLite.

Thin variant of export_to_tflite.py for weights/ctrl_lya_meansub_att.pt
(3ch mean-subtracted-RGB input, attitude+thrust head, 56,836 params — see
docs/58k_meansub_att_retrain.md). Same route as the original export, reusing
its LyapunovV / FusedModel wrappers and Step-2/Step-3 helpers verbatim:

    PyTorch -> ONNX (opset 18) -> onnx2tf SavedModel -> TFLite

Differences vs the original script: the checkpoint path, the controller class
(Controller -> ControllerMeansubAtt), and the artifact names. The fused graph
signature is IDENTICAL to ctrl_lya.tflite — inputs (image, pose, target),
outputs (action, V) — so the existing replay/debug tooling works unchanged.
Action semantics differ: [c m/s^2, roll_sp, pitch_sp, yaw_sp] instead of
[vx, vy, vz, yaw_rate].

Outputs:
    weights/ctrl_lya_meansub_att.tflite      float16 (same quantization
                                             choices as ctrl_lya.tflite)
    weights/ctrl_lya_meansub_att_f32.tflite  float32 reference (no quant)

Run on CPU:
    conda run -n certified_visual_controller \
        python scripts_tflite/export_to_tflite_meansub_att.py
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys
import shutil

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)
# utils_ctrl_meansub_att uses bare `from utils_ctrl_lya_pt import ...` /
# `from render_image import ...`, so scripts_control must be on sys.path too
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "scripts_control"))

import torch

from scripts_control.utils_ctrl_lya_pt import Lyapunov
from utils_ctrl_meansub_att import ControllerMeansubAtt
from export_to_tflite import (LyapunovV, FusedModel, onnx_to_saved_model,
                              saved_model_to_tflite_f16)

CKPT = os.path.join(_PROJECT_ROOT, "weights/ctrl_lya_meansub_att.pt")


# =============================
# STEP 1 — PyTorch → ONNX  (same call as the original, class/path swapped)
# =============================
def export_onnx(weights_dir):
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)

    ctrl = ControllerMeansubAtt().eval()
    ctrl.load_state_dict(ckpt["controller"])

    lya_base = Lyapunov().eval()
    lya_base.load_state_dict(ckpt["lyapunov"])
    lya = LyapunovV(lya_base).eval()

    fused = FusedModel(ctrl, lya).eval()

    fused_onnx = os.path.join(weights_dir, "fused_meansub_att.onnx")
    torch.onnx.export(
        fused,
        (torch.randn(1, 3, 192, 256), torch.randn(1, 6), torch.randn(1, 6)),
        fused_onnx,
        input_names=["image", "pose", "target"],
        output_names=["action", "V"],
        opset_version=18,
        do_constant_folding=True,
    )
    print(f"  fused_meansub_att.onnx → {fused_onnx}")
    return fused_onnx


# =============================
# STEP 3b — SavedModel → TFLite float32 reference (no quantization)
# =============================
def saved_model_to_tflite_f32(saved_model_dir, tflite_path):
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    tflite_bytes = converter.convert()
    with open(tflite_path, "wb") as f:
        f.write(tflite_bytes)
    print(f"  TFLite f32  → {tflite_path}  ({len(tflite_bytes)/1024:.1f} KB)")


# =============================
# MAIN
# =============================
if __name__ == "__main__":
    import tensorflow as tf
    print(f"TensorFlow {tf.__version__}  |  PyTorch {torch.__version__}")

    weights_dir = os.path.join(_PROJECT_ROOT, "weights")

    print("\n=== Step 1: PyTorch → ONNX ===")
    fused_onnx = export_onnx(weights_dir)

    print("\n=== Step 2: ONNX → SavedModel (onnx2tf) ===")
    fused_saved = os.path.join(weights_dir, "fused_meansub_att_tf_saved")
    onnx_to_saved_model(fused_onnx, fused_saved)

    print("\n=== Step 3: SavedModel → TFLite float16 (same quant as original) ===")
    saved_model_to_tflite_f16(
        fused_saved, os.path.join(weights_dir, "ctrl_lya_meansub_att.tflite"))

    print("\n=== Step 3b: SavedModel → TFLite float32 reference ===")
    saved_model_to_tflite_f32(
        fused_saved, os.path.join(weights_dir, "ctrl_lya_meansub_att_f32.tflite"))

    os.remove(fused_onnx)
    data = fused_onnx + ".data"
    if os.path.exists(data):
        os.remove(data)
    shutil.rmtree(fused_saved, ignore_errors=True)

    print("\n=== Done: ctrl_lya_meansub_att{,_f32}.tflite written to weights/ ===")
