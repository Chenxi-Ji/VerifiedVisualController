"""Export Controller and Lyapunov PyTorch models → ONNX → TFLite float16.

Float16 is fully deterministic — weights are cast to f16, all arithmetic
stays in f32 internally, no representative dataset required.
Result: max diff vs PyTorch ≈ 0.002 (vs up to 0.35 for INT8).

Run in pftolite conda env:
    conda run -n pftolite python scripts_control/export_to_tflite.py
"""
import os
import sys
import shutil
from pathlib import Path


# _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# _PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
# sys.path.insert(0, _SCRIPT_DIR)


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)

# add project root (NOT script dir)
sys.path.insert(0, _PROJECT_ROOT)

import torch
import torch.nn as nn
from scripts_control.utils_ctrl_lya_pt import Controller, Lyapunov


# =============================
# LYAPUNOV WRAPPER — export-safe, V only
# =============================
class LyapunovV(nn.Module):
    """Outputs only V; replaces torch.norm with sqrt(sum(x²)) to avoid
    the ReduceL2 axis=np.int64 bug in older onnx2tf versions."""
    def __init__(self, lya):
        super().__init__()
        self.alpha_net = lya.alpha_net
        self.pos_scale = lya.pos_scale
        self.ang_scale = lya.ang_scale
        self.v_scale   = lya.v_scale

    def forward(self, x, target):
        pos_error = x[:, :3] - target[:, :3]
        ang_error = x[:, 3:4] - target[:, 3:4]

        pos_term = (pos_error ** 2).sum(dim=-1)
        ang_term = (1.0 - torch.cos(ang_error)).sum(dim=-1)

        pos_norm = ((pos_error ** 2).sum(dim=-1, keepdim=True) + 1e-12).sqrt()
        ang_norm = ang_term.unsqueeze(-1)

        alpha_input = torch.cat([pos_norm / self.pos_scale,
                                 ang_norm / self.ang_scale], dim=-1)
        alpha = torch.sigmoid(self.alpha_net(alpha_input)).squeeze(-1)

        return self.v_scale * (alpha * pos_term + (1.0 - alpha) * ang_term)


# =============================
# FUSED MODEL — controller + lyapunov in one graph
# =============================
class FusedModel(nn.Module):
    def __init__(self, ctrl, lya_v):
        super().__init__()
        self.ctrl = ctrl
        self.lya  = lya_v

    def forward(self, image, pose, target):
        action = self.ctrl(image)
        V      = self.lya(pose, target)
        return action, V


# =============================
# STEP 1 — PyTorch → ONNX
# =============================
def export_onnx(weights_dir):
    ckpt = torch.load(
        os.path.join(_PROJECT_ROOT, "weights/ctrl_lya.pt"),
        map_location="cpu",
        weights_only=False,
    )

    ctrl = Controller().eval()
    ctrl.load_state_dict(ckpt["controller"])

    lya_base = Lyapunov().eval()
    lya_base.load_state_dict(ckpt["lyapunov"])
    lya = LyapunovV(lya_base).eval()

    fused = FusedModel(ctrl, lya).eval()

    fused_onnx = os.path.join(weights_dir, "fused.onnx")
    torch.onnx.export(
        fused,
        (torch.randn(1, 3, 200, 300), torch.randn(1, 6), torch.randn(1, 6)),
        fused_onnx,
        input_names=["image", "pose", "target"],
        output_names=["action", "V"],
        opset_version=18,
        do_constant_folding=True,
    )
    print(f"  fused.onnx → {fused_onnx}")
    return fused_onnx


# =============================
# STEP 2 — ONNX → SavedModel
# =============================
def onnx_to_saved_model(onnx_path, out_dir):
    import onnx2tf

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    onnx2tf.convert(
        input_onnx_file_path=onnx_path,
        output_folder_path=out_dir,
        non_verbose=True,
        disable_group_convolution=True,
    )
    print(f"  SavedModel → {out_dir}")
    return out_dir


# =============================
# STEP 3 — SavedModel → TFLite float16
# =============================
def saved_model_to_tflite_f16(saved_model_dir, tflite_path):
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]

    tflite_bytes = converter.convert()
    with open(tflite_path, "wb") as f:
        f.write(tflite_bytes)
    print(f"  TFLite f16  → {tflite_path}  ({len(tflite_bytes)/1024:.1f} KB)")


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
    fused_saved = os.path.join(weights_dir, "fused_tf_saved")
    onnx_to_saved_model(fused_onnx, fused_saved)

    print("\n=== Step 3: SavedModel → TFLite float16 ===")
    saved_model_to_tflite_f16(fused_saved, os.path.join(weights_dir, "ctrl_lya.tflite"))

    os.remove(fused_onnx)
    data = fused_onnx + ".data"
    if os.path.exists(data):
        os.remove(data)
    shutil.rmtree(fused_saved, ignore_errors=True)

    print("\n=== Done: fused.tflite written to weights/ ===")
