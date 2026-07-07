"""SPIKE: can a recurrent (GRU) policy with explicit state I/O survive
PT -> ONNX(18) -> onnx2tf -> TFLite fp16, with closed-loop parity?

Decides recurrence vs frame-stacking for the pixel2ctbr policy BEFORE the
design commits. Mirrors scripts_tflite/export_to_tflite.py conventions
(fp16, onnx2tf, tensor-by-name binding). Run in the `tfexport` env:

  CUDA_VISIBLE_DEVICES= ~/miniconda3/envs/tfexport/bin/python pixel2ctbr/spike_gru_export.py

Model shape mimics the intended deployment graph: tiny conv trunk on a
grayscale image + IMU/action-history vector + GRU hidden state in, action +
new hidden state out. Two GRU variants:
  A: torch.nn.GRUCell (let ONNX/onnx2tf decompose it)
  B: hand-rolled GRU cell (explicit matmul/sigmoid/tanh — guaranteed prim ops)
"""

import os
import shutil
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch
import torch.nn as nn

OUT = "pixel2ctbr/spike_out"
H, W, HID, NIMU = 96, 128, 64, 10


class HandGRUCell(nn.Module):
    def __init__(self, nin, nh):
        super().__init__()
        self.x2h = nn.Linear(nin, 3 * nh)
        self.h2h = nn.Linear(nh, 3 * nh)
        self.nh = nh

    def forward(self, x, h):
        gx, gh = self.x2h(x), self.h2h(h)
        xr, xz, xn = gx.chunk(3, 1)
        hr, hz, hn = gh.chunk(3, 1)
        r = torch.sigmoid(xr + hr)
        z = torch.sigmoid(xz + hz)
        n = torch.tanh(xn + r * hn)
        return (1 - z) * n + z * h


class TinyRecurrentPolicy(nn.Module):
    def __init__(self, cell: str):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.AvgPool2d(2),                          # 96x128 -> 48x64
            nn.Conv2d(1, 8, 5, stride=2, padding=2), nn.ReLU(),   # 24x32
            nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.ReLU(),  # 12x16
            nn.Conv2d(16, 24, 3, stride=2, padding=1), nn.ReLU(), # 6x8
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        nin = 24 + NIMU
        self.cell_kind = cell
        self.cell = nn.GRUCell(nin, HID) if cell == "gru" else HandGRUCell(nin, HID)
        self.head = nn.Linear(HID, 4)

    def forward(self, image, imu, h):
        f = self.trunk(image).flatten(1)
        x = torch.cat((f, imu), dim=1)
        h2 = self.cell(x, h)
        return self.head(h2), h2


def export(kind):
    torch.manual_seed(0)
    m = TinyRecurrentPolicy(kind).eval()
    os.makedirs(OUT, exist_ok=True)
    onnx_path = f"{OUT}/{kind}.onnx"
    torch.onnx.export(
        m, (torch.zeros(1, 1, H, W), torch.zeros(1, NIMU), torch.zeros(1, HID)),
        onnx_path, opset_version=18, dynamo=False,
        input_names=["image", "imu", "h_in"], output_names=["action", "h_out"],
    )
    import onnx2tf
    sm_dir = f"{OUT}/{kind}_sm"
    shutil.rmtree(sm_dir, ignore_errors=True)
    onnx2tf.convert(input_onnx_file_path=onnx_path, output_folder_path=sm_dir,
                    disable_group_convolution=True, non_verbose=True)
    # onnx2tf emits <name>_float16.tflite directly; the SavedModel it writes has
    # no serving signature (so from_saved_model() cannot be used here).
    tfl_path = f"{sm_dir}/{kind}_float16.tflite"
    assert os.path.exists(tfl_path), f"onnx2tf did not emit {tfl_path}"
    return m, tfl_path


def closed_loop_parity(m, tfl_path, steps=30):
    import tensorflow as tf
    it = tf.lite.Interpreter(model_path=tfl_path)
    it.allocate_tensors()
    # bind by shape (onnx2tf renames tensors and converts image to NHWC),
    # mirroring the shape/rank-based binding the onboard helper uses
    img_d = imu_d = h_d = None
    for d in it.get_input_details():
        shp = list(d["shape"])
        if len(shp) == 4:
            img_d = d
        elif shp[-1] == HID:
            h_d = d
        elif shp[-1] == NIMU:
            imu_d = d
    assert img_d is not None and imu_d is not None and h_d is not None, \
        [list(d['shape']) for d in it.get_input_details()]
    nhwc = img_d["shape"][-1] == 1  # onnx2tf converts NCHW->NHWC
    outs = it.get_output_details()

    rng = np.random.default_rng(0)
    h_pt = torch.zeros(1, HID)
    h_tf = np.zeros((1, HID), np.float32)
    max_a, max_h = 0.0, 0.0
    for _ in range(steps):
        img = rng.random((1, 1, H, W), dtype=np.float32)
        imu = rng.standard_normal((1, NIMU)).astype(np.float32)
        with torch.no_grad():
            a_pt, h_pt = m(torch.from_numpy(img), torch.from_numpy(imu), h_pt)
        it.set_tensor(img_d["index"],
                      img.transpose(0, 2, 3, 1) if nhwc else img)
        it.set_tensor(imu_d["index"], imu)
        it.set_tensor(h_d["index"], h_tf)
        it.invoke()
        got = {o["index"]: it.get_tensor(o["index"]) for o in outs}
        # identify outputs by shape (4 vs HID), mirroring onboard binding
        a_tf = next(v for v in got.values() if v.shape[-1] == 4)
        h_tf = next(v for v in got.values() if v.shape[-1] == HID)
        max_a = max(max_a, np.abs(a_pt.numpy() - a_tf).max())
        max_h = max(max_h, np.abs(h_pt.numpy() - h_tf).max())
    return max_a, max_h


if __name__ == "__main__":
    for kind in ["gru", "hand"]:
        print(f"=== {kind} ===")
        try:
            m, tfl = export(kind)
            a, h = closed_loop_parity(m, tfl)
            size = os.path.getsize(tfl) / 1024
            print(f"{kind}: EXPORT OK  {size:.0f} KB | 30-step closed-loop parity: "
                  f"max action diff {a:.2e}, max hidden diff {h:.2e}")
        except Exception as e:
            print(f"{kind}: FAILED — {type(e).__name__}: {str(e)[:300]}")
