"""Export PixelCTBRPolicy to TFLite fp16 via the spike-validated chain
(04_design.md §6): PT -> ONNX(18, dynamo=False) -> onnx2tf -> take onnx2tf's
own *_float16.tflite. Then run a LONG closed-loop parity check (hidden state
fed back) — the recurrent analogue of debug_pt_vs_tflite.py.

Run in the tfexport env:
  CUDA_VISIBLE_DEVICES= ~/miniconda3/envs/tfexport/bin/python \
      pixel2ctbr/export_policy.py [--weights weights/pixel_ctbr_bptt.pt]
"""

import argparse
import os
import shutil
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch

sys.path.insert(0, "pixel2ctbr")
from policy import PixelCTBRPolicy, VEC_DIM  # noqa: E402

H_IMG, W_IMG = 96, 128


class ExportWrapper(torch.nn.Module):
    """onnx2tf chokes on per-frame channel Slice ops in features() (axis
    bookkeeping after NCHW->NHWC, same converter-bug genus as the legacy
    non-dividing-pool). This wrapper builds the mean-sub channels with a
    single cat -> channel order [cur, prev, cur-m, prev-m] instead of the
    training order [cur, cur-m, prev, prev-m], and compensates by permuting
    conv1's input-channel weights. Verified numerically by the parity check."""

    def __init__(self, m: PixelCTBRPolicy):
        super().__init__()
        import copy
        self.m = copy.deepcopy(m)
        w = self.m.trunk[0][0].weight.data          # (16, 2*frames, 5, 5)
        f = self.m.frames
        # training order idx of each export-order channel:
        # export ch j in [0..f-1] = raw frame j   -> training idx 2*j
        # export ch f+j          = frame j - mean -> training idx 2*j+1
        perm = [2 * j for j in range(f)] + [2 * j + 1 for j in range(f)]
        self.m.trunk[0][0].weight.data = w[:, perm].contiguous()

    def forward(self, image, vec, h):
        mimg = torch.nn.functional.adaptive_avg_pool2d(image, 1)
        x = torch.cat((image, image - mimg), dim=1)
        fmap = self.m.trunk(x)
        feat = torch.cat((self.m.global_pool(fmap).flatten(1),
                          self.m.lat(fmap).flatten(1),
                          self.m.vert(fmap).flatten(1)), dim=-1)
        z = torch.cat((self.m.img_proj(feat), self.m.vec_mlp(vec)), dim=-1)
        h2 = self.m.gru(z, h)
        raw = self.m.head(torch.cat((h2, z), dim=-1))
        from policy import C_CENTER, C_SPAN, RATE_LIM, clamp_relu
        c = C_CENTER + clamp_relu(raw[:, :1], 1.0) * C_SPAN
        rl = torch.tensor(RATE_LIM)
        w_ = clamp_relu(raw[:, 1:] * rl, rl)
        return torch.cat((c, w_), dim=-1), h2


def export(weights, out_dir="weights", steps=1000, out_base=None):
    """out_base: basename for the artifacts (default: derived from the
    weights filename, e.g. pixel_ctbr_two_gate.pt -> pixel_ctbr_two_gate)."""
    if out_base is None:
        b = os.path.splitext(os.path.basename(weights or ""))[0]
        out_base = b if b.startswith("pixel_ctbr") else "pixel_ctbr"
    torch.manual_seed(0)
    m = PixelCTBRPolicy()
    if weights and os.path.exists(weights):
        m.load_state_dict(torch.load(weights, map_location="cpu",
                                     weights_only=True)["model"])
        print(f"loaded {weights}")
    else:
        print("WARNING: exporting random-init weights (parity test only)")
    m.eval()
    hid = m.hidden
    mx = ExportWrapper(m).eval()

    onnx_path = f"{out_dir}/{out_base}.onnx"
    torch.onnx.export(
        mx, (torch.zeros(1, m.frames, H_IMG, W_IMG), torch.zeros(1, VEC_DIM),
             torch.zeros(1, hid)),
        onnx_path, opset_version=18, dynamo=False,
        input_names=["image", "vec", "h_in"], output_names=["action", "h_out"])

    import onnx2tf
    sm = f"{out_dir}/{out_base}_sm"
    shutil.rmtree(sm, ignore_errors=True)
    onnx2tf.convert(input_onnx_file_path=onnx_path, output_folder_path=sm,
                    disable_group_convolution=True, non_verbose=True)
    tfl = f"{sm}/{out_base}_float16.tflite"
    assert os.path.exists(tfl), "onnx2tf did not emit fp16 tflite"
    final = f"{out_dir}/{out_base}.tflite"
    shutil.copy(tfl, final)
    print(f"exported {final} ({os.path.getsize(final)/1024:.0f} KB)")

    # ---- closed-loop parity ----
    import tensorflow as tf
    it = tf.lite.Interpreter(model_path=final)
    it.allocate_tensors()
    img_d = vec_d = h_d = None
    for d in it.get_input_details():
        shp = list(d["shape"])
        if len(shp) == 4:
            img_d = d
        elif shp[-1] == hid:
            h_d = d
        elif shp[-1] == VEC_DIM:
            vec_d = d
    nhwc = img_d["shape"][-1] == m.frames   # onnx2tf converts NCHW->NHWC
    outs = it.get_output_details()

    rng = np.random.default_rng(0)
    h_pt = torch.zeros(1, hid)
    h_tf = np.zeros((1, hid), np.float32)
    max_a = max_h = 0.0
    with torch.no_grad():
        for _ in range(steps):
            img = rng.random((1, m.frames, H_IMG, W_IMG), dtype=np.float32)
            vec = (rng.standard_normal((1, VEC_DIM)) * 0.5).astype(np.float32)
            a_pt, h_pt = m(torch.from_numpy(img), torch.from_numpy(vec), h_pt)
            it.set_tensor(img_d["index"], img.transpose(0, 2, 3, 1) if nhwc else img)
            it.set_tensor(vec_d["index"], vec)
            it.set_tensor(h_d["index"], h_tf)
            it.invoke()
            got = {o["index"]: it.get_tensor(o["index"]) for o in outs}
            a_tf = next(v for v in got.values() if v.shape[-1] == 4)
            h_tf = next(v for v in got.values() if v.shape[-1] == hid)
            max_a = max(max_a, float(np.abs(a_pt.numpy() - a_tf).max()))
            max_h = max(max_h, float(np.abs(h_pt.numpy() - h_tf).max()))
    print(f"{steps}-step closed-loop parity: max action diff {max_a:.2e}, "
          f"max hidden diff {max_h:.2e}")
    ok = max_a < 1e-2  # actions in physical units (m/s^2, rad/s)
    print("PARITY OK" if ok else "PARITY FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights/pixel_ctbr_bptt.pt")
    ap.add_argument("--steps", type=int, default=1000)
    a = ap.parse_args()
    sys.exit(export(a.weights, steps=a.steps))
