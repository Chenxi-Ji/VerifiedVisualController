"""Recurrent pixel+IMU -> CTBR policy (04_design.md §2).

Grayscale 96x128 frame + 12-D proprio vector + GRU hidden state ->
[c, wx, wy, wz]. All ops on the verified TFLite-export list (dividing pools,
clamp_relu, GRUCell — see spike_gru_export.py). Geometry note: no front
AvgPool (legacy had one because its input was 192x256); conv trunk lands on
the same 6x8 map the legacy readouts were designed around.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from dynamics import G

RATE_LIM = (4.0, 4.0, 2.0)          # rad/s, matches QuadCTBRDynamics.RATE_LIMIT
C_CENTER, C_SPAN = G, 0.9 * G       # c = G + clamp_relu(.,1)*0.9G  in [0.1G, 1.9G]
VEC_DIM = 12                        # gyro3 accel3 tilt2 last_action4


def clamp_relu(x, lim):
    """Exact ReLU-only clamp to [-lim, lim] (CROWN/TFLite-friendly; legacy)."""
    return torch.relu(x + lim) - torch.relu(x - lim) - lim


def normalize_vec(gyro, accel, tilt, last_action):
    """(B,3),(B,3),(B,2),(B,4) -> (B,12) normalized proprio vector."""
    a = last_action
    a_n = torch.cat(((a[:, :1] - C_CENTER) / C_SPAN,
                     a[:, 1:] / torch.tensor(RATE_LIM, device=a.device)), dim=-1)
    return torch.cat((gyro / 4.0, accel / 20.0, tilt / 0.5, a_n), dim=-1)


class PixelCTBRPolicy(nn.Module):
    def __init__(self, hidden=96, frames=2):
        """frames=2: input is [current, previous] grayscale frames — visual
        velocity via frame differencing. The privileged-velocity probe showed
        velocity information was the plateau (05 log); two frames are the
        deployable source of it (model helper caches one frame)."""
        super().__init__()
        self.hidden = hidden
        self.frames = frames
        conv = lambda i, o, k, s, p: nn.Sequential(
            nn.Conv2d(i, o, k, stride=s, padding=p), nn.BatchNorm2d(o), nn.ReLU())
        self.trunk = nn.Sequential(                     # in: (B,2*frames,96,128)
            conv(2 * frames, 16, 5, 2, 2),              # 48x64
            conv(16, 32, 3, 2, 1),                      # 24x32
            conv(32, 48, 3, 2, 1),                      # 12x16
            conv(48, 64, 3, 2, 1),                      # 6x8
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))              # 64
        self.lat = nn.Sequential(nn.Conv2d(64, 8, 1), nn.BatchNorm2d(8),
                                 nn.ReLU(), nn.AdaptiveAvgPool2d((1, 4)))  # 32
        self.vert = nn.Sequential(nn.Conv2d(64, 8, 1), nn.BatchNorm2d(8),
                                  nn.ReLU(), nn.AdaptiveAvgPool2d((3, 1)))  # 24
        self.img_proj = nn.Sequential(nn.Linear(120, 64), nn.ReLU())
        self.vec_mlp = nn.Sequential(nn.Linear(VEC_DIM, 32), nn.ReLU())
        self.gru = nn.GRUCell(64 + 32, hidden)
        # head reads hidden state AND the current fused features (skip
        # connection): rate control wants fresh visual feedback, memory
        # serves estimation — routing everything through the GRU halved the
        # lateral rate response (measured, 05 log). Matches Geles/GRaD-Nav
        # actor designs (features + latent both feed the MLP).
        self.head = nn.Sequential(nn.Linear(hidden + 64 + 32, 64), nn.ReLU(),
                                  nn.Linear(64, 4))
        # start near hover: zero the last layer so pre-clamp logits ~0 -> c≈G
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def features(self, image):
        """image (B,frames,96,128) in [0,1] (dim1: [current, previous]) ->
        (B,120)."""
        # mean-sub via avg-pool, NOT .mean(): ReduceMean exports to a TFLite
        # MEAN op with an INT64 axis the runtime rejects; AveragePool is on
        # the verified-op list (export_policy.py finding, 05 log)
        m = torch.nn.functional.adaptive_avg_pool2d(image, 1)
        # per-frame [raw, raw - mean] pairs, current frame first (so the
        # 2-frame graft can zero-init the previous-frame channels)
        parts = []
        for i in range(image.shape[1]):
            parts += [image[:, i:i + 1], image[:, i:i + 1] - m[:, i:i + 1]]
        x = torch.cat(parts, dim=1)
        f = self.trunk(x)
        return torch.cat((self.global_pool(f).flatten(1),
                          self.lat(f).flatten(1),
                          self.vert(f).flatten(1)), dim=-1)

    def forward(self, image, vec, h, return_feat=False):
        """-> action (B,4) [c m/s^2, w rad/s], new hidden (B,H).
        return_feat additionally returns [h2, z] for train-time auxiliary
        heads (velocity supervision); NEVER set during export — the default
        two-output signature is what gets traced."""
        z = torch.cat((self.img_proj(self.features(image)),
                       self.vec_mlp(vec)), dim=-1)
        h2 = self.gru(z, h)
        feat = torch.cat((h2, z), dim=-1)
        raw = self.head(feat)
        c = C_CENTER + clamp_relu(raw[:, :1], 1.0) * C_SPAN
        rl = torch.tensor(RATE_LIM, device=raw.device)
        w = clamp_relu(raw[:, 1:] * rl, rl)
        a = torch.cat((c, w), dim=-1)
        if return_feat:
            return a, h2, feat
        return a, h2

    def init_hidden(self, B, device="cpu"):
        return torch.zeros(B, self.hidden, device=device)


if __name__ == "__main__":
    m = PixelCTBRPolicy()
    n = sum(p.numel() for p in m.parameters())
    img = torch.rand(2, m.frames, 96, 128)
    vec = torch.zeros(2, VEC_DIM)
    a, h = m(img, vec, m.init_hidden(2))
    print(f"params: {n:,} | action {a.shape} hidden {h.shape}")
    print(f"zero-init action (should be ~hover [{G:.2f},0,0,0]): {a[0].tolist()}")
    assert a[:, 0].min() >= 0.1 * G - 1e-5 and a[:, 0].max() <= 1.9 * G + 1e-5
    big = m(img, vec + 100.0, m.init_hidden(2))[0]
    assert big[:, 1].abs().max() <= RATE_LIM[0] + 1e-5
    print("bounds OK")
