"""Slow 360-degree showcase orbits of the three floater-cleaned splat scenes.

Usage (repo root):  python pixel2ctbr/showcase_video.py [single|two|three|all]
Writes pixel2ctbr/spike_out/showcase/{single,two_gate,three_gate_turn}.mp4
(h264 yuv420p +faststart, 1024x768 @ 30 fps, 720 frames = 24 s per scene).

Camera: the measured 1024x768 fisheye calibration (the twin's honest
camera — SplatRenderer at native resolution). The arena is a walled room,
so a circular orbit "beyond the outermost gate" would exit the captured
volume; instead each scene gets a slow constant-rate ELLIPTICAL orbit
fitted inside the safety nets (mat ~|x|<2.4, y in [-3.3,+3.5]), flying
~0.45 m above gate-center height with a gentle look-at pitch to the track
center. The orbit starts on the well-captured +y side; the -y arc shows
the known view-extrapolation softness (07 doc §5.1) — honest, not a bug.

GPU-frugal: chunk<=2 poses per rasterization call, renderer freed +
cache emptied between scenes, OOM retry at chunk=1, and a short sleep
between chunks whenever other processes hold most of the GPU.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "pixel2ctbr")
sys.path.insert(0, "scripts_control")

from dynamics import quat_from_euler_zyx  # noqa: E402
from render_bridge import SplatRenderer  # noqa: E402
import scene_edit as se  # noqa: E402

OUT_DIR = "pixel2ctbr/spike_out/showcase"
W, H, FPS, N_FRAMES = 1024, 768, 30, 720

# scene name -> (orbit center xy, semi-axis a (x), semi-axis b (y),
#                orbit z, look-at point). Gate-frame METERS, z DOWN.
# Ellipses verified against the room bounds and the gate rings: closest
# approach to any ring center >= 0.95 m (rings reach 0.72 m).
ORBITS = {
    "single": ((0.0, 0.0), 2.1, 2.4, -0.45, (0.0, 0.0, 0.15)),
    "two_gate": ((0.0, -1.1), 2.1, 2.05, -0.45, (0.0, -1.1, 0.2)),
    "three_gate_turn": ((0.456, 0.0), 1.75, 2.9, -0.5, (0.456, 0.0, 0.2)),
}


def orbit_poses(name):
    """(N,3) positions + (N,4) wxyz quats: one slow CCW revolution starting
    on the +y (well-captured) side, camera aimed at the track center."""
    (cx, cy), a, b, z, tgt = ORBITS[name]
    phi = np.pi / 2 + 2 * np.pi * np.arange(N_FRAMES) / N_FRAMES
    px = cx + a * np.cos(phi)
    py = cy + b * np.sin(phi)
    pz = np.full(N_FRAMES, z)
    d = np.stack([tgt[0] - px, tgt[1] - py, tgt[2] - pz], 1)
    yaw = np.arctan2(d[:, 1], d[:, 0])
    pitch = -np.arcsin(d[:, 2] / np.linalg.norm(d, axis=1))
    p = torch.tensor(np.stack([px, py, pz], 1), dtype=torch.float32)
    q = quat_from_euler_zyx(torch.tensor(yaw, dtype=torch.float32),
                            torch.tensor(pitch, dtype=torch.float32),
                            torch.zeros(N_FRAMES))
    return p, q


def scene_for(name):
    if name == "single":
        return se.clean_scene()
    if name == "two_gate":
        import env_two_gate
        return se.multi_gate_scene(env_two_gate.DUP_GATE_POSES)
    import env_three_gate_turn
    return se.multi_gate_scene(env_three_gate_turn.DUP_GATE_POSES)


def encode(name, scene):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = f"{OUT_DIR}/{name}.mp4"
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
         "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-preset", "medium", "-crf", "18",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", path],
        stdin=subprocess.PIPE)
    r = SplatRenderer(width=W, height=H, gray=False, supersample=1,
                      chunk=2, scene=scene)
    p, q = orbit_poses(name)
    t0 = time.time()
    i, step = 0, 2
    while i < N_FRAMES:
        try:
            img = r.render_pose_quat(p[i:i + step], q[i:i + step])
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            step = 1
            print("OOM -> chunk=1")
            continue
        arr = (img.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
        ff.stdin.write(arr.tobytes())
        i += arr.shape[0]
        if i % 60 == 0:
            free, total = torch.cuda.mem_get_info()
            if free < 0.25 * total:      # someone else owns the GPU: yield
                time.sleep(0.1)
            print(f"  {name}: {i}/{N_FRAMES} frames "
                  f"({i / (time.time() - t0):.1f} fps)", flush=True)
    ff.stdin.close()
    ff.wait()
    del r
    torch.cuda.empty_cache()
    print(f"saved {path} ({N_FRAMES} frames @ {FPS} fps, "
          f"{time.time() - t0:.0f} s)")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    names = list(ORBITS) if what == "all" else [
        {"single": "single", "two": "two_gate",
         "three": "three_gate_turn"}[what]]
    torch.manual_seed(0)
    for nm in names:
        encode(nm, scene_for(nm))
