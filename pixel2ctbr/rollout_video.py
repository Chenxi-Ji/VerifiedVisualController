"""Render a closed-loop rollout video: expert (or a policy) flying the CTBR
plant, camera frames from the splat bridge. End-to-end integration check of
dynamics + render_bridge, and the eval-video tool for later policies.

Usage: python pixel2ctbr/rollout_video.py [out.mp4]
"""

import sys

import numpy as np
import torch

sys.path.insert(0, "pixel2ctbr")
from dynamics import DynParams, QuadCTBRDynamics, quat_from_euler_zyx, euler_zyx_from_quat
from expert import GeometricHoverExpert
from render_bridge import SplatRenderer, METERS_PER_UNIT


def main(out="pixel2ctbr/spike_out/expert_rollout.mp4", B=4, T=6.0, fps_view=20):
    torch.manual_seed(1)
    dyn = QuadCTBRDynamics(dt_ctrl=0.025, n_sub=5)
    g = torch.Generator().manual_seed(11)
    par = DynParams.randomized(B, g=g)
    br = SplatRenderer(width=256, height=192, gray=False, supersample=2,
                       mount_jitter_rad=0.5 * np.pi / 180, intrinsics_jitter=1.0)
    br.sample_episode_dr(B, g=torch.Generator(device="cuda").manual_seed(5))

    tgt = torch.tensor([0.0, 1.5 * METERS_PER_UNIT, 0.0]).expand(B, 3)
    ty = torch.full((B,), -np.pi / 2)
    u = lambda lo, hi: lo + (hi - lo) * torch.rand(B, generator=g)
    p0 = torch.stack((u(-1.2, 1.2) * METERS_PER_UNIT,
                      (1.5 + u(-0.8, 1.2)) * METERS_PER_UNIT,
                      u(-0.4, 0.3) * METERS_PER_UNIT), dim=-1)
    q0 = quat_from_euler_zyx(-np.pi / 2 + u(-0.5, 0.5), u(-0.08, 0.08), u(-0.08, 0.08))
    s = dyn.make_state(p0, torch.zeros(B, 3), q0, torch.zeros(B, 3), par)
    ex = GeometricHoverExpert()

    every = max(1, int(round(1.0 / (fps_view * dyn.dt_ctrl))))
    frames, traces = [], {k: [] for k in ["t", "p", "act", "err"]}
    for k in range(int(T / dyn.dt_ctrl)):
        a = ex(s, tgt, ty)
        if k % every == 0:
            img = br.render_state(s)  # (B,3,H,W) color
            frames.append((img.cpu().numpy() * 255).astype(np.uint8))
            traces["t"].append(k * dyn.dt_ctrl)
            traces["p"].append(s.p.numpy().copy())
            traces["act"].append(a.numpy().copy())
            traces["err"].append((s.p - tgt).norm(dim=-1).numpy().copy())
        s = dyn.step(s, a, par)

    import cv2
    H, W = frames[0].shape[2], frames[0].shape[3]
    grid_w, grid_h = 2 * W, 2 * H + 40
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps_view,
                         (grid_w, grid_h))
    for i, fr in enumerate(frames):
        canvas = np.zeros((grid_h, grid_w, 3), np.uint8)
        for b in range(min(B, 4)):
            r, c = divmod(b, 2)
            im = fr[b].transpose(1, 2, 0)[..., ::-1].copy()
            e = traces["err"][i][b]
            act = traces["act"][i][b]
            cv2.putText(im, f"err {e*100:5.1f}cm  c {act[0]:4.1f}  w [{act[1]:+.2f} {act[2]:+.2f} {act[3]:+.2f}]",
                        (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 255, 0), 1)
            canvas[r * H:(r + 1) * H, c * W:(c + 1) * W] = im
        cv2.putText(canvas, f"t = {traces['t'][i]:4.1f} s   expert on randomized CTBR plant, splat camera",
                    (8, grid_h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        vw.write(canvas)
    vw.release()
    fin = np.array(traces["err"][-1])
    print(f"wrote {out} | final errs (cm): {np.round(fin*100,1)}")


if __name__ == "__main__":
    main(*sys.argv[1:2])
