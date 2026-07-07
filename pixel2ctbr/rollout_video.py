"""Render a closed-loop rollout video: expert or a trained policy flying the
CTBR plant, camera frames from the splat bridge.

Usage: python pixel2ctbr/rollout_video.py [out.mp4] [policy_weights.pt]
"""

import sys

import numpy as np
import torch

sys.path.insert(0, "pixel2ctbr")
from dynamics import DynParams, QuadCTBRDynamics, quat_from_euler_zyx, euler_zyx_from_quat
from expert import GeometricHoverExpert
from render_bridge import SplatRenderer, METERS_PER_UNIT


def main(out="pixel2ctbr/spike_out/expert_rollout.mp4", weights=None,
         B=4, T=6.0, fps_view=20):
    torch.manual_seed(1)
    dyn = QuadCTBRDynamics(dt_ctrl=0.025, n_sub=5)
    g = torch.Generator().manual_seed(11)
    par = DynParams.randomized(B, g=g)
    br = SplatRenderer(width=256, height=192, gray=False, supersample=2,
                       mount_jitter_rad=0.5 * np.pi / 180, intrinsics_jitter=1.0)
    br.sample_episode_dr(B, g=torch.Generator(device="cuda").manual_seed(5))

    policy = None
    if weights:
        from env import EnvConfig, HoverEnv
        from policy import PixelCTBRPolicy
        cfg = EnvConfig(B=B)
        penv = HoverEnv(cfg, renderer=SplatRenderer(
            width=128, height=96, chunk=8, gray=True, supersample=2,
            mount_jitter_rad=0.5 * np.pi / 180, intrinsics_jitter=1.0))
        penv.seed(11)
        policy = PixelCTBRPolicy().to("cuda")
        ck = torch.load(weights, weights_only=True, map_location="cuda")
        policy.load_state_dict(ck["model"])
        policy.eval()

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
    if policy is None:
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
    else:
        penv.reset()
        h = policy.init_hidden(B, "cuda")
        tgt_g = penv.tgt_p
        with torch.no_grad():
            for k in range(int(T / penv.cfg.dt_ctrl)):
                img, vec = penv.observe()
                a, h = policy(img, vec, h)
                if k % every == 0:
                    view = br.render_state(penv.state)
                    frames.append((view.cpu().numpy() * 255).astype(np.uint8))
                    traces["t"].append(k * penv.cfg.dt_ctrl)
                    traces["p"].append(penv.state.p.cpu().numpy().copy())
                    traces["act"].append(a.cpu().numpy().copy())
                    traces["err"].append(
                        (penv.state.p - tgt_g).norm(dim=-1).cpu().numpy().copy())
                penv.state = penv.dyn.step(penv.state, a, penv.params)
                penv.last_action = a

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
        who = "policy" if policy is not None else "expert"
        cv2.putText(canvas, f"t = {traces['t'][i]:4.1f} s   {who} on randomized CTBR plant, splat camera",
                    (8, grid_h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        vw.write(canvas)
    vw.release()
    fin = np.array(traces["err"][-1])
    print(f"wrote {out} | final errs (cm): {np.round(fin*100,1)}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
