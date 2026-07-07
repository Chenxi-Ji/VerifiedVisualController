"""Sim evaluation gate for pixel2ctbr policies (04_design.md §5, gate B2).

512 episodes x 8 s under full DR + ablation matrix:
  base       — full DR (the headline number; gate ≥95%)
  no_tilt    — tilt input zeroed for ALL episodes (dropout robustness)
  delay+1    — one extra control step of transport delay
  gain_edges — thrust_gain pinned to 0.85 / 1.15 extremes
  dr_off     — image DR disabled (twin-overfit check: score should stay
               similar; a big JUMP means the policy leans on DR-free renders
               — the legacy gate-swap failure smell)

Run: python pixel2ctbr/eval_policy.py --weights weights/pixel_ctbr_bptt.pt
"""

import argparse
import json
import sys

import torch

sys.path.insert(0, "pixel2ctbr")
from env import EnvConfig, HoverEnv  # noqa: E402
from policy import PixelCTBRPolicy  # noqa: E402


@torch.no_grad()
def run_block(env: HoverEnv, policy, episodes, T, mutate=None):
    B = env.cfg.B
    outs = []
    for r in range(episodes // B):
        env.reset()
        if mutate:
            mutate(env)
        env.policy_rollout(policy, T, no_grad=True)
        outs.append(env.success_metrics())
    return {k: sum(o[k] for o in outs) / len(outs) for k in outs[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights/pixel_ctbr_bptt.pt")
    ap.add_argument("--episodes", type=int, default=512)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--out", default="pixel2ctbr/eval_results.json")
    args = ap.parse_args()

    torch.manual_seed(0)
    cfg = EnvConfig(B=64)
    policy = PixelCTBRPolicy().to(cfg.device)
    ck = torch.load(args.weights, weights_only=True, map_location=cfg.device)
    policy.load_state_dict(ck["model"])
    policy.eval()
    T = int(args.seconds / cfg.dt_ctrl)

    def no_tilt(env):
        env.tilt_mask.zero_()

    def delay_plus(env):
        env.params.delay_steps = (env.params.delay_steps + 1).clamp(
            max=env.dyn.max_delay)

    def gain_edges(env):
        half = env.cfg.B // 2
        env.params.thrust_gain[:half] = 0.85
        env.params.thrust_gain[half:] = 1.15

    results = {}
    env = HoverEnv(cfg)
    env.seed(1234)
    results["base"] = run_block(env, policy, args.episodes, T)
    env.seed(1234)
    results["no_tilt"] = run_block(env, policy, args.episodes, T, no_tilt)
    env.seed(1234)
    results["delay_plus1"] = run_block(env, policy, args.episodes, T, delay_plus)
    env.seed(1234)
    results["gain_edges"] = run_block(env, policy, args.episodes, T, gain_edges)
    env_nodr = HoverEnv(cfg, renderer=env.renderer, image_dr=False)
    env_nodr.seed(1234)
    results["dr_off"] = run_block(env_nodr, policy, args.episodes, T)

    for k, v in results.items():
        print(f"{k:>12}: success {v['success']*100:5.1f}%  "
              f"err_med {v['err_med']*100:5.1f} cm  p95 {v['err_p95']*100:5.1f} cm  "
              f"crash {v['crash']*100:4.1f}%")
    with open(args.out, "w") as f:
        json.dump({"weights": args.weights, "episodes": args.episodes,
                   "results": results}, f, indent=2)
    print(f"wrote {args.out}")
    ok = results["base"]["success"] >= 0.95
    print("GATE B2:", "PASS" if ok else "not yet")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
