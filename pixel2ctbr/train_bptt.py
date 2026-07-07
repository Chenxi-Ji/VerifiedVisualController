"""Phase B — truncated-BPTT fine-tune through the differentiable plant
(04_design.md §4). Warm-started from Phase A weights. Images/IMU detached;
gradients flow policy -> dynamics -> privileged-state losses. Horizon
curriculum; Huber-smoothed losses; visited-state restarts.

Run:  python pixel2ctbr/train_bptt.py [--smoke] [--init weights/pixel_ctbr_bc.pt]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, "pixel2ctbr")
from dynamics import G, QuadState, euler_zyx_from_quat  # noqa: E402
from env import EnvConfig, HoverEnv  # noqa: E402
from policy import PixelCTBRPolicy, RATE_LIM, C_SPAN  # noqa: E402
from train_bc import closed_loop_eval  # noqa: E402

DEV = "cuda"
HUBER = torch.nn.functional.huber_loss
BURN = 6          # unscored GRU warm-up steps per window


def window_loss(env: HoverEnv, states, actions):
    """Losses over one BPTT window (04_design §4). All Huber-smoothed."""
    B = env.cfg.B
    T = len(states)
    zero3 = torch.zeros(B, 3, device=DEV)
    zero1 = torch.zeros(B, device=DEV)
    l_pos = l_vel = l_att = l_act = l_jerk = 0.0
    prev_a = None
    for t, (s, a) in enumerate(zip(states, actions)):
        wt = 0.5 + 1.5 * (t + 1) / T                       # time-increasing
        e_p = s.p - env.tgt_p
        l_pos = l_pos + wt * HUBER(e_p, zero3)
        near = torch.exp(-(e_p.detach().norm(dim=-1)))     # damp v near target
        spd = s.v.norm(dim=-1)
        l_vel = l_vel + wt * (near * spd.clamp(max=5.0) ** 2).mean() * 0.5 \
            + (torch.relu(spd - 1.5) ** 2).mean() * 0.3    # global overspeed:
        # short windows otherwise reward sprinting at the target — kinetic
        # energy at window end is free (the epoch-4 divergence, 05 log)
        yaw, pitch, roll = euler_zyx_from_quat(s.q)
        yerr = 1.0 - torch.cos(yaw - env.tgt_yaw)
        tilt_pen = (torch.relu(pitch.abs() - 0.44) ** 2
                    + torch.relu(roll.abs() - 0.44) ** 2)
        l_att = l_att + wt * (yerr + 4.0 * tilt_pen).mean()
        a_n = torch.cat((((a[:, :1] - G) / C_SPAN),
                         a[:, 1:] / torch.tensor(RATE_LIM, device=DEV)), dim=-1)
        l_act = l_act + (a_n ** 2).mean() * 0.02
        if prev_a is not None:
            l_jerk = l_jerk + ((a - prev_a) ** 2).mean() * 0.002
        prev_a = a
    # terminal cost: the window must END slow and close, or truncated BPTT
    # learns arrive-fast myopia (same role as SHAC's terminal critic, cheaper)
    sT = states[-1]
    l_term = 2.0 * HUBER(sT.p - env.tgt_p, zero3) \
        + 1.5 * HUBER(sT.v, zero3)
    n = T
    parts = {"pos": l_pos / n, "vel": l_vel / n, "att": 0.3 * l_att / n,
             "act": l_act / n, "jerk": l_jerk / max(n - 1, 1), "term": l_term}
    return sum(parts.values()), {k: float(v) for k, v in parts.items()}


def horizon_for_epoch(ep, total):
    """16 -> 24 -> 32 -> 32 curriculum (0.4-0.8 s of physics per window).
    NOT starting at 8: the legacy trainer's shortest window was 0.7 s
    (7 steps at dt=0.1); 8 steps at dt=0.025 is only 0.2 s — too short for
    the position loss to be reducible, so gradients chase noise (run1/run2
    divergence, 05 log)."""
    fr = ep / max(total - 1, 1)
    return [16, 24, 32, 32][min(3, int(fr * 4))]


class VisitedBuffer:
    """Restart states harvested from recent rollouts (ABPT trick) so windows
    cover the approach corridor, not just the start box."""

    def __init__(self, cap=4096):
        self.cap = cap
        self.items = None  # dict of stacked tensors

    @torch.no_grad()
    def add(self, states):
        take = [s.detach() for s in states[:: max(1, len(states) // 4)]]
        new = {
            "p": torch.cat([s.p for s in take]),
            "v": torch.cat([s.v for s in take]),
            "q": torch.cat([s.q for s in take]),
            "w": torch.cat([s.w for s in take]),
        }
        keep = torch.isfinite(new["p"]).all(-1) & (new["p"].norm(dim=-1) < 6.0) \
            & (new["v"].norm(dim=-1) < 2.5)   # never restart from runaway states
        new = {k: v[keep] for k, v in new.items()}
        if self.items is None:
            self.items = new
        else:
            self.items = {k: torch.cat((self.items[k], new[k]))[-self.cap:]
                          for k in new}

    @torch.no_grad()
    def sample_into(self, env: HoverEnv, frac=0.5):
        """Replace a fraction of env's freshly-reset states with buffer states."""
        if self.items is None or self.items["p"].shape[0] < 64:
            return
        B = env.cfg.B
        n = int(B * frac)
        idx = torch.randint(self.items["p"].shape[0], (n,), device=DEV)
        sel = torch.arange(n, device=DEV)
        s = env.state
        for k, attr in [("p", "p"), ("v", "v"), ("q", "q"), ("w", "w")]:
            getattr(s, attr)[sel] = self.items[k][idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--init", default="weights/pixel_ctbr_bc.pt")
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--windows", type=int, default=120)   # per epoch
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--out", default="weights/pixel_ctbr_bptt.pt")
    args = ap.parse_args()
    if args.smoke:
        args.epochs, args.windows = 2, 4

    torch.manual_seed(0)
    env = HoverEnv(EnvConfig(B=48))
    env.seed(7)
    policy = PixelCTBRPolicy().to(DEV)
    if os.path.exists(args.init):
        policy.load_state_dict(torch.load(args.init, weights_only=True)["model"])
        print(f"warm start from {args.init}")
    else:
        print(f"WARNING: no init at {args.init} — cold start (expect divergence risk)")
    # BN frozen: stats were DR-calibrated in Phase A (04_design §4)
    policy.train()
    for m in policy.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eval()

    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    buf = VisitedBuffer()
    for ep in range(args.epochs):
        H = horizon_for_epoch(ep, args.epochs)
        t0 = time.time()
        agg = {}
        for w in range(args.windows):
            env.reset()
            buf.sample_into(env, frac=0.5)
            # burn-in: BURN unscored no-grad steps so the GRU has context
            # before any step is billed — otherwise half the gradients come
            # from an h=0-blind recurrent state that never occurs in steady
            # closed-loop flight (Phase A's chunk burn-in served the same role)
            _, _, h = env.policy_rollout(policy, BURN, no_grad=True)
            env.state = env.state.detach()
            states, actions, _ = env.policy_rollout(policy, H, h0=h.detach())
            loss, parts = window_loss(env, states, actions)
            opt.zero_grad()
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            buf.add(states)
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + v
            agg["gn"] = agg.get("gn", 0.0) + float(gn)
        msg = " ".join(f"{k} {v/args.windows:.4f}" for k, v in agg.items())
        print(f"epoch {ep+1}/{args.epochs} H={H}  {msg}  ({time.time()-t0:.0f}s)")
        if (ep + 1) % 4 == 0 or ep == args.epochs - 1:
            m = closed_loop_eval(env, policy)
            # closed_loop_eval sets BN modules back via policy.train(); re-freeze
            for mod in policy.modules():
                if isinstance(mod, torch.nn.BatchNorm2d):
                    mod.eval()
            print(f"  eval: {m}")
            torch.save({"model": policy.state_dict(), "metrics": m, "epoch": ep},
                       args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
