"""Phase A — behavior cloning of the geometric expert (04_design.md §3).

Collect expert rollouts on the randomized plant with DR'd splat observations,
train the recurrent policy on sequence chunks (burn-in TBPTT), then optional
DAgger rounds where the STUDENT flies and the expert (with its integrator
replayed along the student's trajectory) provides labels.

Run:  python pixel2ctbr/train_bc.py [--smoke]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, "pixel2ctbr")
from dynamics import G  # noqa: E402
from env import EnvConfig, HoverEnv  # noqa: E402
from policy import PixelCTBRPolicy, RATE_LIM, C_SPAN  # noqa: E402

DEV = "cuda"


# ------------------------------------------------------------------ data
@torch.no_grad()
def collect_expert(env: HoverEnv, rounds: int, T: int):
    """rounds × (reset + T-step expert rollout). Returns CPU tensors
    imgs uint8 (N,T,1,H,W), vecs (N,T,12), acts (N,T,4) with N=rounds*B."""
    I, V, A = [], [], []
    t0 = time.time()
    for r in range(rounds):
        env.reset()
        imgs, vecs, acts = env.expert_rollout(T)          # (T,B,...)
        I.append(imgs.transpose(0, 1).contiguous())
        V.append(vecs.transpose(0, 1).cpu())
        A.append(acts.transpose(0, 1).cpu())
        if r == 0 or (r + 1) % 5 == 0:
            fps = (r + 1) * env.cfg.B * T / (time.time() - t0)
            print(f"  collect {r+1}/{rounds}  ({fps:.0f} frames/s)")
    return torch.cat(I), torch.cat(V), torch.cat(A)


@torch.no_grad()
def collect_dagger(env: HoverEnv, policy, rounds: int, T: int):
    """Student flies; expert labels its visited states (expert integrator
    replayed along the student trajectory)."""
    I, V, A = [], [], []
    policy.eval()
    for _ in range(rounds):
        env.reset()
        h = policy.init_hidden(env.cfg.B, DEV)
        imgs, vecs, acts = [], [], []
        for _ in range(T):
            img, vec = env.observe()
            label = env.expert(env.state, env.tgt_p, env.tgt_yaw)
            a, h = policy(img, vec, h)
            imgs.append((img * 255).to(torch.uint8).cpu())
            vecs.append(vec.cpu())
            acts.append(label.cpu())
            env.state = env.dyn.step(env.state, a, env.params)
            env.last_action = a
        I.append(torch.stack(imgs).transpose(0, 1).contiguous())
        V.append(torch.stack(vecs).transpose(0, 1))
        A.append(torch.stack(acts).transpose(0, 1))
    policy.train()
    return torch.cat(I), torch.cat(V), torch.cat(A)


def action_loss(pred, target):
    """Huber on normalized channels: thrust /0.9G, rates /limits."""
    scale = torch.tensor([C_SPAN, *RATE_LIM], device=pred.device)
    return torch.nn.functional.huber_loss(pred / scale, target / scale)


# ------------------------------------------------------------------ train
def train_epochs(policy, data, epochs, chunk=32, burn_in=8, bs=24, lr=3e-4,
                 log=print):
    imgs, vecs, acts = data                                # (N,T,...) CPU
    N, T = imgs.shape[0], imgs.shape[1]
    n_chunks = T // chunk
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    for ep in range(epochs):
        perm = torch.randperm(N * n_chunks)
        tot, nb = 0.0, 0
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            ri, ci = idx // n_chunks, (idx % n_chunks) * chunk
            im = torch.stack([imgs[r, c:c + chunk] for r, c in zip(ri, ci)])
            ve = torch.stack([vecs[r, c:c + chunk] for r, c in zip(ri, ci)])
            ac = torch.stack([acts[r, c:c + chunk] for r, c in zip(ri, ci)])
            im = im.to(DEV).float() / 255.0                # (b,chunk,1,H,W)
            ve, ac = ve.to(DEV), ac.to(DEV)
            h = policy.init_hidden(im.shape[0], DEV)
            loss = 0.0
            for t in range(chunk):
                a, h = policy(im[:, t], ve[:, t], h)
                if t >= burn_in:
                    loss = loss + action_loss(a, ac[:, t])
            loss = loss / (chunk - burn_in)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            tot += loss.item(); nb += 1
        log(f"  epoch {ep+1}/{epochs}  bc_loss {tot/nb:.4f}")
    return policy


@torch.no_grad()
def closed_loop_eval(env: HoverEnv, policy, T=240, resets=4):
    policy.eval()
    ms = []
    for _ in range(resets):
        env.reset()
        env.policy_rollout(policy, T, no_grad=True)
        ms.append(env.success_metrics())
    policy.train()
    agg = {k: sum(m[k] for m in ms) / len(ms) for k in ms[0]}
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--T", type=int, default=160)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--dagger", type=int, default=2)
    ap.add_argument("--out", default="weights/pixel_ctbr_bc.pt")
    args = ap.parse_args()
    if args.smoke:
        args.rounds, args.T, args.epochs, args.dagger = 2, 64, 2, 1

    torch.manual_seed(0)
    env = HoverEnv(EnvConfig(B=64))
    env.seed(42)
    policy = PixelCTBRPolicy().to(DEV)
    policy.train()

    print(f"[phase A] collecting {args.rounds}x64x{args.T} expert frames…")
    data = collect_expert(env, args.rounds, args.T)
    print(f"  dataset: {data[0].shape[0]} rollouts x {data[0].shape[1]} steps "
          f"({data[0].numel() / 2**30:.2f} GiB uint8)")
    train_epochs(policy, data, args.epochs)
    m = closed_loop_eval(env, policy)
    print(f"[phase A] BC closed-loop: {m}")

    for d in range(args.dagger):
        print(f"[phase A] DAgger round {d+1}: student flies, expert labels…")
        new = collect_dagger(env, policy, max(2, args.rounds // 3), args.T)
        data = tuple(torch.cat((a, b)) for a, b in zip(data, new))
        train_epochs(policy, data, max(2, args.epochs // 2))
        m = closed_loop_eval(env, policy)
        print(f"[phase A] after DAgger {d+1}: {m}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"model": policy.state_dict(), "metrics": m}, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
