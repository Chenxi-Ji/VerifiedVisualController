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
BURN = 6          # unscored GRU warm-up steps at chain start
CHAIN = 13        # scored windows per episode chain (13 x 32 steps at
                  # dt=0.025 = 10.4 s — covers the slow-convergence regime
                  # the 8-16 s probe exposed (tail diagnosis, 05 log))


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
        # near-target precision regime (legacy trainer's "V<=0.02: push V^2->0"
        # analogue): Huber's gradient shrinks linearly with error, so at ~0.5 m
        # the pull no longer beats the act/jerk regularizers + DR noise floor —
        # run-5 orbited the target at 0.4-0.6 m forever (05 log). This term
        # keeps the pressure on inside 1 m.
        en = e_p.norm(dim=-1)
        l_pos = l_pos + wt * 5.0 * (torch.exp(-en / 0.3) * en ** 2).mean()
        # soft time-outside-ring: the differentiable version of the gate's
        # position criterion itself — bills every step spent outside 0.15 m,
        # pressuring convergence SPEED (tail diagnosis: success 71%@8s vs
        # 86%@16s — the tail is slow, not lost)
        l_pos = l_pos + wt * 1.5 * torch.sigmoid((en - 0.15) / 0.04).mean()
        near = torch.exp(-en.detach())                     # damp v near target
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
        # gate-visibility gate: the camera is the ONLY position sensor; a
        # restart state whose camera can't see the gate contributes pure
        # noise gradient. bearing(drone->gate origin) vs yaw within ~52°,
        # in front of the gate, inside a sane box.
        from dynamics import euler_zyx_from_quat as _e
        yaw, _, _ = _e(new["q"])
        bearing = torch.atan2(-new["p"][:, 1], -new["p"][:, 0])
        berr = torch.atan2(torch.sin(bearing - yaw), torch.cos(bearing - yaw))
        keep &= (berr.abs() < 0.9) & (new["p"][:, 1] > 0.3) \
            & (new["p"][:, 0].abs() < 2.0) & (new["p"][:, 1] < 3.2) \
            & (new["p"][:, 2].abs() < 1.0)
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
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", default="weights/pixel_ctbr_bptt.pt")
    ap.add_argument("--polish", action="store_true",
                    help="final polish: BN frozen, cosine lr decay, H=32 only")
    args = ap.parse_args()
    if args.smoke:
        args.epochs, args.windows = 2, 4

    torch.manual_seed(0)
    env = HoverEnv(EnvConfig(B=48))
    env.seed(7)
    policy = PixelCTBRPolicy().to(DEV)
    if os.path.exists(args.init):
        sd = torch.load(args.init, weights_only=True)["model"]
        own = policy.state_dict()
        kept = {k: v for k, v in sd.items()
                if k in own and own[k].shape == v.shape}
        missing = [k for k in own if k not in kept]
        policy.load_state_dict(kept, strict=False)
        # skip-connection graft: old head read only h (hidden); new head reads
        # [h, z]. Copy old weights into the h-slice and zero the z-slice so
        # the grafted policy behaves EXACTLY like the checkpoint at load, and
        # training grows the skip from zero.
        hk = "head.0.weight"
        if hk in sd and hk in missing:
            old_w = sd[hk]
            with torch.no_grad():
                policy.head[0].weight.zero_()
                policy.head[0].weight[:, :old_w.shape[1]].copy_(old_w)
                policy.head[0].bias.copy_(sd["head.0.bias"])
            print(f"grafted {hk}: old {tuple(old_w.shape)} into h-slice of "
                  f"{tuple(policy.head[0].weight.shape)}, skip-slice zeroed")
        ck = "trunk.0.0.weight"   # 2-frame graft: conv1 in-ch 2 -> 4
        if ck in sd and ck in missing:
            old_w = sd[ck]
            with torch.no_grad():
                policy.trunk[0][0].weight.zero_()
                policy.trunk[0][0].weight[:, :old_w.shape[1]].copy_(old_w)
            print(f"grafted {ck}: old {tuple(old_w.shape)} into current-frame "
                  f"channels of {tuple(policy.trunk[0][0].weight.shape)}, "
                  f"prev-frame channels zeroed")
        print(f"warm start from {args.init} ({len(kept)}/{len(own)} tensors; "
              f"fresh: {missing})")
    else:
        print(f"WARNING: no init at {args.init} — cold start (expect divergence risk)")
    # BN unfrozen with slow momentum: the two-frame input changed the trunk's
    # input distribution, so Phase-A stats are stale (run-8 finding); slow
    # momentum keeps deployment stats stable while letting them track
    policy.train()
    for m in policy.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            if args.polish:
                m.eval()          # stats adapted during v9; freeze for polish
            else:
                m.momentum = 0.01

    # auxiliary velocity head (train-time only, never exported): forces the
    # trunk+GRU to encode body velocity from the frame pair — the privileged-v
    # probe proved velocity information is what converges this task (05 log:
    # 0.55 m plateau -> 0.113 m with true v). Supervised on true body v.
    aux_v = torch.nn.Linear(policy.hidden + 96, 3).to(DEV)
    opt = torch.optim.Adam(list(policy.parameters()) + list(aux_v.parameters()),
                           lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=args.lr * 0.1) if args.polish else None
    best_succ = -1.0
    scale = torch.tensor([C_SPAN, *RATE_LIM], device=DEV)
    bc_w = torch.tensor([1.0, 2.0, 2.0, 1.5], device=DEV)
    from dynamics import quat_rotate_inv
    for ep in range(args.epochs):
        H = 32 if args.polish else horizon_for_epoch(ep, args.epochs)
        t0 = time.time()
        agg = {}
        n_win = 0
        # EPISODE-CHAINED WINDOWS (v5): each chain = fresh reset + CHAIN
        # consecutive scored windows, state AND GRU hidden carried (detached)
        # across windows. Slow drift accumulates along the chain exactly as
        # in eval (chains start from rest like eval), so later windows SEE
        # and bill integrated errors that 0.4-0.8 s windows alone cannot —
        # the run-4 failure (05 log: slow z-drift invisible to window
        # losses). Replaces the visited-state buffer entirely (chain states
        # are on-policy and current by construction).
        chains = max(1, args.windows // CHAIN)
        for ci in range(chains):
            env.reset()
            if ci % 3 == 2:
                # fine-approach chains: start settled near the target so the
                # terminal-precision regime gets concentrated training signal
                # (the start box almost never samples it)
                B = env.cfg.B
                env.state.p = env.tgt_p + torch.randn(B, 3, device=DEV) * 0.3
                env.state.v = torch.randn(B, 3, device=DEV) * 0.15
                yaw = env.tgt_yaw + torch.randn(B, device=DEV) * 0.15
                from dynamics import quat_from_euler_zyx
                env.state.q = quat_from_euler_zyx(
                    yaw, torch.zeros(B, device=DEV), torch.zeros(B, device=DEV))
            env.expert.reset()
            _, _, h = env.policy_rollout(policy, BURN, no_grad=True)
            for c in range(CHAIN):
                env.state = env.state.detach()
                states, actions, h, labels, feats = env.policy_rollout(
                    policy, H, h0=h.detach(), with_expert=True, with_feat=True)
                loss, parts = window_loss(env, states, actions)
                # chain-position weighting: late-chain windows are the
                # steady-state regime — weighting them up makes a PARKED
                # 20 cm offset expensive (the anti-steady-state-error
                # pressure; tail diagnosis showed failures park just outside
                # the ring with no start/DR pocket)
                loss = loss * (0.6 + 0.8 * c / (CHAIN - 1))
                # feats[i] sees the PRE-step-i observation -> supervise with
                # the pre-step velocity (= states[i-1]); first feat dropped
                l_aux = sum(
                    HUBER(aux_v(f),
                          quat_rotate_inv(s.q.detach(), s.v.detach()) / 2.0)
                    for f, s in zip(feats[1:], states[:-1])) / max(len(feats) - 1, 1)
                loss = loss + 0.5 * l_aux
                parts["aux"] = 0.5 * float(l_aux)
                # expert anchor: dense gradients that bypass the plant
                # (bootstrap-RL-with-IL); the expert's integrator runs along
                # the whole chain, so its labels carry the integral-action
                # signal the policy must reproduce via its hidden state
                # rates weighted 2x thrust: measured deficit is the lateral
                # rate response (corr 0.3 at half the expert magnitude)
                l_bc = sum((HUBER(a / scale, lb / scale, reduction="none")
                            * bc_w).mean()
                           for a, lb in zip(actions, labels)) / len(actions)
                loss = loss + 0.4 * l_bc
                parts["bc"] = 0.4 * float(l_bc)
                opt.zero_grad()
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
                opt.step()
                n_win += 1
                for k, v in parts.items():
                    agg[k] = agg.get(k, 0.0) + v
                agg["gn"] = agg.get("gn", 0.0) + float(gn)
        msg = " ".join(f"{k} {v/n_win:.4f}" for k, v in agg.items())
        print(f"epoch {ep+1}/{args.epochs} H={H} chain={CHAIN}  {msg}  ({time.time()-t0:.0f}s)")
        if sched is not None:
            sched.step()
        if (ep + 1) % 2 == 0 or ep == args.epochs - 1:
            m = closed_loop_eval(env, policy)              # 8 s (the gate)
            m12 = closed_loop_eval(env, policy, T=480, resets=2)
            print(f"  eval12s: {m12}")
            if args.polish:
                for mod in policy.modules():
                    if isinstance(mod, torch.nn.BatchNorm2d):
                        mod.eval()
            print(f"  eval: {m}")
            torch.save({"model": policy.state_dict(), "metrics": m, "epoch": ep},
                       args.out.replace(".pt", "_last.pt"))
            if m["success"] > best_succ:      # keep the BEST, not the last
                best_succ = m["success"]
                torch.save({"model": policy.state_dict(), "metrics": m,
                            "epoch": ep}, args.out)
                print(f"  new best ({best_succ*100:.1f}%) -> {args.out}")
    print(f"saved best={best_succ*100:.1f}% {args.out}")


if __name__ == "__main__":
    main()
