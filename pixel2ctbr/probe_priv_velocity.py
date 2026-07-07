"""DIAGNOSTIC (sim-only, never deployable): does PRIVILEGED true body
velocity in the proprio vector break the ~0.6 m orbit plateau?

If yes -> the deficit is velocity information (GRU-mediated visual velocity
estimation too weak) and the deployable fix is a 2-frame input.
If no  -> the policy's position-servo representation itself is the problem
-> escalate to the reserve critic / structured output.

Trains a v7-style policy whose vec has 3 extra slots = v_body/2, warm-started
from the run-5 snapshot (vec_mlp reinitialized due to width change), short
run. Run: python pixel2ctbr/probe_priv_velocity.py
"""

import sys
import time

import torch

sys.path.insert(0, "pixel2ctbr")
import policy as P
from dynamics import quat_rotate_inv
from env import EnvConfig, HoverEnv
from train_bptt import (BURN, CHAIN, HUBER, horizon_for_epoch, window_loss)

DEV = "cuda"
NV = 15  # 12 + privileged v_body(3)


class PrivPolicy(P.PixelCTBRPolicy):
    def __init__(self, hidden=96):
        super().__init__(hidden)
        self.vec_mlp = torch.nn.Sequential(torch.nn.Linear(NV, 32), torch.nn.ReLU())


def priv_vec(env):
    img, vec = env.observe()
    vb = quat_rotate_inv(env.state.q.detach(), env.state.v.detach()) / 2.0
    return img, torch.cat((vec, vb), dim=-1)


def rollout(env, pol, H, h, with_expert=True):
    states, actions, labels = [], [], []
    for _ in range(H):
        img, vec = priv_vec(env)
        if with_expert:
            with torch.no_grad():
                labels.append(env.expert(env.state.detach(), env.tgt_p, env.tgt_yaw))
        a, h = pol(img, vec, h)
        env.state = env.dyn.step(env.state, a, env.params)
        env.last_action = a.detach()
        states.append(env.state)
        actions.append(a)
    return states, actions, h, labels


@torch.no_grad()
def evaluate(env, pol, T=320, resets=3):
    outs = []
    for _ in range(resets):
        env.reset()
        h = pol.init_hidden(env.cfg.B, DEV)
        for _ in range(T):
            img, vec = priv_vec(env)
            a, h = pol(img, vec, h)
            env.state = env.dyn.step(env.state, a, env.params)
            env.last_action = a
        outs.append(env.success_metrics())
    return {k: sum(o[k] for o in outs) / len(outs) for k in outs[0]}


def main():
    torch.manual_seed(0)
    env = HoverEnv(EnvConfig(B=48))
    env.seed(7)
    pol = PrivPolicy().to(DEV)
    sd = torch.load("weights/pixel_ctbr_bptt_run5.pt", weights_only=True)["model"]
    own = pol.state_dict()
    kept = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
    pol.load_state_dict(kept, strict=False)
    with torch.no_grad():
        pol.head[0].weight.zero_()
        pol.head[0].weight[:, :sd["head.0.weight"].shape[1]].copy_(sd["head.0.weight"])
        pol.head[0].bias.copy_(sd["head.0.bias"])
    print(f"grafted; fresh: {[k for k in own if k not in kept]}")
    pol.train()
    for m in pol.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eval()

    opt = torch.optim.Adam(pol.parameters(), lr=1e-4)
    scale = torch.tensor([P.C_SPAN, *P.RATE_LIM], device=DEV)
    bc_w = torch.tensor([1.0, 2.0, 2.0, 1.5], device=DEV)
    EPOCHS = 8
    for ep in range(EPOCHS):
        H = horizon_for_epoch(ep, EPOCHS)
        t0 = time.time()
        for ci in range(8):  # chains per epoch
            env.reset()
            if ci % 3 == 2:
                B = env.cfg.B
                env.state.p = env.tgt_p + torch.randn(B, 3, device=DEV) * 0.3
                env.state.v = torch.randn(B, 3, device=DEV) * 0.15
            env.expert.reset()
            with torch.no_grad():
                h = pol.init_hidden(env.cfg.B, DEV)
                _, _, h, _ = rollout(env, pol, BURN, h, with_expert=False)
            for c in range(CHAIN):
                env.state = env.state.detach()
                states, actions, h, labels = rollout(env, pol, H, h.detach())
                loss, parts = window_loss(env, states, actions)
                l_bc = sum((HUBER(a / scale, lb / scale, reduction="none") * bc_w).mean()
                           for a, lb in zip(actions, labels)) / len(actions)
                loss = loss + 0.4 * l_bc
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(pol.parameters(), 5.0)
                opt.step()
        m = evaluate(env, pol)
        for mod in pol.modules():
            if isinstance(mod, torch.nn.BatchNorm2d):
                mod.eval()
        print(f"ep {ep+1}/{EPOCHS} H={H} ({time.time()-t0:.0f}s)  eval: {m}")


if __name__ == "__main__":
    main()
