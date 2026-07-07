"""Tail diagnosis: which episodes miss the strict gate, and why?

Runs N episodes, records per-episode start conditions + DR params + outcome
decomposition, then slices failure rate by each factor.
Run: python pixel2ctbr/diagnose_tail.py [weights]
"""

import sys

import torch

sys.path.insert(0, "pixel2ctbr")
from dynamics import euler_zyx_from_quat
from env import EnvConfig, HoverEnv
from policy import PixelCTBRPolicy

DEV = "cuda"


@torch.no_grad()
def main(weights="weights/pixel_ctbr_final.pt", resets=16, T=320):
    torch.manual_seed(0)
    env = HoverEnv(EnvConfig(B=64))
    env.seed(2024)
    pol = PixelCTBRPolicy().to(DEV)
    pol.load_state_dict(torch.load(weights, weights_only=True)["model"])
    pol.eval()

    rows = []
    for r in range(resets):
        env.reset()
        p0 = env.state.p.clone()
        yaw0, _, _ = euler_zyx_from_quat(env.state.q)
        v0 = env.state.v.norm(dim=-1).clone()
        h = pol.init_hidden(env.cfg.B, DEV)
        min_e = torch.full((env.cfg.B,), 1e9, device=DEV)
        t_in = torch.zeros(env.cfg.B, device=DEV)
        for k in range(T):
            img, vec = env.observe()
            a, h = pol(img, vec, h)
            env.state = env.dyn.step(env.state, a, env.params)
            env.last_action = a
            e = (env.state.p - env.tgt_p).norm(dim=-1)
            min_e = torch.minimum(min_e, e)
            t_in += (e < 0.15).float()
        e = (env.state.p - env.tgt_p).norm(dim=-1)
        v = env.state.v.norm(dim=-1)
        yawf, _, _ = euler_zyx_from_quat(env.state.q)
        yerr = torch.atan2(torch.sin(yawf - env.tgt_yaw),
                           torch.cos(yawf - env.tgt_yaw)).abs()
        ok = (e < 0.15) & (v < 0.2) & (yerr < 0.15)
        rows.append({
            "p0": p0.cpu(), "yaw0": yaw0.cpu(), "v0": v0.cpu(),
            "twr": env.params.twr.cpu(), "tau_w": env.params.tau_w.cpu(),
            "tau_c": env.params.tau_c.cpu(), "delay": env.params.delay_steps.cpu(),
            "gain": env.params.thrust_gain.cpu(), "kd": env.params.kd_lin.cpu(),
            "tilt_on": env.tilt_mask.cpu(),
            "e": e.cpu(), "v": v.cpu(), "yerr": yerr.cpu(),
            "min_e": min_e.cpu(), "frac_in": (t_in / T).cpu(), "ok": ok.cpu(),
        })
    cat = {k: torch.cat([r[k] for r in rows]) if rows[0][k].dim() == 1
           else torch.cat([r[k] for r in rows], 0) for k in rows[0]}
    N = cat["ok"].shape[0]
    ok = cat["ok"]
    fail = ~ok
    print(f"N={N}  success {ok.float().mean()*100:.1f}%")

    # which sub-criterion fails?
    e_f = cat["e"] >= 0.15
    v_f = cat["v"] >= 0.2
    y_f = cat["yerr"] >= 0.15
    print(f"failing sub-criteria (of failures): pos {e_f[fail].float().mean()*100:.0f}%"
          f"  vel {v_f[fail].float().mean()*100:.0f}%  yaw {y_f[fail].float().mean()*100:.0f}%")
    # failure taxonomy by trajectory shape
    never = (cat["min_e"] > 0.3)
    visited = (cat["min_e"] < 0.15) & (cat["e"] >= 0.15)
    print(f"failure taxonomy: never-got-close(min_e>0.3) {never[fail].float().mean()*100:.0f}%"
          f" | reached-then-left {visited[fail].float().mean()*100:.0f}%"
          f" | close-but-not-enough {(1 - never[fail].float().mean() - visited[fail].float().mean())*100:.0f}%")
    print(f"median min_e: ok {cat['min_e'][ok].median()*100:.1f} cm | fail {cat['min_e'][fail].median()*100:.1f} cm")
    print(f"median frac-time-inside-0.15m: ok {cat['frac_in'][ok].median()*100:.0f}% | fail {cat['frac_in'][fail].median()*100:.0f}%")

    # factor slicing: failure rate by quartile of each factor
    def slice_by(name, vals):
        qs = torch.quantile(vals.float(), torch.tensor([0.25, 0.5, 0.75]))
        bins = torch.bucketize(vals.float(), qs)
        s = " ".join(f"Q{i+1} {fail[bins == i].float().mean()*100:4.0f}%" for i in range(4))
        print(f"  fail-rate by {name:>10}: {s}")

    print("start-condition slicing:")
    slice_by("x0", cat["p0"][:, 0])
    slice_by("y0", cat["p0"][:, 1])
    slice_by("z0", cat["p0"][:, 2])
    slice_by("|yaw0-t|", (cat["yaw0"] + torch.pi / 2).abs())
    slice_by("v0", cat["v0"])
    print("DR-param slicing:")
    for k in ["twr", "tau_w", "tau_c", "gain", "kd"]:
        slice_by(k, cat[k])
    slice_by("delay", cat["delay"].float())
    print(f"  fail rate tilt-on {fail[cat['tilt_on'] > 0.5].float().mean()*100:.0f}% "
          f"vs tilt-off {fail[cat['tilt_on'] < 0.5].float().mean()*100:.0f}%")
    torch.save(cat, "pixel2ctbr/tail_diag.pt")
    print("saved pixel2ctbr/tail_diag.pt")


if __name__ == "__main__":
    main(*sys.argv[1:2])
