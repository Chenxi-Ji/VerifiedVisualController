"""Quantitative closed-loop eval for the meansub/CTBR-head 58k variant that
is CONSUMED AS VELOCITY by the ORIGINAL kinematic plant (old pinhole camera).

The repo's native test (test_ctrl_lya_pt.py) is qualitative (rollout videos);
this script is its quantitative counterpart on the SAME task with the SAME
metric definitions as logs/eval_ctrl_meansub_att.json (so the two variants
are directly comparable), adapted to the original velocity plant:

  episode  : spawn at a random offset around the target (the native
             test_ctrl_lya_pt.sample_init_poses ranges: x +-1.2 u,
             y -0.8..+1.2 u, z -0.5..+0.4 u, yaw +-0.5 rad, level attitude,
             at rest), roll out 6.0 s (60 steps @ dt=0.1) closed loop:
             OLD-pinhole render (300x200) -> controller (CTBR out) ->
             ctbr_to_velocity -> body_to_world_velocity -> the ORIGINAL
             trainer's plant: FIFO transport delay (latency_steps=1),
             first-order actuator lag (tau=0.15 s), pose += v*dt.
  randomize: the ORIGINAL recipe's randomization = Gaussian actuation noise
             (std 0.02 u/s) on the world-frame velocity command, exactly as
             train_ctrl_lya_pt.py applies it. Images are CLEAN (no image-
             space DR) as in the native test; --image-dr applies the
             training DomainRandomizer instead. --pure drops noise+lag+delay
             (the literal test_ctrl_lya_pt.run_test integrator) as a
             secondary reference number.
  hold_err : mean position error ||p - target|| over the LAST 1.5 s
             (15 steps) of the episode, in meters (1 u = 0.85 m).
             The headline number is the MEDIAN over episodes.
  final_err: position error at t = 6 s.
  yaw_err  : mean |yaw - yaw_target| over the last 1.5 s.
  crash    : at ANY step, z below the mat plane (z_u > 0.65, z-down frame),
             or position error > 3.0 u (~2.55 m, diverged), or NaN state.
             Crashed episodes are excluded from the hold/final statistics
             and reported as a rate.

Acceptance bar for this deliverable: median hold_err <= 0.10 m and
crash < 2% over >= 200 episodes.

Run from the repo root:
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python scripts_control/eval_ctrl_meansub_ctbrvel.py \
      --weights weights/ctrl_lya_meansub_ctbrvel.pt --episodes 240
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from render_image import load_gsplat_scene  # noqa: E402
from utils_ctrl_lya_pt import DomainRandomizer, body_to_world_velocity  # noqa: E402
from utils_ctrl_meansub_ctbrvel import (  # noqa: E402
    ControllerMeansubCtbrVel, ctbr_to_velocity, render_batch_pinhole_gpu,
    METERS_PER_UNIT)

TARGET_POSE = np.array([0.0, 1.5, 0.0, -np.pi / 2, 0.0, 0.0])
MAT_Z_U = 0.65          # mat plane, scene units below gate center (z down +)
DIVERGE_U = 3.0         # position-error ball that counts as gone
HOLD_STEPS = 15         # last 1.5 s at dt=0.1

# the ORIGINAL trainer's plant realism (train_ctrl_lya_pt.Config)
ACTUATOR_TAU = 0.15     # first-order velocity-lag time constant (s)
LATENCY_STEPS = 1       # transport delay in control steps
ACTUATION_NOISE = 0.02  # std of Gaussian noise on the world velocity (u/s)

# established comparison rows (printed next to this variant's numbers)
BANNER_ORIGINAL = "original banner        : 84.2% strict | 3.1 cm median hold | 0 crash"
BANNER_MEANSUB_ATT = ("meansub_att (delivered): 0.0% @10cm | 62.6 cm median hold | "
                      "29.2% crash  (attitude plant, no velocity damping -> orbits)")


@torch.no_grad()
def evaluate(ctrl, scene, device, episodes=240, seed=0, H=60, dt=0.1,
             batch=32, chunk=8, image_dr=False, pure=False,
             actuation_noise=ACTUATION_NOISE, verbose=False):
    """Returns a dict of metrics (definitions in the module docstring).
    pure=True -> instantaneous integrator, no delay/lag/noise (the literal
    test_ctrl_lya_pt.run_test plant)."""
    was_training = ctrl.training
    ctrl.eval()
    rng = np.random.RandomState(seed)
    gen = torch.Generator(device=device).manual_seed(seed * 977 + 13)
    domain_rand = DomainRandomizer() if image_dr else None
    target_u = torch.tensor(TARGET_POSE, device=device,
                            dtype=torch.float32).unsqueeze(0)

    lag_alpha = 1.0 if pure else 1.0 - np.exp(-dt / ACTUATOR_TAU)
    latency = 0 if pure else LATENCY_STEPS
    noise = 0.0 if pure else actuation_noise

    hold_errs, final_errs, yaw_errs, crashes = [], [], [], []
    hold_xy_errs, hold_z_errs = [], []
    done = 0
    bi = 0
    while done < episodes:
        B = min(batch, episodes - done)
        # native test_ctrl_lya_pt.sample_init_poses ranges
        init_u = TARGET_POSE + rng.uniform(
            low=[-1.2, -0.8, -0.5, -0.5, -0.0, -0.0],
            high=[1.2, 1.2, 0.4, 0.5, 0.0, 0.0], size=(B, 6))
        pose = torch.tensor(init_u, device=device, dtype=torch.float32)

        # actuator state, reset each episode (drone starts at rest) --
        # exactly the original trainer's rollout state
        vel_applied = torch.zeros(B, 4, device=device)
        cmd_buffer = deque(
            [torch.zeros(B, 4, device=device) for _ in range(latency)]
        ) if latency > 0 else None

        crashed = torch.zeros(B, dtype=torch.bool, device=device)
        errs_u, errs_xy_u, errs_z_u, yaws = [], [], [], []
        for t in range(H):
            img = render_batch_pinhole_gpu(pose, scene, chunk=chunk,
                                           device=device)
            if domain_rand is not None:
                img = domain_rand(img)
            ctbr = ctrl(img)
            pred_self = ctbr_to_velocity(ctbr)
            pred = body_to_world_velocity(pred_self, pose[:, 3])
            if noise > 0:
                pred = pred + torch.randn(pred.shape, device=device,
                                          generator=gen) * noise
            if cmd_buffer is not None:
                cmd_buffer.append(pred)
                pred = cmd_buffer.popleft()
            vel_applied = vel_applied + lag_alpha * (pred - vel_applied)
            vel_full = torch.cat(
                [vel_applied, torch.zeros(B, 2, device=device)], dim=-1)
            pose = pose + vel_full * dt

            e_u = (pose[:, :3] - target_u[:, :3]).norm(dim=-1)
            bad = (~torch.isfinite(pose).all(dim=-1)) | \
                  (pose[:, 2] > MAT_Z_U) | (e_u > DIVERGE_U)
            crashed |= bad
            errs_u.append(e_u)
            errs_xy_u.append((pose[:, :2] - target_u[:, :2]).norm(dim=-1))
            errs_z_u.append((pose[:, 2] - target_u[:, 2]).abs())
            yaws.append(pose[:, 3])
        errs_u = torch.stack(errs_u, dim=1)          # (B,H)
        errs_xy_u = torch.stack(errs_xy_u, dim=1)
        errs_z_u = torch.stack(errs_z_u, dim=1)
        yaws = torch.stack(yaws, dim=1)              # (B,H)
        hold_u = errs_u[:, -HOLD_STEPS:].mean(dim=1)
        yaw_e = (yaws[:, -HOLD_STEPS:] - TARGET_POSE[3]).abs().mean(dim=1)
        ok = ~crashed
        hold_errs += (hold_u[ok] * METERS_PER_UNIT).tolist()
        hold_xy_errs += (errs_xy_u[ok, -HOLD_STEPS:].mean(dim=1)
                         * METERS_PER_UNIT).tolist()
        hold_z_errs += (errs_z_u[ok, -HOLD_STEPS:].mean(dim=1)
                        * METERS_PER_UNIT).tolist()
        final_errs += (errs_u[ok, -1] * METERS_PER_UNIT).tolist()
        yaw_errs += yaw_e[ok].tolist()
        crashes.append(int(crashed.sum().item()))
        done += B
        bi += 1
        if verbose:
            print(f"  eval batch {bi}: {done}/{episodes} episodes, "
                  f"crashes so far {sum(crashes)}", flush=True)
    if was_training:
        ctrl.train()

    hold = np.array(hold_errs) if hold_errs else np.array([np.inf])
    fin = np.array(final_errs) if final_errs else np.array([np.inf])
    yw = np.array(yaw_errs) if yaw_errs else np.array([np.inf])
    n_crash = sum(crashes)
    return {
        "episodes": episodes,
        "crash_rate": n_crash / episodes,
        "n_crashed": n_crash,
        "hold_err_median_m": float(np.median(hold)),
        "hold_err_mean_m": float(hold.mean()),
        "hold_err_p90_m": float(np.percentile(hold, 90)),
        "hold_xy_err_median_m": float(np.median(hold_xy_errs)) if hold_xy_errs else None,
        "hold_z_err_median_m": float(np.median(hold_z_errs)) if hold_z_errs else None,
        "final_err_median_m": float(np.median(fin)),
        "success_hold_le_0.10m": float((hold <= 0.10).mean()),
        "success_hold_le_0.20m": float((hold <= 0.20).mean()),
        "yaw_err_median_deg": float(np.degrees(np.median(yw))),
        "horizon_s": H * dt,
        "hold_window_s": HOLD_STEPS * dt,
        "image_dr": image_dr,
        "pure_plant": pure,
        "actuation_noise": noise,
        "actuator_tau": 0.0 if pure else ACTUATOR_TAU,
        "latency_steps": latency,
        "camera": "OLD pinhole 300x200 fx=113.258171 (pre-27cd577 path)",
        "seed": seed,
    }


def print_table(m):
    strict = m["success_hold_le_0.10m"]
    print("\n================ this variant vs the established rows ================")
    print(f"meansub_ctbrvel (this) : {strict:.1%} @10cm | "
          f"{m['hold_err_median_m']*100:.1f} cm median hold | "
          f"{m['crash_rate']:.1%} crash   "
          f"({m['episodes']} eps, velocity plant tau={m['actuator_tau']}, "
          f"delay={m['latency_steps']}, noise={m['actuation_noise']})")
    print(BANNER_ORIGINAL)
    print(BANNER_MEANSUB_ATT)
    print("======================================================================")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights/ctrl_lya_meansub_ctbrvel.pt")
    ap.add_argument("--episodes", type=int, default=240)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--image-dr", action="store_true")
    ap.add_argument("--pure", action="store_true",
                    help="no lag/delay/noise: the literal test_ctrl_lya_pt "
                         "integrator (secondary reference)")
    ap.add_argument("--json-out", default="logs/eval_ctrl_meansub_ctbrvel.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.set_per_process_memory_fraction(
            float(os.environ.get("SIDE_MEM_FRAC", "0.25")), 0)

    from render_image import Config as RenderConfig
    scene = load_gsplat_scene(RenderConfig())
    ctrl = ControllerMeansubCtbrVel().to(device)
    ck = torch.load(args.weights, map_location=device, weights_only=False)
    ctrl.load_state_dict(ck["controller"])
    print(f"loaded {args.weights} (epoch {ck.get('epoch')}); "
          f"meta: {ck.get('meta', {}).get('input_spec', 'n/a')}")

    t0 = time.time()
    m = evaluate(ctrl, scene, device, episodes=args.episodes, seed=args.seed,
                 batch=args.batch, chunk=args.chunk, image_dr=args.image_dr,
                 pure=args.pure, verbose=True)
    m["weights"] = args.weights
    m["eval_wall_s"] = round(time.time() - t0, 1)

    print(json.dumps(m, indent=2))
    bar_hold, bar_crash = 0.10, 0.02
    ok = m["hold_err_median_m"] <= bar_hold and m["crash_rate"] < bar_crash
    print(f"\nACCEPTANCE (median hold <= {bar_hold} m AND crash < "
          f"{bar_crash:.0%} over >= 200 eps): {'PASS' if ok else 'FAIL'}  "
          f"[median hold {m['hold_err_median_m']*100:.1f} cm, "
          f"crash {m['crash_rate']:.2%}]")
    print_table(m)
    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out), exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(m, f, indent=2)
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
