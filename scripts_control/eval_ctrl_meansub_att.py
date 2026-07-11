"""Quantitative closed-loop eval for the meansub/attitude 58k variant.

The repo's native test (test_ctrl_lya_pt.py) is qualitative (rollout videos);
this script is its quantitative counterpart on the SAME task, adapted to the
attitude-setpoint plant. Definitions (all reported numbers):

  episode  : spawn at a random offset around the target (the native
             test_ctrl_lya_pt.sample_init_poses ranges: x +-1.2 u,
             y -0.8..+1.2 u, z -0.5..+0.4 u, yaw +-0.5 rad, level attitude,
             at rest), roll out 6.0 s (60 steps @ dt=0.1) closed loop:
             render -> controller -> QuadAttitudeDynamics.
  plant DR : DynParams.randomized per episode (twr 2.0-3.2, tau_w 15-60 ms,
             tau_c 10-45 ms, kd_lin 0.03-0.30, thrust_gain 0.85-1.15)
             + attitude-cascade gain DR +-20%; transport delay fixed at
             1 control step (100 ms), the original trainer's latency_steps.
             Images are CLEAN (no image-space DR) as in the native test;
             --image-dr applies the training DomainRandomizer instead.
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
crash < 2% over >= 200 episodes with the plant DR above.

Run from the repo root:
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python scripts_control/eval_ctrl_meansub_att.py \
      --weights weights/ctrl_lya_meansub_att.pt --episodes 240
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from render_image import load_gsplat_scene  # noqa: E402
from utils_ctrl_lya_pt import DomainRandomizer  # noqa: E402
from utils_ctrl_meansub_att import (  # noqa: E402
    ControllerMeansubAtt, DynParams, QuadAttitudeDynamics,
    quat_from_euler_zyx, render_batch_gpu, METERS_PER_UNIT, YAW_SP_CENTER)

TARGET_POSE = np.array([0.0, 1.5, 0.0, -np.pi / 2, 0.0, 0.0])
MAT_Z_U = 0.65          # mat plane, scene units below gate center (z down +)
DIVERGE_U = 3.0         # position-error ball that counts as gone
HOLD_STEPS = 15         # last 1.5 s at dt=0.1


def randomized_params(B, device, seed, delay_steps=1, plant_dr=True,
                      unit_thrust_gain=False):
    """Seeded plant DR: DynParams.randomized on CPU, delay forced to
    `delay_steps` (the original trainer's latency_steps=1 at 10 Hz control),
    attitude-cascade gains DR'd, then moved to `device`.
    plant_dr=False -> DynParams.nominal + nominal cascade gains (isolates the
    policy's intrinsic precision from DR robustness).
    unit_thrust_gain=True -> DR everywhere except thrust_gain=1.0 (isolates
    the thrust-map-error contribution)."""
    g = torch.Generator().manual_seed(seed)
    dyn = QuadAttitudeDynamics()
    if plant_dr:
        params = DynParams.randomized(B, device="cpu", g=g)
        params = dyn.finalize_params(params, g=g)
        if unit_thrust_gain:
            params.thrust_gain = torch.ones(B)
    else:
        params = DynParams.nominal(B, device="cpu")
        params = dyn.finalize_params(params, g=None)
    params.delay_steps = torch.full((B,), delay_steps, dtype=torch.long)
    for f_ in ("twr", "tau_w", "tau_c", "kd_lin", "delay_steps",
               "thrust_gain", "katt_rp", "katt_y"):
        setattr(params, f_, getattr(params, f_).to(device))
    return params


@torch.no_grad()
def evaluate(ctrl, scene, device, episodes=240, seed=0, H=60, dt=0.1,
             n_sub=20, batch=32, raster_scale=0.5, chunk=8, image_dr=False,
             actuation_noise=0.0, verbose=False, plant_dr=True,
             unit_thrust_gain=False):
    """Returns a dict of metrics (definitions in the module docstring)."""
    was_training = ctrl.training
    ctrl.eval()
    dyn = QuadAttitudeDynamics(dt_ctrl=dt, n_sub=n_sub)
    rng = np.random.RandomState(seed)
    domain_rand = DomainRandomizer() if image_dr else None
    target_u = torch.tensor(TARGET_POSE, device=device,
                            dtype=torch.float32).unsqueeze(0)

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
        pose_u = torch.tensor(init_u, device=device, dtype=torch.float32)
        params = randomized_params(B, device, seed * 100003 + bi,
                                   plant_dr=plant_dr,
                                   unit_thrust_gain=unit_thrust_gain)
        q0 = quat_from_euler_zyx(pose_u[:, 3], pose_u[:, 4], pose_u[:, 5])
        s = dyn.make_state(pose_u[:, :3] * METERS_PER_UNIT,
                           torch.zeros(B, 3, device=device), q0,
                           torch.zeros(B, 3, device=device), params)
        crashed = torch.zeros(B, dtype=torch.bool, device=device)
        errs_u = []          # per-step position error, scene units
        errs_xy_u, errs_z_u = [], []   # per-axis breakdown
        yaws = []
        pose6 = dyn.render_pose(s)
        for t in range(H):
            img = render_batch_gpu(pose6, scene, raster_scale=raster_scale,
                                   chunk=chunk, device=device)
            if domain_rand is not None:
                img = domain_rand(img)
            act = ctrl(img)
            if actuation_noise > 0:
                act = act + torch.randn_like(act) * actuation_noise
            s = dyn.step(s, act, params)
            pose6 = dyn.render_pose(s)
            e_u = (pose6[:, :3] - target_u[:, :3]).norm(dim=-1)
            bad = (~torch.isfinite(pose6).all(dim=-1)) | \
                  (pose6[:, 2] > MAT_Z_U) | (e_u > DIVERGE_U)
            crashed |= bad
            errs_u.append(e_u)
            errs_xy_u.append((pose6[:, :2] - target_u[:, :2]).norm(dim=-1))
            errs_z_u.append((pose6[:, 2] - target_u[:, 2]).abs())
            yaws.append(pose6[:, 3])
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
        "plant_dr": plant_dr,
        "unit_thrust_gain": unit_thrust_gain,
        "seed": seed,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights/ctrl_lya_meansub_att.pt")
    ap.add_argument("--episodes", type=int, default=240)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--raster-scale", type=float, default=0.5)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--image-dr", action="store_true")
    ap.add_argument("--no-plant-dr", action="store_true",
                    help="nominal plant params (isolate intrinsic precision)")
    ap.add_argument("--unit-thrust-gain", action="store_true",
                    help="full DR except thrust_gain=1.0")
    ap.add_argument("--json-out", default="logs/eval_ctrl_meansub_att.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.set_per_process_memory_fraction(
            float(os.environ.get("SIDE_MEM_FRAC", "0.25")), 0)

    from render_image import Config as RenderConfig
    scene = load_gsplat_scene(RenderConfig())
    ctrl = ControllerMeansubAtt().to(device)
    ck = torch.load(args.weights, map_location=device, weights_only=False)
    ctrl.load_state_dict(ck["controller"])
    print(f"loaded {args.weights} (epoch {ck.get('epoch')}); "
          f"meta: {ck.get('meta', {}).get('input_spec', 'n/a')}")

    t0 = time.time()
    m = evaluate(ctrl, scene, device, episodes=args.episodes, seed=args.seed,
                 batch=args.batch, raster_scale=args.raster_scale,
                 chunk=args.chunk, image_dr=args.image_dr, verbose=True,
                 plant_dr=not args.no_plant_dr,
                 unit_thrust_gain=args.unit_thrust_gain)
    m["weights"] = args.weights
    m["eval_wall_s"] = round(time.time() - t0, 1)

    print(json.dumps(m, indent=2))
    bar_hold, bar_crash = 0.10, 0.02
    ok = m["hold_err_median_m"] <= bar_hold and m["crash_rate"] < bar_crash
    print(f"\nACCEPTANCE (median hold <= {bar_hold} m AND crash < "
          f"{bar_crash:.0%} over >= 200 eps): {'PASS' if ok else 'FAIL'}  "
          f"[median hold {m['hold_err_median_m']*100:.1f} cm, "
          f"crash {m['crash_rate']:.2%}]")
    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out), exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(m, f, indent=2)
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
