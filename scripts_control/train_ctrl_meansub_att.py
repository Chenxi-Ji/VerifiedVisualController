"""Retrain the minimally-modified 58k controller (3ch meansub input, attitude
head) on the ORIGINAL task with the ORIGINAL pipeline, against the attitude-
setpoint plant. SIM-ONLY side deliverable (2026-07-11).

This is train_ctrl_lya_pt.py with exactly these changes:
  1. Controller -> ControllerMeansubAtt (3ch meansub input, attitude head).
  2. Plant: the hand-rolled velocity integrator (+ first-order lag + FIFO
     latency) -> QuadAttitudeDynamics (copied from PixelCTBR
     pixel2ctbr/dynamics.py), which owns actuator lag, transport delay and
     per-episode parameter DR. Control period stays dt = 0.1 s (the original
     trainer's rate); physics integrates at dt/n_sub = 5 ms. Transport delay
     = 1 control step = 100 ms (== the original latency_steps=1).
  3. Rendering: batched GPU renderer (render_batch_gpu) instead of the
     per-image cached render() - identical view math, optional raster_scale
     (0.5 rasterizes the fisheye at 512x384 with K/2 then resizes; 1.0 is the
     exact original 1024x768 pipeline), no image cache.
  4. Curriculum/losses/DR: the ORIGINAL 3-phase losses, loss weights, horizon
     schedule, image DomainRandomizer, camera-intrinsics jitter and actuation
     noise - reused verbatim (losses/dataset imported from
     train_ctrl_lya_pt.py); the epoch axis is scaled by epochs/120 so shorter
     runs keep the phase proportions. Losses and the co-trained Lyapunov stay
     in scene units on [x,y,z,yaw] exactly as before (the plant works in
     meters; 1 u = 0.85 m at the boundary).

Artifacts (NEW names, never overwrites existing weights):
  weights/ctrl_lya_meansub_att_last.pt  rolling endpoint (also the resume
                                        source - re-run to continue)
  weights/ctrl_lya_meansub_att.pt       banked best by the periodic quick
                                        eval (64 episodes, plant DR)

GPU etiquette (shared GPU - a campaign training is live):
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (set here as a default),
  torch.cuda.set_per_process_memory_fraction(SIDE_MEM_FRAC, default 0.25).

Run from the repo root:
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python scripts_control/train_ctrl_meansub_att.py --max-minutes 8
(repeat the same command until it prints TRAINING COMPLETE - it resumes)
"""

from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ORIGINAL pipeline pieces, reused verbatim (imported, not copied)
from train_ctrl_lya_pt import (  # noqa: E402
    PoseDataset, compute_traj_loss, compute_final_state_loss,
    compute_lyapunov_decrease_loss, plot_training_curves)
from utils_ctrl_lya_pt import Lyapunov, DomainRandomizer  # noqa: E402
from render_image import load_gsplat_scene, Config as RenderConfig  # noqa: E402

# the modified net + the attitude plant + batched renderer
from utils_ctrl_meansub_att import (  # noqa: E402
    ControllerMeansubAtt, QuadAttitudeDynamics, quat_from_euler_zyx,
    render_batch_gpu, BASE_K, METERS_PER_UNIT,
    C_SPAN, TILT_SP_LIMIT, YAW_SP_SPAN)
from eval_ctrl_meansub_att import evaluate, randomized_params  # noqa: E402


class Config:
    """Mirrors train_ctrl_lya_pt.Config; epoch axis scaled by epochs/120."""

    def __init__(self, args):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = args.batch
        self.epochs = args.epochs
        self.lr = args.lr
        self.lr_decay = 0.95
        self.dt = 0.1                 # control period, as the original
        self.n_sub = 20               # -> 5 ms physics substep
        self.delay_steps = 1          # == original latency_steps (100 ms)
        self.n_poses = args.poses
        self.raster_scale = args.raster_scale
        self.chunk = args.chunk

        self.target_pose = np.array([0.0, 1.5, 0.0, -np.pi / 2, 0.0, 0.0])
        self.gate_pose = np.array([0.0, 0.0, 0.0, -np.pi / 2, 0.0, 0.0])

        self.save_path = "weights/ctrl_lya_meansub_att.pt"        # banked best
        self.last_path = "weights/ctrl_lya_meansub_att_last.pt"   # rolling
        self.figure_path = "training_curves_meansub_att.png"

        # soft mat-plane floor penalty (the ONE loss addition vs the original
        # recipe): the attitude plant sinks while tilting toward the target
        # (thrust vertical component ~ c*cos(tilt)); the original velocity
        # plant could not produce this, so the original losses never see it -
        # measured at ep-77: ~90% of eval episodes clipped the mat plane
        # (z_u > 0.65) during the t~1 s transient, even at nominal params.
        # Penalize trajectory z below a margin above the physical mat:
        #   w_floor * mean(relu(z_u - z_soft)^2)  with z_soft = 0.45 u
        # (mat at +0.65 u below gate center; spawns reach +0.4 u).
        self.z_soft = 0.45
        self.w_floor = args.w_floor
        self.w_vel_scale = args.w_vel_scale
        self.pin_horizon = args.pin_horizon

        # actuation noise: 2% of each channel's full span (the original used
        # 0.02 on a +-1.0 u/s velocity span)
        self.actuation_noise = 0.02
        self.noise_std = torch.tensor(
            [C_SPAN, TILT_SP_LIMIT, TILT_SP_LIMIT, YAW_SP_SPAN],
            device=self.device) * self.actuation_noise

        self.intrinsics_jitter = True

        # ===== original 3-phase curriculum, epoch axis scaled =====
        s = self.epochs / 120.0
        self.curriculum_phase1_end = max(1, round(40 * s))
        self.curriculum_phase2_end = max(2, round(80 * s))
        self.curriculum_phase3_end = self.epochs
        self._h_edges = [max(1, round(e * s)) for e in (20, 35, 50, 70, 95)]
        self._h30_edge = max(2, round(105 * s))
        self._lr_every = max(1, round(10 * s))

    def get_loss_weights(self, epoch):
        # w_vel is the SECOND targeted addition vs the original recipe (with
        # w_floor): a terminal-velocity penalty. The original final-state
        # loss scores POSITION ONLY at t=H - fine for the velocity plant
        # (v_cmd=0 => stay), but on the attitude plant it rewards arriving
        # AT t=H with residual speed, which turns into a growing oscillation
        # past the training horizon (measured: quick-eval hold degraded
        # 60->131 cm through P3 while training losses fell). Hover is a
        # dynamic equilibrium; demand it: penalize ||v||^2 (m/s) over the
        # last 5 rollout steps, phase-gated so it never fights the P1
        # reach-the-target objective.
        if epoch < self.curriculum_phase1_end:
            return {'w_traj': 3.0, 'w_decrease': 0.5, 'w_final_state': 0.05,
                    'w_vel': 0.0}
        elif epoch < self.curriculum_phase2_end:
            alpha = (epoch - self.curriculum_phase1_end) / (
                self.curriculum_phase2_end - self.curriculum_phase1_end)
            return {'w_traj': 3.0 - 2.5 * alpha,
                    'w_decrease': 0.5 + 1.0 * alpha,
                    'w_final_state': 0.05 + 2.95 * alpha,
                    'w_vel': 0.15 * alpha * self.w_vel_scale}
        else:
            return {'w_traj': 0.5, 'w_decrease': 1.5, 'w_final_state': 3.0,
                    'w_vel': 0.5 * self.w_vel_scale}

    def get_horizon(self, epoch):
        if self.pin_horizon > 0:
            # hold-phase fine-tune: rollouts as long as the eval episode so
            # the slow (~5-8 s period) weakly-damped vertical/lateral modes
            # and the mat clips are INSIDE the optimization window - the
            # scheduled 0.7-3 s windows cannot see them (ep-92 diagnosis)
            return self.pin_horizon
        e = self._h_edges
        if epoch < e[0]:
            return 7
        elif epoch < e[1]:
            return 10
        elif epoch < e[2]:
            return 14
        elif epoch < e[3]:
            return 18
        elif epoch < e[4]:
            return 22
        elif epoch < self._h30_edge:
            return 25
        else:
            # late-P3 hold training: 3 s so "arrive AND stay at rest" is
            # actually exercised (the original stopped at 2.5 s)
            return 30


def build_meta(cfg, args):
    return dict(
        arch="ControllerMeansubAtt (scripts_control/utils_ctrl_meansub_att.py)",
        parent=("Controller in scripts_control/utils_ctrl_lya_pt.py "
                "(58,036 params; flight artifact weights/ctrl_lya.pt)"),
        input_spec=("3ch mean-subtracted RGB ONLY: x - x.mean(dim=(2,3), "
                    "keepdim=True) per image per channel; RGB 192x256 in "
                    "[0,1] before meansub (was 6ch [raw, raw-mean])"),
        action_space=("attitude [c, roll_sp, pitch_sp, yaw_sp]: "
                      "c = G + clamp_relu(raw,1)*0.9G in [0.1G,1.9G] m/s^2; "
                      "tilt = clamp_relu(raw*0.35, 0.35) rad; yaw_sp ABSOLUTE"
                      " = -pi/2 + clamp_relu(raw*pi, pi) rad "
                      "(pixel2ctbr_ff/policy_ff.py conventions, copied)"),
        param_count=56836,
        param_count_original=58036,
        plant=(f"QuadAttitudeDynamics dt_ctrl={cfg.dt} n_sub={cfg.n_sub} "
               "delay=1 step (copied from PixelCTBR pixel2ctbr/dynamics.py); "
               "DynParams.randomized per batch + katt +-20% DR"),
        frame=("gate-centered world frame, z DOWN, +y through gate; plant in "
               "meters, render/losses in scene units, 1 u = 0.85 m"),
        target_pose_u=cfg.target_pose.tolist(),
        task="one-gate viewpoint hold: servo to 1.5 u in front of +y gate face and hover",
        raster_scale=cfg.raster_scale,
        loss_deviation=(f"TWO added loss terms vs the original recipe, both "
                        "forced by the velocity->attitude plant change: (1) "
                        f"soft mat-plane floor {cfg.w_floor} * mean(relu(z_u"
                        f" - {cfg.z_soft})^2) over the rollout (mat at +0.65"
                        " u; the attitude plant sinks while tilting, the "
                        "velocity plant cannot - ep-77 diagnosis: ~90% of "
                        "eval episodes clipped the mat). (2) terminal-"
                        "velocity penalty w_vel * mean ||v||^2 over the last"
                        " 5 steps (w_vel: 0 in P1, ramp to 0.15 in P2, 0.5 "
                        "in P3): the original final-state loss is position-"
                        "only - on the velocity plant v_cmd=0 means stay, "
                        "on the attitude plant hover is a dynamic "
                        "equilibrium that must be demanded explicitly. Also "
                        "late-P3 horizon extended 25 -> 30 steps (3 s) to "
                        "exercise the hold. All other losses/curriculum "
                        "verbatim from train_ctrl_lya_pt.py."),
        train_cmd="python scripts_control/train_ctrl_meansub_att.py " + " ".join(
            sys.argv[1:]),
        trainer="scripts_control/train_ctrl_meansub_att.py (adapted from "
                "train_ctrl_lya_pt.py; losses/curriculum/DR reused verbatim)",
    )


def train(args):
    cfg = Config(args)
    device = cfg.device
    if device == "cuda":
        torch.cuda.set_per_process_memory_fraction(
            float(os.environ.get("SIDE_MEM_FRAC", "0.25")), 0)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    scene = load_gsplat_scene(RenderConfig())
    ds = PoseDataset(cfg.target_pose, N=cfg.n_poses)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)

    ctrl = ControllerMeansubAtt().to(device)
    Vnet = Lyapunov().to(device)
    domain_rand = DomainRandomizer()
    dyn = QuadAttitudeDynamics(dt_ctrl=cfg.dt, n_sub=cfg.n_sub)

    opt = torch.optim.Adam(list(ctrl.parameters()) + list(Vnet.parameters()),
                           lr=cfg.lr)

    start_epoch = 0
    best_score = float("inf")
    loss_hist = {'total': [], 'traj': [], 'decrease': [], 'final_state': []}
    if os.path.exists(cfg.last_path) and not args.fresh:
        ckpt = torch.load(cfg.last_path, map_location=device,
                          weights_only=False)
        ctrl.load_state_dict(ckpt["controller"])
        Vnet.load_state_dict(ckpt["lyapunov"])
        start_epoch = ckpt.get("epoch", 0)
        best_score = ckpt.get("best_score", float("inf"))
        loss_hist = ckpt.get("loss_hist", loss_hist)
        if "optimizer" in ckpt and ckpt["optimizer"] is not None:
            opt.load_state_dict(ckpt["optimizer"])
        else:
            # fresh optimizer mid-run: replay the lr schedule deterministically
            for pg in opt.param_groups:
                pg['lr'] = cfg.lr * (cfg.lr_decay ** (start_epoch // cfg._lr_every))
        print(f"[INFO] resumed {cfg.last_path} at epoch {start_epoch} "
              f"(best_score {best_score:.4f}, lr {opt.param_groups[0]['lr']:.2e})")
    if start_epoch >= cfg.epochs:
        print("TRAINING COMPLETE (already at final epoch)")
        return

    target = torch.tensor(cfg.target_pose, device=device,
                          dtype=torch.float32).unsqueeze(0)

    # camera-model DR: per-epoch jittered intrinsics + mount (original values)
    render_K = dict(BASE_K, dyaw=0.0, dpitch=0.0, droll=0.0)

    def jitter_camera_model():
        render_K["fx"] = BASE_K["fx"] * (1.0 + np.random.uniform(-0.004, 0.004))
        render_K["fy"] = BASE_K["fy"] * (1.0 + np.random.uniform(-0.004, 0.004))
        render_K["cx"] = BASE_K["cx"] + np.random.uniform(-0.5, 0.5)
        render_K["cy"] = BASE_K["cy"] + np.random.uniform(-0.5, 0.5)
        render_K["dyaw"] = np.radians(np.random.uniform(-0.5, 0.5))
        render_K["dpitch"] = np.radians(np.random.uniform(-0.5, 0.5))
        render_K["droll"] = np.radians(np.random.uniform(-0.5, 0.5))

    def render_now(pose6):
        """Render the batch with this epoch's camera model (mount offsets are
        camera-only: applied to a detached copy of the render pose)."""
        pr = pose6.detach().clone()
        pr[:, 3] += render_K["dyaw"]
        pr[:, 4] += render_K["dpitch"]
        pr[:, 5] += render_K["droll"]
        return render_batch_gpu(
            pr, scene,
            K={k: render_K[k] for k in ("fx", "fy", "cx", "cy")},
            raster_scale=cfg.raster_scale, chunk=cfg.chunk, device=device)

    def save_ckpt(path, ep):
        tmp = path + ".tmp"
        torch.save({
            "controller": ctrl.state_dict(),
            "lyapunov": Vnet.state_dict(),
            "optimizer": opt.state_dict(),
            "epoch": ep,
            "best_score": best_score,
            "loss_hist": loss_hist,
            "meta": build_meta(cfg, args),
        }, tmp)
        os.replace(tmp, path)  # atomic: a mid-save kill can't corrupt

    t_start = time.monotonic()
    stopped_early = False
    pbar = tqdm(range(start_epoch, cfg.epochs), desc="Training",
                dynamic_ncols=True, leave=True)

    for ep in pbar:
        if cfg.intrinsics_jitter:
            jitter_camera_model()
        weights = cfg.get_loss_weights(ep)
        H = cfg.get_horizon(ep)

        if ep > 0 and ep % cfg._lr_every == 0:
            for pg in opt.param_groups:
                pg['lr'] *= cfg.lr_decay

        tot = {'total': 0.0, 'traj': 0.0, 'decrease': 0.0, 'final_state': 0.0}
        floor_sum = 0.0
        n_batches = 0
        skipped = 0

        for batch_idx, pose_batch in enumerate(dl):
            pose_batch = pose_batch.to(device).float()
            B = pose_batch.size(0)
            target_batch = target.expand(B, -1)
            initial_pose = pose_batch.clone()

            # ===== plant init (at rest; pitch/roll DR of the dataset becomes
            # a real initial tilt the attitude cascade must absorb) =====
            params = randomized_params(
                B, device, seed=args.seed * 7 + ep * 100003 + batch_idx,
                delay_steps=cfg.delay_steps)
            q0 = quat_from_euler_zyx(pose_batch[:, 3], pose_batch[:, 4],
                                     pose_batch[:, 5])
            s = dyn.make_state(pose_batch[:, :3] * METERS_PER_UNIT,
                               torch.zeros(B, 3, device=device), q0,
                               torch.zeros(B, 3, device=device), params)

            pose_curr = dyn.render_pose(s)          # (B,6) scene units
            img_curr = render_now(pose_curr)
            V_curr, alpha_reg_curr = Vnet(pose_curr, target_batch)
            V_list, pose_list = [V_curr], [pose_curr]
            alpha_reg_list = [alpha_reg_curr]
            vel_list = []                           # plant velocity, m/s

            # ===== closed-loop rollout (BPTT through the plant; images are
            # exogenous per-step observations, as in the original) =====
            for step in range(H):
                pred = ctrl(domain_rand(img_curr))
                if cfg.actuation_noise > 0:
                    pred = pred + torch.randn_like(pred) * cfg.noise_std
                s = dyn.step(s, pred, params)
                pose_next = dyn.render_pose(s)
                img_curr = render_now(pose_next)
                V_next, alpha_reg_next = Vnet(pose_next, target_batch)
                pose_curr = pose_next
                V_list.append(V_next)
                pose_list.append(pose_next)
                alpha_reg_list.append(alpha_reg_next)
                vel_list.append(s.v)

            # ===== ORIGINAL losses, verbatim =====
            loss_traj = compute_traj_loss(initial_pose, pose_list[-1],
                                          target_batch, H=H, dt=cfg.dt)
            loss_decrease = compute_lyapunov_decrease_loss(
                V_list, alpha_reg_list, decay_ratio=0.1, scale_increase=5.0,
                w_smooth=0.15, w_alpha_reg=0.1)
            loss_final = compute_final_state_loss(pose_list[-1], target_batch)
            # soft mat-plane floor (see Config.z_soft comment): z-down frame,
            # z_u > z_soft means "too close to the mat"
            z_traj = torch.stack([p6[:, 2] for p6 in pose_list], dim=1)
            loss_floor = torch.relu(z_traj - cfg.z_soft).pow(2).mean()
            # terminal-velocity penalty (see get_loss_weights comment)
            loss_vel = torch.stack(vel_list[-5:], dim=1).pow(2).sum(-1).mean()
            loss_total = (weights['w_traj'] * loss_traj +
                          weights['w_decrease'] * loss_decrease +
                          weights['w_final_state'] * loss_final +
                          cfg.w_floor * loss_floor +
                          weights['w_vel'] * loss_vel)

            if not torch.isfinite(loss_total):
                skipped += 1
                continue

            opt.zero_grad()
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(
                list(ctrl.parameters()) + list(Vnet.parameters()),
                max_norm=1.0)
            opt.step()

            tot['total'] += loss_total.item()
            tot['traj'] += loss_traj.item()
            tot['decrease'] += loss_decrease.item()
            tot['final_state'] += loss_final.item()
            floor_sum += loss_floor.item()
            n_batches += 1

        for k in loss_hist:
            loss_hist[k].append(tot[k] / max(n_batches, 1))
        plot_training_curves(loss_hist, cfg)

        phase = ("P1" if ep < cfg.curriculum_phase1_end else
                 "P2" if ep < cfg.curriculum_phase2_end else "P3")
        pbar.set_description(f"Ep {ep+1:03d} [{phase}|H={H:2d}]")
        pbar.set_postfix({"loss": f"{loss_hist['total'][-1]:.4f}",
                          "traj": f"{loss_hist['traj'][-1]:.4f}",
                          "dec": f"{loss_hist['decrease'][-1]:.4f}",
                          "fin": f"{loss_hist['final_state'][-1]:.4f}",
                          "flr": f"{floor_sum / max(n_batches, 1):.4f}",
                          "skip": skipped})

        # rolling save (resume source)
        save_ckpt(cfg.last_path, ep + 1)

        # periodic quick eval -> bank best
        last_ep = (ep + 1 == cfg.epochs)
        if last_ep or (ep + 1) % args.eval_every == 0:
            m = evaluate(ctrl, scene, device, episodes=args.eval_episodes,
                         seed=123, batch=min(cfg.batch_size, 32),
                         raster_scale=cfg.raster_scale, chunk=cfg.chunk)
            score = m["hold_err_median_m"] + 10.0 * m["crash_rate"]
            print(f"\n[quick eval @ ep {ep+1}] hold_med "
                  f"{m['hold_err_median_m']*100:.1f} cm | crash "
                  f"{m['crash_rate']:.1%} | succ@10cm "
                  f"{m['success_hold_le_0.10m']:.1%} | score {score:.4f} "
                  f"(best {best_score:.4f})", flush=True)
            if score < best_score:
                best_score = score
                save_ckpt(cfg.save_path, ep + 1)
                save_ckpt(cfg.last_path, ep + 1)   # persist best_score
                print(f"[quick eval] BANKED -> {cfg.save_path}", flush=True)

        if args.max_minutes > 0 and \
                (time.monotonic() - t_start) / 60.0 > args.max_minutes:
            print(f"\n[time] {args.max_minutes} min budget reached at epoch "
                  f"{ep+1}/{cfg.epochs}; state saved to {cfg.last_path}. "
                  "Re-run the same command to resume.")
            stopped_early = True
            break

    if not stopped_early:
        if not os.path.exists(cfg.save_path):
            save_ckpt(cfg.save_path, cfg.epochs)
        print("TRAINING COMPLETE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--poses", type=int, default=1024,
                    help="poses per epoch (original: 2000)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--raster-scale", type=float, default=0.5)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--w-floor", type=float, default=5.0,
                    help="weight of the soft mat-plane floor penalty")
    ap.add_argument("--w-vel-scale", type=float, default=1.0,
                    help="scale on the terminal-velocity penalty schedule "
                         "(0 disables; the delivered ep-60 artifact "
                         "effectively trained with 0)")
    ap.add_argument("--pin-horizon", type=int, default=0,
                    help="override the horizon schedule with a fixed H "
                         "(hold-phase fine-tune); 0 = use the schedule")
    ap.add_argument("--eval-every", type=int, default=15)
    ap.add_argument("--eval-episodes", type=int, default=64)
    ap.add_argument("--max-minutes", type=float, default=0,
                    help="clean stop after this wall time (0 = no limit)")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore an existing _last checkpoint")
    args = ap.parse_args()
    train(args)
