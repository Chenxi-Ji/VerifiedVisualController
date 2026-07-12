"""Retrain the minimally-modified 58k controller (3ch meansub input, CTBR-
shaped head consumed as velocity) on the ORIGINAL task with the ORIGINAL
pipeline and plant, under the OLD PINHOLE camera. SIM-ONLY side deliverable
(2026-07-12) for verification practice.

This is train_ctrl_lya_pt.py with exactly these changes:
  1. Controller -> ControllerMeansubCtbrVel (3ch meansub input, CTBR head);
     the net's CTBR output passes through the FIXED affine interface
     `ctbr_to_velocity` (see utils_ctrl_meansub_ctbrvel.py docstring table)
     and from there the pipeline is the original's, verbatim:
     body_to_world_velocity -> +actuation noise -> FIFO transport delay
     (latency_steps=1) -> first-order actuator lag (tau=0.15 s) ->
     pose += vel*dt. The plant, losses, curriculum, horizons, LR schedule,
     image DomainRandomizer, per-epoch camera-intrinsics jitter, dt=0.1 and
     the co-trained Lyapunov are the ORIGINAL's (losses/dataset imported
     from train_ctrl_lya_pt.py, not copied). Default poses/epoch = 2000 and
     batch 32, as the original.
  2. Camera: the OLD PINHOLE model (pre-27cd577 render() path: 300x200
     raster, fx=113.258171 fy=113.347599 cx=158.868074 cy=98.837772,
     camera_model='pinhole' = the gsplat default the old code relied on,
     NO resize -> the net consumes 200x300 frames). The per-epoch
     intrinsics jitter keeps the original's magnitudes (fx,fy +-0.4%,
     cx,cy +-0.5 px, mount +-0.5 deg) applied to the OLD K.
  3. Rendering plumbing (not semantics): the original per-image render() +
     ImageCache is reproduced as a CPU-backed cache (same 2-decimal pose
     key, same 3000-entry FIFO eviction, cleared on every per-epoch
     intrinsics jitter, misses rendered with the CURRENT epoch's jittered
     K/mount at the TRUE unrounded pose) with misses rasterized in chunked
     GPU batches -- parity between the batched and the old single-image
     path is 1.5e-4 max abs (check_pinhole_parity).
  4. Additions that do not change the optimization: resumable foreground
     chunks (--max-minutes; epoch-boundary stop; optimizer state saved),
     periodic quick eval that BANKS the best checkpoint, meta stamping.

Artifacts (NEW names, never overwrites existing weights):
  weights/ctrl_lya_meansub_ctbrvel_last.pt  rolling endpoint (resume source)
  weights/ctrl_lya_meansub_ctbrvel.pt       banked best by the periodic
                                            quick eval (64 episodes)

GPU etiquette (shared GPU - a campaign training is live):
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (set here as a default),
  torch.cuda.set_per_process_memory_fraction(SIDE_MEM_FRAC, default 0.25).

Run from the repo root:
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python scripts_control/train_ctrl_meansub_ctbrvel.py --max-minutes 6
(repeat the same command until it prints TRAINING COMPLETE - it resumes)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque

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
from utils_ctrl_lya_pt import (  # noqa: E402
    Lyapunov, DomainRandomizer, body_to_world_velocity)
from render_image import load_gsplat_scene, Config as RenderConfig  # noqa: E402

# the modified net + the CTBR->velocity interface + old-pinhole renderer
from utils_ctrl_meansub_ctbrvel import (  # noqa: E402
    ControllerMeansubCtbrVel, ctbr_to_velocity, render_batch_pinhole_gpu,
    OLD_PINHOLE_K, OLD_W, OLD_H, G, RATE_LIM)
from eval_ctrl_meansub_ctbrvel import evaluate  # noqa: E402


class Config:
    """Mirrors train_ctrl_lya_pt.Config VERBATIM (only save paths + the
    pinhole base intrinsics differ; epoch axis scaled by epochs/120 so a
    shortened run keeps the phase proportions -- at the default 120 the
    schedule is numerically identical to the original)."""

    def __init__(self, args):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = args.batch          # original: 32
        self.epochs = args.epochs             # original: 120
        self.lr = args.lr                     # original: 1e-3
        self.lr_decay = 0.95
        self.dt = 0.1
        self.n_poses = args.poses             # original: 2000
        self.chunk = args.chunk

        self.target_pose = np.array([0.0, 1.5, 0.0, -np.pi / 2, 0.0, 0.0])
        self.gate_pose = np.array([0.0, 0.0, 0.0, -np.pi / 2, 0.0, 0.0])

        self.save_path = "weights/ctrl_lya_meansub_ctbrvel.pt"        # banked best
        self.last_path = "weights/ctrl_lya_meansub_ctbrvel_last.pt"   # rolling
        self.figure_path = "training_curves_meansub_ctbrvel.png"

        # ORIGINAL plant realism (train_ctrl_lya_pt.Config, verbatim)
        self.actuation_noise = 0.02
        self.actuator_tau = 0.15
        self.latency_steps = 1

        self.intrinsics_jitter = True

        # ===== original 3-phase curriculum, epoch axis scaled =====
        s = self.epochs / 120.0
        self.curriculum_phase1_end = max(1, round(40 * s))
        self.curriculum_phase2_end = max(2, round(80 * s))
        self.curriculum_phase3_end = self.epochs
        self._h_edges = [max(1, round(e * s)) for e in (20, 35, 50, 70, 95)]
        self._lr_every = max(1, round(10 * s))

    def get_loss_weights(self, epoch):
        # ORIGINAL weights, verbatim
        if epoch < self.curriculum_phase1_end:
            return {'w_traj': 3.0, 'w_decrease': 0.5, 'w_final_state': 0.05}
        elif epoch < self.curriculum_phase2_end:
            alpha = (epoch - self.curriculum_phase1_end) / (
                self.curriculum_phase2_end - self.curriculum_phase1_end)
            return {'w_traj': 3.0 - 2.5 * alpha,
                    'w_decrease': 0.5 + 1.0 * alpha,
                    'w_final_state': 0.05 + 2.95 * alpha}
        else:
            return {'w_traj': 0.5, 'w_decrease': 1.5, 'w_final_state': 3.0}

    def get_horizon(self, epoch):
        # ORIGINAL schedule, verbatim (7/10/14/18/22/25)
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
        else:
            return 25


# =====================================================================
# faithful ImageCache (train_ctrl_lya_pt.ImageCache) with CPU storage +
# batched-miss rendering. Same key (pose rounded to 2 decimals), same
# max_size/FIFO eviction, same "render the TRUE pose on miss with the
# current epoch's jittered camera" semantics as the original get_image().
# =====================================================================
class CachedPinholeRenderer:
    def __init__(self, scene, device, chunk=8, max_size=3000):
        self.scene = scene
        self.device = device
        self.chunk = chunk
        self.max_size = max_size
        self.cache = {}                       # key -> (3,200,300) CPU fp32
        self.K = dict(OLD_PINHOLE_K, dyaw=0.0, dpitch=0.0, droll=0.0)

    def jitter(self):
        """Per-epoch camera-model DR: the ORIGINAL jitter magnitudes
        (train_ctrl_lya_pt.jitter_camera_model) applied to the OLD pinhole
        K; drops the cache exactly as the original did."""
        self.K["fx"] = OLD_PINHOLE_K["fx"] * (1.0 + np.random.uniform(-0.004, 0.004))
        self.K["fy"] = OLD_PINHOLE_K["fy"] * (1.0 + np.random.uniform(-0.004, 0.004))
        self.K["cx"] = OLD_PINHOLE_K["cx"] + np.random.uniform(-0.5, 0.5)
        self.K["cy"] = OLD_PINHOLE_K["cy"] + np.random.uniform(-0.5, 0.5)
        self.K["dyaw"] = np.radians(np.random.uniform(-0.5, 0.5))
        self.K["dpitch"] = np.radians(np.random.uniform(-0.5, 0.5))
        self.K["droll"] = np.radians(np.random.uniform(-0.5, 0.5))
        self.cache.clear()

    def __call__(self, poses_np):
        """(B,6) numpy scene-unit poses -> (B,3,200,300) GPU tensor."""
        B = poses_np.shape[0]
        keys = [tuple(np.round(p, 2)) for p in poses_np]
        missing = [i for i, k in enumerate(keys) if k not in self.cache]
        if missing:
            p = poses_np[missing].copy()
            p[:, 3] += self.K["dyaw"]        # mount offsets, as the original
            p[:, 4] += self.K["dpitch"]
            p[:, 5] += self.K["droll"]
            imgs = render_batch_pinhole_gpu(
                p, self.scene,
                K={k: self.K[k] for k in ("fx", "fy", "cx", "cy")},
                chunk=self.chunk, device=self.device).cpu()
            for j, i in enumerate(missing):
                if len(self.cache) >= self.max_size:
                    self.cache.pop(next(iter(self.cache)))
                self.cache[keys[i]] = imgs[j]
        out = torch.stack([self.cache[k] for k in keys])
        return out.to(self.device)


def build_meta(cfg, args):
    return dict(
        arch="ControllerMeansubCtbrVel (scripts_control/utils_ctrl_meansub_ctbrvel.py)",
        parent=("Controller in scripts_control/utils_ctrl_lya_pt.py "
                "(58,036 params; flight artifact weights/ctrl_lya.pt)"),
        input_spec=("3ch mean-subtracted RGB ONLY: x - x.mean(dim=(2,3), "
                    "keepdim=True) per image per channel; RGB 200x300 in "
                    "[0,1] before meansub (OLD pinhole frames; was 6ch "
                    "[raw, raw-mean] at 192x256 fisheye)"),
        camera_model=("OLD PINHOLE (pre-27cd577 render() path): raster "
                      "300x200, fx=113.258171 fy=113.347599 cx=158.868074 "
                      "cy=98.837772, gsplat rasterization camera_model="
                      "'pinhole' (the default the old code relied on; we "
                      "pass it explicitly), NO resize; per-epoch intrinsics"
                      " jitter fx,fy +-0.4% cx,cy +-0.5px mount +-0.5deg "
                      "(original magnitudes) on this K"),
        head=("CTBR-format output [c, wx, wy, wz]: c = G + clamp_relu(raw0,"
              "1)*0.9G in [0.1G,1.9G] m/s^2; w = clamp_relu(raw*RATE_LIM, "
              "RATE_LIM), RATE_LIM=(4,4,2) rad/s (PixelCTBR pixel2ctbr/"
              "policy.py + pixel2ctbr_ff/policy_ff.py ctbr conventions, "
              "copied); last layer zero-init -> hover [G,0,0,0]"),
        ctbr_to_velocity_map=("FIXED affine interface (utils_ctrl_meansub_"
                              "ctbrvel.ctbr_to_velocity): vx = -wy/4, "
                              "vy = +wx/4, vz = -(c-G)/(0.9G), yaw_rate = "
                              "0.3*wz/2 -- each CTBR slot's full range maps"
                              " exactly onto the original velocity head's "
                              "command box ([-1,1] u/s, +-0.3 rad/s); the "
                              "sim consumes the mapped command with the "
                              "ORIGINAL kinematic plant"),
        param_count=56836,
        param_count_original=58036,
        plant=(f"ORIGINAL kinematic velocity plant (train_ctrl_lya_pt.py): "
               f"body_to_world_velocity -> +N(0,{cfg.actuation_noise}) -> "
               f"FIFO delay {cfg.latency_steps} step -> first-order lag "
               f"tau={cfg.actuator_tau}s -> pose += v*dt, dt={cfg.dt}"),
        frame=("gate-centered world frame, z DOWN, +y through gate; "
               "everything in scene units (1 u = 0.85 m for reporting)"),
        target_pose_u=cfg.target_pose.tolist(),
        task="one-gate viewpoint hold: servo to 1.5 u in front of +y gate face and hover",
        loss_deviation=("NONE: losses/curriculum/horizons/DR/LR schedule "
                        "imported from train_ctrl_lya_pt.py and applied "
                        "verbatim (no floor term, no velocity term)"),
        train_cmd="python scripts_control/train_ctrl_meansub_ctbrvel.py " + " ".join(
            sys.argv[1:]),
        trainer="scripts_control/train_ctrl_meansub_ctbrvel.py (adapted from "
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
    ds = PoseDataset(cfg.target_pose, N=cfg.n_poses)   # ORIGINAL ranges
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)

    ctrl = ControllerMeansubCtbrVel().to(device)
    Vnet = Lyapunov().to(device)
    domain_rand = DomainRandomizer()                   # ORIGINAL image DR

    # ORIGINAL first-order actuator lag discretization
    lag_alpha = 1.0 - np.exp(-cfg.dt / cfg.actuator_tau) \
        if cfg.actuator_tau > 0 else 1.0

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
            for pg in opt.param_groups:
                pg['lr'] = cfg.lr * (cfg.lr_decay ** (start_epoch // cfg._lr_every))
        print(f"[INFO] resumed {cfg.last_path} at epoch {start_epoch} "
              f"(best_score {best_score:.4f}, lr {opt.param_groups[0]['lr']:.2e})")
    if start_epoch >= cfg.epochs:
        print("TRAINING COMPLETE (already at final epoch)")
        return

    target = torch.tensor(cfg.target_pose, device=device,
                          dtype=torch.float32).unsqueeze(0)

    renderer = CachedPinholeRenderer(scene, device, chunk=cfg.chunk,
                                     max_size=3000)

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
        # camera-model DR: new render intrinsics + mount this epoch (ORIGINAL)
        if cfg.intrinsics_jitter:
            renderer.jitter()
        weights = cfg.get_loss_weights(ep)
        H = cfg.get_horizon(ep)

        # ORIGINAL LR decay: *0.95 every 10 epochs
        if ep > 0 and ep % cfg._lr_every == 0:
            for pg in opt.param_groups:
                pg['lr'] *= cfg.lr_decay

        tot = {'total': 0.0, 'traj': 0.0, 'decrease': 0.0, 'final_state': 0.0}
        n_batches = 0

        for batch_idx, pose_batch in enumerate(dl):
            pose_batch = pose_batch.to(device).float()
            batch_size = pose_batch.size(0)
            target_batch = target.expand(batch_size, -1)

            initial_pose = pose_batch.clone()

            # ===== Render initial image (cached, ORIGINAL semantics) =====
            img_curr = renderer(pose_batch.detach().cpu().numpy())

            # ===== rollout (ORIGINAL, verbatim; only the head->command
            # step gains the fixed ctbr_to_velocity interface) =====
            pose_curr = pose_batch
            V_curr, alpha_reg_curr = Vnet(pose_curr, target_batch)

            V_list = [V_curr]
            pose_list = [pose_curr]
            alpha_reg_list = [alpha_reg_curr]

            vel_applied = torch.zeros(batch_size, 4, device=device)
            cmd_buffer = deque(
                [torch.zeros(batch_size, 4, device=device)
                 for _ in range(cfg.latency_steps)]
            ) if cfg.latency_steps > 0 else None

            for step in range(H):
                # domain randomization on the (detached) observation only
                ctbr = ctrl(domain_rand(img_curr))
                pred_self = ctbr_to_velocity(ctbr)     # the FIXED interface
                pred = body_to_world_velocity(pred_self, pose_curr[:, 3])
                if cfg.actuation_noise > 0:
                    pred = pred + torch.randn_like(pred) * cfg.actuation_noise

                # transport delay (ORIGINAL FIFO)
                if cmd_buffer is not None:
                    cmd_buffer.append(pred)
                    pred = cmd_buffer.popleft()

                # first-order actuator lag (ORIGINAL)
                vel_applied = vel_applied + lag_alpha * (pred - vel_applied)

                zeros = torch.zeros(*vel_applied.shape[:-1], 2,
                                    device=vel_applied.device,
                                    dtype=vel_applied.dtype)
                vel_full = torch.cat([vel_applied, zeros], dim=-1)

                pose_next = pose_curr + vel_full * cfg.dt

                img_next = renderer(pose_next.detach().cpu().numpy())
                V_next, alpha_reg_next = Vnet(pose_next, target_batch)

                pose_curr = pose_next
                img_curr = img_next

                V_list.append(V_next)
                pose_list.append(pose_next)
                alpha_reg_list.append(alpha_reg_next)

            # ===== ORIGINAL losses, verbatim =====
            loss_traj = compute_traj_loss(initial_pose, pose_list[-1],
                                          target_batch, H=H, dt=cfg.dt)
            loss_decrease = compute_lyapunov_decrease_loss(
                V_list, alpha_reg_list, decay_ratio=0.1, scale_increase=5.0,
                w_smooth=0.15, w_alpha_reg=0.1)
            loss_final = compute_final_state_loss(pose_list[-1], target_batch)
            loss_total = (weights['w_traj'] * loss_traj +
                          weights['w_decrease'] * loss_decrease +
                          weights['w_final_state'] * loss_final)

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
                          "lr": f"{opt.param_groups[0]['lr']:.1e}"})

        # rolling save (resume source)
        save_ckpt(cfg.last_path, ep + 1)

        # periodic quick eval -> bank best
        last_ep = (ep + 1 == cfg.epochs)
        if last_ep or (ep + 1) % args.eval_every == 0:
            m = evaluate(ctrl, scene, device, episodes=args.eval_episodes,
                         seed=123, batch=min(cfg.batch_size, 32),
                         chunk=cfg.chunk)
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
    ap.add_argument("--epochs", type=int, default=120)      # original
    ap.add_argument("--poses", type=int, default=2000,      # original
                    help="poses per epoch (original: 2000)")
    ap.add_argument("--batch", type=int, default=32)        # original
    ap.add_argument("--lr", type=float, default=1e-3)       # original
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=8,
                    help="render chunk (GPU-memory safety)")
    ap.add_argument("--eval-every", type=int, default=15)
    ap.add_argument("--eval-episodes", type=int, default=64)
    ap.add_argument("--max-minutes", type=float, default=0,
                    help="clean stop after this wall time (0 = no limit)")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore an existing _last checkpoint")
    args = ap.parse_args()
    train(args)
