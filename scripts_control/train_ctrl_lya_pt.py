import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader

from render_image import render, render_batch, load_gsplat_scene
from utils_ctrl_lya_pt import Lyapunov,Controller, transform_drone_velocity_to_world_frame

# =============================
# CONFIG (OPTIMIZED FOR THREE-PHASE CURRICULUM)
# =============================
@dataclass
class Config:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # training
    batch_size = 32
    epochs = 120
    lr = 1e-3
    lr_decay = 0.95

    dt = 0.1

    target_pose = np.array([0.0, -3.0, -0.2, 1.57, 0.0, 0.0])

    # gsplat path
    gsplat_path = "nerfstudio/outputs/uturn/splatfacto/2025-05-09_151825"
    checkpoint = "nerfstudio_models/step-000040005.ckpt"

    save_path = "weights/ctrl_lya.pt"
    figure_path = "training_curves.png"

    # ===== CURRICULUM LEARNING WITH 3 PHASES (100 epochs total) =====
    curriculum_phase1_end = 40
    curriculum_phase2_end = 80
    curriculum_phase3_end = 120

    def get_loss_weights(self, epoch):

        if epoch < self.curriculum_phase1_end:
            # ===== Phase 1: Reach the final state =====
            return {
                'w_traj': 3.0,          # 🔥 dominant: reach target
                'w_decrease': 0.5,      # weak Lyapunov constraint
                'w_final_state': 0.05,   # small terminal precision
            }

        elif epoch < self.curriculum_phase2_end:
            # ===== Phase 2: Stable convergence =====
            alpha = (epoch - self.curriculum_phase1_end) / (
                self.curriculum_phase2_end - self.curriculum_phase1_end
            )

            return {
                'w_traj': 3.0 - 2.5 * alpha,                       # still important
                'w_decrease': 0.5 + 1.0 * alpha,     # 🔥 dominant gradually
                'w_final_state': 0.05 + 2.95*alpha,                # moderate terminal pressure
            }

        else:
            # ===== Phase 3: Exact convergence =====
            return {
                'w_traj': 0.5,        # reduced (already near target)
                'w_decrease': 1.5,    # relaxed
                'w_final_state': 3.0, # 🔥 enforce exact precision
            }


    def get_horizon(self, epoch):
        """
        Gradual horizon scaling to allow controller adaptation.
        """
        if epoch < 20:
            return 7
        elif epoch < 35:
            return 10
        elif epoch < 50:
            return 14
        elif epoch < 70:
            return 18
        elif epoch < 95:
            return 22
        else:
            return 25

    def get_learning_rate(self, epoch):
        """
        Exponential decay with Phase 3 boost to counteract diminishing gradients.
        """
        if epoch < 80:
            return self.lr * (self.lr_decay ** (epoch / 10))
        else:
            # Phase 3: slower decay, with slight boost
            phase3_epoch = epoch - 80
            base_lr = self.lr * (self.lr_decay ** 8)  # lr at epoch 80
            return base_lr * (0.95 ** (phase3_epoch / 10))  # Slower decay

        
# =============================
# DATASET
# =============================
class PoseDataset(Dataset):
    def __init__(self, target, N=2000):
        self.target = target
        self.poses = target + np.random.uniform(
            low=[-2.0, -1.5, -1.0, -0.7, -0.0, -0.0],
            high=[2.0, 1.5, 1.0, 0.7, 0.0, 0.0],
            size=(N, 6)
        )

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, i):
        return torch.tensor(self.poses[i], dtype=torch.float32)



# =============================
# LOSS FUNCTIONS
# =============================
def compute_traj_loss(initial_pose, final_pose, target, H, dt,
                      pos_weight=1.0, ang_weight=0.3):
    """
    Trajectory-level loss:
    - encourages progress toward target
    - works even when target is unreachable
    """

    # ===== distances =====
    init_pos_dist = torch.norm(initial_pose[:, :3] - target[:, :3], dim=-1)
    final_pos_dist = torch.norm(final_pose[:, :3] - target[:, :3], dim=-1)

    init_ang_error = initial_pose[:, 3:] - target[:, 3:]
    final_ang_error = final_pose[:, 3:] - target[:, 3:]

    init_ang_dist = (init_ang_error ** 2).sum(dim=-1)
    final_ang_dist = (final_ang_error ** 2).sum(dim=-1)

    # ===== progress =====
    pos_progress = init_pos_dist - final_pos_dist
    ang_progress = init_ang_dist - final_ang_dist

    # ===== reachability =====
    max_reach_pos = 3 * 1.0 * H * dt
    max_reach_ang = 3 * 0.3 * H * dt

    realistic_reach_pos = torch.minimum(
        torch.full_like(init_pos_dist, max_reach_pos),
        init_pos_dist
    )

    realistic_reach_ang = torch.minimum(
        torch.full_like(init_ang_dist, max_reach_ang),
        init_ang_dist
    )

    # ===== normalized progress =====
    pos_progress_ratio = torch.clamp(
        pos_progress / (realistic_reach_pos + 1e-6), 0.0, 1.0
    )

    ang_progress_ratio = torch.clamp(
        ang_progress / (realistic_reach_ang + 1e-6), 0.0, 1.0
    )

    pos_failure = 1.0 - pos_progress_ratio
    ang_failure = 1.0 - ang_progress_ratio

    # ===== loss =====
    loss_pos = (pos_failure ** 2) * final_pos_dist * pos_weight
    loss_ang = (ang_failure ** 2) * final_ang_dist * ang_weight

    return 0.7 * loss_pos.mean() + 0.3 * loss_ang.mean()

def compute_final_state_loss(final_pose, target, epoch=None):
    """
    Progressive precision loss with adaptive weighting.
    - Strong smooth gradient when errors are small
    - Higher angle weight when position error is small
    """
    pos_error = final_pose[:, :3] - target[:, :3]
    ang_error = final_pose[:, 3:] - target[:, 3:]

    pos_dist = torch.norm(pos_error, dim=-1)
    ang_dist = torch.norm(ang_error, dim=-1)

    # # Smooth strong gradient: -log(x) has grad = -1/x (strong but bounded)
    # # Add small offset to prevent log(0)
    # pos_loss = pos_dist 
    # ang_loss = ang_dist

    pos_loss = torch.abs(pos_error).sum(dim=-1)
    ang_loss = torch.abs(ang_error).sum(dim=-1)

    # Adaptive angle weight: increases as position error decreases
    ang_weight = 1.0 #0.2 + 0.8 * torch.exp(-pos_dist / 0.02)

    total_loss = pos_loss.mean() + (ang_weight * ang_loss).mean()

    return total_loss


def compute_lyapunov_decrease_loss(
    V_traj,
    alpha_traj,
    decay_ratio=0.05,
    eps=1e-3,
    scale_increase=5.0,
    w_smooth=0.15,
    w_alpha_reg=0.05,
    V_threshold=0.02,   # 🔥 key threshold
    w_zero=2.0          # 🔥 weight for forcing V → 0
):
    """
    Modified Lyapunov loss with two regimes:

    1) V > threshold:
        enforce decrease

    2) V <= threshold:
        stop decrease constraint
        enforce V → 0
    """

    device = V_traj[0].device
    H = len(V_traj)

    if H < 2:
        return torch.tensor(0.0, device=device)

    V = torch.stack(V_traj, dim=1)   # (B, H)

    # =============================
    # REGION MASKS
    # =============================
    V_prev = V[:, :-1]
    V_next = V[:, 1:]

    high_mask = (V_prev > V_threshold).float()
    low_mask  = (V_prev <= V_threshold).float()

    # =============================
    # 1. DECREASE LOSS (ONLY WHEN FAR)
    # =============================
    dV = V_next - V_prev

    margin = torch.maximum(
        decay_ratio * V_prev,
        torch.full_like(V_prev, eps)
    )

    excess = torch.clamp(dV + margin, min=0.0)

    weight = 1.0 + (dV > 0).float() * (scale_increase - 1.0)

    temporal = torch.linspace(1.0, 1.4, H - 1, device=device).unsqueeze(0)

    loss_dec = (
        excess.pow(2) * weight * temporal * high_mask
    ).sum() / (high_mask.sum() + 1e-6)

    # =============================
    # 2. ZERO-CONVERGENCE LOSS (NEAR TARGET)
    # =============================
    # Directly push V → 0
    loss_zero = (V_next.pow(2) * low_mask).sum() / (low_mask.sum() + 1e-6)

    # =============================
    # 3. SMOOTHNESS
    # =============================
    if H > 2:
        d2V = dV[:, 1:] - dV[:, :-1]
        loss_smooth = d2V.pow(2).mean()
    else:
        loss_smooth = torch.tensor(0.0, device=device)

    # =============================
    # 4. ALPHA REG
    # =============================
    loss_alpha = torch.stack(alpha_traj, dim=0).mean()

    # =============================
    # FINAL
    # =============================
    loss = (
        loss_dec +
        w_zero * loss_zero +
        w_smooth * loss_smooth +
        w_alpha_reg * loss_alpha
    )

    return loss

# =============================
# IMAGE CACHE FOR EFFICIENCY
# =============================
class ImageCache:
    """Cache rendered images to avoid redundant rendering."""
    def __init__(self, max_size=3000):
        self.cache = {}
        self.max_size = max_size
    
    def key(self, pose):
        return tuple(np.round(pose, 2))
    
    def get(self, pose):
        k = self.key(pose)
        return self.cache.get(k)
    
    def set(self, pose, img):
        k = self.key(pose)
        if len(self.cache) >= self.max_size:
            self.cache.pop(next(iter(self.cache)))
        self.cache[k] = img

# =============================
# PLOTTING FUNCTION
# =============================
def plot_training_curves(loss_hist, cfg):
    plt.figure(figsize=(15, 5))

    epochs = range(1, len(loss_hist['total']) + 1)

    # ===== 1. Main losses =====
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss_hist['total'], linewidth=2, color='black', label='Total')
    plt.plot(epochs, loss_hist['traj'], label='Traj', linewidth=2)
    plt.plot(epochs, loss_hist['final_state'], label='Final', linewidth=2)
    plt.plot(epochs, loss_hist['decrease'], label='Lya', linewidth=2)
    plt.xlim(0, cfg.epochs)   # 👈 固定范围
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Primary Losses')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # ===== 2. Curriculum =====
    plt.subplot(1, 2, 2)
    ax = plt.gca()

    # 4 phases
    ax.axvline(cfg.curriculum_phase1_end, linestyle='--', linewidth=2, alpha=0.7, label='P1→P2')
    ax.axvline(cfg.curriculum_phase2_end, linestyle='--', linewidth=2, alpha=0.7, label='P2→P3')

    #plt.plot(epochs, loss_hist['total'], linewidth=2, color='black', label='Total')
    ax.plot(epochs, loss_hist['traj'], alpha=0.6, label='Traj')
    ax.plot(epochs, loss_hist['final_state'], alpha=0.6, label='Final')
    ax.plot(epochs, loss_hist['decrease'], alpha=0.6, label='Lya')

    ax.set_xlim(0, cfg.epochs)   # 👈 固定范围
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Curriculum Phases')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig('figures/'+cfg.figure_path, dpi=150)
    plt.close()   # 👈 非常重要（防止内存爆）

# =============================
# TRAINING
# =============================
def train():
    cfg = Config()
    device = cfg.device

    scene = load_gsplat_scene(cfg)
    ds = PoseDataset(cfg.target_pose)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)

    ctrl = Controller().to(device)
    Vnet = Lyapunov().to(device)
    
    # Single optimizer
    opt = torch.optim.Adam(
        list(ctrl.parameters()) + list(Vnet.parameters()),
        lr=cfg.lr
    )

    start_epoch = 0
    if os.path.exists(cfg.save_path):
        ckpt = torch.load(cfg.save_path, map_location=device)
        ctrl.load_state_dict(ckpt["controller"])
        Vnet.load_state_dict(ckpt["lyapunov"])
        start_epoch = ckpt.get("epoch", 0)
        print(f"[INFO] Checkpoint loaded. Resuming from epoch {start_epoch}")

    target = torch.tensor(cfg.target_pose, device=device, dtype=torch.float32).unsqueeze(0)
    
    # Image cache
    img_cache = ImageCache(max_size=3000)
    
    def get_image(pose_np):
        """Get image with caching to avoid redundant renders."""
        cached = img_cache.get(pose_np)
        if cached is not None:
            return cached
        
        img = render(pose_np, scene, device=device)
        img_cache.set(pose_np, img)
        return img

    # Loss history tracking
    loss_hist = {
        'total': [], 
        'traj': [],
        'decrease': [], 
        'final_state': [], 
    }

    pbar = tqdm(range(start_epoch, cfg.epochs), desc="Training", dynamic_ncols=True, leave=True)

    for ep in pbar:
        # ===== Curriculum weights and horizon =====
        weights = cfg.get_loss_weights(ep)
        H = cfg.get_horizon(ep)

        # ===== LR decay =====
        if ep > 0 and ep % 10 == 0:
            for param_group in opt.param_groups:
                param_group['lr'] *= cfg.lr_decay

        total_loss = 0.0
        total_traj = 0.0
        total_decrease = 0.0
        total_final = 0.0
        n_batches = 0

        for batch_idx, pose_batch in enumerate(dl):
            pose_batch = pose_batch.to(device).float()
            batch_size = pose_batch.size(0)
            target_batch = target.expand(batch_size, -1)

            initial_pose = pose_batch.clone()

            # ===== Render initial image =====
            img_list = []
            for i in range(batch_size):
                pose_np = pose_batch[i].detach().cpu().numpy()
                img = get_image(pose_np)
                img_list.append(img)
            img_curr = torch.stack(img_list)

            # ===== rollout =====
            pose_curr = pose_batch
            # print(target_batch[..., :4].shape)
            V_curr, alpha_reg_curr = Vnet(pose_curr, target_batch)

            V_list = [V_curr]
            pose_list = [pose_curr]
            img_list_traj = [img_curr]
            alpha_reg_list = [alpha_reg_curr]

            for step in range(H):

                pred_self = ctrl(img_curr)
                pred= transform_drone_velocity_to_world_frame(pred_self)
                zeros = torch.zeros(*pred.shape[:-1], 2, device=pred.device, dtype=pred.dtype)
                pred = torch.cat([pred, zeros], dim=-1)

                pose_next = pose_curr + pred * cfg.dt

                # render next
                img_list = []
                for i in range(batch_size):
                    pose_np = pose_next[i].detach().cpu().numpy()
                    img = get_image(pose_np)
                    img_list.append(img)
                img_next = torch.stack(img_list)

                V_next, alpha_reg_next = Vnet(pose_next, target_batch)

                pose_curr = pose_next
                img_curr = img_next

                V_list.append(V_next)
                pose_list.append(pose_next)
                img_list_traj.append(img_next)
                alpha_reg_list.append(alpha_reg_next)

            # =========================================================
            # LOSS 1: TRAJECTORY LOSS (progress / reachability)
            # =========================================================
            loss_traj = compute_traj_loss(
                initial_pose,
                pose_list[-1],
                target_batch,
                H=H,
                dt=cfg.dt
            )

            # =========================================================
            # LOSS 2: LYAPUNOV DECREASE
            # =========================================================
            loss_decrease = compute_lyapunov_decrease_loss(
                V_list,
                alpha_reg_list,
                decay_ratio=0.1,
                scale_increase=5.0,
                w_smooth=0.15,
                w_alpha_reg=0.1
            )

            # =========================================================
            # LOSS 3: FINAL STATE PRECISION
            # =========================================================
            final_pose = pose_list[-1]
            loss_final = compute_final_state_loss(
                final_pose,
                target_batch
            )

            # =========================================================
            # COMBINED LOSS
            # =========================================================
            loss_total = (
                weights['w_traj'] * loss_traj +
                weights['w_decrease'] * loss_decrease +
                weights['w_final_state'] * loss_final
            )

            # ===== backward =====
            opt.zero_grad()
            loss_total.backward()

            torch.nn.utils.clip_grad_norm_(
                list(ctrl.parameters()) + list(Vnet.parameters()),
                max_norm=1.0
            )

            opt.step()

            # ===== logging =====
            total_loss += loss_total.item()
            total_traj += loss_traj.item()
            total_decrease += loss_decrease.item()
            total_final += loss_final.item()
            n_batches += 1

        # ===== epoch stats =====
        avg_loss = total_loss / n_batches
        avg_traj = total_traj / n_batches
        avg_decrease = total_decrease / n_batches
        avg_final = total_final / n_batches

        loss_hist['total'].append(avg_loss)
        loss_hist['traj'].append(avg_traj)
        loss_hist['decrease'].append(avg_decrease)
        loss_hist['final_state'].append(avg_final)

        plot_training_curves(loss_hist, cfg)
        # Logging
        if ep < cfg.curriculum_phase1_end:
            phase = "P1"
        elif ep < cfg.curriculum_phase2_end:
            phase = "P2"
        else:
            phase = "P3"
        
        pbar.set_description(f"Ep {ep+1:03d} [{phase}|H={H:2d}]")

        pbar.set_postfix({
            "loss": f"{avg_loss:.4f}",
            "traj": f"{avg_traj:.4f}",
            "dec": f"{avg_decrease:.4f}",
            "fin": f"{avg_final:.4f}",
            "w_traj": f"{weights['w_traj']:.2f}",
            "w_dec": f"{weights['w_decrease']:.2f}",
            "w_fin": f"{weights['w_final_state']:.2f}",
        })

        # Save checkpoint
        torch.save({
            "controller": ctrl.state_dict(),
            "lyapunov": Vnet.state_dict(),
            "epoch": ep + 1,
        }, cfg.save_path)


if __name__ == "__main__":
    train()
