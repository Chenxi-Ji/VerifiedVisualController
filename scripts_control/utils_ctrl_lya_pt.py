import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def clamp_relu(x, limit):
    """Exact clamp(x, -limit, +limit) written with ReLUs only, so the op is
    natively supported and tightly bounded by alpha-beta-CROWN / auto_LiRPA
    (torch.clamp would export as Clip, which has weaker support)."""
    return torch.relu(x + limit) - torch.relu(x - limit) - limit


# =============================
# VISION CONTROLLER
# =============================
class Controller(nn.Module):
    """CNN vision controller.

    Verifiable + drone-deployable:
      - ops are Conv/BatchNorm/ReLU/AvgPool/Linear + a linear input mean-subtract
        (all alpha-beta-CROWN-supported; BN folds into the conv at inference)
      - per-image per-channel mean subtraction in forward() -> invariant to global
        color/brightness cast (the dominant sim-to-real gap)
      - coarse (3x4) spatial readout (not global pool) so the head sees where the
        gate is, instead of averaging position away
      - action clamping uses the exact ReLU formulation (clamp_relu)
    NOTE: widened backbone (prof-approved) -> more capacity but higher verify cost.
    """
    def __init__(self):
        super().__init__()

        # Widened backbone (prof-approved): ~2x channels for capacity/robustness.
        # Still Conv/BN/ReLU/AvgPool only -> alpha-beta-CROWN-verifiable (costs
        # more to verify). Sizes shown for the 192x256 input.
        self.backbone = nn.Sequential(
            nn.AvgPool2d(2),                       # 192x256 -> 96x128
            nn.Conv2d(6, 32, 5, 2, 2),             # 6ch = raw RGB + mean-sub RGB -> 48x64
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),            # -> 24x32
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 96, 3, 2, 1),            # -> 12x16
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Conv2d(96, 128, 3, 2, 1),           # -> 6x8
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # COARSE spatial readout instead of global (1,1) pool, so the action
            # head can see *where* the gate is (left/center/right, up/mid/down,
            # gate scale/offset/asymmetry). 4x6 = finer layout than 3x4.
            nn.AdaptiveAvgPool2d((4, 6)),          # -> 128 x 4 x 6
            nn.Flatten(),                          # -> 3072
        )

        self.action_head = nn.Sequential(
            nn.Linear(128 * 4 * 6, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )

        self.scale_t = 1.0
        self.scale_r = 0.3

    def forward(self, x):
        """
        x: (B, 3, H, W) RGB images in [0, 1]
        Returns: (B, 4) action [vx, vy, vz, yaw_rate] in the drone body frame
        """
        # Feed BOTH raw RGB and per-channel mean-subtracted RGB (6 channels):
        #   - mean-sub removes ADDITIVE per-image/per-channel color bias (warm/cool cast)
        #   - raw keeps absolute brightness/color, which still carries useful info
        # Mean-sub does NOT remove contrast, gamma, clipping, saturation, or
        # exposure-scale (gain) changes -- DR covers those. Both branches are linear
        # in x, so this stays alpha-beta-CROWN-verifiable.
        x = torch.cat([x, x - x.mean(dim=(2, 3), keepdim=True)], dim=1)

        features = self.backbone(x)
        action = self.action_head(features)

        v_t = clamp_relu(action[:, :3] * self.scale_t, 1.0)
        v_r = clamp_relu(action[:, 3:] * self.scale_r, 0.3)

        return torch.cat([v_t, v_r], dim=-1)


# =============================
# DOMAIN RANDOMIZATION
# =============================
class DomainRandomizer:
    """Image-space domain randomization applied to rendered observations
    during training. Runs on GPU, per-sample, on detached images, so the
    deployed/verified network is untouched — it just sees a wider visual
    distribution (lighting, color balance, sensor noise, blur, occlusion).
    """
    def __init__(self, contrast=0.25, color=0.14, saturation=0.30, exposure=0.08,
                 gamma=0.30, noise_std=0.03, blur_p=0.30,
                 cutout_p=0.3, cutout_frac=0.18,
                 geo_p=0.8, geo_rot_deg=1.0, geo_scale=0.02, geo_trans=0.01):
        # Pure additive brightness was dropped (mean-sub cancels a uniform shift),
        # but `exposure` is kept SMALL on purpose: a shift FOLLOWED BY clamp simulates
        # auto-exposure clipping (saturated highlights / crushed blacks) — nonlinear,
        # so mean-sub canNOT undo it. DR also covers contrast/saturation/gamma/noise/
        # blur/occlusion/geometry, none of which mean-sub removes.
        self.contrast = contrast
        self.color = color
        self.saturation = saturation
        self.exposure = exposure
        self.gamma = gamma
        self.noise_std = noise_std
        self.blur_p = blur_p
        self.cutout_p = cutout_p
        self.cutout_frac = cutout_frac
        # geometric jitter: robustness to the residual between the gsplat
        # equidistant-fisheye render and the real IMX412 lens (k1..k4 ≲1px at
        # the edge, ~0.5px calib error, rolling shutter, lens-unit variation).
        self.geo_p = geo_p
        self.geo_rot_deg = geo_rot_deg
        self.geo_scale = geo_scale
        self.geo_trans = geo_trans

    @torch.no_grad()
    def __call__(self, imgs):
        """imgs: (B, 3, H, W) in [0, 1] -> randomized copy, same shape."""
        B, _, H, W = imgs.shape
        dev = imgs.device
        x = imgs.clone()

        # small geometric jitter (rotation / scale / translation) on the rendered
        # scene before the photometric/sensor effects below — widens the geometric
        # distribution so the net tolerates camera-model / calibration residual.
        geo_mask = torch.rand(B, device=dev) < self.geo_p
        if geo_mask.any():
            idx = torch.where(geo_mask)[0]
            n = idx.numel()
            ang = (torch.rand(n, device=dev) * 2 - 1) * (self.geo_rot_deg * np.pi / 180.0)
            sc = 1.0 + (torch.rand(n, device=dev) * 2 - 1) * self.geo_scale
            tx = (torch.rand(n, device=dev) * 2 - 1) * self.geo_trans * 2.0
            ty = (torch.rand(n, device=dev) * 2 - 1) * self.geo_trans * 2.0
            cos, sin = torch.cos(ang) * sc, torch.sin(ang) * sc
            theta = torch.zeros(n, 2, 3, device=dev)
            theta[:, 0, 0], theta[:, 0, 1], theta[:, 0, 2] = cos, -sin, tx
            theta[:, 1, 0], theta[:, 1, 1], theta[:, 1, 2] = sin, cos, ty
            grid = F.affine_grid(theta, [n, 3, H, W], align_corners=False)
            x[idx] = F.grid_sample(x[idx], grid, mode='bilinear',
                                   padding_mode='border', align_corners=False)

        # gamma (illumination response)
        g = 1.0 + (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1) * self.gamma
        x = x.clamp(1e-4, 1.0) ** g

        # contrast only (additive brightness removed — mean-sub makes it a no-op)
        c = 1.0 + (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1) * self.contrast
        m = x.mean(dim=(2, 3), keepdim=True)
        x = (x - m) * c + m

        # saturation jitter: scale chroma about per-pixel gray. Targets the vivid
        # orange (real) vs muted (render) gate — which mean-subtraction does NOT
        # fix, since it only removes the per-channel mean, not chroma intensity.
        gray = x.mean(dim=1, keepdim=True)
        s = 1.0 + (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1) * self.saturation
        x = gray + (x - gray) * s

        # per-channel color gain (white-balance / relative-channel jitter)
        x = x * (1.0 + (torch.rand(B, 3, 1, 1, device=dev) * 2 - 1) * self.color)

        # exposure jitter WITH clipping: shift then clamp -> simulates auto-exposure
        # saturating highlights / crushing blacks (nonlinear; mean-sub can't undo it)
        x = (x + (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1) * self.exposure).clamp(0.0, 1.0)

        # light blur on a random subset (defocus / motion approximation)
        blur_mask = torch.rand(B, device=dev) < self.blur_p
        if blur_mask.any():
            xb = F.avg_pool2d(x[blur_mask], 3, stride=1, padding=1)
            x[blur_mask] = xb

        # sensor noise
        x = x + torch.randn_like(x) * self.noise_std

        # cutout occlusion (nets/cables/other drones crossing the view)
        cut_mask = torch.rand(B, device=dev) < self.cutout_p
        for i in torch.where(cut_mask)[0]:
            ch = int(self.cutout_frac * H * (0.4 + 0.6 * torch.rand(1).item()))
            cw = int(self.cutout_frac * W * (0.4 + 0.6 * torch.rand(1).item()))
            cy = int(torch.randint(0, H - ch, (1,)).item())
            cx = int(torch.randint(0, W - cw, (1,)).item())
            x[i, :, cy:cy + ch, cx:cx + cw] = torch.rand(3, 1, 1, device=dev)

        return x.clamp_(0.0, 1.0)


# =============================
# POSITIVE DEFINITE LYAPUNOV FUNCTION
# =============================
class Lyapunov(nn.Module):
    """
    V(x) = α(x) * ||pos_error||² + (1 - α(x)) * Σ(1 - cos(angle_error))

    改进点：
    1. α网络输入正则化，确保梯度信息清晰
    2. 添加显式的输入缩放，避免数值不稳定
    3. 保持α ∈ (0,1)，确保凸组合的有效性

    where:
        α(x) = sigmoid(f(norm_pos_dist, norm_ang_dist))
    """
    def __init__(self, hidden_dims=None, pos_scale=2.0, ang_scale=3.0):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [32, 32]

        # α network: input = 2 scalars (normalized distances)
        layers = []
        in_dim = 2
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.alpha_net = nn.Sequential(*layers)

        # 输入正则化参数（基于典型的误差范围）
        self.pos_scale = pos_scale      # 位置误差的典型范围（米）
        self.ang_scale = ang_scale      # 角度误差的典型范围（弧度）

        # Lyapunov函数的整体缩放
        self.v_scale = 3.0


    def forward(self, x, target):
        """
        Args:
            x: (B, 6) 当前pose [pos(3), angle(3)]
            target: (B, 6) 目标pose

        Returns:
            V: (B,) Lyapunov函数值
        """
        pos_error = x[:, :3] - target[:, :3]              # (B, 3)
        ang_error = x[:, 3:4] - target[:, 3:4]            # (B, 1)
        # print(pos_error, ang_error)

        # --- 基础项 ---
        pos_term = (pos_error ** 2).sum(dim=-1)           # (B,)
        ang_term = (1.0 - torch.cos(ang_error)).sum(dim=-1)  # (B,)

        # --- 正则化的距离输入 ---
        pos_norm = torch.norm(pos_error, dim=-1, keepdim=True)  # (B, 1)
        ang_norm = ang_term.unsqueeze(-1)                       # (B, 1)

        # 正则化：将距离映射到合理的输入范围 [0, ~1]
        # 这确保网络接收到有意义的梯度信号
        pos_norm_scaled = pos_norm / self.pos_scale       # (B, 1)
        ang_norm_scaled = ang_norm / self.ang_scale       # (B, 1)

        alpha_input = torch.cat([pos_norm_scaled, ang_norm_scaled], dim=-1)  # (B, 2)

        # --- α计算 ---
        alpha_logit = self.alpha_net(alpha_input)         # (B, 1)
        alpha = torch.sigmoid(alpha_logit).squeeze(-1)    # (B,) ∈ (0, 1)

        # --- Lyapunov函数 (凸组合形式) ---
        V = self.v_scale * (alpha * pos_term + (1.0 - alpha) * ang_term)

        # --- 熵正则化（可选，防止α坍缩到0或1） ---
        eps = 1e-6
        alpha_clamped = torch.clamp(alpha, eps, 1 - eps)
        # 熵最大化：当α=0.5时最大，此时log(4*α*(1-α)) = 0
        alpha_reg = -torch.mean(torch.log(4.0 * alpha_clamped * (1 - alpha_clamped)))

        return V, alpha_reg


# =============================
# BODY <-> WORLD VELOCITY
# =============================
def body_to_world_velocity(vel_body: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Rotate a body-frame velocity command into the world frame.

    Gate-centered world frame (world_frame.json): z down, yaw about z,
    yaw=0 means body x points along world +x. Standard NED-style rotation:
        v_world_xy = Rz(yaw) @ v_body_xy,  vz and yaw_rate pass through.

    Args:
        vel_body: (..., 4) [vx, vy, vz, yaw_rate] in body frame
        yaw: (...,) current yaw
    """
    cy = torch.cos(yaw)
    sy = torch.sin(yaw)
    vx = cy * vel_body[..., 0] - sy * vel_body[..., 1]
    vy = sy * vel_body[..., 0] + cy * vel_body[..., 1]
    return torch.stack([vx, vy, vel_body[..., 2], vel_body[..., 3]], dim=-1)


def body_to_world_velocity_np(vel_body: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """NumPy version of body_to_world_velocity."""
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    vx = cy * vel_body[..., 0] - sy * vel_body[..., 1]
    vy = sy * vel_body[..., 0] + cy * vel_body[..., 1]
    return np.stack([vx, vy, vel_body[..., 2], vel_body[..., 3]], axis=-1)


# =============================
# (LEGACY uturn-scene fixed-axis velocity transform [-vx, vy, -vz; -yaw] REMOVED
#  2026-06-25 — it was dead code and a footgun sitting next to the real
#  body_to_world_velocity above. The active rollout/test path uses the correct
#  yaw-aware body_to_world_velocity / _np ONLY.)
# =============================
