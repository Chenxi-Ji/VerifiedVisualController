import torch
import torch.nn as nn

# =============================
# VISION CONTROLLER
# =============================
class Controller(nn.Module):
    """CNN-based vision controller with action clamping."""
    def __init__(self, img_size=64):
        super().__init__()
        
        # Vision backbone (efficient)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 5, 2, 2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        
        # Action head
        self.action_head = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 6),
        )
        
        self.scale_t = 1.0
        self.scale_r = 0.3

    def forward(self, x):
        """
        x: (B, 3, H, W) RGB images
        Returns: (B, 6) action [vx, vy, vz, wx, wy, wz]
        """
        features = self.backbone(x)
        action = self.action_head(features)
        
        v_t = action[:, :3] * self.scale_t
        v_r = action[:, 3:] * self.scale_r
        
        v_t = torch.clamp(v_t, min=-1.0, max=1.0)
        v_r = torch.clamp(v_r, min=-0.3, max=0.3)
        
        return torch.cat([v_t, v_r], dim=-1)


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
        ang_error = x[:, 3:] - target[:, 3:]              # (B, 3)

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

