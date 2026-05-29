from time import time
from tqdm import tqdm
import torch
from itertools import product
from utils_rational_quad import rational_quad_bound
from utils_alpha_blending import compute_interval_bound_alpha_blending, compute_linear_bound_alpha_blending, compute_alpha_blending, compute_bound_exp  

@torch.no_grad()
def qvec2rotmat_batched(q):
        """Convert batch of quaternions to rotation matrices."""
        q_norm = q / (q.norm(dim=-1, keepdim=True))
        w, x, y, z_q = q_norm.unbind(-1)
        # x, y, z_q, w = q_norm.unbind(-1)
        
        rotmat = torch.zeros(q.shape[0], 3, 3, device=q.device, dtype=q.dtype)
        rotmat[:, 0, 0] = 1 - 2*(y*y + z_q*z_q)
        rotmat[:, 0, 1] = 2*(x*y - w*z_q)
        rotmat[:, 0, 2] = 2*(x*z_q + w*y)
        rotmat[:, 1, 0] = 2*(x*y + w*z_q)
        rotmat[:, 1, 1] = 1 - 2*(x*x + z_q*z_q)
        rotmat[:, 1, 2] = 2*(y*z_q - w*x)
        rotmat[:, 2, 0] = 2*(x*z_q - w*y)
        rotmat[:, 2, 1] = 2*(y*z_q + w*x)
        rotmat[:, 2, 2] = 1 - 2*(x*x + y*y)
        
        return rotmat

@torch.no_grad()
def build_yaw_list(yaw_lb, yaw_ub, k=6):
    if yaw_ub - yaw_lb < 1e-3:
        return torch.tensor([yaw_lb], device=yaw_lb.device, dtype=yaw_lb.dtype) 
    else:
        yaw_list = torch.linspace(yaw_lb, yaw_ub, steps=k, device=yaw_lb.device, dtype=yaw_lb.dtype)
        return yaw_list

@torch.no_grad()
def compute_bound_Z(pose_lb, pose_ub, const, scale):
    device = pose_lb.device
    dtype = pose_lb.dtype
    N = const.shape[0]

    pos_x_lb, pos_y_lb, pos_z_lb, yaw_lb, pitch_lb, roll_lb = pose_lb
    pos_x_ub, pos_y_ub, pos_z_ub, yaw_ub, pitch_ub, roll_ub = pose_ub

    yaw_list = build_yaw_list(yaw_lb, yaw_ub)

    Z_min = torch.full((N,), float("inf"), device=device, dtype=dtype) # (N, )
    Z_max = torch.full((N,), float("-inf"), device=device, dtype=dtype) # (N, )

    for pos_x, pos_y, yaw in product(
        [pos_x_lb, pos_x_ub],
        [pos_y_lb, pos_y_ub],
        yaw_list,
    ):
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)

        # Compute Z for this combination
        Z = cos_yaw*(const[:, 0] - pos_x*scale) - sin_yaw*(const[:, 2] + pos_y*scale)  # (N, )
        # update bounds
        Z_min = torch.minimum(Z_min, Z)
        Z_max = torch.maximum(Z_max, Z)

    # final result
    return Z_min, Z_max

@torch.no_grad()
def compute_bound_XZ(pose_lb, pose_ub, const, scale):
    device = pose_lb.device
    dtype = pose_lb.dtype
    N = const.shape[0]

    pos_x_lb, pos_y_lb, pos_z_lb, yaw_lb, pitch_lb, roll_lb = pose_lb
    pos_x_ub, pos_y_ub, pos_z_ub, yaw_ub, pitch_ub, roll_ub = pose_ub

    yaw_list = build_yaw_list(yaw_lb, yaw_ub)

    XZ_min = torch.full((N,), float("inf"), device=device, dtype=dtype) # (N, )
    XZ_max = torch.full((N,), float("-inf"), device=device, dtype=dtype) # (N, )

    for pos_x, pos_y, yaw in product(
        [pos_x_lb, pos_x_ub],
        [pos_y_lb, pos_y_ub],
        yaw_list
    ):
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)

        # Compute Z for this combination
        Z = cos_yaw*(const[:, 0] - pos_x*scale) - sin_yaw*(const[:, 2] + pos_y*scale)  # (N, )
        X = sin_yaw*(const[:, 0] - pos_x*scale) + cos_yaw*(const[:, 2] + pos_y*scale)  # (N, )
        XZ = X/Z  # (N, )

        # update bounds
        XZ_min = torch.minimum(XZ_min, XZ)
        XZ_max = torch.maximum(XZ_max, XZ)

    # final result
    return XZ_min, XZ_max

@torch.no_grad()
def compute_bound_YZ(pose_lb, pose_ub, const, scale):
    device = pose_lb.device
    dtype = pose_lb.dtype
    N = const.shape[0]

    pos_x_lb, pos_y_lb, pos_z_lb, yaw_lb, pitch_lb, roll_lb = pose_lb
    pos_x_ub, pos_y_ub, pos_z_ub, yaw_ub, pitch_ub, roll_ub = pose_ub

    yaw_list = build_yaw_list(yaw_lb, yaw_ub)

    YZ_min = torch.full((N,), float("inf"), device=device, dtype=dtype) # (N, )
    YZ_max = torch.full((N,), float("-inf"), device=device, dtype=dtype) # (N, )

    for pos_x, pos_y, pos_z, yaw in product(
        [pos_x_lb, pos_x_ub],
        [pos_y_lb, pos_y_ub],
        [pos_z_lb, pos_z_ub],
        yaw_list,
    ):
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)

        # Compute Z for this combination
        Z = cos_yaw*(const[:, 0] - pos_x*scale) - sin_yaw*(const[:, 2] + pos_y*scale)  # (N, )
        Y = -(const[:, 1] - pos_z*scale)  # (N, )
        YZ =  Y/Z  # (N, )

        # update bounds
        YZ_min = torch.minimum(YZ_min, YZ)
        YZ_max = torch.maximum(YZ_max, YZ)

    # final result
    return YZ_min, YZ_max

@torch.no_grad()
def compute_bound_radius(pose_lb, pose_ub, Z_lb, Z_ub, fx, fy, cam_const, gs_const, scale):
    pos_x_lb, pos_y_lb, pos_z_lb, yaw_lb, pitch_lb, roll_lb = pose_lb
    pos_x_ub, pos_y_ub, pos_z_ub, yaw_ub, pitch_ub, roll_ub = pose_ub

    device = Z_lb.device
    dtype = Z_lb.dtype
    N = Z_lb.shape[0]

    F = torch.tensor([[fx, 0.0, 0.0], [0.0, fy, 0.0]], device=device, dtype=dtype)  # (2, 3)
    E = torch.tensor([
        [0.0, 1.0, 0.0],
        [-1.0,0.0, 0.0],
        [0.0, 0.0, 0.0]
    ]).to(device=device, dtype=dtype)  # (3, 3)
    FE = F@E  # (2, 3)

    yaw_list = build_yaw_list(yaw_lb, yaw_ub)
    max_radius = torch.zeros(N, device=device, dtype=dtype)

    for pos_x, pos_y, pos_z, yaw in product(
        [pos_x_lb, pos_x_ub],
        [pos_y_lb, pos_y_ub],
        [pos_z_lb, pos_z_ub],
        yaw_list,
    ):
        x = torch.stack([pos_x, pos_z, -pos_y], dim=-1)  # (3, )
        X = -x[None, :]*scale + cam_const   # (N, 3)
        Xx = compute_skew_matrix(X)  # (N, 3, 3)

        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)

        Rw = torch.stack([torch.stack([sin_yaw, torch.tensor(0.0, device=device), cos_yaw]),
                        torch.tensor([0.0, -1.0, 0.0], device=device),
                        torch.stack([cos_yaw, torch.tensor(0.0, device=device), -sin_yaw])
        ]) # (3, 3)

        RXx = Rw[None, ...]@Xx  # (N, 3, 3)

        semi_cov2d = FE[None, ...]@RXx@gs_const  # (N, 2, 3)
        max_cov2d_eig = torch.linalg.matrix_norm(semi_cov2d, ord=2) #(N,)
        radius = max_cov2d_eig/Z_lb**2  # (N, )

        max_radius = torch.maximum(max_radius, radius)

    return max_radius


@torch.no_grad()
def compute_skew_matrix(v):
    v0 = v[..., 0] # (..., )
    v1 = v[..., 1]
    v2 = v[..., 2] 

    zero = torch.zeros_like(v0)
    skew = torch.stack([
        torch.stack([zero, -v2, v1], dim=-1),
        torch.stack([v2, zero, -v0], dim=-1),
        torch.stack([-v1, v0, zero], dim=-1)
    ], dim=-2)  # (..., 3, 3)

    return skew

@torch.no_grad()
def compute_adj(M):
    m11 = M[..., 0, 0]
    m12 = M[..., 0, 1]
    m13 = M[..., 0, 2]
    m22 = M[..., 1, 1]
    m23 = M[..., 1, 2]
    m33 = M[..., 2, 2]

    adj_M = torch.stack([
        torch.stack([
            m22 * m33 - m23**2,
            m13 * m23 - m12 * m33,
            m12 * m23 - m13 * m22
        ], dim=-1),

        torch.stack([
            m13 * m23 - m12 * m33,
            m11 * m33 - m13**2,
            m12 * m13 - m11 * m23
        ], dim=-1),

        torch.stack([
            m12 * m23 - m13 * m22,
            m12 * m13 - m11 * m23,
            m11 * m22 - m12**2
        ], dim=-1)

    ], dim=-2)  # (N,3,3)

    return adj_M

@torch.no_grad()
def compute_mahal(pose, Z, dx, dy, cam_const, gs_const, adj_gs_const_T, scale):
    device = Z.device
    dtype = Z.dtype
    N = Z.shape[0]
    H, W = dx.shape

    pos_x, pos_y, pos_z, yaw, pitch, roll = pose

    x = torch.stack([pos_x, pos_z, -pos_y], dim=-1)  # (3, )
    X = -x[None, :]*scale + cam_const   # (N, 3)

    ones = torch.ones((H,W), device=device, dtype=dtype)
    d = torch.stack([dx, dy, ones], dim=-1)  # (H,W,3)
    #D = compute_skew_matrix(d)  # (H,W,3,3)

    sin_yaw = torch.sin(yaw)  # (N,)
    cos_yaw = torch.cos(yaw)  # (N,)

    Rw = torch.stack([torch.stack([sin_yaw, torch.tensor(0.0, device=device), cos_yaw]),
        torch.tensor([0.0, -1.0, 0.0], device=device),
        torch.stack([cos_yaw, torch.tensor(0.0, device=device), -sin_yaw])
    ]) # (3, 3)

    #RDR = Rw.T[None, None, ...]@D@Rw[None, None, ...]  # (H, W, 3,3)
    Rd = Rw.T[None, None, ...]@d[..., None]  # (H, W, 3, 1)
    Rd = Rd.squeeze(-1)  # (H, W, 3)
    Rdx = compute_skew_matrix(Rd)  # (H, W, 3, 3)

    semi_P = Rdx[None, ...]@gs_const[:, None, None, ...]  # (N,H,W,3,3)
    semi_Num = X[:, None, None, None, :] @ semi_P  # (N,H,W,1,3)
    Num = semi_Num@semi_Num.transpose(-1, -2)  # (N,H,W,1,1)
    Num = Num.squeeze(-1).squeeze(-1)  # (N,H,W)

    semi_Denom = X[:, None, :] @ adj_gs_const_T  # (N, 1, 3)
    semi_Denom = semi_Denom/Z[:, None, None] # (N, 1, 3)
    Denom = semi_Denom@semi_Denom.transpose(-1, -2)  # (N, 1, 1)
    Denom = Denom.squeeze(-1).squeeze(-1)  # (N, )

    mahal = Num/Denom[:, None, None]  # (N, H, W)
    mahal = mahal.to(dtype)

    return mahal

@torch.no_grad()
def compute_bound_mahal(pose_lb, pose_ub, Z_lb, Z_ub, dx, dy, cam_const, gs_const, adj_gs_const_T, scale):
    pos_x_lb, pos_y_lb, pos_z_lb, yaw_lb, pitch_lb, roll_lb = pose_lb
    pos_x_ub, pos_y_ub, pos_z_ub, yaw_ub, pitch_ub, roll_ub = pose_ub

    device = Z_lb.device
    dtype = Z_lb.dtype
    N = Z_lb.shape[0]
    H, W = dx.shape

    Z_ratio = Z_ub/Z_lb # (N, )
    x_lb = torch.stack([pos_x_lb, pos_z_lb, -pos_y_ub], dim=-1)  # (3, )
    x_ub = torch.stack([pos_x_ub, pos_z_ub, -pos_y_lb], dim=-1)  # (3, )

    if scale >0:
        X_lb = -x_ub[None, :]*scale + cam_const   # (N, 3)
        X_ub = -x_lb[None, :]*scale + cam_const   # (N, 3)
    else:
        X_lb = -x_lb[None, :]*scale + cam_const   # (N, 3)
        X_ub = -x_ub[None, :]*scale + cam_const   # (N, 3)

    ones = torch.ones((H,W), device=device, dtype=dtype)
    d = torch.stack([dx, dy, ones], dim=-1)  # (H,W,3)
    # D = compute_skew_matrix(d)  # (H,W,3,3)

    yaw_list = build_yaw_list(yaw_lb, yaw_ub)

    mahal_lb = torch.full((N,H,W), float("inf"), device=device, dtype=torch.float64) # (N, H, W)
    mahal_ub = torch.full((N,H,W), float("-inf"), device=device, dtype=torch.float64) # (N, H, W)

    for yaw in yaw_list:
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)

        Rw = torch.stack([torch.stack([sin_yaw, torch.tensor(0.0, device=device), cos_yaw]),
            torch.tensor([0.0, -1.0, 0.0], device=device),
            torch.stack([cos_yaw, torch.tensor(0.0, device=device), -sin_yaw])
        ]) # (3, 3)

        Rd = Rw.T[None, None, ...]@d[..., None]  # (H, W, 3, 1)
        Rd = Rd.squeeze(-1)  # (H, W, 3)
        Rdx = compute_skew_matrix(Rd)  # (H, W, 3, 3)

        semi_P = Rdx[None, ...]@gs_const[:, None, None, ...]  # (N,H,W,3,3)
        adj_semi_Q_T = adj_gs_const_T/Z_lb[:, None, None] # (N, 3, 3)
        
        X_lb = X_lb.to(torch.float64)
        X_ub = X_ub.to(torch.float64)
        semi_P = semi_P.to(torch.float64)
        adj_semi_Q_T = adj_semi_Q_T.to(torch.float64)

        mahal_min, mahal_max = rational_quad_bound(X_lb, X_ub, semi_P, adj_semi_Q_T,
                                                   num_bisect=20, max_cap=5e+1,min_cap=1e-3,tol=5e-4)  # (N, H, W)
        mahal_lb = torch.minimum(mahal_lb, mahal_min)
        mahal_ub = torch.maximum(mahal_ub, mahal_max)

    mahal_lb = mahal_lb.to(dtype)
    mahal_ub = mahal_ub.to(dtype)
    mahal_lb = mahal_lb  # (N, H, W)
    mahal_ub = (Z_ratio**2)[:, None, None]*mahal_ub  # (N, H, W)

    return mahal_lb, mahal_ub

@torch.no_grad()
def compute_mahal_chunked(
    pose,
    Z,
    dx,
    dy,
    cam_const,
    gs_const,
    adj_gs_const_T,
    scale,
    chunk_size=1024,
):
    N = Z.shape[0]
    device = pose.device

    H, W = dx.shape[-2], dx.shape[-1]

    out = torch.empty((N, H, W), device=device, dtype=pose.dtype)
    for i in range(0, N, chunk_size):
        j = min(i + chunk_size, N)

        out[i:j] = compute_mahal(
            pose,
            Z[i:j],
            dx,
            dy,
            cam_const[i:j],
            gs_const[i:j],
            adj_gs_const_T[i:j],
            scale,
        )

    return out
    
@torch.no_grad()
def compute_bound_mahal_chunked(
    pose_lb,
    pose_ub,
    Z_lb,
    Z_ub,
    dx,
    dy,
    cam_const,
    gs_const,
    adj_gs_const_T,
    scale,
    chunk_size=1024,
):

    N = Z_lb.shape[0]
    H, W = dx.shape
    device = Z_lb.device

    mahal_lb = torch.empty((N, H, W), device=device, dtype=Z_lb.dtype)
    mahal_ub = torch.empty((N, H, W), device=device, dtype=Z_lb.dtype)

    for i in range(0, N, chunk_size):
        j = min(i + chunk_size, N)

        mahal_lb[i:j], mahal_ub[i:j] = compute_bound_mahal(
            pose_lb,
            pose_ub,
            Z_lb[i:j],
            Z_ub[i:j],
            dx,
            dy,
            cam_const[i:j],
            gs_const[i:j],
            adj_gs_const_T[i:j],
            scale,
        )

    return mahal_lb, mahal_ub


def compute_bound_rgb(pose_lb, pose_ub, Z_lb, Z_ub, k_lb, k_ub, b_lb, b_ub, dx, dy, cam_const, gs_const, adj_gs_const_T, scale):
    pos_x_lb, pos_y_lb, pos_z_lb, yaw_lb, pitch_lb, roll_lb = pose_lb
    pos_x_ub, pos_y_ub, pos_z_ub, yaw_ub, pitch_ub, roll_ub = pose_ub

    device = Z_lb.device
    dtype = Z_lb.dtype
    N = Z_lb.shape[0]
    H, W = dx.shape

    Z_ratio = Z_ub/Z_lb # (N, )

    ones = torch.ones((H,W), device=device, dtype=dtype)
    d = torch.stack([dx, dy, ones], dim=-1)  # (H,W,3)
    # D = compute_skew_matrix(d)  # (H,W,3,3)

    yaw_list = build_yaw_list(yaw_lb, yaw_ub)

    rgb_lin_lb = torch.full((H, W, 3), float("inf"), device=device, dtype=dtype) # (H, W, 3)
    rgb_lin_ub = torch.full((H, W, 3), float("-inf"), device=device, dtype=dtype) # (H, W, 3)

    for pos_x,pos_y,pos_z, yaw in product(
        [pos_x_lb, pos_x_ub],
        [pos_y_lb, pos_y_ub],
        [pos_z_lb, pos_z_ub],
        yaw_list,
    ):
        x = torch.stack([pos_x, pos_z, -pos_y], dim=-1)  # (3, )
        X = -x[None, :]*scale + cam_const   # (N, 3)

        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)

        Rw = torch.stack([torch.stack([sin_yaw, torch.tensor(0.0, device=device), cos_yaw]),
            torch.tensor([0.0, -1.0, 0.0], device=device),
            torch.stack([cos_yaw, torch.tensor(0.0, device=device), -sin_yaw])
        ]) # (3, 3)

        Rd = Rw.T[None, None, ...]@d[..., None]  # (H, W, 3, 1)
        Rd = Rd.squeeze(-1)  # (H, W, 3)
        Rdx = compute_skew_matrix(Rd)  # (H, W, 3, 3)

        semi_P = Rdx[None, ...]@gs_const[:, None, None, ...]  # (N,H,W,3,3)
        semi_P_lb = k_lb[..., None, None]*semi_P[..., None, :, : ] # (N,H,W,3,3,3)
        # print("semi_P_lb shape:", semi_P_lb.shape)
        semi_Num_lb = X[:, None, None, None, None, :] @ semi_P_lb  # (N,H,W,3,1,3)
        Num_lb = semi_Num_lb@semi_Num_lb.transpose(-1, -2)  # (N,H,W,3,1,1)
        Num_lb = Num_lb.squeeze(-1).squeeze(-1)  # (N,H,W,3)

        semi_P_ub = k_ub[..., None, None]*semi_P[..., None, :, : ] # (N,H,W,3,3,3)
        semi_Num_ub = X[:, None, None, None, None, :] @ semi_P_ub  # (N,H,W,3,1,3)
        Num_ub = semi_Num_ub@semi_Num_ub.transpose(-1, -2)  # (N,H,W,3,1,1)
        Num_ub = Num_ub.squeeze(-1).squeeze(-1)  # (N,H,W,3)

        semi_Denom = X[:, None, :] @ adj_gs_const_T/Z_lb[:, None, None]  # (N, 1, 3)
        Denom = semi_Denom@semi_Denom.transpose(-1, -2)  # (N, 1, 1)
        Denom = Denom.squeeze(-1).squeeze(-1)  # (N, )

        mahal_lb = Num_lb/Denom[:, None, None, None]  # (N, H, W, 3)
        mahal_ub = Num_ub/Denom[:, None, None, None]  # (N, H, W, 3)

        mahal_lb = mahal_lb  # (N, H, W, 3)
        mahal_ub = mahal_ub*Z_ratio[:, None, None, None]**2  # (N, H, W, 3)
        mahal_lb = mahal_lb.sum(dim=0) # (H, W, 3)
        mahal_ub = mahal_ub.sum(dim=0) # (H, W, 3)

        rgb_lin_lb = torch.minimum(rgb_lin_lb, mahal_lb)  # (H, W, 3)
        rgb_lin_ub = torch.maximum(rgb_lin_ub, mahal_ub)  # (H, W, 3)

    rgb_lb = b_lb+rgb_lin_lb # (H, W, 3)
    rgb_ub = b_ub+rgb_lin_ub # (H, W, 3)

    print("rgb_lb.min():", rgb_lb.min().item(), "rgb_lb.max():", rgb_lb.max().item())
    print("rgb_ub.min():", rgb_ub.min().item(), "rgb_ub.max():", rgb_ub.max().item())

    return rgb_lb, rgb_ub


def render_bound(pose_lb, pose_ub, scene, width = 300, height = 200,
            fx = 113.258171, fy = 113.347599, 
            cx = 158.868074, cy = 98.837772, 
            near_plane=0.01, far_plane=1e10,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            dtype=torch.float32,
            chunk_size=1024*2,
            linear = False
            ):
    
    means, quats, opacities, scales, colors, transform, scale = scene

    N = means.shape[0]

    TILE = 32
    ALPHA_THRESHOLD = 1e-3
    MIN_RADIUS = 0.1
    MAX_RADIUS =50.0
    COEFF_RADIUS = 3.0
    sample_idx = 30

    scales=torch.exp(scales)
    opacities=torch.sigmoid(opacities).squeeze(-1)
    transform = torch.as_tensor(transform, dtype=dtype, device=device)
    trans_R = transform[:3, :3]
    trans_T = transform[:3, 3:4]

    pose_lb = torch.as_tensor(pose_lb, dtype=dtype, device=device)
    pose_ub = torch.as_tensor(pose_ub, dtype=dtype, device=device)
    pose = (pose_lb + pose_ub) / 2

    pos_x, pos_y, pos_z, yaw, pitch, roll = pose

    # ============================================================
    # 0. Compute View Matrix
    # ============================================================
    sin_yaw = torch.sin(yaw)
    cos_yaw = torch.cos(yaw)

    Rw = torch.stack([
        torch.stack([sin_yaw, torch.tensor(0.0, device=device), cos_yaw]),
        torch.tensor([0.0, -1.0, 0.0], device=device),
        torch.stack([cos_yaw, torch.tensor(0.0, device=device), -sin_yaw])
    ])
    Tw = torch.stack([pos_x, pos_z, -pos_y]).unsqueeze(-1)

    # view_R = Rw@trans_R.T
    # view_T = -view_R@(trans_R@Tw + trans_T)*scale
    # viewmat = torch.eye(4, device=device, dtype=dtype)
    # viewmat[:3, :3] = view_R
    # viewmat[:3, 3:4] = view_T
    
    # ============================================================
    # 1. Transform to Camera Space (Vectorized)
    # ============================================================
    cam_const = trans_R.T[None, ...]@(means[..., None]-trans_T[None, ...]*scale)  # (N, 3, 1)
    means_cam = Rw[None, ...]@(cam_const-Tw[None, ...]*scale)  # (N, 3, 1)

    cam_const = cam_const.squeeze(-1)  # (N, 3)
    means_cam = means_cam.squeeze(-1)  # (N, 3)

    Z_lb, Z_ub = compute_bound_Z(pose_lb, pose_ub, cam_const, scale)
    assert (Z_lb <= Z_ub).all(), "Z_lb should be less than or equal to Z_ub for all Gaussians"

    # Filter valid Gaussians based on Z_lb
    valid_depth = Z_lb > near_plane
    valid_idx = torch.where(valid_depth)[0]
    N_valid = valid_idx.numel()
    print(f"Total Gaussians: {N}, Valid Gaussians: {N_valid}")

    if N_valid == 0:
        print("⚠️  No valid Gaussians!")
        rgb = torch.zeros(height, width, 3, device=device, dtype=dtype)
        return rgb

    quats = quats[valid_idx]
    scales = scales[valid_idx]
    opacities = opacities[valid_idx]
    colors = colors[valid_idx]
    means_cam = means_cam[valid_idx]  # (N_valid, 3)

    Z_lb = Z_lb[valid_idx]
    Z_ub = Z_ub[valid_idx]
    cam_const = cam_const[valid_idx]

    # ============================================================
    # 2. Compute View Directions and SH Colors
    # ============================================================
    C0 = 0.28209479177387814
    colors_rgb = torch.clamp(C0*colors[:, 0, :] + 0.5, 0.0, 1.0)  # (N_valid, 3)

    # ============================================================
    # 3. Camera Projection (Vectorized)
    # ============================================================
    X = means_cam[:, 0]  # (N_valid,)
    Y = means_cam[:, 1]  # (N_valid,)
    Z = means_cam[:, 2]  # (N_valid,)

    XZ=X/Z  # (N_valid,)
    YZ=Y/Z  # (N_valid,)
    px = fx * XZ + cx  # (N_valid,)
    py = fy * YZ + cy  # (N_valid,)

    XZ_lb, XZ_ub = compute_bound_XZ(pose_lb, pose_ub, cam_const, scale)
    YZ_lb, YZ_ub = compute_bound_YZ(pose_lb, pose_ub, cam_const, scale)
    px_lb, px_ub = fx * XZ_lb + cx, fx * XZ_ub + cx
    py_lb, py_ub = fy * YZ_lb + cy, fy * YZ_ub + cy

    assert (px_lb <= px_ub).all(), "px_lb should be less than or equal to px_ub for all Gaussians"
    assert (py_lb <= py_ub).all(), "py_lb should be less than or equal to py_ub for all Gaussians"

    # ============================================================
    # 4. Compute Radius for each 2D Gaussian (Vectorized)
    # ============================================================
    S = torch.diag_embed(scales)  # (N_valid, 3, 3)
    adj_S = compute_adj(S)  # (N_valid, 3, 3)
    R = qvec2rotmat_batched(quats)  # (N_valid, 3, 3)
    gs_const = trans_R.T[None, ...]@R@S  # (N_valid, 3, 3)
    adj_gs_const_T = trans_R.T[None, ...]@R@adj_S.transpose(-1, -2)  # (N_valid, 3, 3)

    radius = compute_bound_radius(pose_lb, pose_ub, Z_lb, Z_ub, fx, fy, cam_const, gs_const, scale)  # (N_valid,)
    radius = COEFF_RADIUS*torch.sqrt(radius)  # (N_valid,)
    radius = torch.clamp(radius, min=MIN_RADIUS, max=MAX_RADIUS) # (N_valid,)

    # ============================================================
    # 6. Sort by Depth (Front-to-Back)
    # ============================================================
    Z_mid= (Z_lb + Z_ub) / 2  # (N_valid,)
    order = torch.argsort(Z_mid, descending=False)  # (N_valid,)
    
    Z = Z[order]
    XZ = XZ[order]
    YZ = YZ[order] 
    px = px[order]
    py = py[order]

    Z_lb = Z_lb[order]
    Z_ub = Z_ub[order]
    px_lb = px_lb[order]
    px_ub = px_ub[order]
    py_lb = py_lb[order]
    py_ub = py_ub[order]

    opacities = opacities[order]
    colors_rgb = colors_rgb[order]
    cam_const = cam_const[order]
    gs_const = gs_const[order]
    adj_gs_const_T = adj_gs_const_T[order]
    radius = radius[order]
    

    # cov2d_inv = cov2d_inv[order]
    # ============================================================
    # 7. Initialize Output and Regularize Covariance
    # ============================================================
    rgb = torch.zeros(height, width, 3, device=device, dtype=torch.float32) # (height, width, 3)

    rgb_lb = torch.zeros(height, width, 3, device=device, dtype=torch.float32) # (height, width, 3)
    rgb_ub = torch.zeros(height, width, 3, device=device, dtype=torch.float32) # (height, width, 3)

    tiles_x = (width + TILE - 1) // TILE
    tiles_y = (height + TILE - 1) // TILE
    total = tiles_y * tiles_x
    # ============================================================
    # 8. Tile-Based Rasterization
    # ============================================================
    for i in tqdm(range(total)):
    # for i in range(total):
        ty = i // tiles_x
        tx = i % tiles_x

        x0, y0 = tx * TILE, ty * TILE
        x1, y1 = min(x0 + TILE, width), min(y0 + TILE, height)

        tile_w = x1 - x0
        tile_h = y1 - y0

        # Create pixel grid for this tile
        xs = torch.arange(x0, x1, device=device, dtype=torch.float32)
        ys = torch.arange(y0, y1, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')  # (tile_h, tile_w)

        # AABB culling
        in_aabb = (
            ((px_ub+radius >= x0) & (px_lb-radius < x1)) &
            ((py_ub+radius >= y0) & (py_lb-radius < y1))
        )

        idx = torch.where(in_aabb)[0]
        if idx.numel() == 0:
            continue  # No valid Gaussians for this tile
        # print(f"Tile ({tx}, {ty}), x=[{x0}, {x1}), y=[{y0}, {y1}): {idx.numel()} Gaussians")

        # Filter Gaussians for this tile
        flt_Z = Z[idx]  # (num_valid,)
        flt_Z_lb = Z_lb[idx]  # (num_valid,)
        flt_Z_ub = Z_ub[idx]  # (num_valid,)
        flt_opacities = opacities[idx]  # (num_valid,)
        flt_colors = colors_rgb[idx]  # (num_valid, 3)
        flt_cam_const = cam_const[idx]  # (num_valid, 3)
        flt_gs_const = gs_const[idx]  # (num_valid, 3, 3)
        flt_adj_gs_const_T = adj_gs_const_T[idx]  # (num_valid, 3, 3)

        dx = (xx-cx)/fx # (tile_h, tile_w)
        dy = (yy-cy)/fy # (tile_h, tile_w)

        ### Compute Mahalanobis Distance
        mahal = compute_mahal_chunked(pose, flt_Z, dx, dy, 
                                      flt_cam_const, flt_gs_const, flt_adj_gs_const_T, scale, 
                                      chunk_size)  # (num_valid, tile_h, tile_w)
        # mahal_lb = torch.clamp(mahal-0.05, min=0.0)  # (num_valid, tile_h, tile_w)
        # mahal_ub = torch.clamp(mahal*3, max=MAX_RADIUS**2)  # (num_valid, tile_h, tile_w)

        # start_time = time()
        mahal_lb, mahal_ub = compute_bound_mahal_chunked(pose_lb, pose_ub, flt_Z_lb, flt_Z_ub, dx, dy, 
                                                        flt_cam_const, flt_gs_const, flt_adj_gs_const_T, scale, 
                                                        chunk_size) # (num_valid, tile_h, tile_w)
        # end_time = time()
        # print(f"Mahalanobis bounds computed in {end_time - start_time:.2f} seconds")
        
        
        w = torch.exp(-0.5 * mahal) * flt_opacities[:, None, None] # (num_valid, tile_h, tile_w)
        
        w_lb = torch.exp(-0.5 * mahal_ub) * flt_opacities[:, None, None] # (num_valid, tile_h, tile_w)
        w_ub = torch.exp(-0.5 * mahal_lb) * flt_opacities[:, None, None] # (num_valid, tile_h, tile_w)

        w_lb = torch.where(w_lb < 1e-3, torch.zeros_like(w_lb), w_lb)
        w_ub = torch.where(w_ub > 1-1e-3, torch.ones_like(w_ub), w_ub)
        w_ub = torch.where((w_ub - w_lb) < 1e-3, w_lb, w_ub)

        # print(f"Tile ({tx}, {ty}), x=[{x0}, {x1}), y=[{y0}, {y1}): {idx.numel()} Gaussians")
        # print(f"w_lb[sample_idx, 0, 0]: {w_lb[sample_idx, 0, 0].item()}, w[sample_idx, 0, 0]: {w[sample_idx, 0, 0].item()}, w_ub[sample_idx, 0, 0]: {w_ub[sample_idx, 0, 0].item()}\n")

        coeff = 0.25
        w_lb = (1-coeff)*w + coeff*w_lb
        w_ub = (1-coeff)*w + coeff*w_ub
       


        if linear:
            w_k_lb, w_k_ub, w_b_lb, w_b_ub = compute_bound_exp(w_lb, w_ub)  # (num_valid, tile_h, tile_w)



        rgb_patch = compute_alpha_blending(w, flt_colors)  # (num_valid, tile_h, tile_w, 3), (num_valid, tile_h, tile_w)
        # rgb_patch_lb = rgb_patch_ub = rgb_patch

        if not linear:
            rgb_patch_lb, rgb_patch_ub = compute_interval_bound_alpha_blending(w_lb, w_ub, flt_colors)  # (tile_h, tile_w, 3)
        else:
            rgb_patch_int_lb, rgb_patch_int_ub, rgb_patch_k_lb, rgb_patch_k_ub, rgb_patch_b_lb, rgb_patch_b_ub = \
            compute_linear_bound_alpha_blending(w_lb, w_ub, w_k_lb, w_k_ub, w_b_lb, w_b_ub, flt_colors)  
            # (tile_h, tile_w, 3), (num_valid, tile_h, tile_w), (tile_h, tile_w, 3)

            print(f"rgb_patch_int_lb[sample_idx, 0, 0]: {rgb_patch_int_lb[sample_idx, 0, 0].item()}, rgb_patch[sample_idx, 0, 0]: {rgb_patch[sample_idx, 0, 0].item()}, rgb_patch_int_ub[sample_idx, 0, 0]: {rgb_patch_int_ub[sample_idx, 0, 0].item()}\n")
            print(f"rgb_patch_k_lb[sample_idx, 0, 0, 0]: {rgb_patch_k_lb[sample_idx, 0, 0, 0].item()}, rgb_patch_k_ub[sample_idx, 0, 0, 0]: {rgb_patch_k_ub[sample_idx, 0, 0, 0].item()}\n")
            print(f"rgb_patch_b_lb[sample_idx, 0, 0]: {rgb_patch_b_lb[sample_idx, 0, 0].item()}, rgb_patch_b_ub[sample_idx, 0, 0]: {rgb_patch_b_ub[sample_idx, 0, 0].item()}\n")
            print(f"rgb_patch_k_lb.min(): {rgb_patch_k_lb.min().item()}, rgb_patch_k_ub.max(): {rgb_patch_k_ub.max().item()}\n")
            print(f"rgb_patch_b_lb.min(): {rgb_patch_b_lb.min().item()}, rgb_patch_b_ub.max(): {rgb_patch_b_ub.max().item()}\n")

            rgb_patch_lb, rgb_patch_ub = compute_bound_rgb(pose_lb, pose_ub, flt_Z_lb, flt_Z_ub, 
                                                           rgb_patch_k_lb, rgb_patch_k_ub, rgb_patch_b_lb, rgb_patch_b_ub, 
                                                           dx, dy, flt_cam_const, flt_gs_const, flt_adj_gs_const_T, scale)
            
        rgb[y0:y1, x0:x1] = rgb_patch
        rgb_lb[y0:y1, x0:x1] = rgb_patch_lb
        rgb_ub[y0:y1, x0:x1] = rgb_patch_ub 

    # ============================================================
    # 9. Finalize Output
    # ============================================================
    img = rgb[..., :3].clamp(0, 1)
    img_lb = rgb_lb[..., :3].clamp(0, 1)
    img_ub = rgb_ub[..., :3].clamp(0, 1)

    img = img.permute(2, 0, 1).to(device)
    img_lb = img_lb.permute(2, 0, 1).to(device)
    img_ub = img_ub.permute(2, 0, 1).to(device)
    return img, img_lb, img_ub