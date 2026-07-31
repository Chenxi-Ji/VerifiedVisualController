from time import time
from tqdm import tqdm
import torch
from itertools import product
from utils_rational_quad import rational_quad_bound
from coordinate_transform import (
    pose_bound_to_camera_bounds,
    pose_to_camera_xyz,
    pose_to_render_coords,
    pose_to_render_rotation,
    transform_to_render_constants,
)

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
def build_angle_list(angle_lb, angle_ub, k=6):
    if angle_ub - angle_lb < 1e-3:
        return torch.tensor([angle_lb], device=angle_lb.device, dtype=angle_lb.dtype) 
    else:
        angle_list = torch.linspace(angle_lb, angle_ub, steps=k, device=angle_lb.device, dtype=angle_lb.dtype)
        return angle_list

@torch.no_grad()
def build_yaw_list(yaw_lb, yaw_ub, k=6):
    return build_angle_list(yaw_lb, yaw_ub, k)

@torch.no_grad()
def build_pitch_list(pitch_lb, pitch_ub, k=6):
    return build_angle_list(pitch_lb, pitch_ub, k)

@torch.no_grad()
def build_roll_list(roll_lb, roll_ub, k=6):
    return build_angle_list(roll_lb, roll_ub, k)

@torch.no_grad()
def compute_bound_Z(pose_lb, pose_ub, const, scale, world_frame=False):
    device = pose_lb.device
    dtype = pose_lb.dtype
    N = const.shape[0]

    pos_x_lb, pos_y_lb, pos_z_lb, yaw_lb, pitch_lb, roll_lb = pose_lb
    pos_x_ub, pos_y_ub, pos_z_ub, yaw_ub, pitch_ub, roll_ub = pose_ub

    yaw_list = build_yaw_list(yaw_lb, yaw_ub)
    pitch_list = build_pitch_list(pitch_lb, pitch_ub)
    roll_list = build_roll_list(roll_lb, roll_ub)

    Z_min = torch.full((N,), float("inf"), device=device, dtype=dtype) # (N, )
    Z_max = torch.full((N,), float("-inf"), device=device, dtype=dtype) # (N, )

    for pos_x, pos_y, pos_z, yaw, pitch, roll in product(
        [pos_x_lb, pos_x_ub],
        [pos_y_lb, pos_y_ub],
        [pos_z_lb, pos_z_ub],
        yaw_list,
        pitch_list,
        roll_list,
    ):
        # Compute Z for this combination
        pose = torch.stack([pos_x, pos_y, pos_z, yaw, pitch, roll])
        _, _, Z = pose_to_camera_xyz(pose, const, scale, world_frame)  # (N, )
        # update bounds
        Z_min = torch.minimum(Z_min, Z)
        Z_max = torch.maximum(Z_max, Z)

    # final result
    return Z_min, Z_max

@torch.no_grad()
def compute_bound_XZ(pose_lb, pose_ub, const, scale, world_frame=False):
    device = pose_lb.device
    dtype = pose_lb.dtype
    N = const.shape[0]

    pos_x_lb, pos_y_lb, pos_z_lb, yaw_lb, pitch_lb, roll_lb = pose_lb
    pos_x_ub, pos_y_ub, pos_z_ub, yaw_ub, pitch_ub, roll_ub = pose_ub

    yaw_list = build_yaw_list(yaw_lb, yaw_ub)
    pitch_list = build_pitch_list(pitch_lb, pitch_ub)
    roll_list = build_roll_list(roll_lb, roll_ub)

    XZ_min = torch.full((N,), float("inf"), device=device, dtype=dtype) # (N, )
    XZ_max = torch.full((N,), float("-inf"), device=device, dtype=dtype) # (N, )

    for pos_x, pos_y, pos_z, yaw, pitch, roll in product(
        [pos_x_lb, pos_x_ub],
        [pos_y_lb, pos_y_ub],
        [pos_z_lb, pos_z_ub],
        yaw_list,
        pitch_list,
        roll_list,
    ):
        # Compute Z for this combination
        pose = torch.stack([pos_x, pos_y, pos_z, yaw, pitch, roll])
        X, _, Z = pose_to_camera_xyz(pose, const, scale, world_frame)  # (N, )
        XZ = X/Z  # (N, )

        # update bounds
        XZ_min = torch.minimum(XZ_min, XZ)
        XZ_max = torch.maximum(XZ_max, XZ)

    # final result
    return XZ_min, XZ_max

@torch.no_grad()
def compute_bound_YZ(pose_lb, pose_ub, const, scale, world_frame=False):
    device = pose_lb.device
    dtype = pose_lb.dtype
    N = const.shape[0]

    pos_x_lb, pos_y_lb, pos_z_lb, yaw_lb, pitch_lb, roll_lb = pose_lb
    pos_x_ub, pos_y_ub, pos_z_ub, yaw_ub, pitch_ub, roll_ub = pose_ub

    yaw_list = build_yaw_list(yaw_lb, yaw_ub)
    pitch_list = build_pitch_list(pitch_lb, pitch_ub)
    roll_list = build_roll_list(roll_lb, roll_ub)

    YZ_min = torch.full((N,), float("inf"), device=device, dtype=dtype) # (N, )
    YZ_max = torch.full((N,), float("-inf"), device=device, dtype=dtype) # (N, )

    for pos_x, pos_y, pos_z, yaw, pitch, roll in product(
        [pos_x_lb, pos_x_ub],
        [pos_y_lb, pos_y_ub],
        [pos_z_lb, pos_z_ub],
        yaw_list,
        pitch_list,
        roll_list,
    ):
        # Compute Z for this combination
        pose = torch.stack([pos_x, pos_y, pos_z, yaw, pitch, roll])
        _, Y, Z = pose_to_camera_xyz(pose, const, scale, world_frame)  # (N, )
        YZ =  Y/Z  # (N, )

        # update bounds
        YZ_min = torch.minimum(YZ_min, YZ)
        YZ_max = torch.maximum(YZ_max, YZ)

    # final result
    return YZ_min, YZ_max

@torch.no_grad()
def compute_radius(pose, Z, fx, fy, cam_const, gs_const, scale, world_frame=False):
    device = Z.device
    dtype = Z.dtype

    F = torch.tensor([[fx, 0.0, 0.0], [0.0, fy, 0.0]], device=device, dtype=dtype)  # (2, 3)
    E = torch.tensor([
        [0.0, 1.0, 0.0],
        [-1.0,0.0, 0.0],
        [0.0, 0.0, 0.0]
    ]).to(device=device, dtype=dtype)  # (3, 3)
    FE = F@E  # (2, 3)

    x = pose_to_render_coords(pose, world_frame)
    X = -x[None, :]*scale + cam_const   # (N, 3)
    Xx = compute_skew_matrix(X)  # (N, 3, 3)

    Rw = pose_to_render_rotation(pose, world_frame) # (3, 3)

    RXx = Rw[None, ...]@Xx  # (N, 3, 3)

    semi_cov2d = FE[None, ...]@RXx@gs_const  # (N, 2, 3)
    max_cov2d_eig = torch.linalg.matrix_norm(semi_cov2d, ord=2) #(N,)
    radius = max_cov2d_eig/Z**2  # (N, )

    return radius

@torch.no_grad()
def compute_bound_radius(pose_lb, pose_ub, Z_lb, Z_ub, fx, fy, cam_const, gs_const, scale, world_frame=False):
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
    pitch_list = build_pitch_list(pitch_lb, pitch_ub)
    roll_list = build_roll_list(roll_lb, roll_ub)
    max_radius = torch.zeros(N, device=device, dtype=dtype)

    for pos_x, pos_y, pos_z, yaw, pitch, roll in product(
        [pos_x_lb, pos_x_ub],
        [pos_y_lb, pos_y_ub],
        [pos_z_lb, pos_z_ub],
        yaw_list,
        pitch_list,
        roll_list,
    ):
        pose = torch.stack([pos_x, pos_y, pos_z, yaw, pitch, roll])
        x = pose_to_render_coords(pose, world_frame)
        X = -x[None, :]*scale + cam_const   # (N, 3)
        Xx = compute_skew_matrix(X)  # (N, 3, 3)

        Rw = pose_to_render_rotation(pose, world_frame) # (3, 3)

        RXx = Rw[None, ...]@Xx  # (N, 3, 3)

        semi_cov2d = FE[None, ...]@RXx@gs_const  # (N, 2, 3)
        max_cov2d_eig = torch.linalg.matrix_norm(semi_cov2d, ord=2) #(N,)
        # Sampling-based radius is tighter but can miss Gaussians near tile boundaries.
        # _, _, Z_sample = pose_to_camera_xyz(pose, cam_const, scale, world_frame)  # (N, )
        # radius = max_cov2d_eig/Z_sample.clamp_min(1e-6)**2  # (N, )

        # Safe but looser radius upper bound:
        radius = max_cov2d_eig/Z_lb.clamp_min(1e-6)**2  # (N, )

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
def compute_mahal(pose, Z, dx, dy, cam_const, gs_const, adj_gs_const_T, scale, world_frame=False):
    device = Z.device
    dtype = Z.dtype
    N = Z.shape[0]
    H, W = dx.shape

    pos_x, pos_y, pos_z, yaw, pitch, roll = pose

    x = pose_to_render_coords(pose, world_frame) #(3, )
    X = -x[None, :]*scale + cam_const   # (N, 3)

    ones = torch.ones((H,W), device=device, dtype=dtype)
    d = torch.stack([dx, dy, ones], dim=-1)  # (H,W,3)
    #D = compute_skew_matrix(d)  # (H,W,3,3)

    Rw = pose_to_render_rotation(pose, world_frame) # (3, 3)

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
def compute_bound_mahal(pose_lb, pose_ub, Z_lb, Z_ub, dx, dy, cam_const, gs_const, adj_gs_const_T, scale, world_frame=False):
    pos_x_lb, pos_y_lb, pos_z_lb, yaw_lb, pitch_lb, roll_lb = pose_lb
    pos_x_ub, pos_y_ub, pos_z_ub, yaw_ub, pitch_ub, roll_ub = pose_ub

    device = Z_lb.device
    dtype = Z_lb.dtype
    N = Z_lb.shape[0]
    H, W = dx.shape

    Z_ratio = Z_ub/Z_lb # (N, )
    X_lb, X_ub = pose_bound_to_camera_bounds(pose_lb, pose_ub, cam_const, scale, world_frame)

    ones = torch.ones((H,W), device=device, dtype=dtype)
    d = torch.stack([dx, dy, ones], dim=-1)  # (H,W,3)
    # D = compute_skew_matrix(d)  # (H,W,3,3)

    yaw_list = build_yaw_list(yaw_lb, yaw_ub)
    pitch_list = build_pitch_list(pitch_lb, pitch_ub)
    roll_list = build_roll_list(roll_lb, roll_ub)

    mahal_lb = torch.full((N,H,W), float("inf"), device=device, dtype=torch.float64) # (N, H, W)
    mahal_ub = torch.full((N,H,W), float("-inf"), device=device, dtype=torch.float64) # (N, H, W)

    for yaw, pitch, roll in product(yaw_list, pitch_list, roll_list):
        pose_angle = torch.stack([pos_x_lb, pos_y_lb, pos_z_lb, yaw, pitch, roll])
        Rw = pose_to_render_rotation(pose_angle, world_frame) # (3, 3)

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
    world_frame=False,
):
    N = Z.shape[0]
    device = pose.device

    H, W = dx.shape[-2], dx.shape[-1]

    mahal = torch.empty((N, H, W), device=device, dtype=pose.dtype)
    for i in range(0, N, chunk_size):
        j = min(i + chunk_size, N)

        mahal[i:j] = compute_mahal(
            pose,
            Z[i:j],
            dx,
            dy,
            cam_const[i:j],
            gs_const[i:j],
            adj_gs_const_T[i:j],
            scale,
            world_frame,
        )

    return mahal
    
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
    world_frame=False,
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
            world_frame,
        )

    return mahal_lb, mahal_ub





@torch.no_grad()
def compute_alpha_blending(w, colors, rgb=None):
    N, H, W = w.shape
    device, dtype = w.device, w.dtype

    if rgb is None:
        rgb = torch.zeros((H, W, 3), device=device, dtype=dtype)

    for i in range(N-1, -1, -1):
        c = colors[i].view(1, 1, 3) # (1, 1, 3)
        d = c - rgb # (H, W, 3)
        rgb = rgb + d * w[i][..., None]

    # print(f"rgb.shape={rgb.shape}, rgb.min={rgb.min():.6f}, rgb.max={rgb.max():.6f}, rgb.mean={rgb.mean():.6f}")
    return rgb

@torch.no_grad()
def compute_interval_bound_alpha_blending(w_lb, w_ub, colors, rgb_lb = None, rgb_ub = None):
    N, H, W = w_lb.shape
    device, dtype = w_lb.device, w_lb.dtype

    if rgb_lb is None:
        rgb_lb = torch.zeros((H, W, 3), device=device, dtype=dtype)
    if rgb_ub is None:
        rgb_ub = torch.zeros((H, W, 3), device=device, dtype=dtype)

    for i in range(N-1, -1, -1):
        c = colors[i].view(1, 1, 3) # (1, 1, 3)

        d_lb = c - rgb_lb # (H, W, 3)
        d_ub = c - rgb_ub

        w_l = w_lb[i][..., None] # (H, W, 1)
        w_u = w_ub[i][..., None]

        m_lb = (d_lb >= 0) # (H, W, 3) -> (H, W, 1) via broadcasting
        w_sel_lb = torch.where(m_lb, w_l, w_u) # (H, W, 1)
        rgb_lb = rgb_lb + d_lb * w_sel_lb # (H, W, 3)

        m_ub = (d_ub >= 0)
        w_sel_ub = torch.where(m_ub, w_u, w_l)
        rgb_ub = rgb_ub + d_ub * w_sel_ub

    return rgb_lb, rgb_ub

@torch.no_grad()
def compute_alpha_blending_chunked(
    w,
    colors,
    rgb = None,
    chunk_size=1024,
):
    N, H, W = w.shape
    device, dtype = w.device, w.dtype

    if rgb is None:
        rgb = torch.zeros((H, W, 3), device=device, dtype=dtype)

    for j in range(N, 0, -chunk_size):
        i = max(j - chunk_size, 0)
        rgb = compute_alpha_blending(
            w[i:j, ...],
            colors[i:j, :],
            rgb
        )

    return rgb

@torch.no_grad()
def compute_interval_bound_alpha_blending_chunked(
    w_lb, w_ub,
    colors,
    rgb_lb = None,
    rgb_ub = None,
    chunk_size=1024,
):
    N, H, W = w_lb.shape
    device, dtype = w_lb.device, w_lb.dtype

    if rgb_lb is None:
        rgb_lb = torch.zeros((H, W, 3), device=device, dtype=dtype)
    if rgb_ub is None:
        rgb_ub = torch.zeros((H, W, 3), device=device, dtype=dtype)

    for j in range(N, 0, -chunk_size):
        i = max(j - chunk_size, 0)
        rgb_lb, rgb_ub = compute_interval_bound_alpha_blending(
            w_lb[i:j, ...],
            w_ub[i:j, ...],
            colors[i:j, :],
            rgb_lb,
            rgb_ub
        )

    return rgb_lb, rgb_ub

def render_bound(pose_lb, pose_ub, scene,
            camera_params = (300, 200, 113.258171, 113.347599, 158.868074, 98.837772),
            near_plane=0.01, far_plane=1e10,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            dtype=torch.float32,
            chunk_size=1024*2,
            tile=32,
            debug = False
            ):
    
    width, height, fx, fy, cx, cy = camera_params
    if len(scene) == 8:
        means, quats, opacities, scales, colors, transform, scale, world_frame = scene
    else:
        means, quats, opacities, scales, colors, transform, scale = scene
        world_frame = False

    N = means.shape[0]

    ALPHA_THRESHOLD = 1e-3
    MIN_RADIUS = 0.1
    MAX_RADIUS = max(width, height)/2
    COEFF_RADIUS = 3.0
    sample_idx = 30

    scales=torch.exp(scales)
    opacities=torch.sigmoid(opacities).squeeze(-1)
    transform, trans_R, trans_T, cam_const = transform_to_render_constants(means, transform, scale, device, dtype)

    pose_lb = torch.as_tensor(pose_lb, dtype=dtype, device=device)
    pose_ub = torch.as_tensor(pose_ub, dtype=dtype, device=device)
    pose = (pose_lb + pose_ub) / 2

    pos_x, pos_y, pos_z, yaw, pitch, roll = pose

    # ============================================================
    # 0. Compute View Matrix
    # ============================================================
    Rw = pose_to_render_rotation(pose, world_frame)
    Tw = pose_to_render_coords(pose, world_frame).unsqueeze(-1)

    # view_R = Rw@trans_R.T
    # view_T = -view_R@(trans_R@Tw + trans_T)*scale
    # viewmat = torch.eye(4, device=device, dtype=dtype)
    # viewmat[:3, :3] = view_R
    # viewmat[:3, 3:4] = view_T
    
    # ============================================================
    # 1. Transform to Camera Space (Vectorized)
    # ============================================================
    means_cam = Rw[None, ...]@(cam_const-Tw[None, ...]*scale)  # (N, 3, 1)

    cam_const = cam_const.squeeze(-1)  # (N, 3)
    means_cam = means_cam.squeeze(-1)  # (N, 3)

    # ============================================================
    # 3. Camera Projection (Vectorized)
    # ============================================================
    X = means_cam[:, 0]  # (N,)
    Y = means_cam[:, 1]  # (N,)
    Z = means_cam[:, 2]  # (N,)
    XZ = X/Z  # (N,)
    YZ = Y/Z  # (N,)
    px = fx * XZ + cx  # (N,)
    py = fy * YZ + cy  # (N,)

    if debug:
        Z_lb = Z_ub = Z
        px_lb = px_ub = px
        py_lb = py_ub = py

        # Filter valid Gaussians based on Z
        valid_depth = Z > near_plane
    else:
        Z_lb, Z_ub = compute_bound_Z(pose_lb, pose_ub, cam_const, scale, world_frame)
        assert (Z_lb <= Z_ub).all(), "Z_lb should be less than or equal to Z_ub for all Gaussians"

        XZ_lb, XZ_ub = compute_bound_XZ(pose_lb, pose_ub, cam_const, scale, world_frame)
        YZ_lb, YZ_ub = compute_bound_YZ(pose_lb, pose_ub, cam_const, scale, world_frame)
        px_lb, px_ub = fx * XZ_lb + cx, fx * XZ_ub + cx
        py_lb, py_ub = fy * YZ_lb + cy, fy * YZ_ub + cy

        assert (px_lb <= px_ub).all(), "px_lb should be less than or equal to px_ub for all Gaussians"
        assert (py_lb <= py_ub).all(), "py_lb should be less than or equal to py_ub for all Gaussians"

        # Filter valid Gaussians based on Z_lb
        valid_depth = Z_lb > near_plane
    valid_idx = torch.where(valid_depth)[0]
    N_valid = valid_idx.numel()
    print(f"Total Gaussians: {N}, Valid Gaussians: {N_valid}")

    if N_valid == 0:
        print("⚠️  No valid Gaussians!")
        rgb = torch.zeros(height, width, 3, device=device, dtype=dtype)
        return rgb, rgb, rgb

    quats = quats[valid_idx]
    scales = scales[valid_idx]
    opacities = opacities[valid_idx]
    colors = colors[valid_idx]
    means_cam = means_cam[valid_idx]  # (N_valid, 3)

    X = X[valid_idx]
    Y = Y[valid_idx]
    Z = Z[valid_idx]
    XZ = XZ[valid_idx]
    YZ = YZ[valid_idx]
    px = px[valid_idx]
    py = py[valid_idx]
    Z_lb = Z_lb[valid_idx]
    Z_ub = Z_ub[valid_idx]
    px_lb = px_lb[valid_idx]
    px_ub = px_ub[valid_idx]
    py_lb = py_lb[valid_idx]
    py_ub = py_ub[valid_idx]
    cam_const = cam_const[valid_idx]

    # ============================================================
    # 2. Compute View Directions and SH Colors
    # ============================================================
    C0 = 0.28209479177387814
    colors_rgb = torch.clamp(C0*colors[:, 0, :] + 0.5, 0.0, 1.0)  # (N_valid, 3)

    # ============================================================
    # 4. Compute Radius for each 2D Gaussian (Vectorized)
    # ============================================================
    S = torch.diag_embed(scales)  # (N_valid, 3, 3)
    adj_S = compute_adj(S)  # (N_valid, 3, 3)
    R = qvec2rotmat_batched(quats)  # (N_valid, 3, 3)
    gs_const = trans_R.T[None, ...]@R@S  # (N_valid, 3, 3)
    adj_gs_const_T = trans_R.T[None, ...]@R@adj_S.transpose(-1, -2)  # (N_valid, 3, 3)

    if debug:
        radius = compute_radius(pose, Z, fx, fy, cam_const, gs_const, scale, world_frame)  # (N_valid,)
    else:
        radius = compute_bound_radius(pose_lb, pose_ub, Z_lb, Z_ub, fx, fy, cam_const, gs_const, scale, world_frame)  # (N_valid,)
    radius = COEFF_RADIUS*radius  # (N_valid,)
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

    tiles_x = (width + tile - 1) // tile
    tiles_y = (height + tile - 1) // tile
    total = tiles_y * tiles_x
    # ============================================================
    # 8. Tile-Based Rasterization
    # ============================================================
    for i in tqdm(range(total)):
    # for i in range(total):
        ty = i // tiles_x
        tx = i % tiles_x

        x0, y0 = tx * tile, ty * tile
        x1, y1 = min(x0 + tile, width), min(y0 + tile, height)

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
                                      chunk_size, world_frame)  # (num_valid, tile_h, tile_w)
        # mahal_lb = torch.clamp(mahal-0.05, min=0.0)  # (num_valid, tile_h, tile_w)
        # mahal_ub = torch.clamp(mahal*3, max=MAX_RADIUS**2)  # (num_valid, tile_h, tile_w)

        w = torch.exp(-0.5 * mahal) * flt_opacities[:, None, None] # (num_valid, tile_h, tile_w)


        rgb_patch = compute_alpha_blending_chunked(w, flt_colors)  # (num_valid, tile_h, tile_w, 3), (num_valid, tile_h, tile_w)
        if debug:
            rgb_patch_lb = rgb_patch_ub = rgb_patch
        else:
            mahal_lb, mahal_ub = compute_bound_mahal_chunked(pose_lb, pose_ub, flt_Z_lb, flt_Z_ub, dx, dy, 
                                                            flt_cam_const, flt_gs_const, flt_adj_gs_const_T, scale, 
                                                            chunk_size, world_frame) # (num_valid, tile_h, tile_w)
            
            w_lb = torch.exp(-0.5 * mahal_ub) * flt_opacities[:, None, None] # (num_valid, tile_h, tile_w)
            w_ub = torch.exp(-0.5 * mahal_lb) * flt_opacities[:, None, None] # (num_valid, tile_h, tile_w)

            w_lb = torch.where(w_lb < 1e-3, torch.zeros_like(w_lb), w_lb)
            w_ub = torch.where(w_ub > 1-1e-3, torch.ones_like(w_ub), w_ub)
            w_ub = torch.where((w_ub - w_lb) < 1e-3, w_lb, w_ub)

            rgb_patch_lb, rgb_patch_ub = compute_interval_bound_alpha_blending_chunked(w_lb, w_ub, flt_colors)  # (tile_h, tile_w, 3)
    
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
