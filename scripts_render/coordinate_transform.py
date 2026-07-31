import torch


def euler_zyx_rotation(yaw, pitch, roll):
    sin_yaw, cos_yaw = torch.sin(yaw), torch.cos(yaw)
    sin_pitch, cos_pitch = torch.sin(pitch), torch.cos(pitch)
    sin_roll, cos_roll = torch.sin(roll), torch.cos(roll)

    return torch.stack([
        torch.stack([
            cos_yaw * cos_pitch,
            cos_yaw * sin_pitch * sin_roll - sin_yaw * cos_roll,
            cos_yaw * sin_pitch * cos_roll + sin_yaw * sin_roll,
        ]),
        torch.stack([
            sin_yaw * cos_pitch,
            sin_yaw * sin_pitch * sin_roll + cos_yaw * cos_roll,
            sin_yaw * sin_pitch * cos_roll - cos_yaw * sin_roll,
        ]),
        torch.stack([
            -sin_pitch,
            cos_pitch * sin_roll,
            cos_pitch * cos_roll,
        ]),
    ])


def legacy_render_rotation(yaw, pitch, roll):
    sin_yaw, cos_yaw = torch.sin(yaw), torch.cos(yaw)
    sin_pitch, cos_pitch = torch.sin(pitch), torch.cos(pitch)
    sin_roll, cos_roll = torch.sin(roll), torch.cos(roll)

    return torch.stack([
        torch.stack([
            cos_roll * sin_yaw - cos_yaw * sin_pitch * sin_roll,
            -cos_pitch * sin_roll,
            cos_roll * cos_yaw + sin_pitch * sin_roll * sin_yaw,
        ]),
        torch.stack([
            -cos_roll * cos_yaw * sin_pitch - sin_roll * sin_yaw,
            -cos_pitch * cos_roll,
            cos_roll * sin_pitch * sin_yaw - cos_yaw * sin_roll,
        ]),
        torch.stack([
            cos_pitch * cos_yaw,
            -sin_pitch,
            -cos_pitch * sin_yaw,
        ]),
    ])


def world_frame_render_rotation(yaw, pitch, roll):
    device = yaw.device
    dtype = yaw.dtype
    cam_axes = torch.tensor([
        [0.0, 0.0, -1.0],
        [1.0, 0.0,  0.0],
        [0.0, -1.0, 0.0],
    ], device=device, dtype=dtype)
    axis_flip = torch.tensor([1.0, -1.0, -1.0], device=device, dtype=dtype)
    local_R = euler_zyx_rotation(yaw, pitch, roll) @ cam_axes
    return torch.diag(axis_flip) @ local_R.T


def pose_to_render_rotation(pose, world_frame=False):
    pos_x, pos_y, pos_z, yaw, pitch, roll = pose
    if world_frame:
        return world_frame_render_rotation(yaw, pitch, roll)
    return legacy_render_rotation(yaw, pitch, roll)


def camera_to_render_coords(x, y, z, world_frame=False):
    if world_frame:
        return torch.stack([x, y, z], dim=-1)  # (3, )
    return torch.stack([x, z, -y], dim=-1)  # (3, )


def camera_bound_to_render_coords(x_lb, y_lb, z_lb, x_ub, y_ub, z_ub, world_frame=False):
    if world_frame:
        X_lb = torch.stack([x_lb, y_lb, z_lb], dim=-1)  # (3, )
        X_ub = torch.stack([x_ub, y_ub, z_ub], dim=-1)  # (3, )
    else:
        X_lb = torch.stack([x_lb, z_lb, -y_ub], dim=-1)  # (3, )
        X_ub = torch.stack([x_ub, z_ub, -y_lb], dim=-1)  # (3, )
    return X_lb, X_ub


def pose_to_render_coords(pose, world_frame=False):
    pos_x, pos_y, pos_z, yaw, pitch, roll = pose
    return camera_to_render_coords(pos_x, pos_y, pos_z, world_frame)


def pose_bound_to_render_coords(pose_lb, pose_ub, world_frame=False):
    pos_x_lb, pos_y_lb, pos_z_lb, yaw_lb, pitch_lb, roll_lb = pose_lb
    pos_x_ub, pos_y_ub, pos_z_ub, yaw_ub, pitch_ub, roll_ub = pose_ub
    return camera_bound_to_render_coords(
        pos_x_lb, pos_y_lb, pos_z_lb,
        pos_x_ub, pos_y_ub, pos_z_ub,
        world_frame,
    )


def transform_to_render_constants(means, transform, scale, device, dtype):
    transform = torch.as_tensor(transform, dtype=dtype, device=device)
    trans_R = transform[:3, :3]
    trans_T = transform[:3, 3:4]
    cam_const = trans_R.T[None, ...]@(means[..., None]-trans_T[None, ...]*scale)  # (N, 3, 1)
    return transform, trans_R, trans_T, cam_const


def pose_to_camera_points(pose, const, scale, world_frame=False):
    Rw = pose_to_render_rotation(pose, world_frame)
    Tw = pose_to_render_coords(pose, world_frame).unsqueeze(-1)
    points_cam = Rw[None, ...]@(const[..., None]-Tw[None, ...]*scale)  # (N, 3, 1)
    return points_cam.squeeze(-1)


def pose_to_camera_xyz(pose, const, scale, world_frame=False):
    points_cam = pose_to_camera_points(pose, const, scale, world_frame)
    X = points_cam[:, 0]  # (N, )
    Y = points_cam[:, 1]  # (N, )
    Z = points_cam[:, 2]  # (N, )
    return X, Y, Z


def pose_bound_to_camera_bounds(pose_lb, pose_ub, cam_const, scale, world_frame=False):
    x_lb, x_ub = pose_bound_to_render_coords(pose_lb, pose_ub, world_frame)

    X_lb = -x_ub[None, :]*scale + cam_const   # (N, 3)
    X_ub = -x_lb[None, :]*scale + cam_const   # (N, 3)

    return X_lb, X_ub


def pose_to_viewmat(pose, transform, scale, world_frame, device, dtype):
    pose = torch.as_tensor(pose, dtype=dtype, device=device)
    transform = torch.as_tensor(transform, dtype=dtype, device=device)
    trans_R = transform[:3, :3]
    trans_T = transform[:3, 3:4]

    Rw = pose_to_render_rotation(pose, world_frame)
    Tw = pose_to_render_coords(pose, world_frame).unsqueeze(-1)

    view_R = Rw @ trans_R.T
    view_T = -(view_R @ (trans_R @ Tw + trans_T))*scale
    viewmat = torch.eye(4, device=device, dtype=dtype)
    viewmat[:3, :3] = view_R
    viewmat[:3, 3:4] = view_T
    return viewmat.unsqueeze(0)
