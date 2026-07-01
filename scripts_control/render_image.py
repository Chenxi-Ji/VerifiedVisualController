import os
import json
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

from dataclasses import dataclass
from scipy.spatial.transform import Rotation
from gsplat.rendering import rasterization


# =============================
# CONFIG
# =============================
@dataclass
class Config:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # gsplat path (cleaned drone-arena scene)
    gsplat_path = "nerfstudio/outputs/Gate_Long_hloc_seq/splatfacto/2026-06-11_015308_cleaned"
    checkpoint = "nerfstudio_models/step-000129999.ckpt"

# =============================
# GSPLAT LOADING（🔥关键）
# =============================
def load_gsplat_scene(cfg, use_world_frame=True):
    """Load gaussians + transforms. use_world_frame=False ignores
    world_frame.json so poses are interpreted in the raw data space
    (needed by the scene-exploration tools that work with transforms.json
    camera poses)."""
    ckpt_path = os.path.join(cfg.gsplat_path, cfg.checkpoint)
    res = torch.load(ckpt_path, weights_only=False, map_location=cfg.device)

    means = res['pipeline']['_model.gauss_params.means']
    quats = res['pipeline']['_model.gauss_params.quats']
    opacities = res['pipeline']['_model.gauss_params.opacities']
    scales = res['pipeline']['_model.gauss_params.scales']

    dc = res['pipeline']['_model.gauss_params.features_dc']
    rest = res['pipeline']['_model.gauss_params.features_rest']
    if rest.shape[1] == 0:
        # sh_degree-0 splatfacto stores sigmoid-space colors, not SH coeffs.
        # Convert to band-0 SH so rasterization(sh_degree=0) reproduces
        # sigmoid(dc) exactly: SH0 color = C0 * coeff + 0.5
        C0 = 0.28209479177387814
        colors = ((torch.sigmoid(dc) - 0.5) / C0)[:, None, :]
    else:
        colors = dc[:, None, :]

    # print(dc.shape, rest.shape, colors.shape)

    # 👉 只加载一次 transform（重要优化）
    with open(os.path.join(cfg.gsplat_path, "dataparser_transforms.json"), "r") as f:
        meta = json.load(f)

    transform = np.vstack([np.array(meta["transform"]), [0, 0, 0, 1]])
    scale = meta["scale"]

    # Optional gate-centered world frame (world_frame.json next to config.yml):
    # poses are then given in a clean frame — origin at the gate center,
    # +y through the gate, z down — and pitch=roll=0 is level flight.
    wf_path = os.path.join(cfg.gsplat_path, "world_frame.json")
    world_frame = use_world_frame and os.path.exists(wf_path)
    if world_frame:
        with open(wf_path) as f:
            W = np.array(json.load(f)["world_transform"])
        transform = transform @ W

    return means, quats, opacities, scales, colors, transform, scale, world_frame


# camera-axes alignment for the world-frame convention:
# yaw=0,pitch=0,roll=0 looks along +x with the image upright (camera up = -z)
CAM_AXES = np.array([[0, 0, -1],
                     [1, 0,  0],
                     [0, -1, 0]], dtype=np.float64)

# =============================
# viewmat转换（🔥关键） - 这个函数将相机位姿转换为gsplat的viewmat格式，使用了torch编译以加速计算
# =============================
def get_viewmat(optimized_camera_to_world, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
   """
   function that converts c2w to gsplat world2camera matrix, using compile for some speed
   """
   R = optimized_camera_to_world[:, :3, :3].to(device) # 3 x 3
   T = optimized_camera_to_world[:, :3, 3:4].to(device)  # 3 x 1
   # flip the z and y axes to align with gsplat conventions
   R = R * torch.tensor([[[1, -1, -1]]], device=R.device, dtype=R.dtype)
   # analytic matrix inverse to get world2camera matrix
   R_inv = R.transpose(1, 2)
   T_inv = -torch.bmm(R_inv, T)
   viewmat = torch.zeros(R.shape[0], 4, 4, device=R.device, dtype=R.dtype)
   viewmat[:, 3, 3] = 1.0  # homogenous
   viewmat[:, :3, :3] = R_inv
   viewmat[:, :3, 3:4] = T_inv
   return viewmat

# =============================
# RENDER（优化版）
# =============================
   
def render(pose, scene, width = 1024, height = 768,
            fx = 504.341405, fy = 503.319815,   # VOXL2 hires_small_color calib (reproj 0.37px)
            cx = 505.485234, cy = 367.606186,
            out_width = 256, out_height = 192,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    means, quats, opacities, scales, colors, transform, scale, world_frame = scene

    px, py, pz, yaw, pitch, roll = pose

    view = np.eye(4)
    R = Rotation.from_euler("ZYX", (yaw, pitch, roll)).as_matrix()
    view[:3, 3] = [px, py, pz]

    if world_frame:
        # clean gate-centered convention (world_frame.json composed in transform)
        view[:3, :3] = R @ CAM_AXES
    else:
        # legacy uturn-scene convention
        view[:3, :3] = R
        tmp = Rotation.from_euler('zyx', [-np.pi/2, np.pi/2, 0]).as_matrix()
        view[:3, :3] = view[:3, :3] @ tmp

        view[0:3,1:3] *= -1
        view = view[np.array([0,2,1,3]),:]
        view[2,:] *= -1

    view = transform @ view
    view[:3,3] *= scale

    view = torch.FloatTensor(view).unsqueeze(0).to(device)

    view = get_viewmat(view)

    Ks = torch.tensor([[fx,0,cx],[0,fy,cy],[0,0,1]], device=device).unsqueeze(0)

    rgb, alpha, _ = rasterization(
        means, quats,
        scales=torch.exp(scales),
        opacities=torch.sigmoid(opacities).squeeze(-1),
        colors=colors,
        viewmats=view,
        Ks=Ks,
        width=width,
        height=height,
        packed = False,
        near_plane=0.01,
        far_plane=1e10,
        render_mode="RGB+ED",
        sh_degree=0,
        sparse_grad=False,
        absgrad=True,
        rasterize_mode="classic",
        camera_model="fisheye",
    )

    img = rgb[0, ..., :3].clamp(0, 1)            # (height, width, 3) at full fisheye res
    # Replicate the on-drone preprocessing EXACTLY: cv2 INTER_LINEAR downscale
    # 1024x768 -> 256x192 (matches model_helper.cpp:302). The image is detached
    # before the controller, so this non-differentiable resize is safe.
    img = cv2.resize(img.detach().cpu().numpy(), (out_width, out_height),
                     interpolation=cv2.INTER_LINEAR)
    return torch.from_numpy(img).permute(2, 0, 1).to(device)
    
def render_batch(poses, scene,
                 width=1024, height=768,
                 fx=504.341405, fy=503.319815,   # VOXL2 hires_small_color calib (reproj 0.37px)
                 cx=505.485234, cy=367.606186,
                 out_width=256, out_height=192,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):

    means, quats, opacities, scales, colors, transform, scale, world_frame = scene

    B = poses.shape[0]

    # =============================
    # 1. 构建 batch view (numpy)
    # =============================
    views = []

    tmp = Rotation.from_euler('zyx', [-np.pi/2, np.pi/2, 0]).as_matrix()

    for i in range(B):
        px, py, pz, yaw, pitch, roll = poses[i]

        view = np.eye(4)

        R = Rotation.from_euler("ZYX", (yaw, pitch, roll)).as_matrix()
        view[:3, 3] = [px, py, pz]

        if world_frame:
            view[:3, :3] = R @ CAM_AXES
        else:
            view[:3, :3] = R

            # ---- 坐标变换 ----
            view[:3, :3] = view[:3, :3] @ tmp

            view[0:3, 1:3] *= -1
            view = view[[0, 2, 1, 3], :]
            view[2, :] *= -1

        view = transform @ view
        view[:3, 3] *= scale

        views.append(view)

    views = np.stack(views, axis=0)  # (B,4,4)

    # =============================
    # 2. 转 torch
    # =============================
    view = torch.from_numpy(views).float().to(device)

    viewmats = get_viewmat(view)

    # =============================
    # 3. Ks batch
    # =============================
    Ks = torch.tensor([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], device=device, dtype=torch.float32)

    Ks = Ks.unsqueeze(0).repeat(B, 1, 1)

    # =============================
    # 4. 确保 gsplat 输入合法（关键优化）
    # =============================
    means_ = means.contiguous()
    quats_ = quats.contiguous()
    scales_ = torch.exp(scales).contiguous()
    opacities_ = torch.sigmoid(opacities).squeeze(-1).contiguous()
    colors_ = colors.contiguous()

    # =============================
    # 5. rasterization (batch!)
    # =============================
    rgb, alpha, _ = rasterization(
        means_,
        quats_,
        scales=scales_,
        opacities=opacities_,
        colors=colors_,
        viewmats=viewmats,
        Ks=Ks,
        width=width,
        height=height,
        packed=False,
        near_plane=0.01,
        far_plane=1e10,
        render_mode="RGB+ED",
        sh_degree=0,
        sparse_grad=False,
        absgrad=True,
        rasterize_mode="classic",
        camera_model="fisheye",
    )

    # =============================
    # 6. 输出 (B,3,H,W)
    # =============================
    imgs = rgb[..., :3].clamp(0, 1)  # (B,H,W,3) at full fisheye res
    # Replicate the on-drone preprocessing EXACTLY: cv2 INTER_LINEAR per frame,
    # 1024x768 -> 256x192 (matches model_helper.cpp:302).
    imgs_np = imgs.detach().cpu().numpy()
    imgs_np = np.stack([cv2.resize(im, (out_width, out_height),
                                   interpolation=cv2.INTER_LINEAR) for im in imgs_np], axis=0)
    imgs = torch.from_numpy(imgs_np).permute(0, 3, 1, 2).to(device)  # (B,3,out_h,out_w)

    return imgs

if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    test_batch = False

    cfg = Config()
    device = cfg.device
    scene = load_gsplat_scene(cfg)

    if not test_batch:

        # gate-centered frame: gate at origin, +y through the gate, z down (+y deploy side)
        random_pose = np.array([0.0, 1.5, 0.0, -np.pi/2, 0.0, 0.0])

        img = render(random_pose, scene, device=cfg.device)

        # 显示图像
        plt.imshow(img.permute(1, 2, 0).cpu().numpy())
        plt.axis('off')
        plt.savefig("figures/example_image.png")
        plt.show()

    else:
        

        poses = np.array([
            [0.0, 2.0, 0.0, -np.pi/2, 0.0, 0.0]
        ])

        imgs = render_batch(poses, scene, device=device)
        print("imgs shape:", imgs.shape)  # (B,3,H,W)

        # show first
        img = imgs[0].permute(1,2,0).cpu().numpy()

        plt.imshow(img)
        plt.axis('off')
        plt.savefig("figures/example_image.png")
        plt.show()

    
    