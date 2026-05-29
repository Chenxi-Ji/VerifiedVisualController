import os
import json
import torch
import numpy as np
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

    # gsplat path
    gsplat_path = "nerfstudio/outputs/uturn/splatfacto/2025-05-09_151825"
    checkpoint = "nerfstudio_models/step-000040005.ckpt"

# =============================
# GSPLAT LOADING（🔥关键）
# =============================
def load_gsplat_scene(cfg):
    ckpt_path = os.path.join(cfg.gsplat_path, cfg.checkpoint)
    res = torch.load(ckpt_path)

    means = res['pipeline']['_model.gauss_params.means']
    quats = res['pipeline']['_model.gauss_params.quats']
    opacities = res['pipeline']['_model.gauss_params.opacities']
    scales = res['pipeline']['_model.gauss_params.scales']

    dc = res['pipeline']['_model.gauss_params.features_dc']
    rest = res['pipeline']['_model.gauss_params.features_rest']
    colors = torch.cat((dc[:, None, :], rest), dim=1)
    colors = dc[:, None, :]

    # print(dc.shape, rest.shape, colors.shape)

    # 👉 只加载一次 transform（重要优化）
    with open(os.path.join(cfg.gsplat_path, "dataparser_transforms.json"), "r") as f:
        meta = json.load(f)

    transform = np.array(meta["transform"])
    scale = meta["scale"]

    return means, quats, opacities, scales, colors, transform, scale

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
# def render(pose, scene, width = 320, height = 240,
#             fx = 273.42082456, fy = 273.789787305, 
#             cx = 174.591581635, cy = 107.77243002, 
#             device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    
def render(pose, scene, width = 300, height = 200,
            fx = 113.258171, fy = 113.347599, 
            cx = 158.868074, cy = 98.837772, 
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    means, quats, opacities, scales, colors, transform, scale = scene

    px, py, pz, yaw, pitch, roll = pose

    view = np.eye(4)
    R = Rotation.from_euler("ZYX", (yaw, pitch, roll)).as_matrix()
    view[:3, :3] = R
    view[:3, 3] = [px, py, pz]

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
    )

    img = rgb[0, ..., :3].clamp(0, 1)
    return img.permute(2, 0, 1).to(device)

# def render_batch(poses, scene,
#                  width=320, height=240,
#                  fx=273.42082456, fy=273.789787305,
#                  cx=174.591581635, cy=107.77243002,
#                  device=torch.device("cuda")):
    
def render_batch(poses, scene,
                 width=300, height=200,
                 fx=113.258171, fy=113.347599,
                 cx=158.868074, cy=98.837772,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):

    means, quats, opacities, scales, colors, transform, scale = scene

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
        view[:3, :3] = R
        view[:3, 3] = [px, py, pz]

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

    print(viewmats)

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
    )

    # =============================
    # 6. 输出 (B,3,H,W)
    # =============================
    imgs = rgb[..., :3].clamp(0, 1)  # (B,H,W,3)
    imgs = imgs.permute(0, 3, 1, 2)  # (B,3,H,W)

    return imgs

if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    test_batch = False

    cfg = Config()
    device = cfg.device
    scene = load_gsplat_scene(cfg)

    if not test_batch:
        
        random_pose = np.array([0.0, -3.0, -0.2, 1.57, 0.0, 0.0])
        # random_pose = np.array([0.5813743, -3.2629182,  -0.10773082,  1.9317428,  -0.04433344,  0.13023579])  # 可以修改为其他pose进行测试  

        img = render(random_pose, scene, device=cfg.device)

        # 显示图像
        plt.imshow(img.permute(1, 2, 0).cpu().numpy())
        plt.axis('off')
        plt.savefig("figures/example_image.png")
        plt.show()
    else:
        

        poses = np.array([
            [0.0, -4.0, 0.0, 1.57, 0.0, 0.0]
        ])

        imgs = render_batch(poses, scene, device=device)
        print("imgs shape:", imgs.shape)  # (B,3,H,W)

        # show first
        img = imgs[0].permute(1,2,0).cpu().numpy()

        plt.imshow(img)
        plt.axis('off')
        plt.savefig("figures/example_image.png")
        plt.show()
        

    