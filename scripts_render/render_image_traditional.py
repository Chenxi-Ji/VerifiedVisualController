import os
import json
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import ClassVar
from scipy.spatial.transform import Rotation
from gsplat.rendering import rasterization


# =============================
# CONFIG
# =============================
@dataclass
class Config:
    scene_name: str = "gate_long"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    SCENES: ClassVar[dict] = {
        "uturn": {
            "gsplat_path": "nerfstudio/outputs/uturn/splatfacto/2025-05-09_151825",
            "checkpoint": "nerfstudio_models/step-000040005.ckpt",
            "camera_params": (300, 200, 113.258171, 113.347599, 158.868074, 98.837772),
            "output_size": None,
            "camera_model": None,
            "use_world_frame": False,
            "pose": np.array([0.0, -3.0, -0.2, 1.57, 0.0, 0.0]),
        },
        "gate_long": {
            "gsplat_path": "nerfstudio/outputs/Gate_Long_hloc_seq/splatfacto/2026-06-11_015308_cleaned",
            "checkpoint": "nerfstudio_models/step-000129999.ckpt",
            "camera_params": (1024, 768, 504.341405, 503.319815, 505.485234, 367.606186),
            "output_size": (256, 192),
            "camera_model": "fisheye",
            "use_world_frame": True,
            # gate-centered frame: gate at origin, +y through the gate, z down (+y deploy side)
            "pose": np.array([0.0, 1.5, 0.0, -np.pi/2, 0.0, 0.0]),
        },
    }

    def __post_init__(self):
        scene = self.SCENES[self.scene_name]
        self.gsplat_path = scene["gsplat_path"]
        self.checkpoint = scene["checkpoint"]
        self.camera_params = scene["camera_params"]
        self.output_size = scene["output_size"]
        self.camera_model = scene["camera_model"]
        self.use_world_frame = scene["use_world_frame"]
        self.pose = scene["pose"]


# =============================
# GSPLAT LOADING（🔥关键）
# =============================
def load_gsplat_scene(cfg):
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
        # Convert to band-0 SH so rasterization(sh_degree=0) reproduces sigmoid(dc).
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

    wf_path = os.path.join(cfg.gsplat_path, "world_frame.json")
    world_frame = cfg.use_world_frame and os.path.exists(wf_path)
    if world_frame:
        with open(wf_path) as f:
            world_transform = np.array(json.load(f)["world_transform"])
        transform = transform @ world_transform

    return means, quats, opacities, scales, colors, transform, scale, world_frame


# camera-axes alignment for the world-frame convention:
# yaw=0,pitch=0,roll=0 looks along +x with the image upright (camera up = -z)
CAM_AXES = np.array([[0, 0, -1],
                     [1, 0,  0],
                     [0, -1, 0]], dtype=np.float64)


# =============================
# viewmat转换（传统备份版，不依赖 coordinate_transform.py）
# =============================
def get_viewmat(optimized_camera_to_world, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
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


def pose_to_viewmat_traditional(pose, transform, scale, world_frame, device):
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
    return get_viewmat(view, device=device)


# =============================
# RENDER（优化版）
# =============================
def render(pose, scene,
            camera_params=(300, 200, 113.258171, 113.347599, 158.868074, 98.837772),
            output_size=None,
            camera_model=None,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    means, quats, opacities, scales, colors, transform, scale, world_frame = scene
    dtype = means.dtype
    width, height, fx, fy, cx, cy = camera_params

    viewmat = pose_to_viewmat_traditional(pose, transform, scale, world_frame, device)

    Ks = torch.tensor([[fx,0,cx],[0,fy,cy],[0,0,1]], device=device, dtype=dtype).unsqueeze(0)

    raster_kwargs = {}
    if camera_model is not None:
        raster_kwargs["camera_model"] = camera_model

    rgb, alpha, _ = rasterization(
        means, quats,
        scales=torch.exp(scales),
        opacities=torch.sigmoid(opacities).squeeze(-1),
        colors=colors,
        viewmats=viewmat,
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
        **raster_kwargs,
    )

    img = rgb[0, ..., :3].clamp(0, 1)
    if output_size is not None:
        img = cv2.resize(img.detach().cpu().numpy(), output_size, interpolation=cv2.INTER_LINEAR)
        return torch.from_numpy(img).permute(2, 0, 1).to(device)
    return img.permute(2, 0, 1).to(device)


def render_batch(poses, scene,
                 camera_params=(300, 200, 113.258171, 113.347599, 158.868074, 98.837772),
                 output_size=None,
                 camera_model=None,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    means, quats, opacities, scales, colors, transform, scale, world_frame = scene
    dtype = means.dtype
    width, height, fx, fy, cx, cy = camera_params

    viewmats = torch.cat([
        pose_to_viewmat_traditional(pose, transform, scale, world_frame, device)
        for pose in poses
    ], dim=0)

    Ks = torch.tensor([[fx,0,cx],[0,fy,cy],[0,0,1]], device=device, dtype=dtype)
    Ks = Ks.unsqueeze(0).repeat(viewmats.shape[0], 1, 1)

    raster_kwargs = {}
    if camera_model is not None:
        raster_kwargs["camera_model"] = camera_model

    rgb, alpha, _ = rasterization(
        means.contiguous(),
        quats.contiguous(),
        scales=torch.exp(scales).contiguous(),
        opacities=torch.sigmoid(opacities).squeeze(-1).contiguous(),
        colors=colors.contiguous(),
        viewmats=viewmats,
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
        **raster_kwargs,
    )

    imgs = rgb[..., :3].clamp(0, 1)
    if output_size is not None:
        imgs_np = imgs.detach().cpu().numpy()
        imgs_np = np.stack([
            cv2.resize(img, output_size, interpolation=cv2.INTER_LINEAR)
            for img in imgs_np
        ], axis=0)
        return torch.from_numpy(imgs_np).permute(0, 3, 1, 2).to(device)
    return imgs.permute(0, 3, 1, 2).to(device)


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)

    cfg = Config()
    device = cfg.device
    scene = load_gsplat_scene(cfg)
        
    img = render(
        cfg.pose,
        scene,
        cfg.camera_params,
        output_size=cfg.output_size,
        camera_model=cfg.camera_model,
        device=cfg.device,
    )

    # 显示图像
    plt.imshow(img.permute(1, 2, 0).cpu().numpy())
    plt.axis('off')
    plt.savefig("figures/example_image_traditional.png")
    plt.show()
