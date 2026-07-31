import os
import math
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import ClassVar
from torch.utils.data import Dataset, DataLoader
from scipy.spatial.transform import Rotation
# from gsplat.rendering import rasterization
from nerfstudio.utils import colormaps
import cv2

from utils_abstract_render import render_bound

# =============================
# CONFIG
# =============================
@dataclass
class Config:
    scene_name: str = "gate_long"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    scenes: ClassVar[dict] = {
        "uturn": {
            "pose_lb": np.array([0.09, -3.01, -0.02, 1.60, 0.0, 0.0]),
            "pose_ub": np.array([0.12, -2.97, -0.00, 1.60, 0.0, 0.0]),
            "camera_params": (300, 200, 113.258171, 113.347599, 158.868074, 98.837772),
            "gsplat_path": "nerfstudio/outputs/uturn/splatfacto/2025-05-09_151825",
            "checkpoint": "nerfstudio_models/step-000040005.ckpt",
            "use_world_frame": False,
        },
        "gate_long": {
            "pose_lb": np.array([-0.01, 1.49, 0.0, -math.pi / 2, 0.0, 0.0]),
            "pose_ub": np.array([0.01, 1.51, 0.0, -math.pi / 2, 0.0, 0.0]),
            "camera_params": (256, 192, 126.08535125, 125.82995375, 126.3713085, 91.9015465),
            "gsplat_path": "nerfstudio/outputs/Gate_Long_hloc_seq/splatfacto/2026-06-11_015308_cleaned",
            "checkpoint": "nerfstudio_models/step-000129999.ckpt",
            "use_world_frame": True,
        },
    }

    chunk_size: int = 1024*2
    tile: int = 32
    debug: bool = False
    output_path: str = "figures/abstract_images.png"

    def __post_init__(self):
        if self.scene_name not in self.scenes:
            raise ValueError(f"Unknown scene_name={self.scene_name}. Available scenes: {list(self.scenes)}")
        scene = self.scenes[self.scene_name]
        self.pose_lb = scene["pose_lb"]
        self.pose_ub = scene["pose_ub"]
        self.camera_params = scene["camera_params"]
        self.gsplat_path = scene["gsplat_path"]
        self.checkpoint = scene["checkpoint"]
        self.use_world_frame = scene["use_world_frame"]

# =============================
# GSPLAT LOADING
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
        colors = (dc.sigmoid() - 0.5) / 0.28209479177387814
        colors = colors[:, None, :]
    else:
        colors = torch.cat((dc[:, None, :], rest), dim=1)
    # colors = dc[:, None, :] #(B, 1, 3)
    
    # print(colors.shape)
    # 👉 只加载一次 transform（重要优化）
    with open(os.path.join(cfg.gsplat_path, "dataparser_transforms.json"), "r") as f:
        meta = json.load(f)

    transform = np.array(meta["transform"])
    transform = np.vstack([transform, np.array([0.0, 0.0, 0.0, 1.0])])
    scale = meta["scale"]

    if cfg.use_world_frame:
        with open(os.path.join(cfg.gsplat_path, "world_frame.json"), "r") as f:
            world_frame_meta = json.load(f)
        world_frame = np.array(world_frame_meta["world_transform"])
        transform = transform @ world_frame

    return means, quats, opacities, scales, colors, transform, scale, cfg.use_world_frame

if __name__ == "__main__":
    cfg = Config()
    os.makedirs(os.path.dirname(cfg.output_path), exist_ok=True)

    device = cfg.device
    scene = load_gsplat_scene(cfg)
    
    start_time = time.time()
    img, img_lb, img_ub = render_bound(
        cfg.pose_lb,
        cfg.pose_ub,
        scene,
        cfg.camera_params,
        device=cfg.device,
        chunk_size=cfg.chunk_size,
        tile=cfg.tile,
        debug=cfg.debug,
    )
    end_time = time.time()
    print(f"Total rendering time: {end_time - start_time:.2f} seconds")

    # 显示图像
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    imgs = [
        img_lb.permute(1, 2, 0).detach().cpu().numpy(),
        img.permute(1, 2, 0).detach().cpu().numpy(),
        img_ub.permute(1, 2, 0).detach().cpu().numpy(),
    ]

    titles = ["img_lb", "img", "img_ub"]

    for ax, im, t in zip(axes, imgs, titles):
        ax.imshow(im)
        ax.axis("off")
        ax.set_title(t)

    plt.tight_layout()
    plt.savefig(cfg.output_path, dpi=300, bbox_inches="tight")
    plt.show()


    
    
