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
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dt = 0.1
    H = 5

    target_pose = np.array([0.0, -3.0, -0.2, 1.57, 0.0, 0.0])

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
    # colors = dc[:, None, :] #(B, 1, 3)
    
    # print(colors.shape)
    # 👉 只加载一次 transform（重要优化）
    with open(os.path.join(cfg.gsplat_path, "dataparser_transforms.json"), "r") as f:
        meta = json.load(f)

    transform = np.array(meta["transform"])
    scale = meta["scale"]

    return means, quats, opacities, scales, colors, transform, scale

if __name__ == "__main__":
    
    os.makedirs("figures", exist_ok=True)
    test_batch = False
    linear = False

    cfg = Config()
    device = cfg.device
    scene = load_gsplat_scene(cfg)
        
    width = 300
    height = 200
    fx = 113.258171
    fy = 113.347599
    cx = 158.868074
    cy = 98.837772

    # width = width // 2
    # height = height // 2
    # fx = fx / 2
    # fy = fy / 2
    # cx = cx / 2
    # cy = cy / 2
    # pose_lb = np.array([0.1, -3.1, -0.1, 1.53, 0.0, 0.0])
    # pose_ub = np.array([0.3, -2.9, -0.3, 1.60, 0.0, 0.0])

    pose_lb = np.array([0.09, -3.01, -0.02, 1.60, 0.0, 0.0])
    pose_ub = np.array([0.12, -2.97, -0.00, 1.60, 0.0, 0.0])
    
    start_time = time.time()
    img, img_lb, img_ub = render_bound(pose_lb, pose_ub, scene, width, height, fx, fy, cx, cy, device=cfg.device, linear = linear)
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
    plt.savefig("figures/abstract_images.png", dpi=300, bbox_inches="tight")
    plt.show()


    
    