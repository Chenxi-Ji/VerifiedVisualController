import os
import math
import json
import sys
import time
from itertools import product
from tqdm import tqdm

from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from scipy.spatial.transform import Rotation
from nerfstudio.utils import colormaps
import cv2

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from scripts_render.render_image import render
from scripts_control.utils_ctrl_lya_pt import Controller, Lyapunov, transform_drone_velocity_to_world_frame

# from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm

# =============================
# CONFIG
# =============================
@dataclass
class Config:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dt = 0.1
    H = 5
    n_split = [20, 20, 10, 1]
    n_split = [ele+1 for ele in n_split]

    target_pose = np.array([0.0, -3.0, -0.2, 1.57, 0.0, 0.0])
    gate_pose = np.array([0.0, -2.0, -0.2, 1.57, 0.0, 0.0])

    save_cert_filename =  "cert_inner"
    if save_cert_filename == "cert":
        pose_lb = np.array([-1.0, -3.8, -0.4, 1.57, 0.0, 0.0])
        pose_ub = np.array([1.0, -2.2, -0.0, 1.57, 0.0, 0.0])
    elif save_cert_filename == "cert_inner":
        pose_lb = np.array([-0.8, -3.8, -0.4, 1.57, 0.0, 0.0])
        pose_ub = np.array([0.8, -3.2, 0.0, 1.57, 0.0, 0.0])

    # gsplat path
    gsplat_path = "nerfstudio/outputs/uturn/splatfacto/2025-05-09_151825"
    checkpoint = "nerfstudio_models/step-000040005.ckpt"

    save_path = "weights/ctrl_lya.pt" #"ctrl_lya.pt"
    video_dir = "videos"
    

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

def run_test(pose_lb, pose_ub, ctrl, Vnet, scene, render_fn, device, video_dir="test_videos"):

    target = cfg.target_pose
    gate = cfg.gate_pose
    dt = cfg.dt
    filename = cfg.save_cert_filename

    ctrl.to(device).eval()
    Vnet.to(device).eval()

    # Create video directory
    os.makedirs(video_dir, exist_ok=True)

    sample_size = 32
    target_t = torch.tensor(target, device=device, dtype=torch.float32).unsqueeze(0)

    # unpack bounds
    pos_x_lb, pos_y_lb, pos_z_lb, yaw_lb, pitch_lb, roll_lb = pose_lb
    pos_x_ub, pos_y_ub, pos_z_ub, yaw_ub, pitch_ub, roll_ub = pose_ub

    # split intervals
    n_split = cfg.n_split
    pos_x_list = torch.linspace(pos_x_lb, pos_x_ub, steps=n_split[0], device=device)
    pos_y_list = torch.linspace(pos_y_lb, pos_y_ub, steps=n_split[1], device=device)
    pos_z_list = torch.linspace(pos_z_lb, pos_z_ub, steps=n_split[2], device=device)
    yaw_list = torch.linspace(yaw_lb, yaw_ub, steps=n_split[3], device=device)

    x_pairs = list(zip(pos_x_list[:-1], pos_x_list[1:]))
    y_pairs = list(zip(pos_y_list[:-1], pos_y_list[1:]))
    z_pairs = list(zip(pos_z_list[:-1], pos_z_list[1:]))
    yaw_pairs = list(zip(yaw_list[:-1], yaw_list[1:]))

    verified_boxes = []
    total = 0
    verified_count = 0

    total_boxes = len(x_pairs) * len(y_pairs) * len(z_pairs) * len(yaw_pairs)
    pbar = tqdm(total=total_boxes)

    for (x_lb, x_ub), (y_lb, y_ub), (z_lb, z_ub), (yaw_lb, yaw_ub) in tqdm(
        product(x_pairs, y_pairs, z_pairs, yaw_pairs),
        total=len(x_pairs) * len(y_pairs) * len(z_pairs) * len(yaw_pairs)
    ):
    
        # print(f"Sampling poses in bounds: x=[{x_lb:.2f}, {x_ub:.2f}], y=[{y_lb:.2f}, {y_ub:.2f}], z=[{z_lb:.2f}, {z_ub:.2f}], yaw=[{yaw_lb_s:.2f}, {yaw_ub_s:.2f}]")

        lb = torch.tensor(
            [x_lb, y_lb, z_lb, yaw_lb, pitch_lb, roll_lb],
            device=device
        )
        ub = torch.tensor(
            [x_ub, y_ub, z_ub, yaw_ub, pitch_ub, roll_ub],
            device=device
        )

        # batched uniform sampling: [batch_size, 6]
        poses = lb + torch.rand(
            sample_size, 6, device=device
        ) * (ub - lb)

        poses = poses.to(torch.float32)

        V_curr, _ = Vnet(poses, target_t.expand_as(poses))
        V_curr = V_curr.squeeze(-1)

        V_curr_min = V_curr.min()
        V_curr_max = V_curr.max()

        # print(f"V range: [{V_curr_min.item():.6f}, {V_curr_max.item():.6f}]")

        imgs = torch.stack([
            render_fn(
                pose.detach().cpu().numpy(),
                scene,
                device=device
            )
            for pose in poses
        ], dim=0)  # [B, C, H, W]

        pred_self = ctrl(imgs)  # [B, 3] (assumed)
        pred = transform_drone_velocity_to_world_frame(pred_self)

        # append zeros for pitch / roll update
        zeros = torch.zeros(
            sample_size, 2,
            device=pred.device,
            dtype=pred.dtype
        )

        pred = torch.cat([pred, zeros], dim=-1)  # [B, 6]
        next_poses = poses + pred * dt

        V_next, _ = Vnet(next_poses, target_t.expand_as(next_poses))
        V_next = V_next.squeeze(-1)
        V_next_min = V_next.min()
        V_next_max = V_next.max()

        # certification
        verified = (V_next_max < V_curr_min-0.01).item()
        # print(f"Certified: {verified}, x_range,y_range,z_range,yaw_range=({x_lb:.2f},{x_ub:.2f}),({y_lb:.2f},{y_ub:.2f}),({z_lb:.2f},{z_ub:.2f}),({yaw_lb_s:.2f},{yaw_ub_s:.2f}), V_curr_range=({V_curr_min.item():.6f}, {V_curr_max.item():.6f}), V_next_range=({V_next_min.item():.6f}, {V_next_max.item():.6f})  ")

        total += 1
        if verified:
            verified_count += 1

        verified_boxes.append({
            "x_lb": x_lb.item(),
            "x_ub": x_ub.item(),
            "y_lb": y_lb.item(),
            "y_ub": y_ub.item(),
            "z_lb": z_lb.item(),
            "z_ub": z_ub.item(),
            "verified": verified
        })

        # print(f"\rVerified: {verified_count / total * 100:.2f}%", end="")
        pbar.set_postfix({
            "verified": f"{verified_count}/{pbar.n + 1}",
            "pct": f"{verified_count / (pbar.n + 1) * 100:.2f}%"
        })

        pbar.update(1)

    save_dict = {
        "verified_boxes": verified_boxes,
        "target": target,
        "gate": gate
    }
    torch.save(save_dict, f"results/{filename}_result.pt")

    # =========================
    # 3D plotting
    # =========================
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")


    def draw_box(ax, x0, x1, y0, y1, z0, z1, color):
        verts = [
            [(x0,y0,z0),(x1,y0,z0),(x1,y1,z0),(x0,y1,z0)],
            [(x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1)],
            [(x0,y0,z0),(x1,y0,z0),(x1,y0,z1),(x0,y0,z1)],
            [(x0,y1,z0),(x1,y1,z0),(x1,y1,z1),(x0,y1,z1)],
            [(x0,y0,z0),(x0,y1,z0),(x0,y1,z1),(x0,y0,z1)],
            [(x1,y0,z0),(x1,y1,z0),(x1,y1,z1),(x1,y0,z1)],
        ]

        ax.add_collection3d(
            Poly3DCollection(
                verts,
                alpha=0.25,
                facecolor=color,
                edgecolor=color
            )
        )


    for box in verified_boxes:
        color = "green" if box["verified"] else "red"

        draw_box(
            ax,
            box["x_lb"], box["x_ub"],
            box["y_lb"], box["y_ub"],
            box["z_lb"], box["z_ub"],
            color=color
        )

    ax.scatter(*target[:3], c='red', s=30, marker='*', label='Target')
    ax.scatter(*gate[:3], c='black', s=30, marker='*', label='Gate')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Certified Regions (green=verified, red=not verified)")

    plt.tight_layout()
    plt.savefig(f"figures/{filename}_regions.png", dpi=300, bbox_inches="tight")
    plt.show()



if __name__ == "__main__":
    
    os.makedirs("results", exist_ok=True)
    test_batch = False
    linear = False

    cfg = Config()
    device = cfg.device
    scene = load_gsplat_scene(cfg)

    ctrl = Controller().to(device)
    Vnet = Lyapunov().to(device)
    render_fn = render

    # load weights
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(PROJECT_ROOT, "../", cfg.save_path)
    ckpt = torch.load(save_path, map_location=device)
    ctrl.load_state_dict(ckpt["controller"])
    Vnet.load_state_dict(ckpt["lyapunov"])  

    pose_lb= cfg.pose_lb
    pose_ub = cfg.pose_ub

    dt = cfg.dt
    run_test(pose_lb, pose_ub, ctrl, Vnet, scene, render_fn, device)
    

    
    