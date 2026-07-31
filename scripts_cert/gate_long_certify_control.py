import os
import sys
from itertools import product
from collections import deque
from tqdm import tqdm

from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

SCRIPT_DIR = Path(__file__).resolve().parent          # <root>/scripts_cert
PROJECT_ROOT = SCRIPT_DIR.parent                       # <root>
CONTROL_DIR = PROJECT_ROOT / "scripts_control"
RENDER_DIR = PROJECT_ROOT / "scripts_render"

# certify_control.py lives in <root>/scripts_cert, while the controller utilities
# and renderer live in sibling folders. Add the project paths explicitly so the
# script works when launched as:
#   python scripts_cert/certify_control.py
for path in (PROJECT_ROOT, CONTROL_DIR, RENDER_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

try:
    from scripts_render.render_image import (
        Config as RenderConfig,
        render,
        load_gsplat_scene,
    )
except ModuleNotFoundError as exc:
    if exc.name not in {"scripts_render", "scripts_render.render_image"}:
        raise
    from render_image import Config as RenderConfig, render, load_gsplat_scene

try:
    from scripts_control.utils_ctrl_lya_pt import (
        Controller,
        Lyapunov,
        transform_drone_velocity_to_world_frame,
    )
except ModuleNotFoundError as exc:
    if exc.name not in {"scripts_control", "scripts_control.utils_ctrl_lya_pt"}:
        raise
    from utils_ctrl_lya_pt import Controller, Lyapunov, transform_drone_velocity_to_world_frame

# from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm


# =============================
# CONFIG
# =============================
@dataclass
class Config:
    scene_name = "gate_long"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dt = 0.1
    H = 5

    n_split = [20, 20, 10, 1]
    n_split = [ele + 1 for ele in n_split]

    min_cell_size = 0.01
    max_depth = 10
    sample_size = 64
    lyapunov_margin = 0.01
    lyapunov_verified_threshold = 0.05

    # Match test_ctrl_lya_pt.py: gate-centered world frame, gate at the origin,
    # +y on the deployment side, z down, and yaw=-pi/2 facing the gate.
    target_pose = np.array([0.0, 1.5, 0.0, -np.pi / 2, 0.0, 0.0])
    gate_pose = np.array([0.0, 0.0, 0.0, -np.pi / 2, 0.0, 0.0])

    save_path = "weights/ctrl_lya.pt"
    video_dir = f"videos_{scene_name}"
    results_dir = "results"
    figures_dir = "figures"

    region_size = "small"
    save_cert_filename = f"{scene_name}_{region_size}_cert"

    # Use the same state domain sampled by sample_init_poses() in
    # test_ctrl_lya_pt.py: target + [low, high].

    if region_size == "large":
        pose_lb = target_pose + np.array([-0.5, -0.5, -0.3, -0.0, 0.0, 0.0])
        pose_ub = target_pose + np.array([0.5, 0.5, 0.3, 0.0, 0.0, 0.0])
    elif region_size == "mid":
        pose_lb = target_pose + np.array([-0.3, -0.3, -0.2, 0.0, 0.0, 0.0])
        pose_ub = target_pose + np.array([0.3, 0.3, 0.2, 0.0, 0.0, 0.0])
    elif region_size == "small":
        pose_lb = target_pose + np.array([0.1, -0.2, -0.1, 0.0, 0.0, 0.0])
        pose_ub = target_pose + np.array([0.3, 0.2, 0.1, 0.0, 0.0, 0.0])
    else:
        raise ValueError(f"Unknown region_size: {region_size}")


# =============================
# ADAPTIVE CELL SPLIT
# =============================
def subdivide_cell(lb, ub, threshold):
    size = ub - lb  # (6,)
    split_dims = torch.where(size > threshold)[0]  # (K,)

    if split_dims.numel() == 0:
        return []

    mid = 0.5 * (lb + ub)  # (6,)
    children = []

    for choice in product([0, 1], repeat=split_dims.numel()):
        child_lb = lb.clone()  # (6,)
        child_ub = ub.clone()  # (6,)

        for dim, side in zip(split_dims, choice):
            if side == 0:
                child_ub[dim] = mid[dim]
            else:
                child_lb[dim] = mid[dim]

        children.append((child_lb, child_ub))

    return children


def make_initial_queue(pose_lb, pose_ub, cfg, device):
    pos_x_lb, pos_y_lb, pos_z_lb, yaw_lb, pitch_lb, roll_lb = pose_lb
    pos_x_ub, pos_y_ub, pos_z_ub, yaw_ub, pitch_ub, roll_ub = pose_ub

    n_split = cfg.n_split

    pos_x_list = torch.linspace(pos_x_lb, pos_x_ub, steps=n_split[0], device=device)
    pos_y_list = torch.linspace(pos_y_lb, pos_y_ub, steps=n_split[1], device=device)
    pos_z_list = torch.linspace(pos_z_lb, pos_z_ub, steps=n_split[2], device=device)
    yaw_list = torch.linspace(yaw_lb, yaw_ub, steps=n_split[3], device=device)

    x_pairs = list(zip(pos_x_list[:-1], pos_x_list[1:]))
    y_pairs = list(zip(pos_y_list[:-1], pos_y_list[1:]))
    z_pairs = list(zip(pos_z_list[:-1], pos_z_list[1:]))
    yaw_pairs = list(zip(yaw_list[:-1], yaw_list[1:]))

    queue = deque()

    for (x_lb, x_ub), (y_lb, y_ub), (z_lb, z_ub), (yaw_lb_i, yaw_ub_i) in product(
        x_pairs,
        y_pairs,
        z_pairs,
        yaw_pairs,
    ):
        lb = torch.tensor(
            [
                x_lb.item(),
                y_lb.item(),
                z_lb.item(),
                yaw_lb_i.item(),
                pitch_lb,
                roll_lb,
            ],
            device=device,
            dtype=torch.float32,
        )  # (6,)

        ub = torch.tensor(
            [
                x_ub.item(),
                y_ub.item(),
                z_ub.item(),
                yaw_ub_i.item(),
                pitch_ub,
                roll_ub,
            ],
            device=device,
            dtype=torch.float32,
        )  # (6,)

        queue.append((lb, ub, 0))

    return queue


def verify_one_cell(lb, ub, ctrl, Vnet, scene, render_fn, cfg, target_t, device):
    poses = lb + torch.rand(
        cfg.sample_size,
        6,
        device=device,
        dtype=torch.float32,
    ) * (ub - lb)  # (B,6)

    target_batch = target_t.expand_as(poses)  # (B,6)

    V_curr, _ = Vnet(poses, target_batch)
    V_curr = V_curr.squeeze(-1)  # (B,)

    V_curr_min = V_curr.min()
    V_curr_max = V_curr.max()

    if V_curr_max <= cfg.lyapunov_verified_threshold:
        stats = {
            "V_curr_min": V_curr_min.item(),
            "V_curr_max": V_curr_max.item(),
            "V_next_min": float("nan"),
            "V_next_max": float("nan"),
            "threshold_verified": True,
        }
        return True, stats

    imgs = torch.stack(
        [
            render_fn(
                pose.detach().cpu().numpy(),
                scene,
                cfg.camera_params,
                output_size=cfg.output_size,
                camera_model=cfg.camera_model,
                device=device,
            )
            for pose in poses
        ],
        dim=0,
    )

    # Match the current utils_ctrl_lya_pt.py convention: controller output is
    # transformed from drone frame to world frame, then zero pitch/roll rates
    # are appended and the full 6D pose is integrated with forward Euler.
    pred_self = ctrl(imgs)  # (B,4): [vx, vy, vz, yaw_rate] in body frame
    if pred_self.ndim != 2 or pred_self.shape[-1] != 4:
        raise ValueError(
            "Controller must return shape (B, 4) containing "
            f"[vx, vy, vz, yaw_rate], but got {tuple(pred_self.shape)}"
        )

    pred = transform_drone_velocity_to_world_frame(pred_self)  # (B,4)
    zeros = torch.zeros(
        *pred.shape[:-1],
        2,
        device=pred.device,
        dtype=pred.dtype,
    )  # (B,2)
    pred = torch.cat([pred, zeros], dim=-1)  # (B,6)
    next_poses = poses + pred * cfg.dt       # (B,6)

    V_next, _ = Vnet(next_poses, target_batch)
    V_next = V_next.squeeze(-1)  # (B,)

    V_next_min = V_next.min()
    V_next_max = V_next.max()

    verified = (V_next_max < V_curr_min - cfg.lyapunov_margin).item()

    stats = {
        "V_curr_min": V_curr_min.item(),
        "V_curr_max": V_curr_max.item(),
        "V_next_min": V_next_min.item(),
        "V_next_max": V_next_max.item(),
        "threshold_verified": False,
    }

    return verified, stats


def box_to_dict(lb, ub, verified, depth, stats):
    return {
        "x_lb": lb[0].item(),
        "x_ub": ub[0].item(),
        "y_lb": lb[1].item(),
        "y_ub": ub[1].item(),
        "z_lb": lb[2].item(),
        "z_ub": ub[2].item(),
        "yaw_lb": lb[3].item(),
        "yaw_ub": ub[3].item(),
        "pitch_lb": lb[4].item(),
        "pitch_ub": ub[4].item(),
        "roll_lb": lb[5].item(),
        "roll_ub": ub[5].item(),
        "verified": verified,
        "depth": depth,
        "V_curr_min": stats["V_curr_min"],
        "V_curr_max": stats["V_curr_max"],
        "V_next_min": stats["V_next_min"],
        "V_next_max": stats["V_next_max"],
        "threshold_verified": stats.get("threshold_verified", False),
    }


# =============================
# MAIN VERIFICATION
# =============================
def run_test(pose_lb, pose_ub, ctrl, Vnet, scene, render_fn, cfg, device, video_dir="test_videos"):
    target = cfg.target_pose
    gate = cfg.gate_pose
    filename = cfg.save_cert_filename

    ctrl.to(device).eval()
    Vnet.to(device).eval()

    video_dir = Path(video_dir)
    results_dir = PROJECT_ROOT / cfg.results_dir
    figures_dir = PROJECT_ROOT / cfg.figures_dir

    video_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    target_t = torch.tensor(
        target,
        device=device,
        dtype=torch.float32,
    ).unsqueeze(0)  # (1,6)

    queue = make_initial_queue(pose_lb, pose_ub, cfg, device)

    verified_boxes = []
    evaluated_count = 0
    verified_count = 0
    terminal_failed_count = 0
    subdivided_count = 0

    pbar = tqdm(total=len(queue), desc="Adaptive verification")

    with torch.no_grad():
        while len(queue) > 0:
            lb, ub, depth = queue.popleft()

            verified, stats = verify_one_cell(
                lb,
                ub,
                ctrl,
                Vnet,
                scene,
                render_fn,
                cfg,
                target_t,
                device,
            )

            evaluated_count += 1

            if verified:
                verified_count += 1

                verified_boxes.append(
                    box_to_dict(
                        lb,
                        ub,
                        verified=True,
                        depth=depth,
                        stats=stats,
                    )
                )

            else:
                children = []

                if depth < cfg.max_depth:
                    children = subdivide_cell(
                        lb,
                        ub,
                        cfg.min_cell_size,
                    )

                if len(children) > 0:
                    subdivided_count += 1

                    for child_lb, child_ub in children:
                        queue.append((child_lb, child_ub, depth + 1))

                    pbar.total += len(children)
                    pbar.refresh()

                else:
                    terminal_failed_count += 1

                    verified_boxes.append(
                        box_to_dict(
                            lb,
                            ub,
                            verified=False,
                            depth=depth,
                            stats=stats,
                        )
                    )

            pbar.update(1)
            pbar.set_postfix(
                {
                    "queue": len(queue),
                    "verified": verified_count,
                    "failed": terminal_failed_count,
                    "eval": evaluated_count,
                    "subdiv": subdivided_count,
                }
            )

    pbar.close()

    save_dict = {
        "verified_boxes": verified_boxes,
        "target": target,
        "gate": gate,
        "cfg": {
            "scene_name": cfg.scene_name,
            "region_size": cfg.region_size,
            "dt": cfg.dt,
            "n_split": cfg.n_split,
            "min_cell_size": cfg.min_cell_size,
            "max_depth": cfg.max_depth,
            "sample_size": cfg.sample_size,
            "lyapunov_margin": cfg.lyapunov_margin,
            "lyapunov_verified_threshold": cfg.lyapunov_verified_threshold,
        },
        "summary": {
            "evaluated_count": evaluated_count,
            "verified_count": verified_count,
            "terminal_failed_count": terminal_failed_count,
            "subdivided_count": subdivided_count,
            "final_leaf_count": len(verified_boxes),
        },
    }

    result_path = results_dir / f"{filename}_result.pt"
    torch.save(save_dict, result_path)

    print("\n========== Verification Summary ==========")
    print(f"Evaluated cells        : {evaluated_count}")
    print(f"Verified leaf cells    : {verified_count}")
    print(f"Failed leaf cells      : {terminal_failed_count}")
    print(f"Subdivided cells       : {subdivided_count}")
    print(f"Final leaf cells       : {len(verified_boxes)}")
    print(f"Saved to               : {result_path}")
    print("==========================================\n")

    # =========================
    # 3D plotting
    # =========================
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    def draw_box(ax, x0, x1, y0, y1, z0, z1, color):
        verts = [
            [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],
            [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],
            [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],
            [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)],
            [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)],
            [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],
        ]

        ax.add_collection3d(
            Poly3DCollection(
                verts,
                alpha=0.25,
                facecolor=color,
                edgecolor=color,
            )
        )

    for box in verified_boxes:
        color = "green" if box["verified"] else "red"

        draw_box(
            ax,
            box["x_lb"],
            box["x_ub"],
            box["y_lb"],
            box["y_ub"],
            box["z_lb"],
            box["z_ub"],
            color=color,
        )

    ax.scatter(*target[:3], c="red", s=30, marker="*", label="Target")
    ax.scatter(*gate[:3], c="black", s=30, marker="*", label="Gate")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Adaptive Certified Regions: green=verified, red=failed")
    ax.legend()

    plt.tight_layout()
    figure_path = figures_dir / f"{filename}_regions.png"
    plt.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    cfg = Config()
    device = cfg.device

    assert cfg.scene_name == "gate_long", "This certification script is configured for scene_name='gate_long' only."

    render_cfg = RenderConfig(
        scene_name=cfg.scene_name,
        device=cfg.device,
    )
    cfg.camera_params = render_cfg.camera_params
    cfg.output_size = render_cfg.output_size
    cfg.camera_model = render_cfg.camera_model

    scene = load_gsplat_scene(render_cfg)

    ctrl = Controller().to(device)
    Vnet = Lyapunov().to(device)

    # Match test_ctrl_lya_pt.py's checkpoint loading and architecture check.
    save_path = Path(cfg.save_path)
    if not save_path.is_absolute():
        save_path = PROJECT_ROOT / save_path

    ckpt = torch.load(save_path, map_location=device)
    try:
        ctrl.load_state_dict(ckpt["controller"])
    except RuntimeError as exc:
        raise RuntimeError(
            f"{save_path} does not match the current Controller architecture "
            "(it may predate the vertical-readout/TFLite-safe pooling upgrade). "
            "Train new weights with scripts_control/train_ctrl_lya_pt.py and "
            "copy the intended checkpoint to weights/ctrl_lya.pt."
        ) from exc
    Vnet.load_state_dict(ckpt["lyapunov"])

    run_test(
        pose_lb=cfg.pose_lb,
        pose_ub=cfg.pose_ub,
        ctrl=ctrl,
        Vnet=Vnet,
        scene=scene,
        render_fn=render,
        cfg=cfg,
        device=device,
        video_dir=PROJECT_ROOT / cfg.video_dir,
    )
