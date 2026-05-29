import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import sys

# =============================
# IMPORT
# =============================
sys.path.append('.')
from utils_ctrl_lya_pt import Lyapunov


@dataclass
class Config:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dt = 0.1

    # [x, y, z, yaw, pitch, roll]
    target_pose = np.array([0.0, -3.0, -0.2, 1.57, 0.0, 0.0])

    save_path = "weights/ctrl_lya.pt"


# =============================
# LOAD MODEL
# =============================
def load_model(path, device):
    model = Lyapunov().to(device)

    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["lyapunov"])

    model.eval()
    print("✅ Loaded model")

    return model


def eval_V(model, pose, target, device):
    with torch.no_grad():
        p = torch.tensor(
            pose, dtype=torch.float32, device=device
        ).unsqueeze(0)

        t = torch.tensor(
            target, dtype=torch.float32, device=device
        ).unsqueeze(0)

        V_val, _ = model(p, t)

        return V_val.item()


# =============================
# GENERIC 2D SLICE
# =============================
def compute_slice(
    Vnet,
    target,
    device,
    dim_i,
    dim_j,
    range_i,
    range_j,
    resolution,
    freeze_mode,
):
    """
    freeze_mode:
        "yaw" -> freeze yaw
        "xyz" -> freeze xyz
    """

    grid_i = np.linspace(range_i[0], range_i[1], resolution)
    grid_j = np.linspace(range_j[0], range_j[1], resolution)

    II, JJ = np.meshgrid(grid_i, grid_j)
    V = np.zeros_like(II)

    for a in range(resolution):
        for b in range(resolution):

            pose = target.copy()

            pose[dim_i] = II[a, b]
            pose[dim_j] = JJ[a, b]

            if freeze_mode == "yaw":
                pose[3] = target[3]

            elif freeze_mode == "xyz":
                pose[:3] = target[:3]

            V[a, b] = eval_V(
                Vnet,
                pose,
                target,
                device
            )

    return II, JJ, V


# =============================
# MAIN PLOT
# =============================
def plot_lya_2d(
    Vnet,
    target,
    device,
    resolution=100
):

    fig, axes = plt.subplots(
        2, 3,
        figsize=(15, 8)
    )

    # ==================================
    # FIRST ROW: XYZ SLICES (yaw fixed)
    # ==================================
    xyz_pairs = [
        (0, 1, "x-y"),
        (1, 2, "y-z"),
        (2, 0, "z-x"),
    ]

    xyz_ranges = [
        [target[0] - 3, target[0] + 3],
        [target[1] - 3, target[1] + 3],
        [target[2] - 2, target[2] + 2],
    ]

    for idx, (i, j, name) in enumerate(xyz_pairs):

        ax = axes[0, idx]

        II, JJ, V = compute_slice(
            Vnet,
            target,
            device,
            i,
            j,
            xyz_ranges[i],
            xyz_ranges[j],
            resolution,
            freeze_mode="yaw"
        )

        ax.contour(II, JJ, V, levels=20)
        ax.contourf(II, JJ, V, levels=20, alpha=0.6)

        ax.scatter(
            target[i],
            target[j],
            c='red',
            s=80,
            marker='*'
        )

        ax.set_title(f"{name} (yaw fixed)")
        ax.set_xlabel(name.split('-')[0])
        ax.set_ylabel(name.split('-')[1])
        ax.grid(True)

    # ==================================
    # SECOND ROW: XYZ-YAW SLICES
    # ==================================
    yaw_pairs = [
        (0, 3, "x-yaw"),
        (1, 3, "y-yaw"),
        (2, 3, "z-yaw"),
    ]

    yaw_ranges = [
        [target[0] - 3, target[0] + 3],
        [target[1] - 3, target[1] + 3],
        [target[2] - 2, target[2] + 2],
        [target[3] - np.pi, target[3] + np.pi],
    ]

    for idx, (i, j, name) in enumerate(yaw_pairs):

        ax = axes[1, idx]

        II, JJ, V = compute_slice(
            Vnet,
            target,
            device,
            i,
            j,
            yaw_ranges[i],
            yaw_ranges[j],
            resolution,
            freeze_mode="xyz"
        )

        ax.contour(II, JJ, V, levels=20)
        ax.contourf(II, JJ, V, levels=20, alpha=0.6)

        ax.scatter(
            target[i],
            target[j],
            c='red',
            s=80,
            marker='*'
        )

        ax.set_title(f"{name} (xyz fixed)")
        ax.set_xlabel(name.split('-')[0])
        ax.set_ylabel(name.split('-')[1])
        ax.grid(True)

    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/lya_2d.png", dpi=150)

    print("✅ Saved figure: figures/lya_2d.png")


# =============================
# MAIN
# =============================
def main():

    cfg = Config()
    device = cfg.device

    target = cfg.target_pose

    print("Loading model...")
    Vnet = load_model(
        cfg.save_path,
        device
    )

    print("Plotting 2D slices...")
    plot_lya_2d(
        Vnet,
        target,
        device,
        resolution=80
    )


if __name__ == "__main__":
    main()