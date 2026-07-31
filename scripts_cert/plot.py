import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# =========================
# Config
# =========================
@dataclass
class Config:
    scene_name: str = "uturn"  # "uturn" or "gate_long"
    region_size: str = "small"
    results_dir: str = "results"
    figures_dir: str = "figures"

    @property
    def filename(self):
        return f"{self.scene_name}_{self.region_size}_cert"

    @property
    def result_filename(self):
        return os.path.join(self.results_dir, f"{self.filename}_result.pt")

    @property
    def figure_filename(self):
        return os.path.join(self.figures_dir, f"{self.filename}_partition_and_verification.png")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot certification result for uturn or gate_long.")
    parser.add_argument(
        "--scene",
        choices=["uturn", "gate_long"],
        default="uturn",
        help="Scene name to plot.",
    )
    parser.add_argument(
        "--region",
        default="small",
        help="Region size tag used in the result filename.",
    )
    return parser.parse_args()


# =========================
# Load data
# =========================
args = parse_args()
cfg = Config(scene_name=args.scene, region_size=args.region)

data = torch.load(cfg.result_filename, map_location="cpu")

verified_boxes = data["verified_boxes"]
target = np.asarray(data["target"][:3])
gate = np.asarray(data["gate"][:3])

os.makedirs(cfg.figures_dir, exist_ok=True)

print(f"Loaded result from: {cfg.result_filename}")
print(f"Scene             : {cfg.scene_name}")
print(f"Region            : {cfg.region_size}")
print(f"Total boxes       : {len(verified_boxes)}")


# =========================
# Helper functions
# =========================
def get_box_max_size(box):
    dx = box["x_ub"] - box["x_lb"]
    dy = box["y_ub"] - box["y_lb"]
    dz = box["z_ub"] - box["z_lb"]

    return max(dx, dy, dz)


def get_global_axis_limits(boxes):
    x_min = min(box["x_lb"] for box in boxes)
    x_max = max(box["x_ub"] for box in boxes)

    y_min = min(box["y_lb"] for box in boxes)
    y_max = max(box["y_ub"] for box in boxes)

    z_min = min(box["z_lb"] for box in boxes)
    z_max = max(box["z_ub"] for box in boxes)

    x_mid = 0.5 * (x_min + x_max)
    y_mid = 0.5 * (y_min + y_max)
    z_mid = 0.5 * (z_min + z_max)

    radius = 0.5 * max(
        x_max - x_min,
        y_max - y_min,
        z_max - z_min,
    )

    x_lim = (x_mid - radius, x_mid + radius)
    y_lim = (y_mid - radius, y_mid + radius)
    z_lim = (z_mid - radius, z_mid + radius)

    return x_lim, y_lim, z_lim


def draw_box(
    ax,
    x0,
    x1,
    y0,
    y1,
    z0,
    z1,
    facecolor,
    edgecolor="black",
    alpha=0.35,
    linewidth=0.15,
):
    verts = [
        [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],
        [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],
        [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],
        [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)],
        [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)],
        [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],
    ]

    poly = Poly3DCollection(
        verts,
        facecolor=facecolor,
        edgecolor=edgecolor,
        alpha=alpha,
        linewidth=linewidth,
    )

    ax.add_collection3d(poly)


def setup_axis(ax, x_lim, y_lim, z_lim, title):
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.set_zlim(*z_lim)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    ax.set_title(title)

    ax.view_init(elev=25, azim=-60)


# =========================
# Shared axis limits
# =========================
x_lim, y_lim, z_lim = get_global_axis_limits(verified_boxes)


# =========================
# Cell size colormap
# =========================
cell_sizes = np.asarray(
    [get_box_max_size(box) for box in verified_boxes],
    dtype=np.float64,
)
cell_sizes = np.round(cell_sizes / 1e-5) * 1e-5

if np.isclose(cell_sizes.min(), cell_sizes.max()):
    size_norm = Normalize(
        vmin=cell_sizes.min() - 1e-12,
        vmax=cell_sizes.max() + 1e-12,
    )
else:
    size_norm = Normalize(
        vmin=cell_sizes.min(),
        vmax=cell_sizes.max(),
    )

size_cmap = cm.get_cmap("viridis")


# =========================
# Create one figure with two subfigures
# =========================
fig = plt.figure(figsize=(18, 8))

ax_size = fig.add_subplot(1, 2, 1, projection="3d")
ax_verify = fig.add_subplot(1, 2, 2, projection="3d")


# =========================
# Subfigure 1:
# color by cell size
# =========================
for box, size in zip(verified_boxes, cell_sizes):
    color = size_cmap(size_norm(size))

    draw_box(
        ax_size,
        box["x_lb"],
        box["x_ub"],
        box["y_lb"],
        box["y_ub"],
        box["z_lb"],
        box["z_ub"],
        facecolor=color,
        edgecolor="black",
        alpha=0.45,
        linewidth=0.2,
    )

ax_size.scatter(*target, c="red", s=70, marker="*", label="Target")
ax_size.scatter(*gate, c="black", s=70, marker="*", label="Gate")

setup_axis(
    ax_size,
    x_lim,
    y_lim,
    z_lim,
    title=f"{cfg.scene_name}: Pose Cell Partition Colored by Cell Size",
)

ax_size.legend()


# =========================
# Subfigure 2:
# color by verified / unverified
# =========================
for box in verified_boxes:
    if not box["verified"]:
        color = "red"
    elif box.get("threshold_verified", False):
        color = "lightgreen"
    else:
        color = "darkgreen"

    draw_box(
        ax_verify,
        box["x_lb"],
        box["x_ub"],
        box["y_lb"],
        box["y_ub"],
        box["z_lb"],
        box["z_ub"],
        facecolor=color,
        edgecolor="black",
        alpha=0.35,
        linewidth=0.2,
    )

ax_verify.scatter([], [], [], c="darkgreen", s=70, marker="s", label="Verified: V decreases")
ax_verify.scatter([], [], [], c="lightgreen", s=70, marker="s", label="Verified: V <= threshold")
ax_verify.scatter([], [], [], c="red", s=70, marker="s", label="Failed")
ax_verify.scatter(*target, c="red", s=70, marker="*", label="Target")
ax_verify.scatter(*gate, c="black", s=70, marker="*", label="Gate")

setup_axis(
    ax_verify,
    x_lim,
    y_lim,
    z_lim,
    title=f"{cfg.scene_name}: Verification Reason",
)

ax_verify.legend()


# =========================
# Shared colorbar for size plot
# =========================
sm = cm.ScalarMappable(
    norm=size_norm,
    cmap=size_cmap,
)

sm.set_array([])

cbar = fig.colorbar(
    sm,
    ax=ax_size,
    shrink=0.65,
    pad=0.08,
)

cbar.set_label("max(dx, dy, dz)")


# =========================
# Save and show
# =========================
plt.tight_layout()

plt.savefig(cfg.figure_filename, dpi=300, bbox_inches="tight")

print(f"Saved figure to: {cfg.figure_filename}")

plt.show()
