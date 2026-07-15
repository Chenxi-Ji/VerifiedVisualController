import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# =========================
# Load data
# =========================
filename = "cert"
result_filename = f"results/{filename}_result.pt"
data = torch.load(result_filename, map_location="cpu")

verified_boxes = data["verified_boxes"]
target = np.asarray(data["target"][:3])
gate = np.asarray(data["gate"][:3])

os.makedirs("figures", exist_ok=True)


# =========================
# Filter only unverified boxes
# =========================
unverified_boxes = [
    box for box in verified_boxes
    if not box["verified"]
]

print(f"Total boxes: {len(verified_boxes)}")
print(f"Unverified boxes: {len(unverified_boxes)}")

if len(unverified_boxes) == 0:
    raise RuntimeError("No unverified boxes found. Nothing to plot.")


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

    if radius <= 0:
        radius = 1e-6

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
# Use all boxes to keep the same global spatial range
# If you want zoom-in view, replace verified_boxes with unverified_boxes
# =========================
x_lim, y_lim, z_lim = get_global_axis_limits(verified_boxes)


# =========================
# Cell size colormap
# Only normalize over unverified cells
# =========================
cell_sizes = np.asarray(
    [get_box_max_size(box) for box in unverified_boxes],
    dtype=np.float64,
)

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
# only unverified cells, colored by cell size
# =========================
for box, size in zip(unverified_boxes, cell_sizes):
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
        alpha=0.55,
        linewidth=0.25,
    )

ax_size.scatter(*target, c="red", s=70, marker="*", label="Target")
ax_size.scatter(*gate, c="black", s=70, marker="*", label="Gate")

setup_axis(
    ax_size,
    x_lim,
    y_lim,
    z_lim,
    title="Unverified Pose Cells: Colored by Cell Size",
)

ax_size.legend()


# =========================
# Subfigure 2:
# only unverified cells, all red
# =========================
for box in unverified_boxes:
    draw_box(
        ax_verify,
        box["x_lb"],
        box["x_ub"],
        box["y_lb"],
        box["y_ub"],
        box["z_lb"],
        box["z_ub"],
        facecolor="red",
        edgecolor="red",
        alpha=0.45,
        linewidth=0.25,
    )

ax_verify.scatter(*target, c="red", s=70, marker="*", label="Target")
ax_verify.scatter(*gate, c="black", s=70, marker="*", label="Gate")

setup_axis(
    ax_verify,
    x_lim,
    y_lim,
    z_lim,
    title="Unverified Pose Cells Only",
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

save_path = f"figures/{filename}_unverified_partition_only.png"
plt.savefig(save_path, dpi=300, bbox_inches="tight")

print(f"Saved figure to: {save_path}")

plt.show()