import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# =========================
# Load data
# =========================
filename = "cert_inner"
result_filename = f"results/{filename}_result.pt"
data = torch.load(result_filename, map_location="cpu")

verified_boxes = data["verified_boxes"]
target = data["target"]
gate = data["gate"]


# convert tensors if needed
target = target[:3]#.cpu().numpy()
gate = gate[:3]#.cpu().numpy()


# =========================
# draw cube helper
# =========================
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
        Poly3DCollection(verts, alpha=0.2, facecolor=color, edgecolor=color)
    )


# =========================
# plot
# =========================
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")


# plot verified boxes
for box in verified_boxes:
    color = "green" if box["verified"] else "red"

    draw_box(
        ax,
        box["x_lb"], box["x_ub"],
        box["y_lb"], box["y_ub"],
        box["z_lb"], box["z_ub"],
        color
    )


# plot target & gate
ax.scatter(*target, c='red', s=60, marker='*', label='Target')
ax.scatter(*gate, c='black', s=60, marker='*', label='Gate')


# labels
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title("Verified Region Visualization")
ax.legend()

plt.tight_layout()
plt.show()