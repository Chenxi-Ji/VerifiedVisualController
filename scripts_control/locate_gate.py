"""Locate the gate and build the gate-centered world frame for a scene.

This is the tool that produced world_frame.json for Gate_Long_hloc_seq.
To redo it for a new scene/splat:

  1. Run explore_scene.py, look at the real-frame contact sheets, and find
     frames where the ring gate is clearly visible from well-separated
     viewpoints (include at least one face-on view and one far view).
  2. Read the ring-center pixel (u, v) of each chosen frame off a
     grid-overlaid image (explore_scene saves images_4-scale frames:
     960x540 = full res / 4) and put them in MANUAL_CENTERS below.
  3. For one face-on frame, also read the ring's inner-opening top/bottom/
     left/right rim pixels into RIM_PIXELS (used to get the true vertical
     and the ring size).
  4. Run this script. It triangulates the gate center, derives the scene's
     true vertical from the ring rim (do NOT trust nerfstudio's up: with
     lots of downward-looking footage it can be way off), writes
     world_frame.json next to the checkpoint, and renders a verification
     grid - check that the ring is centered on the crosshair.

World frame convention (consumed by render_image.py when world_frame.json
exists): origin = gate center, +y = through the gate, z = down,
yaw = pi/2 faces the gate, pitch = roll = 0 is level flight.

Run from the repo root:
    python scripts_control/locate_gate.py
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt

from render_image import Config, load_gsplat_scene, render
from explore_scene import TRANSFORMS_JSON, pose_to_c2w, load_frames

OUT_DIR = "figures/explore"

# ring-center pixels (u, v) at images_4 scale (960x540), read manually
MANUAL_CENTERS = {
    "frame_00299.png": (380, 175),
    "frame_00317.png": (540, 250),
    "frame_00335.png": (530, 248),
    "frame_00341.png": (610, 240),
    "frame_01531.png": (508, 120),
    "frame_01543.png": (510, 115),
}

# inner-opening rim pixels of face-on frames: (center, top, bottom, left, right)
RIM_PIXELS = {
    "frame_00335.png": ((530, 248), (530, 85), (530, 410), (365, 248), (695, 248)),
    "frame_01531.png": ((508, 120), (508, 45), (508, 193), (435, 120), (582, 120)),
}

FACE_ON = "frame_00335.png"   # used for the through-gate direction


def triangulate(origins, dirs):
    """Least-squares point closest to all rays (o_i + t d_i)."""
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for o, d in zip(origins, dirs):
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ o
    return np.linalg.solve(A, b)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cfg = Config()
    poses, names, _ = load_frames()
    by_name = dict(zip(names, poses))
    with open(TRANSFORMS_JSON) as f:
        meta = json.load(f)
    # intrinsics at images_4 scale
    fx, fy, cx, cy = (meta["fl_x"] / 4, meta["fl_y"] / 4,
                      meta["cx"] / 4, meta["cy"] / 4)

    def ray(frame, u, v):
        """World-space ray through pixel (u, v) of a dataset frame."""
        c2w = pose_to_c2w(by_name[frame])
        d = c2w[:3, :3] @ np.array([(u - cx) / fx, -(v - cy) / fy, -1.0])
        return c2w[:3, 3], d / np.linalg.norm(d)

    # ---- 1. gate center: triangulate manual ring-center rays ----
    rays = [ray(n, u, v) for n, (u, v) in MANUAL_CENTERS.items()]
    gate = triangulate(*map(np.array, zip(*rays)))
    print("gate center:", gate.round(4))
    for (n, _), (o, d) in zip(MANUAL_CENTERS.items(), rays):
        res = np.linalg.norm((gate - o) - np.dot(gate - o, d) * d)
        print(f"  {n}: residual {res:.4f}  dist {np.dot(gate - o, d):.3f}")

    # ---- 2. true vertical from the ring rim (top-bottom is plumb) ----
    ups, diams = [], []
    for frame, (ctr, top, bot, left, right) in RIM_PIXELS.items():
        o, _ = ray(frame, *ctr)
        n_pl = gate - o
        n_pl /= np.linalg.norm(n_pl)          # gate plane ~ faces this camera

        def hit(px):
            oo, dd = ray(frame, *px)
            t = ((gate - oo) @ n_pl) / (dd @ n_pl)
            return oo + t * dd

        p_top, p_bot = hit(top), hit(bot)
        v = p_top - p_bot
        ups.append(v / np.linalg.norm(v))
        diams += [np.linalg.norm(p_top - p_bot),
                  np.linalg.norm(hit(right) - hit(left))]
        print(f"{frame}: vertical {ups[-1].round(4)}")
    print("up agreement (dot):", np.round(ups[0] @ ups[1], 4))
    up = np.mean(ups, axis=0)
    up /= np.linalg.norm(up)
    print("inner-opening diameters:", np.round(diams, 3),
          "-> mean", np.round(np.mean(diams), 3), "scene units")

    # ---- 3. through-gate direction (horizontal w.r.t. true up) ----
    o_face, _ = ray(FACE_ON, *RIM_PIXELS[FACE_ON][0])
    thru = gate - o_face
    thru /= np.linalg.norm(thru)
    thru -= (thru @ up) * up
    thru /= np.linalg.norm(thru)

    # ---- 4. build + save the world frame ----
    e_z = -up                    # z down
    e_y = thru                   # +y flies through the gate
    e_x = np.cross(e_y, e_z)     # right-handed
    W = np.eye(4)
    W[:3, 0], W[:3, 1], W[:3, 2], W[:3, 3] = e_x, e_y, e_z, gate
    out_json = os.path.join(cfg.gsplat_path, "world_frame.json")
    with open(out_json, "w") as f:
        json.dump({"world_transform": W.tolist(),
                   "comment": "gate-centered frame: origin=gate center, "
                              "+y=through gate, z=down; built by locate_gate.py"},
                  f, indent=1)
    print(f"\nwrote {out_json}")

    # ---- 5. verification renders in the world frame ----
    scene = load_gsplat_scene(cfg)   # picks up the world_frame.json just written
    cases = [(f"y={y}", [0, y, 0, np.pi / 2, 0, 0]) for y in (-2.5, -2.0, -1.5, -1.0)]
    cases += [("x=+0.8", [0.8, -1.5, 0, np.pi / 2, 0, 0]),
              ("x=-0.8", [-0.8, -1.5, 0, np.pi / 2, 0, 0]),
              ("z=-0.5 (higher)", [0, -1.5, -0.5, np.pi / 2, 0, 0]),
              ("yaw+0.4", [0, -1.5, 0, np.pi / 2 + 0.4, 0, 0])]
    fig, axes = plt.subplots(2, 4, figsize=(20, 7))
    for ax, (lbl, pose) in zip(axes.ravel(), cases):
        img = render(np.array(pose, float), scene, device=cfg.device)
        ax.imshow(img.permute(1, 2, 0).cpu().numpy())
        ax.plot([150], [100], 'r+', markersize=14)
        ax.set_title(lbl, fontsize=9)
        ax.axis('off')
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "gate_verification.png")
    plt.savefig(out, dpi=110)
    plt.close()
    print(f"wrote {out} - the ring should sit on the crosshair in every tile")


if __name__ == "__main__":
    main()
