"""Explore a gsplat scene to locate the gate and choose pose ranges.

Converts dataset camera poses (transforms.json) into the (x, y, z, yaw,
pitch, roll) convention used by render(), verifies the conversion is exact,
prints pose statistics, and renders contact sheets of trajectory frames so
the gate can be located visually.

Run from the repo root:
    python scripts_control/explore_scene.py
"""
import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from render_image import Config, load_gsplat_scene, render

TRANSFORMS_JSON = "nerfstudio/Gate_Long_hloc_seq_data/transforms.json"
OUT_DIR = "figures/explore"

# world-axis permutation render() applies: view = view[[0,2,1,3]]; view[2]*=-1
P4 = np.array([[1, 0, 0, 0],
               [0, 0, 1, 0],
               [0, -1, 0, 0],
               [0, 0, 0, 1]], dtype=np.float64)
D = np.diag([1.0, -1.0, -1.0])
TMP = Rotation.from_euler('zyx', [-np.pi / 2, np.pi / 2, 0]).as_matrix()


def pose_to_c2w(pose):
    """Reproduce render()'s view construction (before dataparser transform)."""
    px, py, pz, yaw, pitch, roll = pose
    view = np.eye(4)
    view[:3, :3] = Rotation.from_euler("ZYX", (yaw, pitch, roll)).as_matrix() @ TMP
    view[:3, 3] = [px, py, pz]
    view[0:3, 1:3] *= -1
    view = view[[0, 2, 1, 3], :]
    view[2, :] *= -1
    return view


def c2w_to_pose(c2w, applied_transform=None):
    """Recover (x, y, z, yaw, pitch, roll) from a transforms.json
    camera-to-world matrix.

    The checkpoint's dataparser transform maps from ORIGINAL colmap space
    (it has the dataset's applied_transform composed into it), while
    transforms.json matrices are in post-applied space — so the
    applied_transform must be undone before inverting render()'s view
    construction."""
    c2w = np.asarray(c2w, dtype=np.float64)
    if applied_transform is not None:
        Ta = np.vstack([np.asarray(applied_transform, dtype=np.float64),
                        [0, 0, 0, 1]])
        c2w = np.linalg.inv(Ta) @ c2w
    M = P4.T @ c2w
    t = M[:3, 3]
    R_zyx = M[:3, :3] @ D @ TMP.T
    yaw, pitch, roll = Rotation.from_matrix(R_zyx).as_euler("ZYX")
    return np.array([t[0], t[1], t[2], yaw, pitch, roll])


def camera_forward(pose):
    """Camera viewing direction in pose/world coordinates (unit vector)."""
    c2w = pose_to_c2w(pose)
    return -c2w[:3, 2]  # OpenGL convention: camera looks along -z


def load_frames():
    with open(TRANSFORMS_JSON) as f:
        meta = json.load(f)
    applied = meta.get("applied_transform")
    frames = sorted(meta["frames"], key=lambda fr: fr["file_path"])
    poses, names = [], []
    for fr in frames:
        c2w = np.array(fr["transform_matrix"])
        poses.append(c2w_to_pose(c2w, applied))
        names.append(os.path.basename(fr["file_path"]))
    return np.array(poses), names, frames


def verify_roundtrip(poses, frames, n=20):
    with open(TRANSFORMS_JSON) as f:
        applied = json.load(f).get("applied_transform")
    Ta = np.vstack([np.asarray(applied, dtype=np.float64), [0, 0, 0, 1]])
    idx = np.linspace(0, len(poses) - 1, n).astype(int)
    errs = []
    for i in idx:
        c2w = np.array(frames[i]["transform_matrix"])
        if c2w.shape[0] == 3:
            c2w = np.vstack([c2w, [0, 0, 0, 1]])
        errs.append(np.abs(Ta @ pose_to_c2w(poses[i]) - c2w).max())
    print(f"round-trip max error over {n} frames: {max(errs):.2e}")


def render_depth(pose, scene, device):
    """Render and return (rgb_img_hwc, center_depth_model_units)."""
    img = render(pose, scene, device=device)  # (3,H,W), rgb only
    return img.permute(1, 2, 0).cpu().numpy()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cfg = Config()
    # poses here are raw transforms.json camera poses -> bypass world_frame.json
    scene = load_gsplat_scene(cfg, use_world_frame=False)
    poses, names, frames = load_frames()

    verify_roundtrip(poses, frames)

    pos = poses[:, :3]
    print(f"\n{len(poses)} camera poses (data/world units)")
    print("position min :", pos.min(0).round(3))
    print("position max :", pos.max(0).round(3))
    print("yaw   range  :", poses[:, 3].min().round(3), "..", poses[:, 3].max().round(3))
    print("pitch range  :", poses[:, 4].min().round(3), "..", poses[:, 4].max().round(3))
    print("roll  range  :", poses[:, 5].min().round(3), "..", poses[:, 5].max().round(3))

    # ---- contact sheet of every Nth frame ----
    step = max(1, len(poses) // 36)
    sel = list(range(0, len(poses), step))[:36]
    fig, axes = plt.subplots(6, 6, figsize=(24, 16))
    for ax, i in zip(axes.ravel(), sel):
        img = render_depth(poses[i], scene, cfg.device)
        ax.imshow(img)
        p = poses[i]
        ax.set_title(f"#{i} {names[i]}\n[{p[0]:.2f},{p[1]:.2f},{p[2]:.2f}] "
                     f"y{p[3]:.2f} p{p[4]:.2f} r{p[5]:.2f}", fontsize=7)
        ax.axis('off')
    for ax in axes.ravel()[len(sel):]:
        ax.axis('off')
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "contact_sheet.png")
    plt.savefig(out, dpi=90)
    plt.close()
    print(f"\nwrote {out}")

    np.save(os.path.join(OUT_DIR, "trajectory_poses.npy"), poses)
    print(f"wrote {OUT_DIR}/trajectory_poses.npy")


if __name__ == "__main__":
    main()
