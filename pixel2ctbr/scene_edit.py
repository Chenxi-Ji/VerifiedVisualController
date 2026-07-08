"""Splat-scene editing: gate extraction + duplication (FalconGym-2.0 recipe).

FalconGym 2.0 (UIUC, edit_gsplat_api.py) mass-produces track variants by
box-selecting the gaussians of an object in a METRIC frame, copying the five
per-gaussian tensors, rotating means+quats, translating, and concatenating
back onto the scene (~ms per op, no retraining). This module is that recipe
adapted to OUR scene tuple (scripts_control/render_image.load_gsplat_scene):

  (means, quats, opacities, scales, colors, transform, scale, world_frame)

means/quats live in the RAW checkpoint space. The gate-centered frame (origin
= gate center, +y through the gate toward the deploy side, z down; METERS =
units * METERS_PER_UNIT) is reached through the composed dataparser+
world_frame `transform` (rigid, verified in __main__) and uniform `scale`:

  p_ckpt = scale * (transform @ [p_units, 1])[:3]        (render_image.py)

so gate-frame edits conjugate through that chain:
  points   : crop/duplicate via M = diag(scale,scale,scale,1) @ transform
  rotations: R_ckpt_delta = R_t @ R_gate @ R_t^T   (R_t = transform[:3,:3])
  quats    : q_new = q(R_ckpt_delta) (x) q_old     (Hamilton, w-x-y-z)
  log-scales/opacities/colors copy unchanged (rigid move, uniform scale).

Unlike FalconGym (whose 'world' frame needs a hardcoded axis-permutation
chain from their Aruco convention), our transform already IS the gate frame,
so no extra permutations appear anywhere.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts_control"))
from render_image import Config, load_gsplat_scene  # noqa: E402

from dynamics import quat_mul  # noqa: E402
from render_bridge import METERS_PER_UNIT  # noqa: E402

# Gate crop box in gate-frame METERS (xmin,xmax),(ymin,ymax),(zmin,zmax).
# Ring + frame occupy ~|x|,|z| < 0.7, |y| < 0.25. z DOWN: the outer wire
# hoop closes at z≈+0.72 (an earlier +0.60 cut sliced the lower hoop arc +
# mounting collar off every duplicate); the short stand legs run from the
# collar to the FLOOR MAT at z≈+0.855 (mocap: gate center 0.855 m above the
# floor — NOT 1.2 m as previously believed). z-max +0.82 keeps the full
# ring + collar + near-full legs (ending ~3 cm above the mat) and excludes
# the mat layer: at +0.86 copies provably drag mat/tape-line fragments
# along (pixel-diffed renders; see docs/pixel2ctbr/07_multigate_envs.md).
GATE_BOX = ((-0.72, 0.72), (-0.28, 0.28), (-0.72, 0.82))


# ------------------------------------------------------------- frame algebra
def _units_to_ckpt_mat(scene) -> torch.Tensor:
    """4x4 M: gate-frame SCENE-UNIT points -> ckpt space (see module doc)."""
    means, transform, scale = scene[0], scene[5], scene[6]
    T = torch.as_tensor(np.asarray(transform), dtype=means.dtype,
                        device=means.device)
    M = T.clone()
    M[:3] *= scale
    return M


def gate_frame_means_m(scene) -> torch.Tensor:
    """(N,3) gaussian centers in gate-frame METERS."""
    means = scene[0]
    Minv = torch.linalg.inv(_units_to_ckpt_mat(scene))
    p = means @ Minv[:3, :3].T + Minv[:3, 3]
    return p * METERS_PER_UNIT


def gate_pose(t_m, yaw: float = 0.0) -> torch.Tensor:
    """4x4 gate pose in gate-frame METERS: R_z(yaw) then translate. yaw
    rotates the gate normal +y -> (-sin yaw, cos yaw, 0); the matching drone
    transit yaw is -pi/2 + yaw (see env_multigate.py)."""
    c, s = math.cos(yaw), math.sin(yaw)
    T = torch.eye(4)
    T[0, 0], T[0, 1], T[1, 0], T[1, 1] = c, -s, s, c
    T[:3, 3] = torch.as_tensor(t_m, dtype=torch.float32)
    return T


# ------------------------------------------------------------------ edit ops
def extract_gate_gaussians(scene, box=GATE_BOX) -> torch.Tensor:
    """(N,) bool mask of gaussians inside the gate-frame-METERS box.
    FalconGym's world-space bbox segmentation, in our frame."""
    p = gate_frame_means_m(scene)
    (x0, x1), (y0, y1), (z0, z1) = box
    return ((p[:, 0] >= x0) & (p[:, 0] <= x1) &
            (p[:, 1] >= y0) & (p[:, 1] <= y1) &
            (p[:, 2] >= z0) & (p[:, 2] <= z1))


def duplicate_gate(scene, T_new_gate: torch.Tensor, box=GATE_BOX, mask=None):
    """Copy the gate's gaussians to a new gate-frame pose (4x4, METERS —
    build with gate_pose()). The original gate sits at the gate-frame ORIGIN,
    so the new ring center lands exactly at T_new_gate[:3,3].
    Returns (means, quats, opacities, scales, colors) of the copy, in raw
    ckpt/param space, ready for compose()."""
    means, quats, opac, scl, colors = scene[:5]
    dev, dt = means.device, means.dtype
    if mask is None:
        mask = extract_gate_gaussians(scene, box)
    Tn = T_new_gate.to(device=dev, dtype=dt)

    # means: ckpt -> gate m -> new pose -> ckpt (all rigid + uniform scale)
    M = _units_to_ckpt_mat(scene)
    Minv = torch.linalg.inv(M)
    p = (means[mask] @ Minv[:3, :3].T + Minv[:3, 3]) * METERS_PER_UNIT
    p = p @ Tn[:3, :3].T + Tn[:3, 3]
    p = p / METERS_PER_UNIT
    new_means = p @ M[:3, :3].T + M[:3, 3]

    # quats: conjugate the gate-frame rotation into ckpt space, left-multiply
    # (FalconGym rotate(): R_world2 = R @ R_world, done in their world frame)
    R_t = M[:3, :3] / scene[6]                 # rigid rotation of `transform`
    R_d = (R_t @ Tn[:3, :3] @ R_t.T).cpu().numpy()
    from scipy.spatial.transform import Rotation
    qx, qy, qz, qw = Rotation.from_matrix(R_d).as_quat()   # scipy: x,y,z,w
    q_d = torch.tensor([qw, qx, qy, qz], dtype=dt, device=dev)
    new_quats = quat_mul(q_d.expand_as(quats[mask]), quats[mask])

    return (new_means, new_quats, opac[mask].clone(), scl[mask].clone(),
            colors[mask].clone())


def compose(scene, *extras):
    """New scene tuple with extra gaussian groups appended (transform/scale/
    world_frame unchanged, so SplatRenderer consumes it as-is)."""
    means, quats, opac, scl, colors = scene[:5]
    for em, eq, eo, es, ec in extras:
        means = torch.cat((means, em))
        quats = torch.cat((quats, eq))
        opac = torch.cat((opac, eo))
        scl = torch.cat((scl, es))
        colors = torch.cat((colors, ec))
    return (means, quats, opac, scl, colors) + tuple(scene[5:])


def delete_gaussians(scene, mask):
    """Scene without the masked gaussians (verification aid)."""
    keep = ~mask
    return tuple(t[keep] for t in scene[:5]) + tuple(scene[5:])


def multi_gate_scene(gate_poses):
    """Load the pristine scene and append a gate copy per pose in
    gate_poses (list of 4x4 gate-frame-METERS, originals excluded)."""
    scene = load_gsplat_scene(Config())
    mask = extract_gate_gaussians(scene)
    extras = [duplicate_gate(scene, T, mask=mask) for T in gate_poses]
    return compose(scene, *extras)


# ------------------------------------------------------------- verification
if __name__ == "__main__":
    # numeric gate for the coordinate-space math (renders: spike_multigate.py)
    torch.manual_seed(0)
    scene = load_gsplat_scene(Config())
    means = scene[0]
    print(f"scene: {means.shape[0]:,} gaussians on {means.device}")

    M = _units_to_ckpt_mat(scene)
    R_t = (M[:3, :3] / scene[6])
    ortho = (R_t @ R_t.T - torch.eye(3, device=R_t.device)).abs().max().item()
    print(f"transform rigidity |R Rt - I|_max = {ortho:.2e}")
    assert ortho < 1e-4, "transform is not rigid — conjugation math invalid"

    # round trip gate-frame meters -> ckpt -> gate-frame meters
    pts = (torch.rand(1024, 3, device=means.device) - 0.5) * 4.0
    ck = (pts / METERS_PER_UNIT) @ M[:3, :3].T + M[:3, 3]
    Minv = torch.linalg.inv(M)
    back = ((ck @ Minv[:3, :3].T + Minv[:3, 3])) * METERS_PER_UNIT
    rt = (back - pts).abs().max().item()
    print(f"round-trip |err|_max = {rt:.2e} m")
    assert rt < 1e-4

    # identity duplicate must reproduce the source gaussians exactly
    mask = extract_gate_gaussians(scene)
    print(f"gate box {GATE_BOX} -> {int(mask.sum()):,} gaussians "
          f"({100 * mask.float().mean():.2f}% of scene)")
    dm, dq, *_ = duplicate_gate(scene, gate_pose((0, 0, 0), 0.0), mask=mask)
    em = (dm - means[mask]).abs().max().item()
    eq = (dq - scene[1][mask]).abs().max().item()
    print(f"identity-duplicate err: means {em:.2e} quats {eq:.2e}")
    assert em < 1e-4 and eq < 1e-4

    # duplicated copy lands where asked (centroid check, in gate-frame m)
    T2 = gate_pose((0.0, -2.2, 0.0), 0.0)
    two = compose(scene, duplicate_gate(scene, T2, mask=mask))
    p2 = gate_frame_means_m(two)
    c_src = p2[: means.shape[0]][mask].mean(0)
    c_new = p2[means.shape[0]:].mean(0)
    print(f"src centroid  {c_src.tolist()}")
    print(f"copy centroid {c_new.tolist()} (expect src + [0,-2.2,0])")
    d = (c_new - c_src - torch.tensor([0, -2.2, 0], device=c_new.device))
    assert d.abs().max().item() < 1e-3
    print("ALL PASS")
