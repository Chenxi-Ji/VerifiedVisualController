#!/usr/bin/env python3
"""
Plot a REAL ctrl_lya flight the same way the sim test scripts do: trajectory +
Lyapunov V rollout video, plus trajectory-only / V-only figures.

Data sources
------------
1. OptiTrack ground truth (REQUIRED)  — ~/mocap_ws/record_flight.py CSV
   (raw Motive Z-up poses of the `starling2` rigid body, meters).
2. Drone logs (OPTIONAL)              — written by ctrl_lya_offboard.py to
   /tmp/ctrl_lya_logs/ on the VOXL:
     flight_*_ctrl.csv   actions + onboard V + commands (sent/held + events)
     flight_*_telem.csv  PX4 local-NED position/velocity (used to auto-align
                         the drone clock to the laptop clock via the speed
                         profile — frame-invariant, so the session-dependent
                         PX4 local frame never matters)

Everything is converted OFFLINE into the gate-centered frame the model was
trained in (+y through the gate toward the deploy side, z DOWN, scene units,
1 u = meters_per_unit meters), and V(t) is recomputed from the mocap pose with
the trained Lyapunov network — the onboard V is NOT trusted (the onboard
VIO->gate transform is still an identity placeholder).

One-time gate calibration (see record_flight.py docstring):
  python scripts_control/plot_flight.py --gate-from gate_center.csv gate_front.csv \\
      --save-gate gate_mocap.json

Typical use:
  python scripts_control/plot_flight.py --mocap mocap_20260701_130102.csv \\
      --gate gate_mocap.json \\
      --ctrl flight_20260701_170059_ctrl.csv --telem flight_20260701_170059_telem.csv \\
      --out flights/run1

Outputs (stem = --out):
  <stem>_rollout.mp4     animated: 3D trajectory | V(t) | actions   (like rollout_pt_XX.mp4)
  <stem>_traj.png        trajectory only: 3D + top-down over the V contour
  <stem>_V.png           V(t) (mocap-recomputed; pass --onboard-v to also overlay the
                         onboard V from the ctrl log, dashed)
  <stem>_actions.png     onboard actions/commands timeline (if --ctrl given)
  <stem>_gateframe.csv   t, x,y,z (scene u), yaw, V — for downstream analysis
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from scipy.spatial.transform import Rotation

# gate-centered frame constants (match train/test configs)
TARGET_POSE = np.array([0.0, 1.5, 0.0, -np.pi / 2, 0.0, 0.0])
GATE_POSE   = np.array([0.0, 0.0, 0.0, -np.pi / 2, 0.0, 0.0])
GATE_INNER_RADIUS_U = 0.44          # ring inner opening 0.88 u in scene units


# =============================================================================
# CSV LOADERS
# =============================================================================
def _read_csv(path):
    """Rows of a CSV, skipping '#' comment lines; returns (header, rows)."""
    with open(path, newline='') as f:
        rows = [r for r in csv.reader(f) if r and not r[0].startswith('#')]
    return rows[0], rows[1:]


def load_mocap(path, body):
    header, rows = _read_csv(path)
    col = {n: i for i, n in enumerate(header)}
    t, p, q = [], [], []
    names = set()
    for r in rows:
        names.add(r[col['name']])
        if r[col['name']] != body:
            continue
        t.append(float(r[col['t_wall']]))
        p.append([float(r[col[k]]) for k in ('x', 'y', 'z')])
        q.append([float(r[col[k]]) for k in ('qx', 'qy', 'qz', 'qw')])
    if not t:
        sys.exit(f"no rows for body '{body}' in {path} (bodies present: {sorted(names)})")
    t, p, q = np.array(t), np.array(p), np.array(q)
    order = np.argsort(t)
    return t[order], p[order], q[order]


def load_ctrl_ctbr(path):
    """pixel2ctbr flight CSV (ctbr_offboard.py) -> same dict shape as
    load_ctrl so downstream plotting works: action=[c m/s^2, wx, wy, wz],
    cmd=[roll_dps, pitch_dps, yaw_dps, thrust_norm]. Event rows have empty
    numeric fields and the note in the last column."""
    header, rows = _read_csv(path)
    col = {n: i for i, n in enumerate(header)}
    data = {k: [] for k in ('t', 'action', 'V', 'cmd', 'sent')}
    events = []
    for r in rows:
        note = r[col['event']] if len(r) > col['event'] else ''
        if r[col['c_ms2']] == '':
            if note:
                events.append((float(r[col['t_wall']]), note))
            continue
        data['t'].append(float(r[col['t_wall']]))
        data['action'].append([float(r[col[k]]) for k in ('c_ms2', 'wx', 'wy', 'wz')])
        data['V'].append(0.0)
        data['cmd'].append([float(r[col[k]]) for k in
                            ('roll_dps', 'pitch_dps', 'yaw_dps', 'thrust_norm')])
        data['sent'].append(0 if note == 'hover_hold' else 1)
    return {k: np.array(v) for k, v in data.items()}, events


def load_ctrl(path):
    """ctrl CSV -> dict of arrays + list of (t, note) event markers.
    Auto-detects the pixel2ctbr log format (header contains 'c_ms2')."""
    header, rows = _read_csv(path)
    if 'c_ms2' in header:
        return load_ctrl_ctbr(path)
    col = {n: i for i, n in enumerate(header)}
    data = {k: [] for k in ('t', 'action', 'V', 'cmd', 'sent')}
    events = []
    for r in rows:
        note = r[col['note']] if col.get('note') is not None and len(r) > col['note'] else ''
        if r[col['vx_u']] == '':                       # event marker row
            if note:
                events.append((float(r[col['t_wall']]), note))
            continue
        data['t'].append(float(r[col['t_wall']]))
        data['action'].append([float(r[col[k]]) for k in ('vx_u', 'vy_u', 'vz_u', 'vyaw_rad')])
        data['V'].append(float(r[col['V']]))
        data['cmd'].append([float(r[col[k]]) for k in ('fwd_mps', 'right_mps', 'down_mps', 'yaw_dps')])
        data['sent'].append(int(r[col['sent']]))
    return {k: np.array(v) for k, v in data.items()}, events


def load_telem(path):
    header, rows = _read_csv(path)
    col = {n: i for i, n in enumerate(header)}
    t, vel = [], []
    for r in rows:
        t.append(float(r[col['t_wall']]))
        vel.append([float(r[col[k]]) for k in ('vn_mps', 've_mps', 'vd_mps')])
    return np.array(t), np.array(vel)


# =============================================================================
# GATE FRAME
# =============================================================================
def gate_axes(center, front):
    """Gate-frame axes expressed in the mocap frame (Z-up).

    +y = horizontal direction gate->deploy side (from the two calibration
         points), +z = DOWN, +x = y × z (right-handed; matches 'x = right when
         facing the gate' of GATE_ARENA_SETUP.md).
    Returns R_gm (3x3): p_gate = R_gm @ (p_mocap - center).
    """
    d = np.asarray(front, float) - np.asarray(center, float)
    d[2] = 0.0                                   # through-gate axis is horizontal
    n = np.linalg.norm(d)
    if n < 0.3:
        sys.exit(f"gate calibration points are only {n:.2f} m apart horizontally "
                 "— re-record with the front point ~1 m in front of the gate")
    y = d / n
    z = np.array([0.0, 0.0, -1.0])               # mocap is Z-up -> gate z is DOWN
    x = np.cross(y, z)
    return np.stack([x, y, z])                   # rows


def mocap_to_gateframe(p_m, q_m, R_gm, center, mpu, body_rot_inv):
    """Positions (N,3) + xyzw quats (N,4) -> gate-frame pose6 (N,6), scene units.

    yaw is the heading of the body FRD +x axis about gate-frame z (down):
    forward_mocap = R(q) @ body_rot_inv @ e_x  (same convention as the
    mocap->PX4 bridge frames.py), then atan2 of its gate-frame xy components.
    pitch is filled for the heading arrow in plots; roll is not needed (the
    Lyapunov V is yaw-only) and set to 0.
    """
    pos_g = (R_gm @ (p_m - center).T).T / mpu
    fwd_body = body_rot_inv.apply(np.array([1.0, 0.0, 0.0]))
    fwd_m = Rotation.from_quat(q_m).apply(np.tile(fwd_body, (len(q_m), 1)))
    fwd_g = (R_gm @ fwd_m.T).T
    yaw = np.arctan2(fwd_g[:, 1], fwd_g[:, 0])
    pitch = np.arcsin(np.clip(-fwd_g[:, 2], -1.0, 1.0))
    pose6 = np.zeros((len(pos_g), 6))
    pose6[:, :3] = pos_g
    pose6[:, 3] = yaw
    pose6[:, 4] = pitch
    return pose6


def calibrate_gate(center_csv, front_csv, body, save_path):
    """Average two still captures into a gate.json; print a sanity summary."""
    out = {}
    for key, path in (('center_m', center_csv), ('front_m', front_csv)):
        t, p, _ = load_mocap(path, body)
        std = p.std(axis=0)
        if std.max() > 0.05:
            print(f"WARNING: {path} moves up to {std.max()*100:.1f} cm during the "
                  "capture — hold the drone still for calibration")
        out[key] = p.mean(axis=0).tolist()
    R_gm = gate_axes(out['center_m'], out['front_m'])
    out['note'] = ('gate-centered frame: +y through gate toward deploy side, '
                   'z down; p_gate = R_gm @ (p_mocap - center_m)')
    with open(save_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"gate center (mocap): {np.round(out['center_m'], 3).tolist()}")
    print(f"deploy-side point   : {np.round(out['front_m'], 3).tolist()}")
    print(f"gate +y axis (mocap): {np.round(R_gm[1], 3).tolist()}")
    print(f"saved -> {save_path}")


# =============================================================================
# LYAPUNOV V (offline recompute, trusted path)
# =============================================================================
def compute_V(pose6, weights):
    try:
        import torch
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from utils_ctrl_lya_pt import Lyapunov
        Vnet = Lyapunov()
        ckpt = torch.load(weights, map_location='cpu')
        Vnet.load_state_dict(ckpt['lyapunov'])
        Vnet.eval()
        with torch.no_grad():
            p = torch.tensor(pose6, dtype=torch.float32)
            tgt = torch.tensor(TARGET_POSE, dtype=torch.float32).expand_as(p)
            V, _ = Vnet(p, tgt)
        return V.numpy(), Vnet
    except Exception as e:
        print(f"WARNING: V recompute skipped ({type(e).__name__}: {e}) — "
              "trajectory plots will still be produced")
        return None, None


def v_contour_grid(Vnet, half_x=2.0, y_lo=-0.5, y_hi=3.5, res=90):
    """V over the x-y plane at target z/yaw (for the top-down underlay)."""
    import torch
    xs = np.linspace(-half_x, half_x, res)
    ys = np.linspace(y_lo, y_hi, res)
    XX, YY = np.meshgrid(xs, ys)
    poses = np.tile(TARGET_POSE, (XX.size, 1))
    poses[:, 0] = XX.ravel()
    poses[:, 1] = YY.ravel()
    with torch.no_grad():
        p = torch.tensor(poses, dtype=torch.float32)
        tgt = torch.tensor(TARGET_POSE, dtype=torch.float32).expand_as(p)
        V, _ = Vnet(p, tgt)
    return XX, YY, V.numpy().reshape(XX.shape)


# =============================================================================
# CLOCK ALIGNMENT (drone wall clock -> laptop wall clock)
# =============================================================================
def auto_align(t_mocap, p_mocap, t_telem, v_telem, max_lag_s=5.0, grid_hz=20.0):
    """Cross-correlate |velocity| profiles (frame-invariant) to find the clock
    offset such that t_drone + offset ≈ t_laptop. Returns offset seconds.

    Both CSVs stamp wall-clock epoch and the VOXL is NTP-synced, so the true
    offset is near zero — the search window is deliberately NARROW and a
    non-zero lag is accepted only when it clearly beats zero-lag. (A wide
    window on short, low-motion flights locks onto spurious peaks: a real
    flight came back +12 s mis-aligned before this gate existed.) If the
    drone clock is genuinely far off, pass --t-offset explicitly."""
    if len(t_telem) < grid_hz * 3 or len(t_mocap) < grid_hz * 3:
        return 0.0
    # mocap speed by finite difference (light smoothing)
    dt = np.gradient(t_mocap)
    dt[dt <= 0] = np.nan
    vm = np.linalg.norm(np.gradient(p_mocap, axis=0) / dt[:, None], axis=1)
    vm = np.nan_to_num(vm, nan=0.0)
    k = max(1, int(grid_hz * 0.25))
    vm = np.convolve(vm, np.ones(k) / k, mode='same')
    vt = np.linalg.norm(v_telem, axis=1)

    # common uniform grid spanning both, padded by the search window
    lo = min(t_mocap[0], t_telem[0]) - max_lag_s
    hi = max(t_mocap[-1], t_telem[-1]) + max_lag_s
    grid = np.arange(lo, hi, 1.0 / grid_hz)
    a = np.interp(grid, t_mocap, vm, left=0.0, right=0.0)
    b = np.interp(grid, t_telem, vt, left=0.0, right=0.0)
    if a.std() < 1e-6 or b.std() < 1e-6:
        return 0.0
    a = (a - a.mean()) / a.std()
    b = (b - b.mean()) / b.std()
    max_lag = int(max_lag_s * grid_hz)
    lags = np.arange(-max_lag, max_lag + 1)
    corr = np.array([np.dot(a[max(0, L):len(a) + min(0, L)],
                            b[max(0, -L):len(b) - max(0, L)]) / (len(a) - abs(L))
                     for L in lags])
    zero = corr[max_lag]                    # correlation at lag 0
    best = int(np.argmax(corr))
    offset = lags[best] / grid_hz           # drone + offset -> laptop time
    if corr[best] < 0.25 or corr[best] < zero + 0.10:
        print(f"clock auto-align: keeping 0.00s (peak {corr[best]:.2f} at "
              f"{offset:+.2f}s does not clearly beat zero-lag {zero:.2f}; "
              f"clocks are epoch-stamped — use --t-offset to force)")
        return 0.0
    print(f"clock auto-align: drone {offset:+.2f}s -> laptop  "
          f"(peak {corr[best]:.2f}, zero-lag {zero:.2f})")
    return offset


# =============================================================================
# PLOTS
# =============================================================================
def _draw_gate_topdown(ax):
    ax.plot([-GATE_INNER_RADIUS_U, GATE_INNER_RADIUS_U], [0, 0],
            color='k', lw=4, solid_capstyle='butt', label='Gate')
    ax.scatter(*TARGET_POSE[:2], c='red', s=120, marker='*', zorder=5, label='Target')


def _traj3d_panel(ax, pose6, title="Trajectory", line=True):
    if line:
        ax.plot(pose6[:, 0], pose6[:, 1], pose6[:, 2], 'b-', lw=1.5)
    ax.scatter(*TARGET_POSE[:3], c='red', s=60, marker='*', label='Target')
    ax.scatter(*GATE_POSE[:3], c='black', s=60, marker='*', label='Gate')
    ax.scatter(*pose6[0, :3], c='green', s=40, marker='o', label='Start')
    ax.scatter(*pose6[-1, :3], c='purple', s=40, marker='s', label='End')
    th = np.linspace(0, 2 * np.pi, 60)          # gate ring (x-z plane at y=0)
    ax.plot(GATE_INNER_RADIUS_U * np.cos(th), np.zeros_like(th),
            GATE_INNER_RADIUS_U * np.sin(th), 'k-', lw=2)
    ax.set_xlabel('X (u)')
    ax.set_ylabel('Y (u)')
    ax.set_zlabel('Z (u, down)')
    ax.invert_zaxis()                            # z is down -> plot up = up
    ax.set_title(title)
    ax.legend(fontsize=8)


def plot_static(stem, t, pose6, V, Vnet):
    # ---- trajectory-only figure: 3D + top-down (over V contour) ----
    fig = plt.figure(figsize=(13, 6))
    ax3 = fig.add_subplot(1, 2, 1, projection='3d')
    _traj3d_panel(ax3, pose6)
    ax2 = fig.add_subplot(1, 2, 2)
    if Vnet is not None:
        half_x = max(2.0, np.abs(pose6[:, 0]).max() + 0.5)
        y_lo = min(-0.5, pose6[:, 1].min() - 0.5)
        y_hi = max(3.5, pose6[:, 1].max() + 0.5)
        XX, YY, VV = v_contour_grid(Vnet, half_x, y_lo, y_hi)
        cs = ax2.contourf(XX, YY, VV, levels=20, alpha=0.55, cmap='viridis')
        ax2.contour(XX, YY, VV, levels=20, colors='k', linewidths=0.3, alpha=0.4)
        plt.colorbar(cs, ax=ax2, label='V (at target z, yaw)')
    ax2.plot(pose6[:, 0], pose6[:, 1], 'w-' if Vnet is not None else 'b-', lw=2)
    ax2.plot(pose6[:, 0], pose6[:, 1], 'b-', lw=1)
    ax2.scatter(pose6[0, 0], pose6[0, 1], c='green', s=50, marker='o', zorder=5, label='Start')
    ax2.scatter(pose6[-1, 0], pose6[-1, 1], c='purple', s=50, marker='s', zorder=5, label='End')
    _draw_gate_topdown(ax2)
    ax2.set_xlabel('X (u)')
    ax2.set_ylabel('Y (u)   (deploy side = +y)')
    ax2.set_title('Top-down (gate frame)')
    ax2.set_aspect('equal')
    ax2.legend(fontsize=8, loc='upper left')
    fig.suptitle(f"{os.path.basename(stem)} — {t[-1] - t[0]:.1f}s flight, gate-centered frame")
    fig.tight_layout()
    fig.savefig(stem + "_traj.png", dpi=140)
    plt.close(fig)
    print(f"saved {stem}_traj.png")


def plot_V(stem, t, V, ctrl=None, t_ctrl=None, events=None, show_onboard=False):
    fig, ax = plt.subplots(figsize=(11, 4))
    t0 = t[0]
    ax.plot(t - t0, V, 'g-', lw=1.8, label='V (mocap pose, recomputed)')
    if show_onboard and ctrl is not None and len(ctrl['V']):
        ax.plot(t_ctrl - t0, ctrl['V'], 'r--', lw=1.0, alpha=0.7,
                label='V onboard (uncalibrated transform)')
    for te, note in (events or []):
        if t[0] - 5 < te < t[-1] + 5:
            ax.axvline(te - t0, color='0.4', ls=':', lw=1)
            ax.text(te - t0, ax.get_ylim()[1] * 0.95, note, rotation=90,
                    fontsize=7, va='top', ha='right')
    ax.set_xlabel('t (s)')
    ax.set_ylabel('V(x)')
    ax.set_title('Lyapunov function along the real trajectory')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(stem + "_V.png", dpi=140)
    plt.close(fig)
    print(f"saved {stem}_V.png")


def plot_actions_ctbr(stem, t0, ctrl, t_ctrl, events):
    """CTBR variant: thrust (m/s^2) + body rates (rad/s) / sent commands."""
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    axes[0].plot(t_ctrl - t0, ctrl['action'][:, 0], lw=1.2, color='tab:red',
                 label='c (m/s²)')
    axes[0].axhline(9.81, color='0.5', lw=0.8, ls='--', label='hover g')
    axes[0].set_ylabel('collective thrust (m/s²)')
    axes[0].set_title('Model CTBR action — thrust')
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    for j, lab in enumerate(['ωx (roll)', 'ωy (pitch)', 'ωz (yaw)']):
        axes[1].plot(t_ctrl - t0, ctrl['action'][:, 1 + j], lw=1.1, label=lab)
    held = ctrl['sent'] == 0
    if held.any():
        axes[1].scatter((t_ctrl - t0)[held], np.zeros(held.sum()), marker='x',
                        c='red', s=18, label='hover-hold')
    axes[1].axhline(0, color='0.5', lw=0.8)
    axes[1].set_ylabel('body rates (rad/s)')
    axes[1].set_title('Model CTBR action — rates')
    axes[1].legend(fontsize=8, ncol=4)
    axes[1].grid(alpha=0.3)
    for j, lab in enumerate(['roll °/s', 'pitch °/s', 'yaw °/s']):
        axes[2].plot(t_ctrl - t0, ctrl['cmd'][:, j], lw=1.0, label=lab)
    ax2 = axes[2].twinx()
    ax2.plot(t_ctrl - t0, ctrl['cmd'][:, 3], lw=1.0, color='tab:red',
             label='thrust01')
    ax2.set_ylabel('thrust (norm)', color='tab:red')
    axes[2].set_ylabel('sent rates (°/s)')
    axes[2].set_xlabel('t (s)')
    axes[2].set_title('Commands sent to PX4 (set_attitude_rate)')
    axes[2].legend(fontsize=8, ncol=3, loc='upper left')
    axes[2].grid(alpha=0.3)
    for ax in axes:
        for te, note in events:
            ax.axvline(te - t0, color='0.4', ls=':', lw=1)
    for te, note in events:
        axes[0].text(te - t0, axes[0].get_ylim()[1], note, rotation=90,
                     fontsize=7, va='top')
    fig.tight_layout()
    fig.savefig(stem + "_actions.png", dpi=140)
    plt.close(fig)
    print(f"saved {stem}_actions.png (CTBR)")


def plot_actions(stem, t0, ctrl, t_ctrl, events):
    # pixel2ctbr logs: thrust scale makes the velocity plot meaningless —
    # detect by action[0] living around g rather than inside ±1 u/s
    if len(ctrl['action']) and np.median(np.abs(ctrl['action'][:, 0])) > 3.0:
        return plot_actions_ctbr(stem, t0, ctrl, t_ctrl, events)
    labels = ['vx (fwd+)', 'vy (right+)', 'vz (down+)', 'yaw rate']
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for j in range(4):
        axes[0].plot(t_ctrl - t0, ctrl['action'][:, j], lw=1.1, label=labels[j])
    held = ctrl['sent'] == 0
    if held.any():
        axes[0].scatter((t_ctrl - t0)[held], np.zeros(held.sum()), marker='x',
                        c='red', s=18, label='held (not sent)')
    axes[0].axhline(0, color='0.5', lw=0.8)
    axes[0].set_ylabel('action (scene u/s, rad/s)')
    axes[0].set_ylim(-1.15, 1.15)
    axes[0].set_title('Model actions')
    axes[0].legend(fontsize=8, ncol=5)
    axes[0].grid(alpha=0.3)
    for j, lab in enumerate(['fwd m/s', 'right m/s', 'down m/s', 'yaw °/s']):
        axes[1].plot(t_ctrl - t0, ctrl['cmd'][:, j], lw=1.1, label=lab)
    axes[1].axhline(0, color='0.5', lw=0.8)
    axes[1].set_ylabel('commanded')
    axes[1].set_xlabel('t (s)')
    axes[1].set_title('Commands sent to PX4')
    axes[1].legend(fontsize=8, ncol=4)
    axes[1].grid(alpha=0.3)
    for ax in axes:
        for te, note in events:
            ax.axvline(te - t0, color='0.4', ls=':', lw=1)
    if events:
        for te, note in events:
            axes[0].text(te - t0, 1.05, note, rotation=90, fontsize=7, va='bottom')
    fig.tight_layout()
    fig.savefig(stem + "_actions.png", dpi=140)
    plt.close(fig)
    print(f"saved {stem}_actions.png")


def render_video(stem, t, pose6, V, ctrl, t_ctrl, fps, show_onboard=False):
    """Animated rollout in the test_ctrl_lya_pt.py style:
    3D trajectory | V(t) | actions (actions panel only when ctrl log given)."""
    n_panels = 3 if ctrl is not None else 2
    fig = plt.figure(figsize=(6 * n_panels, 5))
    ax_traj = fig.add_subplot(1, n_panels, 1, projection='3d')
    ax_V = fig.add_subplot(1, n_panels, 2)
    ax_act = fig.add_subplot(1, n_panels, 3) if ctrl is not None else None

    t0, t1 = t[0], t[-1]
    frames_t = np.arange(t0, t1, 1.0 / fps)

    # static dressing (markers + gate ring; the trajectory line grows per frame)
    _traj3d_panel(ax_traj, pose6, line=False)
    line3d, = ax_traj.plot([], [], [], 'b-', lw=1.8)
    ax_traj.set_xlim(min(-1.5, pose6[:, 0].min() - 0.3), max(1.5, pose6[:, 0].max() + 0.3))
    ax_traj.set_ylim(min(-0.5, pose6[:, 1].min() - 0.3), max(3.0, pose6[:, 1].max() + 0.3))
    zlo = min(-1.0, pose6[:, 2].min() - 0.3)
    zhi = max(1.0, pose6[:, 2].max() + 0.3)
    ax_traj.set_zlim(zhi, zlo)                       # inverted (z down)

    ax_V.set_xlabel('t (s)')
    ax_V.set_ylabel('V(x)')
    ax_V.set_title('Lyapunov Function')
    ax_V.grid(alpha=0.3)
    ax_V.set_xlim(0, t1 - t0)
    vmax = (np.nanmax(V) if V is not None else 1.0)
    ax_V.set_ylim(0, vmax * 1.15 + 0.1)
    lineV, = ax_V.plot([], [], 'g-', lw=1.8)
    lineVo = None
    if show_onboard and ctrl is not None and len(ctrl['V']):
        lineVo, = ax_V.plot([], [], 'r--', lw=0.9, alpha=0.7, label='onboard')
        ax_V.legend(fontsize=8, loc='upper right')

    act_lines = []
    if ax_act is not None:
        for lab in ['vx', 'vy', 'vz', 'yaw']:
            ln, = ax_act.plot([], [], lw=1.1, label=lab)
            act_lines.append(ln)
        ax_act.axhline(0, color='0.5', lw=0.8)
        ax_act.set_xlim(0, t1 - t0)
        ax_act.set_ylim(-1.15, 1.15)
        ax_act.set_xlabel('t (s)')
        ax_act.set_title('Actions (scene u/s)')
        ax_act.legend(fontsize=8, ncol=4)
        ax_act.grid(alpha=0.3)

    # yuv420p + faststart: playable in every player / PowerPoint / browsers
    writer = FFMpegWriter(fps=fps, metadata=dict(artist='plot_flight'), bitrate=2500,
                          extra_args=['-pix_fmt', 'yuv420p', '-movflags', '+faststart'])
    quiv = [None]
    with writer.saving(fig, stem + "_rollout.mp4", dpi=100):
        for ft in frames_t:
            k = np.searchsorted(t, ft, side='right')
            if k < 2:
                writer.grab_frame()
                continue
            line3d.set_data(pose6[:k, 0], pose6[:k, 1])
            line3d.set_3d_properties(pose6[:k, 2])
            # heading arrow at the current pose
            if quiv[0] is not None:
                quiv[0].remove()
            p = pose6[k - 1]
            R = Rotation.from_euler("ZYX", (p[3], p[4], p[5])).as_matrix()
            fwd = R @ np.array([1.0, 0.0, 0.0]) * 0.35
            quiv[0] = ax_traj.quiver(p[0], p[1], p[2], fwd[0], fwd[1], fwd[2],
                                     color='g', linewidth=1.5, arrow_length_ratio=0.3)
            if V is not None:
                lineV.set_data(t[:k] - t0, V[:k])
            if lineVo is not None:
                kc = np.searchsorted(t_ctrl, ft, side='right')
                lineVo.set_data(t_ctrl[:kc] - t0, ctrl['V'][:kc])
            if ax_act is not None:
                kc = np.searchsorted(t_ctrl, ft, side='right')
                for j, ln in enumerate(act_lines):
                    ln.set_data(t_ctrl[:kc] - t0, ctrl['action'][:kc, j])
            vtxt = f"{V[k-1]:.3f}" if V is not None else "n/a"
            fig.suptitle(f"t={ft - t0:5.1f}s | V={vtxt} | pose=[{p[0]:+.2f}, {p[1]:+.2f}, "
                         f"{p[2]:+.2f}, yaw {p[3]:+.2f}]", fontsize=11)
            writer.grab_frame()
    plt.close(fig)
    print(f"saved {stem}_rollout.mp4  ({len(frames_t)} frames @ {fps} fps)")


# =============================================================================
# MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mocap', help='mocap CSV from record_flight.py')
    ap.add_argument('--gate', help='gate.json from --gate-from calibration')
    ap.add_argument('--gate-from', nargs=2, metavar=('CENTER_CSV', 'FRONT_CSV'),
                    help='calibrate the gate frame from two still captures and exit')
    ap.add_argument('--save-gate', default='gate_mocap.json')
    ap.add_argument('--ctrl', help='flight_*_ctrl.csv from ctrl_lya_offboard.py')
    ap.add_argument('--telem', help='flight_*_telem.csv (enables clock auto-align)')
    ap.add_argument('--body', default='starling2')
    ap.add_argument('--body-rot', default='180,0,180',
                    help='mocap rigid-body -> FRD rotation, deg — MUST match the bridge\'s '
                         'body_rotation_rpy_deg (default matches the asset re-created '
                         '2026-07-01, nose along -X; the pre-2026-07-01 asset was 180,0,-90)')
    ap.add_argument('--weights', default=None,
                    help='ctrl_lya .pt for the Lyapunov net (default <repo>/weights/ctrl_lya.pt)')
    ap.add_argument('--meters-per-unit', type=float, default=0.85)
    ap.add_argument('--t0', type=float, default=None, help='trim start (s from file start)')
    ap.add_argument('--t1', type=float, default=None, help='trim end (s from file start)')
    ap.add_argument('--no-auto-trim', action='store_true',
                    help='keep the full recording instead of auto-trimming to the '
                         'offboard_start -> pilot_takeover window from the ctrl log')
    ap.add_argument('--t-offset', type=float, default=None,
                    help='manual drone-clock offset (s); overrides auto-align')
    ap.add_argument('--hz', type=float, default=30.0, help='resample rate for plots/V')
    ap.add_argument('--video-fps', type=float, default=20.0)
    ap.add_argument('--no-video', action='store_true')
    ap.add_argument('--onboard-v', action='store_true',
                    help='overlay the onboard V from the ctrl log (dashed) in the V plot '
                         'and video — off by default since the onboard VIO->gate transform '
                         'is still an uncalibrated placeholder')
    ap.add_argument('--out', default=None, help='output stem (default alongside mocap CSV)')
    args = ap.parse_args()

    if args.gate_from:
        calibrate_gate(args.gate_from[0], args.gate_from[1], args.body, args.save_gate)
        return

    if not args.mocap or not args.gate:
        ap.error('--mocap and --gate are required (or use --gate-from to calibrate)')

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    weights = args.weights or os.path.join(root, 'weights', 'ctrl_lya.pt')
    stem = args.out or os.path.splitext(args.mocap)[0]
    os.makedirs(os.path.dirname(os.path.abspath(stem)), exist_ok=True)

    # ---- load mocap (full file; the analysis window is chosen below) ----
    t, p, q = load_mocap(args.mocap, args.body)
    file_t0 = t[0]

    # ---- drone logs + clock alignment (on the full series) ----
    ctrl = events = None
    t_ctrl = None
    offset = 0.0
    if args.ctrl:
        ctrl, events = load_ctrl(args.ctrl)
        if args.t_offset is not None:
            offset = args.t_offset
            print(f"clock offset (manual): drone {offset:+.2f}s -> laptop")
        elif args.telem:
            t_tel, v_tel = load_telem(args.telem)
            offset = auto_align(t, p, t_tel, v_tel)
        t_ctrl = ctrl['t'] + offset
        events = [(te + offset, note) for te, note in events]
    events = events or []

    # ---- analysis window: explicit --t0/--t1 > event auto-trim > full file.
    # Auto-trim cuts the manual-flight tail (fly-to-start / pilot landing) out
    # of the figures: [offboard_start - 1 s, first end event]. The end margin
    # is ZERO on purpose — even 1.5 s of pilot-takeover motion wrecks the
    # end-pose/V stats.
    END_EVENTS = ('pilot_takeover', 'stale_handoff', 'hold_handoff',
                  'land', 'land_fallback', 'ctrl_c', 'mpa_reader_closed')
    w0, w1 = t[0], t[-1]
    if args.t0 is not None or args.t1 is not None:
        if args.t0 is not None:
            w0 = file_t0 + args.t0
        if args.t1 is not None:
            w1 = file_t0 + args.t1
    elif not args.no_auto_trim:
        starts = [te for te, n in events if n == 'offboard_start']
        if starts:
            ends = [te for te, n in events if n in END_EVENTS and te > starts[0]]
            w0 = max(w0, starts[0] - 1.0)
            if ends:
                w1 = min(w1, min(ends))
            print(f"auto-trim to offboard window: {w0 - file_t0:.1f}s -> "
                  f"{w1 - file_t0:.1f}s of the recording  (--no-auto-trim for full)")
    m = (t >= w0) & (t <= w1)
    t, p, q = t[m], p[m], q[m]
    if len(t) < 10:
        sys.exit('fewer than 10 mocap samples after trimming')

    if ctrl is not None:
        # keep only ctrl rows inside the analysis window (with a small margin)
        mc = (t_ctrl >= t[0] - 1.0) & (t_ctrl <= t[-1] + 1.0)
        for k in ('t', 'action', 'V', 'cmd', 'sent'):
            ctrl[k] = ctrl[k][mc]
        t_ctrl = t_ctrl[mc]
        if len(t_ctrl) == 0:
            print('WARNING: no ctrl rows overlap the mocap window — check clock '
                  'offset (--t-offset) or trim; dropping the actions panel')
            ctrl = None

    # ---- resample mocap to a uniform grid (240 Hz raw is overkill) ----
    grid = np.arange(t[0], t[-1], 1.0 / args.hz)
    p_g = np.stack([np.interp(grid, t, p[:, i]) for i in range(3)], axis=1)
    # nearest-sample quaternions (avoids slerp; fine at 240->30 Hz)
    idx = np.clip(np.searchsorted(t, grid), 0, len(t) - 1)
    q_g = q[idx]

    # ---- gate frame + V ----
    gate = json.load(open(args.gate))
    R_gm = gate_axes(gate['center_m'], gate['front_m'])
    body_rot_inv = Rotation.from_euler(
        'xyz', [float(v) for v in args.body_rot.split(',')], degrees=True).inv()
    pose6 = mocap_to_gateframe(p_g, q_g, R_gm, np.asarray(gate['center_m']),
                               args.meters_per_unit, body_rot_inv)
    V, Vnet = compute_V(pose6, weights)

    d_end = np.linalg.norm(pose6[-1, :3] - TARGET_POSE[:3])
    print(f"flight: {grid[-1] - grid[0]:.1f}s, {len(grid)} samples @ {args.hz:.0f} Hz")
    print(f"start (gate frame): {np.round(pose6[0, :4], 2).tolist()}  "
          f"end: {np.round(pose6[-1, :4], 2).tolist()}  |end-target|={d_end:.2f} u")
    if V is not None:
        print(f"V: start {V[0]:.3f}  min {V.min():.3f}  end {V[-1]:.3f}")

    # ---- outputs ----
    with open(stem + "_gateframe.csv", 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['t_s', 'x_u', 'y_u', 'z_u', 'yaw_rad', 'pitch_rad', 'V'])
        for i in range(len(grid)):
            w.writerow([f'{grid[i] - grid[0]:.4f}',
                        *[f'{pose6[i, j]:.5f}' for j in range(5)],
                        f'{V[i]:.5f}' if V is not None else ''])
    print(f"saved {stem}_gateframe.csv")

    plot_static(stem, grid, pose6, V, Vnet)
    if V is not None:
        plot_V(stem, grid, V, ctrl, t_ctrl, events, show_onboard=args.onboard_v)
    if ctrl is not None and len(t_ctrl):
        plot_actions(stem, grid[0], ctrl, t_ctrl, events)
    if not args.no_video:
        render_video(stem, grid, pose6, V, ctrl, t_ctrl, args.video_fps,
                     show_onboard=args.onboard_v)


if __name__ == '__main__':
    main()
