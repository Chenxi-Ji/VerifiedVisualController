"""Rollout videos for the meansub-att 58k variant — EXACT mirror of
test_ctrl_lya_pt.py's run_test (same Config values, same init-pose
sampler, same 3-panel figure: 3D trajectory + Lyapunov curve + rendered
view, same suptitle/labels/fps/bitrate/naming pattern, videos/ dir).
The ONLY divergences are what the variant's head demands: ctrl(img)
returns attitude+thrust, integrated by QuadAttitudeDynamics at nominal
plant params instead of the original velocity-head Euler step.

Run:  SIDE_MEM_FRAC=0.25 python scripts_control/videos_ctrl_meansub_att.py
Out:  videos/rollout_meansub_att_{00..04}.mp4
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import numpy as np
import torch
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_image import load_gsplat_scene, render  # noqa: E402
from test_ctrl_lya_pt import (Config, sample_init_poses,  # noqa: E402
                              draw_frame)
from utils_ctrl_lya_pt import Lyapunov  # noqa: E402
from utils_ctrl_meansub_att import (ControllerMeansubAtt,  # noqa: E402
                                    QuadAttitudeDynamics, DynParams,
                                    quat_from_euler_zyx)
from eval_ctrl_meansub_att import METERS_PER_UNIT  # noqa: E402


def run_test(ctrl, Vnet, scene, target, gate, render_fn,
             device, dt=0.1, H=30, sample_num=5, video_dir="videos"):

    ctrl.to(device).eval()
    Vnet.to(device).eval()

    os.makedirs(video_dir, exist_ok=True)

    target_t = torch.tensor(target, device=device, dtype=torch.float32).unsqueeze(0)
    init_poses = sample_init_poses(target, n=sample_num)
    dyn = QuadAttitudeDynamics(dt_ctrl=dt, n_sub=20)
    params = DynParams.nominal(1, device)

    for idx, init_pose in enumerate(init_poses):

        pose = torch.tensor(init_pose, device=device, dtype=torch.float32).unsqueeze(0)
        q0 = quat_from_euler_zyx(pose[:, 3], pose[:, 4], pose[:, 5])
        s = dyn.make_state(pose[:, :3] * METERS_PER_UNIT,
                           torch.zeros(1, 3, device=device), q0,
                           torch.zeros(1, 3, device=device), params)

        traj, V_list = [], []

        # =========================
        # FIGURE SETUP (identical to test_ctrl_lya_pt)
        # =========================
        fig = plt.figure(figsize=(18, 5))

        ax_traj = fig.add_subplot(1, 3, 1, projection='3d')
        ax_V    = fig.add_subplot(1, 3, 2)
        ax_img  = fig.add_subplot(1, 3, 3)

        traj_line, = ax_traj.plot([], [], [], 'b-', linewidth=2)

        ax_traj.scatter(*target[:3], c='red', s=30, marker='*', label='Target')
        ax_traj.scatter(*gate[:3], c='black', s=30, marker='*', label='Gate')
        ax_traj.set_xlabel('X')
        ax_traj.set_ylabel('Y')
        ax_traj.set_zlabel('Z')
        ax_traj.set_title("Trajectory")
        ax_traj.legend()

        ax_V.set_title("Lyapunov Function")
        ax_V.set_xlabel("Time Step")
        ax_V.set_ylabel("V(x)")
        ax_V.grid(True, alpha=0.3)

        V_line, = ax_V.plot([], [], 'g-', linewidth=2)

        video_path = os.path.join(video_dir, f"rollout_meansub_att_{idx:02d}.mp4")
        writer = FFMpegWriter(fps=3, metadata=dict(artist='Controller'), bitrate=1500)

        frame_count = 0

        with torch.no_grad():
            with writer.saving(fig, video_path, dpi=100):

                for t in range(H):

                    # =========================
                    # render
                    # =========================
                    img = render_fn(
                        pose[0].detach().cpu().numpy(),
                        scene,
                        device=device
                    )
                    img = img.unsqueeze(0)

                    # =========================
                    # control (variant: attitude head -> attitude plant)
                    # =========================
                    a = ctrl(img)
                    s = dyn.step(s, a, params)
                    pose = dyn.render_pose(s)

                    img_np = img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()

                    # =========================
                    # Lyapunov
                    # =========================
                    V, _ = Vnet(pose, target_t.expand_as(pose))

                    # =========================
                    # record
                    # =========================
                    p = pose[0].cpu().numpy()
                    traj.append(p)
                    V_list.append(V.item())

                    traj_np = np.array(traj)

                    # =========================
                    # update TRAJ
                    # =========================
                    traj_line.set_data(traj_np[:, 0], traj_np[:, 1])
                    traj_line.set_3d_properties(traj_np[:, 2])

                    ax_traj.set_xlim(target[0] - 1.0, target[0] + 1.0)
                    ax_traj.set_ylim(target[1] - 2.0, target[1] + 2.0)
                    ax_traj.set_zlim(target[2] - 1.0, target[2] + 1.0)

                    for c in ax_traj.collections[:]:
                        c.remove()

                    ax_traj.scatter(*target[:3], c='red', s=30, marker='*', label='Target')
                    ax_traj.scatter(*gate[:3], c='black', s=30, marker='*', label='Gate')

                    draw_frame(ax_traj, p[:3], p[3:], scale=0.2)

                    # =========================
                    # update IMAGE
                    # =========================
                    ax_img.clear()
                    ax_img.imshow(img_np)
                    ax_img.set_title("Rendered View")
                    ax_img.axis("off")

                    # =========================
                    # update LYAPUNOV
                    # =========================
                    V_line.set_data(np.arange(len(V_list)), V_list)
                    ax_V.set_xlim(0, H - 1)
                    if len(V_list) > 0:
                        ax_V.set_ylim(0, max(V_list) * 1.2 + 0.1)
                    ax_V.relim()
                    ax_V.autoscale_view()

                    info_text = f"Step: {t+1}/{H} | V: {V.item():.4f} | Pose: [{p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f},{p[3]:.2f}, {p[4]:.2f}, {p[5]:.2f}]"
                    fig.suptitle(info_text, fontsize=10)

                    plt.tight_layout()
                    plt.draw()

                    writer.grab_frame()
                    frame_count += 1

        plt.close(fig)

        print(f"[DONE] rollout {idx} - Video saved: {video_path} ({frame_count} frames)")


if __name__ == "__main__":
    cfg = Config()
    device = cfg.device
    if device == "cuda":
        torch.cuda.set_per_process_memory_fraction(
            float(os.environ.get("SIDE_MEM_FRAC", "0.25")), 0)

    scene = load_gsplat_scene(cfg)

    ctrl = ControllerMeansubAtt().to(device)
    Vnet = Lyapunov().to(device)

    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(PROJECT_ROOT, "../", "weights/ctrl_lya_meansub_att.pt")
    ckpt = torch.load(save_path, map_location=device, weights_only=False)
    ctrl.load_state_dict(ckpt["controller"])
    Vnet.load_state_dict(ckpt["lyapunov"])

    np.random.seed(7)   # reproducible episode set

    run_test(
        ctrl=ctrl,
        Vnet=Vnet,
        scene=scene,
        target=cfg.target_pose,
        gate=cfg.gate_pose,
        render_fn=render,
        device=device,
        dt=cfg.dt,
        H=cfg.H,
        sample_num=cfg.sample_num,
        video_dir=cfg.video_dir
    )
