import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass
from scipy.spatial.transform import Rotation
import cv2
from matplotlib.animation import FFMpegWriter
import io

from render_image import render, render_batch, load_gsplat_scene
from utils_ctrl_lya_pt import Controller, Lyapunov, transform_drone_velocity_to_world_frame

# =============================
# CONFIG
# =============================
@dataclass
class Config:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dt = 0.1
    H = 30
    sample_num = 5

    target_pose = np.array([0.0, -3.0, -0.2, 1.57, 0.0, 0.0])
    gate_pose = np.array([0.0, -2.0, -0.2, 1.57, 0.0, 0.0])

    # gsplat path
    gsplat_path = "nerfstudio/outputs/uturn/splatfacto/2025-05-09_151825"
    checkpoint = "nerfstudio_models/step-000040005.ckpt"

    save_path = "weights/ctrl_lya.pt" #"ctrl_lya.pt"
    video_dir = "videos"

# =========================
# INIT POSES
# =========================
def sample_init_poses(target, n=10):
    return target + np.random.uniform(
        low=[-1.5, -1.2, -0.7, -0.5, -0.0, -0.0],
        high=[ 1.5,  1.2,  0.7,  0.5,  0.0,  0.0],
        size=(n, 6)
    )

def draw_frame(ax, pos, rpy, scale=0.1):
    """
    Draw camera viewing direction matching the rendering pipeline.
    """
    px, py, pz = pos
    yaw, pitch, roll = rpy
    
    # Step 1: ZYX Euler to rotation matrix (same as render function)
    R = Rotation.from_euler("ZYX", (yaw, pitch, roll)).as_matrix()
    
    # Step 2: Apply the same coordinate transformation as in render
    tmp = Rotation.from_euler('zyx', [-np.pi/2, np.pi/2, 0]).as_matrix()
    R = R @ tmp
    
    # Step 3: Apply the axis flips (matching view[0:3,1:3] *= -1)
    R[:, 1:3] *= -1
    
    # Step 4: Apply the coordinate swap (view = view[np.array([0,2,1,3]),:])
    # This swaps rows: X, Z, Y order
    R = R[np.array([0, 2, 1]), :]
    
    # Step 5: Negate Z (matching view[2,:] *= -1)
    R[2, :] *= -1
    
    # Step 6: Camera forward direction is along X axis
    forward = R[1,:]#-R[2, :]  # First row of transformed R matrix
    forward = forward / (np.linalg.norm(forward) + 1e-8)
    
    origin = np.asarray([px, py, pz])
    
    ax.quiver(
        origin[0], origin[1], origin[2],
        forward[0] * scale,
        forward[1] * scale,
        forward[2] * scale,
        color='g',
        linewidth=1.0,
        arrow_length_ratio=0.3
    )


def fig_to_image(fig):
    """Convert matplotlib figure to numpy image array"""
    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(
        fig.canvas.get_width_height()[::-1] + (3,)
    )
    return image_from_plot

# =========================
# TEST LOOP
# =========================
def run_test(ctrl, Vnet, scene, target, gate, render_fn,
             device, dt=0.1, H=10, sample_num=10, video_dir="test_videos"):

    ctrl.to(device).eval()
    Vnet.to(device).eval()

    # Create video directory
    os.makedirs(video_dir, exist_ok=True)

    plt.ion()

    target_t = torch.tensor(target, device=device, dtype=torch.float32).unsqueeze(0)
    init_poses = sample_init_poses(target, n=sample_num)

    for idx, init_pose in enumerate(init_poses):

        pose = torch.tensor(init_pose, device=device, dtype=torch.float32).unsqueeze(0)

        traj, V_list = [], []

        # =========================
        # FIGURE SETUP
        # =========================
        fig = plt.figure(figsize=(18, 5))

        # Horizontal layout: traj (left), lyapunov (middle), rendered image (right)
        ax_traj = fig.add_subplot(1, 3, 1, projection='3d')
        ax_V    = fig.add_subplot(1, 3, 2)
        ax_img  = fig.add_subplot(1, 3, 3)

        # trajectory line
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

        img_artist = None

        # Video writer setup
        video_path = os.path.join(video_dir, f"rollout_pt_{idx:02d}.mp4")
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
                    # control
                    # =========================
                    pred_self = ctrl(img)
                    pred= transform_drone_velocity_to_world_frame(pred_self)
                    zeros = torch.zeros(*pred.shape[:-1], 2, device=pred.device, dtype=pred.dtype)
                    pred = torch.cat([pred, zeros], dim=-1)
                    next_pose = pose + pred * dt
                    # print(next_pose)

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

                    # Set axis limits for better visualization
                    ax_traj.set_xlim(target[0] - 1.0, target[0] + 1.0)
                    ax_traj.set_ylim(target[1] - 2.0, target[1] + 2.0)
                    ax_traj.set_zlim(target[2] - 1.0, target[2] + 1.0)

                    # camera direction (single vector)
                    for c in ax_traj.collections[:]:
                        c.remove()

                    # Replot target every frame
                    ax_traj.scatter(*target[:3], c='red', s=30, marker='*',label='Target')
                    ax_traj.scatter(*gate[:3], c='black', s=30, marker='*',label='Gate')

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
                    ax_V.set_xlim(0, H - 1)  # Display entire x horizon
                    if len(V_list) > 0:
                        ax_V.set_ylim(0, max(V_list) * 1.2 + 0.1)
                    ax_V.relim()
                    ax_V.autoscale_view()

                    # Add current info text
                    info_text = f"Step: {t+1}/{H} | V: {V.item():.4f} | Pose: [{p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f},{p[3]:.2f}, {p[4]:.2f}, {p[5]:.2f}]"
                    fig.suptitle(info_text, fontsize=10)

                    # =========================
                    # refresh and save frame
                    # =========================
                    plt.tight_layout()
                    plt.draw()
                    
                    # Save frame to video
                    writer.grab_frame()
                    frame_count += 1

                    plt.pause(0.01)  # Brief pause for rendering

                    pose = next_pose

        plt.ioff()
        plt.close(fig)

        print(f"[DONE] rollout {idx} - Video saved: {video_path} ({frame_count} frames)")

if __name__ == "__main__":
    cfg = Config()
    device = cfg.device

    scene = load_gsplat_scene(cfg)

    ctrl = Controller().to(device)
    Vnet = Lyapunov().to(device)

    # load weights
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(PROJECT_ROOT, "../", cfg.save_path)
    ckpt = torch.load(save_path, map_location=device)
    ctrl.load_state_dict(ckpt["controller"])
    Vnet.load_state_dict(ckpt["lyapunov"])

    target = cfg.target_pose
    gate = cfg.gate_pose   

    run_test(
        ctrl=ctrl,
        Vnet=Vnet,
        scene=scene,
        target=target,
        gate=gate,
        render_fn=render,
        device=device,
        dt=cfg.dt,
        H=cfg.H,
        sample_num=cfg.sample_num,
        video_dir=cfg.video_dir
    )
