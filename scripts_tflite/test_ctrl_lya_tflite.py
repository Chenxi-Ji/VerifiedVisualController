"""TFLite rollout test — uses fused.tflite (controller + lyapunov) for inference.
gsplat rendering still runs on GPU via PyTorch.

Run in pftolite conda env:
    conda run -n pftolite python scripts_control/test_ctrl_lya_tflite.py
"""
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from dataclasses import dataclass
from scipy.spatial.transform import Rotation
from matplotlib.animation import FFMpegWriter

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)

# add project root (NOT script dir)
sys.path.insert(0, _PROJECT_ROOT)
from scripts_control.utils_ctrl_lya_pt import transform_drone_velocity_to_world_frame_np
from scripts_control.render_image import render, load_gsplat_scene

try:
    from ai_edge_litert.interpreter import Interpreter as TFLiteInterp
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter as TFLiteInterp
    except ImportError:
        import tensorflow as tf
        TFLiteInterp = tf.lite.Interpreter


# =============================
# CONFIG
# =============================
@dataclass
class Config:
    device       = "cuda" if torch.cuda.is_available() else "cpu"
    dt           = 0.1
    H            = 30
    sample_num   = 5
    target_pose  = np.array([0.0, -3.0, -0.2, 1.57, 0.0, 0.0])
    gate_pose    = np.array([0.0, -2.0, -0.2, 1.57, 0.0, 0.0])
    gsplat_path  = os.path.join(_PROJECT_ROOT, "nerfstudio/outputs/uturn/splatfacto/2025-05-09_151825")
    checkpoint   = "nerfstudio_models/step-000040005.ckpt"
    fused_tflite = os.path.join(_PROJECT_ROOT, "weights/ctrl_lya.tflite")
    video_dir    = os.path.join(_PROJECT_ROOT, "videos")


# =============================
# FUSED TFLITE MODEL
# =============================
class TFLiteFusedModel:
    """Runs controller + Lyapunov in a single TFLite inference call."""

    def __init__(self, model_path):
        self.interp = TFLiteInterp(model_path=model_path)
        self.interp.allocate_tensors()

        inp_details = self.interp.get_input_details()
        out_details = self.interp.get_output_details()

        # --- inputs: image (4-D), pose (2-D), target (2-D) ---
        self._img_det    = None
        self._pose_det   = None
        self._target_det = None
        unnamed_2d = []
        for det in inp_details:
            if len(det["shape"]) == 4:
                self._img_det = det
            else:
                n = det["name"].lower()
                if "target" in n:
                    self._target_det = det
                elif "pose" in n:
                    self._pose_det = det
                else:
                    unnamed_2d.append(det)
        # positional fallback for unlabelled 2-D inputs (order: pose, target)
        if self._pose_det is None and unnamed_2d:
            self._pose_det = unnamed_2d.pop(0)
        if self._target_det is None and unnamed_2d:
            self._target_det = unnamed_2d.pop(0)

        # image layout
        s = self._img_det["shape"]
        if s[1] == 3:           # NCHW
            self.h, self.w = int(s[2]), int(s[3])
            self.nhwc = False
        else:                   # NHWC
            self.h, self.w = int(s[1]), int(s[2])
            self.nhwc = True

        # --- outputs: action (1,4) vs V (1,) detected by shape ---
        self._action_det = None
        self._v_det      = None
        for det in out_details:
            if len(det["shape"]) >= 2 and det["shape"][-1] == 4:
                self._action_det = det
            else:
                self._v_det = det
        if self._action_det is None or self._v_det is None:
            self._action_det, self._v_det = out_details[0], out_details[1]

    def __call__(self, img_tensor, pose_np, target_np):
        """
        img_tensor  : (1, 3, H, W) float torch
        pose_np     : (1, 6) float32 numpy
        target_np   : (1, 6) float32 numpy
        Returns     : action (1, 4) float32, V scalar float
        """
        img = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
        img = cv2.resize(img, (self.w, self.h))
        x = (img[np.newaxis] if self.nhwc
             else img.transpose(2, 0, 1)[np.newaxis]).astype(np.float32)

        self.interp.set_tensor(self._img_det["index"],    x)
        self.interp.set_tensor(self._pose_det["index"],   pose_np.astype(np.float32))
        self.interp.set_tensor(self._target_det["index"], target_np.astype(np.float32))
        self.interp.invoke()

        action = self.interp.get_tensor(self._action_det["index"]).astype(np.float32)
        V      = float(self.interp.get_tensor(self._v_det["index"]).flat[0])
        return action, V


# =============================
# HELPERS
# =============================
def sample_init_poses(target, n=10):
    return target + np.random.uniform(
        low= [-1.5, -1.2, -0.7, -0.5, -0.0, -0.0],
        high=[ 1.5,  1.2,  0.7,  0.5,  0.0,  0.0],
        size=(n, 6),
    )


def draw_frame(ax, pos, rpy, scale=0.1):
    px, py, pz = pos
    yaw, pitch, roll = rpy
    R = Rotation.from_euler("ZYX", (yaw, pitch, roll)).as_matrix()
    tmp = Rotation.from_euler("zyx", [-np.pi/2, np.pi/2, 0]).as_matrix()
    R = R @ tmp
    R[:, 1:3] *= -1
    R = R[np.array([0, 2, 1]), :]
    R[2, :] *= -1
    forward = R[1, :] / (np.linalg.norm(R[1, :]) + 1e-8)
    origin = np.array([px, py, pz])
    ax.quiver(
        origin[0], origin[1], origin[2],
        forward[0]*scale, forward[1]*scale, forward[2]*scale,
        color="g", linewidth=1.0, arrow_length_ratio=0.3,
    )


# =============================
# TEST LOOP
# =============================
def run_test(model, scene, target, gate, render_fn,
             device, dt=0.1, H=10, sample_num=10, video_dir="videos"):

    os.makedirs(video_dir, exist_ok=True)
    plt.ion()

    target_np  = target.reshape(1, 6).astype(np.float32)
    init_poses = sample_init_poses(target, n=sample_num)

    for idx, init_pose in enumerate(init_poses):
        pose_np = init_pose.reshape(1, 6).astype(np.float32)

        traj, V_list = [], []

        fig     = plt.figure(figsize=(18, 5))
        ax_traj = fig.add_subplot(1, 3, 1, projection="3d")
        ax_V    = fig.add_subplot(1, 3, 2)
        ax_img  = fig.add_subplot(1, 3, 3)

        traj_line, = ax_traj.plot([], [], [], "b-", linewidth=2)
        ax_traj.scatter(*target[:3], c="red",   s=30, marker="*", label="Target")
        ax_traj.scatter(*gate[:3],   c="black", s=30, marker="*", label="Gate")
        ax_traj.set_xlabel("X"); ax_traj.set_ylabel("Y"); ax_traj.set_zlabel("Z")
        ax_traj.set_title("Trajectory"); ax_traj.legend()

        ax_V.set_title("Lyapunov Function")
        ax_V.set_xlabel("Time Step"); ax_V.set_ylabel("V(x)")
        ax_V.grid(True, alpha=0.3)
        V_line, = ax_V.plot([], [], "g-", linewidth=2)

        video_path  = os.path.join(video_dir, f"rollout_tflite_{idx:02d}.mp4")
        writer      = FFMpegWriter(fps=3, metadata=dict(artist="Controller"), bitrate=1500)
        frame_count = 0

        with writer.saving(fig, video_path, dpi=100):
            for t in range(H):
                # render
                img   = render_fn(pose_np[0], scene, device=device)  # (3,H,W)
                img_t = img.unsqueeze(0)                               # (1,3,H,W)

                # single fused inference: action + V together
                action_np, V_val = model(img_t, pose_np, target_np)
                action_np = transform_drone_velocity_to_world_frame_np(action_np)  # in-place transform
                zeros = np.zeros(action_np.shape[:-1] + (2,), dtype=action_np.dtype)
                action_np = np.concatenate([action_np, zeros], axis=-1)
                next_pose_np = pose_np + action_np * dt
                

                img_np = img_t.squeeze(0).permute(1, 2, 0).cpu().numpy()

                traj.append(pose_np[0].copy())
                V_list.append(V_val)
                traj_arr = np.array(traj)

                # trajectory
                traj_line.set_data(traj_arr[:, 0], traj_arr[:, 1])
                traj_line.set_3d_properties(traj_arr[:, 2])
                ax_traj.set_xlim(target[0]-1.0, target[0]+1.0)
                ax_traj.set_ylim(target[1]-2.0, target[1]+2.0)
                ax_traj.set_zlim(target[2]-1.0, target[2]+1.0)
                for c in ax_traj.collections[:]:
                    c.remove()
                ax_traj.scatter(*target[:3], c="red",   s=30, marker="*", label="Target")
                ax_traj.scatter(*gate[:3],   c="black", s=30, marker="*", label="Gate")
                draw_frame(ax_traj, pose_np[0, :3], pose_np[0, 3:], scale=0.2)

                # rendered image
                ax_img.clear()
                ax_img.imshow(img_np)
                ax_img.set_title("Rendered View")
                ax_img.axis("off")

                # lyapunov plot
                V_line.set_data(np.arange(len(V_list)), V_list)
                ax_V.set_xlim(0, H - 1)
                ax_V.set_ylim(0, max(V_list) * 1.2 + 0.1)

                p    = pose_np[0]
                info = (f"Step: {t+1}/{H} | V: {V_val:.4f} | "
                        f"Pose: [{p[0]:.2f},{p[1]:.2f},{p[2]:.2f},"
                        f"{p[3]:.2f},{p[4]:.2f},{p[5]:.2f}]")
                fig.suptitle(info, fontsize=10)

                plt.tight_layout()
                plt.draw()
                writer.grab_frame()
                frame_count += 1
                plt.pause(0.01)

                pose_np = next_pose_np

        plt.ioff()
        plt.close(fig)
        print(f"[DONE] rollout {idx} — {video_path} ({frame_count} frames)")


# =============================
# MAIN
# =============================
if __name__ == "__main__":
    np.random.seed(0)

    cfg    = Config()
    device = cfg.device
    print(f"Device: {device}")

    print("Loading gsplat scene ...")
    scene = load_gsplat_scene(cfg)

    print("Loading fused TFLite model ...")
    model = TFLiteFusedModel(cfg.fused_tflite)
    print(f"  image  input : {model._img_det['shape']}  (nhwc={model.nhwc})")
    print(f"  pose   input : {model._pose_det['shape']}")
    print(f"  target input : {model._target_det['shape']}")
    print(f"  action output: {model._action_det['shape']}")
    print(f"  V      output: {model._v_det['shape']}")

    run_test(
        model      = model,
        scene      = scene,
        target     = cfg.target_pose,
        gate       = cfg.gate_pose,
        render_fn  = render,
        device     = device,
        dt         = cfg.dt,
        H          = cfg.H,
        sample_num = cfg.sample_num,
        video_dir  = cfg.video_dir,
    )
