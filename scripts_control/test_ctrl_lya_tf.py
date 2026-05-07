import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from dataclasses import dataclass
from scipy.spatial.transform import Rotation
from matplotlib.animation import FFMpegWriter

from render_image import render, load_gsplat_scene


# =============================
# CONFIG
# =============================
@dataclass
class Config:

    device = "cuda" if tf.config.list_physical_devices("GPU") else "cpu"

    dt = 0.1
    H = 30
    sample_num = 5

    target_pose = np.array([0.0, -4.0, 0.0, 1.57, 0.0, 0.0], dtype=np.float32)
    gate_pose = np.array([0.0, -2.0, 0.0, 1.57, 0.0, 0.0], dtype=np.float32)

    gsplat_path = "nerfstudio/outputs/uturn/splatfacto/2025-05-09_151825"
    checkpoint = "nerfstudio_models/step-000040005.ckpt"

    controller_tflite = "weights/ctrl.tflite"
    lyapunov_tflite = "weights/lya.tflite"

    video_dir = "videos"


# =============================
# POSE SAMPLING
# =============================
def sample_init_poses(target, n=10):

    return target + np.random.uniform(
        low=[-1.5, -1.5, -0.7, -0.5, -0.3, -0.3],
        high=[ 1.5,  1.5,  0.7,  0.5,  0.3,  0.3],
        size=(n, 6)
    ).astype(np.float32)


# =============================
# TFLITE WRAPPER (CORE CHANGE)
# =============================
class TFLiteModel:

    def __init__(self, path):

        self.interpreter = tf.lite.Interpreter(model_path=path)
        self.interpreter.allocate_tensors()

        self.inp = self.interpreter.get_input_details()
        self.out = self.interpreter.get_output_details()

    def __call__(self, *inputs):

        for i, x in enumerate(inputs):
            self.interpreter.set_tensor(
                self.inp[i]["index"],
                x.astype(np.float32)
            )

        self.interpreter.invoke()

        outputs = [
            self.interpreter.get_tensor(o["index"])
            for o in self.out
        ]

        return outputs if len(outputs) > 1 else outputs[0]


# =============================
# DRAW CAMERA FRAME
# =============================
def draw_frame(ax, pos, rpy, scale=0.1):

    px, py, pz = pos
    yaw, pitch, roll = rpy

    R = Rotation.from_euler("ZYX", (yaw, pitch, roll)).as_matrix()

    tmp = Rotation.from_euler('zyx', [-np.pi/2, np.pi/2, 0]).as_matrix()

    R = R @ tmp
    R[:, 1:3] *= -1
    R = R[np.array([0, 2, 1]), :]
    R[2, :] *= -1

    forward = R[1] / (np.linalg.norm(R[1]) + 1e-8)

    ax.quiver(
        px, py, pz,
        forward[0]*scale,
        forward[1]*scale,
        forward[2]*scale,
        color='g'
    )


# =============================
# ROLLOUT LOOP (UNCHANGED LOGIC)
# =============================
def run_test(ctrl, Vnet, scene, target, gate, render_fn,
             device, dt, H, sample_num, video_dir):

    os.makedirs(video_dir, exist_ok=True)

    target_t = target[None, :].astype(np.float32)

    init_poses = sample_init_poses(target, sample_num)

    for idx, init_pose in enumerate(init_poses):

        pose = init_pose[None, :].astype(np.float32)

        traj, V_list = [], []

        fig = plt.figure(figsize=(18, 5))

        ax_traj = fig.add_subplot(1, 3, 1, projection='3d')
        ax_V = fig.add_subplot(1, 3, 2)
        ax_img = fig.add_subplot(1, 3, 3)

        traj_line, = ax_traj.plot([], [], [], 'b-')
        V_line, = ax_V.plot([], [], 'g-')

        video_path = os.path.join(video_dir, f"rollout_tf_{idx:02d}.mp4")

        writer = FFMpegWriter(fps=3)

        with writer.saving(fig, video_path, dpi=100):

            for t in range(H):

                # ================= render =================
                img = render_fn(pose[0], scene, device=device)
                img = img.detach().cpu().numpy() 
                img = np.transpose(img, (1, 2, 0)) # CHW → HWC
                img = img[None].astype(np.float32)  # add batch dim
                #print(img.shape)

                # ================= control =================
                pred = ctrl(img)
                if isinstance(pred, list):  # safety
                    pred = pred[0]

                next_pose = pose + pred * dt

                img_np = img[0]#.transpose(1, 2, 0)

                # ================= lyapunov =================
                V = Vnet(pose, target_t)[0]

                # ================= log =================
                p = pose[0]
                traj.append(p.copy())
                V_list.append(float(V[0]))

                traj_np = np.array(traj)

                # ================= trajectory =================
                traj_line.set_data(traj_np[:, 0], traj_np[:, 1])
                traj_line.set_3d_properties(traj_np[:, 2])

                ax_traj.set_xlim(target[0]-1, target[0]+1)
                ax_traj.set_ylim(target[1]-2, target[1]+2)
                ax_traj.set_zlim(target[2]-1, target[2]+1)

                for c in ax_traj.collections[:]:
                    c.remove()

                ax_traj.scatter(*target[:3], c='red', s=30, marker='*')
                ax_traj.scatter(*gate[:3], c='black', s=30, marker='*')

                draw_frame(ax_traj, p[:3], p[3:], scale=0.2)

                # ================= image =================
                ax_img.clear()
                ax_img.imshow(img_np)
                ax_img.axis("off")

                # ================= V =================
                V_line.set_data(np.arange(len(V_list)), V_list)
                ax_V.set_xlim(0, H-1)
                ax_V.set_ylim(0, max(V_list)*1.2 + 1e-6)

                fig.suptitle(
                    f"Step {t} | V={V[0]:.4f} | "
                    f"Pose={p.round(2)}"
                )

                plt.tight_layout()
                writer.grab_frame()
                plt.pause(0.01)

                pose = next_pose

        plt.close(fig)

        print(f"[DONE] rollout {idx} saved: {video_path}")


# =============================
# MAIN
# =============================
if __name__ == "__main__":

    cfg = Config()

    scene = load_gsplat_scene(cfg)

    root = os.path.dirname(os.path.abspath(__file__))

    ctrl_path = os.path.join(root, "..", cfg.controller_tflite)
    lya_path = os.path.join(root, "..", cfg.lyapunov_tflite)

    ctrl = TFLiteModel(ctrl_path)
    Vnet = TFLiteModel(lya_path)

    print("[INFO] controller:", ctrl_path)
    print("[INFO] lyapunov:", lya_path)

    run_test(
        ctrl=ctrl,
        Vnet=Vnet,
        scene=scene,
        target=cfg.target_pose,
        gate=cfg.gate_pose,
        render_fn=render,
        device=cfg.device,
        dt=cfg.dt,
        H=cfg.H,
        sample_num=cfg.sample_num,
        video_dir=cfg.video_dir
    )