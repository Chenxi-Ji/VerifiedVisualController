import os
import numpy as np
import tensorflow as tf
from dataclasses import dataclass
from render_image import render, load_gsplat_scene

@dataclass
class Config:
    device = "cuda" if tf.config.list_physical_devices("GPU") else "cpu"

    target_pose = np.array([0.0, -4.0, 0.0, 1.57, 0.0, 0.0], dtype=np.float32)

    # gsplat path
    gsplat_path = "nerfstudio/outputs/uturn/splatfacto/2025-05-09_151825"
    checkpoint = "nerfstudio_models/step-000040005.ckpt"

    controller_tflite = "weights/ctrl.tflite"
    lyapunov_tflite = "weights/lya.tflite"

    

# =============================
# TFLITE WRAPPER
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

        return outputs[0] if len(outputs) == 1 else outputs


# =============================
# IMAGE GENERATION (SEPARATE FUNCTION)
# =============================
def generate_image(cur_pose, scene, device="cpu"):
    """
    Input:
        cur_pose : np.ndarray (6,) or (1,6), float32
            [x, y, z, roll, pitch, yaw]

    Output:
        image : np.ndarray
            shape: (1, H, W, 3), float32
    """

    if cur_pose.ndim == 1:
        cur_pose = cur_pose[None, :]

    img = render(cur_pose[0], scene, device=device)

    # torch -> numpy + HWC format
    img = img.detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))  # CHW → HWC
    img = img[None].astype(np.float32)  # (1, H, W, 3)

    return img


# =============================
# CONTROL + LYPUNOV FUNCTION (SEPARATE)
# =============================
def compute_control_and_lyapunov(image, cur_pose, target_pose, ctrl_model, Vnet):
    """
    Inputs
    ------
    image:
        shape (1, H, W, 3), float32

    cur_pose:
        shape (6,) or (1, 6), float32

    target_pose:
        shape (6,) or (1, 6), float32

    Returns
    -------
    control:
        shape (6,), float32

    V:
        scalar float
    """

    # ---------------- reshape safety ----------------
    if cur_pose.ndim == 1:
        cur_pose = cur_pose[None, :]
    if target_pose.ndim == 1:
        target_pose = target_pose[None, :]

    # ---------------- control ----------------
    control = ctrl_model(image)
    if isinstance(control, list):
        control = control[0]
    control = control.astype(np.float32)

    # ---------------- lyapunov ----------------
    V = Vnet(cur_pose, target_pose)
    if isinstance(V, list):
        V = V[0]

    V = float(V[0])

    return control, V


# =============================
# SINGLE STEP PIPELINE
# =============================
def step(cur_pose, scene, ctrl_model, Vnet, target_pose, device="cpu"):
    """
    ONE FULL STEP:

    cur_pose -> image -> control + V

    Returns:
        image, control, V
    """

    # 1. render image from pose
    image = generate_image(cur_pose, scene, device=device)

    # 2. compute control + Lyapunov
    control, V = compute_control_and_lyapunov(
        image=image,
        cur_pose=cur_pose,
        target_pose=target_pose,
        ctrl_model=ctrl_model,
        Vnet=Vnet
    )

    return image, control, V


# =============================
# EXAMPLE USAGE
# =============================
if __name__ == "__main__":
    cfg = Config()

    # scene
    scene = load_gsplat_scene(cfg)

    # models
    root = os.path.dirname(os.path.abspath(__file__))

    ctrl_path = os.path.join(root, "..", cfg.controller_tflite)
    lya_path = os.path.join(root, "..", cfg.lyapunov_tflite)

    ctrl = TFLiteModel(ctrl_path)
    Vnet = TFLiteModel(lya_path)

    print("[INFO] controller:", ctrl_path)
    print("[INFO] lyapunov:", lya_path)

    # input pose
    cur_pose = np.array([1, -5, 0.2, 0.1, 0, 0], dtype=np.float32)

    # single step
    image, control, V = step(
        cur_pose=cur_pose,
        scene=scene,
        ctrl_model=ctrl,
        Vnet=Vnet,
        target_pose=cfg.target_pose,
        device=cfg.device
    )

    print("image shape:", image.shape)
    print("control:", control)
    print("V:", V)