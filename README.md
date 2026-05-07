# Scripts

This folder contains scripts for running a verified visual controller using Gaussian Splatting (gsplat) scene rendering and TFLite neural network models.

---

## Files

### `render_image.py`

Handles loading and rendering from a pre-trained 3D Gaussian Splatting scene (nerfstudio splatfacto format).

**Key functions:**

- `load_gsplat_scene(cfg)` — Loads Gaussian parameters (means, quats, opacities, scales, colors) and the dataparser transform from a nerfstudio checkpoint.
- `get_viewmat(optimized_camera_to_world)` — Converts a camera-to-world matrix into the gsplat world-to-camera viewmat format.
- `render(pose, scene, ...)` — Renders a single image from a 6-DOF pose `[x, y, z, yaw, pitch, roll]`. Returns a `(3, H, W)` float32 tensor.
- `render_batch(poses, scene, ...)` — Batched version of `render` for multiple poses at once. Returns `(B, 3, H, W)`.

**Default camera intrinsics:** `320×240`, `fx=273.42`, `fy=273.79`, `cx=174.59`, `cy=107.77`.

---

### `ctrl_lya_tf.py`

Provides a modular single-step pipeline combining scene rendering, control prediction, and Lyapunov value estimation.

**Config (`Config` dataclass):**

| Field | Default | Description |
|---|---|---|
| `device` | `cuda` / `cpu` | Compute device |
| `target_pose` | `[0, -4, 0, 1.57, 0, 0]` | Goal pose `[x, y, z, roll, pitch, yaw]` |
| `gsplat_path` | `nerfstudio/outputs/uturn/...` | Path to the splatfacto output directory |
| `checkpoint` | `step-000040005.ckpt` | Checkpoint filename inside `nerfstudio_models/` |
| `controller_tflite` | `weights/ctrl.tflite` | Path to the controller TFLite model |
| `lyapunov_tflite` | `weights/lya.tflite` | Path to the Lyapunov network TFLite model |

**Key classes and functions:**

- `TFLiteModel(path)` — Thin wrapper around `tf.lite.Interpreter`. Callable with NumPy arrays as inputs.
- `generate_image(cur_pose, scene, device)` — Renders a `(1, H, W, 3)` image from a pose using the gsplat scene.
- `compute_control_and_lyapunov(image, cur_pose, target_pose, ctrl_model, Vnet)` — Runs the controller on the image and the Lyapunov network on the current/target poses. Returns `(control, V)`.
- `step(cur_pose, scene, ctrl_model, Vnet, target_pose, device)` — Full single-step pipeline: render → control + Lyapunov. Returns `(image, control, V)`.

**Usage:**
```bash
python3 scripts/ctrl_lya_tf.py
```

---

### `test_ctrl_lya_tf.py`

Runs closed-loop rollout tests, visualizes trajectories and Lyapunov values, and saves videos.

**Config (`Config` dataclass):**

| Field | Default | Description |
|---|---|---|
| `dt` | `0.1` | Integration timestep |
| `H` | `30` | Rollout horizon (steps) |
| `sample_num` | `5` | Number of random initial poses to test |
| `target_pose` | `[0, -4, 0, 1.57, 0, 0]` | Goal pose |
| `gate_pose` | `[0, -2, 0, 1.57, 0, 0]` | Intermediate gate pose (visualized) |
| `gsplat_path` | `nerfstudio/outputs/uturn/...` | Path to splatfacto output |
| `checkpoint` | `step-000040005.ckpt` | Checkpoint filename |
| `controller_tflite` | `weights/ctrl.tflite` | Controller TFLite model |
| `lyapunov_tflite` | `weights/lya.tflite` | Lyapunov network TFLite model |
| `video_dir` | `videos/` | Output directory for rollout videos |

**Key functions:**

- `sample_init_poses(target, n)` — Samples `n` random initial poses near the target with bounded uniform noise.
- `draw_frame(ax, pos, rpy, scale)` — Draws a camera forward-direction arrow on a 3D matplotlib axis.
- `run_test(...)` — Main rollout loop. For each sampled initial pose, steps the controller for `H` steps, renders the 3D trajectory, Lyapunov curve, and current camera image side-by-side, and saves an `.mp4` video.

**Output:** Videos saved to `videos/rollout_tf_00.mp4`, `rollout_tf_01.mp4`, etc.

**Usage:**
```bash
python3 scripts/test_ctrl_lya_tf.py
```

---

## Dependencies

- `torch`, `gsplat` — 3D Gaussian Splatting rendering
- `tensorflow` — TFLite model inference
- `numpy`, `scipy` — Numerical utilities and rotation math
- `matplotlib` — Visualization and video export
- `ffmpeg` — Required by `matplotlib.animation.FFMpegWriter` for video saving

---

## Directory Layout (expected)

```
VerifiedVisualController/
├── nerfstudio/
│   └── outputs/
│       └── uturn/splatfacto/2025-05-09_151825/
│           ├── config.yml
│           ├── dataparser_transforms.json
│           └── nerfstudio_models/
│               └── step-000040005.ckpt
├── weights/
│   ├── ctrl.tflite
│   └── lya.tflite
├── videos/              # rollout videos written here
└── scripts/
    ├── render_image.py
    ├── ctrl_lya_tf.py
    └── test_ctrl_lya_tf.py
```
