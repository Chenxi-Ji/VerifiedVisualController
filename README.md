# VerifiedVisualControllerTF Scripts

This repository contains two script groups:

- `scripts_control/`: PyTorch training/testing utilities for the vision controller and Lyapunov model.
- `scripts_tflite/`: Export, validation, and rollout scripts for the fused TFLite model.

## scripts_control

### `scripts_control/render_image.py`

Loads a nerfstudio gsplat checkpoint and renders RGB images from a 6-DoF pose.

Key functionality:

- `load_gsplat_scene(cfg)`: loads Gaussian parameters + dataparser transform.
- `get_viewmat(...)`: converts camera pose into gsplat view matrix format.
- `render(pose, scene, ...)`: renders a single image `(3, H, W)`.
- `render_batch(poses, scene, ...)`: batch rendering for multiple poses.

### `scripts_control/utils_ctrl_lya_pt.py`

Core neural network definitions and velocity-frame transform helpers.

Key functionality:

- `Controller`: CNN image-based controller predicting 4D action, mapped to `(vx, vy, vz, yaw_rate)` style output.
- `Lyapunov`: positive-definite Lyapunov network using pose/target errors.
- `transform_drone_velocity_to_world_frame(...)`: torch version of frame conversion.
- `transform_drone_velocity_to_world_frame_np(...)`: NumPy version of frame conversion.

### `scripts_control/train_ctrl_lya_pt.py`

Joint training script for the PyTorch controller and Lyapunov model using gsplat-rendered observations.

Key functionality:

- Three-phase curriculum over 120 epochs (different loss weight schedules).
- Rollout-based optimization with trajectory, Lyapunov decrease, and final-state losses.
- Image cache to reduce repeated rendering overhead.
- Checkpoint save/resume via `weights/ctrl_lya.pt`.
- Training-curve plotting to `figures/training_curves.png`.

Run:

```bash
python3 scripts_control/train_ctrl_lya_pt.py
```

### `scripts_control/test_ctrl_lya_pt.py`

Closed-loop rollout test with PyTorch models. Generates videos that show trajectory, rendered view, and Lyapunov trend.

Key functionality:

- Samples random initial poses around the target.
- Runs controller-driven rollout for a fixed horizon.
- Evaluates Lyapunov value at each step.
- Saves per-rollout videos in `videos/`.

Run:

```bash
python3 scripts_control/test_ctrl_lya_pt.py
```

### `scripts_control/draw_lya_2d.py`

Visualizes 2D slices of the learned Lyapunov function around the target pose.

Key functionality:

- Loads Lyapunov weights from `weights/ctrl_lya.pt`.
- Computes contour maps for `(x,y)`, `(y,z)`, `(z,x)`, and `(x,yaw)`, `(y,yaw)`, `(z,yaw)`.
- Saves figure to `figures/lya_2d.png`.

Run:

```bash
python3 scripts_control/draw_lya_2d.py
```

## scripts_tflite

### `scripts_tflite/export_to_tflite.py`

Exports the trained PyTorch controller + Lyapunov model into one fused float16 TFLite file.

Pipeline:

1. PyTorch -> ONNX (`fused.onnx`)
2. ONNX -> TensorFlow SavedModel (via `onnx2tf`)
3. SavedModel -> float16 TFLite (`weights/ctrl_lya.tflite`)

Also includes cleanup of temporary export artifacts.

Run:

```bash
python3 scripts_tflite/export_to_tflite.py
```

### `scripts_tflite/debug_pt_vs_tflite.py`

Compares numerical outputs of PyTorch models and fused TFLite model.

Key functionality:

- Controller output diff statistics (`max`, `mean`, `p95`).
- Lyapunov value diff statistics across random pose/target pairs.
- Quick sanity summary of PyTorch vs TFLite consistency.

Run:

```bash
python3 scripts_tflite/debug_pt_vs_tflite.py
```

### `scripts_tflite/test_ctrl_lya_tflite.py`

Runs rollout testing with TFLite inference (fused controller + Lyapunov), while still using PyTorch gsplat rendering.

Key functionality:

- Uses one TFLite call to get both action and Lyapunov value.
- Applies frame conversion and integrates pose dynamics.
- Produces rollout videos in `videos/rollout_tflite_*.mp4`.

Run:

```bash
python3 scripts_tflite/test_ctrl_lya_tflite.py
```

## Notes

- `__pycache__/` directories are auto-generated and are not part of the main code logic.
- Most scripts assume existing assets under `nerfstudio/outputs/...` and `weights/`.
