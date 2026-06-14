# VerifiedVisualControllerTF Scripts

> **2026-06-11 update:** the pipeline now trains in the cleaned drone-arena
> gate scene (`Gate_Long_hloc_seq`) using a **gate-centered world frame**
> (gate at origin, +y through the gate, z down, yaw=π/2 faces the gate), an
> upgraded alpha-beta-CROWN-friendly controller, and domain randomization.
> **Read [`GATE_ARENA_SETUP.md`](GATE_ARENA_SETUP.md)** for all changes,
> coordinate conventions, and the Starling 2 (VOXL 2) deployment plan.
> Working conda env for training/testing: `certified_visual_controller`.

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

- `Controller`: CNN image-based controller predicting body-frame `(vx, vy, vz, yaw_rate)`; 52k params, alpha-beta-CROWN-supported ops only, exact ReLU-based action clamp.
- `DomainRandomizer`: training-time image randomization (lighting/color/noise/blur/cutout).
- `Lyapunov`: positive-definite Lyapunov network using pose/target errors.
- `body_to_world_velocity(...)` / `body_to_world_velocity_np(...)`: yaw-aware body→world conversion (gate-centered frame).
- `transform_drone_velocity_to_world_frame*(...)`: legacy fixed-flip conversion (old uturn scene only).

### `scripts_control/explore_scene.py` / `scripts_control/locate_gate.py`

Scene bring-up tools for a new splat: pose conversion from `transforms.json`
(with validation renders), gate triangulation from manually-read ring pixels,
true-vertical recovery from the ring rim, and `world_frame.json` generation.
See their docstrings and `GATE_ARENA_SETUP.md`.

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

## Python Environment And Dependencies

Recommended Python version:

- Python 3.10 or 3.11

Install core dependencies (training/testing with PyTorch + gsplat):

```bash
python3 -m pip install numpy scipy matplotlib tqdm opencv-python torch gsplat
```

Install TFLite inference dependencies (for `scripts_tflite/test_ctrl_lya_tflite.py` and `scripts_tflite/debug_pt_vs_tflite.py`):

```bash
python3 -m pip install tensorflow tflite-runtime
```

Install export dependencies (for `scripts_tflite/export_to_tflite.py`):

```bash
python3 -m pip install onnx onnx2tf tensorflow
```

System dependency for video writing:

- `ffmpeg` (required by `matplotlib.animation.FFMpegWriter`)

## Notes

- `__pycache__/` directories are auto-generated and are not part of the main code logic.
- Most scripts assume existing assets under `nerfstudio/outputs/...` and `weights/`.

## Project Structure

**# To run scripts in this repo, please download scene data from https://drive.google.com/drive/folders/1koY1TL30Bty2x0U6VpszKRgMXk61oTkG?usp=drive_link and put it aligned with the project tree diagram at the end of this readme file.**

```text
VerifiedVisualControllerTF/
├── README.md
├── GATE_ARENA_SETUP.md          # 2026-06-11 arena/gate scene guide + deployment plan
├── figures/
│   └── explore/                 # scene bring-up diagnostics (see GATE_ARENA_SETUP.md)
├── nerfstudio/
│   ├── Gate_Long_hloc_seq_data -> ~/certified_visual_controller/video/data/Gate_Long_hloc_seq
│   └── outputs/
│       ├── Gate_Long_hloc_seq -> ~/certified_visual_controller/video/outputs/Gate_Long_hloc_seq
│       │   └── splatfacto/2026-06-11_015308_cleaned/
│       │       ├── config.yml
│       │       ├── dataparser_transforms.json
│       │       ├── world_frame.json          # gate-centered frame (generated by locate_gate.py)
│       │       └── nerfstudio_models/step-000129999.ckpt
│       └── uturn/
│           └── splatfacto/2025-05-09_151825/   # legacy scene (still supported)
├── scripts_control/
│   ├── draw_lya_2d.py
│   ├── explore_scene.py
│   ├── locate_gate.py
│   ├── render_image.py
│   ├── test_ctrl_lya_pt.py
│   ├── train_ctrl_lya_pt.py
│   └── utils_ctrl_lya_pt.py
├── scripts_tflite/
│   ├── debug_pt_vs_tflite.py
│   ├── export_to_tflite.py
│   └── test_ctrl_lya_tflite.py
├── videos/
└── weights/
	├── ctrl_lya.pt              # promoted weights (must match current Controller!)
	├── ctrl_lya_<stamp>.pt      # training run outputs
	└── ctrl_lya.tflite
```
