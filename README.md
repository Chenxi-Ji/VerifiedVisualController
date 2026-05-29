# Verified Visual Controller

This repository contains a vision-based controller pipeline built on Gaussian Splatting rendering, a PyTorch controller/Lyapunov model, TFLite export/inference, and a certification workflow.

## Project Tree

```text
VerifiedVisualController/
├── auto_LiRPA/
├── backups/
├── figures/
├── nerfstudio/
│   └── outputs/uturn/splatfacto/2025-05-09_151825/
│       ├── config.yml
│       ├── dataparser_transforms.json
│       └── nerfstudio_models/
│           └── step-000040005.ckpt
├── results/
├── scripts_cert/
│   ├── certify_control.py
│   └── plot.py
├── scripts_control/
│   ├── draw_lya_2d.py
│   ├── render_image.py
│   ├── test_ctrl_lya_pt.py
│   ├── train_ctrl_lya_pt.py
│   └── utils_ctrl_lya_pt.py
├── scripts_render/
│   ├── abstract_render_image.py
│   ├── render_image.py
│   ├── utils_abstract_render.py
│   ├── utils_alpha_blending.py
│   └── utils_rational_quad.py
├── scripts_tflite/
│   ├── debug_pt_vs_tflite.py
│   ├── export_to_tflite.py
│   └── test_ctrl_lya_tflite.py
├── test_videos/
├── videos/
└── weights/
```

## Scripts

### `scripts_control/render_image.py`
Loads a nerfstudio Gaussian Splatting checkpoint and renders RGB images from 6-DoF poses. This module is used by the control and certification scripts.

Run: imported by other scripts; no standalone entry point.

### `scripts_control/utils_ctrl_lya_pt.py`
Defines the PyTorch controller network, the Lyapunov network, and pose/velocity frame transforms.

Run: imported by other scripts; no standalone entry point.

### `scripts_control/train_ctrl_lya_pt.py`
Trains the PyTorch controller and Lyapunov models end-to-end.

Input:
- Training data sampled around the target pose
- A nerfstudio Gaussian Splatting scene checkpoint in `nerfstudio/outputs/uturn/splatfacto/2025-05-09_151825/`

Output:
- `weights/ctrl_lya.pt` containing the controller and Lyapunov weights
- `training_curves.png` with loss and training diagnostics

Functionality:
- Builds pose samples with `PoseDataset`
- Renders images from the gsplat scene with `render_batch`
- Optimizes the controller and Lyapunov networks with a curriculum schedule
- Logs trajectory, Lyapunov, and terminal-state losses during training

Run:
```bash
python3 scripts_control/train_ctrl_lya_pt.py
```

### `scripts_control/test_ctrl_lya_pt.py`
Runs closed-loop rollout tests with the PyTorch model, renders trajectories, and saves rollout videos in `videos/`.

Run:
```bash
python3 scripts_control/test_ctrl_lya_pt.py
```

### `scripts_control/draw_lya_2d.py`
Evaluates the Lyapunov function on 2D slices and saves a contour plot to `figures/lya_2d.png`.

Run:
```bash
python3 scripts_control/draw_lya_2d.py
```

### `scripts_render/render_image.py`
Rendering helper used by the abstract rendering and certification pipeline.

Input:
- A nerfstudio Gaussian Splatting checkpoint and `dataparser_transforms.json`
- A 6-DoF pose or pose interval depending on the helper that calls it

Output:
- A rendered RGB tensor for nominal rendering
- Lower/upper image bounds when called through the abstract rendering pipeline

Functionality:
- Loads Gaussian parameters, opacities, scales, and colors from the checkpoint
- Converts camera poses into the gsplat view-matrix convention
- Produces rendered images used by certification and visualization scripts

Run: imported by other scripts; no standalone entry point.

### `scripts_render/utils_abstract_render.py`
Contains bound-propagation helpers for abstract rendering and certification-style analysis.

Run: imported by other scripts; no standalone entry point.

### `scripts_render/utils_alpha_blending.py`
Implements alpha-blending bound utilities used by the abstract renderer.

Run: imported by other scripts; no standalone entry point.

### `scripts_render/utils_rational_quad.py`
Provides rational quadratic bound utilities used by the abstract renderer.

Run: imported by other scripts; no standalone entry point.

### `scripts_render/abstract_render_image.py`
Demonstrates abstract rendering by computing lower, nominal, and upper images for a pose interval.

Input:
- A pose lower bound and upper bound defined in the script
- The gsplat scene checkpoint under `nerfstudio/outputs/uturn/splatfacto/2025-05-09_151825/`

Output:
- Three images: lower bound, nominal, and upper bound renderings
- `figures/abstract_images.png`

Functionality:
- Calls `render_bound` to propagate pose uncertainty through the renderer
- Visualizes the resulting image interval side by side
- Measures and prints the total rendering time

Run:
```bash
python3 scripts_render/abstract_render_image.py
```

### `scripts_tflite/export_to_tflite.py`
Exports the PyTorch controller and Lyapunov models to a fused float16 TFLite model.

Input:
- `weights/ctrl_lya.pt`
- The PyTorch `Controller` and `Lyapunov` definitions in `scripts_control/utils_ctrl_lya_pt.py`

Output:
- `weights/fused.onnx` during export
- `weights/fused_tf_saved/` as an intermediate SavedModel directory
- `weights/ctrl_lya.tflite` as the final fused TFLite model

Functionality:
- Wraps the controller and Lyapunov network into a single fused graph
- Exports the fused graph from PyTorch to ONNX
- Converts ONNX to TensorFlow SavedModel with `onnx2tf`
- Converts the SavedModel to float16 TFLite for deployment

Run:
```bash
python3 scripts_tflite/export_to_tflite.py
```

### `scripts_tflite/test_ctrl_lya_tflite.py`
Runs closed-loop rollout tests using the fused TFLite model, renders trajectories, and saves rollout videos in `videos/`.

Run:
```bash
python3 scripts_tflite/test_ctrl_lya_tflite.py
```

### `scripts_tflite/debug_pt_vs_tflite.py`
Numerically compares the PyTorch controller/Lyapunov outputs against the fused TFLite model.

Run:
```bash
python3 scripts_tflite/debug_pt_vs_tflite.py
```

### `scripts_cert/certify_control.py`
Splits the pose space into boxes, checks Lyapunov decrease under the learned controller, and saves certification results.

Input:
- Pose bounds defined in the script
- `weights/ctrl_lya.pt` for the controller and Lyapunov model
- The nerfstudio Gaussian Splatting scene in `nerfstudio/outputs/uturn/splatfacto/2025-05-09_151825/`

Output:
- `results/cert_result.pt` with the verified boxes and metadata
- A 3D visualization of verified and non-verified regions

Functionality:
- Tiles the pose space into smaller boxes
- Samples states inside each box and evaluates the controller and Lyapunov decrease
- Marks boxes that satisfy the decrease condition
- Saves the certification result for later plotting

Run:
```bash
python3 scripts_cert/certify_control.py
```

### `scripts_cert/plot.py`
Loads a saved certification result and plots the verified and non-verified boxes in 3D.

Run:
```bash
python3 scripts_cert/plot.py
```

## Dependencies

The current imports require the following Python libraries:

- `torch`
- `numpy`
- `scipy`
- `matplotlib`
- `tqdm`
- `opencv-python` (`cv2`)
- `gsplat`
- `nerfstudio`
- `tensorflow` or `tflite-runtime` or `ai-edge-litert` for TFLite inference/export
- `onnx` and `onnx2tf` for the export pipeline

System dependency:

- `ffmpeg` for `matplotlib.animation.FFMpegWriter`

Optional:

- `auto_LiRPA` is present in the repo for experimentation, but it is not required by the currently active certification script.

Example install set:

```bash
pip install torch numpy scipy matplotlib tqdm opencv-python gsplat nerfstudio tensorflow onnx onnx2tf ai-edge-litert
```

If you use `tflite-runtime` instead of `tensorflow`, install that package in place of `tensorflow`.
