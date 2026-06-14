# Drone Arena Gate Scene — Setup, Changes & Deployment Guide

**Date:** 2026-06-11
**Scene:** `Gate_Long_hloc_seq` (netted drone arena, one octagonal ring gate on blue mats)
**Replaces:** the old `uturn` scene (still works — see [Backward compatibility](#backward-compatibility))
**Hardware target:** ModalAI **Starling 2** (VOXL 2)

This document explains everything that changed in this repo to train the verified
visual controller in the new arena gaussian-splat environment, why each change was
needed, how to run the pipeline, and what is left to do for the real drone.

---

## 0. TL;DR

| What | Value |
|---|---|
| Splat checkpoint | `nerfstudio/outputs/Gate_Long_hloc_seq/splatfacto/2026-06-11_015308_cleaned/` (1.61M gaussians, floaters removed) |
| Coordinate frame | **Gate-centered**: gate center = origin, **+y = through the gate**, **z = down**, yaw = π/2 faces the gate, pitch = roll = 0 is level |
| Gate pose | `[0, 0, 0, π/2, 0, 0]` |
| Target (hover) pose | `[0, -1.5, 0, π/2, 0, 0]` (1.5 u before the gate) |
| Scene scale | ring inner opening = **0.88 scene units** → if the real opening is *D* meters, **1 unit = D / 0.88 m** |
| Controller | new 52k-param CNN, all ops alpha-beta-CROWN-supported, exact ReLU clamp |
| Training extras | image domain randomization + actuation noise + yaw-aware dynamics |
| Train | `conda activate certified_visual_controller && python scripts_control/train_ctrl_lya_pt.py` |
| After training | copy `weights/ctrl_lya_<stamp>.pt` → `weights/ctrl_lya.pt`, then test / export |

---

## 1. The splat environment

The arena was reconstructed with `splatfacto-big` (nerfstudio, conda env `imcooked`)
from the 4K video `Gate_Long.mov`, processed with hloc sequential matching.
Stray "floater" gaussians in empty space were removed with
`~/certified_visual_controller/video/clean_splats.py`:

- 22,646 spatially isolated splats (KNN statistical outliers — the floaters)
- 735,701 splats with opacity < 0.005 (invisible)
- 2,361,875 → **1,611,551 gaussians**, saved as a sibling nerfstudio run
  `2026-06-11_015308_cleaned` (original untouched)

Further manual cleanup is possible at any time via the SuperSplat round-trip
(`clean_splats.py export-ply` → edit at https://superspl.at/editor →
`clean_splats.py import-ply`). The repo consumes the scene through symlinks:

```
nerfstudio/outputs/Gate_Long_hloc_seq  ->  ~/certified_visual_controller/video/outputs/Gate_Long_hloc_seq
nerfstudio/Gate_Long_hloc_seq_data     ->  ~/certified_visual_controller/video/data/Gate_Long_hloc_seq
```

---

## 2. The three coordinate traps (read this before touching poses)

Getting poses right in this scene required fixing three independent problems.
They are documented here because they will bite again on the next scene.

### Trap 1 — `applied_transform` is baked into the saved dataparser transform

`dataparser_transforms.json` (what `render()` applies) maps from the **original
COLMAP space**, but `transforms.json` camera poses are stored **after** the
dataset's `applied_transform`. Feeding a `transforms.json` pose straight through
the render transform double-applies that permutation and puts the camera in a
wrong-but-plausible-looking place.

```python
# WRONG: c2w from transforms.json used directly
view = dataparser_transform @ c2w_json

# RIGHT: undo applied_transform first (explore_scene.c2w_to_pose does this)
Ta   = np.vstack([meta["applied_transform"], [0, 0, 0, 1]])
view = dataparser_transform @ np.linalg.inv(Ta) @ c2w_json
```

Validation: rendering from a dataset frame's pose must reproduce the real photo —
see `figures/explore/pose_validation.png`.

### Trap 2 — nerfstudio's "up" is ~75° off in this scene

Nerfstudio's `orientation_method: up` aligns the **mean camera up-axis** with +z.
This video looks *down at the mats* much of the time, so the model's +z ended up
nearly horizontal. Consequence: in raw scene coordinates,
`pitch = roll = 0` pointed ~70° off-level (a real level, gate-facing dataset frame
had `pitch≈1.16, roll≈2.8`). The true vertical was recovered from the **gate
itself**: the ring's top and bottom rim points are plumb, so intersecting their
pixel rays with the gate plane in two independent frames gives the gravity
direction (the two estimates agree to dot = 0.97).

### Trap 3 — where the gate actually is

The gate center was triangulated as the least-squares intersection of rays cast
through **manually-read ring-center pixels** of 6 well-separated dataset frames
(automatic Hough circle detection was unreliable). Residuals: 0.02–0.17 units at
distances of 1.8–6.4 units.

```
gate center (raw scene coords): [-0.3185, -0.2935, 2.4099]
ring inner opening:              0.88 scene units (0.85–0.91 across 4 measurements)
```

All three fixes are wrapped into one artifact so nothing downstream has to know
about them: **the gate-centered world frame**.

---

## 3. The gate-centered world frame (`world_frame.json`)

A 4×4 rigid transform `W` stored next to the checkpoint:

```
nerfstudio/outputs/Gate_Long_hloc_seq/splatfacto/2026-06-11_015308_cleaned/world_frame.json
```

`load_gsplat_scene()` detects the file and composes it into the render transform.
From then on **every pose in this repo lives in a clean frame**:

| Axis | Meaning |
|---|---|
| origin | gate center |
| +y | horizontal, through the gate (flight direction) |
| +x | horizontal, to the right when facing the gate |
| +z | **down** (NED-style) — so "0.3 above gate center" is `z = -0.3` |
| yaw | about vertical; `yaw = π/2` faces the gate; `yaw=0` faces +x |
| pitch = roll = 0 | true level flight |

The mats are ~0.65 u below gate center (`z ≈ +0.65`).

Verification: `figures/explore/gate_verification.png` — renders from the approach
line and offset poses; the ring sits on the image-center crosshair in every tile.

**Regenerating for a new scene:** run `scripts_control/explore_scene.py` to get
real-frame contact sheets and grid-overlaid frames, fill in the pixel tables at
the top of `scripts_control/locate_gate.py`, run it, and check the verification
grid it writes. The whole procedure is documented in that file's docstring.

---

## 4. File-by-file changes

### `scripts_control/render_image.py`

1. **Paths** point at the cleaned arena checkpoint (before: `uturn/.../step-000040005.ckpt`).
2. **Color fix.** This checkpoint was trained with `sh_degree 0`, where splatfacto
   stores *sigmoid-space* colors, not SH coefficients. The old code passed
   `features_dc` to the rasterizer as SH band-0 — correct for the old scene, wrong
   colors for this one:

   ```python
   # before
   colors = dc[:, None, :]

   # after (sh_degree-0 checkpoints)
   C0 = 0.28209479177387814
   colors = ((torch.sigmoid(dc) - 0.5) / C0)[:, None, :]   # exact: SH0(c) == sigmoid(dc)
   ```
3. **Device fix.** `torch.load(..., map_location=cfg.device)` — the cleaned
   checkpoint stores CPU tensors (the original happened to store CUDA tensors).
4. **World-frame rendering.** When `world_frame.json` exists, poses are
   interpreted in the gate frame with a minimal camera convention
   (`view[:3,:3] = R_zyx @ CAM_AXES`, position used directly). The legacy
   axis-juggling path is kept verbatim for the `uturn` scene.
   `load_gsplat_scene(cfg, use_world_frame=False)` opts out (used by the
   exploration tools, which work in raw dataset coordinates).
5. The scene tuple grew by one element (`world_frame` flag); `render` /
   `render_batch` unpack it. A leftover debug `print(viewmats)` was removed.

**Unchanged on purpose:** render resolution and intrinsics
(`300×200, fx=113.26, fy=113.35, cx=158.87, cy=98.84`, ≈106° HFOV).
These model the **drone's own camera**, which is what the controller sees at
deployment — do not replace them with the 4K capture-video intrinsics.

### `scripts_control/utils_ctrl_lya_pt.py`

**New `Controller`** (old: 3 convs → 32 features, ~18k params, `torch.clamp`):

```python
self.backbone = nn.Sequential(
    nn.AvgPool2d(2),              # 200x300 -> 100x150: 4x fewer neurons to verify
    nn.Conv2d(3, 16, 5, 2, 2),  nn.BatchNorm2d(16), nn.ReLU(),   # -> 50x75
    nn.Conv2d(16, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),   # -> 25x38
    nn.Conv2d(32, 48, 3, 2, 1), nn.BatchNorm2d(48), nn.ReLU(),   # -> 13x19
    nn.Conv2d(48, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),   # -> 7x10
    nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
)
self.action_head = nn.Sequential(
    nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 4),
)
```

- **52,180 parameters** (~110 KB as float16 TFLite) — still trivial for VOXL 2.
- Every op is in alpha-beta-CROWN / auto_LiRPA's supported set:
  Conv, BatchNorm (folds into conv at inference), ReLU, AvgPool, Linear, Concat.
- Action clamping rewritten as an **exact ReLU identity** instead of `torch.clamp`
  (Clip has weaker verifier support):

  ```python
  def clamp_relu(x, limit):           # == torch.clamp(x, -limit, +limit), exactly
      return torch.relu(x + limit) - torch.relu(x - limit) - limit
  ```
- The leading `AvgPool2d(2)` is the single biggest verification-cost lever: it
  quarters the number of neurons in the widest layers.
- Output is `(B, 4)` = `[vx, vy, vz, yaw_rate]` in the **drone body frame**,
  clamped to ±1.0 u/s translation and ±0.3 rad/s yaw.

**New `DomainRandomizer`** — applied to rendered observations during training only
(images are detached, so this never touches the deployed/verified network; it just
widens the visual distribution):

- gamma / brightness / contrast / per-channel white-balance jitter
- gaussian sensor noise, random light blur
- random cutout rectangles (occlusions: nets, cables, other drones)

**New `body_to_world_velocity(vel_body, yaw)`** — the old transform was a *fixed*
axis flip with no yaw dependence (only approximately valid near one heading):

```python
# before (legacy, kept for the uturn scene)
linear_world = [-vx, vy, -vz];  yaw_rate_world = -yaw_rate

# after (correct rigid-body kinematics, NED-style frame)
vx_w = cos(yaw) * vx - sin(yaw) * vy
vy_w = sin(yaw) * vx + cos(yaw) * vy
vz_w, yaw_rate_w = vz, yaw_rate
```

The `Lyapunov` network is unchanged (already small; sigmoid/cos are supported by
auto_LiRPA).

### `scripts_control/train_ctrl_lya_pt.py`

- `target_pose = [0, -1.5, 0, π/2, 0, 0]`, `gate_pose = [0, 0, 0, π/2, 0, 0]`.
- Sampling box around the target (was ±[2.0, 1.5, 1.0], yaw ±0.7 in old scene):

  ```python
  low  = [-1.5, -1.5, -0.5, -0.6, 0, 0]   # x lateral | y: 3.0u..0.5u before gate
  high = [ 1.5,  1.0,  0.4,  0.6, 0, 0]   # z: 0.5 above .. 0.4 below gate center
  ```
  (z capped at +0.4 because the mats are at z ≈ +0.65.)
- Rollout now uses domain randomization + correct kinematics + actuation noise:

  ```python
  pred_self = ctrl(domain_rand(img_curr))
  pred = body_to_world_velocity(pred_self, pose_curr[:, 3])
  pred = pred + torch.randn_like(pred) * cfg.actuation_noise   # 0.02
  ```

Curriculum, losses, horizons, optimizer are unchanged.

### `scripts_control/test_ctrl_lya_pt.py`

- New target/gate poses, tightened init-pose sampling, yaw-aware kinematics.
- `draw_frame()` simplified — in the gate frame the camera forward is just the
  rotated +x axis.
- Loading `weights/ctrl_lya.pt` now raises a clear error if the file holds
  pre-upgrade (old-architecture) weights.

### `scripts_tflite/*`

- `export_to_tflite.py`: **unchanged** — same `(1, 3, 200, 300)` input, same
  PyTorch → ONNX → onnx2tf → float16 TFLite pipeline. Verified: the new fused
  model exports to ONNX cleanly (64 KB).
- `test_ctrl_lya_tflite.py`: new scene path, new poses, yaw-aware kinematics,
  simplified `draw_frame`.
- `debug_pt_vs_tflite.py`: pose comparison ranges updated to the new training box.

### New tools: `scripts_control/explore_scene.py`, `scripts_control/locate_gate.py`

Scene-bring-up tools (run from the repo root, env `certified_visual_controller`):

- `explore_scene.py` — converts `transforms.json` poses to render poses
  (handling Trap 1), validates the round-trip, prints pose statistics, renders a
  trajectory contact sheet.
- `locate_gate.py` — triangulates the gate from manual ring-center pixels,
  derives true vertical from the ring rim (Trap 2), writes `world_frame.json`,
  and renders the verification grid. Re-running it reproduces the committed
  `world_frame.json` to ~1e-5.

Diagnostic imagery from the bring-up lives in `figures/explore/`.

---

## 5. How to run everything

```bash
conda activate certified_visual_controller     # torch 2.11 cu128 + gsplat 1.5.3
cd ~/certified_visual_controller/VerifiedVisualControllerTF

# sanity render (writes figures/example_image.png — gate centered, 1.5u out)
python scripts_control/render_image.py

# train (120 epochs, 3-phase curriculum; checkpoint: weights/ctrl_lya_<stamp>.pt)
python scripts_control/train_ctrl_lya_pt.py

# promote the trained weights, then closed-loop test (videos/rollout_pt_*.mp4)
cp weights/ctrl_lya_<stamp>.pt weights/ctrl_lya.pt
python scripts_control/test_ctrl_lya_pt.py

# Lyapunov landscape figure
python scripts_control/draw_lya_2d.py

# export fused controller+Lyapunov to float16 TFLite (env: pftolite)
conda run -n pftolite python scripts_tflite/export_to_tflite.py

# numerical parity PyTorch vs TFLite, then TFLite closed-loop rollout
conda run -n pftolite python scripts_tflite/debug_pt_vs_tflite.py
conda run -n pftolite python scripts_tflite/test_ctrl_lya_tflite.py
```

A 2-epoch smoke run of the full pipeline (render → DR → controller → dynamics →
Lyapunov losses → backward) was verified on the RTX 5080.

---

## 6. Verification with alpha-beta-CROWN

The controller was built so that the graph contains only well-supported
operations: `Conv2d`, `BatchNorm2d` (folded at inference), `ReLU`, `AvgPool2d`,
`AdaptiveAvgPool2d((1,1))` (= global average pool), `Linear`, `Flatten`,
scalar multiply, `Concat`, and the ReLU-based clamp. Suggested flow:

```bash
# 1. export the controller (or the fused model) to ONNX — already part of
#    scripts_tflite/export_to_tflite.py (fused.onnx before conversion), or:
python - <<'PY'
import torch, sys; sys.path.insert(0, 'scripts_control')
from utils_ctrl_lya_pt import Controller
ctrl = Controller().eval()
ctrl.load_state_dict(torch.load('weights/ctrl_lya.pt', map_location='cpu')['controller'])
torch.onnx.export(ctrl, torch.rand(1,3,200,300), 'weights/controller.onnx',
                  input_names=['image'], output_names=['action'], opset_version=18)
PY

# 2. clone https://github.com/Verified-Intelligence/alpha-beta-CROWN and write a
#    config with model: onnx_path: weights/controller.onnx, spec: L_inf
#    perturbation on the image input + linear constraints on the action output.
```

Notes for the verification side:

- The input is 200×300×3 = 180k dimensions; the leading AvgPool keeps the first
  conv's activation volume manageable, but expect verification to be the
  bottleneck — patch/brightness specs over image subregions are a pragmatic
  starting point before full-image L∞.
- Properties worth verifying: action-bound consistency (output stays in the
  clamp box — should verify trivially given the ReLU-clamp), sign/monotonicity
  of `vy` over a region of approach images, and Lyapunov-decrease style
  properties on the fused model (the Lyapunov net's `cos`/`sigmoid` are
  supported by auto_LiRPA).

---

## 7. Next steps — deploying on the Starling 2 (VOXL 2)

The trained artifact is one file: `weights/ctrl_lya.tflite`
(float16, inputs `image (1,3,200,300)`, `pose (1,6)`, `target (1,6)`;
outputs `action (1,4)` body-frame `[vx, vy, vz, yaw_rate]` + scalar `V`).

### 7.1 Calibrate the scale (one tape measure)

Measure the ring's inner opening diameter `D` in meters. Then:

```
meters_per_unit = D / 0.88
```

All scene-unit quantities (positions, velocities) convert by this factor. E.g. if
`D = 0.9 m`, then 1 u ≈ 1.02 m and the ±1.0 u/s velocity clamp ≈ ±1 m/s.

### 7.2 Confirm the camera model

The controller was trained on renders with `300×200, fx≈113.3, fy≈113.3,
cx≈158.9, cy≈98.8` (≈106°×83° FOV) — these should match the Starling 2 camera
that will feed the network. Action items:

- Pull the actual calibration from the drone (`/data/modalai/` camera intrinsics
  on VOXL 2) for the camera you intend to use. The wide-FOV color option on the
  Starling 2 is the hires sensor; the tracking cameras are grayscale and would
  need a retrain on grayscale renders.
- If the real FOV differs noticeably, either retrain with matched render
  intrinsics (one-line change in `render()`'s defaults) or undistort+crop the
  camera stream to the trained model. Matching at render time is the cleaner
  path — that is the entire point of training in the splat.

### 7.3 Align the gate frame with VIO

The network's `pose` input and the body→world conversion live in the
gate-centered frame. On the drone, VIO (qVIO/OpenVINS on VOXL 2) gives pose in
its own local frame, so you need one rigid transform `T_vio→gate`:

- Simplest: place the drone at a measured spot relative to the gate (e.g., on the
  mats, 2 m in front of the gate center, facing it), zero VIO there, and compose
  the measured offset. The gate frame is NED-like (z down) — VOXL/PX4 local NED
  matches the convention directly if you align the +y axis with the through-gate
  direction.
- Better: an AprilTag on/near the gate stand and `voxl-tag-detector` to estimate
  the gate pose online, then `T_vio→gate` continuously.

Note the *image* path needs no alignment at all — only the Lyapunov monitor and
any pose-based logging do. The action is body-frame and goes straight to the
flight controller.

### 7.4 Onboard inference + control loop (MPA)

On VOXL 2 the natural shape is a small MPA service (or a fork of
`voxl-tflite-server`, which already manages camera pipes + the TFLite delegate):

1. Subscribe to the camera pipe; convert to RGB float `[0,1]`, resize to
   200×300, layout to the model's input (the exporter emits NHWC after onnx2tf —
   check with `debug_pt_vs_tflite.py`'s input-detail dump).
2. Run the fused TFLite model (XNNPACK/GPU delegate; at 52k params expect
   ≲ a few ms — comfortably faster than the 10 Hz training timestep).
3. Scale the action: `v_cmd_mps = action[:3] * meters_per_unit`, yaw rate is
   rad/s as-is.
4. Publish as PX4 **offboard velocity setpoints in the body frame**
   (via `voxl-vision-hub` figure-of-merit path or MAVSDK offboard
   `VelocityBodyYawspeed`) at ≥10 Hz.
5. Use the Lyapunov output `V` as a runtime monitor: command a hold/abort if `V`
   increases persistently or exceeds a threshold — this is the "certified"
   safety hook the architecture is designed around.

### 7.5 Bring-up sequence (safety first)

1. **Parity:** `debug_pt_vs_tflite.py` after every export (expect ~1e-3 max diff).
2. **Replay test:** feed recorded onboard camera frames through the TFLite model
   offline; check actions point the right way before any flight.
3. **Tethered / hand-held test:** hold the drone in the arena, watch live action
   vectors and `V` on a laptop.
4. **First flights:** velocity scale at 25–50% (`meters_per_unit * 0.25`),
   geofence + kill switch, start from poses inside the training box
   (x ±1.2, y −3.0…−0.5, z within ±0.4 of gate center, yaw ±0.5 of facing).
5. Iterate: if sim-to-real gap shows up (lighting, exposure), widen the
   `DomainRandomizer` ranges and retrain — that is exactly what it is for.

### 7.6 Open items checklist

- [ ] Measure ring inner diameter → set `meters_per_unit`
- [ ] Confirm/recalibrate drone camera intrinsics vs render intrinsics
- [ ] Full 120-epoch training run + closed-loop test videos
- [ ] Promote weights → `weights/ctrl_lya.pt` → TFLite export + parity check
- [ ] alpha-beta-CROWN property specs + verification runs on `controller.onnx`
- [ ] VIO↔gate frame alignment procedure (fixed offset or AprilTag)
- [ ] MPA inference service + PX4 offboard plumbing on VOXL 2
- [ ] Tethered bring-up, then low-speed free flight

---

## 8. Known limitations & notes

- **hloc scale drift:** the reconstruction is metrically self-consistent near the
  gate (where all training poses live) but the far ends of the long capture
  trajectory show drift. Don't trust scene coordinates more than ~3–4 u from the
  gate.
- **Second ring:** a spare ring gate leans against the far wall in the scene
  (visible from some angles). The training box keeps the real gate dominant in
  view, but be aware it exists if you widen the sampling ranges.
- **Kinematic model:** training integrates `pose += v·dt` (velocity-level,
  dt = 0.1 s). The Starling 2's velocity tracking is fast enough for this to be
  reasonable at ≤1 m/s, and the actuation-noise term covers some tracking error,
  but aggressive flight would need a dynamics-aware rollout.
- **Backward compatibility:** the old `uturn` scene renders through the legacy
  code path (no `world_frame.json` → legacy pose convention + legacy fixed-flip
  velocity transform are still in the codebase). Old `ctrl_lya.pt` weights do
  **not** load into the new Controller (different architecture) — the test script
  tells you so explicitly.

## 9. Figure index (`figures/explore/`)

| File | What it shows |
|---|---|
| `pose_validation.png` | render-at-dataset-pose vs real photo (Trap 1 fixed) |
| `gate_verification.png` | world-frame verification grid (ring on crosshair) |
| `real_frames_sheet.jpg`, `real_280_460.jpg`, `real_1380_1560.jpg` | real-video contact sheets used to find the gate |
| `grid_frame_*.jpg` | grid-overlaid frames used to read ring pixels manually |
| `scene_layout_zoom.png` | splat point cloud + camera path, 3 orthogonal views |
| `contact_sheet.png` | rendered trajectory contact sheet (raw data space) |
