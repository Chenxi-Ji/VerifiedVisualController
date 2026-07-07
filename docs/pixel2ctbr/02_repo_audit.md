# 02 — Repo Audit: what exists, what's reusable for pixels+IMU → CTBR

*2026-07-07. Sources: code-level audit of the three repos (agent-assisted, all claims
verified against source) + render benchmarks run on this machine today. Line refs are
clickable `file:line`.*

## 1. The three repos, one line each

- `VerifiedVisualController_small_clone` — training/eval/export + the splat twin. LIVE.
- `Starling2` — onboard C++ (voxl-tflite-server model helper) + dockerized MAVSDK
  offboard runner. LIVE.
- `~/mocap_ws` — OptiTrack→PX4 bridge + flight recorder. Becomes **evaluation-only**
  in this phase (mocap leaves the control loop; the recorder/plotter never were in it).

## 2. Renderer (the digital twin interface) — REUSE, with a speed upgrade

- Entry points: `load_gsplat_scene()` at `scripts_control/render_image.py:27`,
  `render()` at `:103`, `render_batch()` at `:166`.
- Scene tuple = (means, quats, opacities, scales, colors, transform, scale,
  world_frame); ckpt `nerfstudio/outputs/.../2026-06-11_015308_cleaned/nerfstudio_models/step-000129999.ckpt`,
  **1,611,551 gaussians**; gate-centered frame baked via `world_frame.json`.
- Pose convention `[px,py,pz,yaw,pitch,roll]`, gate frame (+y through gate, z down),
  camera axes via `CAM_AXES` (`render_image.py:75`).
- Current path renders **1024×768 fisheye** (measured calib fx 504.341/fy 503.320/
  cx 505.485/cy 367.606) then `cv2.resize INTER_LINEAR → 256×192` on CPU to
  byte-match `model_helper.cpp:302`.

### Render throughput measured today (RTX 5080 Laptop 16 GB, this scene)

`pixel2ctbr/bench_render.py`, `torch.no_grad`, fisheye, RGB:

| path | res | batch | img/s |
|---|---|---|---|
| legacy 1024×768 RGB+ED + CPU resize | 1024×768→256×192 | 1 | 186 |
| legacy | same | 8 | 121 |
| legacy | same | 32 | 114 |
| direct low-res (K scaled), packed=False | 256×192 | 32 | 367 |
| direct, packed=False | 128×96 | 32 | 422 |
| direct, **packed=True** | 128×96 | 32 | 503 |
| direct, packed=True | 128×96 | 128 | **568** |
| direct, packed=True | 256×192 | 128 | 523 |
| direct, any | any | ≥256 | OOM |

Findings:
1. **~500–570 img/s is the ceiling** on this GPU for this scene. Cost is dominated by
   projecting 1.6 M gaussians per camera, not by pixels (128×96 ≈ 256×192).
2. `packed=True` + batch 128 is the sweet spot; batch ≥256 OOMs (projection buffers
   scale with #gaussians × #cameras). `radius_clip` didn't rescue batch 256.
3. The legacy full-res+CPU-resize path *loses* throughput with batch (CPU resize
   serializes); direct low-res render with scaled K is ~4–5× faster than legacy.
4. Consequence for training-strategy math: 1 M rendered frames ≈ 35 min; 10 M ≈ 5.5 h;
   400 M (UZH pixel-PPO scale) ≈ 9 days of pure rendering — **raw-pixel PPO at
   literature scale is out; render budgets must stay ≤ ~10–30 M frames**, i.e.
   distillation or BPTT-style approaches, or heavy cache reuse.
5. Not yet tried (headroom if needed): scene pruning (opacity/size threshold on the
   1.6 M gaussians), fp16 gaussian params, rendering RGB at sh_degree 0 only (already
   done), splitting batch across CUDA streams.

Geometric note: rendering directly at 128×96 with K scaled by (out/full) is the same
fisheye projection as render-big-then-resize, minus the anti-aliasing of the box
filter; if aliasing shows up as a sim2real term we can render 2× and avg-pool on GPU
(still ~4× cheaper than legacy).

## 3. Training plant — REBUILD (this is the hard gap #1)

The current "plant" (`train_ctrl_lya_pt.py:537-569`) is a **kinematic single
integrator on pose with a velocity actuator model**: action → body→world rotate →
+noise 𝒩(0,0.02²) → 1-step FIFO delay (0.1 s) → first-order lag τ=0.15 s → integrate;
pitch/roll frozen at per-rollout DR values; dt = 0.1 s; horizon 7→25 curriculum.

**There is no rigid-body/attitude dynamics, no gravity, no thrust, no inertia, no
motor model anywhere in any repo** (verified by grep). A CTBR policy cannot be trained
against this plant. We must build a quadrotor rigid-body sim (and identify/randomize
its parameters: mass, inertia, thrust map, motor τ, PX4 rate-loop response).

Reusable *patterns* from the trainer: closed-loop BPTT rollouts with privileged-pose
losses and detached images; horizon curriculum; `ImageCache` (`train_ctrl_lya_pt.py:344`,
pose-rounded-to-0.01 key, FIFO eviction, cleared on per-epoch intrinsics jitter);
per-epoch camera DR `jitter_camera_model()` (`:468`). Caveat: the image cache's hit
rate will collapse for attitude-varying CTBR rollouts (6-DOF pose visits repeat far
less than yaw-only kinematic ones) — plan around batched fresh renders instead.

## 4. Controller & Lyapunov nets — REUSE the spatial encoder pattern, new head/IO

- `Controller` (`utils_ctrl_lya_pt.py:17-113`): 6-ch mean-sub input trick → 4-conv
  backbone → 64×6×8 map → global/lateral/vertical pooled readouts (120-D) → MLP head
  → `clamp_relu` bounded 4-D velocity action. 58k params, all CROWN/TFLite-safe ops.
- What transfers to CTBR: the mean-sub input, the conv trunk, the pooled-readout
  philosophy (each pooling's invariance chosen for a control cue), `clamp_relu`
  bounded outputs, dividing-pool export constraint (`:63-69`).
- What must change: **single RGB frame in, 4-D velocity out** becomes
  **image(s)+IMU+action-history in, [thrust, ωx, ωy, ωz] out**, with temporal context
  (stack or recurrence) and different output scaling. The head and input plumbing are
  new; the trunk is a starting point.
- `Lyapunov` (`utils_ctrl_lya_pt.py:224-300`, 1,185 params) — reusable as-is as a
  *monitor* concept; whether V keeps its role in phase 1 is a strategy question
  (attitude states would need to enter V for CTBR convergence claims).

## 5. Image & camera DR — REUSE AS-IS

- `DomainRandomizer` (`utils_ctrl_lya_pt.py:119-218`): geometric affine, gamma,
  contrast, saturation, per-channel gain, exposure+clamp, blur, noise, cutout — GPU,
  per-sample, no_grad. Action-agnostic; battle-tested through a real sim2real
  crossing. Grayscale variant needed if we drop to gray input (color items collapse
  to gain/contrast).
- Camera-model DR: per-epoch fx/fy ±0.4%, cx/cy ±0.5 px, mount ±0.5°
  (`train_ctrl_lya_pt.py:468-481`). Reuse; consider widening mount DR for a
  down-tilted or tracking camera if the camera choice changes.

## 6. Export chain — REUSE with new I/O signature

`scripts_tflite/export_to_tflite.py`: PT → ONNX opset 18 → onnx2tf
(`disable_group_convolution=True`) → TFLite **fp16** (132 KB today; int8 rejected,
0.35 err). Encoded gotchas that still bind:
- pooling bin counts must divide the feature map (the 1×3 corruption bug);
- `torch.norm` → `sqrt(sum(x²)+ε)` for onnx2tf ReduceL2;
- fused-graph tensor identification onboard is by rank/name/shape, not index.
New for this phase: recurrent state or frame-stack tensors must survive onnx2tf →
TFLite 2.8 (GRU may need to be expressed as explicit matmul/sigmoid ops with state as
an explicit input/output tensor — to be validated in a spike before committing to
recurrence).

## 7. Onboard + offboard deployment plumbing — REUSE with targeted swaps

- `ctrl_lya_model_helper.cpp`: camera MPA pipe (`/run/mpa/hires_small_color/`) →
  YUV→RGB → resize → NHWC float tensor; a second pthread reads a pose pipe
  (`vio_reader_loop`, `:196-273`) — **this thread pattern is exactly what an IMU
  reader needs** (`/run/mpa/imu_apps` instead of `vvhub_body_wrt_local`).
- Output `CtrlLyaMsg` (`.h:27-33`): 40 B, magic `'CLYA'`, **6 float action slots (only
  4 used)**, V, camera-timestamp ns → `[thrust,ωx,ωy,ωz]` fits with **no wire-format
  change**; spare slots can carry an OOD/health flag.
- `mpa_reader.c` magic-scan → stdout: reuse unchanged.
- `ctrl_lya_offboard.py`: keep the entire safety skeleton (pilot-in-the-loop
  arm-skip, staleness watchdog 0.4 s / handoff 5 s, pilot-takeover detection, HOLD
  handoff, dual CSV logging). Swap the single command call
  `set_velocity_body(VelocityBodyYawspeed…)` (`:363-370`) →
  `set_attitude_rate(AttitudeRate(roll°/s, pitch°/s, yaw°/s, thrust₀₁))` —
  **MAVSDK-Python 2.8.0 in the existing docker image already provides it** (README
  even references an `offboard_attitude_rate.py` template that was never written).
- ⚠ Failsafe semantics change fundamentally: "re-send zero action" is a safe hover
  for velocity control but **zero body rates + stale thrust is not a hover**, and
  offboard.stop()→HOLD may behave differently with no position estimate. The safe
  fallback under rate control (likely: RC takeover in Stabilized/Altitude + a
  hover-thrust-hold or immediate handoff) is a design item, pending the PX4/VOXL
  deployment research.
- `telemetry.attitude_euler()` is already subscribed for logging (`:204-221`) — PX4's
  attitude estimate is reachable via MAVSDK; whether the *policy* consumes PX4
  attitude (vs raw IMU vs nothing) is a strategy decision.

## 8. Evaluation pipeline — REUSE AS-IS (mocap becomes instrument-only)

`record_flight.py` (raw Motive `/poses` → CSV) + `plot_flight.py` (gate-frame
transform, clock auto-align, auto-trim, `_rollout.mp4`/`_traj.png`/`_V.png`/
`_actions.png`) need zero changes to evaluate CTBR flights; extend `_actions.png` to
plot thrust+rates. The `mocap_to_px4_bridge` ODOMETRY path is what gets *removed
from the control loop* — keep the workspace for ground truth only.

## 9. Hard gaps (confirmed absent, must be created)

1. **Quadrotor rigid-body dynamics + PX4-rate-loop + motor model** — nothing exists;
   no PX4 param dump or .ulg log exists anywhere to identify from. Need: mass,
   inertia, thrust/weight & thrust map, motor time constant, rate-loop response;
   sources = ModalAI docs/PX4 params (deployment research), a bench/log-based ID
   pass, and generous DR in the meantime.
2. **IMU ingestion** — no code anywhere reads gyro/accel (train or deploy). Need: sim
   IMU model (gyro bias/noise, accel bias/noise/gravity) + onboard MPA IMU reader +
   timestamp alignment with frames (both MPA CLOCK_MONOTONIC — same clock domain,
   good).
3. **CTBR offboard runner** — template referenced in Starling2 README but never
   written; to be built on the ctrl_lya_offboard.py skeleton.
4. **Rate-mode failsafe story** — see §7 ⚠.

## 10. Environment facts (verified today)

- `imcooked` env: torch 2.7.1+cu128, CUDA OK on RTX 5080 Laptop 16 GB, gsplat 1.4.0.
- Export env `pftolite` (a.k.a. tfexport listed in conda) for onnx2tf/TFLite.
- Repo branch for this phase: `pixel2ctbr` (created today from `Krishna_good`).
