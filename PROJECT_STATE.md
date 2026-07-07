# Certified Visual Gate Controller — Project State (as of 2026-07-02)

> **2026-07-07 — NEW PHASE ACTIVE on branch `pixel2ctbr`:** the project is
> evolving to mocap-free, VIO-free flight — pixels+IMU → collective thrust +
> body rates (CTBR). This document remains the accurate record of the
> velocity-controller phase it describes; the new phase's state lives in
> `docs/pixel2ctbr/` (00 mission, 01 research report, 02 repo audit,
> 03 strategy, 04 design, 05 implementation log, 06 verification).

## What this is
A ~58k-parameter vision-only neural controller that flies a ModalAI Starling 2 to a
hover point in front of a gate from camera images alone, co-trained with a learned
Lyapunov function V as convergence certificate/monitor. Trained entirely in a
Gaussian-splat digital twin of the real mocap arena; deployed as a 132 KB TFLite
model onboard (~7–10 Hz). Real flights are measured with OptiTrack and evaluated in
the same gate-centered frame / V(t) plots as sim rollouts. **Status: working on
hardware — first instrumented flight converged (2026-07-01).**

## Repos / locations
- `~/certified_visual_controller/VerifiedVisualController_small_clone` — the LIVE repo:
  training, export, plotting, weights (`weights/ctrl_lya.pt`, epoch 120, + `.tflite`), all docs.
- `~/certified_visual_controller/Starling2` — deployment: `voxl-tflite-server` (C++ model
  helper) + `voxl-docker-mavsdk-python` (mpa_reader.c, ctrl_lya_offboard.py, docker).
- `~/mocap_ws` — ROS 2 Jazzy workspace: `motion_capture_tracking` (OptiTrack → `/poses`)
  + `mocap_to_px4_bridge` (→ MAVLink ODOMETRY) + `record_flight.py` (trajectory recorder).
- `~/certified_visual_controller/VerifiedVisualController` — deprecated large-architecture
  repo, NOT used.
- Conda env for training/plotting/rendering: `imcooked` (torch, gsplat, cv2, CUDA);
  TFLite export runs in `pftolite`. Drone IP `192.168.1.225`.

## Frames, units, conventions
- Gate-centered frame: origin = gate center, **+y through the gate toward the deploy
  side**, **z DOWN**, +x = right when facing the gate; yaw = −π/2 faces the gate.
- Deploy side = **+y face** (the −y face is washed out in the splat and unlearnable).
- Scene units: ring inner opening 0.88 u = 0.75 m ⇒ **1 u = 0.85 m** (`meters_per_unit`).
- Target pose `[0, +1.5, 0, −π/2, 0, 0]` (≈1.3 m in front of gate, facing it).
- Action = body-FRD `[vx, vy, vz, yaw_rate]`, clamped ±1.0 u/s translation, ±0.3 rad/s yaw.
- **Action is computed from the image ONLY; pose+target feed only the Lyapunov V.**

## Controller network (58,036 params: backbone 48,960 / readouts 536+536 / head 8,004)
- Input RGB 192×256 in [0,1] → concat `[raw, raw − per-image-per-channel mean]` (6 ch;
  exact invariance to global color/brightness cast; linear, verifiable).
- Backbone: AvgPool2 → Conv16(5×5,s2) → Conv32(3×3,s2) → Conv48(3×3,s2) → Conv64(3×3,s2),
  each ·BN·ReLU → 64×6×8 feature map (each cell ≈66×66 px receptive field).
- Three pooled readouts (what each average destroys defines what it reports):
  - global AvgPool(1,1) → 64: fraction-of-image feature mass = gate apparent size =
    distance cue (drives vx); position-invariant; best sim→real transfer (cos 0.88 vs 0.42).
  - lateral: 1×1 conv 64→8 ·BN·ReLU · AvgPool(1,4) → 32: 4 full-height column slabs
    (6×2 cells each); left/right feature-mass asymmetry (drives vy, yaw).
  - vertical: twin with AvgPool(3,1) → 24: 3 full-width row bands (drives vz); without
    it the feature vector is provably blind to vertical gate position.
- Head: Linear(120→64)·ReLU·Dropout(0.1)·Linear(64→4) → clamp_relu (±1.0 / ±0.3),
  `clamp_relu(x,L)=relu(x+L)−relu(x−L)−L` (exact ReLU-only clamp, α,β-CROWN-friendly).
- Pool bin counts DIVIDE the 6×8 map (4|8, 3|6) → export as single native TFLite
  AveragePool (non-dividing 1×3 was a real onnx2tf/TFLite-2.8 corruption bug).
- All ops Conv/BN/ReLU/AvgPool/Linear/Concat; BN folds at export.

## Lyapunov function (1,185 params)
- `V = 3·[α·‖pos_err‖² + (1−α)·(1−cos Δyaw)]`, `α = σ(MLP(‖pos_err‖/2, (1−cos Δyaw)/3))`,
  MLP 2→32→32→1. Yaw-only angles (pitch/roll are DR'd and uncontrollable).
- Measured (2026-07-01, current weights): strictly non-decreasing along all rays from
  target to 50 u (V(1)=1.36, V(5)=57, V(50)=7500, α→1) — valid CLF behavior empirically;
  no structural guarantee (α bound not yet applied).
- Runtime role: monitor only. V_MAX hook in offboard exists but is disabled (1e6).

## Training (train_ctrl_lya_pt.py)
- Closed-loop rollouts through the twin; losses backprop via pose dynamics; images
  detached (privileged-pose supervision). Adam 1e-3 ×0.95/10 ep, batch 32, 120 epochs,
  dt = 0.1 s, grad-clip 1.0, N = 2000 start poses.
- Plant per step (order): controller → body→world (current yaw) → **+ noise**
  𝒩(0, 0.02²) per channel → **1-step transport delay** (FIFO, 0.1 s) → **first-order
  lag** v ← v + α(u − v), α = 1 − e^(−Δt/τ), τ = 0.15 s ⇒ α ≈ 0.49 → integrate
  Xₜ₊₁ = Xₜ + vₜΔt (pitch/roll frozen). Delay = dead time (matches measured ~100 ms
  camera→command latency); τ = actuator response (engineering estimate, not identified).
- Start box (offsets around target): x ±1.5, y −1.0…+1.5 (0.5–3 u from gate),
  z −0.5…+0.4, yaw ±0.6, pitch/roll ±0.20 rad held per rollout.
- Horizon curriculum 7→10→14→18→22→25 (ep 20/35/50/70/95); 3-phase loss weights
  (traj/decrease/final): P1 3.0/0.5/0.05 → P2 linear ramp → P3 0.5/1.5/3.0.
- Losses: ① normalized trajectory progress vs reachable distance; ② Lyapunov decrease,
  two regimes (V>0.02: (ΔV+margin)⁺², margin max(0.1V,1e-3), increases ×5, temporal
  weight →1.4; V≤0.02: push V²→0) + d²V smoothness 0.15 + α-entropy reg 0.1;
  ③ L1 final position + L1 final yaw.
- Image-space DR (per sample, per step, detached): geometric affine p=0.8 (rot ±1°,
  scale ±2%, shift ±1%×2) · gamma ±0.30 · contrast ±0.25 · saturation ±0.30 ·
  per-channel gain ±0.14 · exposure ±0.08 then clamp (AE clipping, survives mean-sub) ·
  blur 3×3 p=0.30 · noise σ=0.03 · cutout p=0.3 (rect 7–18%/dim). p = per-image apply
  probability; no-p items apply always with magnitude drawn from ± range.
- Camera-model DR (per epoch, in the renderer): fx/fy ±0.4%, cx/cy ±0.5 px, mount ±0.5°.
- ImageCache keyed on pose rounded to 0.01, cleared each epoch on intrinsics jitter.

## Digital twin / rendering
- nerfstudio splatfacto scene `Gate_Long_hloc_seq/.../2026-06-11_015308_cleaned`,
  ckpt step-129999; `world_frame.json` bakes the gate frame (verified ~1e-5 reproj).
- Render = gsplat `camera_model="fisheye"` at 1024×768 with the drone camera's measured
  calibration (fx 504.341, fy 503.320, cx 505.485, cy 367.606; reproj 0.37 px), then
  `cv2.INTER_LINEAR` → 256×192 — byte-identical to onboard preprocessing
  (model_helper.cpp resize). Figure `figures/camera_model_old_vs_new.png` shows
  pinhole-vs-fisheye at same pose/K/pipeline (old pre-fix training was 106° rectilinear).

## Export
- `export_to_tflite.py`: FusedModel (Controller ⊕ LyapunovV) → ONNX opset 18 → onnx2tf
  → TFLite **float16**, 132 KB. Inputs image(1,192,256,3)/pose(1,6)/target(1,6);
  outputs action(1,4) + V. PT↔TFLite max diff ≈0.002 (≤0.006 on real frames);
  INT8 rejected (0.35 error). Parity tools: debug_pt_vs_tflite.py, replay_real_video.py.

## Deployment stack
- State: Motive (viewport Y-up, **streams Z-up**) → `motion_capture_tracking` →
  `/poses` (NamedPoseArray, ~240 Hz, best-effort QoS) → `mocap_to_px4_bridge`
  (world_rot [180,0,0]; **body_rot [180,0,180]** since the rigid body was RE-CREATED
  2026-07-01 nose-along-−X — the old asset was [180,0,−90]; a stale value caused a
  Position-mode backwards runaway; mode zero_at_startup) → ODOMETRY 30 Hz → PX4 EKF2
  (EV pos+yaw, height=Vision, mag off; VIO disabled). Watchdog auto-restarts the mocap
  node on NatNet stalls (WiFi dropouts are a known issue; wired Ethernet is the real fix).
- Vision→action: hires 1024×768 fisheye → voxl-tflite-server (CPU/XNNPACK; GPU delegate
  corrupts outputs) → resize 256×192 → fused tflite → CtrlLyaMsg `'=4s7fQ'` 40 B
  {magic 'CLYA', action[6], V, timestamp_ns=camera CLOCK_MONOTONIC} on MPA pipe →
  mpa_reader (C, magic-scan, raw structs to stdout) → `ctrl_lya_offboard.py`
  (MAVSDK in docker; mounts: /tmp, /run/mpa, /etc/modalai, /dev, /statelog) →
  VelocityBodyYawspeed at inference rate; v_mps = action·0.85·SAFETY_SCALE(1.0).
- Onboard model helper: VIO→gate transform is an **identity placeholder**
  (T_R_=I, T_t_=0, no unit scale, no yaw offset) ⇒ onboard V is garbage ⇒ V_MAX
  disabled. Helper target [0,1.5,0,−1.5708]; wrong V degrades monitoring only, never
  flight (action is image-only).

## ctrl_lya_offboard.py (rewritten 2026-07-01, pilot-in-the-loop flow)
- Expects the drone already flying in Position mode; skips arming if armed (arms only
  from ground start). Enters OFFBOARD in place; drone flies itself to the target.
- RC flip out of OFFBOARD ⇒ detected as `pilot_takeover`: stops commanding, closes
  logs, exits. **Never auto-lands.** Ctrl-C / stale stream / mpa_reader death ⇒
  offboard.stop() = PX4 HOLD hover + RC handoff (land only if stop() fails).
- Staleness watchdog: no CtrlLyaMsg for 0.4 s ⇒ zero-velocity hold re-sent at 5 Hz
  (inside PX4 COM_OF_LOSS_T 0.5 s); dead 5 s ⇒ HOLD handoff.
- Always-on logging to /tmp/ctrl_lya_logs (wiped at VOXL reboot): `flight_*_ctrl.csv`
  (t_wall epoch, ts_ns, latency_ms, action, onboard V, cmd m/s, sent/held, event rows:
  offboard_start / pilot_takeover / stale_* / hold_handoff / ctrl_c / land_fallback)
  + `flight_*_telem.csv` (NED pos/vel, yaw, mode, armed @20 Hz).
- Script is baked into the docker image via COPY; the updated copy is pushed to VOXL
  /tmp and run as `python3 /tmp/ctrl_lya_offboard.py` (re-push after reboot).

## Measurement / plotting pipeline
- `~/mocap_ws/record_flight.py`: logs `/poses` raw (Motive frame, Z-up, m, xyzw) to CSV;
  `--duration N` still-capture mode for calibration; `--bodies`, 1 Hz status prints.
- Gate calibration (done 2026-07-01 20:07): two 3 s hand-held stills (gate center +
  ~1 m in front on +y side) → `plot_flight.py --gate-from … --save-gate gate_mocap.json`.
- `scripts_control/plot_flight.py`: mocap → gate frame (ŷ = horizontal(front−center),
  ẑ = −up, x̂ = ŷ×ẑ, /0.85; yaw via fwd = R(q)·B⁻¹·e₁, B = `--body-rot`, default
  **180,0,180** matching the current bridge value); **V recomputed offline** with the
  .pt Lyapunov (onboard V untrusted; `--onboard-v` overlays it, opt-in); drone↔laptop
  clock auto-align via speed cross-correlation, window ±5 s + quality gate vs zero-lag
  (both clocks are epoch/NTP-synced; a wide window once mis-aligned a real flight by
  +12 s), `--t-offset` override; **auto-trim** to [offboard_start − 1 s, first end
  event] from ctrl-log events (`--no-auto-trim`, `--t0/--t1` override). Outputs:
  `_rollout.mp4` (3D traj | V(t) | actions, h264 yuv420p +faststart, test-video style),
  `_traj.png` (3D + top-down over V contour), `_V.png`, `_actions.png`, `_gateframe.csv`.
  End-to-end validated on synthetic data (gate pose + 37.4 s offset recovered exactly).

## First instrumented real flight (2026-07-01, artifacts flights/run1_fixed_*)
- Converged: V 0.68 → 0.017, final error 0.27 u ≈ 23 cm from target, smooth approach
  into the V basin, unsaturated actions near target, camera→command latency ≈100 ms
  (matches the trained 0.1 s delay). Onboard V column ≈150k–480k (garbage, as expected
  from the identity transform). Recording included manual fly-to-start and landing
  tails (now handled by auto-trim + the pilot_takeover flow).

## Sim-to-real history (each failure measured, then fixed)
1. GPU delegate corrupted outputs (V=inf, 0.6 Hz) → CPU/XNNPACK (~7 Hz).
2. Wrong camera (106° rectilinear trained vs 121° fisheye real) → calibrate + fisheye
   render + byte-matched resize.
3. Global color cast → mean-sub input channel; gate-size cue → saturation DR.
4. 592k net memorized the splat gate (gate-swap test; real logits 40× sim) → 58k redesign
   (global pool + tiny readouts). AdaBN removed ~40% of the blow-up, not sufficient alone.
5. −y splat face washed out → deploy side flipped to +y.
6. Altitude blindness (global pool shift-invariant) → vertical readout added.
7. Non-dividing pool (1×3) broke TFLite export → dividing pools (1×4, 3×1).
8. Ideal-plant training rewards oscillation → actuator lag + latency in rollout.

## Docs in the repo
- `SYSTEM_OVERVIEW.md` — full technical explanation (arch, V, training, DR, twin,
  export, deploy, sim2real timeline, how to read result figures).
- `PRESENTATION_NOTES.md` — concise crib sheet + likely Q&A.
- `ROBUSTNESS_REVIEW.md` — evidence-based review of current system + prioritized gaps.
- `FLIGHT_RECORDING.md` — recording/plotting pipeline usage.
- `run.md` — exact command runbook (Phase R recal / 0 setup / A gate cal / B per-flight
  pilot-in-the-loop / C plots / D metrics + sim comparison).
- `SIM2REAL_DIAGNOSIS.md`, `TRAINING_FIXES.md` — June failure analyses (historical).
- `GATE_ARENA_SETUP.md` — partially stale (still says −y deploy side, 300×200/fx≈113
  camera); code and newer docs are authoritative.

## Known open items (state, not plan)
- P1: onboard VIO→gate transform still identity ⇒ onboard V meaningless, V_MAX gate
  disarmed; fix sketch (constants + yaw offset from gate_mocap.json + start pose) is in
  ROBUSTNESS_REVIEW.md; validation would be one flight comparing onboard V vs
  recomputed V (`--onboard-v`).
- P2: image-side OOD monitor unimplemented (pre-clamp |logit|∞: in-dist ≤1.6, broken
  regime 4.6–8.3, threshold ≈2.5; could ship in the unused action[4] slot with no wire
  format change).
- No α,β-CROWN verification artifact exists yet despite the verification-friendly design.
- Training hygiene open: no seed, dead resume block, dead get_learning_rate(), no
  held-out eval rollout, yaw-wrap-unsafe traj/final losses (V itself is wrap-safe),
  no logit-magnitude penalty, structural α bound on V not applied.
- τ = 0.15 s actuator lag is an estimate; commanded-vs-actual velocity in the new logs
  would support system identification.
- Bridge body_rot [180,0,180] applied + rebuilt after the 2026-07-01 rigid-body
  re-creation (old [180,0,−90] caused the runaway); consistent with the successful
  instrumented flight the same evening.
