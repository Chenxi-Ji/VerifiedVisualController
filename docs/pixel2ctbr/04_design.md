# 04 — System Design (pixel2ctbr)

*2026-07-07. Implements the 03_strategy decision. Numbers marked ⚠ are
placeholders pending the deployment-research thread / system ID; they are
config values, not architecture.*

## 0. One-paragraph summary

A ~80 k-param recurrent network consumes a grayscale 128×96 fisheye frame,
IMU (gyro, accel), an optional tilt estimate, and its last action, and emits
`[collective_thrust, ωx, ωy, ωz]` at 30–40 Hz. It is trained entirely in the
splat twin on a batched differentiable rigid-body plant: first behavior-cloned
from a geometric expert (Phase A), then fine-tuned with truncated BPTT through
the plant with images out-of-graph (Phase B). Deployment reuses the
voxl-tflite-server / MPA / MAVSDK stack with the velocity call swapped for
`set_attitude_rate` and a new failsafe design. OptiTrack remains
evaluation-only.

## 1. Simulation environment (`pixel2ctbr/env.py`)

State/plant: `QuadCTBRDynamics` (verified, 05 log) — quaternion rigid body,
first-order rate-loop tracking (τ_ω), thrust lag (τ_c), thrust-map gain, linear
drag, control-step FIFO delay; dt_ctrl 0.025 s ⚠ (40 Hz; switchable to 0.033
if the camera pins us to 30 Hz), n_sub 5.

Episode:
- Start box (gate frame, meters = units×0.85): x ±1.5 u, y target+(−1.0…+1.5) u,
  z −0.5…+0.4 u, yaw −π/2±0.6, pitch/roll ±0.1 rad, v ±0.5 m/s per axis,
  ω = 0. Same box as legacy (comparability) + velocity randomization (new —
  CTBR must catch moving starts).
- Target: hover at `[0, 1.5 u, 0]`, yaw −π/2 (facing gate). No target tensor
  is fed to the policy — the task is baked (mission doc).
- Horizon: Phase B curriculum 8→32 steps (0.2→0.8 s windows); eval episodes
  8 s.
- Termination (eval only): z > 1.2 m below gate center (floor), |p| > 5 m,
  |tilt| > 80° (training uses fixed windows; termination masks in losses).

DR (per episode unless noted):
- Dynamics: `DynParams.randomized` — TWR 1.6–2.6 ⚠, τ_ω 0.02–0.10 s ⚠,
  τ_c 0.015–0.06 s ⚠, k_drag 0–0.3, delay 2–4 ctrl steps (50–100 ms),
  thrust_gain 0.85–1.15.
- IMU: gyro bias σ 0.02 rad/s, accel bias σ 0.2 m/s², white noise per read,
  tilt drift/noise; **tilt dropout p=0.2 per episode** (feeds zeros) so the
  policy never *requires* the PX4 tilt estimate.
- Camera: per-episode intrinsics ±0.4% f, ±0.5 px c (scaled), mount ±0.5°
  (quaternion-composed); image DR per frame: the legacy `DomainRandomizer`
  reduced to its grayscale-meaningful subset (gamma, contrast, exposure+clamp,
  blur, noise, cutout, small affine) ⚠ magnitudes copied from legacy.
- Latency jitter: delay_steps resampled per episode (2–4).

Rendering: `SplatRenderer` (verified) — direct 128×96 fisheye, supersample=2,
grayscale, ~295 img/s at chunk 64. Renders under `torch.no_grad`, detached
from the policy graph (D.Va lesson; legacy pattern).

## 2. Policy network (`pixel2ctbr/policy.py`)

```
image 1×96×128 [0,1]
  → concat[img, img − mean(img)]                    # 2 ch; legacy invariance trick, gray version
  → AvgPool2 → Conv16(5×5,s2)·BN·ReLU → Conv32(3×3,s2)·BN·ReLU
  → Conv48(3×3,s2)·BN·ReLU → Conv64(3×3,s2)·BN·ReLU  # → 64×6×8 map (identical geometry to legacy)
  → readouts: global AvgPool(1,1)→64 | lateral 1×1conv→8·AvgPool(1,4)→32
             | vertical 1×1conv→8·AvgPool(3,1)→24    # 120-D, dividing pools only (export constraint)
vec = [gyro(3), accel(3), tilt(2), last_action(4)]   # 12-D, normalized (gyro/4, accel/20, tilt/0.5, act pre-scaled)
  → Linear(12→32)·ReLU
GRUCell(120+32 → 128)                                # the only stateful element
  → Linear(128→64)·ReLU → Linear(64→4)
  → scale & clamp_relu:  c = 9.81·(1 + tanh-free clamp_relu(a₀,1)·0.9) ∈ [0.98,18.6] m/s² ⚠
                         ω = clamp_relu(a₁..₃·[4,4,2], [4,4,2]) rad/s
```

Parameter count ≈ 84 k ⚠ (computed at build). All ops on the verified TFLite
list; GRUCell parity 1e-4 (spike). Hidden state is an explicit I/O tensor
onboard; reset = zeros at OFFBOARD entry.

Output-scaling note: thrust head is centered at hover-ish (9.81) with
symmetric clamp reach so a zero-initialized head starts near hover instead of
free-fall — the same trick that made the expert's FIFO prefill matter
(05 log). Exact form finalized in code with a unit test.

## 3. Phase A — expert BC (`pixel2ctbr/train_bc.py`)

- Data: on-policy expert rollouts on the randomized plant (B=64 envs ×
  ~150–300 steps × ~30–60 resets ≈ 3e5 (obs, action) pairs ≈ 20–40 min
  rendering). Store as sequences (GRU needs temporal structure): tensors
  [seq, B, ...] chunked to length 32.
- Loss: Huber(action_pred, expert_action) with per-channel weights
  (thrust/9.81, rates/rate-limit normalized); GRU trained by TBPTT over the
  chunks (state carried, detached at chunk boundaries).
- DAgger: 1–2 rounds — roll the *student*, label with expert at visited
  states, mix 50/50 with round-0 data.
- Gate to Phase B: student-only rollouts ≥80% reach-and-hold (looser than
  final gate; it just has to be a sane initialization).

## 4. Phase B — truncated BPTT fine-tune (`pixel2ctbr/train_bptt.py`)

The legacy trainer's skeleton with the new plant:
- Rollout: window of H control steps (curriculum 8→16→24→32 by epoch ⚠),
  batch 48–64 (render-bound), starts drawn from the start box ∪ a
  visited-state buffer (ABPT trick; refreshed from recent rollouts so windows
  cover the whole approach, not just starts).
- Images rendered no_grad per step; policy + dynamics differentiable;
  gradients flow action→dynamics→state→losses (never through renderer).
- Losses (per step + terminal, Huber-smoothed):
  ① position progress / proximity: ρ(‖p−p*‖) with time-increasing weight;
  ② velocity damping near target: ρ(‖v‖)·exp(−‖p−p*‖);
  ③ attitude/yaw: 1−cos(yaw−yaw*) + tilt penalty beyond 25°;
  ④ action regularization: ‖a−a_hover‖² small + ‖aₜ−aₜ₋₁‖² (jerk; Geles);
  ⑤ perception (phase-B.2, for gates later): keep gate bearing in FOV.
  Weights start from legacy phase-2 ratios, tuned by rollout inspection.
- Optimizer Adam 3e-4 ⚠, grad clip 1.0 (legacy), BN frozen from Phase A
  (stats already DR-calibrated), 20–40 epochs × 2000 windows.
- Reserve (not built unless needed): terminal value head on privileged state
  (SHAC-proper) if window-truncation bias visibly caps performance.

## 5. Evaluation (`pixel2ctbr/eval_policy.py`, feeds 06_verification.md)

- Sim gate: 512 episodes, full DR, 8 s: success = final err < 0.15 m,
  |v| < 0.2 m/s, no crash, ≥95%. Report median/95th err, action smoothness,
  V-style convergence plots (legacy plotting conventions).
- Ablations recorded: no-tilt (dropout robustness), delay +1 step, thrust_gain
  edges, image-DR-off (twin-overfit check — legacy gate-swap lesson).
- Videos: rollout_video.py grid + err/action overlays.

## 6. Export & onboard (`pixel2ctbr/export_policy.py`, Starling2 repo changes)

- Export: PT → ONNX (opset 18, **dynamo=False**) → onnx2tf → **use onnx2tf's
  own `*_float16.tflite`** (spike findings). I/O: image (1,96,128,1 NHWC after
  conversion), vec (1,12), h (1,128) → action (1,4), h_out (1,128). Parity
  harness = spike's closed-loop check, 1000 steps, must stay <1e-3.
- Model helper (new `pixel_ctbr_model_helper.cpp` from ctrl_lya template):
  camera pipe → **grayscale + INTER_AREA resize to 128×96** (AA-matched to
  supersampled training renders — 05 log); IMU thread on `/run/mpa/imu_apps` ⚠
  (pipe name pending deployment thread) with a small ring buffer, gyro/accel
  averaged over the frame interval; hidden-state tensor kept between invokes,
  zeroed on stream (re)start; publishes CtrlLyaMsg with action =
  [c_norm, ωx, ωy, ωz] (wire format unchanged — 6 float slots exist).
- Thrust normalization: sim action c [m/s²] → PX4 normalized thrust via
  `thrust_norm = c / (TWR_measured · g)` ⚠ with a bench-measured hover point;
  battery compensation noted as future work (policy's thrust_gain DR is the
  first line of defense).
- Offboard runner (`ctbr_offboard.py` from ctrl_lya_offboard.py skeleton):
  `set_attitude_rate(AttitudeRate(roll_deg_s, pitch_deg_s, yaw_deg_s,
  thrust_norm))` at message rate; **failsafe redesign**: stale >0.15 s ⚠ ⇒
  send (hover_thrust_trim, 0,0,0); stale >1 s or mpa death ⇒ offboard.stop()
  → RC Stabilized/Altitude handoff; RC mode-flip ⇒ pilot_takeover (never
  auto-land). Exact PX4 no-aiding behavior pends deployment thread; bench
  test (props off) required before flight.
- Safety ladder to first flight: (i) bench: props off, verify rates/thrust
  scaling + failsafes; (ii) hand-held: log actions vs manual motion, sanity
  signs; (iii) tethered/low hover with pilot cover; (iv) start-box flight,
  mocap recording for gate-frame eval via plot_flight.py.

## 7. What is deliberately NOT in v1

- No onboard OOD monitor (P2 stays open; slot reserved in CtrlLyaMsg).
- No Lyapunov co-training — V doesn't extend trivially to attitude states;
  revisit after hover works (CROWN-friendliness preserved in the net anyway).
- No battery-voltage thrust compensation (DR + trim only).
- No gate trajectories (designed-for, not built: §03 path-to-gates).
