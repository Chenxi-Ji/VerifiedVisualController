# 05 — Implementation Log (pixel2ctbr)

*Running log, newest at the bottom. Records what was built, why, and what was
verified at each step. Design rationale lives in 04_design.md; this is the diary.*

## 2026-07-07 — branch + docs bootstrap

- Branch `pixel2ctbr` created off `Krishna_good`.
- `docs/pixel2ctbr/00_mission_and_constraints.md` — problem statement, constraint
  analysis (why velocity commands die with mocap, why CTBR), strategy space S1–S4,
  open questions Q1–Q7.
- Research fan-out launched: anchor paper (2406.12505), literature sweep, training
  infra / RL-in-splats, Starling2/PX4 CTBR deployment, repo audit.

## 2026-07-07 — render throughput benchmark (`pixel2ctbr/bench_render.py`)

Decides render-in-the-loop feasibility. Results in 02_repo_audit.md §2. Headline:
**~500–570 img/s ceiling** (1.6 M gaussians dominate; resolution nearly free),
`packed=True` + batch ≤128 required, batch ≥256 OOM. Raw-pixel PPO at literature
scale (400 M frames) would need ~9 days of rendering — excluded; distillation/BPTT
budgets (1–30 M frames ≈ 0.5–15 h) are fine.

## 2026-07-07 — dynamics core built before the strategy decision (deliberately)

Every candidate strategy (S1 end-to-end RL, S2 teacher–student, even S4
flow-abstraction) trains against the same plant: quadrotor rigid body + PX4
rate-loop-in-the-loop + motor/thrust lag + latency + IMU model. None of that exists
in any repo (audit §9). So `pixel2ctbr/dynamics.py` + `pixel2ctbr/imu.py` + tests
are built now, parameterized so research results only change *numbers* (τ_ω, TWR,
delays, noise), not structure.

Decisions taken (provisional values marked ⚠, to be updated from the deployment
research + system ID):

- **Units/frames: SI + gate frame.** Dynamics run in meters/seconds in the
  gate-centered frame (z DOWN — conveniently NED-like, so gravity = +9.81 ẑ and FRD
  body conventions carry over). Scene units (1 u = 0.85 m) appear ONLY at the render
  boundary (`pose_units = pose_m / 0.85`). The old trainer worked in scene units;
  mixing units into physics invites bugs.
- **State**: p(3), v(3) world, q(4) unit quaternion body→world (FRD body), ω(3) body
  rates, plus actuator internal states (filtered ω_track, filtered thrust, command
  FIFO). Batched torch tensors, differentiable end to end.
- **Action**: `[c, ωx_cmd, ωy_cmd, ωz_cmd]` with c = mass-normalized collective
  thrust in m/s² (policy-side semantic; the cmd→PX4-normalized-thrust map is a
  deployment-side calibration). Bounds: c ∈ [0, a_max], a_max = TWR·g with TWR DR'd
  ⚠ (placeholder 2.0 until Starling numbers land); ω_cmd ∈ ±[4,4,2] rad/s initial
  stabilization limits ⚠ (racing works use up to ±8–10; hover doesn't need it).
- **Inner-loop model: closed-loop abstraction, not motor-level.** PX4's rate
  controller runs at ~1 kHz onboard; we model its *closed-loop effect*: body rates
  first-order-track ω_cmd with time constant τ_ω ⚠ (placeholder 0.05 s) + rate
  saturation; collective thrust first-order-tracks c with τ_T ⚠ (placeholder 0.03 s).
  This is the standard CTBR sim2real abstraction (Kaufmann benchmark 2202.10796;
  UZH racing line). Hooks left for an explicit torque/inertia/mixer model (option B)
  if system ID later justifies it.
- **Transport delay**: FIFO on actions in sim ticks, length ⌈latency/dt_sim⌉,
  latency DR'd ⚠ (placeholder 60–120 ms, covers old measured 100 ms camera→command).
- **Aerodynamics**: linear rotor-drag term −k_d·v (k_d DR'd ⚠ 0.0–0.3 s⁻¹). No
  blade-element model; hover ≠ racing.
- **Integration**: semi-implicit Euler at dt_sim = 0.005 s with n substeps per
  control step (control dt 0.02–0.04 s); quaternion update via exact exponential map
  (normalized each step); pitch/roll NOT frozen (that was the kinematic sim's trick —
  the whole point here is attitude dynamics).
- **IMU model** (`imu.py`): gyro = ω + bias(per-episode const, σ_b ⚠ 0.02 rad/s) +
  white noise (σ ⚠ 0.005 rad/s/sample); accel = specific force R⁻¹(a − g) + bias
  (σ_b ⚠ 0.2 m/s²) + noise (σ ⚠ 0.1 m/s²). At hover reads [0,0,−g] (FRD). Optional
  gravity-aligned attitude tilt estimate (roll/pitch from PX4's complementary
  filter) modeled as true tilt + slow drift + noise ⚠ — whether the policy gets it
  is a design decision, but the sim can produce it.

Verification for this step (test_dynamics.py, all must pass before anything builds
on it): hover fixed point; free fall; rate-step response matches τ_ω; yaw rotation
preserves level attitude; quaternion norm stability over 10 s; delay FIFO exactness;
gradient flow (BPTT through 50 steps finite, nonzero); batch-shape/broadcast sanity.

**Result: 22/22 PASS** (`pixel2ctbr/test_dynamics.py`). One test initially failed
for a good reason: BPTT gradient probed *at the hover fixed point*, where the
quadratic loss gradient is analytically zero — fixed by perturbing the action
(θ=0.05) off the fixed point; gradient then 9.2, finite. Quaternion math verified
against scipy `Rotation.from_euler("ZYX",…)` (renderer convention) to 5e-7.

## 2026-07-07 — recurrence export spike (`pixel2ctbr/spike_gru_export.py`) ✅

Question: can a GRU policy with explicit hidden-state I/O survive
PT → ONNX(18) → onnx2tf → TFLite fp16? (If not, the policy must frame-stack.)

Findings:
1. torch 2.12's `torch.onnx.export` now DEFAULTS to the dynamo exporter, whose
   graph onnx2tf 1.19.16 cannot digest (`axes don't match array` on the first
   Conv). **`dynamo=False` restores the legacy exporter and the known-good path.**
   The existing `export_to_tflite.py` predates this default — it will need the same
   flag whenever it's next run in a fresh env. ⚠ footgun recorded.
2. onnx2tf emits `*_float16.tflite` directly; its SavedModel has **no serving
   signature** in this version, so step-3 `TFLiteConverter.from_saved_model` (the
   old script's approach) fails with "Only support at least one signature key".
   Use onnx2tf's own fp16 tflite artifact instead.
3. **`nn.GRUCell` (image-trunk + IMU concat + GRU + head, state as explicit
   input/output tensor): EXPORT OK, 56 KB**, 30-step closed-loop parity vs PyTorch:
   max action diff 7.8e-5, max hidden diff 1.1e-4 — fp16-noise level (deployed
   velocity model flew with 2e-3 parity). Ops are all ancient builtins (FC,
   Logistic, Tanh, Mul/Add, Slice, Concat) — expected fine on onboard TFLite 2.8,
   to be re-verified on-device.
4. Hand-rolled GRU variant hit an input-binding mismatch in the spike harness
   (`Invalid tensor size`); not debugged — moot given (3), fallback not needed.

**Design consequence: recurrence is allowed.** Frame-stack vs GRU is now a pure
learning/sim2real question, not an export question.

## 2026-07-07 — geometric hover expert + plant-feasibility gate ✅

`pixel2ctbr/expert.py` — Lee/Mellinger-style cascade: position PID → desired
specific force → desired tilt+yaw attitude + collective → attitude-error P →
rate commands. Privileged sim state (p,v,q), but deliberately blind to the
per-episode actuator params (thrust_gain, τ's, delay). Roles: feasibility oracle
now, teacher candidate for distillation, debugging baseline forever.

Iteration story (each step measured, `pixel2ctbr/test_expert.py`, B=256
randomized plants, start box from the old project, 40 Hz):
1. PD-only: **27.3%** success, median err 17 cm, zero crashes. Diagnosis:
   steady-state offset = thrust_gain error × G / kp ≈ 37 cm worst case — a PD
   loop cannot reject the ±15% thrust-map DR. **Transferable lesson: any policy
   on this plant needs integral action / adaptation; for the learned policy
   that is exactly what recurrence provides (hidden state can estimate the
   thrust residual from observed accel).**
2. + position-error integrator (ki 1.5, anti-windup ±2): **94.1%**, median
   1.1 cm — but stuck tail. Diagnosed via param correlation: failures cluster at
   τ_ω≈0.09–0.10 s AND delay≈95 ms (≈200 ms combined lag) with katt=8 →
   bounded ~12 cm limit cycle, never a crash.
3. Softened gains (kp 3.5, kd 3.2, katt 5.5): **99.2%**, median 0.2 cm, worst
   7.3 cm, 0 crashes, 95th pct 0.8 cm at 8 s.

Verdict: control problem GREEN across the whole DR box at 40 Hz. Two design
notes carried forward: (a) τ_ω DR upper bound 0.10 s may be unrealistically
slow for PX4's rate loop — revisit with deployment numbers; (b) if the expert
becomes the distillation teacher, keep the softened gains (a teacher that
limit-cycles teaches limit cycles).

## 2026-07-07 — render bridge (`pixel2ctbr/render_bridge.py`) ✅

QuadState → batched low-res fisheye frames, replacing the legacy
render-at-1024-then-CPU-resize path for training. Direct render at policy res
with scaled K, `packed=True`, chunked ≤64/call, per-episode camera DR
(intrinsics + mount as quaternion) folded in. Verification
(`test_render_bridge.py`) caught two real issues:

1. **CAM_AXES quaternion was wrong** (hand-guessed; maxdiff 2.0 vs the matrix).
   Fixed with the value from `Rotation.from_matrix(CAM_AXES)` = (w,x,y,z)
   (−.5,.5,.5,−.5); now exact to 0.
2. **Aliasing gap vs deployed preprocessing**: direct low-res rendering skips
   the box filter of the legacy 1024→256 INTER_LINEAR path — images
   geometrically aligned to 0.1 px (phase correlation, response 0.85) but
   0.042 mean|diff| of speckle. Old sim2real history (fix #2: byte-matched
   preprocessing) says don't ship that gap. Fix: `supersample=2` — render 2×,
   avg-pool 2×2 (≈free: throughput is projection-bound) → parity 0.0155.
   **Deployment spec consequence: the new onboard preprocessing must use an
   area-average resize (cv2 INTER_AREA), not INTER_LINEAR**, so the real
   pipeline low-passes like the training images. Recorded for the model-helper
   rewrite.

Throughput at policy res (128×96, ss=2, chunk 64, RTX 5080): **295 img/s**.
Budget math: distillation (~300 k frames) ≈ 17 min; SHAC/BPTT (1–10 M) ≈ 1–9 h.

## 2026-07-07 — policy + env + both trainers built and smoke-tested ✅

- `policy.py` — 118,932 params; gray mean-sub 2-ch input; legacy trunk minus
  the front AvgPool (at 96×128 the pool would land on 3×4, not the 6×8 map the
  readouts are designed for — caught at spec time); 120-D readouts → +12-D
  proprio → GRUCell(96) → head with hover-centered thrust scaling; zero-init
  head ⇒ exact hover output at init (verified). Bounds clamp verified.
- `env.py` — `HoverEnv`: reset (start box + velocity randomization + per-episode
  DynParams/IMU/camera DR + tilt-dropout mask), `observe()` (render → gray DR →
  12-D normalized proprio), `expert_rollout` (BC data, uint8 CPU),
  `policy_rollout` (differentiable, images detached), `success_metrics`.
  Everything on GPU (device-mixing refactored away). Gray DR = legacy subset:
  affine/gamma/contrast/exposure-clamp/blur/noise/cutout.
- `train_bc.py` — Phase A: expert collection (measured **400 frames/s** incl.
  DR), chunk-32 TBPTT with 8-step burn-in, Huber on normalized channels,
  DAgger rounds with the expert's integrator replayed along student
  trajectories. Smoke: mechanics verified; metrics meaningless at smoke scale.
- `train_bptt.py` — Phase B: window losses (position/velocity-near-target/
  yaw+tilt/action/jerk, Huber), horizon curriculum 8→32, BN frozen from
  Phase A, visited-state restart buffer (ABPT), grad-clip 1.0. Smoke: grads
  flow (gn 0.03→0.96 across horizons), ~3.5 s/window at H=32 B=48 ⇒ full run
  2–3.5 h.
- **Full Phase A launched** (30×64×160 ≈ 307 k frames, 8 epochs, 2 DAgger) →
  logs/bc_run1.log.

## 2026-07-07 — deployment addendum (agent report, partial)

ModalAI px4-firmware fork adds `MC_ROLL/PITCH/YAW_CUTOFF` first-order LPF **on
rate-PID torque output** (active in offboard rate mode); Starling 2 ships
30/30/10 Hz ⇒ extra pole τ≈5.3 ms roll/pitch, ≈15.9 ms yaw. Our τ_ω DR
(20–100 ms) dominates these poles — covered, but noted for system ID. No-mocap
arming recipe: `EKF2_MAG_TYPE=5` (mag present+calibrated but unfused; passes
arming checks; yaw = drifting gyro integral — fine for body-rate policy) or
remove mag + `SYS_HAS_MAG=0`; `EKF2_HGT_REF=0` (baro). MAVSDK
`AttitudeRate(roll_deg_s, pitch_deg_s, yaw_deg_s, thrust_value)` confirmed
(PX4 v1.14.3 / MAVSDK v2.12.2 pins). Full main report re-requested.
