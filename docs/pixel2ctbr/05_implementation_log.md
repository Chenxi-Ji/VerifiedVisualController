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

## 2026-07-07 — deployment report landed → DR retuned, expert 100% ✅

Full verified report merged into 01_research_report §4. Consequences applied:
- `DynParams.randomized` retuned to measured reality: twr 2.0–3.2 (was
  1.6–2.6; real T/W ≈2.6–2.9), τ_ω 0.015–0.06 s (was 0.02–0.10; real rate
  loop ~10–20 Hz bw + torque LPF 5–16 ms + motor 10–30 ms), τ_c 0.010–0.045,
  delay 1–4 steps (tracking-cam path ~20–35 ms glass→cmd).
- Re-ran gates: dynamics 22/22; **expert now 100.0% / worst 0.9 cm** — the old
  4% failure tail lived in the pessimistic slow-τ corner that reality doesn't
  have.
- NOTE: the Phase-A BC run launched earlier trains against the OLD (harder)
  ranges — fine for a warm start; Phase B and any Phase-A rerun use the new.

## 2026-07-07 — deployment scaffold + eval gate written

- `pixel2ctbr/deploy/ctbr_offboard.py` — MAVSDK rate runner on the
  ctrl_lya_offboard skeleton: PX4 param-recipe preflight check, ≥1 s setpoint
  stream before offboard.start(), rate-mode failsafe ladder (stale 0.15 s →
  hover-hold frames at 50 Hz; 0.6 s → offboard.stop() → Stabilized; RC flip →
  takeover), thrust map `thrust01 = 0.34·c/g` clamp 0.60. NOT hardware-run;
  gated by 06 C-ladder.
- `pixel2ctbr/deploy/README.md` — the full param recipe + camera/model-helper
  spec (tracking-cam target with calibration blocker; INTER_AREA preprocessing
  to match trained AA; IMU pipe recipe; 1–2 XNNPACK threads, never GPU
  delegate) + bench ladder.
- `pixel2ctbr/eval_policy.py` — 512-episode gate + ablations (no-tilt,
  delay+1, gain-edges, DR-off overfit check).

## 2026-07-07 — Phase A results (logs/bc_run1.log, weights/pixel_ctbr_bc.pt)

307 k frames (30×64×160, OLD harder DR ranges), 8 epochs, 2 DAgger rounds:
- **Pure BC: textbook compounding-error catastrophe** — bc_loss →0.0000 while
  closed-loop = 0% success, median err 35 m, 98% crash. (Training loss near
  zero + closed-loop failure = distribution shift, exactly the S2 weakness
  called out in 03_strategy.)
- **DAgger cures it order-of-magnitude per round**: round 1 → median 1.92 m,
  15% crash; round 2 → **median 0.83 m, p95 2.2 m, 2% crash** (success
  metric still 0% — threshold is 15 cm).
- Verdict: sane warm start achieved; the residual (~0.8 m hover bias, no
  tight convergence) is the imitation-gap residue Phase B's closed-loop
  objective targets. Phase B launched from these weights with the corrected
  DR ranges (logs/bptt_run1.log).

## 2026-07-07 — Phase B divergence: diagnosis across three runs

**Run 1** (initial losses): epoch-4 eval median err 19.4 m, 17% crash —
*worse* than the warm start. Window losses flat, grads small. Diagnosis:
**short-horizon myopia** — on 0.2 s windows the cheapest position-loss
reduction is to sprint at the target; terminal kinetic energy is never
billed; the visited-state buffer (filtered only on |p|<6 m) then recycles
runaway states as restarts. Patch v2: terminal position+velocity window cost
("arrive gently" — the poor man's SHAC critic), global overspeed penalty
relu(|v|−1.5)², buffer filter |v|<2.5, lr 5e-5.

**Run 2** (v2): epoch-4 eval 9.7 m / 21% crash — halved, still regressing.
Window losses *flat across 5 epochs* while closed-loop collapses ⇒ the
gradients chase something the windows can't measure. Two structural flaws
found:
1. **h=0-blind windows**: every window starts with a zero GRU state — for
   buffer-restart windows that means acting on a moving plant with no
   velocity estimate and a stale last-action input, a condition that never
   occurs in steady flight. Half of all gradients came from this regime
   (Phase A's chunk burn-in existed precisely to mask it).
2. **Unit-translation bug in the curriculum**: legacy horizons 7→25 were in
   dt=0.1 s steps = 0.7–2.5 s of physics; my 8→32 at dt=0.025 s = 0.2–0.8 s.
   The H=8 phase trains on windows 3.5× shorter in *seconds* than anything
   the legacy recipe ever used — position loss mostly irreducible in-window.

Patch v3: 6-step **no-grad burn-in** per window (GRU + plant warm, then the
scored H steps start from a detached state), curriculum floor raised to
**16→24→32→32** (0.4–0.8 s). Run 3 = v3 from the same Phase-A weights
(logs/bptt_run3.log).

Meta-lesson recorded: dt-relative hyperparameters (horizons, delays in
steps) must be translated in *seconds*, not steps, when the control rate
changes 4×.

## 2026-07-07 — research fan-out post-mortem

4 of 5 agent threads delivered (anchor deep-dive, infra survey, deployment —
all merged into 01). The broad literature-sweep thread died twice to
rate-limits/timeouts after its sub-threads completed but before emitting;
§3 was instead written from primary sources directly (Swift Nature PDF read
in full) + cross-references already in hand. Items it would have verified
are marked [UNVERIFIED] inline in 01. Report considered complete.

## 2026-07-07 — Phase B v4 (run 4, logs/bptt_run4.log)

Three fixes stacked on v3, each tied to a measured run-3 symptom:
1. **Gate-visibility filter on restart buffer** (bearing-to-gate within
   ~52° of yaw, in front of gate, sane box) — the camera is the only
   position sensor; restarts that can't see the gate produce noise
   gradients.
2. **Expert-anchor loss** (λ=0.2 Huber on normalized actions vs DAgger-style
   expert labels along the student's own windows) — dense well-conditioned
   gradient that bypasses the plant; bootstrap-RL-with-IL pattern.
3. **clip 1.0→5.0, lr→1e-4** — run-3 grads sat at 2–5 so every update was
   clipped to direction-only at ~¼ the nominal step.

**Run-4 outcome**: crashes ~solved (17.7%→0.5% by epoch 8) but median err
*rose* to 6.1 m — "safe but lost". Live diagnostic (16-drone probe with
per-timestep bearing/altitude/thrust stats) showed the real mechanism: NOT
gate blindness (blind fraction 0–6% throughout) but **slow vertical drift**
— z_med −0.69 m at t=3 s → −4.3 m at t=6 s at |v| under the 1.5 m/s penalty
cap with thrust pinned ≈9.2 m/s². Constant small thrust bias + drift slow
enough that *no loss term at 0.4–0.8 s windows can see it*: over one window
it costs centimeters of Huber; over a 6 s eval it compounds to 4 m. The
policy needs INTEGRAL action (same conclusion as the expert experiment:
PD 27% → PID 99%), and integral behavior is only learnable when training
exhibits the *integrated* error.

## 2026-07-07 — Phase B v5: episode-chained windows (run 5)

Fix: each training episode = fresh reset + **CHAIN=10 consecutive scored
windows**, state AND GRU hidden carried (detached) between windows = 4–8 s
of continuous on-policy flight per chain. Drift accumulates along the chain
exactly as at eval (chains start from rest, matching eval distribution by
construction); late-chain windows see and bill the integrated offset;
gradients stay window-local (BPTT stability preserved); the recurrent state
trains on multi-second histories (integral action learnable). Replaces the
visited-state buffer entirely; burn-in only at chain start; expert anchor
raised to λ=0.4 (its integrator runs along the whole chain, carrying exactly
the integral-action signal). Render cost per scored step unchanged.

Smoke (2 epochs): eval 0.72 m median / 0.5% crash — **first configuration to
improve on the 0.83 m warm start**, and chain losses show drift being billed
(vel/term terms large in late windows). Full run: logs/bptt_run5.log.

**Run-5 outcome**: divergence and drift fixed for good — evals 0.71 m/0
crash (ep 4), 0.65 m/1% success (ep 8), 0.76 m (ep 12): **stable orbit
around the target at 0.4–0.6 m that an 18 s probe shows never converging**
(|v| ~0.2 m/s persists). Not slow convergence — a wander equilibrium.

## 2026-07-07 — v6 (precision regime) didn't break the orbit; v7 does surgery

v6 added a near-target precision loss (exp(−e/0.4)·e², the legacy
"V≤0.02 push V²→0" analogue) + fine-approach chain starts (⅓ of chains
start settled within 0.3 m) + 8 s evals. Epoch-4 eval: 0.69 m — no change.

**Channel-level diagnosis** (settled-state probe, policy vs expert actions
over 100 steps × 16 drones): thrust corr +0.61 at full magnitude (vertical
control fine — the chains fixed it), but **lateral rates wx/wy corr only
+0.28/+0.34 at HALF the expert magnitude** — the policy under-corrects
lateral offsets weakly and noisily. That's the orbit. Architectural cause
candidate: the head read ONLY the GRU state — fresh visual features reach
the rate channels through the recurrent bottleneck alone, whereas
Geles/GRaD-Nav feed current features to the actor directly alongside memory.

v7: **head skip connection** — head input = [h, current fused features].
Grafted so behavior is EXACTLY the run-5 checkpoint at load (old head
weights copied into the h-slice, skip-slice zeroed; zero-init graft
collapsed to open-loop hover in smoke, 96% crash — the exact-behavior graft
starts at run-5 level 0.57 m). Anchor made per-channel with rates ×2
(targets the measured deficit). Run 7 = v7 from the run-5 snapshot
(logs/bptt_run7.log). Epoch-4 eval: 0.67 m — same plateau.

## 2026-07-07 — THE ROOT CAUSE: velocity information (privileged probe)

Ablations killed the remaining suspects: IMU biases zeroed + image DR off
moved the orbit only 0.55→0.47 m — the floor is not observation corruption.
First-principles re-derivation: the legacy image→velocity-command controller
never needed to KNOW velocity (its plant integrated commands); a CTBR policy
must implement the damping term (expert's k_d·v) and can only get v by
remembering previous-frame features through the 96-d GRU bottleneck.

**Decisive probe** (`probe_priv_velocity.py`): append privileged true body
velocity to the proprio vector (sim-only), same recipe, same run-5 init,
8 epochs: 0.53 → 0.39 → 0.40 → 0.31 → 0.20 → 0.164 → **0.142 → 0.113 m
median, 65% strict success, p95 0.32 m, zero crashes** — smashes through the
0.55–0.7 m plateau that seven training-side interventions couldn't dent, and
was still improving at cutoff. Velocity information IS the bottleneck.

## 2026-07-07 — v8: two-frame input + auxiliary velocity head (run 8)

Deployable version of the same information:
- **Input = [current, previous] grayscale frames** (visual velocity by frame
  differencing; conv1 2→4 channels with per-frame [raw, raw−mean] pairs).
  Onboard cost: the model helper caches one preprocessed frame. Graft: old
  conv1 weights into the current-frame channels, previous-frame channels
  zeroed ⇒ load-time behavior identical to run 5 again.
- **Auxiliary velocity head** (train-time only, Linear on [h,z] → body v/2,
  Huber vs true pre-step velocity, weight 0.5; never exported): forces the
  trunk+GRU to actually extract velocity — the probe proved that's the
  convergent representation. Env DRs each frame once and reuses it as
  `previous` (exactly the deployment pipeline's behavior).
- Run 8: 28 epochs × 100 windows from the run-5 snapshot
  (logs/bptt_run8.log). Target: approach the probe's 0.113 m ceiling.

**Run-8 outcome**: broke the plateau (0.68 → 0.47 → 0.44 → 0.42 m by ep 16)
but stalled far above the probe. Physical oversight found by arithmetic: at
40 Hz and hover speeds 0.2–0.5 m/s, **consecutive frames differ by 0.3–0.8
px** — the added velocity signal was sub-pixel, nearly invisible to stride-2
convs. (The probe's velocity was macroscopic; the aux loss sat at 0.009 with
nothing learnable to chew on.)

## 2026-07-07 — v9: 150 ms visual baseline (run 9) — GATE-LEVEL REACHED

Pair the current frame with frame(t−6 steps) = 150 ms ago (env ring buffer;
deployment = ring of ~6 preprocessed frames ≈ 72 KB in the model helper).
Hover-speed motion becomes 1.5–4 px — learnable. Also unfroze trunk BNs
(momentum 0.01): their stats predated the two-frame distribution. Warm start
from run-8.

Result trajectory (evals every 4 epochs): 0.82 (transient: prev-frame
semantics changed 25→150 ms) → **0.318** → **0.239** → 0.397 (oscillation:
BN drift + no lr decay) → **0.124 m / 57% success (ep 20)** → 0.151 m / 45%
(ep 24, the auto-saved one — save-last-not-best flaw noted and fixed).
Aux velocity loss immediately 2–4× larger than run 8 (signal present),
crashes ≈ 0 throughout. **The deployable policy touched the privileged
probe's ceiling** (0.124 vs 0.113 m).

## 2026-07-07 — polish run (weights/pixel_ctbr_final.pt)

From run-9 end: BN refrozen (stats now adapted), cosine lr 6e-5→6e-6, all
epochs H=32, evals every 2 epochs, **best-checkpoint-by-success saving**
(run-9's ep-20 peak was overwritten by a worse final save — fixed).
logs/bptt_polish.log. Best: **75.5% success, 0.092 m median, p95 0.36 m,
0 crashes** at epoch 4; stable 65–75% band thereafter.

## 2026-07-07 (later) — 95% push: tail diagnosis + v10/v11/v12

**Tail diagnosis** (`diagnose_tail.py`, 1024 episodes on the 75.5% ckpt):
NO start-condition or DR pocket — failures uniform. 92% are position-only;
median failing episode parks at 21 cm (just outside the 15 cm ring); 53%
graze 0.15–0.3 m, 30% stall farther, 17% reach-then-leave. Longer-horizon
probe: success 71.4% @8 s → 83.9% @12 s → 86.2% @16 s ⇒ **the tail is
substantially SLOW, not lost** — plus a steady-state parking offset.

**v10** (chain-position weighting = steady-state pressure; sharper ring
precision term; CHAIN 13 = 10.4 s; dual-horizon evals): best **83.9% @8 s /
99.0% @12 s peak gate, median 4–6 cm** — steady-state solved; residual gap
is arrival speed. (Also survived a mid-run laptop suspend, 1m22s — CUDA
context held.)

**v11** (snappier verified teacher kp4.2/ki2.2/katt7 + soft time-outside-ring
loss): REGRESSION — stuck 60–75% with crash flickers; the aggressive teacher
labels conflict with the converged smooth policy at low lr. Teacher reverted
to soft gains. **Recorded negative result.**

**v12** (attribution run: ring loss ONLY, soft teacher, from v10-best) —
running (logs/bptt_v12.log). Deliverable checkpoint meanwhile remains
v10-best `weights/pixel_ctbr_final2.pt`; re-exported → parity 4.8e-3.

Honest framing pinned: the 8 s deadline in the strict gate is OUR design
choice (mirrors the expert's timescale); at 12 s the policy already clears
95%+ in peak gates. If 8 s tops out below 95%, report both horizons rather
than silently moving the goalpost.

## 2026-07-07 (later) — Starling2 repo made hardware-test-ready

Committed `bc7d74e` in `~/certified_visual_controller/Starling2` (user's
uncommitted ctrl_lya edits untouched):
- **`pixel_ctbr_model_helper.{h,cpp}`** — the missing onboard piece: gray +
  INTER_AREA 128×96 (AA-matched to training), frame ring (PIXEL_CTBR_RING,
  default 5 ≈ 167 ms @30 fps), IMU reader thread on `/run/mpa/imu_apps`
  (imu_data_t magic-scan, averaged per frame), 12-D vec EXACTLY matching
  `policy.py::normalize_vec` (tilt slots zeroed — dropout-trained), GRU
  hidden carry with 0.5 s-gap reset, publishes wire-compatible `CLYA` msg
  (mpa_reader unchanged). Wired into enum/factory/model-path (sources are
  CMake-globbed).
- **`ctbr_offboard.py`** — rate-mode runner: PX4 param-recipe preflight,
  ≥1 s setpoint stream, failsafe ladder (0.15 s hover-hold / 0.6 s
  stale-exit → Stabilized / RC-flip takeover / reader-death), CSV logging,
  background telemetry watcher.
- **`pixel_ctbr.tflite`** (264 KB, v10-best weights, parity 4.8e-3) into
  `misc_files/usr/bin/dnn/`.
- **`PIXEL_CTBR_DEPLOY.md`** — config, param recipe, C1–C7 bench ladder,
  v1 limitations. First gate C1 = on-device build + tensor-discovery check
  (cannot compile here — needs voxl-cross + TFLite 2.8 headers).

## 2026-07-07 (later) — "whatever can be coded now" batch

1. **Onboard tilt** (Starling2 `38bd33b`): complementary filter (gyro
   propagate + accel-gravity correct, 1 g-gated) inside the helper's IMU
   thread fills the tilt vec slots — removes the v1 zeroed-tilt limitation
   (~5 pt sim ablation cost). Env knobs `PIXEL_CTBR_TILT[_GAIN]`. Sign
   conventions derived for FRD/z-down and matched to the sim's tilt model.
2. **plot_flight.py CTBR support**: auto-detects ctbr_offboard.py logs
   (`c_ms2` header), new 3-panel actions plot (thrust vs hover-g line /
   body rates + hover-hold marks / sent °/s + thrust01). Verified on a
   synthetic log end-to-end.
3. **Milestone-2 v0 scaffolding** (`env_transit.py` + trainer wiring):
   GateTransitEnv — phase machine (approach wp → commit gate: centered
   <0.12 m, lateral |v|<0.2, yaw<0.12 → exit hover 0.8 m past the plane),
   per-step plane-crossing detection (through-opening vs frame-strike),
   phase-aware `tgt_p` property that transparently retargets the existing
   expert/losses. Splat-validity constraint documented (short overshoot;
   -y face unlearnable). **Expert oracle: first cut 85.9% with 13.7%
   frame-strikes (commit gate on position only → drift over the runway);
   tightened commit gate + shorter runway → 100% success, 100% clean
   crossings, 0.2 cm median exit error (B=256, 13 s).** train_bptt gains
   `--task transit` (transit eval = crossing quality + exit hover at 13 s)
   and a Swift/Geles perception term (keep gate near optical axis, approach
   phase only). Training run pends GPU (v12 in flight).

## 2026-07-07 (later) — v13 consolidation run: new best 85.9%

Design: from v10-best; ring loss RAMPED 0→1.0 over 6 epochs (v11/v12
attribution: the abrupt introduction was the regressor, independent of the
teacher — v12 ring-only also regressed); **distance-scaled speed allowance**
(cap 1.2+0.6·min(dist,2) m/s: fast approach, tight arrival — the flat
1.5 m/s cap was billing transit speed); lr 8e-5 cosine→8e-6, 24 epochs,
CHAIN 13, BN frozen, best-checkpoint saving.

Trajectory (8 s gate): 56.8 (ramp start, = v10's own ep-2) → 80.7 (ep 4! —
v10 needed 12 epochs to reach this) → oscillation band 68–84 through the
mid-run → **85.9% @ ep 18 (new overall best)** → 81–84 tail. 12 s gates
touched 91.7% with p95 8–9 cm; medians tightened to 2.9–3.6 cm. At ep 12
the 8 s p95 hit 13.8 cm — position is inside the ring for ~95% of episodes;
the strict composite is increasingly velocity/yaw-limited at the timestamp.

Deliverable selection: definitive fixed-seed 512-episode evals running on
BOTH v13-best (pixel_ctbr_final5.pt) and v10-best (pixel_ctbr_final2.pt) —
v10-best never had its own definitive pass (the earlier 69.1% table was the
first polish checkpoint). Winner becomes the exported tflite.

## 2026-07-07 (later) — v13-vs-v10 definitive + v14 closes milestone 1

Definitive tables (06 §B): v13 82.8%/3.5 cm but brittle (no-tilt −10.7,
delay+1 −20.9 — its faster approaches spend the latency margin); v10
84.0%/5.6 cm with a flat robustness profile. **v14** = v13-best + train-time
wide-delay DR (25–125 ms; evals unchanged for comparability) + constant ring
weight: best-of-both — **84.2% / 3.1 cm base, delay sensitivity halved
(−10.0), gain-edges and DR-off best-in-family; in-run 12 s gates hit 95.8%
(p95 5.7 cm) twice**. Chosen as the flight deliverable; re-exported
(parity 6.7e-3) and shipped to Starling2 (`cb17430`). The 8 s strict gate
stands at ~84% — the residual is an arrival-speed tail on far/awkward
starts; next levers if sim-side push resumes: even wider delay DR promoted
into eval, speed-cap tuning, or accepting 10–12 s as the operational
convergence budget (12 s meets the 95% bar).

- Definitive 512-episode eval + ablations: see 06_verification.md §B
  (base 69.1% / 9.5 cm / 0 crashes; DR-off ≈ base ⇒ no twin-overfit;
  graceful degradation on no-tilt / +latency / gain-edges).
- Export: one more converter landmine — onnx2tf rejects channel-axis Slice
  ops (the per-frame mean-sub); fixed with an export-time wrapper that
  rebuilds the graph slice-free and permutes conv1 input channels to match
  (numerically proven by the 1000-step parity: 5.4e-3). `pixel_ctbr.tflite`
  264 KB.
- Artifact: policy_rollout.mp4 (trained policy, DR'd plants, splat camera).
- **Open to reach the 95% gate**: tail diagnosis (p95 44 cm — which start
  pocket produces the slow episodes?), possibly +epochs / tail-weighted
  chain sampling / terminal critic. Then the C-ladder (06) to hardware, and
  the milestone-2 gate-trajectory extension (03 §path-to-gates).
- `pixel2ctbr/export_policy.py` — export + 1000-step closed-loop parity
  (random-weights parity 2.5e-3). Found two landmines: torch≥2.9 dynamo
  exporter default breaks onnx2tf (`dynamo=False`); `.mean(dim)` → TFLite
  MEAN(INT64 axis) runtime rejection → mean-sub reexpressed as AvgPool.
- Memory + PROJECT_STATE pointers to the new phase.

## 2026-07-07 (later) — milestone-2 v1: multi-gate tracks via splat editing

Full write-up: 07_multigate_envs.md. The one measured gate becomes N-gate
tracks with FalconGym-2.0's editable-gsplat recipe (box-select in a metric
frame → copy the five gaussian tensors → rigid-transform means+quats →
concat). Read both local FalconGym copies: editing code byte-identical
(mini = code + data + extra plane-DR demos); their Aruco-frame permutation
chain doesn't apply to us — our composed dataparser+world_frame transform
IS the gate frame, so `scene_edit.py` reduces the math to one rigid 4×4 +
uniform scale (rigidity 1.2e-7, round-trip 5e-7 m, identity-duplicate
exact; asserted in __main__). Crop box ((−.72,.72),(−.28,.28),(−.72,.60))
m = 38,839 gaussians: ring+hoop+collar, stand legs EXCLUDED (would drag a
floor patch under every copy) ⇒ duplicates float — accepted. SplatRenderer
gains `scene=`.

`env_multigate.MultiGateEnv` generalizes the transit phase machine to a
waypoint list: per-gate pre-gate commit waypoints, the tuned commit gate
evaluated in each gate's own plane coordinates, per-gate signed-plane
crossing detection with a NEW frame-annulus bound (FRAME_R 0.75 m —
oblique infinite planes otherwise bill phantom strikes). Same
rollout/metrics interface as GateTransitEnv; `--task
two_gate|three_gate_turn` wired in train_bptt (perception term now aims at
the current phase's gate).

Tracks + expert oracles (B=256, DR'd plants, state-only):
- **two_gate** (second gate at y=−2.2 m): **100/99.6/100/99.6%** over
  seeds 5/1/11/42, all crossings clean, exit err ~3 mm. Residual 1/256 = a
  far-corner start overshooting wp₀ through the plane at |x|=0.39 m
  (honest frame-risk; recovers and finishes).
- **three_gate_turn** (40°/gate toward +x, 2.0 m chords — next gate ~20°
  off-axis at each pass; +x arc keeps the track shallow in −y): 87.5% at
  20 s was pure unsettled braking → **100% at 22 s** (EVAL_T 880).

Visual gate (spike_multigate.py, 512×384 color + 128×96 policy-eye):
duplicated rings crisp everywhere, gate 3 visible THROUGH gate 2 from its
pre-wp, gate 2 visible inside gate 1 at policy resolution. Worst imagery:
two-gate exit (y=−3 m facing −y, smeared extrapolation — braking phase
only); three-gate exit faces +x and is much better. Throughput at training
settings: 400→367→316 img/s (pristine/two/three; −8%/−21%, still ≥ the 295
img/s design number).

## 2026-07-07 (evening) — crop-box bottom fix + three-gate re-anchor

User feedback on the multi-gate scenes, two fixes (07 doc §2/§3 updated).
(1) Duplicated gates were visibly sliced at the BOTTOM: the outer wire
hoop closes at gate-frame z≈+0.72, so the +0.60 crop cut the lower hoop
arc, bottom marker and collar off every copy. Iterating z-max with
close-up renders + per-increment pixel diffs exposed a wrong prior — the
floor mat is at z≈+0.855 (mocap-confirmed), not 1.2: legs are short
(0.70→0.86) and the X-feet/tape lines lie ON the mat, so +0.86 already
drags mat/tape fragments under copies (the floor-patch failure mode) while
+0.82 adds exactly the leg ends. GATE_BOX z-max 0.60 → **0.82** (38,839 →
41,405 gaussians, 2.57%): duplicates now carry ring + hoop + collar +
near-full legs ending ~3 cm above the mat — grounded look, no patch.
(2) The turn track's last gate pressed against the +x/−y safety net (old
c₃ (2.42,−2.88), exit (3.20,−3.02); mat/net at ~x∈[−2.3,2.5], y∈[−3.3,3.5]
from splat probes). Re-anchored the arc with the REAL gate as the MIDDLE
gate (env_multigate REAL_GATE; upstream duplicate (0.684,+1.879) at −40°,
downstream (0.684,−1.879) at +40°, same 40°/2.0 m arc): extremes now wp₁
(1.13,+2.41) / exit (1.20,−2.49), ≥~1 m inside the net; start box
re-expressed in gate-1's approach frame (±1.0 m × 0.45–1.25 m runway,
rotated corners on-mat). Oracles: two-gate bit-identical (100%, seed 5);
three-gate v2 **100% on seeds 5/1/11/42**, zero strikes, exit err ~3 mm.
Bonus: the old threegate_past_g1 hires frame (unusable −y smear) is now a
crisp +y-space view; QA + 1024×768 hires renders regenerated. Throughput
432/338/359 img/s (pristine/two/three, v14 sharing the GPU) ≥ design 295.

## 2026-07-07 (night) — splat floater cleanup + showcase orbits

User feedback: blue floaty artifacts around the gates/arena (worst at the
gate BOTTOM) in the multi-gate renders. Point-cloud probe located them:
mat-blue reconstruction fuzz hovering 5–25 cm ABOVE the mat plane
(nothing exists at z∈[0.30,0.60) near the gate — the fuzz is a
z∈[0.60,0.85] layer), sparse mid-air fog (median opacity 0.04), a haze
band ~0.1–0.6 m in front of the door wall, and blue wisps inside the
gate OPENING that `duplicate_gate` copied into every duplicate. Recolor
diagnostics also showed the deep-−y hover layer IS the rendered mat
there (extrapolated surface reconstructed 10–30 cm high) — so wholesale
above-mat deletion was rejected; the far field only sheds its
low-opacity halo (op<0.35).

`scene_edit.clean_floaters(scene)`: deterministic mask from gate-frame
geometry + blue-excess color + opacity (five groups, thresholds in 07
§6.2), **9,768 gaussians = 0.606%**; `clean_scene()` feeds HoverEnv's
default renderer and `multi_gate_scene()` cleans BEFORE extraction
(gate box 41,405 → 39,305; copies lose their blue freight). __main__
asserts determinism, <1.1% budget, and that non-blue gate-ring content
can never be masked. Verified over 3 visual rounds (13 views, pixel
diffs, before/after strips in `spike_out/floater_cleanup/`): gate-bottom
blobs/door haze/shelf streaks/mid-air fog GONE (a shelf drone model
resurfaced from under a streak); mat texture/tape, walls, roof, cables,
towel, furniture intact. Policy-eye 128×96: mean |diff| 0.2–0.4%,
localized — training stats unchanged. Regressions: two-/three-gate
oracles 100% (state-only path untouched), throughput unchanged within
noise (raw 666 vs cleaned 601–630 img/s, ±5% run variance, idle GPU).
QA + hires sets regenerated (`spike_multigate.py` gains `hires` — incl.
a new `single_gate_bottom` close-up — and a raw-vs-cleaned bench row).

Plus: `showcase_video.py` — slow 360° orbit mp4s of all three cleaned
scenes (24 s / 720 frames @ 30 fps, 1024×768 calibrated fisheye, h264
yuv420p +faststart) to `spike_out/showcase/`; elliptical orbits fitted
inside the safety nets (a circle past the outermost gate would leave the
walled capture volume), ring clearance ≥0.95 m, framing probe-verified.
