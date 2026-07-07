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
logs/bptt_polish.log. Remaining gap to the ≥95% gate is the fat tail
(p95 ~0.7–1 m: a minority of episodes converge slowly or stall) + the strict
composite criterion (err<0.15 m AND |v|<0.2 AND yaw<0.15 at t=8 s).
- `pixel2ctbr/export_policy.py` — export + 1000-step closed-loop parity
  (random-weights parity 2.5e-3). Found two landmines: torch≥2.9 dynamo
  exporter default breaks onnx2tf (`dynamo=False`); `.mean(dim)` → TFLite
  MEAN(INT64 axis) runtime rejection → mean-sub reexpressed as AvgPool.
- Memory + PROJECT_STATE pointers to the new phase.
