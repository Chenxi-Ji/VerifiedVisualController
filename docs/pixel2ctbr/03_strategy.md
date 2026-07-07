# 03 — Strategy: options weighed, decision, rationale

*2026-07-07. Inputs: 00 (mission), 01 (research), 02 (repo audit + benchmarks),
and the feasibility experiments in 05 (expert 99.2%, GRU export parity 1e-4,
render bridge 295 img/s verified). Deployment-numbers thread still in flight;
it can move parameters (rates, camera, τ ranges), not this decision.*

## The strategy space, revisited with evidence

**S1 — end-to-end pixel PPO in the twin.** Killed as the *primary* path.
Evidence: Geles needed 400 M mask-frames (no rendering!) for racing; even
optimistic hover budgets (~3e7 rendered frames) cost days per run on our
measured 295 img/s with high variance per reward/hparam iteration; the
strongest group in the field chose an abstraction specifically to avoid
render-in-the-loop PPO. Retained only as a possible *fine-tune* stage on a
warm-started policy (where budgets shrink orders of magnitude).

**S2 — teacher–student distillation.** Viable and cheap (SOUS VIDE: 100–300 k
frames, 105 real flights; FalconGym 2.0: 98.6% on our airframe). Weakness:
imitation gap under information asymmetry — our expert acts on true v and an
error integral; the student sees pixels+IMU history. DAgger closes the
state-distribution gap, not the realizability gap; the student may need
closed-loop objective pressure to find its *own* (history-based) solution
rather than chasing unrealizable targets.

**S3 — keep velocity net + IMU-only "velocity executor".** Confirmed unsound
by research: no literature instance of IMU-only velocity tracking (drift);
every no-VIO system moves the loop to rates/attitude (Geles/Heeg/MonoRace).
Dead.

**S4 — flow/feature abstraction instead of raw pixels.** Legitimate lineage
(Deep Drone Acrobatics; Geles's mask IS an abstraction). But it inserts an
onboard flow/feature module we'd have to build and benchmark on the VOXL2 CPU,
and the mission is *direct pixel* control; our splat twin + verified DR is
precisely the asset that makes raw pixels credible without an abstraction
layer. Fallback if raw-pixel transfer fails in a way DR can't absorb.

**S5 (new, from research) — GRaD-Nav-style short-horizon differentiable
training.** BPTT through our differentiable rigid-body CTBR plant (built,
tested) with splat renders out-of-graph (bridge built, verified), truncated
horizons + curriculum, privileged losses; optionally a terminal critic
(SHAC-proper) if plain BPTT stalls. Hardware-validated precedent (GRaD-Nav
zero-shot; Heeg real hand-throw recovery in the feature variant); budgets
(1–10 M frames = 1–9 h) fit our GPU; and **it is architecturally the same
pattern as our legacy trainer that already crossed sim2real once** (detached
images, privileged supervision, horizon curriculum) — minimum new machinery,
maximum reuse of proven parts.

## Decision

**Hybrid S2→S5, in two phases, with the asymmetric principle throughout:**

- **Phase A — BC warm start (S2, hours):** roll the geometric expert (softened
  gains, 99.2% success) on the randomized plant; render observations through
  the bridge with full image/camera DR; train the recurrent pixel+IMU policy
  by behavior cloning (+1–2 DAgger rounds under the student's own rollouts).
  Budget ~2–4e5 frames ≈ 20–40 min of rendering per round. Deliverable: a
  policy that hovers in sim, probably imperfectly.
- **Phase B — short-horizon BPTT fine-tune (S5, the main event):** closed-loop
  rollouts through dynamics+renderer, horizons curriculum ~8→32 control steps,
  images detached, losses on privileged state (position/velocity/attitude
  trajectory costs + action smoothness + rate/thrust regularization — Huber
  where quadratic explodes). Warm start from Phase A makes early rollouts
  non-catastrophic (the known cure for BPTT-through-stiff-dynamics
  divergence, and what Bootstrapping-RL-with-IL found). Add a terminal value
  head ONLY if plain truncated BPTT + curriculum stalls — legacy evidence says
  it may not be needed for a hover basin.
- **Evaluation gate to hardware** mirrors the legacy protocol: ≥95% success
  from the start box under full DR in sim; onboard-rate feasibility bench;
  then pilot-in-the-loop flight with mocap as measurement only.

Why not pure S2: the information-asymmetry realizability gap above.
Why not pure S5: cold-start BPTT through attitude dynamics is the documented
failure mode; a BC warm start costs 30 minutes and removes it.
Why not PPO+critic from day one: complexity budget — every moving part
(critic fitting, advantage normalization, entropy tuning) multiplies debug
time on a one-GPU schedule; the literature's working splat pipelines used
either imitation (SOUS VIDE) or windowed BPTT+critic (GRaD-Nav); we take the
simplest composition that the evidence supports and hold the critic in
reserve.

## Observation / action / architecture (frozen for 04_design.md)

- **Obs:** grayscale 128×96 fisheye frame (camera: hires pipeline assumed,
  final call on tracking-cam vs hires pends deployment thread); IMU gyro (3);
  accel (3, low-pass); tilt estimate roll/pitch (2) *with train-time dropout*
  so the policy degrades gracefully if PX4's tilt is unavailable/laggy;
  last action (4). No position, no velocity, no yaw, no target pose — the
  target is implicit in the task (hover at the trained offset in front of the
  gate, facing it).
- **Action:** `[c, ωx, ωy, ωz]`, c ∈ [0, TWR·g] m/s² mass-normalized,
  ω ∈ ±[4,4,2] rad/s, bounded by `clamp_relu` (verified CROWN/TFLite-safe).
- **Net (~60–90 k params):** legacy-pattern trunk on gray input (mean-sub
  1-ch → AvgPool2 → Conv16/32/48/64 s2 → 64×6×8) → the three pooled readouts
  (global 64 / lateral 32 / vertical 24 = 120) → concat [gyro, accel, tilt,
  last-action] → GRUCell(96–128) → Linear head → clamp_relu. Export path
  verified by spike (1e-4 closed-loop parity, 56 KB at similar size).
  Frame-stacking (2–3 frames) is the ablation/fallback if GRU training or
  onboard state-carry proves awkward.
- **Rates:** control 40 Hz target (25 ms), camera at 20–40 Hz per deployment
  findings; delay FIFO covers 50–100 ms, retuned when measured.

## Path to gates (milestone 2, designed-for now)

The same recipe extends: replace the hover target with progress-along-path
losses through gate waypoints (legacy loss ① pattern), add Geles-style
perception-keep-gate-in-view term, extend the start-state distribution to
post-gate states (initial-state buffer). FalconGym 2.0's editable-splat trick
(duplicate/translate the gate) is available to us for track variety without
new scans. Nothing in Phase A/B's architecture blocks this — the policy's
task conditioning changes, not its I/O.

## Risks & pre-committed mitigations

1. **BPTT divergence** → warm start (A), short windows + curriculum, Huber
   losses, grad clip (legacy had 1.0), rate-loop-in-graph already smooth
   (first-order), critic in reserve.
2. **Sim2real thrust map error** → wide TWR/thrust_gain DR (expert survives
   it; policy must too), hover-thrust trim measured on the bench before
   flight, battery-voltage note in deployment doc.
3. **Onboard rate shortfall** (net too slow at 40 Hz) → input already small;
   fallback 20 Hz control with retrained delay model (SOUS VIDE flew CTBR at
   20 Hz), or tracking-cam res-native input.
4. **Rate-offboard failsafe semantics** (stale rate cmd ≠ hover) → deployment
   doc must define: watchdog sends hover-thrust zero-rate frame, then RC
   handoff; verified on the bench before any free flight. (Pends deployment
   thread for PX4 behavior specifics.)
5. **Gate leaves frame during transients** (Geles failure mode) → hover task
   keeps gate roughly centered by construction; add perception penalty in
   Phase B losses; recurrence carries short blind gaps.
