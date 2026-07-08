# MASTER — The Complete Brain Dump (pixel2ctbr)

*2026-07-07. This is the handoff document: everything I know, believe, and
would tell a successor over a long dinner — the mental model, the scars, the
taste. The stage docs (00–07) are the evidence; this is the synthesis. If you
read one document, read this; if you disagree with a stage doc, this one wins
on philosophy and loses on specifics (specifics rot faster).*

---

## 0. What this is, in one breath

A ~126k-parameter recurrent network that flies a real quadrotor (ModalAI
Starling 2) to a hover in front of a gate — and now through multi-gate
tracks in sim — using ONLY the onboard camera and IMU, commanding collective
thrust + body rates directly to PX4's rate loop. No motion capture, no VIO,
no position estimate anywhere in the loop. Trained entirely inside a
Gaussian-splat digital twin of the real arena. As of handoff: **works in sim
(84.2% strict-8s / 95.8% at 12 s, 3.1 cm median, zero crashes in thousands of
episodes), exported and staged on the drone, NOT yet flown.** The bench
ladder (C1–C7 in 06_verification.md) is the remaining path to first flight.

## 1. The central mental model — where every bit of information comes from

A CTBR policy must implicitly reconstruct what the position/velocity loops
used to know. Everything in this project falls out of asking, for each
quantity: *where does the network get it?*

| Quantity | Source | Design consequence |
|---|---|---|
| Position (gate-relative) | The image, directly — gate apparent size/offset | The splat twin must be *calibration-faithful* (measured fisheye K, byte-matched preprocessing). This is the project's moat. |
| Velocity | **NOT observable from one frame.** Visual motion between frames + accel integration | THE bottleneck we hit (runs 5–8). At 40 Hz, consecutive frames move sub-pixel at hover speeds — pair frames **150 ms apart** (ring buffer), plus an aux velocity head at train time to force extraction. Proven by a privileged-velocity probe before building anything. |
| Attitude (tilt) | PX4's rate loop holds commanded rates; tilt observed via IMU gravity direction | Policy gets a tilt estimate with 20% train-time dropout → deployment ships a complementary filter but degrades gracefully without it. |
| Yaw (gate-relative) | The image (gate bearing) | Absolute yaw is irrelevant — the mag can stay unfused (EKF2_MAG_TYPE=5). |
| Thrust-to-weight / battery | **Nobody tells you.** Must be inferred from observed accel vs command | Needs *integral action* — the expert proved it (PD 27% → PID 99%); for the network that's the GRU integrating over long horizons, which only trains if training exhibits *integrated* errors (→ chained windows). |
| Latency | Modeled as a FIFO delay in sim, DR'd 25–125 ms | Fast flying spends latency margin — v13's lesson. Speed and delay-robustness trade off through one budget. |

If you extend this system, run every new idea through this table first.

## 2. The sim2real philosophy (why this has a chance of flying)

Inherited from the previous phase (which DID cross sim2real on hardware,
velocity-command version, 2026-07-01) and hardened here:

1. **Byte-match what you can measure.** Camera K measured (0.37 px reproj);
   render at policy resolution with supersampled AA *because* the onboard
   resize will be INTER_AREA (we control both ends — spec'd together).
   History: sim2real failure #2 of the old phase was a camera-model mismatch.
2. **DR what you can't measure, centered on evidence.** Every plant parameter
   (TWR 2.0–3.2, τ_ω 15–60 ms, τ_c 10–45 ms, gain ±15%, delay 25–125 ms) is
   centered on *verified Starling 2 numbers* (shipped PX4 params, ESC source,
   ModalAI fork code — 01_research_report §4) and widened for ignorance.
   DR is not a substitute for measurement; it's interest paid on unmeasured
   quantities.
3. **The renderer stays out of the gradient graph.** Unanimous in the
   literature (D.Va measured 1e15 gradient norms through a differentiable
   renderer); gradients flow through *dynamics only*, images are detached
   observations. Our legacy trainer independently invented this.
4. **The policy trains on the deployment representation.** No mask
   abstraction, no detector — raw (well, DR'd) rendered pixels. The
   professor's own PixelPilot thread proved the alternative fails: policies
   trained on idealized masks got 0% on realistic detector masks even with
   IoU-matched DR. We carry only the splat-vs-reality gap, which this lab has
   crossed before.
5. **Mocap is an instrument, never an input.** The entire recording/plotting
   stack works unchanged; just never start the ODOMETRY bridge.

## 3. Map of the world

- `VerifiedVisualController_small_clone` (branch **pixel2ctbr**) — everything:
  - `pixel2ctbr/` — `dynamics.py` (differentiable CTBR plant, 22 tests),
    `imu.py`, `expert.py` (geometric PID teacher/oracle), `render_bridge.py`
    (splat → policy frames), `env.py` (hover), `env_transit.py`,
    `env_multigate.py` + `env_two_gate.py` + `env_three_gate_turn.py`,
    `scene_edit.py` (gate duplication + floater cleanup; FalconGym-2.0
    recipe), `policy.py` (the network), `train_bc.py` (Phase A),
    `train_bptt.py` (Phase B — the main trainer, all tasks),
    `eval_policy.py` (definitive gate + ablations), `diagnose_tail.py`,
    `export_policy.py`, `rollout_video.py`, `deploy/` (runner + runbook),
    benches and spikes.
  - `docs/pixel2ctbr/00..07 + MASTER.md` — the paper trail. 05 is the diary;
    read it to understand *why* any line of code looks the way it does.
  - `weights/` — `pixel_ctbr_final6.pt` = **v14, the flight deliverable**;
    `final2` = v10 (robust baseline), `final5` = v13 (precision line);
    `pixel_ctbr.tflite` = exported v14.
- `Starling2` repo — onboard C++ (`pixel_ctbr_model_helper.{h,cpp}` in
  voxl-tflite-server: gray+INTER_AREA, frame ring, IMU thread + tilt filter,
  GRU state carry, CLYA wire msg), `ctbr_offboard.py` (MAVSDK rate runner +
  failsafe ladder), the tflite, `PIXEL_CTBR_DEPLOY.md` (param recipe +
  C1–C7). Wire format unchanged from ctrl_lya ⇒ mpa_reader/docker reused.
- `~/mocap_ws` — evaluation instrument only.
- `Gen-Drone-Racing-Research`, `FalconGym-2.0`, `FalconGym2.0-mini` — the
  group's platforms; sim-only. We mined: gate-editing recipe (adopted),
  plant-DR cross-check (passed), PixelPilot's negative result (validates our
  representation choice). Their render-free gate-mask is the pretraining
  trick to steal if trajectory RL ever needs scale.

## 4. Load-bearing numbers (memorize these)

- Render: **~500 img/s** raw at policy res; **295–432 img/s** with AA through
  the bridge; batch ≤128 packed, ≥256 OOMs (16 GB, 1.6 M gaussians). This
  killed raw-pixel PPO (400 M frames ≈ 9 days) and chose BPTT+distillation
  (1–10 M ≈ hours).
- Plant: gate frame is z-DOWN (NED-like — gravity +9.81ẑ, FRD body, thrust
  along −z_body). 1 scene unit = 0.85 m. Floor at z=+0.855 (measured; the
  1.2 m crash threshold predates this — revision queued).
- Control: 40 Hz (dt 0.025), 5 physics substeps, rate limits ±4,4,2 rad/s,
  thrust head spans 0.1–1.9 g centered at hover.
- Deployment: hover ≈ 0.34 normalized thrust; `thrust01 = 0.34·c/9.81`;
  SET_ATTITUDE_TARGET type_mask=128; PX4 holds stale rates for COM_OF_LOSS_T
  (set 0.3–0.5 s!); VOXL2 inference ~0.3–3 ms on 1–2 A77 threads, never the
  GPU delegate.
- Results: v14 definitive 84.2%/3.1 cm/0 crashes (8 s), 95.8% at 12 s;
  expert oracle 100%/0.6 cm; transit oracles 100% clean crossings.

## 5. The failure museum (every scar, and the transferable lesson)

This section is the most valuable thing I can leave you. Each entry:
symptom → diagnosis → fix → *general lesson*.

1. **Pure BC collapsed** (loss→0, 98% crash closed-loop) → distribution
   shift → DAgger rounds (35 m → 0.83 m median). *Training loss on-policy
   distribution ≠ closed-loop competence. Always eval closed-loop.*
2. **BPTT run 1 diverged** (19 m!) → 0.2 s windows reward sprinting; kinetic
   energy at window end is free → terminal position+velocity cost, overspeed
   penalty. *Truncated horizons need terminal shaping or a critic; "arrive
   gently" must be in the objective.*
3. **The dt-unit curriculum bug**: I copied horizon numbers (8→32 steps) from
   a 0.1 s/step trainer into a 0.025 s/step trainer — 4× shorter in seconds
   than anything that ever worked. *Translate dt-relative hyperparameters in
   SECONDS, never steps.*
4. **h=0-blind windows**: every training window started with a zero GRU state
   mid-flight — a condition that never occurs in deployment. → burn-in, then
   properly: **episode-chained windows** (state+hidden carried detached,
   10×32 steps = 8–10 s of continuous on-policy flight; gradients stay
   window-local). *Train the recurrence on the temporal distribution it will
   face. Chains also made slow drift visible to losses — you cannot learn
   integral action from windows that never show integrated error.*
5. **The 0.55–0.7 m orbit plateau** — seven training-side fixes failed
   identically. Ablations cleared sensor noise. The decisive move: a
   **privileged probe** (feed true velocity, sim-only) — plateau collapsed
   0.55→0.113 m in 8 epochs. Then make the information deployable: two-frame
   input. *When many orthogonal training fixes fail the same way, the
   bottleneck is information, not optimization. Prove it with a privileged
   probe BEFORE building the deployable version.*
6. **Consecutive frames were useless** (run 8 stalled): at 40 Hz and 0.2–0.5
   m/s, inter-frame motion is 0.3–0.8 px — sub-pixel, invisible to stride-2
   convs. → 150 ms frame baseline (ring buffer). *Do the arithmetic on your
   signal's magnitude in PIXELS before feeding it to a CNN.*
7. **Weak lateral rate response** (measured: corr 0.3 at half expert
   magnitude) → the head read only the GRU state; fresh visual features
   reached the rates through a 96-d bottleneck → **skip connection** (head
   reads [h, features]), grafted so load-time behavior was bit-identical
   (zero-init graft collapsed; exact-behavior grafts are the way).
   *Architecture surgery mid-project: always graft to exact equivalence,
   never re-initialize behavior you paid for.*
8. **Clip starvation**: grad norms sat at 2–5 against clip 1.0 — every update
   direction-only at ¼ scale, loss flat. *When losses plateau, check whether
   your updates are being silently truncated before concluding anything.*
9. **Ring-loss shock** (v11/v12 regressed ~15 pts): adding a new loss term at
   full weight to a converged policy at low lr = re-equilibration chaos. →
   **ramp new pressures in** (v13/v14 recovered instantly). And v11 stacked
   TWO changes (teacher swap + ring loss) — v12 existed only to attribute.
   *One lever at a time when cheap; explicit attribution runs when not.*
10. **The teacher's own laziness**: the soft-gain expert settles in 6–8 s;
    the student inherited its timescale through the anchor. A snappier
    teacher, though, destabilized the converged student (v11). *Your student
    can't outrun its teacher via imitation terms; but swapping teachers
    under a converged policy is surgery, not a hyperparameter.*
11. **Speed-vs-latency budget** (v13's brittleness: −20.9 pts at +25 ms
    delay): faster approaches spend the latency margin. → distance-scaled
    speed allowance (2.4 m/s far, 1.2 near) + **train-time-only wide delay
    DR** (evals unchanged for comparability) = v14 kept the speed AND halved
    the brittleness. *Robustness is bought at train time, verified at fixed
    eval time; never change both distributions at once.*
12. **Gate-time evals lie**: v13's 85.9% "best" beat v10's 83.9%, but the
    definitive fixed-seed table + ablations reversed the verdict (v13 was
    brittle). Per-gate evals are 192 episodes of DR luck. *Selection
    decisions come only from the definitive protocol: fixed seeds, 512+
    episodes, THE ABLATION MATRIX. The base number alone is never the
    decision.*
13. **Save-best-not-last**: run 9's peak (0.124 m) was overwritten by its
    final save. Embarrassing, classic. *Checkpoint selection by metric,
    always.*
14. **Assumed floor height was wrong** (1.2 m vs measured 0.855). Caught by
    a subagent that measured instead of trusting my brief. *Every "known"
    constant deserves one measurement. Subordinates who verify premises are
    worth their tokens.*
15. **Export landmines** (each cost an hour; all encoded in
    export_policy.py): torch≥2.9 dynamo exporter breaks onnx2tf
    (`dynamo=False`); `.mean(dim)` → TFLite MEAN with INT64 axis (express as
    AvgPool); onnx2tf's SavedModel has no signature (use its own fp16
    tflite); channel-axis Slice ops break conversion (ExportWrapper builds a
    slice-free graph + permutes conv1 weights, proven by 1000-step
    closed-loop parity). Plus inherited: dividing pools only, fp16 not int8,
    never the GPU delegate, TFLite 2.8 on-device re-check. *The export
    parity harness is not optional tooling; it is the only thing standing
    between you and silently-wrong onboard behavior.*

## 6. How we operate (the method, distilled)

- **Measure, then fix.** Every intervention in this project traces to a
  measured symptom (channel correlations, param-sliced failure rates,
  per-timestep state probes, pixel arithmetic). If you can't say what you
  measured, you're guessing.
- **Feasibility oracles before learning.** The geometric expert is oracle
  (is the plant controllable? 100%), teacher (BC/anchor labels), and
  debugging baseline. Every new task starts by making the EXPERT pass it
  (transit: 85.9%→100% by tightening the commit gate — a control-design
  iteration that cost minutes, not GPU-days).
- **Gates before building on anything.** Dynamics had 22 tests before the
  env existed; the render bridge had sub-pixel parity vs the legacy renderer
  before training on it; exports have closed-loop parity before shipping.
  When a gate fails, the finding goes in the log (the diary is the asset).
- **Protected checkpoints + honest tables.** Best-by-metric saving; one
  definitive eval protocol; revision-queue discipline for anything that
  breaks comparability (06 §C2).
- **Think in budgets**: render throughput → sample budget → algorithm class.
  The whole strategy (BPTT+distillation, not PPO) came from one benchmark
  on day one. Re-run `bench_render.py` if the scene or GPU changes.
- **Fly the pipeline end-to-end early.** The rollout video (expert + splat +
  dynamics, day one) caught coordinate/convention bugs no unit test would.
- **Write while you think.** 05_implementation_log.md is the project's real
  memory. Every failed run is documented WITH its diagnosis; v11's negative
  result is as recorded as v14's win. Successors: keep this habit or lose
  the plot.

## 7. Deployment state & the road to first flight

Everything is staged; NOTHING has run on the vehicle. The ladder
(PIXEL_CTBR_DEPLOY.md, gates C1–C7): on-device build + tensor discovery →
timing (need ≥20 Hz, glass→cmd ≤100 ms) → props-off failsafe exercise
(EVERY branch: hover-hold, stale-exit, RC flip, reader death) → hand-held
sign check → tethered thrust trim (the 0.34 hover fraction is battery-state
dependent) → first free flight with mocap recording → sim-vs-real analysis.

Failsafe philosophy (understand before touching): under rate control a stale
command is NOT a hover — PX4 holds last rates for COM_OF_LOSS_T (ships 1.0 s
= a crash; set 0.3–0.5). Our ladder: 0.15 s → hover-hold frames; 0.6 s →
offboard.stop() → Stabilized (COM_OBL_RC_ACT=2 — Position/Altitude fallbacks
need estimates that don't exist). Two traps verified in source: MUORB
keep-alive starvation triggers a blind descent (don't nice-up inference);
the land detector can false-disarm a low-thrust hover near the floor.
EKF2_HGT_REF ships as 3 (vision) — MUST be 0 (baro) or the EKF is waiting
for mocap that never comes.

After the first flights: fit residuals from real logs (Swift's GP/kNN
recipe — 01 §3), tighten τ_ω/latency DR from measured step responses, then
retrain. That loop beats any further sim-side polishing.

## 8. What's not done, and how I'd do it

1. **The 8 s tail (84% vs the 95% gate).** The 12 s number already clears
   95%. Options in order of my preference: (a) accept 10–12 s as the
   operational convergence budget for hover (it's a hover — nothing racing
   about it) and re-express the gate; (b) another v14-style run with delay
   DR promoted into eval + slightly higher approach speed cap; (c) the
   terminal critic (SHAC-proper) — held in reserve all project, never
   needed; reach for it only if (b) plateaus.
2. **Trajectory training (milestone 2).** Everything is staged: two verified
   multi-gate envs (`--task two_gate|three_gate_turn`), 100% oracles,
   phase-aware targets feeding the existing losses, perception term already
   generalized. My plan: BC warm start from the transit expert (~30 min) →
   chained BPTT exactly like hover (expect the same failure modes — the
   chains/ramp/anchor machinery all transfers) → curriculum single-transit →
   two-gate → three-gate-turn. Known risks with one-line remedies in 07:
   identical-clone gate aliasing (jitter duplicate colors slightly) and
   baked lighting on rotated duplicates (photometric DR already covers
   much). Exit-region splat degradation is real — keep exit hovers short of
   y=−3 and prefer arcs that curve back into captured space.
3. **Tracking camera switch.** Calibrate the AR0144 (fisheye + extrinsics),
   set the bridge's K, retrain (~2 h), re-export. Buys global shutter +
   20–30 ms latency. Do it after the first hires flight, not before — one
   variable at a time.
4. **OOD/health monitor** (the reserved V slot in the wire format). The old
   phase's pre-clamp-logit thresholds don't transfer; recalibrate on the new
   net (in-dist vs gate-occluded/lights-off renders) and ship as a
   soft-abort signal to the runner.
5. **Certification story.** Every op is still CROWN-friendly by
   construction (clamp_relu, no exotics); nobody has run α,β-CROWN on the
   recurrent policy. If that matters academically, the hover task with the
   Lyapunov-style analysis from the old phase is the venue; don't let it
   block flying.

## 9. Things I'd never do again / final warnings

- Never stack two interventions on a converged policy without an attribution
  plan (v11 cost 3 GPU-hours to un-confuse).
- Never trust an in-run eval gate for a selection decision.
- Never write a curriculum in steps without converting to seconds.
- Never assume a physical constant that one measurement could check.
- Never let "the loss went down" substitute for a closed-loop eval, and
  never let a closed-loop base number substitute for the ablation matrix.
- Don't keep polishing sim past the point where a flight would teach more.
  **We are past that point now.** The single highest-information next action
  for this project is gate C1 on the bench, then C3's props-off failsafe
  hour. Sim iterations after that should be driven by real logs.
- The splat is a map, not the territory. It has already fooled us gently
  (floaters, washed −y face, baked lighting). The DR + the old phase's
  history say the transfer should hold — but hold the first flight with the
  humility of someone who has been wrong 15 documented times this project
  (see §5).

## 10. Quick-start (exact commands)

```bash
# env: ~/miniconda3/envs/imcooked (torch+gsplat), tfexport (export chain)
cd ~/certified_visual_controller/VerifiedVisualController_small_clone

# tests / gates
python pixel2ctbr/test_dynamics.py           # plant (22)
python pixel2ctbr/test_expert.py             # oracle 100%
python pixel2ctbr/test_render_bridge.py      # renderer parity
python pixel2ctbr/env_two_gate.py            # transit oracles
python pixel2ctbr/env_three_gate_turn.py

# train (hover, the full recipe = v14 config)
python pixel2ctbr/train_bc.py                                   # Phase A (fresh starts only)
python pixel2ctbr/train_bptt.py --polish --no-ring-ramp --wide-delay \
  --init weights/pixel_ctbr_final6.pt --epochs 16 --windows 104 --lr 5e-5 \
  --out weights/NEXT.pt
# trajectory tasks: add --task two_gate | three_gate_turn (BC-warm-start first)

# the verdict protocol (the ONLY numbers that count)
python pixel2ctbr/eval_policy.py --weights weights/NEXT.pt --episodes 512 --seconds 8

# export + ship
CUDA_VISIBLE_DEVICES= ~/miniconda3/envs/tfexport/bin/python \
  pixel2ctbr/export_policy.py --weights weights/NEXT.pt --steps 1000
cp weights/pixel_ctbr.tflite ../Starling2/voxl-tflite-server/misc_files/usr/bin/dnn/

# fly: follow Starling2/PIXEL_CTBR_DEPLOY.md C1→C7, in order, no skipping.
```

*— end of watch. The diary (05) has everything else. Measure first, ramp new
pressures, protect your checkpoints, and go fly the thing.*
