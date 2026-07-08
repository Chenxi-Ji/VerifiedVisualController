# 06 — Verification (pixel2ctbr)

*What is verified, how, and what gates remain before hardware. Updated as
evidence lands. Principle inherited from the legacy project: every sim2real
crossing step gets its own measured checkpoint.*

## A. Verified so far (all on 2026-07-07, commands reproducible)

| # | Claim | Evidence |
|---|---|---|
| A1 | Rigid-body CTBR plant is correct | `test_dynamics.py` 22/22: quat math vs scipy (5e-7), hover fixed point, free-fall g·t²/2, τ_ω step response, delay FIFO exactness, BPTT grads finite |
| A2 | Control problem feasible across DR box | `test_expert.py`: geometric PID expert, no knowledge of per-episode params — 100% success, worst 0.9 cm (measured-range DR); 99.2% on the older pessimistic ranges |
| A3 | Thrust-map error needs integral action/adaptation | PD-only expert: 27.3% w/ 37 cm bias (= gain err·G/kp, analytic match); +integrator → 94–100%. Motivates recurrence in the policy |
| A4 | Splat bridge geometrically faithful | `test_render_bridge.py`: CAM_AXES quaternion exact; vs legacy renderer 0.1 px phase-corr alignment; supersample=2 closes AA gap (0.042→0.0155) |
| A5 | Render throughput sufficient | bench_render.py: 500–570 img/s raw, 295 img/s at policy res w/ AA — distillation ≈ 17 min, BPTT 1–9 h budgets |
| A6 | Recurrent policy exports to TFLite | spike + `export_policy.py`: GRUCell chain OK, **1000-step closed-loop parity 2.5e-3 max action diff** (fp16); found+fixed ReduceMean INT64-axis landmine (mean-sub → AvgPool) and dynamo-exporter default regression (`dynamo=False`) |
| A7 | Full training loop mechanically sound | BC + BPTT smoke runs: collection 400 f/s, gradients flow, eval/save work |
| A8 | Dynamics+bridge+expert cohere visually | `rollout_video.py`: 4-tile video, tilted horizons during transients, all tiles converge to canonical gate-centered hover view, 2–4.7 cm |

## B. Training results (2026-07-07, definitive 512-episode eval)

- B1 ✅ Phase A BC+DAgger: 0.83 m median / 2% crash warm start (bc_run1.log).
- B2 ◐ Phase B through 9 iterations + polish (full diagnosis chain in 05):
  best checkpoint `weights/pixel_ctbr_final.pt`, eval_policy.py, 512
  episodes × 8 s, full DR (`pixel2ctbr/eval_results.json`):

  | condition | success | err_med | p95 | crash |
  |---|---|---|---|---|
  | **base** | **69.1%** | **9.5 cm** | 44.0 cm | 0.0% |
  | no_tilt | 63.5% | 9.3 cm | 40.6 cm | 0.0% |
  | delay +25 ms | 58.2% | 11.7 cm | 57.2 cm | 0.2% |
  | thrust-gain edges | 62.1% | 11.5 cm | 51.7 cm | 0.0% |
  | image-DR off | 70.1% | 9.7 cm | 54.9 cm | 0.0% |

  Reading: median well inside the 15 cm radius; **DR-off ≈ base ⇒ no
  twin-overfit** (legacy gate-swap smell absent); graceful degradation on
  every ablation; zero crashes at 2560 episodes. **Gate (≥95% strict
  composite) NOT yet met** — limiter is the slow-episode tail (p95 44 cm),
  not the typical case. Tail diagnosis is the next training-side task.

### Final round (v10 vs v13, definitive 512-ep fixed-seed evals, 8 s)

  | condition | v13-best (final5) | v10-best (final2) |
  |---|---|---|
  | base | 82.8% / 3.5 cm / p95 20.1 | **84.0%** / 5.6 cm / p95 32.2 |
  | no-tilt | 72.1% (−10.7) | **83.4% (−0.6)** |
  | delay +25 ms | 61.9% (−20.9) | **77.1% (−6.9)** |
  | gain edges | **81.4%** | 78.7% |
  | image-DR off | 80.5% | 77.7% |
  Crashes ≈ 0 across all 5120 episodes.

  **Flight deliverable: v10-best (`weights/pixel_ctbr_final2.pt`)** — the
  headline is a statistical tie, but v13 bought its 3.5 cm precision with
  brittleness (tilt-dependent, 3× the latency sensitivity — its faster
  approaches spend the latency margin). Robustness wins for hardware; the
  exported `pixel_ctbr.tflite` in Starling2 is already built from v10-best
  (parity 4.8e-3). v13-best kept as the precision line; identified next
  training lever: widen delay DR (train at 25–125 ms) to buy the margin
  back, then re-run this table.
  At 12 s horizons both models hover ~87–92% (peak gates 99%); the residual
  8 s gap is arrival-speed for a minority of far/awkward starts.
- Export ✅ `weights/pixel_ctbr.tflite` 264 KB fp16; 1000-step closed-loop
  parity vs PyTorch: max action diff 5.4e-3 (≈0.06% of thrust span) — C1
  passed on desktop (on-device TFLite-2.8 re-check pending, C2).
- Artifact: `pixel2ctbr/spike_out/policy_rollout.mp4` (trained policy, 4
  random DR'd plants, splat camera).

## C. Bench/hardware ladder (ordered gates; each blocks the next)

1. **C1 export parity on trained weights** — export_policy.py, 1000-step
   closed-loop, <1e-2 action units. (Tool ready.)
2. **C2 onboard timing** — model on VOXL2, XNNPACK 1–2 threads pinned cores
   4–6: measure ms/invoke + end-to-end pipe latency (`voxl-inspect-cam`
   style); need ≥30 Hz with total glass→command ≤100 ms (trained DR ceiling).
3. **C3 PX4 param recipe + SITL/props-off bench** — apply deploy/README.md
   recipe; `ctbr_offboard.py` param check passes; verify with
   `px4-listener vehicle_rates_setpoint` that commanded rates/thrust arrive
   scaled correctly; exercise EVERY failsafe branch: inference kill →
   hover-hold frames; stale-exit → offboard.stop() → Stabilized; RC flip mid
   stream; confirm `MUORB_KAF_LAND` keep-alive under full inference load.
4. **C4 hand-held sign check** — motors off, policy live: tilt/translate the
   drone by hand, verify action signs and magnitudes react sanely (legacy
   flipcheck analogue).
5. **C5 thrust trim** — brief tethered/hand-guarded hover in OFFBOARD:
   measure actual hover thrust fraction vs 0.34 map; adjust THRUST_PER_MS2.
6. **C6 first free flight** — pilot-in-the-loop (Stabilized hover → OFFBOARD
   in start box), mocap recording ON (measurement only), evaluate with
   plot_flight.py gate-frame pipeline + `_actions` panel extended to
   thrust/rates.
7. **C7 sim-vs-real comparison** — same plots as legacy first-flight
   (convergence, action traces, latency measurement from logs); feed
   measured latency/τ back into DynParams and retrain if outside DR.

## C2. Eval-protocol revision queue (apply together, then re-baseline)

Changes that would make new eval tables incomparable with the
v10/v13/v14 ones — batched here deliberately, to apply in ONE protocol
revision alongside the next DR update:

1. **Crash threshold z > 1.2 m is ~35 cm below the real floor**: the
   multi-gate scene work measured the mat at gate-frame z ≈ +0.855 m
   (gate_mocap.json + scene z-histogram) — the 1.2 m value predates that
   measurement. Impact on existing tables ≈ none (crash rates were ~0 and
   training uses fixed windows, not termination), but the honest threshold
   is z > ~0.7 (floor minus prop radius). 
2. Wide-delay DR (v14's train-time 25–125 ms) promoted into the standard
   eval DR once flight logs confirm the real latency envelope.

## D. Known-unknowns being carried (with owner)

- Tracking-cam calibration (K, fisheye coeffs, mount extrinsics) — required
  before switching the policy input off hires; until then hires + its 45 ms
  latency is inside the trained delay DR but eats margin. (User task: run
  ModalAI calibration + a splat-render sanity check like test_render_bridge.)
- Motor τ / rate-loop bandwidth system ID — C3/C5 logs (rate-step responses)
  feed `DynParams` refinement.
- TFLite 2.8 runtime compatibility of the exported graph — desktop parity is
  on TF 2.13; C2 must re-run parity on-device (op-version risk documented in
  deployment report).
- Battery-sag thrust drift — DR ±15% covers sag per the ESC's RPM loop
  headroom claim; C5/C6 logs to confirm.
