# 11 — Is this a paper? (viability analysis, 2026-07-07)

*Verdict: plausibly yes — gated on real flights, and requiring careful
positioning against two close neighbors. This doc records the analysis so
the framing survives until writing time.*

## 1. Nearest neighbors (engage head-on, never bury)

| work | what it did | what it did NOT do (our room) |
|---|---|---|
| GRaD-Nav (2503.03984) | pixels→CTBR trained in splats, diff-RL (SHAC-style), zero-shot to hardware | used a depth prior; Jetson-class compute; not IMU-only-proprio; per-config 6–7/10 success |
| SOUS VIDE (2412.16346) | BC in splat (FiGS), →thrust+rates, 105 real flights | inputs include optical FLOW module + PARTIAL STATE (velocity!); BC-only (no closed-loop fine-tune) |
| Geles RSS'24 (2406.12505) | pixels→CTBR, no state estimation | mask abstraction (not raw pixels); OFFBOARD RTX 3090; no IMU by choice |
| MonoRace (2601.15222) | camera+IMU→motor cmds onboard, race winner | racing pipeline; details [UNVERIFIED in our survey]; not splat-twin-trained |
| FalconGym 2.0 (2510.02248) | splat-trained, flew on a Starling 2 (our airframe!) | velocity commands with VIO still in the loop — exactly the dependency we remove |

## 2. The defensible delta (the claim, if flights land)

**Raw pixels + raw IMU → CTBR with no state estimation of any kind (no
VIO/mocap/depth/flow-module/velocity input), running fully onboard a 285 g
CPU-only vehicle, trained entirely in a calibration-matched Gaussian-splat
twin of the deployment arena.** The deployment research found no prior
onboard-NN→CTBR system on VOXL2 at all — that conjunction (no-state +
onboard-CPU + sub-300 g + photoreal-twin-trained) is unoccupied as of
2026-07.

Supporting contributions with standalone value (method section meat):
1. **Privileged-probe diagnosis**: proving the observation bottleneck
   (velocity) with a sim-only privileged input BEFORE building the
   deployable fix — a reusable pattern (05 log runs 5–8).
2. **The sub-pixel finding**: consecutive-frame pairing at control rate is
   informationless at hover speeds (0.3–0.8 px); a ~150 ms visual baseline
   fixes it. Crisp, general, quotable.
3. **Episode-chained windows**: truncated BPTT cannot teach integral action
   unless training exhibits integrated errors; carrying state+hidden across
   detached windows does, at window-local gradient cost.
4. **Precision-vs-robustness via the latency budget**: the v13/v14 tables
   (speed spends delay margin; train-time-only delay-DR widening buys it
   back without touching eval comparability).
5. A documented negative-results trail (v11/v12 attribution) reviewers tend
   to reward when hardware-backed.

## 3. What is REQUIRED before submission

1. **Real flights** (non-negotiable): hover transfer = solid systems paper;
   an onboard two-/three-gate transit = clearly conference-grade.
2. One differentiator executed: multi-gate on hardware, and/or the
   monitored-flight angle (08_lyapunov L1 analysis-V + L3b OOD monitor) —
   no close neighbor ships a runtime monitor story.
3. Honest related-work: GRaD-Nav and SOUS VIDE discussed in the first
   paragraph of related work, deltas explicit (the table above).

## 4. Venue guidance

- With flights: **ICRA / IROS / RA-L** natural fit (systems + method +
  hardware validation + ablations).
- RSS/CoRL bar: needs broader generality (multiple arenas — the splat
  pipeline supports it via re-scan) or the certification thread matured.
- Without flights: arXiv/workshop only — the sim-validated competition
  (GRaD-Nav, VisFly, GaussGym) is too strong.

## 5. The lab-coordination note (do this first)

This project lives in the professor's group's orbit: FalconGym is theirs,
and their PixelPilot branch is the *planned* version of what this repo has
built (their docs list it as thread 3). The strongest and most collegial
path is to frame the paper WITH the lab — as the no-VIO, onboard-CPU
realization of that research line — and that conversation should happen
BEFORE writing. Their M2/M3 findings (mask-trained policies fail on real
detector masks) also strengthen our representation-choice argument and
deserve citation as motivation.

## 6. Already-written assets

Docs 00–10 + MASTER ≈ the methods section and ablation tables; the
showcase/rollout videos ≈ supplementary; eval JSONs are camera-ready-shaped.
The critical path to the paper is the critical path to everything: C1 →
C7, then write.
