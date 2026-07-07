# 01 — Research Report: Pixels(+IMU) → CTBR for Quadrotors

*2026-07-07. Compiled from a five-thread research fan-out (anchor-paper deep dive,
literature sweep, training-infrastructure survey, Starling 2 / PX4 deployment
research, local repo audit). Every load-bearing claim carries a source. Sections
marked ⏳ are awaiting a research thread still in flight and will be filled in.*

---

## 1. The anchor paper — Geles et al., RSS 2024 (arXiv 2406.12505)

**"Demonstrating Agile Flight from Pixels without State Estimation"**, Geles,
Bauersfeld, Romero, Xing, Scaramuzza (UZH-RPG). RSS 2024, Outstanding Demo Award.
Drone racing up to 40 km/h / 2 g with **no VIO, no SLAM, no IMU input** —
"directly map pixels to low-level control commands without explicit state
estimation or access to IMU" (https://arxiv.org/abs/2406.12505,
https://arxiv.org/html/2406.12505v1).

What it actually does (details that matter for us):

- **Observation is NOT raw RGB.** The policy consumes an **84×84 continuous
  gate-segmentation mask** (inner gate edges, confidence-valued) + the **last 3
  actions**. No image stack, no recurrence, no IMU, no attitude.
- **Action: 4-D CTBR** (mass-normalized collective thrust + body rates) at
  **50 Hz**, executed by Betaflight's onboard rate loop. CTBR chosen citing the
  action-space benchmark (Kaufmann et al., arXiv 2202.10796): most agile learned
  interface that needs no state estimation.
- **Training never renders an image.** The mask is synthesized *geometrically*
  (project gate edges through a calibrated double-sphere fisheye model,
  occlusion-sorted, 10% of segments corrupted to random places; <100 µs/frame).
  PPO, **asymmetric actor-critic**: critic additionally sees a 20-D privileged
  state (pose, 6-D rotation, v, ω, gate index encoding, next-gate vector).
  **400 M env steps, 100 parallel envs, ~1 day** (state-based baseline: 3 h).
  Reward: progress + perception (keep gate in view: exp(−δ_cam⁴)) − action-mag −
  action-delta − crash; initial-state buffer instead of a curriculum.
- **The symmetric ablation gets 0% success on every track** — the single
  strongest evidence in the literature that pixel-input flight policies need a
  privileged critic.
- **Deployment is OFFBOARD**: 720p60 video link (33 ms) → ground-station RTX
  3090 → SwinV2-B gate segmenter (TensorRT, 4 ms; trained on 80 k labeled
  images, none from the deployment arena) → policy → RF uplink → Betaflight.
  No onboard-compute variant exists.
- Sim2real: dynamics DR ±20% thrust/drag/inertia, ±5% mass, gates ±5 cm; the
  segmentation abstraction absorbs the appearance gap; no latency modeling, no
  motor-delay randomization (first-order motor model in sim, unrandomized).
- Results: 90–100% success across tracks (sim/HIL/real), real Figure-8 100%
  over 20 laps, MGE 0.49 m vs 0.37 m for state-based. Known failure mode: if no
  gate is visible for several frames the policy is lost (3-action history is
  the only memory) — the paper itself proposes recurrence as future work.
- **No code, weights, or dataset released** (checked arXiv, RPG pages, GitHub
  incl. org search). Closest usable code from the same lab:
  https://github.com/uzh-rpg/agilicious (platform) and
  https://github.com/uzh-rpg/rpg_flightning (JAX diff-sim of the follow-up).

**Take for our project.** The paper proves pixels→CTBR closed-loop flight is
learnable and transfers — but its recipe (task-specific mask abstraction, no
rendering in training, RTX 3090 on the ground, 400 M samples) is built around
*racing at scale with offboard compute*. Our constraints (onboard VOXL2 CPU,
photorealistic splat twin instead of a mask simulator, hover first) point to its
*descendants* rather than the paper itself:

- **Heeg, Song, Scaramuzza, ICRA 2025 (arXiv 2410.15979)** — same lab replaces
  PPO with **BPTT through a differentiable sim** (visual features + double-sphere
  camera model): CTBR at 50 Hz, vision policy in **9 minutes** of training,
  real-world stabilization from hand throws. Code: rpg_flightning (GPL-3.0).
- **Dream to Fly (arXiv 2501.14377)** — model-based RL from raw pixels.
- **MonoRace, TU Delft (arXiv 2601.15222)** — mono camera + IMU → **direct motor
  commands, fully onboard**; won the 2025 A2RL race. The design point closest to
  "runs on the drone".

## 2. Can we train inside our Gaussian splat? (training-infrastructure survey)

Headline: **yes — it has been done, with hardware transfer, at least three
times**, and the recipe converges across groups: *own dynamics in PyTorch/JAX,
splat renders as out-of-graph observations, batched rasterization, short-horizon
gradient methods or imitation*.

### 2.1 Direct precedents (drones, splat/NeRF in the loop)

- **GRaD-Nav** (Stanford, arXiv 2503.03984; code
  https://github.com/Qianzhong-Chen/grad_nav): **the closest thing to our plan
  that exists.** Differentiable short-horizon RL (SHAC-style, truncated windows
  h=32, terminal critic, asymmetric privileged critic) through rigid-body
  dynamics **with a PD attitude loop and motor delay inside the graph**, dt
  0.05 s; **gsplat renders per step as observations (not differentiated)**;
  outputs **CTBR**. 128 parallel drones on an RTX 4090, 0.07 s wall per sim
  step, rendering 55.7% of compute (~0.30 ms/render amortized ≈ 3.3 k renders/s).
  Zero-shot transfer to a Pixracer/Jetson drone, 6–7/10 per gate config. Bonus
  trick: splat means double as a free point cloud for collision/reward.
- **SOUS VIDE / FiGS** (Stanford MSL, arXiv 2412.16346; code+data
  https://stanfordmsl.github.io/SousVide/): gsplat scene + light 10-D drone
  model ("FiGS", up to 130 fps unbatched); **behavior cloning from a privileged
  MPC expert**; student consumes **images + optical flow + IMU + partial state →
  thrust + body rates at 20 Hz**; **100–300 k rendered pairs per scene**;
  **105 real flights**, robust to 30% mass change, wind, 60% brightness change.
- **FalconGym 1.0/2.0** (UIUC, arXiv 2503.02198 / 2510.02248; v1 code
  https://github.com/IllinoisReliableAutonomyGroup/FalconGym): NeRF→(2.0)
  splatfacto; pose-estimator + DAgger IL; 2.0 adds an **editable-splat API**
  (move/duplicate gates, ~4 ms/op) to mass-produce track variants; actions are
  body-velocity+yaw — and it flew **98.6% gate success (69/70) on a ModalAI
  Starling 2**. Our airframe, our renderer family, mocap-free flight: strong
  external validation, albeit at velocity level (their drone kept VIO for the
  velocity loop — exactly the dependency we are removing).
- Same pattern beyond drones: **GaussGym** (arXiv 2510.15352, code
  https://github.com/escontra/gauss_gym — gsplat batched inside IsaacGym,
  asymmetric AC RL from RGB, zero-shot A1 stairs), **VR-Robo** (2502.01536),
  **SplatSim** (2409.10161), **NeRF2Real** (2210.04932), **RialTo** (2403.03949),
  **GS-Playground** (2604.25459: ~10 k fps at 640×480 after ~90% gaussian
  pruning — pruning is the big lever).

### 2.2 The renderer-gradient question, settled

**Nobody backprops through the renderer.** D.Va (arXiv 2505.10646) tried
differentiable-renderer SHAC and measured gradient norms exploding past 1e15;
dropping the ∂obs/∂state Jacobian (renderer out of graph) was both stabler and
2–3× faster. GRaD-Nav, GaussGym, VisFly all treat renders as observations.
This matches our legacy trainer's detached-image/privileged-pose trick — the
pattern carries over unchanged.

### 2.3 Sample budgets vs our measured render ceiling

Our bench (02_repo_audit.md §2): ~500–570 raw img/s, **295 img/s** at policy res
with supersampled AA on the RTX 5080 (1.6 M gaussians, unpruned). Literature
budgets:

| Method class | Budget (frames) | At ~300 img/s |
|---|---|---|
| BC/DAgger distillation (SOUS VIDE, FalconGym) | 1e5–3e5 | **minutes–17 min** |
| Short-horizon BPTT/SHAC (GRaD-Nav, Heeg, D.Va) | 1e6–1e7 | **1–9 h** |
| Pixel PPO, hover (extrapolated between Heeg 8.25 M and racing 400 M) | 3e7± | 1–3 days, high variance |
| Pixel PPO, racing-grade (Geles) | 4e8 | ~9–16 days — excluded |

(Headroom if needed: gaussian pruning à la GS-Playground, radius_clip 1–3 px,
camera-rate < control-rate decoupling à la GaussGym/GRaD-Nav.)

### 2.4 Simulator survey (why we build, not adopt)

Aerial Gym 2.0 (BSD-3, active) renders only ray-cast depth/seg, not photoreal
RGB; OmniDrones/Isaac Lab weld rendering to Omniverse; Flightmare is dormant;
gym-pybullet-drones renders 64×48 at 24 fps CPU; RotorPy (MIT, active) is a
good *dynamics-fidelity reference* (rotor drag, motor dynamics) with no
photoreal camera; Crazyflow/rpg_flightning are JAX (awkward with our torch
splat). Every successful splat-training project built exactly what we already
have: own vectorized dynamics + external gsplat calls. (Sources:
https://github.com/ntnu-arl/aerial_gym_simulator, arXiv 2503.01471,
https://github.com/btx0424/OmniDrones, arXiv 2309.12825,
https://github.com/uzh-rpg/flightmare, https://github.com/utiasDSL/gym-pybullet-drones,
https://github.com/spencerfolk/rotorpy, arXiv 2306.09262,
https://github.com/learnsyslab/crazyflow, https://github.com/uzh-rpg/rpg_flightning.)

### 2.5 gsplat facts checked against docs/issues

- fisheye `camera_model` since v1.4.0 (PR #398), OpenCV-fisheye convention;
  distortion coefficients need the 3DGUT path (`with_ut=True`, ≥1.5.2) which
  forbids `packed`/`sparse_grad`. Our measured-K-only usage (residual lens error
  absorbed by camera DR) avoids that constraint.
- fisheye **pose gradients are reported broken** (issue #824) — irrelevant,
  renders stay out of our graph.
- batched multi-camera in one call since v1.0 (~6.4× vs looping at low res,
  docs/batch.md); kernels fp32-only (#277); `radius_clip` called out in API
  docs as the low-res speed lever.
- Official profiling: 171.8 fps forward at 1080p/2.8 M gaussians (TITAN RTX) —
  consistent with our projection-bound ~500/s at low res on a faster GPU.

### 2.6 Differentiable-sim stability catalog (for the BPTT path)

Known failure: exploding/chaotic gradients through long rollouts of stiff
attitude dynamics. Fix catalog with provenance:

- short truncated windows (h≈16–32) + terminal critic — SHAC (ICLR 2022),
  AHAC (2405.17784), D.Va, GRaD-Nav;
- curriculum on horizon — Wiedemann et al. APG (arXiv 2209.13052, code
  https://github.com/lis-epfl/apg_trajectory_tracking) *and* our own legacy
  trainer (7→25 curriculum) — convergent evolution;
- smooth (Huber) losses; simple-dynamics backward under high-fidelity forward —
  Heeg 2410.15979;
- PD attitude loop inside the graph to tame stiffness — GRaD-Nav;
- value-auxiliary + visited-state inits — ABPT (VisFly-Lab, 2603.21123);
- first-order motor-lag smoothing — GRaD-Nav, Heeg.

## 3. Literature sweep: vision-to-low-level-control across groups ⏳

*(Thread still in flight — teacher–student lineage (Deep Drone Acrobatics,
Learning High-Speed Flight in the Wild), Swift's architecture and why it kept
VIO+gate-detector instead of raw pixels, IBVS/optical-flow hover, nano-drone
works, and the 2024–2026 frontier. Will be merged here with per-paper
obs/action/rate/code tables and lessons.)*

## 4. Starling 2 / PX4 deployment facts ⏳

*(Thread still in flight — camera/IMU specs and MPA latencies, MAVSDK
set_attitude_rate semantics and PX4 failsafe behavior without EKF aiding,
VOXL2 inference options and achievable rates, hover-thrust numbers for the
sim's thrust map. Will be merged here; numbers feed 04_design.md and the DR
ranges in dynamics.py.)*

## 5. Synthesis (updated as threads land)

1. **Pixels+IMU→CTBR trained in our splat is not a research bet; it is a
   reproduction** of GRaD-Nav/SOUS VIDE with a better-calibrated camera model
   and a harder deployment target (onboard CPU instead of Jetson/offboard).
2. **The asymmetric critic is non-negotiable** (Geles symmetric ablation: 0%).
   Whatever the algorithm, the value/teacher side sees privileged state; the
   actor sees exactly the deployment observation.
3. **Keep the renderer out of the autodiff graph** (D.Va's 1e15 gradients;
   unanimous practice). Our legacy detached-image BPTT already does this.
4. **Budgets favor first-order/BPTT (1–10 M frames) or distillation
   (1e5–3e5)** on our single GPU; pixel PPO from scratch is the risky outlier.
   This kills strategy S1 (pure pixel PPO) and elevates S2 (teacher–student)
   and the GRaD-Nav-style differentiable-RL variant of S1/S2 hybrids
   (03_strategy.md makes the call).
5. **Temporal context is required** for velocity observability from pixels
   (and thrust-residual adaptation — our own expert experiment reproduced the
   need for integral action quantitatively: PD-only 27%→PID 99.2%). Literature
   ships either short action-history (Geles), GRU/LSTM (GaussGym, forest-nav
   2602.07101), or stacked frames+flow (SOUS VIDE). Our export spike shows
   GRUCell survives the TFLite chain with 1e-4 parity, so recurrence is open.
