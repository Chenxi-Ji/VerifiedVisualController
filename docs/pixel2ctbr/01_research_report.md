# 01 — Research Report: Pixels(+IMU) → CTBR for Quadrotors

*2026-07-07. Compiled from a five-thread research fan-out (anchor-paper deep
dive, literature sweep, training-infrastructure survey, Starling 2 / PX4
deployment research, local repo audit) plus direct primary-source reads
(Swift Nature PDF). Every load-bearing claim carries a source; items a
timed-out sweep couldn't verify are marked [UNVERIFIED] inline.*

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

## 3. The broader vision-to-control lineage

*(Primary sources: Swift read directly from the Nature PDF; teacher–student
and diff-sim entries cross-referenced from the §1/§2 threads. A wider
agent-driven sweep was cut short by rate-limit/timeout issues; the entries
here are the load-bearing ones for our design.)*

### Swift — Kaufmann et al., Nature 620:982 (2023), read from the paper

The strongest existing datapoint for *CTBR policies on real hardware at the
limit*, and the clearest argument about where to put the sim2real burden:

- **Perception is an abstraction, not pixels**: VIO at 100 Hz + a CNN
  gate-corner detector at 30 Hz; corners → 3D gate pose via camera
  resectioning + a track map; fused with VIO in a **Kalman filter**.
- **Policy is tiny**: 2-layer MLP (2×128), input = filtered state + previous
  action, output = **collective thrust + body rates at 100 Hz** — "the same
  control modality that the human pilots use." All onboard (Jetson-class
  computer on the Agilicious platform; total sensorimotor latency 40 ms vs
  ~220 ms for human champions).
- **Sim2real by empirical residuals, not just DR**: perception residuals
  modeled as **Gaussian processes**, dynamics residuals by **k-NN
  regression**, both *fit from real flight data* recorded under mocap; the
  policy is then fine-tuned in the residual-augmented sim. The authors state
  perception residuals were stochastic while dynamics residuals were largely
  deterministic.
- Reward includes a **perception term (keep the next gate in the camera
  FOV)** — the same term reappears in Geles 2024; vision-in-the-loop policies
  need to be *taught* to protect their own observability.
- Stated brittleness: "Swift's perception system assumes that the appearance
  of the environment is consistent with what was observed during training;
  if this assumption fails, the system can fail." Even with an abstraction,
  appearance robustness bounded the system — motivating (a) their detector's
  training-set diversity and (b) our heavy image DR.

**Lessons for us**: CTBR at policy level is proven on hardware at 100 Hz
onboard; previous-action input and perception-aware rewards are standard;
and the *residual-fitting* recipe (fit GP/kNN residuals from real logs, then
fine-tune in-sim) is the natural phase-2 of our sim2real plan once first
flights produce logs — it slots exactly into our existing
mocap-as-measurement pipeline.

### Teacher–student sensorimotor policies (the S2 lineage)

- **Deep Drone Acrobatics** (Kaufmann et al., RSS 2020, arXiv 2006.05768):
  privileged MPC teacher → student on **abstracted vision (feature tracks) +
  IMU**; abstraction chosen explicitly because raw-pixel students transferred
  poorly; acrobatic maneuvers at the platform limit, zero-shot.
- **Learning High-Speed Flight in the Wild** (Loquercio et al., Science
  Robotics 2021, arXiv 2110.05113): privileged teacher → student on **depth +
  state**, trained entirely in sim (Flightmare renders), zero-shot to forests
  at 10 m/s; outputs receding-horizon trajectories (not CTBR) tracked by a
  classical controller. The canonical existence proof that *simulation-only
  vision training transfers* when the observation is chosen well.
- **Bootstrapping RL with IL for vision-based agile flight** (Xing et al.,
  arXiv 2403.12203): BC warm start + RL fine-tune beats either alone for
  vision policies — the exact Phase-A/Phase-B composition we adopted (and
  our Phase-B divergence-then-anchor experience empirically reproduced its
  premise).
- **SOUS VIDE / FalconGym** (§2.1): the same pattern executed inside splats.

### Differentiable-sim CTBR (the S5 lineage)

- **Heeg, Song, Scaramuzza** (ICRA 2025, arXiv 2410.15979, code
  rpg_flightning): BPTT through dynamics + differentiable camera on **visual
  features → CTBR at 50 Hz**, minutes of training, real hand-throw recovery.
- **Wiedemann et al. APG** (ICRA 2023, arXiv 2209.13052, code
  lis-epfl/apg_trajectory_tracking): BPTT beats model-free on tracking with
  10× less compute; stability via curriculum — convergent with our legacy
  trainer's design.
- **GRaD-Nav / D.Va / SHAC / AHAC** (§2.2, §2.6): the stability catalog.

### Classical no-pose visual stabilization (context; none output CTBR)

- **IBVS for hover** (Hamel & Mahony 2002 line): image-moment/spherical-
  projection servoing with attitude from IMU — proves gate-relative hover is
  observable from image features + tilt alone, but assumes an inner velocity
  or attitude loop and known feature geometry; brittle to appearance.
- **Optical-flow landing/hover** (de Croon et al., bee-inspired): flow
  divergence regulates descent/hover without metric state; known oscillation
  instability near touchdown (the flow-gain/height ambiguity) — an argument
  for learned policies with memory over fixed-gain flow laws.
- Nano-drone CNN works (PULP-Dronet etc.) do steering classes, not
  closed-loop thrust/rate control — not load-bearing here.

### 2024–2026 frontier (beyond §2.1's splat-training entries)

- **Dream to Fly** (arXiv 2501.14377): model-based RL from raw pixels for
  drone flight — the UZH answer to sample cost; world model instead of
  abstraction. [Real-flight specifics UNVERIFIED in our sweep.]
- **MonoRace** (TU Delft, arXiv 2601.15222): mono camera + IMU → **direct
  motor commands onboard**, won the 2025 A2RL race — the most aggressive
  onboard end-to-end design point publicly known. [venue details UNVERIFIED]
- **Multi-task quadrotor RL** (arXiv 2412.12442): one policy for
  stabilization + tracking + racing via multi-critic — relevant to our
  milestone-2 extension pattern.

## 4. Starling 2 / PX4 deployment facts (verified against pinned sources)

*Sources pinned: PX4 v1.14.3, MAVSDK v2.12.2, modalai/px4-firmware `voxl-dev`
@12b5c2c6c42, shipped `voxl-px4-params` (D0014_Starling_2.params, SDK 1.7) and
`voxl-esc` files, docs.modalai.com / forum staff posts. Two adversarial
verification passes; every load-bearing claim 2–3 independent confirmations.*

### Sensors
- **Hires = Sony IMX412** (not IMX214), 12.3 MP **rolling shutter** color,
  ~146° D-FOV; our current 1024×768 input is its `hires_small_color` stream;
  stock latency ~45–50 ms (low-latency driver: 11–16 ms for the encode path).
- **Tracking = onsemi AR0144, 1280×800 GLOBAL shutter grayscale, 162° fisheye,
  30 fps stock / 50 max** (staff-confirmed limit), RAW8 `preview` pipe ≈
  15–20 ms glass-to-client — the lowest-latency CV path (what QVIO consumes).
  Dual (C26) or triple (C27) config — check the unit.
- **IMU**: 2× ICM-42688-P — IMU0 owned by PX4 on the DSP (8 kHz FIFO → 800 Hz
  loop); IMU1 → `/run/mpa/imu_apps` at 1 kHz (~976 Hz actual), default FIFO
  drain 100 Hz (≈10-sample batches; raise poll to 500 Hz or write `"read"` to
  the control pipe per frame — the voxl-qvio-server recipe). `imu_data_t` =
  40 B packed, **CLOCK_MONOTONIC — same clock domain as camera timestamps**
  (camera stamps = start-of-exposure; use +exposure/2).
- Baro ICP-101xx on the DSP; no onboard mag (mag on the GPS puck).

### PX4 offboard body-rate path (the load-bearing facts)
- MAVSDK `set_attitude_rate(AttitudeRate(roll°/s, pitch°/s, yaw°/s,
  thrust01))` → SET_ATTITUDE_TARGET **type_mask=128**, rad/s on wire, thrust =
  normalized collective 0–1 → `thrust_body[2] = −thrust`, no clamping.
  MAVSDK auto-resends the last setpoint at 20 Hz; PX4 needs >2 Hz and ~1 s of
  streaming before `offboard.start()`.
- **Body-rate offboard needs NO position/velocity estimate on v1.14**
  (source-verified: offboardCheck.cpp gates only pos/vel/accel setpoint types;
  mode_requirements.cpp asks only angular-velocity + attitude + offboard
  signal). Attitude tilt-init is IMU-only.
- **Param recipe** (no-mocap): `EKF2_HGT_REF=0` (**ships as 3=vision!**),
  `EKF2_GPS_CTRL=0`, `EKF2_EV_CTRL=0`, `EKF2_MAG_TYPE=5` (mag connected+
  calibrated passes arming but is never fused; yaw drifts — irrelevant for a
  body-rate policy) or remove mag + `SYS_HAS_MAG=0`; `COM_OBL_RC_ACT=2`
  (Stabilized); `COM_OF_LOSS_T` 0.3–0.5 s.
- **Failsafe traps**: PX4 **holds the last body-rate setpoint until
  COM_OF_LOSS_T** (default 1.0 s) — a stale aggressive rate command is a
  crash; ModalAI ships **`MUORB_KAF_LAND=1`** (apps↔DSP keep-alive timeout
  1 s ⇒ blind descent) — our inference service must not starve voxl-px4;
  land-detector (`COM_DISARM_LAND=0.1 s`, `LNDMC_ROT_MAX=30°/s`) can
  false-disarm a low-thrust hover near the floor — tether test.
- RC mode-switch always exits OFFBOARD instantly; brief the pilot to flip to
  **Stabilized** (not Position) — safety modes without estimates: Manual/
  Stabilized, Acro, Altitude(baro).

### Rate loop to replicate in sim (shipped Starling 2 tune)
- Rate PID (K=1): roll 0.072/0.171/0.0009, pitch 0.097/0.228/0.0011, yaw
  0.15/0.5/0; attitude P 16/16/2.8; loop at **800 Hz** (`IMU_GYRO_RATEMAX`);
  gyro LPF 80 Hz + D-term 60 Hz + ESC-RPM dynamic notch.
- **ModalAI-fork-only `MC_ROLL/PITCH/YAW_CUTOFF` = 30/30/10 Hz: first-order
  LPF on the rate-PID torque output** (active in offboard rate mode) ⇒ extra
  pole τ ≈ 5.3/5.3/15.9 ms.
- **No shaping of offboard rate setpoints** in mainline (`MC_*RATE_MAX`
  130/130/150 °/s applies only to the attitude controller's output, NOT to
  offboard rate setpoints; no thrust slew; battery comp off).
- Motor τ 10–30 ms [low confidence — RPM-closed-loop ESC, no published
  number]; closed-loop rate bandwidth ~10–20 Hz [estimate]. End-to-end
  command→force on comparable platforms ~35–40 ms.

### Physical / thrust numbers (sim table, confidence-tagged in the source)
- TOW **285 g**; motors 1504-3000KV, 120 mm props, 2S; rotor arms ±0.085/
  ±0.0625 m; κ_yaw 0.05.
- **`MPC_THR_HOVER 0.34`**, thrust curve `rel_thrust = 0.9·s² + 0.1·s`,
  ESC RPM-closed-loop 2000–15000 RPM ⇒ hover ≈ 9300 RPM, per-motor 0.70 N,
  k_T ≈ 8.1e-9 N/RPM², **T/W ≈ 2.6–2.9** [derived, medium confidence].
- Thrust map for deployment: `thrust01 = 0.34·c/g` linear v1; full-curve
  inversion available.

### Onboard inference
- voxl-tflite-server = TFLite **2.8.0** on SDK 1.x (master: 2.17.1) — convert
  against matching TF or hit op-version errors; delegates: CPU=XNNPACK
  8 threads, GPU=OpenCL (documented **custom-model corruption pattern** —
  matches our history), NNAPI=Hexagon (int8 only).
- Benchmarks: MobileNetV1-224 CPU 19.7 ms; small models don't win on GPU.
  **Our ~119 k-param 96×128 net: est. 0.3–3 ms on 1–2 A77 threads
  (pin to cores 4–6, `voxl-set-cpu-mode perf`) ⇒ 30–50 Hz comfortable.**
  Anchors: 200 Hz state-based NN on VOXL2 Mini (2510.04724); FalconGym 2.0's
  U-Net at 8 Hz on a Starling 2.
- **No published system runs an onboard NN → CTBR offboard on VOXL2 — this
  would be a first.** Nearby: NTNU's motor-RPM MLP as a custom PX4 module at
  250 Hz (2503.01471); E2E-Fly 30 Hz CTBR via Betaflight bridge; SimpleFlight
  100 Hz CTBR (2412.11764); SkyJEPA claims 100 Hz minimum for their stack —
  counter-evidence: SOUS VIDE flew CTBR at 20 Hz, E2E-Fly at 30 Hz; our 40 Hz
  target with delay-in-training is defensible.

## 4b. Cross-check: the group's own platform (Gen-Drone-Racing-Research, local)

*(Added 2026-07-07 after mining the professor's repo + branches — the
FalconGym group's research platform. Full agent report in session log;
verdicts:)*

- **Sim-only**: no Starling/VOXL/PX4/MAVSDK/TFLite code on any of 14
  branches; hardware deployment is our own ground to break (consistent with
  the deployment thread's "no prior art for onboard-NN→CTBR on VOXL2").
- **Command-path cross-check ✓**: their AIGP branch's offboard CTBR loop
  uses SET_ATTITUDE_TARGET `type_mask=128`, rates rad/s FRD, thrust 0–1 —
  identical to our ctbr_offboard.py plan (via pymavlink instead of MAVSDK),
  with no failsafe machinery (ours has the ladder).
- **Plant cross-check ✓**: their high-fidelity AIGP model uses motor
  first-order lag τ=0.03 s, DR τ∈[0.01,0.06] s (ours 0.010–0.045), plant DR
  as the explicit sim2real strategy (mass ±40%, k_w ±35%); their main-branch
  model has NO motor lag/delay/drag. Neither branch models transport delay
  or IMU noise (their docs flag exact-IMU "flattery") — our plant is
  strictly richer where sim2real bites.
- **PixelPilot thread** (their pixels→CTBR line): Geles-style 84×84
  analytic gate-mask + asymmetric PPO; solved in-sim, but **0% zero-shot on
  realistic detector masks; IoU-matched DR failed** (their M2/M3 findings) —
  the representation gap between idealized and real perception is
  structural. Validates our train-on-photoreal-renders choice: we carry
  only the splat-vs-reality gap the legacy project already crossed.
- **Adoptable for milestone 2**: their render-free projected gate-mask
  (<100 µs/frame) as a cheap RL pretraining stage before splat fine-tuning;
  their SE3→CTBR reference generator; their gate-count metric-bug warning.

## 5. Synthesis

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
