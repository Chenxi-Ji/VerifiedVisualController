# 00 — Mission & Constraints (pixel2ctbr)

*Written 2026-07-07, before any research or code. This is the problem statement and the
initial state of thinking. Later docs revise it; where they conflict, later docs win.*

## The goal

Evolve the certified visual gate controller into a system that flies the Starling 2
**without OptiTrack and without VIO**. Allowed inputs at flight time:

- the onboard camera,
- the IMU (gyro + accel, and whatever PX4's IMU-only attitude estimator provides),
- the drone's internal low-level control (PX4's body-rate loop and mixer).

The network outputs **CTBR commands** — Collective Thrust + Body Rates
`[thrust, ωx, ωy, ωz]` — instead of body-velocity setpoints.

Milestone 1: **stabilization** — hover at a visual target (in front of the gate) from
pixels + IMU alone.
Milestone 2: extend the same policy class to **trajectories through gates**.

## Why the current system cannot just drop mocap

The current controller is already "vision-only" in the sense that the *action* is
computed from the image alone. But the action is a **velocity command**, and executing
a velocity command requires PX4 to know its own velocity. Today that knowledge comes
from OptiTrack → `mocap_to_px4_bridge` → ODOMETRY → EKF2. Remove mocap and PX4 has no
velocity estimate: `VelocityBodyYawspeed` offboard control simply stops being available
(EKF2 rejects offboard position/velocity modes without aiding). VIO is excluded by the
project goal. So the *interface* between the network and the drone must move down the
control stack, to the point where the autopilot needs only the gyro:

```
today:   image → NN → v_cmd  → PX4 velocity ctrl (needs EKF2 velocity ← mocap)  → rates → motors
goal:    image(+IMU) → NN → [thrust, ω_cmd] → PX4 rate ctrl (needs only gyro) → motors
```

CTBR is the standard interface in the learned-agile-flight literature precisely because
it is the lowest-level command that is still platform-portable and doesn't require any
state estimation beyond the gyro.

## What the network must now implicitly do

Moving the interface down means the network absorbs everything the velocity loop and
position loop used to do:

1. **Attitude stabilization / gravity alignment.** Either the net learns it from IMU
   history, or we feed PX4's IMU-only attitude estimate (roll/pitch usable without any
   external aiding; yaw drifts — but yaw relative to the gate is observable from the
   image). This is "internal drone control" and is allowed.
2. **Velocity damping.** Velocity is not observable from a single image. It *is*
   observable from image motion (optical flow of the gate/arena texture) + IMU
   integration over short windows. ⇒ the policy needs **temporal context**: frame
   stacking or recurrence. This is the single biggest architectural change.
3. **Thrust/weight knowledge.** Hover thrust in normalized units depends on mass,
   battery voltage, prop condition. The sim must randomize it and the policy must
   infer/adapt from accel + observed motion (or we accept a trim knob).

## Hard constraints

- **Runs onboard Starling 2** (VOXL2, QRB5165). Current 192×256 net runs ~7–10 Hz via
  voxl-tflite-server on CPU/XNNPACK (GPU delegate corrupts outputs — measured). A CTBR
  policy realistically needs 30–100 Hz ⇒ smaller input (grayscale, ≤ ~128×96?) and/or
  the tracking camera. **Open question Q1: what policy rate do the reference works use,
  and what can VOXL2 sustain?**
- **No mocap, no VIO in the control loop.** OptiTrack stays as a *measurement
  instrument* for evaluation only (the flight-recording pipeline survives unchanged —
  it never fed the controller anyway).
- **Safety without Position mode.** Without mocap/VIO there is no Position-mode
  babysitting: the pilot holds the drone in Stabilized/Altitude(baro) mode, flips to
  OFFBOARD(rate), and must be able to flip back instantly. The pilot-in-the-loop flow
  from `ctrl_lya_offboard.py` (2026-07-01 rewrite) is the template; failsafe behavior
  under *rate* offboard needs its own review (a stale rate command is far more
  dangerous than a stale velocity command).
- **Trainable with what we have**: the Gaussian-splat twin of the real arena + gsplat
  fisheye rendering with the measured calibration + the battle-tested image-DR
  pipeline. This asset is the project's moat — every strategy should be evaluated
  partly on how well it exploits it.

## Assets carried over (from PROJECT_STATE.md, verified on hardware 2026-07-01)

- Splat digital twin + fisheye render pipeline, byte-matched to onboard preprocessing.
- Image DR + camera-model DR recipes that survived a real sim2real crossing.
- 58k-param verification-friendly backbone (global/lateral/vertical pooled readouts)
  — reusable as the *spatial* encoder, with temporal machinery added.
- TFLite export path (ONNX→onnx2tf→fp16) incl. known pitfalls (dividing pools only,
  no non-dividing AveragePool, fp16 not int8).
- Deployment plumbing: voxl-tflite-server model_helper, MPA pipe protocol, dockerized
  MAVSDK offboard script with watchdogs/logging, flight recorder + gate-frame plotting.
- Measured plant facts: ~100 ms camera→command latency; velocity-loop lag τ≈0.15 s
  (estimate). For CTBR we need *new* plant facts: rate-loop bandwidth, thrust map,
  motor time constant. **Q2: what does PX4 on Starling 2 give us (params, logs) and
  what must be identified/randomized?**

## Initial strategy space (to be settled by research, not vibes)

- **S1 — end-to-end RL from pixels** in the splat twin (PPO + asymmetric actor-critic:
  critic sees privileged pose, actor sees pixels+IMU). Honest question: is gsplat
  rendering throughput enough for RL sample counts? (**Q3** — measure fps at 96×128.)
- **S2 — teacher–student**: state-based CTBR teacher (cheap, no rendering, PPO or
  BPTT/APG like the current trainer) → distill to pixel+IMU student with DAgger on
  rendered rollouts. Orders of magnitude fewer rendered frames.
- **S3 — keep the current velocity-command net, add a learned "velocity executor"**
  (v_cmd + IMU → CTBR). Almost certainly unsound — tracking a velocity command
  requires velocity feedback, which IMU alone can't give (drift). Listed to be
  explicitly killed or rescued by research (**Q4**: has anyone made IMU-only velocity
  tracking work? e.g. via the *image* carrying the velocity information — at which
  point it collapses into S1/S2 anyway).
- **S4 — intermediate visual abstraction**: optical flow / feature tracks as network
  input instead of raw pixels (Deep Drone Acrobatics style). More sim2real-robust,
  cheaper to render (flow from splat renders), but adds an onboard flow module and
  the user asked for *direct pixel* control. Keep as fallback / ablation.

Recurrence vs frame-stacking (**Q5**), which camera — hires fisheye vs tracking cam
(**Q6**, latency/global-shutter/fps vs "the twin was calibrated for hires"), and
whether any certification story survives the move to attitude dynamics (**Q7**,
secondary — keep ops CROWN-friendly where free, don't block on it) are open.

## The user-provided anchor paper

https://arxiv.org/abs/2406.12505 — (to be confirmed by research; believed to be UZH-RPG
"flying from pixels without state estimation" line of work). Research must extract:
exact observation/action spaces, policy rate, architecture, training recipe, sim2real
measures, hardware, compute location (onboard?), and code availability.

## Success criteria for milestone 1 (stabilization)

1. In sim (splat twin, full dynamics + DR): from the current start box, reach and hold
   the hover target in front of the gate — position error comparable to the velocity
   controller (final err ≈0.27 u real / ~0.1 u sim today), no crashes, for ≥95% of
   start poses.
2. Policy + preprocessing runs onboard at the design rate with measured latency inside
   the trained latency budget.
3. One real flight, pilot-in-the-loop, that converges to hover in front of the gate
   with mocap recording *as measurement only* — evaluated with the existing
   plot_flight.py pipeline (which needs no mocap-in-the-loop changes).

## Doc map (will grow)

- `00_mission_and_constraints.md` — this file.
- `01_research_report.md` — papers + code survey (the deliverable research report).
- `02_repo_audit.md` — what exists in the repos, what's reusable.
- `03_strategy.md` — options weighed, decision + rationale.
- `04_design.md` — the chosen system, end to end.
- `05_implementation_log.md` — running log while building.
- `06_verification.md` — tests, sim evals, onboard benches, flight protocol.
- `07_multigate_envs.md` — milestone-2 multi-gate tracks (splat editing, envs).
