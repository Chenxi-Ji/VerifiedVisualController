# 09 — Network I/O: the exact contract (and the no-cheating audit)

*2026-07-07. The definitive record of what the policy consumes and emits,
where each bit comes from in sim vs on the drone, and the explicit audit
that flight-time inputs are camera + IMU + the policy's own memory — nothing
else. Code ground truth: `pixel2ctbr/policy.py` (esp. `normalize_vec`,
`PixelCTBRPolicy.forward`), `pixel2ctbr/env.py::observe`, and the onboard
`Starling2/voxl-tflite-server/src/model_helper/pixel_ctbr_model_helper.cpp`
(`build_vec`, `run_inference`). If this doc and code disagree, the code wins
— then fix this doc.*

## 1. Inputs (three tensors per control step)

### 1.1 Image — `(1, 2, 96, 128)` float32 in [0,1]

Two GRAYSCALE fisheye frames stacked in the channel dimension:

| channel | content |
|---|---|
| 0 | current frame |
| 1 | the frame from ≈150 ms ago |

- Resolution 128×96, from the measured hires calibration (fx 504.341 …
  scaled), preprocessing = grayscale + **area-average** downscale (sim:
  2× supersampled render + avg-pool; drone: `cv2.INTER_AREA` — spec'd
  together so the anti-aliasing matches; 05 log).
- The 150 ms pairing is the **visual velocity** source: at 40 Hz and hover
  speeds (0.2–0.5 m/s), consecutive frames move 0.3–0.8 px (sub-pixel,
  unlearnable — run-8 finding); a 6-frame gap makes motion 1.5–4 px.
  Onboard this is a ring buffer of the helper's own past frames
  (`PIXEL_CTBR_RING`, default 5 at 30 fps ≈ 167 ms; set `round(0.15·fps)`).
  At stream start the ring is primed with the first frame (pair = static),
  identical to the sim's episode start.
- Inside the network, `features()` expands each frame to
  `[raw, raw − its own spatial mean]` → the conv trunk sees 4 channels.
  The mean-sub pair is the exact global-brightness-invariance trick carried
  from the previous phase (and it is implemented as AvgPool, not `.mean()`,
  for TFLite reasons — 05 log).

### 1.2 Proprio vector — `(1, 12)` float32

Order and scaling MUST match `policy.py::normalize_vec` (the onboard
`build_vec()` mirrors it constant-for-constant):

| idx | quantity | normalization | source (deploy) |
|---|---|---|---|
| 0–2 | gyro ωx,ωy,ωz [rad/s] | ÷ 4.0 | `/run/mpa/imu_apps`, averaged over the frame interval |
| 3–5 | accel (specific force) [m/s²] | ÷ 20.0 | same IMU (hover reads ≈[0,0,−9.81], FRD) |
| 6–7 | tilt roll, pitch [rad] | ÷ 0.5 | complementary filter **inside the helper**, computed from that same gyro+accel (gravity direction + gyro integration; 1 g-gated). Trained with 20% per-episode dropout ⇒ zeroing it (`PIXEL_CTBR_TILT=0`) degrades gracefully (84.2%→77.7% in the definitive ablation) |
| 8 | last thrust cmd | (c − 9.81)/8.829 | policy's own previous output |
| 9–11 | last rate cmds | ÷ [4, 4, 2] | policy's own previous output |

### 1.3 Recurrent state — `(1, 96)` float32

The GRU hidden state, fed back from the previous step's `h_out`. Onboard it
is an explicit input/output tensor pair; the helper carries it across
invokes and **zeroes it on stream start or a >0.5 s frame gap** (matching
the sim's episode-start semantics). This is where velocity estimation,
thrust-residual integration (the "learned integral action"), and short
memory live.

## 2. Outputs

- `action (1, 4)` = `[c, ωx, ωy, ωz]`: collective thrust in **m/s²**
  (mass-normalized; range 0.1 g–1.9 g via `clamp_relu`, centered so a
  zero-weight head outputs exact hover) and body rates in **rad/s**
  (clamped ±[4,4,2]). Deployment maps thrust to PX4's normalized command:
  `thrust01 = 0.34·c/9.81` (bench-trimmed at gate C5).
- `h_out (1, 96)`: next hidden state (fed back).
- Wire message (`CLYA`, 40 B, unchanged from the legacy phase): action in
  slots 0–3, slots 4–5 zero, `V` slot reserved (0 for now; earmarked for
  the OOD/health monitor — 08_lyapunov.md L3b).

### Exported-graph note (the deployable contract)

After ONNX→onnx2tf conversion the image tensor is **NHWC** `(1, 96, 128, 2)`
with channels `[current, previous]` interleaved per pixel, and tensor names
are mangled — the helper binds **by rank/shape** (4-D→image, last-dim 12→vec,
96→h). The export wrapper also reorders conv1's input channels to build the
mean-sub pairs slice-free (converter limitation); parity is proven by a
1000-step closed-loop check on every export.

## 3. The no-cheating audit (flight-time information sources)

| input | source | external state estimate? |
|---|---|---|
| image pair | camera + the helper's own past frames | **no** |
| gyro, accel | IMU | **no** |
| tilt | derived from the same IMU inside our helper (not PX4 EKF, no mag, no vision) | **no** |
| last action | the policy itself | **no** |
| hidden state | the policy itself | **no** |

Surrounding system: PX4 executes the rate commands using **only its gyro**
(the reason CTBR was chosen — 00_mission); mag connected but unfused
(`EKF2_MAG_TYPE=5`); no GPS/VIO; the mocap→PX4 bridge is **never started** —
OptiTrack is a post-flight measurement instrument only. The barometer serves
only the pilot's Altitude fallback mode, never the policy.

**Deliberately absent from the inputs**: position, velocity, yaw, target
pose, gate coordinates, task ID. The task (hover / two-gate / three-gate) is
baked into the weights by training — which is exactly why the tasks ship as
separate weight files (`pixel_ctbr_{one,two,three}_gate.tflite`) rather
than one conditioned model.

**Where privileged information DOES exist — training time only** (the
standard asymmetric setup): the geometric expert teacher, all window losses,
and the auxiliary velocity head read true simulator state. Every such
pathway is severed at export (the aux head is deleted; the shipped graph is
parity-checked to compute from exactly the three tensors above).
Behavioral corroboration that no hidden crutch remains: the definitive
ablations degrade gracefully (no-tilt −6.5 pts, +25 ms latency −10 pts,
DR-off ≈ base) instead of collapsing.

## 4. One-line history of why each input exists

- **Two frames, 150 ms apart** — velocity is unobservable from one frame;
  proven the bottleneck by a privileged probe (0.55 m plateau → 0.113 m
  with true v), consecutive frames proven useless by pixel arithmetic
  (05 log, runs 5–9).
- **Gyro/accel** — damping and thrust-residual cues; accel bias DR forces
  the vision to stay authoritative.
- **Tilt with dropout** — cheap attitude prior worth ~6 pts, optional by
  training so the complementary filter is a bonus, not a dependency.
- **Last action** — standard delay compensation (Geles, Swift both feed it).
- **GRU(96)** — the integral action the expert experiment proved necessary
  (PD 27% → PID 99%), plus state estimation across the 150 ms baseline.
