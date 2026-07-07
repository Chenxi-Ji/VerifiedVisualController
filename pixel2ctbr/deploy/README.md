# pixel2ctbr deployment (Starling 2)

Files here get pushed to the drone / the Starling2 repo, mirroring how
`ctrl_lya_offboard.py` is pushed to VOXL `/tmp` today. **Nothing here has run
on hardware yet** — the bench ladder in `docs/pixel2ctbr/06_verification.md`
gates every step. Sources for all numbers: deployment research in
`docs/pixel2ctbr/01_research_report.md` §4 (pinned PX4 v1.14.3 / MAVSDK
v2.12.2 / modalai voxl-dev sources).

## PX4 parameter recipe (estimator-less rate offboard)

Set (QGC or `param set` on voxl-px4 shell):

| param | value | why |
|---|---|---|
| `EKF2_HGT_REF` | **0** (baro) | ships as 3=vision; without mocap ODOMETRY the EKF has no height source |
| `EKF2_GPS_CTRL` | 0 | no GPS fusion indoors |
| `EKF2_EV_CTRL`  | 0 | external-vision fusion off once mocap bridge stops |
| `EKF2_MAG_TYPE` | 5 | mag stays connected+calibrated (passes arming check) but is never fused; yaw = drifting gyro integral — fine for body-rate policy |
| `COM_OBL_RC_ACT`| 2 | offboard-loss failsafe → Stabilized (Position/Altitude fallbacks need estimates we don't have) |
| `COM_OF_LOSS_T` | 0.3–0.5 | default 1.0 s of HELD LAST BODY RATES is a crash; shorten |

Verify already-shipped values: `COM_ARM_WO_GPS=1`, `MAV_FWDEXTSP=1`,
`IMU_GYRO_RATEMAX=800`, `MC_ROLL/PITCH/YAW_CUTOFF=30/30/10`,
`MUORB_KAF_LAND=1` (note: apps-proc keep-alive starvation → blind descent —
don't run the inference service at RT priority that can starve voxl-px4),
`COM_DISARM_LAND=0.1` + `LNDMC_ROT_MAX=30°/s` (land-detector false-positive
risk during low-thrust hover near floor — tether test).

Mocap stays available as *measurement*: `record_flight.py` + `plot_flight.py`
unchanged; just do NOT start the `mocap_to_px4_bridge` ODOMETRY stream.

## Command path

`pixel_ctbr` model helper (voxl-tflite-server or standalone MPA service) →
`CtrlLyaMsg` (unchanged 40 B wire format, action = `[c m/s², ωx, ωy, ωz]`) →
`mpa_reader` → `ctbr_offboard.py` (this dir) → MAVSDK
`set_attitude_rate(AttitudeRate(roll°/s, pitch°/s, yaw°/s, thrust01))` →
SET_ATTITUDE_TARGET type_mask=128 → PX4 rate loop (800 Hz).

Thrust map v1 (bench-refine): `thrust01 = 0.34 · c/9.81`, clamp ≤ 0.60
(`MPC_THR_MAX`). Curve refinement available: invert `0.9s²+0.1s`.

## Camera / model-helper spec (to implement in Starling2 repo)

- Target camera: `tracking_front` AR0144 RAW8 `preview` (global shutter, gray,
  1280×800, 162° fisheye, ~15–20 ms to client, fps 30→raise toward 50).
  **Blocker: needs fisheye calibration + splat-render retrain with that K;
  until then hires (`hires_small_color`, known K) is the fallback.**
- Preprocess: grayscale + **cv2 INTER_AREA** to 128×96 (matches supersampled
  training AA — 05 log), /255.
- **Frame ring buffer**: keep the last ~6 preprocessed frames (≈72 KB); the
  model input is [current, frame(t−150 ms)] stacked in channels (visual
  velocity baseline — 05 log v9). At stream start, prime the ring with the
  first frame. If the deployed camera rate ≠ 40 Hz, pick the ring depth that
  keeps the pair baseline ≈ 150 ms.
- IMU: `/run/mpa/imu_apps` (1 kHz; raise `imu_apps_fifo_poll_rate_hz` to 500
  or write `"read"` to `/run/mpa/imu_apps/control` per frame — qvio recipe);
  `imu_data_t` = 40 B packed, CLOCK_MONOTONIC (same domain as camera
  timestamps). Average gyro/accel over the frame interval → 12-D vec (order
  and normalization EXACTLY as `policy.py::normalize_vec`; tilt slots = 0
  unless PX4 attitude is wired in later — policy trained with tilt dropout).
- Hidden state: fp32 tensor kept across invokes; zero on stream (re)start and
  on entering OFFBOARD.
- Inference: XNNPACK **1–2 threads** pinned to A77 cores (4–6),
  `voxl-set-cpu-mode perf`; **never the GPU delegate** (documented corruption
  pattern). Expected 0.3–3 ms.

## Bench ladder (must pass in order — see 06_verification.md)

1. SITL or props-off bench: param recipe check (script does it), arm in
   Stabilized, enter OFFBOARD, verify rates/thrust scaling on `px4-listener
   vehicle_rates_setpoint`, exercise all failsafe branches (kill inference →
   hover-hold frames → stale-exit → Stabilized handoff; RC flip mid-stream).
2. Hand-held: policy live, motors off — check action signs against manual
   motion (tilt right ⇒ negative roll-rate command, etc.).
3. Tether, low hover, pilot cover; then start-box flights with mocap
   recording for gate-frame evaluation.
