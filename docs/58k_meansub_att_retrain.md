# 58k controller, meansub-only / attitude-head retrain (sim-only)

*2026-07-11. Side deliverable: the repo's original ~58k vision controller,
minimally modified (input 6ch -> 3ch meansub-RGB, output velocity -> attitude
+thrust) and retrained in sim on its original one-gate viewpoint-hold task, so
a verification pipeline can be practiced on it. It does NOT need to fly.*

## Artifacts

| file | what |
|---|---|
| `weights/ctrl_lya_meansub_att.pt` | **the deliverable** — banked best (epoch 60 of the 120-epoch run; selector = quick-eval `hold_median + 10*crash` on 64 DR episodes). Checkpoint keys: `controller`, `lyapunov` (co-trained V net), `epoch`, `meta` (input spec, action space, param count, plant, eval results). |
| `weights/ctrl_lya_meansub_att_last.pt` | rolling endpoint of the full 120-epoch run (later phases scored worse — see "what was tried"). |
| `logs/eval_ctrl_meansub_att*.json` | the four definitive 240-episode evals below. |
| `figures/rollouts_meansub_att.png` | 16 DR episodes, 6 s closed loop. |
| `figures/renderer_parity_meansub_att.png` | render-path parity check. |

The original flight artifact `weights/ctrl_lya.pt` is untouched.

## Exact architecture delta

Original: `Controller` in `scripts_control/utils_ctrl_lya_pt.py` (the ~58k
net that flew hardware with OptiTrack closing the velocity loop; docs in
`SYSTEM_OVERVIEW.md` section 4). Modified copy: `ControllerMeansubAtt` in
`scripts_control/utils_ctrl_meansub_att.py`. Three changes, nothing else:

1. **Input 6ch -> 3ch.** Original forward concatenated
   `[x, x - x.mean(dim=(2,3), keepdim=True)]` (raw RGB + per-image
   per-channel mean-subtracted RGB). Now the input is the **mean-subtracted
   RGB only** (same meansub expression, raw branch dropped);
   `conv1 = Conv2d(6,16,5,2,2)` -> `Conv2d(3,16,5,2,2)`. Still consumes RGB
   192x256 in [0,1]; meansub happens inside `forward`. (Exact global
   brightness/color-cast invariance now holds by construction — asserted in
   the module self-test.)
2. **Output head: velocity -> attitude+thrust** with the FF-campaign
   conventions (copied from PixelCTBR `pixel2ctbr_ff/policy_ff.py`
   `action_center_span` / attitude branch — copied, not imported):
   - `c = G + clamp_relu(raw0, 1) * 0.9G` — collective thrust, [0.1G, 1.9G] m/s^2
   - `roll_sp, pitch_sp = clamp_relu(raw * 0.35, 0.35)` — tilt setpoints, +-0.35 rad
   - `yaw_sp = -pi/2 + clamp_relu(raw * pi, pi)` — **absolute** yaw setpoint
     (-pi/2 = facing the gate = this repo's target yaw)
   - last linear layer zero-initialized -> exact hover `[G, 0, 0, -pi/2]` at
     init (inherited FF contract).
   The head layers themselves (Linear 120->64, ReLU, Dropout 0.1, Linear
   64->4) are unchanged — only the output squashing differs.
3. Trunk / readouts / everything else: untouched (AvgPool front, 16/32/48/64
   backbone, global + lateral(1x4) + vertical(3x1) readouts, 120-d head
   input, all clamp_relu/CROWN-friendly ops).

**Param count: 56,836 trainable (original: 58,036; delta = conv1 shrink
16x3x5x5 vs 16x6x5x5 = -1,200).** 57,188 stored floats incl. BN running stats.

## Plant, task, trainer

- Plant: `QuadAttitudeDynamics` — PX4 quaternion-P attitude cascade on top of
  a rate/thrust-lag rigid-body model, FIFO transport delay, per-episode DR
  (twr 2.0-3.2, tau_w 15-60 ms, tau_c 10-45 ms, kd_lin 0.03-0.30,
  thrust_gain 0.85-1.15, K_att +-20%). Copied verbatim from PixelCTBR
  `pixel2ctbr/dynamics.py` into `utils_ctrl_meansub_att.py`. Control period
  0.1 s (the original trainer's dt, ~ the deployed 7-10 Hz), physics substep
  5 ms, transport delay fixed 1 control step (100 ms) = the original
  `latency_steps=1`. Plant works in meters; renderer/losses in scene units
  (1 u = 0.85 m).
- Task (unchanged): servo to `[0, +1.5, 0, -pi/2]` — 1.5 u in front of the
  +y gate face, facing it — from random offsets (x +-1.5, y -1.0..+1.5,
  z -0.5..+0.4, yaw +-0.6, pitch/roll +-0.20), in the gsplat twin
  (`nerfstudio/outputs/Gate_Long_hloc_seq/.../2026-06-11_015308_cleaned`).
- Trainer: `scripts_control/train_ctrl_meansub_att.py` =
  `train_ctrl_lya_pt.py` with the plant swapped in. Losses (trajectory
  progress, co-trained Lyapunov decrease, final-state precision), 3-phase
  curriculum, horizon schedule 7->25 steps, image-space `DomainRandomizer`,
  per-epoch camera-intrinsics jitter and 2%-of-span actuation noise are the
  original's (losses/dataset literally imported from `train_ctrl_lya_pt.py`).
  Rendering via a batched GPU path (`render_batch_gpu`): same view math,
  **bit-exact** vs the original `render()` at `raster_scale=1.0` (verified;
  `figures/renderer_parity_meansub_att.png`); training/eval ran at
  `raster_scale=0.5` (fisheye rasterized at 512x384 with K/2, ~4x faster,
  mean image delta 1.5%) — the 240-episode eval repeated at scale 1.0 agrees
  within noise (below), so nothing is tied to the fast path.

### Training command

```bash
cd ~/certified_visual_controller/VerifiedVisualController_small_clone
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts_control/train_ctrl_meansub_att.py --max-minutes 8
# repeat until "TRAINING COMPLETE" (resumes from _last; ~3.5 h on the RTX 5080
# laptop at SIDE_MEM_FRAC=0.25). Defaults: 120 epochs, 1024 poses/epoch,
# batch 32, raster_scale 0.5.
```

The delivered artifact banked at **epoch 60** (mid-P2) of this run. Two loss
terms were added to the trainer *after* that point while chasing the
acceptance bar (`--w-floor`, soft mat-plane floor; `--w-vel-scale`,
terminal-velocity penalty — both forced by the plant change and documented in
the code); every post-ep-60 configuration (P3 precision phase, floor/vel
terms, H=30 polish, H=55 long-horizon fine-tune) scored WORSE on the
quick-eval selector, so the bank was never overwritten. To reproduce the
delivered recipe exactly: `--w-floor 0 --w-vel-scale 0` and take the ep-60
bank.

## Eval — definitions and results

`scripts_control/eval_ctrl_meansub_att.py` (the repo's native test,
`test_ctrl_lya_pt.py`, is qualitative videos; this is its quantitative
counterpart on the same task):

- **episode**: spawn at rest at the native test offsets (x +-1.2, y -0.8..+1.2,
  z -0.5..+0.4 u, yaw +-0.5, level), roll out **6.0 s** (60 steps @ 0.1 s)
  closed loop against the attitude plant.
- **plant DR** (standard): `DynParams.randomized` + K_att +-20% per episode,
  delay 1 step; images clean (as the native test).
- **hold_err**: mean position error over the **last 1.5 s**, meters. Headline
  = median over episodes (crashed episodes excluded from hold stats).
- **crash**: at any step, below the mat plane (z_u > 0.65), >3 u from target
  (diverged), or NaN.

240 episodes each, seed 0, artifact `ctrl_lya_meansub_att.pt`:

| eval | hold median | hold xy / z | crash | succ@20cm | yaw err |
|---|---|---|---|---|---|
| **standard DR** (acceptance eval) | **62.6 cm** | 36 / 46 cm | **29.2%** | 0.6% | **0.71 deg** |
| standard DR, raster_scale 1.0 | 61.5 cm | — | 30.4% | — | — |
| nominal plant (no DR) | 59.2 cm | 40 / 39 cm | 26.3% | 0.6% | — |
| DR but thrust_gain=1.0 | 53.1 cm | 33 / 37 cm | 14.2% | 2.4% | — |

**Acceptance bar (median hold <= 10 cm, crash < 2%): NOT MET.** Yaw is
solved (<1 deg); position hold plateaus at ~0.5-0.6 m with ~15-30% mat
crashes.

### Why (the honest blocker)

The bar is structurally out of reach for THIS architecture on THIS plant, not
a matter of more epochs/lr (10+ recipe iterations all slid along the same
Pareto wall):

1. **No velocity feedback.** The net is memoryless (one frame). In velocity
   mode the OptiTrack/PX4 velocity loop below supplied the damping; in
   attitude mode the position loop closes through a double integrator whose
   only damping is rotor drag (kd_lin 0.03-0.30 /s). Any static image->thrust
   /tilt map is a conservative spring: rollouts show slow (~5-8 s period)
   weakly-damped xy/z oscillations of ~0.5 m amplitude — consistent with
   PixelCTBR's single-image hover line plateauing at 45.4 cm (gray 96x128,
   40 Hz) after multi-day tuning (its two-frame/GRU lines, which DO see
   velocity, reach 3-11 cm).
2. **Delay-capped stiffness vs unidentifiable thrust gain.** A memoryless net
   cannot identify the episode's thrust_gain (+-15% = +-1.5 m/s^2 bias), so
   its altitude error scales as gain-error/stiffness, and stiffness is capped
   by the 100 ms transport delay + lags at 10 Hz control (loop goes unstable
   near ~4-5 rad/s — measured: every push toward precision traded directly
   into crash rate). At the cap, the residual is a few tens of cm — matching
   the measured 53->62 cm and the halving of crashes when thrust_gain is
   pinned to 1.
3. **BPTT horizon**: the oscillation period exceeds any horizon this recipe
   can train (H=55 attempts diverged — BPTT through 1100 chained dynamics
   steps; the PixelCTBR campaign caps at ~32 with checkpointing for the same
   reason), so the optimizer cannot even see the mode it would need to damp.

What WOULD move it (out of scope here): two-frame input or any recurrence
(velocity observability — the PixelCTBR img2/GRU hover lines hit 3-11 cm on
this scene), 40 Hz control, or an integral/adaptive element for the thrust
map. All are architecture changes; the brief was to keep the 58k net as-is.

For practicing a verification pipeline the artifact is fully serviceable: the
net is CROWN/TFLite-friendly (same op set as the original), memoryless, with
bounded outputs, a co-trained Lyapunov function in the same checkpoint, and a
reproducible sim eval; it reaches and holds the gate viewpoint (median 6-s
endpoint within ~0.6 m, yaw locked) rather than flying precision hover.

## Load / run

```python
import sys, torch
sys.path.insert(0, "scripts_control")
from utils_ctrl_meansub_att import ControllerMeansubAtt
ctrl = ControllerMeansubAtt()
ck = torch.load("weights/ctrl_lya_meansub_att.pt", map_location="cpu", weights_only=False)
ctrl.load_state_dict(ck["controller"]); ctrl.eval()
print(ck["meta"]["input_spec"], ck["meta"]["action_space"], ck["meta"]["definitive_eval"]["standard_dr"])
# img: (B,3,192,256) RGB in [0,1]  ->  action (B,4) [c, roll_sp, pitch_sp, yaw_sp]
act = ctrl(torch.rand(1, 3, 192, 256))
```

Closed-loop / eval:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts_control/eval_ctrl_meansub_att.py \
    --weights weights/ctrl_lya_meansub_att.pt --episodes 240
# variants: --no-plant-dr | --unit-thrust-gain | --image-dr | --raster-scale 1.0
```

Module self-tests: `python scripts_control/utils_ctrl_meansub_att.py`
(param count, hover-at-init, bounds, cast invariance, plant hover).

GPU etiquette (a campaign training shares this GPU): every entry point sets
`torch.cuda.set_per_process_memory_fraction(SIDE_MEM_FRAC (default 0.25))`
and defaults `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

## TFLite export (2026-07-11)

Same route as the original `ctrl_lya.tflite` export, run in the
`certified_visual_controller` conda env (the "pftolite" name in the old
docstrings is stale), all on CPU (`CUDA_VISIBLE_DEVICES=""`):

**PyTorch -> ONNX (torch.onnx.export, opset 18, dynamo/torch.export path of
torch 2.11) -> onnx2tf 1.29.24 SavedModel (`disable_group_convolution=True`)
-> TF 2.19 TFLiteConverter.** The exported graph is the same FUSED model as
the original: `FusedModel(ControllerMeansubAtt, LyapunovV(ckpt["lyapunov"]))`
with inputs `image (1,192,256,3 NHWC)`, `pose (1,6)`, `target (1,6)` and
outputs `action (1,4)`, `V (1,)` — byte-identical IO signature to
`ctrl_lya.tflite`, so all existing tflite tooling works unchanged. Only the
ACTION SEMANTICS differ: `[c m/s^2, roll_sp, pitch_sp, yaw_sp(abs)]` instead
of `[vx, vy, vz, yaw_rate]`. No unsupported ops; the in-graph mean-subtract
exports as a plain ReduceMean.

| artifact | precision | size |
|---|---|---|
| `weights/ctrl_lya_meansub_att.tflite` | float16 weights (`Optimize.DEFAULT` + `supported_types=[tf.float16]`, same choices as `ctrl_lya.tflite`) | 131.4 KB |
| `weights/ctrl_lya_meansub_att_f32.tflite` | float32 reference (no quantization) | 240.9 KB |

Export script: `scripts_tflite/export_to_tflite_meansub_att.py` (thin variant
of `export_to_tflite.py`, reuses its Step-2/3 helpers; original untouched).

### Parity (pt vs tflite), `scripts_tflite/debug_pt_vs_tflite_meansub_att.py`

Per-channel MAX abs delta; "norm" = delta / channel span
(c: 0.9G = 8.83 m/s^2, tilt: 0.35 rad, yaw: pi rad).

| input set | file | c | roll_sp | pitch_sp | yaw_sp | max norm |
|---|---|---|---|---|---|---|
| 256 random images | f16 | 2.4e-3 | 2.6e-5 | 3.0e-5 | 4.9e-4 | **2.8e-4** |
| 256 random images | f32 | 1.0e-4 | 3.1e-6 | 6.9e-6 | 3.0e-6 | **2.0e-5** |
| 21 real frames (real_t*.png + video samples) | f16 | 3.4e-3 | 2.0e-4 | 6.8e-5 | 5.3e-4 | **5.6e-4** |
| 21 real frames | f32 | 6.0e-3 | 1.7e-4 | 7.9e-5 | 9.3e-5 | **6.8e-4** |
| all 530 starling_video frames | f16 | 1.2e-2 | 3.0e-4 | 1.8e-4 | 5.6e-4 | **1.4e-3** |

Lyapunov V (100 random pose/target pairs): max 2.5e-4 (f16), 9.5e-7 (f32).

Acceptance (<1e-3 normalized, f32): PASS (2.0e-5 random, 6.8e-4 real). The
f16 file matches on the PNG set (5.6e-4) and only exceeds 1e-3 on the
worst thrust sample over the full 530-frame video sweep (1.4e-3 normalized =
0.012 m/s^2 on a 8.8 m/s^2 span) — the expected f16 weight-quantization
cost; use the f32 reference where <1e-3 is required. On real frames the f32
delta is op-reassociation noise (ReduceMean/conv accumulation order), not
quantization — it is the same order as f16 there.

### Real-frame replay

`replay_real_frame.py` runs UNCHANGED (pass the tflite path as argv); note
its printed labels are the old velocity-head text — for this model read
`[vx, vy, vz, yaw_rate]` as `[c m/s^2, roll_sp, pitch_sp, yaw_sp]`:

| frame | c (m/s^2) | roll_sp | pitch_sp | yaw_sp |
|---|---|---|---|---|
| real_t0.0 (full + pre-resized identical) | 7.948 (0.81 g) | +0.175 | -0.005 | -1.594 |
| real_t1.0 | 7.675 (0.78 g) | +0.163 | -0.017 | -1.591 |
| real_t0.0_colorshift | 8.779 | +0.135 | -0.010 | -1.595 |

Bounded, non-NaN, tilts well inside +-0.35, yaw pinned at -pi/2 (target
heading), thrust slightly below hover — plausible for these frames.

Video replay: `scripts_tflite/replay_real_video_meansub_att.py` (thin variant
of `replay_real_video.py`: ControllerMeansubAtt on the pt side, attitude
labels, thrust plotted as c/G-1, outputs suffixed `_meansub_att` so the
original model's artifacts stay put). Over all 530 frames of
`starling_video.mp4`: no NaN; c in [4.9, 11.6] m/s^2 (mean 9.06 = 0.92 g),
roll_sp in [-0.14, +0.29], pitch_sp in [-0.06, +0.09], yaw_sp in
[-1.60, -1.41] (mean -1.53). Artifacts:
`starling_video_meansub_att_actions.{csv,png}`. (The clip's mp4 header
claims 320 fps, so the t axis spans 1.66 s — same as the original model's
`starling_video_actions.csv`.)

The closed-loop reference numbers for this model remain the PyTorch eval in
the table above / `logs/eval_ctrl_meansub_att*.json` (a tflite-in-the-loop
sim eval was descoped).

### Commands

```bash
cd ~/certified_visual_controller/VerifiedVisualController_small_clone
PY=~/miniconda3/envs/certified_visual_controller/bin/python

# 1. export (CPU): writes weights/ctrl_lya_meansub_att{,_f32}.tflite
CUDA_VISIBLE_DEVICES="" $PY scripts_tflite/export_to_tflite_meansub_att.py

# 2. parity: 256 random images + real frames + Lyapunov, per-channel deltas
CUDA_VISIBLE_DEVICES="" $PY scripts_tflite/debug_pt_vs_tflite_meansub_att.py \
    --tflite weights/ctrl_lya_meansub_att.tflite      # and _f32.tflite

# 3. single-frame replay (existing script, unchanged)
CUDA_VISIBLE_DEVICES="" $PY scripts_tflite/replay_real_frame.py \
    real_t0.0_full.png weights/ctrl_lya_meansub_att.tflite

# 4. video replay (tflite + pt side by side, CSV + plot)
CUDA_VISIBLE_DEVICES="" $PY scripts_tflite/replay_real_video_meansub_att.py \
    starling_video.mp4
```
