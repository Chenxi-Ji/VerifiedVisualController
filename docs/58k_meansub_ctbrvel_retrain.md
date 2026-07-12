# 58k controller, meansub-input / CTBR-as-velocity retrain (sim-only)

*2026-07-12. Second verification-practice variant of the repo's original ~58k
vision controller. Deltas vs the original are deliberately just two: (1) input
6ch -> 3ch mean-subtracted RGB, (2) output squashing -> CTBR-format head. The
4 CTBR numbers are then CONSUMED AS THE VELOCITY COMMAND by the ORIGINAL
kinematic plant, and images are rendered with the OLD PINHOLE camera. Because
the velocity head + kinematic integrator provide implicit damping, this variant
reaches ORIGINAL-CLASS sim quality. It does NOT need to fly.*

## Headline result

| variant | @10cm hold | median hold | crash | plant / notes |
|---|---|---|---|---|
| **meansub_ctbrvel (this)** | **100.0%** | **1.4 cm** | **0.0%** | velocity plant (tau=0.15, delay=1, noise=0.02), 240 eps |
| original banner | 84.2% strict | 3.1 cm | 0 | the flight artifact `weights/ctrl_lya.pt` |
| meansub_att (prior side variant) | 0.0% | 62.6 cm | 29.2% | attitude plant, no velocity damping -> orbits |

**Acceptance bar (median hold <= 10 cm AND crash < 2% over >= 200 eps): PASS**,
with margin — this variant beats the original banner's 3.1 cm median hold.

## Artifacts

| file | what |
|---|---|
| `weights/ctrl_lya_meansub_ctbrvel.pt` | **the deliverable** — banked best (epoch 105 of the 120-epoch run; selector = quick-eval `hold_median + 10*crash` on 64 velocity-plant episodes). Keys: `controller`, `lyapunov` (co-trained V net), `optimizer`, `epoch`, `best_score`, `loss_hist`, `meta` (input spec, camera model, head, the CTBR->velocity map, param count, plant). |
| `weights/ctrl_lya_meansub_ctbrvel_last.pt` | rolling endpoint of the full 120-epoch run (resume source). |
| `logs/eval_ctrl_meansub_ctbrvel.json` | definitive 240-episode eval (original-recipe randomization). |
| `logs/eval_ctrl_meansub_ctbrvel_imagedr.json` | 240 eps + training image DR (robustness reference). |
| `logs/eval_ctrl_meansub_ctbrvel_pure.json` | 240 eps, pure integrator (no lag/delay/noise) — the literal `test_ctrl_lya_pt.run_test` plant. |
| `videos/rollout_meansub_ctbrvel_{00..04}.mp4` | 5 rollouts, H=30, original 3-panel figure (3D traj + Lyapunov + rendered view), fps 3, OLD-pinhole renders in the view panel. |
| `figures/pinhole_view_meansub_ctbrvel.png` | old-pinhole vs current-fisheye render comparison at two poses. |
| `figures/training_curves_meansub_ctbrvel.png` | training loss curves. |

The original flight artifact `weights/ctrl_lya.pt`, and the prior side variant's
`weights/ctrl_lya_meansub_att*`, are untouched. No originals were modified — all
new code lives in `*_meansub_ctbrvel.py` files.

## Exact architecture delta vs the original

Original: `Controller` in `scripts_control/utils_ctrl_lya_pt.py` (~58k params,
the net that flew hardware). Modified copy: `ControllerMeansubCtbrVel` in
`scripts_control/utils_ctrl_meansub_ctbrvel.py`. **Two** changes, nothing else:

1. **Input 6ch -> 3ch.** The original `forward` concatenated
   `[x, x - x.mean(dim=(2,3), keepdim=True)]` (raw RGB + per-image per-channel
   mean-subtracted RGB, 6 ch). Now the input is the **mean-subtracted RGB only**
   (identical meansub expression, raw branch dropped);
   `conv1 = Conv2d(6,16,5,2,2)` -> `Conv2d(3,16,5,2,2)`. Exact global
   brightness/color-cast invariance now holds by construction (asserted in the
   module self-test). Same 3ch-meansub input as the delivered `meansub_att`.
2. **Output squashing: original velocity -> CTBR head.** The head *layers*
   (Linear 120->64, ReLU, Dropout 0.1, Linear 64->4) are the original's; only
   the output squashing is replaced by the PixelCTBR **CTBR** convention
   (`pixel2ctbr/policy.py` constants + `pixel2ctbr_ff/policy_ff.py` `ctbr`
   branch — **copied, not imported**):
   - `c  = G + clamp_relu(raw0, 1) * 0.9G` — collective thrust, in [0.1G, 1.9G] m/s^2
   - `w  = clamp_relu(raw_{1:} * RATE_LIM, RATE_LIM)`, `RATE_LIM = (4, 4, 2)` rad/s
   - last linear layer **zero-initialized** -> exact hover CTBR `[G, 0, 0, 0]`
     at init (campaign contract; maps to the zero velocity command below).

   The net genuinely outputs **CTBR-format numbers** — a verifier can state
   "this network emits `[c, wx, wy, wz]` in the deployment CTBR interface."

Trunk / readouts / everything else: untouched (AvgPool front, 16/32/48/64
backbone, global + lateral(1x4) + vertical(3x1) readouts, 120-d head input,
all `clamp_relu`/CROWN-friendly ops). Adaptive pools make the trunk
resolution-agnostic, so the same weights consume the old pinhole 200x300 frames.

**Param count: 56,836 trainable** (original 58,036; delta = conv1 shrink
16x3x5x5 vs 16x6x5x5 = -1,200). 57,194 stored floats incl. BN running stats.
Identical count to `meansub_att` (same conv1 change; the head squashing is
parameter-free).

## The FIXED CTBR -> velocity interface (the load-bearing convention)

The 4 CTBR-slot numbers are **not** flown through a rate loop here. The sim
**consumes them as the body-frame velocity command** of the ORIGINAL plant.
`utils_ctrl_meansub_ctbrvel.ctbr_to_velocity` is the fixed, affine,
verification-friendly map; each slot's full CTBR range maps **exactly** onto
the original velocity head's clamp range (FRD body frame, z DOWN):

| CTBR slot (net output) | range | -> original velocity slot | formula | range |
|---|---|---|---|---|
| `c`  collective thrust | [0.1G, 1.9G] m/s^2 | `vz` (climb, z-down) | `-(c - G)/(0.9G)` | [-1, 1] u/s |
| `wx` roll rate | +-4 rad/s | `vy` (right) | `+wx/4` | [-1, 1] u/s |
| `wy` pitch rate | +-4 rad/s | `vx` (forward) | `-wy/4` | [-1, 1] u/s |
| `wz` yaw rate | +-2 rad/s | `yaw_rate` | `0.3*wz/2` | [-0.3, 0.3] rad/s |

Sign rationale: above-hover thrust = climb = `-z`; roll-right = translate
right; nose-down pitch = translate forward; yaw shares the axis, rescaled onto
the original +-0.3 rad/s authority. The reachable command set is therefore
**identical** to the original `Controller`'s ([-1,1] u/s per translation axis,
+-0.3 rad/s yaw), and hover CTBR `[G,0,0,0]` maps to the zero command ("stay").
The map is differentiable, so BPTT flows through it during training. It is
stamped in the checkpoint `meta.ctbr_to_velocity_map` and is the single fact
the verification person needs: *the net emits CTBR-format numbers; the sim
treats them as velocities via this fixed 1:1 mapping.*

## Camera: the OLD PINHOLE model (the switch mechanics, precisely)

The repo migrated camera models at **commit `27cd577`** ("updates to camera and
domain randomisation", 2026-06-21). That commit **added
`camera_model="fisheye"`** to both `gsplat.rasterization(...)` calls in
`scripts_control/render_image.py` and moved to a 1024x768 raster
(fx=504.34...) downscaled to 256x192. **Before** that commit (the `70c113e`
era, `weights/old/ctrl_lya_20260611_035813.pt`), `render()`:
- passed **no** `camera_model` kwarg — and gsplat's default is
  `camera_model='pinhole'` (verified against the installed gsplat signature:
  `Literal['pinhole','ortho','fisheye'] = 'pinhole'`), i.e. rectilinear
  projection, ~91 deg FOV;
- rasterized directly at `width=300, height=200` with
  `fx=113.258171, fy=113.347599, cx=158.868074, cy=98.837772`;
- returned the frame **unresized**.

`render_pinhole` / `render_batch_pinhole_gpu` in
`scripts_control/utils_ctrl_meansub_ctbrvel.py` replicate that old code path
exactly — same view math (`get_viewmat`, `CAM_AXES`, world-frame branch), same
`rasterization` kwargs — and pass **`camera_model="pinhole"` explicitly** where
the old code relied on the default. The net therefore consumes `(3, 200, 300)`
pinhole frames, exactly as the original pinhole-era training did.
`figures/pinhole_view_meansub_ctbrvel.png` shows the projection difference
(pinhole/rectilinear ~91 deg vs the current fisheye/equidistant ~116 deg), and
`figures/camera_model_old_vs_new.png` (pre-existing) documents the migration.
Batched-vs-single-image render parity: **1.5e-4** max abs (`check_pinhole_parity`).

## Plant, task, trainer — original recipe, verbatim

- **Plant (unchanged from the original).** `body_to_world_velocity` (yaw-aware
  rotation) -> Gaussian actuation noise (std 0.02 u/s) -> FIFO transport delay
  (`latency_steps=1`, 100 ms) -> first-order actuator lag (`tau=0.15` s,
  zero-order-hold discretized) -> `pose += world_velocity * dt`, `dt=0.1`.
  This is `train_ctrl_lya_pt.py`'s rollout integration, copied line-for-line
  into the rollout loop (only the head->command step gains the fixed
  `ctbr_to_velocity` shim).
- **Task (unchanged).** Servo to `[0, +1.5, 0, -pi/2]` — 1.5 u in front of the
  +y gate face, facing it — from the original `PoseDataset` offsets
  (x +-1.5, y -1.0..+1.5, z -0.5..+0.4, yaw +-0.6, pitch/roll +-0.20), in the
  gsplat twin `nerfstudio/outputs/Gate_Long_hloc_seq/.../2026-06-11_015308_cleaned`.
- **Losses / curriculum / DR (unchanged, imported).** `compute_traj_loss`,
  `compute_lyapunov_decrease_loss`, `compute_final_state_loss`,
  `PoseDataset`, `plot_training_curves`, the `DomainRandomizer`, and the
  3-phase weight schedule + horizon schedule (7/10/14/18/22/25) + LR schedule
  (1e-3, x0.95 / 10 ep) are `train_ctrl_lya_pt.py`'s, **imported not copied**.
  The co-trained `Lyapunov` net is optimized jointly, as the original.
  `meta.loss_deviation = "NONE"` — unlike `meansub_att`, this variant needed
  **no** added loss terms, because the velocity plant is the plant the
  original losses were designed for.
- **Rendering plumbing.** The original per-image `render()` + `ImageCache`
  (2-decimal pose key, 3000-entry FIFO, cleared on per-epoch intrinsics jitter)
  is reproduced as `CachedPinholeRenderer`, with cache misses rasterized in
  chunked GPU batches. Per-epoch camera-model DR uses the **original
  magnitudes** (fx,fy +-0.4%, cx,cy +-0.5 px, mount yaw/pitch/roll +-0.5 deg)
  applied to the OLD pinhole K.

### Training command

```bash
cd ~/certified_visual_controller/VerifiedVisualController_small_clone
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
SIDE_MEM_FRAC=0.25 \
python scripts_control/train_ctrl_meansub_ctbrvel.py --max-minutes 6
# repeat the same command until it prints "TRAINING COMPLETE" (resumes from
# _last). Defaults reproduce the original recipe: 120 epochs, 2000 poses/epoch,
# batch 32, lr 1e-3. Ran in resumable ~5-6 min foreground chunks on the shared
# RTX 5080 laptop at SIDE_MEM_FRAC=0.25 (peak ~2.4 GiB of the 3.86 GiB budget).
```

Quick-eval trajectory over the run (64 DR episodes, banked when improved):
ep15 9.7 cm -> ep30 6.6 cm -> **ep75 2.8 cm** -> ep90 2.3 cm -> **ep105 1.4 cm**
(banked) -> ep120 1.4 cm. The bank is **epoch 105**.

## Eval table (native definitions)

Metric definitions match `logs/eval_ctrl_meansub_att.json` exactly (episode =
random offset spawn from the native `sample_init_poses` ranges, 6.0 s closed
loop; `hold_err` = mean `||p-target||` over the last 1.5 s, median over
episodes, meters at 1 u = 0.85 m; crash = mat-plane clip `z_u>0.65`, or
`>3.0 u` divergence, or NaN), so the two variants are directly comparable.

| condition (240 eps) | median hold | mean | p90 | @10cm | crash | yaw err |
|---|---|---|---|---|---|---|
| original recipe (noise+lag+delay) | **1.4 cm** | 1.4 cm | 1.8 cm | 100.0% | 0.0% | 0.22 deg |
| + training image DR | 1.7 cm | — | — | 100.0% | 0.0% | — |
| pure integrator (no lag/delay/noise) | 1.3 cm | — | — | 100.0% | 0.0% | — |
| **original banner** | 3.1 cm | — | — | 84.2% strict | 0 | — |
| meansub_att (prior variant) | 62.6 cm | 64.2 cm | 95.8 cm | 0.0% | 29.2% | — |

## Why this variant parks and `meansub_att` orbits (velocity-plant damping)

Both variants share the 3ch-meansub input and a memoryless CNN (no recurrence,
no velocity state fed in). The difference is the **plant the head drives**:

- **This variant (velocity plant).** The head commands a body-frame *velocity*,
  and the kinematic integrator is a pure single integrator: `v_cmd = 0` means
  `pose` stops. Hover is a *static* fixed point — the controller only has to
  drive the command to zero at the target, which the original position/final-
  state losses directly reward. That is *implicit damping*: the plant itself
  removes energy when the command shrinks. Result: monotonic V decay to ~0 and
  a dead stop at the target (see any rollout video; final pose ~
  `[0, 1.5, 0, -pi/2]`), 1.4 cm hold.
- **`meansub_att` (attitude plant).** The head commands attitude+thrust; hover
  is a *dynamic* equilibrium (thrust must exactly cancel gravity while the tilt
  returns to level, with the drone's momentum carrying it past the target). A
  memoryless net with no velocity feedback cannot damp that second-order mode,
  so it limit-cycles — the documented ~0.6 m orbit and 29% mat clips, even with
  the added floor/velocity loss terms.

So reverting the plant to the original velocity integrator (this variant) is
exactly what buys back original-class quality — the point of the exercise.

## Load / run snippet

```python
import torch
from scripts_control.utils_ctrl_meansub_ctbrvel import (
    ControllerMeansubCtbrVel, ctbr_to_velocity, CtbrAsVelocity, render_pinhole)

net = ControllerMeansubCtbrVel()
ck = torch.load("weights/ctrl_lya_meansub_ctbrvel.pt", map_location="cpu",
                weights_only=False)
net.load_state_dict(ck["controller"]); net.eval()

# 1) as a CTBR-format net (what a verifier sees): img (B,3,200,300) in [0,1]
ctbr = net(img)                 # (B,4) [c m/s^2, wx, wy, wz rad/s]

# 2) as the sim uses it: CTBR consumed as the original velocity command
vel_body = ctbr_to_velocity(ctbr)   # (B,4) [vx, vy, vz, yaw_rate], the original box
#   ... then body_to_world_velocity(vel_body, yaw); pose += vel_world * dt

# 3) drop-in for the ORIGINAL test_ctrl_lya_pt.run_test (velocity-head API):
ctrl = CtbrAsVelocity(net)          # forward(img) == ctbr_to_velocity(net(img))
```

The rollout videos are produced by calling the **original**
`test_ctrl_lya_pt.run_test` verbatim (imported) with `ctrl = CtbrAsVelocity(net)`
and `render_fn = render_pinhole` — see `scripts_control/videos_ctrl_meansub_ctbrvel.py`.

## Files added (no originals modified)

- `scripts_control/utils_ctrl_meansub_ctbrvel.py` — `ControllerMeansubCtbrVel`,
  `ctbr_to_velocity`, `CtbrAsVelocity`, `render_pinhole` /
  `render_batch_pinhole_gpu` / `check_pinhole_parity`. Self-test:
  `python scripts_control/utils_ctrl_meansub_ctbrvel.py`.
- `scripts_control/train_ctrl_meansub_ctbrvel.py` — trainer (original recipe;
  losses/dataset/DR imported from `train_ctrl_lya_pt.py`).
- `scripts_control/eval_ctrl_meansub_ctbrvel.py` — quantitative eval
  (`--image-dr`, `--pure` switches).
- `scripts_control/videos_ctrl_meansub_ctbrvel.py` — rollout videos via the
  original `run_test`.
- `docs/58k_meansub_ctbrvel_retrain.md` — this doc.
