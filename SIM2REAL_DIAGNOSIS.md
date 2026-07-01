# Sim-to-Real Diagnosis — Why the Controller Saturates on the Real Drone

> **Date:** 2026-06-25
> **Scope:** Root-cause of the lateral/approach failure of `weights/ctrl_lya.{pt,tflite}`
> on real onboard frames, with live evidence, and the fix list (BN re-adaptation
> explained in full).
> **Repro env:** `conda activate imcooked` (torch + gsplat + cv2 + CUDA). TFLite
> parity was checked separately; the PyTorch `Controller` is a faithful stand-in
> (PT↔TFLite max action diff ≤ 0.006 on the real video).

---

## 0. TL;DR

- **The controller is correct in sim and saturates into garbage on real frames.**
  At the *same* centered pose, sim → near-zero hover action; the real frame →
  `[vx,vy,vz,yaw] = [-1, -1, +1, -0.3]` (full reverse + hard left + dive).
- **It is a pure vision sim-to-real (out-of-distribution) failure**, not export,
  not quantization, not control logic, and **not** global color.
- The mechanism is measurable: the real image drives the **pre-clamp action logits
  to ≈4.6 vs ≈0.1 in sim — a ~40× blow-up** — and the ReLU clamp pins every channel
  to its limit. **The "safety" clamp is converting OOD into confident, wrong,
  full-speed commands.**
- Cheapest first fix to try: **BatchNorm re-adaptation (AdaBN)** on real frames
  (Section 5). Bigger fixes: structure/background domain randomization + retrain,
  shrink the model, an OOD/abort gate, and a hybrid VIO+vision deploy.

---

## 1. What was tested

Two complementary sources of real-world behavior were used:

1. **Recorded real flight video** `gate_test.mp4` (729 frames, 24.3 s) replayed
   through both models by `scripts_tflite/replay_real_video.py`
   (→ `gate_test_actions.csv`). Preprocessing matches the drone exactly
   (BGR→RGB, `cv2.INTER_LINEAR` → 256×192, `/255`).
2. **A live, controlled experiment** in `imcooked`: render the gsplat scene at
   known offsets and push each render through the PyTorch `Controller`; then push
   the saved **real** frames (`real_t0.0_256x192.png`, `real_t0.0_colorshift.png`)
   through the *same* network. For every input we also logged the **backbone
   feature magnitude** and the **pre-clamp action logit magnitude** to detect OOD.

Action convention: body-frame `[vx (fwd+), vy (right+), vz (down+), yaw_rate (CW+)]`,
scene units / s. Target hover pose is `x=0, y=-1.5, z=0, yaw=π/2` (facing the gate).

---

## 2. Decisive evidence

### 2.1 The controller is correct *in sim*

**Lateral sweep** (drone at `y=-1.5` facing gate; `vy` should pass through ~0 at
center and flip sign — accounting for the `yaw=π/2` body→world rotation the signs
are correct, i.e. it drives back toward x=0):

```
 x_off    vx      vy      vz     yaw  | feat  raw_t raw_r
 -1.0   -1.000  -1.000  -1.000  -0.300 | 0.26  4.45  2.86
 -0.5   -0.427  -1.000  +0.617  -0.300 | 0.22  2.00  1.13
 +0.0   -0.113  -0.019  +0.059  +0.142 | 0.18  0.11  0.47   <- centered: ~hover
 +0.5   +0.210  +1.000  +1.000  +0.300 | 0.22  1.69  2.53
 +1.0   -0.487  +1.000  +0.862  +0.300 | 0.23  1.76  2.98
   vy range across x = 2.000  -> TRACKS in sim
```

**Forward sweep** (centered; `vx` should be + when far, ~0 at target `y=-1.5`,
− when too close):

```
 y_off    vx      vy      vz     yaw
 -2.5   +1.000  ...                      far  -> approach (+)   ✅
 -2.0   +1.000  ...                      far  -> approach (+)   ✅
 -1.5   -0.113  -0.019  +0.059  +0.142   target -> ~hover       ✅
 -1.0   -1.000  ...                      too close -> back up   ✅
 -0.5   -1.000  ...                      too close -> back up   ✅
```

So the **control logic, frame conventions, distance cue, and lateral tracking all
work in simulation**. In-distribution images give controlled, unsaturated outputs
(`raw_t≈0.1` at the operating point).

### 2.2 The same network on the matched real frame

```
 frame                vx      vy      vz     yaw  | feat  raw_t raw_r
 SIM  x=0           -0.113  -0.019  +0.059  +0.142 | 0.18  0.11  0.47   ✅ hover
 REAL raw           -1.000  -1.000  +1.000  -0.300 | 0.33  4.58  5.84   ❌ saturated
 REAL colorshift    -1.000  -1.000  +1.000  -0.300 | 0.33  4.74  6.13   ❌ identical
```

Same approximate pose (drone centered, ~1.5 u in front, facing the gate). Sim →
hover. Real → **every channel slammed to its clamp limit.**

### 2.3 The real-video phases confirm it (from `gate_test_actions.csv`)

```
 phase                         vx      vy      vz     yaw   n
 LEFT of center  (want vy>0)  -1.000  +0.234  +1.000  -0.300  105
 RIGHT past mid  (want vy<0)  -1.000  -0.031  +0.801  -0.220  180
 back to MID     (want vy~0)  -1.000  -0.633  +0.853  -0.300   60   <- hard left at center
 MID + right yaw (want yaw>0) -0.892  -0.573  +0.584  -0.300  120   <- yaw pinned wrong way
 fwd/close->orig (want vx<0)  -1.000  -0.311  +0.803  -0.133  264
```

Overall on the real video: `vx mean = -0.982` (≈ pinned full reverse the entire
clip), `vz mean = +0.799` (constant dive), `yaw` pinned at the ∓0.3 limit. On a
real drone this is "fly backward, descend, spin" — the opposite of approaching.

---

## 3. What this rules in and out

| Hypothesis | Verdict | Evidence |
|---|---|---|
| TFLite export / float16 quantization | **Ruled out** | PT↔TFLite max diff ≤ 0.006 on real frames |
| Control logic / coordinate frames | **Ruled out** | Correct lateral + forward behavior in sim |
| Global color / white-balance gap | **Ruled out** | Color-shifted real frame → *byte-identical* saturated output (raw 4.58→4.74) |
| **Vision sim-to-real OOD** | **CONFIRMED** | Pre-clamp logits 0.1 (sim) → 4.6 (real), ~40×; clamp saturates all channels |

The gap is **structural/textural, not photometric.** Look at the inputs: the splat
gate is a smooth, hazy, low-contrast torus on muted blue mats; the real frame is a
sharp **segmented polygonal** ring with bright **windows** and arena clutter on the
right that the splat never reconstructed. The network keyed on splat-specific
statistics that simply do not exist in the real image.

---

## 4. Root causes in the network / training

1. **Gradient never flows through the image.** `render()` does
   `.detach().cpu().numpy()` + `cv2.resize`, so the controller is trained as
   *supervised regression from image → an action label derived from privileged
   ground-truth pose* (`compute_traj_loss` / `compute_final_state_loss` read the
   true pose). Nothing forces it to use the **causal, transferable** cue (gate
   position/size in the image) — any splat-specific feature that correlates with
   pose on the training set minimizes the loss equally well, and those features
   vanish on real images.

2. **Massively oversized for one fixed scene: 592,584 params** (not the "52k"
   claimed in `README.md`, `GATE_ARENA_SETUP.md`, and the `Controller` docstring).
   The `Linear(3072→128)` head alone is ~393k. Huge capacity + a single
   gate/target/background ⇒ it memorizes the splat. The 2× channel widen likely
   made sim-to-real *worse*. (It also kills the alpha-beta-CROWN "verifiable"
   premise — 592k params over a 147k-dim image input is not practically
   verifiable.)

3. **BatchNorm runs on sim statistics.** At inference BN normalizes with the
   running mean/var estimated from sim renders; real low-level statistics differ
   (measured backbone feature magnitude 0.18 → 0.33), compounding the shift. This
   is the part the **BN re-adaptation** in Section 5 targets.

4. **Domain randomization is photometric + tiny-geometric only.** It covers
   contrast / color / saturation / exposure / gamma / noise / blur / cutout +
   ±1° rotation / ±2% scale / ±8% intrinsics — but **not** real background
   structure (windows, clutter), the gate's true segmented appearance, real
   sharpness/contrast, or lens-distortion residual. The colorshift no-op proves
   photometric DR is not the binding constraint; **structure is**.

5. **The mean-subtract "invariance" is not enforced.** The 6-channel input keeps
   the raw absolute-color channel `[raw, raw - mean]`, and the conv freely uses
   the raw channel because it is informative in sim.

6. **The Lyapunov "certified" monitor cannot catch this.** `V` is computed from
   **pose (VIO)**, not the image. When vision fails, pose can still read
   near-target, so `V` stays low and greenlights garbage actions. The certified
   monitor does not observe the failing modality.

---

## 5. BN re-adaptation (AdaBN) — what it is and why it is the cheapest first fix

### 5.1 What BatchNorm actually does

Each `BatchNorm2d` layer transforms a channel's activations as:

```
y = gamma * (x - mu) / sqrt(var + eps) + beta
```

- **Training mode** (`model.train()`): `mu`, `var` are the **current batch's**
  statistics, and the layer *also* maintains a running estimate
  `running_mean`, `running_var` by exponential moving average (`momentum`).
- **Eval mode** (`model.eval()`): the layer uses the stored
  `running_mean` / `running_var` (NOT the batch) with the learned `gamma`, `beta`.

Crucially, those running statistics were estimated **entirely from the training
distribution = gsplat/sim renders.**

### 5.2 Why that breaks on real images

Real frames have different per-channel low-level statistics (brightness, contrast,
texture energy, the windows, the sharper edges). So the activations entering each
BN layer have a different true mean/variance than `running_mean` / `running_var`
captured from sim. The normalization `(x - mu_sim)/sqrt(var_sim)` therefore does
**not** map real activations back to ~unit scale — they come out shifted and
mis-scaled, the next conv+ReLU sees out-of-range inputs, and the error **cascades
layer by layer** until the action head emits the ≈4.6 logits we measured (vs ≈0.1
in sim). The clamp then pins everything to ±limit.

This is a well-known effect. **AdaBN** (Li et al., 2016, *"Revisiting Batch
Normalization for Practical Domain Adaptation"*) showed that simply **recomputing
the BN statistics on the target domain — with zero labels and zero gradient
steps — recovers a large fraction of the domain gap.**

### 5.3 The fix, mechanically

Keep **all learned weights frozen** (conv kernels, `gamma`, `beta`, the FC head).
Only refresh the BN running statistics by forward-passing **real** frames:

```python
import torch, cv2, glob, numpy as np
import sys; sys.path.insert(0, "scripts_control")
from utils_ctrl_lya_pt import Controller

dev = "cuda"
ctrl = Controller().to(dev)
ckpt = torch.load("weights/ctrl_lya.pt", map_location=dev)
ctrl.load_state_dict(ckpt["controller"])

# 1) reset BN running stats and switch them to *cumulative* averaging.
#    momentum=None => running stat becomes the exact mean over all batches seen.
for m in ctrl.modules():
    if isinstance(m, torch.nn.BatchNorm2d):
        m.reset_running_stats()
        m.momentum = None
        m.train()          # BN updates stats ...
# everything else stays in eval (Dropout off); no optimizer, no backprop.
for m in ctrl.modules():
    if not isinstance(m, torch.nn.BatchNorm2d):
        m.eval()

# 2) feed REAL frames (e.g. every frame of gate_test.mp4) in train mode.
def load_real(p):
    bgr = cv2.imread(p); rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (256, 192), interpolation=cv2.INTER_LINEAR)
    return torch.from_numpy(rgb.astype(np.float32)/255.).permute(2,0,1)

cap = cv2.VideoCapture("gate_test.mp4"); frames = []
while True:
    ok, bgr = cap.read()
    if not ok: break
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (256,192), interpolation=cv2.INTER_LINEAR)
    frames.append(torch.from_numpy(rgb.astype(np.float32)/255.).permute(2,0,1))
batch = torch.stack(frames).to(dev)

with torch.no_grad():
    for i in range(0, len(batch), 64):       # a few hundred frames is plenty
        ctrl(batch[i:i+64])                   # updates BN running_mean/var only

# 3) back to eval: BN now normalizes with REAL statistics.
ctrl.eval()
torch.save({**ckpt, "controller": ctrl.state_dict()}, "weights/ctrl_lya_bnadapt.pt")
```

Notes:
- The controller's `forward` concatenates `[raw, raw - mean]` → 6 channels before
  the first conv/BN, so feeding real frames through `ctrl()` exercises every BN
  layer with the real distribution — exactly what we want.
- `momentum=None` makes each `running_*` a true cumulative average over all frames
  seen (order-independent, no need to tune momentum).
- Use frames that span the **operating** distribution (the `gate_test` onboard
  frames are ideal — real sensor, near the gate).

### 5.4 How to measure whether it worked

Re-run the Section 2.2 comparison on the BN-adapted weights. Success looks like:

- The **pre-clamp logit magnitude on the real centered frame drops from ≈4.6
  toward O(1)**, and
- The centered real frame produces a **near-hover** action instead of
  `[-1,-1,+1]`, and `vy` starts varying with the real lateral phases in
  `gate_test_actions.csv` instead of sitting pinned.

### 5.5 Caveats / where BN-adaptation stops

- BN-adaptation fixes a **distribution-statistics** mismatch, not a **semantic**
  one. If the network is partly keying on the background windows or the
  splat-vs-real gate *shape*, recomputing BN stats cannot invent the right
  feature — it only re-centers/re-scales the ones that exist. So treat it as
  **necessary but possibly not sufficient**: the expected outcome is "much less
  saturated, partially tracking," after which the structural DR retrain (Section
  6) closes the rest.
- **Re-export after adapting.** BN folds into the preceding conv at inference /
  TFLite export, so the folded conv weights change — you must re-run
  `export_to_tflite.py` and re-check parity. (The inference graph stays conv-only,
  so it remains verification-friendly.)
- A more permanent cure is to **remove the batch dependence entirely**: switch
  `BatchNorm2d` → `GroupNorm` (or LayerNorm), which use no batch/running stats and
  so have no train-vs-deploy statistics gap. That needs a retrain but removes this
  failure mode by construction.

### 5.6 Measured result (this repo, 2026-06-25)

BN stats were recalibrated on the 729 `gate_test.mp4` frames (`momentum=None`,
4 BatchNorm2d layers, no gradient), tested on the **held-out** `real_t0.0` still
and the sim sweeps. Adapted weights saved to `weights/ctrl_lya_bnadapt.pt`
(`ctrl_lya.pt` and the `.tflite` left untouched).

| metric (real held-out frame) | BEFORE | AFTER | change |
|---|---|---|---|
| backbone feature magnitude | 0.33 | **0.22** | → toward sim's ~0.15 |
| pre-clamp `|v_t|` logit | 4.58 | **2.90** | **−37%** |
| pre-clamp `|yaw|` logit | 5.84 | **3.32** | **−43%** |
| final action `[vx,vy,vz,yaw]` | `[-1,-1,+1,-0.3]` | `[-1,-1,+1,-0.3]` | still saturated |

Sim behavior preserved (lateral `vy` range 2.0, still tracks; centered ≈ hover).
Real-video per-phase `vy` after adaptation: LEFT `+0.31` (correct sign), RIGHT
`+0.09` (should be negative), MID `-0.26`; `vx` still pinned `-1.0` everywhere.

**Conclusion — necessary, cheap, confirms the diagnosis, not sufficient alone.**
BN re-adaptation removed ~40% of the OOD activation blow-up *for free*, proving the
BN-statistics mismatch is a real and significant slice of the gap. But the residual
logits (2.9) still exceed the ±1 clamp, so the action stays saturated, and the
**forward/distance channel `vx` stays pinned** — it leans hardest on splat-specific
cues (gate apparent size / FOV / scale). The remaining gap is **semantic/structural**
and needs the Section 6 work: structure+background DR retrain, a smaller model, and
the OOD/abort gate. Run BN-adaptation as a calibration step *after* that retrain,
not as a substitute for it.

---

## 6. Full fix list, by leverage

**Highest leverage — close the gap at the data / statistics level:**
- **BN re-adaptation on real frames** (Section 5). Minutes, no labels, no retrain.
- **Structure + background domain randomization, then retrain.** Composite random
  real backgrounds behind a segmented gate, randomize the gate texture toward the
  real hazard-stripe look, and widen contrast/sharpness to *bracket* real. This
  targets the actual (structural) gap that color DR misses.
- **Match the camera model.** Pull the real IMX412 intrinsics + distortion off the
  drone and render with the matching fisheye model **at 192×256**. Also fix the
  deploy doc: `GATE_ARENA_SETUP.md §7` says `200×300 / fx≈113` while the model is
  `192×256 / fx≈484` (`export_to_tflite.py` exports `1×3×192×256`). A wrong input
  size at deploy is itself a geometry gap.

**Make it transfer instead of memorize:**
- **Shrink the model** (drop the giant FC head / channels). Smaller + heavy aug
  generalizes better here and restores verifiability. Correct the "52k" claims to
  the real count once re-sized.
- **Auxiliary gate-localization head:** also predict the gate's pixel
  position/size (the label is free — you know the gate projection). This forces
  the net to actually *localize the gate*, which transfers far better than
  end-to-end pose regression and gives the image→action coupling that detached
  rendering currently removes.
- **GroupNorm instead of BatchNorm** (removes the train/deploy stats gap entirely).

**Deployment architecture — use what the drone already has:**
- **Runtime OOD / abort gate** on the pre-clamp logit norm or feature norm:
  sim≈0.1 vs real≈4.6 is *trivially separable today*. If the input is OOD →
  HOLD/abort instead of commanding full speed. Wire it into the Lyapunov monitor
  so "certified" actually covers the vision modality.
- **Hybrid VIO + vision:** VIO already provides metric pose (it feeds `V`). Let
  VIO handle range/height/approach (`vx`, `vz`) and let the fragile vision net do
  only the residual **lateral / yaw alignment** — don't ask it to regress all four
  DoF from pixels.

---

## 7. Reproduce this diagnosis

```bash
conda activate imcooked        # torch + gsplat + cv2 + CUDA

# sim sweeps + sim-vs-real on the matched frame, with OOD (feature/logit) stats
python scripts_tflite/sim_lateral_test.py            # live renders (needs gsplat)

# real video -> action sequence + per-phase means (needs the recorded clip)
python scripts_tflite/replay_real_video.py gate_test.mp4

# single real frame sanity
python scripts_tflite/replay_real_frame.py real_t0.0_256x192.png
```

Key artifacts already in the repo: `gate_test_actions.csv`,
`real_t0.0_256x192.png`, `real_t0.0_colorshift.png`, `sim_lat_*.png`,
`sim_fwd_*.png`.
