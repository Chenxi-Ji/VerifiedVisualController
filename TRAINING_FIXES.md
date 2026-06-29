# Controller / Training / Network — Fix Plan (4-Lane Audit)

> **Date:** 2026-06-25
> **Companion to:** [`SIM2REAL_DIAGNOSIS.md`](SIM2REAL_DIAGNOSIS.md) (the original
> diagnosis + BN re-adaptation). This doc is the deeper audit: four parallel
> Opus analysts each took one subsystem — **(A)** network architecture,
> **(B)** training loop / gradient flow, **(C)** losses + Lyapunov, **(D)** domain
> randomization + rendering — and every claim below is backed by a live experiment
> in the `imcooked` env, not by reading code. The lanes were run independently and
> converged; where they independently agree, that is noted as a confidence signal.

---

## 0. TL;DR — what the audit changed

The original diagnosis said "vision sim-to-real OOD; structural not photometric;
fix is mostly the net." The 4-lane audit **sharpens and partly overturns** that:

1. **It is the GATE, not the background or the color.** A composite/swap experiment
   (Lane D) proved the network **memorized the specific sim gate's appearance**.
   Dropping the *sim* gate into the *real* photo recovers a hover action; the *real*
   gate breaks it regardless of background. **Gate = #1 OOD driver, background = #2,
   photometry = not a factor** (DR already brackets every global statistic).
2. **The rendered gate is objectively wrong:** a **double-ring "ghost"** (a splat
   defect that survives the `_cleaned` scene), too small, and ~9° too high in frame.
   So the cheapest high-impact fix is a **rendering** fix, not net surgery.
3. **The detached renderer is NOT the cure** (Lane B). gsplat's gradient flows fine
   once you swap `cv2.resize`→`F.interpolate`, but that adds no transfer signal. The
   binding causes are the degenerate single-scene distribution + privileged-pose-only
   supervision + an oversized head + a loss that *prefers* saturation.
4. **The "certificate" is mathematically invalid and unwired** (Lane C): `V` is not
   radially unbounded (collapses to 0 far away), is never gated on at deploy, and is
   pose-only so it cannot see the vision failure.

The reordered plan is in [Section 6](#6-unified-fix-plan-re-ordered). The headline:
**make the rendered gate look like the real gate first**, then rebuild the net to
transfer rather than memorize, then fix the monitor.

---

## 1. Lane D — Domain Randomization & Rendering (the decisive lane)

### 1.1 The swap test (single most important measurement)

Composite gate-region vs. background between the matched sim render (`y=-1.5`) and
the real frame (`real_t0.0`), push each through `weights/ctrl_lya.pt`, log the
pre-clamp action-head logit norm (OOD metric; ~0.1 sim, ~4.6 real):

| input | pre-clamp logit‖·‖ | action |
|---|---|---|
| SIM gate + SIM bg (baseline) | **0.49** | ≈ hover ✅ |
| **SIM gate + REAL bg** | **0.54** | ≈ hover ✅ |
| REAL gate + SIM bg | **2.46** | saturated ❌ |
| REAL gate + grey bg | 6.25 | saturated ❌ |
| REAL gate + REAL bg (real frame) | **8.28** | `[-1,-1,1,-0.3]` ❌ |

**The sim gate hovers on any background; the real gate breaks it on any background.**
The network memorized the sim gate. Background is a strong secondary (+5.8 logit on
top of the real gate). Per-cell ablation localizes the OOD energy to the
bottom-center gate arc + floor.

### 1.2 Global photometric/structural stats MATCH — DR already brackets them

| metric | sim | real | verdict |
|---|---|---|---|
| luminance std (RMS contrast) | 0.184 | 0.188 | match |
| Laplacian var (sharpness) | 2716 | 2718 | match (real is **not** sharper) |
| FFT high-freq fraction | 0.105 | 0.107 | match |
| edge fraction (Canny) | 0.170 | 0.131 | sim has **more** edges (ghost ring) |
| mean R / G / B | .43/.53/.54 | .46/.45/.43 | only real separator (color cast) |

Applying `DomainRandomizer` to sim renders, **every** real global stat falls inside
the augmented p2.5–p97.5 range. **Photometric DR is not the binding constraint and
should not be widened** (confirms the colorshift no-op from the diagnosis).

### 1.3 The gate geometry/structure mismatch (the `vx`-pinned killer)

| gate metric | sim @ target | real | note |
|---|---|---|---|
| hazard-pixel area | 0.050 | **0.089** | sim never reaches it even at y=−0.6 |
| bbox height | 113 px | **144 px** | sim matches only ~0.5 u closer |
| vertical center cy | 101 | **82** | needs pitch ≈ **−9°**; DR is ±2.5° |
| ring structure | **double-ring ghost** | single | splat defect, not aug-fixable |
| yellow components | 147 | 308 | real chevron is finer |

Real gate = **bigger, higher, single-ring, finer**; sim gate = **smaller, centered,
double-ringed, coarser**. The joint real combination is OOD → `vx` pins to −1.

### 1.4 Lane-D fixes (training-only; ops unconstrained)
- **P1 — Fix the gate.** Replace the splat-rendered gate with a **randomized CAD gate
  primitive** composited over the background: kills the ghost, lets you bracket
  texture/size/ring-thickness/saturation/sharpness so the net can't memorize one
  gate, and gives a **free segmentation mask** for the aux-localization head (Lane A/B).
- **P2 — Background compositing** behind the gate mask (real arena crops, the 2nd
  ring as a distractor, windows, clutter, procedural). Destroys the fixed-location
  background features the head exploits.
- **P3 — Pitch/roll DR ±10–15°** (currently exactly 0) + cy jitter → closes the ~9°
  vertical gap. One-line-ish change.
- **P4 — Matched camera model @ 192×256.** Pull real IMX412 intrinsics + k1..k4 from
  `/data/modalai/`, set `render()` to match, verify rendered gate apparent-size ==
  real, and **fix the stale `200×300 / fx≈113` doc** (actual is `192×256`, render
  fx=484@1024 ≈ 121@256). Must precede the retrain or the geometry stays wrong.
- **P6 — Leave photometric DR as-is** (already brackets real; if anything reduce
  saturation). Optional cheap insurance: directional motion blur, rolling-shutter shear.

```python
# P1+P2 sketch (training-only)
gate = render_cad_gate(pose, K, tex=choice(GATE_TEXTURES),
                       sat=U(0.5,1.5), sharp=U(0.5,1.5), ring_thick=U(0.8,1.2))
bg   = splat_render(pose) if rand()<0.5 else random_background(pose)
img  = composite(gate.rgb, gate.alpha, bg)     # alpha = free localization label
```

---

## 2. Lane A — Network Architecture

### 2.1 Where the 40× OOD blow-up is born (measured, layer by layer)

| stage | sim meanabs | real meanabs | real/sim |
|---|---|---|---|
| pooled feature (3072-d) | 0.175 | 0.334 | **1.9×** |
| head Linear(3072→128) out | 6.44 | 9.01 | 1.4× |
| after head ReLU #1 | 0.071 | 0.869 | **12.2×** |
| pre-clamp logits (L2) | 0.49 | 8.28 | **16.8×** |

**The backbone is only ~1.9× OOD; the 393k head manufactures the rest.** At the sim
operating point only 5/128 head units fire (the net balances "hover" on a razor's
edge); real frames light up silent units → blow-up → clamp saturation. Signature of
**memorization**. Also: the controller is **591,940 params, not the "52k" the docs
claim**, and the `Linear(3072→128)` head is **66%** of it.

**Pooling resolution controls transfer:** feature cos(sim,real) = **0.42** at the 4×6
readout vs **0.88** at global (1,1) — the fine spatial cells expose fixed-location
background structure. Receptive field is *not* a limiter (final cells see 66×66 px).

### 2.2 Lane-A fixes
- **P1 (DEPLOY) — Replace the 3072-d readout + 393k head** with a hybrid: global pool
  (64-d, transfers at 0.88, drives vx/vz) + a tiny **learned** lateral branch
  (`Conv2d 64→8, AdaptiveAvgPool (1,3)` = 24-d, drives vy/yaw) → 88-d head input,
  `Linear(88,64)` = 5.7k (vs 393k). Biggest sim2real *and* verifiability win.
- **P2 (DEPLOY) — Expose pre-clamp magnitude as a verifiable OOD score**
  (`relu(z)+relu(-z)`, then max) for a runtime abort gate. sim ≈0.5 vs real ≈8.3.
- **P3 (DEPLOY) — BN: AdaBN then fold+freeze at export.** Within the allowed op set
  this is the right cure for the train/deploy stats gap; **GroupNorm/LayerNorm do NOT
  fold and give loose, expensive α-β-CROWN bounds — not recommended.**
- **P4 (DEPLOY) — Drop the raw branch; per-image mean-sub only** (3-ch). Removes a
  non-transferring absolute-color crutch. (Lane D goes further — see §5.)
- **P5 (DEPLOY) — Halve backbone channels** 32/64/96/128 → 16/32/48/64 (~2× fewer
  unstable ReLUs to verify; less memorization).

Proposed re-architected `Controller`: **~55k params (10.8× smaller), ~2× fewer
conv-ReLU neurons to verify**, all ops in the allowed set. Full nn.Module sketch is
in the Lane-A report; key shape: `mean-sub(3ch) → 4 conv blocks(16/32/48/64) →
[global_pool(64) ⊕ lat_readout(24)] → Linear(88,64) → Linear(64,4) → clamp_relu`,
plus a train-only `aux_gate = Linear(88,3)`.

---

## 3. Lane B — Training Loop & Gradient Flow

### 3.1 Measured evidence

| probe | result | meaning |
|---|---|---|
| `render()` output | `grad_fn=None` (detached) | image is a hard constant |
| `F.interpolate` render → `means.grad` | 2.3M nonzero | gsplat **is** differentiable; cv2/cpu roundtrip breaks it |
| rollout backward: ctrl grads | 22/22 nonzero | controller trained **only via pose-dynamics path** |
| rollout backward: scene grads | **None** | **zero gradient through the image** |
| grad clip @1.0 | always binds (pre-clip 7→53 over H) | **clip, not LR, controls step size** |
| ImageCache hit-rate | **4.9%** | near-dead weight |
| `render_batch(32)` vs 32×`render()` | 269 vs 169 ms | **render_batch is SLOWER** (red herring) |
| differentiable render fwd | 4.26 vs 5.43 ms | GPU-interp path is also **faster** |

### 3.2 Lane-B fixes (all training-only — deployed graph untouched)
- **#1 (HIGH, do first) — Aux gate-localization head + loss.** Predict gate (u,v,size)
  in image space; label is free exact geometry; backprops only through the net (no
  differentiable renderer needed); dropped at export. Converts "regress pose from
  whole image" → "find the gate, then act" → transferable. **Independently proposed
  by Lane A.**
- **#2 (HIGH) — Fix the degenerate distribution.** `PoseDataset` uses
  `pitch/roll = 0` exactly and one fixed gate/background; vary gate pose, background,
  and **pitch/roll**, and inject body tilt during rollout (a real quad pitches 10–20°
  to translate; it sees frames the net never trained on).
- **#3 (MEDIUM, trivial) — Hygiene.** No seed anywhere (every run is a different
  experiment). `get_learning_rate()` is **dead code** (never called; real schedule is
  `lr*=0.95`/10ep). Resume is **dead** (`save_path` embeds `datetime.now()`). Add a
  fixed-seed held-out validation rollout — there is currently **no eval signal**.
- **#4 (LOW for transfer) — Differentiable render** (`F.interpolate`): correct, cheap,
  faster, unblocks future image-space losses, but **not the cure** on its own (adds
  temporal credit assignment, no new gate signal). Do as hygiene, don't expect it to
  move the needle.
- **#5 (MEDIUM) — Clip threshold** ~10–20 (or horizon-normalized) and **log pre-clip
  grad-norm**; align horizon jumps with the loss-weight phase boundaries.
- **#6 (LOW, perf) — Drop/repair the cache; render near 256×192** (rendering is ~6h/run
  and dominates); ignore `render_batch`.

---

## 4. Lane C — Losses & Lyapunov

### 4.1 `V` is not a valid CLF — far-field collapse (measured)

```
 pos_norm     V        alpha
   0.00     0.000     0.50
   5.58     7.318  <- PEAK
  20.00     0.353     0.0003
  50.00     0.000     0.0000   <- identical to V(target)!
```

`α = sigmoid(net(...))` saturates to 0 far out, and exp-decay of α beats the
polynomial `‖pos_err‖²`. So `V(target) = V(50 u away) = 0`, `{V≤0.5}` is **two
disjoint regions** (ball at target **and** the whole far field) → **not radially
unbounded, sublevel sets non-compact, no valid region-of-attraction**. As the real
drone backs away (vx pinned −1), V rises to ~7 then falls back toward 0, **re-arming a
naive "V<thresh ⇒ safe" check while it flies away.** The 1185-param `alpha_net` is
~constant (≈0.5) in-region and only breaks things out-of-region.

### 4.2 Lane-C fixes
- **Issue 1 (DEPLOY, CRITICAL) — Make `V` valid:** delete the alpha-net, use a fixed
  convex weight (or bound `α∈[0.25,0.75]`, or add a quadratic floor `c·pos_term`).
  Provably PD, radially unbounded, simpler, still verifiable. Re-export + re-parity.
- **Issue 2 (DEPLOY, CRITICAL) — `V` is never wired and is modality-blind.** At deploy
  `V` is only *plotted*; nothing gates/aborts on `V` or `dV`, and it's pose-only so it
  can't see the vision failure. Add a verifiable safety predicate combining `V`, `dV`,
  **and** Lane-A's pre-clamp OOD score (`SAFE = ood≤τ ∧ V≤V_env ∧ dV≤margin` → else
  HOLD). A *true* image-space CLF is **not** recommended (intractable verification +
  itself OOD).
- **Issue 3 (TRAIN, HIGH) — Penalize logit/action magnitude.** Nothing stops the loss
  from preferring saturation; the inflated reachability cap (Issue 5) actively rewards
  max speed. Add `w_logit·‖pre-clamp‖²` (~1e-3..1e-2). Keeps in-dist logits O(1) and
  **sharpens the OOD gap the monitor relies on.** This is the loss-side enabler of the
  `[-1,-1,+1]` failure.
- **Issue 4 (TRAIN, latent) — Yaw wrap.** traj uses `Δyaw²`, final uses `|Δyaw|`, V
  uses `1−cos` (only this is wrap-safe). Latent now (`|yaw_err|<π` in-box) but a
  landmine at deploy/wider ranges. Make both losses use `1−cos`.
- **Issue 5/6/7 (low) — `3×` reachability factor** defeats the progress loss
  (collapses to plain distance); thresholds are unit-fragile; the alpha-net is a no-op
  with a mislabeled entropy term; `V` ignores pitch/roll while the losses include them.

---

## 5. Cross-lane agreements (confidence signals)

Independent lanes converging on the same fix is strong evidence:

| recommendation | lanes that independently reached it |
|---|---|
| **The GATE appearance is the dominant OOD driver** | D (swap test) + A (gate-cell OOD) |
| **Mean-sub branch hurts — drop it** | A (80% of conv1 shift) + D (it's the structural conduit) |
| **Aux gate-localization head** | A + B |
| **Pre-clamp-logit OOD score + safety gate** | A + C |
| **Shrink the net / global pool over 4×6** | A + D |
| **Pitch/roll is missing (always 0)** | B + D |
| **Photometric DR is not the problem** | D (bracketing) + diagnosis (colorshift no-op) |

---

## 6. Unified fix plan (re-ordered)

| Pri | Fix | Lanes | Tag |
|---|---|---|---|
| **1** | **Fix the rendered gate**: kill double-ring ghost; randomized CAD gate primitive (texture/size/ring/sharpness) → can't memorize one gate; free seg mask | D | TRAIN |
| **2** | **Matched camera model @ 192×256** + fix stale `200×300/fx113` doc (precede retrain) | D | TRAIN/doc |
| **3** | **Background compositing** behind gate mask + distractors (2nd ring, windows) | D | TRAIN |
| **4** | **Pitch/roll DR ±10–15°** + body tilt in rollout (currently exactly 0) | D + B | TRAIN |
| **5** | **Aux gate-localization head + loss** (free label from the mask) | A + B | TRAIN |
| **6** | **Net redesign ~55k** (global pool + compact lateral readout, halved channels) + **drop mean-sub** | A + D | DEPLOY |
| **7** | **Logit-magnitude penalty** + **yaw wrap-safety** + seed/eval/dead-code hygiene | C + B | TRAIN |
| **8** | **Valid CLF** (delete alpha-net) + **wire verifiable OOD/V safety gate** | A + C | DEPLOY |
| **9** | **AdaBN → fold/freeze**, then export + re-verify | A | DEPLOY |

**Execution sequence:** #2 (camera) → one retrain doing #1,#3,#4,#5,#6,#7 →
#8 (CLF + gate) → #9 (calibrate) → export → verify the final shipped net.

**Framing shift:** this is **less a neural-net problem than a rendering problem**. The
swap test recovered hover just by substituting the gate pixels — so the highest-impact,
lowest-effort opening move is **#1–#2: make the rendered gate match the real gate.**

---

## 7. Not yet covered / open items

- **Real camera intrinsics + distortion (k1..k4)** are not in the repo (TODO in
  `GATE_ARENA_SETUP.md`). #2 is blocked on pulling them from the drone.
- **A CAD gate primitive** (or a clean re-capture/splat-cleanup of the gate) needs to
  be built — this is the load-bearing asset for #1/#3/#5.
- **The verification scripts still do not exist** (the "Verified" claim has no runnable
  artifact — see `SIM2REAL_DIAGNOSIS.md`). #6/#8 reset the verification target, so the
  verifier must run on the final shipped net.
- **Per-lane full reports** (with complete code sketches and reproduce scripts) live in
  the agent transcripts; scratch experiment scripts are under the session scratchpad.
