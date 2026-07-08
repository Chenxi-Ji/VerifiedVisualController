# 10 — Recurrence: who uses what in the literature, and why ours is a GRU

*2026-07-07. Companion note to 01_research_report.md and 09_network_io.md,
answering "which papers used a GRU?" precisely and recording the actual
provenance of our recurrence decision.*

## 1. Recurrence across the surveyed vision-flight literature

| work | memory mechanism | role | note |
|---|---|---|---|
| Forest navigation via relightable 3DGS (arXiv 2602.07101) | **GRU** | reactive RGB→RL policy memory | the explicit GRU precedent in our survey (01 §2.1) |
| GaussGym (arXiv 2510.15352) | **LSTM** | DinoV2 features + LSTM + auxiliary reconstruction, asymmetric AC RL from RGB | same role, different cell; also the aux-loss precedent we reused (velocity head) |
| Dream to Fly (2501.14377) / SkyDreamer (2510.14783) | recurrent **world model** (Dreamer-family RSSM; GRU-based internally) | memory lives in the model, not a reactive policy | different paradigm (MBRL) |
| Geles et al., RSS 2024 (2406.12505 — our anchor) | **none** — last 3 actions only | — | documented failure mode: lost when the gate leaves the frame for a few steps; the paper itself names a recurrent architecture as future work |
| Swift, Nature 2023 | **none** — 2×128 MLP on Kalman-filtered state + prev action | — | memory unnecessary because a state estimator (VIO+detector+KF) supplies velocity etc. |
| SOUS VIDE (2412.16346) | frame history via **optical flow input** + partial state | flow carries the motion information explicitly | the "explicit temporal feature" alternative to recurrence |
| Deep Drone Acrobatics (2006.05768) | temporal **feature-track** abstraction + IMU | ditto | |

Reading: every system that flies from vision either (a) carries memory
(GRU/LSTM/world model), (b) feeds an explicitly temporal input (flow,
tracks, action history), or (c) has a state estimator doing that job
upstream (Swift). The memoryless-with-action-history corner (Geles) works
for racing with a perception reward that keeps gates in frame, and its
brittleness when perception drops out is documented by its own authors.

## 2. Why OUR policy is recurrent (actual provenance — not a citation)

The honest chain, in order:

1. **Information analysis** (00_mission, 09_network_io): with no state
   estimator, velocity and the thrust-map residual are only recoverable by
   integrating observations over time → the policy needs memory of some
   kind. This was written down before any experiment.
2. **The expert experiment** (05 log): a memoryless PD expert stalls at 27%
   with a 37 cm steady-state offset; adding an integrator → 99%+. Integral
   action ≈ memory. Whatever the network is, it must be able to integrate.
3. **The export spike** — the decision *gate*: recurrence was only allowed
   into the design after `nn.GRUCell` (explicit hidden-state I/O tensors)
   survived PT→ONNX→onnx2tf→TFLite fp16 with 1e-4 closed-loop parity
   (spike_gru_export.py). Had it failed, the design fell back to frame
   stacking (Q5 in 00_mission).
4. **The privileged-velocity probe** (05 log, runs 5–8) later *proved* the
   information story: the 0.55–0.7 m plateau collapsed to 0.113 m when true
   velocity was injected — and the deployable fix (150 ms frame pair + aux
   velocity head) explicitly helps the GRU do its estimation job rather
   than replacing it.

## 3. Why a GRU specifically (vs LSTM / stack / transformer-ish)

- **GRU vs LSTM**: fewer parameters and fewer ops for the same role at this
  scale (hidden 96); one state tensor instead of two (h only, no cell
  state) — meaningfully simpler as an explicit input/output pair in the
  TFLite graph and the C++ helper. The GaussGym LSTM shows either works;
  nothing in our task needed LSTM's extra gate.
- **GRU vs frame stacking**: stacking N frames buys fixed-lag memory only,
  scales input linearly, and cannot integrate over seconds (the
  thrust-residual estimation needs multi-second horizons — the chained-
  window training exists precisely to teach that). We DID end up adding a
  minimal 2-frame stack anyway — for the *instantaneous velocity* signal —
  the hybrid: stack for fast signals, recurrence for slow estimation.
- **Kept deliberately boring**: GRUCell decomposes into matmul/sigmoid/tanh
  — all ancient TFLite builtins and all α,β-CROWN-supported ops, preserving
  the export path and the (future) verification story.

## 4. Cross-references

- Export proof: 05 log "recurrence export spike"; export_policy.py parity.
- Hidden-state runtime semantics (zero on gap, carry across invokes):
  09_network_io.md §1.3; pixel_ctbr_model_helper.cpp.
- Training the recurrence properly (chained windows, burn-in): 05 log v5;
  MASTER.md §5 items 4–6.
