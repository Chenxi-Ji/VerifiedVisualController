# 08 — Lyapunov in pixel2ctbr: decisions, analysis, and the incorporation plan

*2026-07-07. The explicit record of what happened to the Lyapunov machinery
from the velocity-controller phase, why, what each incorporation option
costs and buys, and the exact plan for the pieces worth doing. Companion to
04_design.md §7 (where "no V in v1" was decided) and MASTER.md §8.*

---

## 1. What the original system had (baseline facts)

From the previous phase (PROJECT_STATE.md, `utils_ctrl_lya_pt.py:224-300`):

- **Form**: `V = 3·[α·‖e_p‖² + (1−α)·(1−cos Δyaw)]` with
  `α = σ(MLP(‖e_p‖/2, (1−cosΔyaw)/3))`, MLP 2→32→32→1, 1,185 params.
  Inputs: privileged pose error only (position, yaw). Never saw images.
- **Roles**:
  1. *Training*: decrease-condition losses along BPTT rollouts (two regimes:
     (ΔV+margin)⁺² above a floor, push V²→0 below it) + smoothness + α-entropy.
  2. *Analysis*: V(t) plots for sim and real flights; the real first flight
     was judged by V 0.68→0.017.
  3. *Runtime monitor (nominal)*: a `V_MAX` abort in the offboard script —
     **disabled in practice** because onboard V consumed a VIO pose through
     an identity-placeholder transform ⇒ garbage values (150k–480k).
- **Status of guarantees**: empirically valid CLF behavior (measured
  monotone along rays to 50 u) but **no structural guarantee** (the α bound
  was never applied). It was a *learned monitor with certificate-shaped
  training*, not a proof.
- **Trusted path in practice**: V recomputed **offline** from mocap ground
  truth (`plot_flight.py`); the onboard value was never load-bearing.

## 2. Why v1 of pixel2ctbr shipped without it (the decision, restated precisely)

1. **The runtime input ceased to exist.** V is a function of pose error;
   this phase's premise is *no pose estimate in the loop* (no mocap, no
   VIO). The old "onboard V is garbage" bug became structural: there is
   nothing onboard to evaluate V(e_p, Δyaw) on. Only the offline/mocap path
   survives — and that path never needed changes.
2. **The state space outgrew the form.** The old plant was a velocity
   integrator: (e_p, Δyaw) *was* essentially the closed-loop state. The CTBR
   plant adds velocity, tilt, body rates, actuator lag states, and a
   150 ms-history-carrying GRU. A function that certifies convergence must
   decrease over the full closed-loop state; ‖e_p‖² + (1−cosΔyaw) does not
   extend trivially (e.g. a drone AT the target moving 2 m/s has V≈0 and is
   about to leave — V must weigh velocity).
3. **Its training role had a cheaper replacement.** The decrease losses'
   function — "make monotone progress, arrive gently" — is served by the
   chained-window losses: time-weighted position Huber (progress), terminal
   position+velocity cost (the code comments call it the poor man's
   SHAC-critic/decrease condition), the near-ring precision regime (the old
   "V≤0.02 ⇒ push V²→0" analogue — the lineage is explicit in the code),
   and the time-outside-ring term. Empirically these carried the project to
   3 cm hover; adding a co-trained V during the run-1..14 debugging era
   would have been one more moving part in a period where *every* extra
   pressure needed attribution runs and ramps (see MASTER §5 items 9–11).

None of these reasons is "Lyapunov is worthless here" — they are ordering
decisions. What follows is the honest menu.

## 3. The option space, analyzed

### L1 — Offline/analysis V over the full CTBR state  ✅ DO (cheap, high value)

**What**: extend the Lyapunov class to privileged state
`V(e_p, v, tilt, e_yaw)` (e.g. `V = α₁‖e_p‖² + α₂‖v‖² + α₃(1−cosΔyaw) +
α₄‖tilt‖²` with a small α-net as before, or a slightly wider MLP with a
positive-definite head), and **fit it to the converged policy** ("certificate
extraction": train V so ΔV<0 along successful eval rollouts and V=0 only at
the target set — the policy is FROZEN; V adapts to it, not vice versa).

**Buys**: the V(t) convergence plots for sim and real flights (like-for-like
comparison with the previous phase's first-flight evidence — reviewers and
the lab already speak this language); a scalar convergence metric for eval
tables (fraction-of-episodes-monotone, time-to-sublevel-set); a principled
basin visualization per task (hover, two-gate phase segments, three-gate).

**Costs**: ~a day. No effect on training or deployment. Uses the existing
offline-recompute path in `plot_flight.py` (which already recomputes V from
mocap for the old phase — swap the function).

**Multi-gate note**: the transit phase machine gives piecewise targets; V is
evaluated per phase segment against the current phase target and RESETS at
phase switches (piecewise-CLF reading; the crossing events mark the
switching surfaces). Report per-segment decrease fractions.

### L2 — Co-training V (decrease losses in the trainer)  ⏸ HOLD (only on evidence)

**What**: the old phase's arrangement — V trains alongside the policy and
the policy is penalized for ΔV>0 along windows.

**Analysis**: the chained-window losses already encode the same pressure
(§2.3). The project's hard-won process lesson (v11/v12: 15-pt regressions
from abruptly added pressures) says a new co-trained loss enters only with a
ramp and an attribution run — that's ~3 GPU-hours per experiment. There is
currently **no observed failure it would fix**: hover converges, transit
oracles are 100%, and the trajectory trainings are mid-run. Where it MIGHT
earn its place: if trajectory training exhibits phase-boundary instability
(oscillating between committing/retreating at gate approaches) — a
per-phase decrease condition is a natural regularizer for exactly that.

**Decision**: hold. Trigger = a measured instability in two/three-gate
training that terminal+ring shaping doesn't fix. If triggered: fit V per L1
first, then add the decrease loss with weight ramped over ≥6 epochs, one
change at a time, definitive-eval before/after.

### L3 — Runtime monitor (the "V slot")

Three sub-options, very different in viability:

- **L3a — pose-based V onboard**: DEAD by construction (no pose). Revisit
  only if a future estimator returns to the stack (e.g. gate-PnP from the
  group's MapPilot line — at which point the whole architecture
  conversation reopens).
- **L3b — observation-space health/OOD monitor**: ✅ DO — this is the
  *deployable analogue* of the old V_MAX and the old phase's open P2 item.
  The network's internals (pre-clamp head logits, GRU hidden norm/drift)
  separate in-distribution from broken-input regimes; the OLD phase
  measured |logit|∞ ≤1.6 in-dist vs 4.6–8.3 broken — **those thresholds do
  NOT transfer to the new net and must be recalibrated**. New capability
  since then: `scene_edit.delete_gaussians` can render the arena **without
  the gate** — a perfect true-OOD probe (plus lights-down gamma, occlusion
  cutouts, empty-wall aim). Ship the scalar in the reserved `V` slot of the
  wire message (zero format change), and wire the abort into
  `ctbr_offboard_common.py`'s EXISTING ladder: sustained V>threshold for
  N frames ⇒ treat exactly like staleness (hover-hold ⇒ handoff). Cost:
  ~half a day + a calibration script; zero new deployment plumbing.
- **L3c — learned image-conditioned certificate V(o_t, h_t)**: research
  (neural certificates with perception in the loop). Publishable; not
  flight-blocking; would build directly on L1's extracted V as the
  supervision target. Park it.

### L4 — Formal verification (α,β-CROWN)  📌 PRESERVED, NOT PURSUED

The architecture was kept verification-friendly on purpose (clamp_relu,
convs, dividing pools, GRU = matmul/sigmoid/tanh — all CROWN-supported).
Honest scoping: what is verifiable near-term is **per-step output bounds
under input sets** (e.g. "for any image and bounded IMU in this set, rate
commands stay within X of hover") — useful for a safety argument about
bounded aggressiveness, NOT a closed-loop stability proof (that would need
the certificate of L3c plus reachability through the plant — a paper, or
two). The old phase's never-executed CROWN run remains never-executed; the
door remains open.

## 4. The concrete incorporation plan (if/when executed)

Ordered; each step independently shippable; none blocks the C1–C7 flight
ladder:

1. **L1 (analysis V)** — new `pixel2ctbr/lyapunov.py`: `LyapunovCTBR`
   class (α-net pattern, extended state), `fit_certificate.py` (frozen
   policy, decrease + boundary losses over eval-rollout state streams from
   all three tasks), validation gates: ≥98% of successful episodes
   monotone after a 1 s transient; V=0 iff in the success set (tolerance);
   level-set plots per task. Wire into `eval_policy.py` (report
   monotone-fraction) and `plot_flight.py` (V(t) panel via the existing
   offline-recompute path). *Do after the trajectory trainings converge, on
   their eval rollouts, so one fit covers all three tasks.*
2. **L3b (OOD monitor)** — `calibrate_ood.py`: collect head-logit/hidden
   stats over (a) in-dist eval episodes, (b) gate-deleted scene, (c)
   photometric extremes, (d) cutout-occluded frames; pick the statistic
   with the cleanest margin (start with pre-clamp |logit|∞, the old
   phase's winner); threshold at ≥3× in-dist p99. Export unchanged
   (compute the statistic in the C++ helper from the action tensor's
   pre-clamp values — NOTE: clamp is inside the graph, so ship the
   statistic via the V slot computed helper-side from h_out norm instead
   if logits prove unreachable; decide during implementation). Runner: N=5
   consecutive frames over threshold ⇒ hover-hold, 20 ⇒ handoff; log every
   trip. Bench-test in C3 with the lens covered.
3. **L2 (co-training)** — only on its trigger (see §3), with the
   ramp+attribution discipline.
4. **L3c / L4** — park; revisit post-first-flights, ideally with real-log
   residuals folded into the sim first (Swift recipe, 01 §3), because a
   certificate fitted to an unfaithful sim certifies the wrong system.

## 5. Decision table (the whole doc in six rows)

| option | verdict | trigger/when | cost | blocks flight? |
|---|---|---|---|---|
| L1 offline analysis V (full state) | **do** | after trajectory trainings converge | ~1 day | no |
| L2 co-trained decrease losses | hold | measured phase-boundary instability | ~3 GPU-h/experiment | no |
| L3a pose-based onboard V | dead | a pose estimator returns to the stack | — | no |
| L3b OOD/health monitor in V slot | **do** | before/with first flights (C3 testable) | ~0.5 day | no |
| L3c image-conditioned certificate | park | post-flight, post-residual-fitting | research | no |
| L4 α,β-CROWN output bounds | preserved | when the safety argument needs it | days | no |

*The through-line: the old phase's real, load-bearing Lyapunov value was the
offline analysis and the discipline it imposed — both survive and extend
(L1). The runtime ambition survives in its only honest no-pose form (L3b).
Everything certificate-flavored beyond that is genuinely open research, kept
deliberately unblocked by architecture choices already made.*
