# THEORY — Everything in this project, explained from first principles

*A teaching document. Goal: you can read this once, slowly, and then
explain every design decision, every equation, and every lesson to someone
else — a lab meeting, a committee, a reviewer. Every symbol is defined at
first use. Sections build on each other but each has a "so what" summary.
Code cross-references point at the ground truth.*

---

## Table of contents

1. Frames, rotations, and quaternions
2. Quadrotor dynamics and the CTBR abstraction
3. The IMU: what an accelerometer actually measures
4. Classical control: the geometric expert, PID, and integral action
5. Information & observability: what a camera can and cannot tell you
6. The network, block by block, with the math
7. Learning: BC, DAgger, BPTT, and why each failure happened
8. The loss functions, term by term
9. Domain randomization: the theory of why it works
10. Sim2real: aliasing, latency, and parity
11. Lyapunov functions, gently
12. The lessons, restated as principles

---

## 1. Frames, rotations, and quaternions

### 1.1 The two frames

**World frame (the "gate frame")**: origin at the gate center; **+y points
through the gate toward the side we fly from**; **z points DOWN**; x right
(when facing the gate). Units: meters (the splat's internal "scene units"
satisfy 1 unit = 0.85 m and appear only at the render boundary).

Why z **down**? Convention from aerospace (NED: North-East-Down). Two
payoffs: gravity is simply **g = [0, 0, +9.81] m/s²** (a positive number in
the z slot — falling means z increases), and it matches PX4/MAVLink
conventions so nothing flips sign at the deployment boundary.

**Body frame (FRD: Front-Right-Down)**: attached to the drone; x out the
nose, y out the right side, z down through the belly. The propellers push
along **−z_body** ("up" in body terms).

### 1.2 Rotations: what a quaternion is

A rotation in 3D can be written as a **unit quaternion**
`q = (w, x, y, z)` with `w² + x² + y² + z² = 1`. Interpretation: to rotate
by angle θ (theta) about a unit axis `n = (n₁,n₂,n₃)`:

```
q = ( cos(θ/2),  n₁·sin(θ/2),  n₂·sin(θ/2),  n₃·sin(θ/2) )
      ^ scalar    ^--------- vector part ---------^
```

Our `q` maps **body coordinates to world coordinates**: if `v_b` is a
vector expressed in the body frame, then `v_w = R(q)·v_b` is the same
physical vector in world coordinates, where `R(q)` is the 3×3 rotation
matrix built from q (the explicit formula is in `expert.py:90-96`).

**Composition** is the Hamilton product (code: `quat_mul`):
rotating by q₁ then q₂ is the quaternion `q₂ ⊗ q₁`. It is NOT commutative —
order matters, exactly like rotation matrices.

**Why quaternions and not Euler angles or matrices?** Matrices: 9 numbers
with 6 constraints — drift off the manifold under integration. Euler
angles: gimbal lock (at pitch = ±90° two axes collapse). Quaternions: 4
numbers, 1 constraint (renormalize each step — one line), no singularities,
cheap composition. We use Euler **ZYX (yaw ψ, pitch θ, roll φ)** only at
interfaces: the renderer's pose convention and human-readable logs.
`quat_from_euler_zyx` builds q = q_z(ψ) ⊗ q_y(θ) ⊗ q_x(φ) — yaw first
about world-z, then pitch about the new y, then roll about the newest x
("intrinsic" rotations). Verified against scipy to 5e-7.

### 1.3 Integrating a rotation: the exponential map

If the body rotates with **angular velocity ω = (ωx, ωy, ωz) rad/s**
(expressed in the body frame) for a short time Δt, the attitude update is

```
q ← q ⊗ exp_map(ω·Δt),   exp_map(u) = ( cos(‖u‖/2),  (u/‖u‖)·sin(‖u‖/2) )
```

where `‖u‖` is the vector's length. Read it as: "rotate about the axis
u/‖u‖ by angle ‖u‖." For tiny ‖u‖ the sine ratio → 1/2·u (the code uses a
Taylor guard `1 − θ²/6` to avoid 0/0 — `dynamics.py::quat_exp_map`). We
renormalize q after every step so floating-point drift never accumulates.

---

## 2. Quadrotor dynamics and the CTBR abstraction

### 2.1 The state

```
p ∈ R³   position (m, world/gate frame)
v ∈ R³   velocity (m/s, world frame)
q ∈ S³   attitude quaternion (body→world)
ω ∈ R³   body angular rates (rad/s, body frame)
```

plus actuator internals (a filtered thrust value and a queue of delayed
commands — below).

### 2.2 The action: CTBR

`a = [c, ωx^cmd, ωy^cmd, ωz^cmd]`

- **c** = commanded **collective thrust, mass-normalized**, in m/s². That
  means: "the acceleration the props would produce if the drone were
  weightless." Hover requires the thrust to cancel gravity ⇒ **c ≈ g =
  9.81 m/s²** at hover. Using acceleration units (not Newtons or a 0–1
  throttle) keeps the policy portable: the mapping to PX4's normalized
  throttle (`thrust01 = 0.34·c/9.81` — 0.34 is the measured hover fraction)
  is a one-line deployment calibration.
- **ω^cmd** = desired body rates. PX4's onboard rate controller (an 800 Hz
  PID using only the gyro) makes the actual rates track these.

**Why CTBR?** It is the *lowest-level* command that (a) needs **no state
estimate** on the autopilot (the rate loop only needs the gyro) and (b)
stays platform-portable (any PX4/Betaflight vehicle takes it). Everything
above it (velocity, position modes) requires the autopilot to know its own
velocity — which is exactly the mocap/VIO dependency this project removes.
This is also the standard conclusion of the action-space literature
(Kaufmann et al. benchmark).

### 2.3 The closed-loop actuator abstraction (first-order lags)

We do NOT simulate motors and mixers. We simulate what the *policy
experiences*: the closed rate loop behaves approximately like a
**first-order system** — the actual rate ω exponentially approaches the
commanded rate:

```
ω ← ω + α_ω · (ω^cmd − ω),      α_ω = 1 − e^(−Δt/τ_ω)
```

**Symbols**: Δt = simulation substep (0.005 s); **τ_ω (tau)** = the *time
constant* — the time for the response to cover 63% of the remaining gap.
After 3τ you're at 95%. Our τ_ω is domain-randomized 15–60 ms, centered on
the measured Starling 2 rate-loop bandwidth (~10–20 Hz; note **bandwidth f
and time constant relate as τ ≈ 1/(2πf)**). Thrust tracks its command the
same way with its own τ_c (10–45 ms — motor spin-up).

Why the exponential form for α? It's the exact discretization of the
continuous first-order ODE `τ·ẋ = (u − x)`: over a step Δt the solution
decays by e^(−Δt/τ). Using `1 − e^(−Δt/τ)` instead of the naive `Δt/τ`
keeps the model exact for any step size.

### 2.4 Transport delay

Cameras, inference, and radio take time. We model a **dead time**: the
plant applies the command issued L control steps ago (a FIFO queue,
`dynamics.py`). L is randomized 1–5 steps = 25–125 ms. Delay is *different*
from lag: lag smooths, delay shifts. Both destabilize feedback loops —
that's why the latency budget shows up in every robustness table.

### 2.5 The translational dynamics

```
a_world = R(q)·[0, 0, −T]  +  [0, 0, +g]  −  k_d·v
          ^ thrust, rotated   ^ gravity      ^ linear drag
v ← v + a_world·Δt ;   p ← p + v·Δt      (semi-implicit Euler)
```

**Symbols**: T = the *produced* thrust (the lag-filtered, gain-scaled
version of c); the thrust vector is [0,0,−T] in the body frame because
props push along −z_body; `k_d` ∈ [0, 0.3] s⁻¹ is a linear rotor-drag
coefficient (air resistance roughly proportional to speed at these speeds).
**Semi-implicit Euler** means: update v first, then update p with the *new*
v — this tiny ordering choice is much more energy-stable than naive Euler.

**The hover fixed point** (sanity anchor for everything): level attitude
(R = I), T = g, v = 0 ⇒ a_world = 0. Our test suite literally asserts a
drone commanded this way stays put for 4 s to machine precision.

**Thrust-map error**: the real map from command to produced thrust is
uncertain (battery sag, prop wear). We model `T = gain·c` with gain
randomized ±15%. To hover, the policy must discover the *actual* gain —
"command 9.81" fails by up to ±1.5 m/s². This single line of DR is what
forces the policy to have integral action (§4.3, §7.5).

---

## 3. The IMU: what an accelerometer actually measures

An accelerometer does **not** measure acceleration. It measures **specific
force**: the non-gravitational force per unit mass acting on it,

```
f_body = Rᵀ(q) · (a_world − g_vec)
```

(Rᵀ, the transpose, rotates world→body.) Intuition: in free fall,
a_world = g_vec, so f = 0 — the accelerometer reads zero even though you
accelerate at g. **At hover**, a_world = 0, so f_body = −Rᵀ·g_vec =
[0, 0, −9.81] in FRD: the sensor "feels" the props pushing it up.

The **gyro** measures ω directly. Both sensors carry a **bias** (a
per-power-cycle constant offset — randomized per episode: gyro σ = 0.02
rad/s, accel σ = 0.2 m/s²) and white noise per read. Bias is the deep
problem: integrating a biased accel gives velocity drift growing linearly
in time — this is *why* IMU-only velocity estimation fails and why vision
must carry the position/velocity truth.

### 3.1 The complementary filter (the onboard tilt estimate)

Tilt (roll φ, pitch θ) is observable from the IMU alone: gravity's
direction in body frame tells you which way "down" is. Two noisy sources:

- **Gyro path**: integrate φ̇ ≈ ωx — smooth, drifts (bias integrates).
- **Accel path**: at near-hover, from f ≈ −Rᵀ g_vec, algebra gives
  `φ_acc = atan2(−f_y, −f_z)`, `θ_acc = atan2(f_x, √(f_y²+f_z²))` —
  drift-free, but noisy and WRONG during accelerations (f then contains
  motion, not just gravity).

The **complementary filter** blends them: propagate with the gyro, then nudge
toward the accel answer with a small gain k (ours 0.02), *only when*
`‖f‖ ≈ g` (i.e., we're probably not maneuvering hard):

```
φ ← φ + ωx·Δt ;   if 7 < ‖f‖ < 12.6:   φ ← φ + k·(φ_acc − φ)
```

"Complementary" because the gyro supplies the high-frequency content and
the accel the low-frequency (DC) truth — their strengths are complementary.
This IS (a simplified form of) what PX4's attitude estimator does; we
implemented it in the helper so the tilt input needs nothing but the IMU.

---

## 4. Classical control: the geometric expert

The expert (`expert.py`) is a textbook cascade. Understanding it teaches
most of the control theory in the project.

### 4.1 The cascade

```
position error ──PID──► desired acceleration a_des (world)
a_des ──geometry──► desired thrust magnitude T and desired attitude R_des
attitude error (R vs R_des) ──P──► body-rate commands ω^cmd
```

**Step 1 — outer loop (PID)**, with e_p = p − p_target:

```
a_des = −k_p·e_p − k_d·v − k_i·∫e_p dt
```

k_p pulls toward the target (a spring), k_d damps (friction), and the
integral term k_i·∫e accumulates persistent error (see §4.3).

**Step 2 — from acceleration to attitude.** We need the *thrust vector* to
produce a_des while also cancelling gravity. From §2.5 with drag ignored:

```
T·ẑ_b^des = g_vec − a_des        (ẑ_b = body z-axis in world coords)
⇒ T = ‖g_vec − a_des‖,  ẑ_b^des = (g_vec − a_des)/T
```

Check the hover case: a_des = 0 ⇒ ẑ_b^des = [0,0,1] (level, since z is
down) and T = g. To *move forward*, a_des points forward ⇒ ẑ_b^des tilts
backward-…-wait, forward: the drone leans into the direction of travel.
That's the deep fact of quadrotors: **lateral acceleration is produced by
tilting** — position control is secretly attitude control.

The desired full attitude R_des is built from ẑ_b^des plus the desired
yaw ψ*: choose x_c = [cosψ*, sinψ*, 0] (where the nose should point,
projected level), then `ŷ = ẑ×x_c / ‖·‖`, `x̂ = ŷ×ẑ` — a right-handed
frame with the prescribed z and (approximately) the prescribed yaw. (× is
the vector cross product.)

**Step 3 — attitude error to rates.** With R the current and R_des the
desired rotation, the error rotation is Rᵀ·R_des. For small errors, the
**vee map** extracts its axis-angle vector:

```
e_R = ½·(RᵀR_des − R_desᵀR)^∨      (the ∨ takes the 3 independent
                                     entries of a skew-symmetric matrix)
ω^cmd = k_att · e_R
```

This is the Lee/Mellinger geometric attitude P-controller: command rates
proportional to how far, and about which axis, the attitude must rotate.

### 4.2 Why the gains were softened, then re-hardened

A P(D) loop with total loop delay τ_total goes unstable roughly when its
gain demands a response faster than the delay allows (phase margin — the
correction arrives too late and pushes the wrong way). With the original
pessimistic DR (τ_ω up to 100 ms + delay up to 100 ms), katt = 8 produced
limit cycles in 4% of plants; softening to 5.5 fixed it. When the
*measured* rate-loop numbers arrived (τ_ω ≤ 60 ms), 100% held. Lesson:
**gains and delays trade off through phase margin; know your delays.**

### 4.3 Integral action — the most important 10 lines in the project

Suppose a constant disturbance d (e.g., the thrust-gain error: commanded
hover produces g·(gain−1) ≈ ±1.5 m/s² of unwanted vertical acceleration).
A P-controller settles where spring force balances disturbance:

```
steady state:  k_p·e_ss = d   ⇒   e_ss = d/k_p
```

With d = 1.5 and k_p = 4 that's **37 cm of permanent error** — precisely
what we measured (27% success). No finite k_p removes it. The integral term
does: as long as e ≠ 0, ∫e grows, adding ever more correction until e = 0
exactly. (Anti-windup — clamping the integral — prevents the accumulated
term from causing huge overshoot after saturations.)

The learned policy has no explicit integrator — but the **GRU can
implement one** (its state can accumulate). It only *learns* to if training
shows it the integrated consequence of a bias — which 0.8-second training
windows never did. That single observation begat the chained-window
trainer (§7.5).

---

## 5. Information & observability: what a camera can and cannot tell you

### 5.1 Position: yes, richly

At the hover point (~1.3 m from the gate), the gate nearly fills the
128-px-wide image. The scale factor is the focal length in pixels,
**fx ≈ 63 px** (at 128-wide resolution). A lateral offset Δx at distance Z
shifts the gate in the image by approximately

```
Δu ≈ fx · Δx / Z    pixels
```

For Δx = 10 cm at Z = 1.3 m: Δu ≈ 5 px — plainly visible. Apparent gate
*size* similarly codes distance. So single-image position information is
plentiful; that part transferred directly from the previous phase.

### 5.2 Velocity: not from one frame — and the sub-pixel trap

Velocity only appears as *motion between frames*:

```
Δu ≈ fx · v · Δt_pair / Z
```

With consecutive frames (Δt_pair = 25 ms at 40 Hz), v = 0.3 m/s, Z = 1.3 m:
Δu ≈ **0.36 px**. Sub-pixel. A stride-2 conv trunk effectively cannot see
it (its first layer already halves resolution). This is why run 8 stalled
— the "velocity input" we'd added carried almost no signal. Setting
Δt_pair = 150 ms (pair the current frame with the one 6 steps back) makes
Δu ≈ 2–4 px — learnable. **Do this arithmetic for every visual signal you
ever feed a CNN.**

### 5.3 Why we PROVED it before building it: the privileged probe

Seven training-side fixes had failed identically at a 0.5–0.7 m orbit. The
hypothesis "the policy lacks velocity information" was tested by injecting
*true velocity* into the observation (possible only in sim — that's what
"privileged" means) and retraining briefly: the plateau collapsed
0.55 → 0.113 m. That isolates the cause with near-certainty: the failures
were informational, not optimization. THEN we engineered the deployable
carrier of the same information (the 150 ms pair + an auxiliary head that
*forces* the features to encode velocity by predicting it from them, train
time only). **Pattern: when many orthogonal fixes fail the same way,
suspect the observation; prove it with a privileged input; then build the
sensor-legal version.**

### 5.4 Recurrence as estimation

A GRU carrying state h_t across steps can implement (approximately) any
recursive estimator — in particular a Luenberger-observer/Kalman-like
velocity filter (predict with accel, correct with visual motion) and a
disturbance integrator (§4.3). That's the theoretical reason the policy is
recurrent; the practical confirmations were the probe and the expert's
PD→PID experiment. See 10_recurrence_notes.md for the literature.

---

## 6. The network, block by block

Input tensors (exact contract in 09_network_io.md): image (2, 96, 128) =
[current, 150 ms-old] grayscale; 12-D proprio vector; 96-D GRU state.

### 6.1 The mean-subtraction trick

Each frame I is expanded to the pair `[I, I − mean(I)]` (mean over all
pixels of that frame). Why: if global illumination scales/shifts all
pixels (camera auto-exposure, lighting change), `I − mean(I)` is invariant
to the shift while `I` retains absolute level; giving the first conv BOTH
lets it *choose* invariant features where useful. It's linear ⇒ costs
nothing, stays verification-friendly, and survived a real sim2real
crossing in the previous phase. (Implemented as an average-pool rather than
`.mean()` purely because of a TFLite converter bug — a good example of
implementation vs math.)

### 6.2 The conv trunk and the pooled readouts

Four stride-2 conv+BN+ReLU blocks: 96×128 → 48×64 → 24×32 → 12×16 → a
**6×8 feature map** with 64 channels. Each cell of that map sees a large
receptive field (~66 px square). Then three *pooled readouts*, each chosen
by what its averaging destroys — the signature idea inherited from the
previous phase:

- **Global average (→64 numbers)**: averages away all spatial layout;
  what survives is "how much gate-ness is in view" = apparent size =
  **distance** cue. Position-invariant ⇒ transfers best across sim2real.
- **Lateral readout (1×1 conv to 8 ch, then pool rows away → 8×4 = 32)**:
  keeps only left↔right structure (4 column bands) ⇒ codes **horizontal
  offset/yaw**.
- **Vertical readout (pool columns away → 8×3 = 24)**: keeps only up↕down
  structure ⇒ codes **height** (its absence made the old net provably
  blind to altitude — a real bug they fixed this way).

The 120-D image code joins a 32-D embedding of the proprio vector, feeds a
**GRUCell(96)**, and the action head reads **[h, current features]**
concatenated — the *skip connection*. The skip exists because we measured
the rate channels under-responding when everything was routed through the
GRU (memory should serve estimation, not gate all fresh evidence).

### 6.3 The GRU equations (so you can explain the cell)

With input x_t (the 96+32-D fused features) and previous state h_{t−1}:

```
r_t = σ(W_r x_t + U_r h_{t−1})     reset gate    (σ = logistic sigmoid)
z_t = σ(W_z x_t + U_z h_{t−1})     update gate
ĥ_t = tanh(W_h x_t + U_h (r_t ⊙ h_{t−1}))    candidate  (⊙ = elementwise)
h_t = (1 − z_t) ⊙ ĥ_t + z_t ⊙ h_{t−1}
```

Read: z decides how much old state to KEEP (z→1: pure memory — that's how
an integrator is implemented), r decides how much old state to consult when
forming the new candidate. All W, U are learned matrices. Everything is
matmul/sigmoid/tanh — old, boring ops that export and verify cleanly.

### 6.4 Bounded outputs: clamp_relu

```
clamp_relu(x, L) = relu(x + L) − relu(x − L) − L      ∈ [−L, +L]
```

Check the three regimes: x < −L gives 0 − 0 − L = −L; |x| ≤ L gives
(x+L) − 0 − L = x; x > L gives (x+L) − (x−L) − L = L. An exact clamp built
from ReLUs only — no min/max ops — which keeps the network inside the
α,β-CROWN-verifiable op set. The thrust head is centered at hover:
`c = 9.81 + clamp_relu(·)·0.9·9.81`, so a zero-initialized head outputs
exactly hover (a free warm start and a safe failure mode).

---

## 7. Learning: the algorithms and why each failure happened

### 7.1 Behavior cloning (BC) and compounding error

BC = supervised learning: minimize the difference between the policy's
action and the expert's action *on states the expert visited*. The trap:
when the learned policy flies, its small errors take it to states slightly
OFF the expert's distribution, where it was never trained; errors grow,
excursions grow — a compounding spiral. Theory says the closed-loop cost
of an ε-accurate imitator can grow like **T²·ε** over horizon T (Ross &
Bagnell) rather than T·ε. We saw the dramatic version: training loss
→ 0.0000, closed loop 98% crash.

**DAgger** (Dataset Aggregation) fixes the distribution: let the *student*
fly, ask the expert "what would you do HERE," add those labels, retrain.
Each round trains exactly where the student actually goes. Two rounds took
us 35 m → 0.83 m median. Note the subtlety we handled: our expert is
stateful (its integrator), so its labels along a student trajectory use
the integral accumulated along THAT trajectory.

### 7.2 Asymmetric learning (privileged teacher, sensor-limited student)

The expert/losses/aux heads may read true sim state; the policy may not.
This asymmetry is the industry-standard trick (teacher-student distillation,
asymmetric actor-critic — Geles' ablation shows the symmetric version gets
literally 0%). All privileged paths are severed at export; 09_network_io
carries the audit.

### 7.3 BPTT through the plant (and NOT the renderer)

Because our dynamics are differentiable torch ops, we can compute the exact
gradient of a trajectory loss with respect to policy weights **through the
physics**: action → next state → next loss, chained over the window
(BackPropagation Through Time). The images are *detached* (no gradient
through the renderer): D.Va measured renderer-gradient norms >1e15 — splat
rasterization is far too nonsmooth. So the gradient signal is: "had you
tilted a bit more at t=3, your position at t=12 would have been better" —
via physics, with vision treated as a fixed observation channel. Cheap,
stable, and hardware-validated by GRaD-Nav and (feature-version) Heeg.

### 7.4 Truncation myopia and terminal costs

You can't backprop through minutes (memory, and gradients through long
chaotic rollouts explode — the SHAC literature). So you truncate to windows
of H steps. New problem: the window can't see beyond its end, so behavior
that looks great inside the window but ends badly (arriving *fast*) is
rewarded — we watched it sprint and diverge. Two standard cures: a learned
**terminal value function** (SHAC's critic — estimates the future beyond
the cutoff), or a hand-built **terminal cost** on the window's final state.
We used the second (position AND velocity at window end — "end close AND
slow"), holding the critic in reserve all project; it was never needed.

### 7.5 Episode-chained windows (our main structural invention here)

Even with terminal costs, a *slow* drift (cm/s from a small thrust bias)
costs almost nothing inside 0.8 s — but compounds to meters over 8 s.
Losses literally cannot see it. Fix: run windows **consecutively along one
episode** — reset once, then 13 windows of 32 steps, carrying the physical
state AND the GRU state across windows (detached — gradients stay
window-local, so BPTT stability is preserved). Later windows then *start*
from drifted states and bill the accumulated error; the GRU experiences
multi-second histories (so integral action is learnable); and the training
temporal distribution matches evaluation by construction (episodes start
from rest, like deployment). One mechanism, three fixes.

### 7.6 Optimization mechanics worth knowing

- **Gradient clipping**: rescale the gradient if its norm exceeds a cap —
  protects against occasional exploding windows. Trap we hit: if the
  *typical* norm sits above the cap, every step is truncated and learning
  silently crawls (loss flat). Check your norms against your cap.
- **Ramping new losses**: adding a new pressure at full weight to a
  converged policy at small lr = shock and regression (measured twice).
  Ramp weights over ~6 epochs.
- **Cosine decay + best-checkpoint saving**: the last epochs at small lr do
  the precision work; save by metric, never "latest".
- **Huber loss** (used everywhere): quadratic for |e| ≤ δ, linear beyond —
  `H(e) = ½e² if |e|≤δ else δ(|e|−½δ)`. Quadratic near zero (smooth,
  well-conditioned), linear far away (outliers and transients don't
  dominate the gradient like a pure L2 would).

### 7.7 The auxiliary velocity head

A tiny linear layer, train-time only, reads the policy's internal features
and is trained to predict true body velocity. Its gradient flows INTO the
trunk/GRU, forcing them to *represent* velocity (representation shaping).
At export it's deleted. Why it helps: the main task's gradient signal for
"encode velocity" is indirect and weak; the aux makes it direct. (GaussGym
does the same with a reconstruction loss.)

---

## 8. The loss functions, term by term (`train_bptt.py::window_loss`)

For each step t in a window, with e = p − p*, en = ‖e‖, v the velocity,
and wt a weight growing linearly with t (later steps in a window matter
more; a second multiplier grows along the chain — steady-state windows
matter more):

1. `Huber(e)` — get to the target.
2. `5·exp(−en/0.3)·en²` — **near-ring precision**: Huber's pull weakens
   linearly as en shrinks, and eventually loses to the action
   regularizers + noise floor (the measured 0.5 m "orbit"). This term's
   exponential gate turns ON near the target and its en² keeps the
   pressure. (Direct descendant of the old phase's "if V small, push V²→0".)
3. `RING_W·sigmoid((en − 0.15)/0.04)` — **time outside the ring**: a
   smooth 0/1 indicator of "not yet at the target," summed over time ⇒
   directly penalizes SLOWNESS (the differentiable version of the success
   criterion). Introduced with a ramp (§7.6).
4. Velocity: `near·‖v‖²` (damp when close — near = exp(−en)) plus
   `relu(‖v‖ − cap)²` with **cap = 1.2 + 0.6·min(en, 2)** — the
   distance-scaled speed allowance: fly fast far away (2.4 m/s), arrive
   slow (1.2). The earlier flat cap was taxing healthy transit speed.
5. Attitude: `1 − cos(ψ − ψ*)` (yaw error — the cosine form is smooth and
   wrap-safe: 0 at aligned, max at 180°) + a tilt penalty beyond 25°.
6. Action regularizers: `‖a_normalized‖²` small (don't waste control
   authority) and `‖a_t − a_{t−1}‖²` (jerk — smoothness; Geles/Swift use
   the same pair).
7. **Terminal cost**: `2·Huber(e_T) + 1.5·Huber(v_T)` at the window's last
   state — §7.4.
8. **Expert anchor**: `Huber((a − a_expert)/scale)` with per-channel
   weights (rates ×2 — where we measured the deficit): dense,
   well-conditioned supervision that bypasses the plant entirely;
   the "bootstrap RL with IL" pattern.
9. (Transit tasks) **Perception term**: with β the bearing angle from the
   drone to the current gate (β = atan2(Δy, Δx) in the plane) and ψ the
   yaw, penalize `relu(|β − ψ| − 0.35)²` — keep the gate within ~20° of
   the optical axis while it's still ahead. Vision-in-the-loop policies
   must be *taught* to protect their own observability (Swift and Geles
   both learned this; their rewards contain the same term).

---

## 9. Domain randomization: the theory of why it works

Train on a *distribution* over plants (mass margin, lags, delays, gains,
sensor biases, camera intrinsics, photometrics). Two distinct mechanisms:

1. **Robustness**: the optimal policy for a distribution hedges — it must
   work acceptably on ALL sampled plants, so the real plant (if inside the
   distribution) is just another sample. This is a min-average (sometimes
   min-worst) game against nature.
2. **Adaptation (the subtler one)**: a *recurrent* policy can do better
   than hedging — it can IDENTIFY the current plant from the transient
   (how did the drone respond to my last commands?) and adapt. DR +
   recurrence ⇒ the training pressure to become an online system
   identifier. Our thrust-gain DR forcing learned integral action is the
   cleanest example.

Discipline that made it work here: center every range on a measured value
(01 §4 tables), widen for ignorance, and **never change the eval
distribution casually** — train-time-only widenings (v14's delay DR) keep
tables comparable; protocol changes are batched and re-baselined (06 §C2).
Also: DR is for *residual* uncertainty. Anything you can measure or
byte-match (camera model, preprocessing), you match — DR-ing a knowable is
paying interest on a debt you could just repay.

## 10. Sim2real: aliasing, latency, parity

- **Aliasing/AA**: rendering directly at 128×96 samples the scene too
  sparsely — thin structures shimmer (violating Nyquist: signal content
  above half the sampling rate folds into artifacts). The deployed camera
  path *low-passes* (its downscale averages many sensor pixels). So the
  sim renders at 2× and average-pools, and the onboard resize is specified
  as INTER_AREA (a box filter) — both ends low-pass alike. The general
  principle: **match the whole measurement chain, not just the geometry.**
- **Latency budgeting**: measured glass→command ≈ 20–45 ms (camera
  dependent) + inference (~1–3 ms) + transport — trained delay DR covers
  25–125 ms, so the real pipeline sits inside the training distribution
  with margin. Faster flight eats margin (v13's lesson): speed and latency
  robustness share one budget.
- **Export parity**: the shipped TFLite is fp16 and structurally rewritten
  by converters; the ONLY acceptable proof of equivalence is closed-loop —
  1000 steps feeding the hidden state back, max action deviation ~5e-3
  m/s² (≈0.06% of the thrust span). Every export runs it. Four converter
  landmines are documented in MASTER §5.15.

## 11. Lyapunov functions, gently

A **Lyapunov function** V(x) is an "energy-like" scalar over the state:
V > 0 away from the goal, V = 0 at it, and — the crucial property — V
strictly DECREASES along the system's trajectories (V̇ < 0). If such a
function exists, trajectories must slide down its landscape into the goal:
**stability, proved without solving the dynamics.** A *control* Lyapunov
function (CLF) is the same idea when a controller is choosing actions to
make V decrease.

The previous phase *learned* a V(e_p, Δyaw) alongside the controller
(decrease enforced as a training loss) — giving convergence *plots* and a
monitor, though not a formal proof (the structural positivity condition was
never enforced). This phase dropped it because (a) evaluating V at runtime
needs a pose estimate — abolished by the project's premise; and (b) the
state grew (a drone AT the target moving 2 m/s must have V > 0, so V needs
v, tilt, rates). The full option analysis and the two pieces worth doing
(offline analysis-V, observation-space OOD monitor) are in 08_lyapunov.md.

## 12. The lessons, restated as principles

1. **Ask where each bit of information comes from** before designing
   anything (the §1-of-MASTER table). Most failures were information
   failures wearing optimization costumes.
2. **Do the pixel arithmetic** (§5.2) for every visual signal.
3. **Prove bottlenecks with privileged probes** before engineering fixes.
4. **Integral action must be learnable**: if a bias matters over seconds,
   training must exhibit seconds (chained windows).
5. **Truncated optimization is myopic**: bill the end state.
6. **One lever at a time; ramp new pressures; graft to exact equivalence;
   protect best checkpoints; decide only on the fixed-seed ablation
   matrix.** (Each clause is one specific scar — MASTER §5.)
7. **Match measured reality, randomize residuals, and keep eval
   distributions sacred.**
8. **Closed-loop or it didn't happen** — training loss and even open-loop
   parity prove nothing about flying.
9. Sims flatter you (exact IMUs, clean masks — see the neighbors' own
   documented gaps). Fly early; let real logs drive the next sim.

*If you can re-derive §4.3 (why P-control leaves e = d/k_p and why the
integrator kills it), §5.2 (pixels per m/s), and §7.5 (why chained windows
teach integration), you understand this project well enough to defend
every decision in it.*
