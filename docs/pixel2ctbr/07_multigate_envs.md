# 07 — Multi-gate track environments (pixel2ctbr, milestone-2 v1)

*2026-07-07. Extends the single-gate transit task (05 log, env_transit.py) to
multi-gate tracks using splat-scene editing. Everything here is verified by
code in the repo; renders in `pixel2ctbr/spike_out/multigate/` (gitignored —
regenerate with `python pixel2ctbr/spike_multigate.py [crop|two|three|bench]`).*

*Rev 2026-07-07 (evening): crop box z-max 0.60 → 0.82 (duplicates' sliced
bottoms fixed — §2) and the three-gate arc re-anchored with the real gate in
the MIDDLE (old gate 3 pressed against the safety net — §3). Oracles re-run.*

## 0. Summary

The arena splat contains exactly one gate. FalconGym 2.0's editable-gsplat
trick — box-select an object's gaussians in a metric frame, copy, rigidly
transform, concatenate — gives us N-gate tracks from the one measured gate
without recapturing or retraining the twin. `scene_edit.py` implements that
recipe against our scene tuple; `env_multigate.py` generalizes the transit
phase machine to a waypoint list over N gates (the real gate may sit at any
index of the track — `REAL_GATE`); `env_two_gate.py` /
`env_three_gate_turn.py` are the first two tracks. Expert oracle: 99.6–100%
(two-gate, seeds 5/1/11/42) and 100% (three-gate, same four seeds) clean
transits on the randomized plant. Rendering cost at training settings:
−22% (two-gate) / −17% (three-gate) img/s vs pristine (GPU shared with the
v14 run; three-gate is cheaper than two-gate because its upstream gate is
behind the camera for most of the track).

## 1. FalconGym 2.0 survey (both local copies read)

`~/FalconGym-2.0` and `~/FalconGym2.0-mini` (Yan Miao, UIUC): the editing
code is **byte-identical** between them — `edit_gsplat_api.py`, `utils.py`,
`generate_4D_gsplat.py` have equal md5s; `quick_render.py` differs by one
debug print. The mini additionally carries the data (E1_144 scene ckpt,
object pickles, aruco-calibration scripts) and `plane_only_generations.py`
(mass-produces randomized UMX-plane variants — the "editable splat as DR"
pattern). So: code-wise the copies are equivalent; mini is the runnable one.
Neither runs against our scene as-is — their world frame is an
Aruco-derived convention reached through a hardcoded axis-permutation chain
(`tmp` rotation + COLMAP row reorder) baked into every function.

What we reused (the transferable ideas, from `edit_gsplat_api.py`):
- segment by **axis-aligned bbox in a metric frame** (their `world`, our
  gate frame), reusable as a saved mask;
- `duplicate()`: copy the five per-gaussian tensors, rotate means about a
  pivot + rotate quats by the same rotation, translate, `torch.cat` onto
  the scene — opacities/log-scales/colors copy unchanged;
- quats transform by **left-multiplying** the edit rotation after
  conjugating it into ckpt space (`batch_rotm_world_to_quat_ckpt`).

What we did differently (`scene_edit.py`):
- our chain is two ops, not five: `p_ckpt = scale·(transform @ p_units)`
  with `transform = dataparser @ world_frame.json` already gate-centered —
  no permutation legs; verified rigid to 1.2e-7, round-trip 4.8e-7 m;
- pivot at the **gate-frame origin** (= ring center), not the subset
  centroid: `gate_pose(t, yaw)` then means exactly where the env thinks the
  gate is (FalconGym's centroid pivot needs a follow-up recentering);
- quat delta built once per edit (single 3×3 conjugation + scipy), applied
  with the repo's batched Hamilton `quat_mul`;
- identity-duplicate reproduces sources to 4.5e-8 (means) / 0 (quats), and
  a (0,−2.2,0) copy's centroid lands within 1e-3 m — `scene_edit.py`
  __main__ asserts all of this.

## 2. Gate extraction (crop box)

`GATE_BOX = ((−0.72, 0.72), (−0.28, 0.28), (−0.72, 0.82))` gate-frame
meters → **41,405 gaussians (2.57%** of 1.61 M). The original cut at
z-max **+0.60 sliced the duplicates' bottoms** (user-visible): the outer
wire hoop closes at z≈+0.72, so every copy lost the lower hoop arc, the
bottom orange marker and the mounting collar — bare bottom segment with
wire stubs. Fixing it exposed a wrong prior: the floor is **not** at
z≈+1.2 — the mocap calibration (gate_mocap.json) puts the gate center
0.855 m above the floor, and the whole-scene z-histogram peaks at
z≈+0.86 (mat surface; the counts below are reconstruction fuzz). So the
stand is short: collar ≈+0.6→0.7, legs +0.70→0.86, X-feet + tape lines
ON the mat at ≈+0.83–0.88.

z-max iterated visually at 640×480 close-ups (original beside + behind a
copy), with per-increment pixel diffs to attribute what each cut adds:
- **+0.72** restores hoop arc + collar + bottom marker (the big delta);
- **+0.78 → +0.82** adds exactly the lower leg ends (diff pixels sit at
  the leg bottoms) — legs end ~3 cm above the mat, copies look grounded;
- **+0.86** starts dragging **mat/tape-line fragments** (diff pixels lie
  ON the mat around the base) — the floor-patch failure mode, rejected.

z-max = **0.82** chosen. Consequences: duplicates no longer float — they
carry ring + hoop + collar + near-full legs and read as standing on the
mat (feet rods and their tape shadows stay behind — invisible at flight
viewing distances, verified in the two/three-track renders + 1024×768
hires samples). Deletion QA still shows intact background (feet + mat
remain, no holes).

Duplication fidelity (crop + two/three renders): rings crisp at all tested
view angles, checkered texture intact, no smearing added by the copy
itself. Residual physics: sh_degree-0 splats have **baked lighting** — a
rotated duplicate carries the original's illumination (the ±40° gates
are lit as if facing +y). Visually minor at our angles.

## 3. Environments

`env_multigate.MultiGateEnv(HoverEnv)` generalizes `env_transit`:

- **Geometry per gate i**: center c_i, gate yaw φ_i about z; approach
  normal n_i = Rz(φ_i)·ŷ, in-plane axis u_i = Rz(φ_i)·x̂, transit yaw
  ψ_i = −π/2 + φ_i.
- **Phase machine**: waypoint list [wp_0 … wp_{n−1}, exit], wp_i = c_i +
  0.7·n_i, exit = c_last − 0.8·n_last. Phase k targets wp_k with yaw ψ_k;
  the tightened commit gate from env_transit (centered <0.12 m, in-plane
  |v|<0.2, yaw <0.12, near <0.40 — the 85.9%→100% tuning) is evaluated **in
  gate k's own plane coordinates** and latches k+1. `tgt_p`/`tgt_yaw` are
  phase-indexed properties, so the expert, losses and HoverEnv machinery
  retarget transparently (env_transit pattern).
- **Start box**: HoverEnv's box serves a gate at the origin facing +y;
  when the track's FIRST gate is a duplicate elsewhere (`REAL_GATE > 0`)
  reset resamples the pose in gate 1's approach frame (`START_X/Y/Z`
  meters, rotated by φ₁, shifted to c₁, yaw ψ₁ ± 0.6) so every episode
  starts on gate 1's +n side. Tracks whose first gate IS the origin gate
  skip the branch and stay bit-exact with the previous behavior.
- **Crossing detection**: signed plane offset s_k = n_k·(p−c_k) flipping
  +→−; in-plane offset <0.30 m = clean, <0.75 m = frame strike (`FRAME_R`,
  **new vs env_transit**: once gate planes are oblique, an infinite-plane
  crossing far from the ring is not a physical hit — without it the turn
  track would bill phantom "frame strikes" for ordinary flight), farther =
  wide miss (no strike, but success needs every gate clean).
- **Success**: all gates crossed inside the opening, zero strikes, final
  ≤0.15 m of exit, |v|<0.25, no crash. Metrics add per-gate crossing
  fractions and mean phase; interface (transit_expert_rollout /
  transit_metrics) unchanged from GateTransitEnv → trainer drop-in.
- **Rendering**: constructed with `renderer=None`, the env composites its
  duplicates via `scene_edit.multi_gate_scene` into a standard
  SplatRenderer (new optional `scene=` parameter); `renderer=False` stays
  state-only and never touches the checkpoint.

### Two-gate straight (`env_two_gate.py`)
Second gate at (0, −2.2, 0), same orientation; exit at y=−3.0 m.

### Three-gate 40° turn (`env_three_gate_turn.py`)
Gates on a circular arc toward +x: heading change 40°/gate (spec 30–40°),
chord 2.0 m (spec 2.0–2.5), **real gate anchored in the MIDDLE**
(`REAL_GATE = 1`; v2 — it used to be first): c₁=(0.684,+1.879), φ₁=−40°
(duplicate, passed first, on the +y approach side); c₂=(0,0), φ₂=0 (the
real gate); c₃=(0.684,−1.879), φ₃=+40°; exit (1.198,−2.492). The v1
anchoring ran the arc to c₃=(2.416,−2.879) / exit (3.204,−3.018) —
**pressed against the +x/−y safety net**: probe renders put the mat/net
boundary at roughly x∈[−2.3,+2.5], y∈[−3.3,+3.5], and the old pre-g3
hires frame shows the net hugging the gate. Re-anchoring recenters the
whole track (extremes wp₁ (1.13,+2.41) and exit (1.20,−2.49), ≥~1 m
inside the net) with the same turn/chord. Start box: `START_X = ±1.0 m`,
runway `START_Y = (0.45, 1.25) m` in gate-1's frame — rotated corners
(worst ~(0.7, 3.5)) stay on the mat, in well-captured +y splat space.
Chords sit TURN/2 = 20° off the transit heading, so the next gate is
~20° off the optical axis at each pass — deep inside the 146° fisheye
FOV (renders confirm: gates 2 AND 3 visible *through* gate 1's opening
from wp₁). Turning toward +x keeps the track shallow in −y and points
the exit view at the well-captured +x arena side.

## 4. Verification

**Expert oracle** (state-only, B=256, full plant DR, CPU):

| track | horizon | success | all-clean crossings | strikes | exit err med |
|---|---|---|---|---|---|
| two-gate, seed 5 | 16 s | 100% | 100% | 0 | 0.31 cm |
| two-gate, seeds 1/11/42 | 16 s | 99.6/100/99.6% | 100% | 0.39/0/0.39% | 0.33 cm |
| three-gate v2, seeds 5/1/11/42 | 22 s | **100% (all four)** | 100% | 0 | 0.27–0.34 cm |

- Re-run after the crop/REAL_GATE changes: the two-gate seed-5 numbers are
  **bit-identical** to v1 (its code path is untouched and dynamics never
  see the gaussians); three-gate v2 is clean on every seed — the shrunk
  rotated start box has no far-corner pocket, so even the 1/256 overshoot
  strike the two-gate box exhibits doesn't arise.
- EVAL_T = 880 (22 s) retained from v1 (at 20 s the slowest DR plants were
  still braking at the exit: 87.5% → 100% with all-clean either way).
- The two-gate residual (1/256 on two seeds): a far-corner start with
  initial velocity pointing away overshoots wp₀ **through** the gate plane
  at |x| = 0.39 m — inside the frame annulus → billed as a strike; the
  drone recovers and finishes (crossed_all stays 100%). This is an honest
  frame-risk event, the same exposure env_transit had (its plane test was
  infinite!); left as-is, ≥97% gate holds.

**Visual gate** (512×384 color frames along each waypoint chain, 7 + 9
frames, plus 1024×768 hires samples in `spike_out/multigate_hires/`): all
duplicated rings render clean **with complete bottoms** (closed hoop +
collar + legs, §2); the next gate is in-frame at every pass — from wp₁ of
the turn track gates 2 and 3 are both visible through gate 1's opening;
policy-eye check (128×96 gray, DR'd, `30_policy_eye…png`) shows gate 2
inside gate 1's opening. Re-anchoring also fixed the worst v1 frame: the
old `threegate_past_g1` hires (camera just past the origin gate, deep
−y-facing extrapolation) was an unusable smear — its v2 counterpart sits
in well-captured +y space and is crisp. Remaining worst frame: two-gate
**exit** (y=−3.0 m facing −y), unchanged; the turn track's exit
(y=−2.5 m, facing +x−y) is markedly better.

**Throughput** (128×96 ss=2 gray, B=8, chunk=8, GPU shared with the v14
training run): pristine 432 img/s → two-gate 338 (−22%) → three-gate 359
(−17%). Slowdown outruns the gaussian count (+2.6%/+5.1%) because gate
pixels are the expensive ones; the v2 turn track is *cheaper* than
two-gate because its upstream gate is behind the camera for most poses.
Still ≥ the 295 img/s design number (04 §1).

**Trainer**: `--task two_gate|three_gate_turn` wired (env selection,
EVAL_T-aware eval, perception term aimed at the current phase's gate);
window_loss backward verified finite on the rendered three-gate env at
B=8; hover/transit regressions pass (transit oracle 100%,
test_render_bridge ALL PASS).

## 5. Known limitations / open items

1. **−y view extrapolation** is now load-bearing: past gate 1 every
   background pixel is extrapolated from +y captures, degrading with
   depth; duplicated gates are the only crisp anchors there (that's the
   design bet — same one PixelPilot's mask-gap finding supports). Exit
   hover deep in −y (two-gate) has the worst imagery; the turn track was
   shaped to avoid exactly this.
2. **Identical gates**: duplicates are exact clones — nothing visual
   disambiguates gate 2 from gate 1 except context/geometry; the GRU +
   phase-consistent trajectories must carry that. If aliasing bites in
   training, per-duplicate slight color/scale jitter is a one-line edit.
3. **Baked lighting** on rotated duplicates (§2); duplicates stand on
   near-full legs since the z-max 0.82 crop but still cast no shadow and
   carry no feet rods — invisible at flight distances.
4. Phase machine + crossing bookkeeping read privileged state — training
   scaffolding only, as in env_transit; nothing new leaks to the policy.
5. Oracle residual: the 1/256 pre-commit overshoot strike (§4). A
   mid-approach speed cap on the expert would close it; not worth touching
   the verified teacher for now.
6. Training on these tasks not yet run (GPU owned by hover-task v13);
   first candidate recipe: warm-start from the transit-task checkpoint
   once it exists, curriculum two_gate → three_gate_turn.
