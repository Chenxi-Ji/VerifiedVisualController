"""Three-gate smooth turn: the real gate at the origin as the MIDDLE gate +
two scene_edit duplicates on a circular arc curving toward +x (heading
change TURN_DEG per gate, chord SPACING between centers, gates tangent to
the arc). Gate 1 (passed first) is a duplicate upstream on the +y approach
side at yaw -TURN; gate 3 is a duplicate downstream at yaw +TURN.

Anchoring the ORIGINAL gate in the middle (v2; it used to be first) centers
the whole track on the arena: the old arc ran to gate 3 at (2.42, -2.88) m
with the exit at (3.20, -3.02) — pressed against the +x/-y safety net (mat
extent ~|x|<2.4, y in [-3.3, +3.5]; probed visually in the splat). Now the
extremes are wp1 (1.13, +2.41) and exit (1.20, -2.49), all >~1 m inside the
net. The start box is re-expressed in gate 1's approach frame (env_multigate
START_*), shrunk to x +-1.0 m / runway 0.45-1.25 m so its rotated corners
stay on the mat in well-captured +y splat space.

Arc choice (04-design constraint set): consecutive chords sit only TURN/2 =
20 deg off the transit heading, so the NEXT gate is ~20 deg off the optical
axis when passing the previous one — deep inside the 146-deg hires fisheye
FOV. Turning toward +x keeps the track shallow in -y (exit y ~-2.5 m): the
arena was captured from +y, so -y depth = view extrapolation (07 doc).
Verified frames: spike_out/multigate/20_three_*.png.

Gate i sits at yaw phi_i = (i - 1)*TURN about z; drone transit yaw
-pi/2 + phi_i.
"""

from __future__ import annotations

import math

from env_multigate import MultiGateEnv, oracle_main
from env_multigate import verification_poses as _vp

TURN_DEG = 40.0     # heading change per gate (spec 30-40)
SPACING = 2.0       # m chord between consecutive gate centers (spec 2.0-2.5)


def _arc_gates(n=3, mid=1, turn=math.radians(TURN_DEG), chord=SPACING):
    """Arc through n gates with gate `mid` (the real one) at the origin,
    yaw 0; gate i at yaw (i-mid)*turn; chord i->i+1 along the mean of the
    two headings (gates tangent to the arc)."""
    x = y = 0.0
    pos = {mid: (0.0, 0.0)}
    for i in range(mid + 1, n):                    # walk downstream
        a = (i - 0.5 - mid) * turn
        x += chord * math.sin(a)
        y -= chord * math.cos(a)
        pos[i] = (x, y)
    x = y = 0.0
    for i in range(mid - 1, -1, -1):               # walk upstream
        a = (i + 0.5 - mid) * turn
        x -= chord * math.sin(a)
        y += chord * math.cos(a)
        pos[i] = (x, y)
    return [((pos[i][0], pos[i][1], 0.0), (i - mid) * turn) for i in range(n)]


class ThreeGateTurnEnv(MultiGateEnv):
    GATES = _arc_gates()
    REAL_GATE = 1     # the original splat gate is the MIDDLE gate
    EVAL_T = 880      # 22 s: at 20 s the slowest DR plants are still braking
                      # at the exit (oracle 87.5% -> 100%, all-clean either way)
    # rotated start box (gate-1 approach frame): corners stay on the mat
    START_X = 1.0
    START_Y = (0.45, 1.25)


def _dup_poses():
    import scene_edit as se
    return [se.gate_pose(c, y) for i, (c, y) in enumerate(ThreeGateTurnEnv.GATES)
            if i != ThreeGateTurnEnv.REAL_GATE]


DUP_GATE_POSES = _dup_poses()


def verification_poses():
    return _vp(ThreeGateTurnEnv)


if __name__ == "__main__":
    oracle_main(ThreeGateTurnEnv, T_s=22.0)
