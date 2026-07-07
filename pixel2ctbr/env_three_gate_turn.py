"""Three-gate smooth turn: the real gate at the origin + two scene_edit
duplicates on a circular arc curving toward +x (heading change TURN_DEG per
gate, chord SPACING between centers, gates tangent to the arc).

Arc choice (04-design constraint set): consecutive chords sit only TURN/2 =
20 deg off the transit heading, so the NEXT gate is ~20 deg off the optical
axis when passing the previous one — deep inside the 146-deg hires fisheye
FOV. Turning toward +x keeps the track as shallow in -y as the 30-40 deg
spec allows (y reaches ~-3.0 m vs -5.2 m for a straight 3-gate line): the
arena was captured from +y, so -y depth = view extrapolation (07 doc).
Verified frames: spike_out/multigate/20_three_*.png.

Gate i sits at yaw phi_i = i*TURN about z; drone transit yaw -pi/2 + phi_i.
"""

from __future__ import annotations

import math

from env_multigate import MultiGateEnv, oracle_main
from env_multigate import verification_poses as _vp

TURN_DEG = 40.0     # heading change per gate (spec 30-40)
SPACING = 2.0       # m chord between consecutive gate centers (spec 2.0-2.5)


def _arc_gates(n=3, turn=math.radians(TURN_DEG), chord=SPACING):
    gates, x, y = [((0.0, 0.0, 0.0), 0.0)], 0.0, 0.0
    for i in range(1, n):
        a = (i - 0.5) * turn          # chord direction = mean of headings
        x += chord * math.sin(a)
        y -= chord * math.cos(a)
        gates.append(((x, y, 0.0), i * turn))
    return gates


class ThreeGateTurnEnv(MultiGateEnv):
    GATES = _arc_gates()
    EVAL_T = 880      # 22 s: at 20 s the slowest DR plants are still braking
                      # at the exit (oracle 87.5% -> 100%, all-clean either way)


def _dup_poses():
    import scene_edit as se
    return [se.gate_pose(c, y) for c, y in ThreeGateTurnEnv.GATES[1:]]


DUP_GATE_POSES = _dup_poses()


def verification_poses():
    return _vp(ThreeGateTurnEnv)


if __name__ == "__main__":
    oracle_main(ThreeGateTurnEnv, T_s=22.0)
