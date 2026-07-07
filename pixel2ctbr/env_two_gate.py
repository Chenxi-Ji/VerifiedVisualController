"""Two-gate straight track: the real gate at the origin + one scene_edit
duplicate 2.2 m further along the flight direction (gate-frame y = -2.2 m,
same orientation). Approach from the +y start box -> through gate 1 ->
through gate 2 -> brake to hover EXIT_Y = 0.8 m past gate 2 (y = -3.0 m).

Splat-validity note: past gate 1 the camera faces -y content captured only
from +y viewpoints; the duplicated gate is the visual anchor there, the
background is view-extrapolated and degrades with depth (07 doc). Verified
frames: spike_out/multigate/10_two_*.png.
"""

from __future__ import annotations

from env_multigate import MultiGateEnv, oracle_main
from env_multigate import verification_poses as _vp

GATE_SPACING = 2.2   # m between the gate planes


class TwoGateStraightEnv(MultiGateEnv):
    GATES = [((0.0, 0.0, 0.0), 0.0),
             ((0.0, -GATE_SPACING, 0.0), 0.0)]
    EVAL_T = 640      # 16 s


def _dup_poses():
    import scene_edit as se
    return [se.gate_pose(c, y) for c, y in TwoGateStraightEnv.GATES[1:]]


DUP_GATE_POSES = _dup_poses()


def verification_poses():
    return _vp(TwoGateStraightEnv)


if __name__ == "__main__":
    oracle_main(TwoGateStraightEnv, T_s=16.0)
