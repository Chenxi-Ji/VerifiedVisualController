"""Simulated IMU (and optional tilt estimate) from QuadCTBRDynamics states.

gyro  = w_body + bias_g + noise                      [rad/s]
accel = specific force = R^T (a_world - g) + bias_a + noise   [m/s^2]
        (FRD body, z down: hover reads ~[0, 0, -9.81])

Biases are per-episode constants (redrawn on reset); noise is white per read.
Placeholder magnitudes ⚠ — see docs/pixel2ctbr/05_implementation_log.md.
An optional 'tilt' output models PX4's IMU-only attitude estimate (roll/pitch
observable without external aiding, yaw NOT included): true roll/pitch + slow
random-walk drift + noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from dynamics import G, QuadState, quat_rotate_inv, euler_zyx_from_quat


@dataclass
class ImuParams:
    gyro_bias: torch.Tensor    # (B,3) rad/s, per-episode
    accel_bias: torch.Tensor   # (B,3) m/s^2, per-episode
    gyro_noise: float = 0.005  # ⚠ rad/s per read
    accel_noise: float = 0.10  # ⚠ m/s^2 per read
    tilt_noise: float = 0.005  # ⚠ rad per read
    tilt_drift_rate: float = 0.002  # ⚠ rad/sqrt(s) random walk

    @staticmethod
    def randomized(batch: int, device="cpu", g: torch.Generator | None = None,
                   gyro_bias_sigma: float = 0.02, accel_bias_sigma: float = 0.2) -> "ImuParams":
        rn = lambda s: torch.randn((batch, 3), device=device, generator=g) * s
        return ImuParams(gyro_bias=rn(gyro_bias_sigma), accel_bias=rn(accel_bias_sigma))


class ImuSim:
    """Stateful only for the tilt-drift random walk."""

    def __init__(self, params: ImuParams, dt: float):
        self.p = params
        self.dt = dt
        self.tilt_drift = torch.zeros_like(params.gyro_bias[:, :2])

    def reset(self, params: ImuParams | None = None):
        if params is not None:
            self.p = params
        self.tilt_drift = torch.zeros_like(self.p.gyro_bias[:, :2])

    @torch.no_grad()
    def read(self, s: QuadState, g: torch.Generator | None = None):
        """Returns dict of detached sensor tensors (sensors carry no gradient:
        policies must not backprop through 'reality')."""
        dev = s.p.device
        rnd = lambda shape, sig: torch.randn(shape, device=dev, generator=g) * sig
        gyro = s.w.detach() + self.p.gyro_bias + rnd(s.w.shape, self.p.gyro_noise)
        g_vec = torch.tensor([0.0, 0.0, G], device=dev).expand_as(s.v)
        a_w = s.a_world.detach() if s.a_world is not None else torch.zeros_like(s.v)
        accel = quat_rotate_inv(s.q.detach(), a_w - g_vec) \
            + self.p.accel_bias + rnd(s.v.shape, self.p.accel_noise)
        _, pitch, roll = euler_zyx_from_quat(s.q.detach())
        self.tilt_drift = self.tilt_drift + rnd(self.tilt_drift.shape,
                                                self.p.tilt_drift_rate) * (self.dt ** 0.5)
        tilt = torch.stack((roll, pitch), dim=-1) + self.tilt_drift \
            + rnd(self.tilt_drift.shape, self.p.tilt_noise)
        return {"gyro": gyro, "accel": accel, "tilt": tilt}
