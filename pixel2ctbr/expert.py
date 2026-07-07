"""Privileged-state geometric hover expert for the CTBR plant.

Classic cascaded geometric controller (Lee/Mellinger style, hover regime):
position/velocity PD -> desired specific-force vector -> desired attitude
(tilt + yaw) + collective thrust -> attitude-error P -> body-rate commands.

Uses FULL sim state (p, v, q) — this is a sim-side oracle/teacher, never
deployed. It deliberately does NOT know per-episode actuator params
(thrust_gain, tau, delay): if it can hover the randomized plant anyway, the
control problem is feasible under our DR; a learned policy with the same
interface has an existence proof.

Frame notes (gate frame, z DOWN, FRD body):
  dynamics:  a = thrust_world + g_vec,  thrust_world = -T * z_b^w,  g_vec=[0,0,+G]
  want a -> a_des  =>  T * z_b^w_des = g_vec - a_des
  hover: a_des=0 => z_b_des=[0,0,1] (level), T=G.
"""

from __future__ import annotations

import torch

from dynamics import G, QuadState, quat_rotate_inv


class GeometricHoverExpert:
    def __init__(self, kp=4.2, kd=3.5, ki=2.2, kyaw=2.0, katt=7.0,
                 a_lat_max=4.5, tilt_max=0.45, c_min=2.0, c_max=18.0,
                 integ_cap=2.0, dt=0.025):
        self.kp, self.kd, self.ki, self.kyaw, self.katt = kp, kd, ki, kyaw, katt
        self.a_lat_max = a_lat_max      # m/s^2 cap on commanded plane accel
        self.tilt_max = tilt_max        # rad cap on commanded tilt
        self.c_min, self.c_max = c_min, c_max
        self.integ_cap = integ_cap      # m*s anti-windup clamp
        self.dt = dt
        self.g_vec = None
        self.integ = None               # (B,3) position-error integral

    def reset(self):
        self.integ = None

    def __call__(self, s: QuadState, p_target: torch.Tensor,
                 yaw_target: torch.Tensor) -> torch.Tensor:
        """s: QuadState (B,...); p_target (B,3) m; yaw_target (B,) rad.
        Returns action (B,4) = [c m/s^2, wx, wy, wz rad/s]."""
        B = s.p.shape[0]
        dev = s.p.device
        if self.g_vec is None or self.g_vec.device != dev:
            self.g_vec = torch.tensor([0.0, 0.0, G], device=dev)

        # --- outer loop: desired acceleration (world), saturated.
        # Integral term rejects constant force errors the expert can't know
        # (thrust_gain DR): without it, steady-state err = gain_err*G/kp ~ 0.4 m.
        e_p = s.p - p_target
        if self.integ is None:
            self.integ = torch.zeros_like(e_p)
        self.integ = (self.integ + e_p * self.dt).clamp(-self.integ_cap, self.integ_cap)
        a_des = -self.kp * e_p - self.kd * s.v - self.ki * self.integ
        a_lat = a_des[:, :2]
        lat_norm = a_lat.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        a_des = torch.cat(
            (a_lat * (self.a_lat_max / lat_norm).clamp(max=1.0),
             a_des[:, 2:].clamp(-6.0, 6.0)), dim=-1)

        # --- desired thrust vector & attitude
        f_des = self.g_vec - a_des                    # = T * z_b_des^w
        T = f_des.norm(dim=-1)
        z_des = f_des / T.unsqueeze(-1).clamp_min(1e-9)
        # cap tilt: blend toward [0,0,1] if angle too large
        cos_tilt = z_des[:, 2].clamp(-1, 1)
        tilt = torch.acos(cos_tilt)
        over = (tilt > self.tilt_max)
        if over.any():
            # shrink horizontal component to hit tilt_max exactly
            horiz = z_des[:, :2]
            hn = horiz.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            capped = torch.cat(
                (horiz / hn * torch.sin(torch.full_like(hn, self.tilt_max)),
                 torch.cos(torch.full_like(hn, self.tilt_max))), dim=-1)
            z_des = torch.where(over.unsqueeze(-1), capped, z_des)

        # desired rotation: z_b = z_des, yaw = yaw_target
        x_c = torch.stack((torch.cos(yaw_target), torch.sin(yaw_target),
                           torch.zeros(B, device=dev)), dim=-1)
        y_des = torch.linalg.cross(z_des, x_c, dim=-1)
        y_des = y_des / y_des.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        x_des = torch.linalg.cross(y_des, z_des, dim=-1)
        R_des = torch.stack((x_des, y_des, z_des), dim=-1)  # columns

        # current R from quaternion: columns are body axes in world
        w_, x_, y_, zq = s.q.unbind(-1)
        R = torch.stack((
            torch.stack((1 - 2 * (y_ * y_ + zq * zq), 2 * (x_ * y_ - w_ * zq), 2 * (x_ * zq + w_ * y_)), -1),
            torch.stack((2 * (x_ * y_ + w_ * zq), 1 - 2 * (x_ * x_ + zq * zq), 2 * (y_ * zq - w_ * x_)), -1),
            torch.stack((2 * (x_ * zq - w_ * y_), 2 * (y_ * zq + w_ * x_), 1 - 2 * (x_ * x_ + y_ * y_)), -1),
        ), dim=-2)

        # attitude error (SO(3) log-style, small-angle form): e_R in body frame
        Rt_Rd = R.transpose(-1, -2) @ R_des
        e_R = 0.5 * torch.stack((
            Rt_Rd[:, 2, 1] - Rt_Rd[:, 1, 2],
            Rt_Rd[:, 0, 2] - Rt_Rd[:, 2, 0],
            Rt_Rd[:, 1, 0] - Rt_Rd[:, 0, 1],
        ), dim=-1)
        w_cmd = self.katt * e_R
        w_cmd = torch.cat((w_cmd[:, :2].clamp(-4.0, 4.0),
                           (self.kyaw / self.katt * self.katt * e_R[:, 2:]).clamp(-2.0, 2.0)), dim=-1)

        # collective: project desired force on actual body z (tilt compensation)
        z_b = R[:, :, 2]
        c = (f_des * z_b).sum(-1).clamp(self.c_min, self.c_max)
        return torch.cat((c.unsqueeze(-1), w_cmd), dim=-1)
