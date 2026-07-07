"""Feasibility check: geometric expert must hover the RANDOMIZED plant from the
project start box. If this fails, the DR ranges / latency make the control
problem itself infeasible at this rate — a learned policy would have no chance.
Run: python pixel2ctbr/test_expert.py
"""

import sys

import torch

sys.path.insert(0, "pixel2ctbr")
from dynamics import DynParams, QuadCTBRDynamics, quat_from_euler_zyx, euler_zyx_from_quat
from expert import GeometricHoverExpert

torch.manual_seed(0)

MPU = 0.85  # meters per scene unit
B = 256
dyn = QuadCTBRDynamics(dt_ctrl=0.025, n_sub=5)   # 40 Hz control
g = torch.Generator().manual_seed(3)
par = DynParams.randomized(B, g=g)

# start box from the old project (scene units -> meters), around hover target
# target: [0, 1.5, 0] u in front of gate, yaw -pi/2 (facing gate)
tgt_p = torch.tensor([0.0, 1.5 * MPU, 0.0]).expand(B, 3)
tgt_yaw = torch.full((B,), -torch.pi / 2)

u = lambda lo, hi: lo + (hi - lo) * torch.rand(B, generator=g)
p0 = torch.stack((u(-1.5, 1.5) * MPU, (1.5 + u(-1.0, 1.5)) * MPU,
                  u(-0.5, 0.4) * MPU), dim=-1)
yaw0 = -torch.pi / 2 + u(-0.6, 0.6)
q0 = quat_from_euler_zyx(yaw0, u(-0.1, 0.1), u(-0.1, 0.1))
v0 = torch.stack((u(-0.5, 0.5), u(-0.5, 0.5), u(-0.3, 0.3)), dim=-1)
w0 = torch.zeros(B, 3)

s = dyn.make_state(p0, v0, q0, w0, par)
expert = GeometricHoverExpert()

T = 8.0
crash = torch.zeros(B, dtype=torch.bool)
for k in range(int(T / dyn.dt_ctrl)):
    a = expert(s, tgt_p, tgt_yaw)
    s = dyn.step(s, a, par)
    crash |= s.p[:, 2] > 1.2                      # floor ~1.2 m below gate center
    crash |= ~torch.isfinite(s.p).all(dim=-1)

err_p = (s.p - tgt_p).norm(dim=-1)
yaw_f, pitch_f, roll_f = euler_zyx_from_quat(s.q)
yaw_err = torch.atan2(torch.sin(yaw_f - tgt_yaw), torch.cos(yaw_f - tgt_yaw)).abs()

ok = (~crash) & (err_p < 0.10) & (s.v.norm(dim=-1) < 0.15) & (yaw_err < 0.10)
print(f"expert hover from start box (B={B}, randomized plant, 40 Hz, delays 50-100 ms):")
print(f"  success (err<10 cm, |v|<0.15, yaw<0.1 rad, no crash): {ok.float().mean()*100:.1f}%")
print(f"  median final pos err: {err_p.median()*100:.1f} cm | worst: {err_p.max()*100:.1f} cm")
print(f"  crashes: {crash.sum().item()}")
q05 = torch.quantile(err_p, 0.95)
print(f"  95th pct err: {q05*100:.1f} cm")
sys.exit(0 if ok.float().mean() > 0.97 else 1)
