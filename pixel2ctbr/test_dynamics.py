"""Verification for dynamics.py / imu.py. Run: python pixel2ctbr/test_dynamics.py"""

import math
import sys

import torch

sys.path.insert(0, "pixel2ctbr")
from dynamics import (DynParams, G, QuadCTBRDynamics, euler_zyx_from_quat,
                      quat_from_euler_zyx, quat_mul, quat_normalize,
                      quat_rotate, quat_rotate_inv)
from imu import ImuParams, ImuSim

torch.manual_seed(0)
FAILURES = []


def check(name, cond, detail=""):
    ok = bool(cond)
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not ok:
        FAILURES.append(name)


def hover_state(dyn, params, B, dev="cpu"):
    p = torch.zeros(B, 3)
    v = torch.zeros(B, 3)
    q = quat_from_euler_zyx(torch.zeros(B), torch.zeros(B), torch.zeros(B))
    w = torch.zeros(B, 3)
    return dyn.make_state(p, v, q, w, params)


def hover_action(params, B):
    c = (G / params.thrust_gain).unsqueeze(-1)
    return torch.cat((c, torch.zeros(B, 3)), dim=-1)


# 1. quaternion sanity vs scipy ------------------------------------------------
try:
    from scipy.spatial.transform import Rotation
    ypr = torch.tensor([[0.3, -0.2, 0.1], [-1.5708, 0.0, 0.0], [2.9, 0.4, -0.3]])
    q = quat_from_euler_zyx(ypr[:, 0], ypr[:, 1], ypr[:, 2])
    R_scipy = Rotation.from_euler("ZYX", ypr.numpy()).as_matrix()
    v = torch.tensor([[1.0, 2.0, 3.0]]).expand(3, 3)
    v_mine = quat_rotate(q, v)
    v_ref = torch.tensor((R_scipy @ v.numpy()[..., None]).squeeze(-1), dtype=torch.float32)
    check("quat matches scipy ZYX", torch.allclose(v_mine, v_ref, atol=1e-5),
          f"maxdiff {(v_mine - v_ref).abs().max():.2e}")
    y2, p2, r2 = euler_zyx_from_quat(q)
    back = torch.stack((y2, p2, r2), -1)
    check("euler roundtrip", torch.allclose(back, ypr, atol=1e-5),
          f"maxdiff {(back - ypr).abs().max():.2e}")
    vi = quat_rotate_inv(q, v_mine)
    check("rotate_inv inverts rotate", torch.allclose(vi, v, atol=1e-5))
except ImportError:
    print("SKIP  scipy not available")

# 2. hover fixed point ---------------------------------------------------------
B = 4
dyn = QuadCTBRDynamics(dt_ctrl=0.025, n_sub=5)
par = DynParams.nominal(B)
par.kd_lin = torch.zeros(B)  # drag irrelevant at v=0 but be exact
s = hover_state(dyn, par, B)
a = hover_action(par, B)
for _ in range(int(4.0 / dyn.dt_ctrl)):
    s = dyn.step(s, a, par)
check("hover: |p| stays ~0 for 4 s", s.p.abs().max() < 1e-3, f"max |p| {s.p.abs().max():.2e}")
check("hover: |v| ~0", s.v.abs().max() < 1e-3, f"{s.v.abs().max():.2e}")
check("hover: level attitude", (s.q[:, 1:3].abs()).max() < 1e-6)

# 2b. hover with thrust_gain != 1 (command G/gain -> produced G) ---------------
par_g = DynParams.nominal(B)
par_g.thrust_gain = torch.full((B,), 1.12)
s = hover_state(dyn, par_g, B)
a = hover_action(par_g, B)
for _ in range(80):
    s = dyn.step(s, a, par_g)
check("hover under thrust-map gain", s.v.abs().max() < 1e-3, f"{s.v.abs().max():.2e}")

# 3. free fall -----------------------------------------------------------------
par0 = DynParams.nominal(B)
par0.kd_lin = torch.zeros(B)
par0.delay_steps = torch.ones(B, dtype=torch.long)  # 1-step FIFO = immediate
s = hover_state(dyn, par0, B)
s.thrust = torch.zeros(B)
s.cmd_fifo = torch.zeros_like(s.cmd_fifo)
par0.tau_c = torch.full((B,), 1e-6)  # kill thrust lag; command 0 thrust
a0 = torch.zeros(B, 4)
T = 1.0
for _ in range(int(T / dyn.dt_ctrl)):
    s = dyn.step(s, a0, par0)
z_expect = 0.5 * G * T * T  # z DOWN so falling = +z
check("free fall z=+g t^2/2", torch.allclose(s.p[:, 2], torch.full((B,), z_expect), rtol=0.02),
      f"z {s.p[0,2]:.3f} vs {z_expect:.3f}")

# 4. rate-loop step response ----------------------------------------------------
par = DynParams.nominal(B)
par.delay_steps = torch.ones(B, dtype=torch.long)
s = hover_state(dyn, par, B)
w_cmd = torch.tensor([1.0, 0.0, 0.0])
a = torch.cat(((G / par.thrust_gain).unsqueeze(-1), w_cmd.expand(B, 3)), dim=-1)
t63 = None
t = 0.0
for _ in range(200):
    s = dyn.step(s, a, par)
    t += dyn.dt_ctrl
    if t63 is None and s.w[0, 0] >= 0.632 * 1.0:
        t63 = t
check("rate step ~tau_w to 63%", t63 is not None and abs(t63 - 0.05) <= dyn.dt_ctrl,
      f"t63 {t63}")
check("rate converges to cmd", abs(s.w[0, 0] - 1.0) < 1e-3, f"{s.w[0,0]:.4f}")

# 5. yaw spin keeps level -------------------------------------------------------
par = DynParams.nominal(B)
s = hover_state(dyn, par, B)
a = torch.cat(((G / par.thrust_gain).unsqueeze(-1),
               torch.tensor([0.0, 0.0, 0.5]).expand(B, 3)), dim=-1)
for _ in range(160):  # 4 s
    s = dyn.step(s, a, par)
yaw, pitch, roll = euler_zyx_from_quat(s.q)
check("yaw-only spin: pitch/roll ~0", max(pitch.abs().max(), roll.abs().max()) < 1e-4)
check("yaw-only spin: |v| small", s.v.abs().max() < 5e-3, f"{s.v.abs().max():.2e}")
check("quat norm stable", (s.q.norm(dim=-1) - 1).abs().max() < 1e-5)

# 6. delay FIFO exactness -------------------------------------------------------
par = DynParams.nominal(1)
par.delay_steps = torch.tensor([3])
dyn2 = QuadCTBRDynamics(dt_ctrl=0.025, n_sub=1)
s = hover_state(dyn2, par, 1)
probe = torch.tensor([[G, 0.9, 0.0, 0.0]])
hold = hover_action(par, 1)
s1 = dyn2.step(s, probe, par)     # cmd enters FIFO
s2 = dyn2.step(s1, hold, par)
check("delay: rate cmd not yet applied at t+2", s2.w[0, 0].abs() < 1e-9, f"{s2.w[0,0]:.2e}")
s3 = dyn2.step(s2, hold, par)     # 3rd step: applied
check("delay: rate cmd applied at t+3", s3.w[0, 0] > 0.1, f"{s3.w[0,0]:.3f}")

# 7. BPTT gradient flow (perturb off the fixed point or grads are legitimately 0)
par = DynParams.nominal(2)
s = hover_state(dyn, par, 2)
theta = torch.full((2, 4), 0.05, requires_grad=True)
st = s
for _ in range(50):
    act = hover_action(par, 2) + theta
    st = dyn.step(st, act, par)
loss = (st.p ** 2).sum() + (st.v ** 2).sum()
loss.backward()
gnorm = theta.grad.norm().item()
check("BPTT 50 steps: grad finite & nonzero", math.isfinite(gnorm) and gnorm > 1e-6,
      f"|grad| {gnorm:.3e}")

# 8. IMU ------------------------------------------------------------------------
par = DynParams.nominal(B)
s = hover_state(dyn, par, B)
a = hover_action(par, B)
for _ in range(40):
    s = dyn.step(s, a, par)
ip = ImuParams.randomized(B)
ip.gyro_bias = torch.zeros(B, 3); ip.accel_bias = torch.zeros(B, 3)
imu = ImuSim(ip, dyn.dt_ctrl)
reads = [imu.read(s) for _ in range(200)]
acc = torch.stack([r["accel"] for r in reads]).mean(0)
gyr = torch.stack([r["gyro"] for r in reads]).mean(0)
check("IMU hover accel ~[0,0,-g]",
      torch.allclose(acc, torch.tensor([[0.0, 0.0, -G]]).expand(B, 3), atol=0.05),
      f"mean z {acc[0,2]:.3f}")
check("IMU hover gyro ~0", gyr.abs().max() < 0.01)
check("IMU is detached", not reads[0]["accel"].requires_grad)

# 9. randomized params: shapes + stability under hover controller ---------------
g = torch.Generator().manual_seed(7)
parr = DynParams.randomized(64, g=g)
s = hover_state(dyn, parr, 64)
kp_v = 3.0
for _ in range(120):  # crude accel feedback hover using true state (sanity only)
    c = (G / parr.thrust_gain) - kp_v * (-s.v[:, 2])
    act = torch.cat((c.unsqueeze(-1), torch.zeros(64, 3)), dim=-1)
    s = dyn.step(s, act, parr)
check("randomized batch: finite states", torch.isfinite(s.p).all() and torch.isfinite(s.q).all())
check("randomized batch: alt drift bounded (<1 m in 3 s w/ crude fb)",
      s.p[:, 2].abs().max() < 1.0, f"max |z| {s.p[:,2].abs().max():.3f}")

print("\n" + ("ALL PASS" if not FAILURES else f"FAILURES: {FAILURES}"))
sys.exit(1 if FAILURES else 0)
