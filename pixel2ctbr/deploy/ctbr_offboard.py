"""CTBR offboard runner for Starling 2 — pixel2ctbr deployment.

Adapted from Starling2/voxl-docker-mavsdk-python/ctrl_lya_offboard.py
(pilot-in-the-loop skeleton, watchdogs, CSV logging) with the command path
swapped to body rates + thrust and the failsafe ladder redesigned for rate
control (04_design.md §6, deployment report 01_research_report.md §4).

  NN (onboard, MPA)  →  CtrlLyaMsg action = [c m/s^2, wx, wy, wz rad/s]
                     →  this script  →  MAVSDK set_attitude_rate()

FAILSAFE LADDER (rate-mode; a stale rate command ≠ hover!):
  msg age > STALE_HOLD_S   → stream HOVER-HOLD frame (hover thrust, 0 rates)
                             at KEEPALIVE_HZ (PX4 must keep seeing setpoints
                             or it holds the LAST rates for COM_OF_LOSS_T)
  msg age > STALE_EXIT_S   → offboard.stop() → pilot in Stabilized
  RC mode flip             → pilot_takeover: stop commanding, exit (no land)
  Ctrl-C / reader death    → offboard.stop() handoff

RUN ONLY AFTER: the PX4 param recipe is applied (see check_params below /
deploy/px4_params.md), bench props-off test, tether test. NOT yet run on
hardware — bench validation is the next step in 06_verification.md.
"""

import asyncio
import csv
import math
import os
import struct
import subprocess
import sys
import time

from mavsdk import System
from mavsdk.offboard import AttitudeRate, OffboardError

# ---- policy/action semantics (must match training: policy.py) --------------
G = 9.81
HOVER_THRUST_NORM = 0.34      # shipped MPC_THR_HOVER; refine on bench
THRUST_PER_MS2 = HOVER_THRUST_NORM / G   # linear map around hover (v1)
RATE_LIMIT_DPS = (229.0, 229.0, 115.0)   # ±4,4,2 rad/s in deg/s, mirrors sim clamp
SAFETY_RATE_SCALE = 1.0       # first flights may use <1.0
MSG_FMT = "=4s7fQ"            # CtrlLyaMsg: magic, action[6], V, ts_ns (40 B unchanged)
MSG_SIZE = struct.calcsize(MSG_FMT)
MAGIC = b"CLYA"

STALE_HOLD_S = 0.15           # hover-hold begins
STALE_EXIT_S = 0.60           # offboard.stop() handoff
KEEPALIVE_HZ = 50.0           # our own resend rate (PX4 sees fresh setpoints)
MPA_READER = "/root/mpa_reader"   # inside the docker image, as before

# Params that MUST hold for estimator-less rate offboard (report §(a)):
PARAM_RECIPE = {
    "EKF2_HGT_REF": 0,        # ships 3=vision — MUST be baro w/o mocap
    "EKF2_GPS_CTRL": 0,
    "EKF2_EV_CTRL": 0,
    "EKF2_MAG_TYPE": 5,       # mag present+calibrated but unfused
    "COM_OBL_RC_ACT": 2,      # offboard-loss → Stabilized
    "COM_ARM_WO_GPS": 1,
}
PARAM_WARN = {"COM_OF_LOSS_T": (0.3, 0.5)}   # want short; default 1.0 s


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def action_to_setpoint(a):
    """[c m/s^2, wx, wy, wz rad/s] (body FRD) -> AttitudeRate (deg/s + thrust01)."""
    thrust = clamp(a[0] * THRUST_PER_MS2, 0.0, 0.60)      # MPC_THR_MAX shipped 0.60
    r = [clamp(math.degrees(a[1 + i]) * SAFETY_RATE_SCALE,
               -RATE_LIMIT_DPS[i], RATE_LIMIT_DPS[i]) for i in range(3)]
    return AttitudeRate(r[0], r[1], r[2], thrust)


HOVER_HOLD = AttitudeRate(0.0, 0.0, 0.0, HOVER_THRUST_NORM)


class Log:
    def __init__(self, path="/tmp/ctbr_logs"):
        os.makedirs(path, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.f = open(f"{path}/flight_{ts}_ctbr.csv", "w", newline="")
        self.w = csv.writer(self.f)
        self.w.writerow(["t_wall", "ts_ns", "age_ms", "c_ms2", "wx", "wy", "wz",
                         "thrust_norm", "sent", "event"])

    def row(self, ts_ns, age_ms, a, sp, sent, event=""):
        self.w.writerow([f"{time.time():.4f}", ts_ns, f"{age_ms:.1f}",
                         *(f"{x:.4f}" for x in a[:4]),
                         f"{sp.thrust_value:.4f}", int(sent), event])

    def event(self, name):
        self.w.writerow([f"{time.time():.4f}", "", "", "", "", "", "", "", "", name])
        self.f.flush()


async def check_params(drone):
    ok = True
    for name, want in PARAM_RECIPE.items():
        try:
            got = await drone.param.get_param_int(name)
        except Exception:
            got = int(await drone.param.get_param_float(name))
        if got != want:
            print(f"[PARAM FAIL] {name} = {got}, need {want}")
            ok = False
    for name, (lo, hi) in PARAM_WARN.items():
        got = await drone.param.get_param_float(name)
        if not (lo <= got <= hi):
            print(f"[PARAM WARN] {name} = {got}, recommend {lo}-{hi}")
    return ok


def read_msg(proc, buf):
    """Magic-scan CtrlLyaMsg from mpa_reader stdout (ctrl_lya_offboard pattern)."""
    while True:
        i = buf.find(MAGIC)
        if i >= 0 and len(buf) >= i + MSG_SIZE:
            m = struct.unpack(MSG_FMT, bytes(buf[i:i + MSG_SIZE]))
            del buf[:i + MSG_SIZE]
            return m, buf
        chunk = proc.stdout.read(512)
        if not chunk:
            return None, buf
        buf.extend(chunk)


async def main():
    drone = System()
    await drone.connect(system_address="udp://:14551")
    async for st in drone.core.connection_state():
        if st.is_connected:
            break
    print("connected")
    if not await check_params(drone):
        print("param recipe not satisfied — fix params, then rerun")
        return

    log = Log()
    armed = False
    async for a in drone.telemetry.armed():
        armed = a
        break
    if not armed:
        print("ARM the drone in Stabilized and hover manually first "
              "(pilot-in-the-loop flow); exiting.")
        return

    proc = subprocess.Popen([MPA_READER], stdout=subprocess.PIPE, bufsize=0)
    buf = bytearray()

    # stream setpoints ≥1 s before offboard.start() (PX4 requirement)
    for _ in range(int(1.2 * KEEPALIVE_HZ)):
        await drone.offboard.set_attitude_rate(HOVER_HOLD)
        await asyncio.sleep(1.0 / KEEPALIVE_HZ)
    try:
        await drone.offboard.start()
    except OffboardError as e:
        print(f"offboard start failed: {e._result.result}")
        proc.kill()
        return
    log.event("offboard_start")
    print("OFFBOARD (rate) — policy flying; RC mode flip = instant takeover")

    last_msg_t = time.monotonic()
    last_sp = HOVER_HOLD
    loop = asyncio.get_event_loop()
    try:
        while True:
            # non-blocking-ish read via executor keeps the keepalive honest
            m, buf = await loop.run_in_executor(None, read_msg, proc, buf)
            now = time.monotonic()
            if m is not None:
                a = m[1:5]
                sp = action_to_setpoint(a)
                last_sp, last_msg_t = sp, now
                await drone.offboard.set_attitude_rate(sp)
                log.row(m[7], 0.0, a, sp, True)
            age = now - last_msg_t
            if m is None or age > STALE_HOLD_S:
                if age > STALE_EXIT_S:
                    log.event("stale_exit_handoff")
                    break
                await drone.offboard.set_attitude_rate(HOVER_HOLD)
                log.row(0, age * 1e3, (0, 0, 0, 0), HOVER_HOLD, True, "hover_hold")
                await asyncio.sleep(1.0 / KEEPALIVE_HZ)
            # pilot takeover?
            async for fm in drone.telemetry.flight_mode():
                if str(fm) != "OFFBOARD":
                    log.event("pilot_takeover")
                    print("pilot takeover — exiting without commanding")
                    proc.kill()
                    return
                break
    except (KeyboardInterrupt, asyncio.CancelledError):
        log.event("ctrl_c")
    finally:
        try:
            await drone.offboard.stop()   # PX4 → COM_OBL_RC_ACT (Stabilized)
            log.event("offboard_stop_handoff")
        except OffboardError:
            log.event("offboard_stop_failed")
        proc.kill()
        log.event("exit")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
