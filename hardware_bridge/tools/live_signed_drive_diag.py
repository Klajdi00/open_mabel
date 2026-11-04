#!/usr/bin/env python3
"""live_signed_drive_diag.py — decisive test: are the wheels FIGHTING on a straight cmd?

For FORWARD and BACK micro commands, log per module [back, FR=ID1, FL]:
  * commanded steer (post-calib) vs ACTUAL steer  -> did the firmware FLIP it (+0.5)?
  * SIGNED actual drive velocity (Teensy encoder) -> if signs are MIXED across modules,
    the wheels are driving against each other = the straight-line stall.

Hypothesis: the firmware's per-module steer wheel-flip is not matched by a drive-sign
flip, so when modules land in mixed flip states (position-dependent => intermittent) the
flipped wheel spins backwards and fights the others. Rotation is less affected because
its per-module headings differ enough that the flip pattern stays consistent.

SAFE: VMAX=0.10 m/s, ~0.8 s, ramped, zeroed on exit, no strafe.
"""
import sys, time
import numpy as np

ROOT = "/home/mabel/open_mabel/hardware_bridge"
sys.path.insert(0, ROOT)
from mabel_hw.drivers import base_ik                                  # noqa: E402
from mabel_hw.drivers.maxswerve_host.swerve import SwerveClient        # noqa: E402

MOD = ["back", "FR(ID1)", "FL"]
VMAX, LOOP_HZ, PHASE_S = 0.10, 50.0, 0.8


def flipped(actual, cmd):
    d = abs((float(actual) - float(cmd)) % 1.0)
    d = min(d, 1.0 - d)
    return d > 0.25     # closer to cmd+0.5 than cmd


def run_phase(sw, label, vx):
    prev = np.array(sw.getState().steer_angle, float)
    drive, steer = base_ik.inverse_kinematics(vx, 0.0, 0.0, prev)
    cdrive, csteer = base_ik.apply_calibration(drive, steer)
    n = int(PHASE_S * LOOP_HZ); dt = 1.0 / LOOP_HZ
    vel_hist = []
    for k in range(n):
        tic = time.time()
        ramp = min(1.0, k / (0.2 * LOOP_HZ), (n - k) / (0.2 * LOOP_HZ))
        cd = np.asarray(cdrive, float) * ramp
        sw.setCommand(cd, np.asarray(csteer, float))
        st = sw.getState()
        vel_hist.append(np.array(st.drive_velocity, float))   # SIGNED
        while time.time() - tic < dt:
            time.sleep(0.001)
    st = sw.getState()
    act_steer = np.array(st.steer_angle, float)
    vel = np.mean(np.array(vel_hist)[10:-5], axis=0)          # SIGNED mean velocity
    print(f"\n[{label}] vx={vx:+.2f}  cmd_steer={[round(float(x),3) for x in csteer]}")
    signs = []
    for i in range(3):
        fl = flipped(act_steer[i], csteer[i])
        signs.append(np.sign(vel[i]) if abs(vel[i]) > 0.01 else 0)
        print(f"   {MOD[i]:8s} act_steer={act_steer[i]:.3f} {'FLIPPED' if fl else 'normal ':7s}"
              f"  signed_vel={vel[i]:+.3f} m/s")
    pos = sum(1 for s in signs if s > 0); neg = sum(1 for s in signs if s < 0)
    if pos and neg:
        print(f"   ==> WHEELS FIGHTING: {pos} forward + {neg} reverse  (= straight-line STALL)")
    else:
        print(f"   ==> consistent ({pos} fwd, {neg} rev, {3-pos-neg} idle)")
    return signs


def main():
    sw = SwerveClient(teensy_ip="192.168.124.10", port=5800, stream_hz=200).connect()
    time.sleep(0.5)
    if not sw.connected():
        print("NOT CONNECTED — abort."); sw.close(); return
    print(f"connected. rest steer={[round(float(x),3) for x in sw.getState().steer_angle]}")
    try:
        for label, vx in [("FWD", VMAX), ("BACK", -VMAX), ("FWD2", VMAX)]:
            run_phase(sw, label, vx)
            # idle between phases so wheels settle wherever they stop (vary flip state)
            for _ in range(int(0.6 * LOOP_HZ)):
                sw.setCommand(np.zeros(3), np.array(sw.getState().steer_angle, float))
                time.sleep(1.0 / LOOP_HZ)
    finally:
        sw.setCommand(np.zeros(3), np.array(sw.getState().steer_angle, float))
        time.sleep(0.1)
        sw.close()


if __name__ == "__main__":
    main()
