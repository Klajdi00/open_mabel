#!/usr/bin/env python3
"""live_steer_settle_diag.py — SAFE steer-only probe (drive = 0 throughout).

Wheels pivot in place; the base does NOT translate. For each target heading set we hold
the steer command with ZERO drive and watch whether each module's steer angle CONVERGES
to its target, logging the residual error + steer current. If a module (esp. ID1=FR)
cannot reach the "straight-ahead" heading, that explains a translation-specific stall:
on a straight command the wheel sits mis-aligned and the drive fights it (high current,
no motion), while rotation headings happen to be reachable.

Targets are the post-calibration steer values base_ik produces for each body motion, so
they are exactly what the real teleop sends. NO drive is ever commanded.
"""
import sys, time
import numpy as np

ROOT = "/home/mabel/open_mabel/hardware_bridge"
sys.path.insert(0, ROOT)
from mabel_hw.drivers import base_ik                                  # noqa: E402
from mabel_hw.drivers.maxswerve_host.swerve import SwerveClient        # noqa: E402

MOD = ["back", "FR(ID1)", "FL"]
HOLD_S = 2.0
LOOP_HZ = 50.0


def steer_target(vx, vy, wz):
    drive, steer = base_ik.inverse_kinematics(vx, vy, wz, np.zeros(3))
    _, csteer = base_ik.apply_calibration(drive, steer)
    return np.asarray(csteer, float)


def ang_err(a, b):
    """Smallest signed distance on the [0,1) circle (rotations)."""
    d = (a - b) % 1.0
    return d - 1.0 if d > 0.5 else d


def main():
    sw = SwerveClient(teensy_ip="192.168.124.10", port=5800, stream_hz=200).connect()
    time.sleep(0.5)
    if not sw.connected():
        print("NOT CONNECTED — abort."); sw.close(); return
    print(f"connected. rest steer={[round(float(x),3) for x in sw.getState().steer_angle]}")

    targets = [
        ("FORWARD", steer_target(0.5, 0.0, 0.0)),
        ("BACK",    steer_target(-0.5, 0.0, 0.0)),
        ("ROTATE",  steer_target(0.0, 0.0, 0.5)),
        ("FORWARD2", steer_target(0.5, 0.0, 0.0)),   # repeat to see repeatability
    ]
    dt = 1.0 / LOOP_HZ
    try:
        for label, tgt in targets:
            err_hist, scur_hist = [], []
            n = int(HOLD_S * LOOP_HZ)
            for k in range(n):
                tic = time.time()
                sw.setCommand(np.zeros(3), tgt)            # ZERO drive, hold target heading
                st = sw.getState()
                err = np.array([ang_err(float(st.steer_angle[i]), tgt[i]) for i in range(3)])
                err_hist.append(err)
                if k % 10 == 0:
                    h = sw.getHealthState(timeout=0.03)
                    if h is not None:
                        scur_hist.append(np.array(h.steer_current, float))
                while time.time() - tic < dt:
                    time.sleep(0.001)
            settle = np.array(err_hist)
            final_err = np.mean(np.abs(settle[-15:]), axis=0)        # last 0.3 s residual
            jitter = np.std(settle[-25:], axis=0)                     # last 0.5 s wobble
            scur = np.mean(scur_hist, axis=0) if scur_hist else np.zeros(3)
            print(f"\n[{label}] target(rot)={[round(float(x),3) for x in tgt]}")
            for i in range(3):
                stuck = final_err[i] > 0.03            # >~11 deg residual = not converged
                wob = jitter[i] > 0.02                  # hunting
                flag = "  <-- NOT CONVERGED" if stuck else ("  <-- HUNTING" if wob else "")
                print(f"   {MOD[i]:8s} residual={final_err[i]*360:5.1f}deg  "
                      f"jitter={jitter[i]*360:4.1f}deg  steerI={scur[i]:4.1f}A{flag}")
    finally:
        sw.setCommand(np.zeros(3), np.asarray(sw.getState().steer_angle, float))
        time.sleep(0.1)
        sw.close()
    print("\n(steer-only: base did not translate)")


if __name__ == "__main__":
    main()
