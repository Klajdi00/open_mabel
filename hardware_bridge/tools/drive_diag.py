#!/usr/bin/env python3
"""drive_diag.py — REAL-base drive velocity tracking + current diagnostic.

Answers "is the drive command true to real life?": it commands a ladder of drive
velocity setpoints through the SAME IK+calibration path ps5_teleop/the bridge use, and
reads back the ACTUAL wheel velocity (encoder) + drive current per module. The gap
between commanded and actual tells you what is wrong:

  * actual ≈ commanded (free wheels, off the ground)        -> command path is correct;
    if it then stalls on the ground, it is LOAD/TORQUE (the +20 kg) — raise the SPARK
    drive smart-current-limit + velocity kP/kFF in REV Hardware Client.
  * actual << commanded even with wheels OFF the ground      -> the SPARK velocity loop /
    conversion factor is mis-provisioned (firmware side), not the host.
  * current pinned at a low ceiling while actual < commanded -> current-limited: raise it.

SAFETY: commands real motion. Put the WHEELS OFF THE GROUND for the first run. It ramps
gently, holds each setpoint ~1.5 s, and ALWAYS zeros + closes on exit. Stop the bridge /
any other client first (single owner). Run:

    python3 tools/drive_diag.py --teensy-ip 192.168.124.10 --max 2.0
"""
import argparse
import os
import sys
import time

import numpy as np

# Use the canonical proven path (same code ps5_teleop + the bridge run).
_REPO = os.path.expanduser("~/projects/upstream/maxswerve-host")
if os.path.isdir(_REPO):
    sys.path.insert(0, _REPO)
from maxswerve_host import SwerveClient            # noqa: E402
from ps5_teleop import inverse_kinematics, apply_calibrated  # noqa: E402

NAMES = ["back", "front_right", "front_left"]


def hold(sw, vx, secs, prev, settle=False):
    """Command body twist (vx,0,0) for `secs`; return (mean_actual[3], max_cur[3], prev)."""
    t0 = time.time()
    vels, curs = [], []
    while time.time() - t0 < secs:
        drive, steer = inverse_kinematics(vx, 0.0, 0.0, prev); prev = steer
        apply_calibrated(sw, drive, steer)
        st = sw.getState()
        if st is not None and not settle:
            vels.append(np.abs(st.drive_velocity))
        time.sleep(0.02)
    if not settle:
        h = sw.getHealthState(timeout=0.3)
        if h is not None:
            curs.append(np.abs(h.drive_current))
    mean_v = np.mean(vels, axis=0) if vels else np.zeros(3)
    max_c = np.max(curs, axis=0) if curs else np.zeros(3)
    return mean_v, max_c, prev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teensy-ip", default="192.168.124.10")
    ap.add_argument("--max", type=float, default=2.0, help="top setpoint m/s")
    args = ap.parse_args()

    sw = SwerveClient(teensy_ip=args.teensy_ip).connect()
    time.sleep(0.6)
    if not sw.connected():
        print("NO TELEMETRY — robot powered? bridge still running (not single owner)? Aborting.")
        sw.close(); return 1
    prev = np.array(sw.getState().steer_angle, dtype=float)

    setpoints = [s for s in (0.2, 0.5, 1.0, 1.5, 2.0, 3.0) if s <= args.max + 1e-6]
    print(f"\ncommanding forward through IK+calib (wheels OFF the ground recommended).")
    print(f"{'cmd m/s':>8} | {'actual m/s (back / FR / FL)':>34} | {'drive A':>20} | track%")
    print("-" * 80)
    try:
        for vx in setpoints:
            _, _, prev = hold(sw, vx * 0.1, 0.6, prev, settle=True)   # steer-settle, no drive read
            v, c, prev = hold(sw, vx, 1.6, prev)
            track = 100.0 * (np.mean(v) / vx) if vx > 0 else 0.0
            print(f"{vx:8.2f} | " +
                  "  ".join(f"{v[i]:5.2f}" for i in range(3)) + "   | " +
                  "  ".join(f"{c[i]:4.1f}" for i in range(3)) + " | " +
                  f"{track:5.0f}%")
            hold(sw, 0.0, 0.5, prev, settle=True)     # stop between
    finally:
        try:
            hold(sw, 0.0, 0.4, prev, settle=True)
        except Exception:
            pass
        sw.close()
    print("\nReading: track% = actual/commanded. ~100% off the ground = command is correct;")
    print("a big drop ON the ground (+20 kg) = torque/current-limited -> tune the SPARK")
    print("drive current limit + velocity kP/kFF (firmware/REV Hardware Client).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
