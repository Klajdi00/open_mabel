#!/usr/bin/env python3
"""live_optimize_fix_test.py — validate the host-side steer-optimize fix on real hardware.

The straight-line stall is the steer 180-deg wheel-flip landing modules in inconsistent
flip states (position-dependent => intermittent). Standard fix (WPILib optimize): command
the NEAREST heading to the MEASURED steer, negating drive when we flip, so the firmware
never has to choose a flip and all modules stay consistent.

This drives a small FORWARD micro-move from several different rest positions, WITHOUT then
WITH optimize, and reports per-trial forward tracking + drive current. The fix should make
forward tracking high and CONSISTENT across rest positions (no more stall lottery).

SAFE: VMAX=0.10 m/s, ~0.8 s/trial, ramped, zeroed on exit, no strafe. To vary the rest
flip-state we first nudge a tiny rotation so wheels settle in a new place each trial.
"""
import sys, time
import numpy as np

ROOT = "/home/mabel/open_mabel/hardware_bridge"
sys.path.insert(0, ROOT)
from mabel_hw.drivers import base_ik                                  # noqa: E402
from mabel_hw.drivers.maxswerve_host.swerve import SwerveClient        # noqa: E402

VMAX, LOOP_HZ, PHASE_S = 0.40, 50.0, 0.45   # ~0.18 m/burst, above the kF=0 stiction floor


def optimize(cdrive, csteer, meas_steer):
    """Nearest-heading + signed drive vs the MEASURED steer (rotations)."""
    d = np.array(cdrive, float); s = np.array(csteer, float)
    for i in range(3):
        err = (csteer[i] - meas_steer[i]) % 1.0
        if 0.25 < err < 0.75:          # target is on the far side -> flip
            s[i] = (csteer[i] + 0.5) % 1.0
            d[i] = -cdrive[i]
    return d, s


def settle_new_restpos(sw, wz, secs=0.6):
    """Tiny in-place rotation so the wheels stop in a fresh flip-state, then idle."""
    n = int(secs * LOOP_HZ)
    prev = np.array(sw.getState().steer_angle, float)
    for k in range(n):
        drive, steer = base_ik.inverse_kinematics(0.0, 0.0, wz, prev); prev = steer
        cd, cs = base_ik.apply_calibration(drive, steer)
        sw.setCommand(np.asarray(cd, float), np.asarray(cs, float))
        time.sleep(1.0 / LOOP_HZ)
    for _ in range(int(0.4 * LOOP_HZ)):     # idle, hold steer
        sw.setCommand(np.zeros(3), np.array(sw.getState().steer_angle, float))
        time.sleep(1.0 / LOOP_HZ)


def fwd_trial(sw, use_opt):
    prev = np.array(sw.getState().steer_angle, float)
    drive, steer = base_ik.inverse_kinematics(VMAX, 0.0, 0.0, prev)
    cdrive, csteer = base_ik.apply_calibration(drive, steer)
    n = int(PHASE_S * LOOP_HZ); dt = 1.0 / LOOP_HZ
    vel_hist, cur_hist = [], []
    for k in range(n):
        tic = time.time()
        ramp = min(1.0, k / (0.2 * LOOP_HZ), (n - k) / (0.2 * LOOP_HZ))
        meas = np.array(sw.getState().steer_angle, float)
        if use_opt:
            d, s = optimize(np.asarray(cdrive, float), np.asarray(csteer, float), meas)
        else:
            d, s = np.asarray(cdrive, float), np.asarray(csteer, float)
        sw.setCommand(d * ramp, s)
        st = sw.getState()
        vel_hist.append(np.array(st.drive_velocity, float))
        if k % 10 == 0:
            h = sw.getHealthState(timeout=0.03)
            if h is not None:
                cur_hist.append(np.array(h.drive_current, float))
        while time.time() - tic < dt:
            time.sleep(0.001)
    vel = np.array(vel_hist)[10:-5]
    # forward speed magnitude = mean |per-wheel speed| (all should translate together)
    speed = np.mean(np.abs(vel))
    cur = np.mean(np.max(cur_hist, axis=1)) if cur_hist else 0.0
    track = speed / VMAX * 100
    return track, cur


def main():
    sw = SwerveClient(teensy_ip="192.168.124.10", port=5800, stream_hz=200).connect()
    time.sleep(0.5)
    if not sw.connected():
        print("NOT CONNECTED — abort."); sw.close(); return
    print("Comparing FORWARD micro-move from 4 different rest positions.\n")
    try:
        for mode, use_opt in [("WITHOUT optimize", False), ("WITH optimize", True)]:
            print(f"=== {mode} ===")
            tracks = []
            for trial, wz in enumerate([0.4, -0.4, 0.6, -0.5]):
                settle_new_restpos(sw, wz)
                track, cur = fwd_trial(sw, use_opt)
                tracks.append(track)
                print(f"  trial{trial+1}: forward tracking={track:4.0f}%  peakI={cur:4.1f}A")
            tr = np.array(tracks)
            print(f"  -> mean={tr.mean():.0f}%  min={tr.min():.0f}%  "
                  f"spread={tr.max()-tr.min():.0f}%  "
                  f"({'STABLE' if tr.min() > 25 else 'STALLS in some configs'})\n")
    finally:
        sw.setCommand(np.zeros(3), np.array(sw.getState().steer_angle, float))
        time.sleep(0.1)
        sw.close()


if __name__ == "__main__":
    main()
