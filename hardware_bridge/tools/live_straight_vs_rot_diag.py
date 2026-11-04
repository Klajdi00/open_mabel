#!/usr/bin/env python3
"""live_straight_vs_rot_diag.py — REAL-HARDWARE micro-movement diagnostic.

Single-owner only (kill the bridge + swerve_node first). Uses the EXACT ps5_teleop
path (base_ik.inverse_kinematics + apply_calibration) over the fixed SwerveClient, and
runs a scripted sequence of MICRO movements:

    rest -> fwd -> rest -> back -> rest -> rotate+ -> rest -> rotate- -> rest

For every phase it logs, per module [back, FR=ID1, FL], the COMMANDED drive (post-calib),
the ACTUAL drive velocity (Teensy encoder), the steer angle, and the drive CURRENT
(health poll). A motor that STALLS reads ~0 actual velocity while commanded != 0 and
draws high current — that's the signature we compare between translation and rotation.

SAFETY: VMAX=0.12 m/s, WMAX=0.30 rad/s, ~1.0 s/phase (<0.15 m travel), ramped, zeroed
on exit. No strafe. Re-sends at 50 Hz to feed the command watchdog.
"""
import os, sys, time
import numpy as np

ROOT = "/home/mabel/open_mabel/hardware_bridge"
sys.path.insert(0, ROOT)
from mabel_hw.drivers import base_ik                                  # noqa: E402
from mabel_hw.drivers.maxswerve_host.swerve import SwerveClient        # noqa: E402

VMAX, WMAX = 0.12, 0.30
LOOP_HZ = 50.0
PHASE_S = 1.0
MOD = ["back", "FR(ID1)", "FL"]


def main():
    sw = SwerveClient(teensy_ip="192.168.124.10", port=5800, stream_hz=200).connect()
    time.sleep(0.5)
    if not sw.connected():
        print("NOT CONNECTED to Teensy — abort."); sw.close(); return
    st = sw.getState()
    print(f"connected. live_mask={st.live_mask:#05b}  rest steer(rot)="
          f"{[round(float(x),3) for x in st.steer_angle]}")

    prev_steer = np.array(sw.getState().steer_angle, dtype=float)  # start from ACTUAL heading
    dt = 1.0 / LOOP_HZ

    # (label, vx, vy, wz)
    phases = [
        ("REST",      0.0,  0.0,  0.0),
        ("FWD",       VMAX, 0.0,  0.0),
        ("REST",      0.0,  0.0,  0.0),
        ("BACK",     -VMAX, 0.0,  0.0),
        ("REST",      0.0,  0.0,  0.0),
        ("ROT+",      0.0,  0.0,  WMAX),
        ("REST",      0.0,  0.0,  0.0),
        ("ROT-",      0.0,  0.0, -WMAX),
        ("REST",      0.0,  0.0,  0.0),
    ]

    summary = []
    try:
        for label, vx, vy, wz in phases:
            cmd_hist, act_hist, cur_hist, steer_hist = [], [], [], []
            n = int(PHASE_S * LOOP_HZ)
            for k in range(n):
                tic = time.time()
                # ramp in/out over the first/last 0.2 s to avoid current spikes
                ramp = min(1.0, k / (0.2 * LOOP_HZ), (n - k) / (0.2 * LOOP_HZ))
                drive, steer = base_ik.inverse_kinematics(vx * ramp, vy, wz * ramp, prev_steer)
                prev_steer = steer
                cdrive, csteer = base_ik.apply_calibration(drive, steer)
                sw.setCommand(np.asarray(cdrive, float), np.asarray(csteer, float))
                st = sw.getState()
                cmd_hist.append(np.abs(cdrive))
                act_hist.append(np.abs(np.array(st.drive_velocity, float)))
                steer_hist.append(np.array(st.steer_angle, float))
                if k % 10 == 0:
                    h = sw.getHealthState(timeout=0.03)
                    if h is not None:
                        cur_hist.append(np.array(h.drive_current, float))
                while time.time() - tic < dt:
                    time.sleep(0.001)
            cmd = np.mean(cmd_hist[5:-5] or cmd_hist, axis=0)
            act = np.mean(act_hist[5:-5] or act_hist, axis=0)
            cur = np.mean(cur_hist, axis=0) if cur_hist else np.zeros(3)
            track = np.where(cmd > 1e-3, act / np.maximum(cmd, 1e-6), np.nan)
            print(f"\n[{label:5s}] vx={vx:+.2f} wz={wz:+.2f}")
            for i in range(3):
                stall = (cmd[i] > 0.02 and act[i] < 0.2 * cmd[i])
                flag = "  <-- STALL" if stall else ""
                print(f"   {MOD[i]:8s} cmd={cmd[i]:.3f}  act={act[i]:.3f}  "
                      f"track={track[i]*100 if not np.isnan(track[i]) else 0:5.0f}%  "
                      f"I={cur[i]:5.1f}A{flag}")
            summary.append((label, cmd, act, track, cur))
    finally:
        # zero drive, hold steer, then disable
        sw.setCommand(np.zeros(3), np.asarray(prev_steer, float))
        time.sleep(0.1)
        sw.close()

    # focused verdict on ID1 (module 1) translation vs rotation
    print("\n==== ID1 (front-right) translation vs rotation ====")
    def avg_track(labels):
        ts = [s[3][1] for s in summary if s[0] in labels and not np.isnan(s[3][1])]
        return np.mean(ts) * 100 if ts else float("nan")
    print(f"  ID1 tracking on TRANSLATION (FWD/BACK): {avg_track(['FWD','BACK']):.0f}%")
    print(f"  ID1 tracking on ROTATION  (ROT+/ROT-): {avg_track(['ROT+','ROT-']):.0f}%")


if __name__ == "__main__":
    main()
