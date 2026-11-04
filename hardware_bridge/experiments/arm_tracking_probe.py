#!/usr/bin/env python3
"""arm_tracking_probe.py — OPERATOR-RUN position-tracking probe for J1/J2/J4.

⚠ THIS MOVES THE REAL ARM. Run it yourself, with eyes on the robot and a hand on the
e-stop. It is deliberately NOT run from automation (hardware_bridge/README.md: "No
robot/hand motion from automation — user-run checklist"), and that matters more than
usual right now: the HAL no longer carries a thermal cut, stall guard, or slew guard,
and l_j4's motor is already impaired (see the arm-2pi-stall-and-gravity-ff memory).

WHAT IT ANSWERS
  1. Does the joint TRACK a commanded angle, and with how much steady-state error?
  2. Is that error explained by gravity  (e = tau_g / kp)?  -> gravity FF should cancel it
  3. Does the pose WALK AWAY in ~2 s steps? (the ratchet) -> prints a per-2s drift table
  4. Is the feedforward or the error accumulating? -> logs cmd / measured / tau each tick

METHOD — small angles only. For each joint it steps to rest+delta for a few seconds,
samples cmd vs measured vs tau at 20 Hz, then returns to rest and de-energises. It
NEVER commands a far target: every step is <= --delta from the joint's RESTING pose, so
kp*error stays small by construction even before the controller's rate limit applies.

USAGE (on thor, bridge running, arms powered, e-stop in reach):
    python3 experiments/arm_tracking_probe.py --joint l_j2 --delta 0.15 --dwell 6
    python3 experiments/arm_tracking_probe.py --joint l_j2 --delta 0.15 --no-torque   # dry run

  --no-torque   read-only: streams NOTHING, just records the resting pose + drift.
                RUN THIS FIRST. If the pose drifts in 2 s steps with torque OFF and no
                commands, the problem is upstream (the gateway ratchet), not tracking.

READ THE OUTPUT
  steady-state error e and the predicted gravity error tau_g/kp are printed side by
  side. e ~= tau_g/kp  => gravity FF is the fix (it is not a tuning problem: kd is
  hardware-capped at 5, which caps kp ~= 15, so PD alone CANNOT remove the droop).
  A DRIFT column that grows in ~2 s steps is the reconcile ratchet
  (server/sim_teleop_bridge.py `_reconcile_resumed`), not a tracking failure.
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FW = ("l_j1", "l_j2", "l_j3", "l_j4", "l_j5", "l_j6", "l_j7",
      "r_j1", "r_j2", "r_j3", "r_j4", "r_j5", "r_j6", "r_j7")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--joint", default="l_j2", choices=FW)
    ap.add_argument("--delta", type=float, default=0.15,
                    help="rad from the RESTING pose (default 0.15 ~= 8.6 deg). Keep small.")
    ap.add_argument("--dwell", type=float, default=6.0, help="s to hold each step")
    ap.add_argument("--rate", type=float, default=20.0, help="sample Hz")
    ap.add_argument("--no-torque", action="store_true",
                    help="READ-ONLY: never energise, never command. Run this first.")
    args = ap.parse_args()

    if abs(args.delta) > 0.35:
        print("refusing: --delta > 0.35 rad is not a 'small angle' probe.")
        return 2

    from mabel_hw import MabelHardware
    side = "left_arm" if args.joint.startswith("l_") else "right_arm"

    print(__doc__.split("USAGE")[0])
    if not args.no_torque:
        print(f"!! ABOUT TO ENERGISE {side} and move {args.joint} by {args.delta:+.3f} rad.")
        if input("   Type MOVE to continue: ").strip() != "MOVE":
            print("aborted.")
            return 1

    hw = MabelHardware(subsystems=[side], logger=lambda m: print("  [hw]", m))
    hw.open()
    try:
        handle = getattr(hw, side)
        if not handle.available:
            print(f"{side} not available — is MABEL_HW_{side.upper()} enabled and the "
                  "arms Teensy streaming? (tools/arms_link_check.py)")
            return 1
        for _ in range(50):
            if handle.connected():
                break
            time.sleep(0.1)

        rest = handle.state().as_dict().get(args.joint)
        if rest is None:
            print(f"{args.joint} not in this group: {handle.names}")
            return 1
        lim = handle.commandable_ranges().get(args.joint)
        print(f"\n{args.joint}: resting at {rest:+.4f} rad; calibrated range {lim}")

        # ── phase 0: READ-ONLY drift. Torque OFF, nothing commanded. ──────────
        print("\n── phase 0: drift with torque OFF, NOTHING commanded (10 s) ──")
        print("   if this drifts in ~2 s steps, the fault is upstream, not tracking.")
        t0 = time.monotonic()
        base = rest
        while time.monotonic() - t0 < 10.0:
            time.sleep(1.0)
            q = handle.state().as_dict()[args.joint]
            print(f"   t={time.monotonic()-t0:4.1f}s  q={q:+.4f}  drift={q-base:+.4f}")
        if args.no_torque:
            print("\n--no-torque: done (nothing was energised or commanded).")
            return 0

        # ── phase 1: energise + hold, then step ──────────────────────────────
        with handle.claim("arm_tracking_probe") as w:
            w.set_torque(True)
            time.sleep(0.5)
            for label, target in (("hold", rest), ("step", rest + args.delta),
                                  ("back", rest)):
                print(f"\n── {label}: commanding {target:+.4f} rad for {args.dwell}s ──")
                print("      t     cmd        q         err      tau      drift/2s")
                t0 = time.monotonic()
                errs, last2, prev_q = [], t0, None
                while time.monotonic() - t0 < args.dwell:
                    w.set_joint_positions({args.joint: target})
                    time.sleep(1.0 / args.rate)
                    st = handle.state()
                    q = st.as_dict()[args.joint]
                    tau = dict(zip(st.names, st.effort))[args.joint]
                    e = target - q
                    errs.append(e)
                    now = time.monotonic()
                    if now - last2 >= 2.0:
                        d = "" if prev_q is None else f"{q - prev_q:+.4f}"
                        print(f"   {now-t0:5.1f}  {target:+.4f}  {q:+.4f}  "
                              f"{e:+.4f}  {tau:+6.2f}   {d}")
                        prev_q, last2 = q, now
                if errs:
                    ss = statistics.fmean(errs[-int(args.rate):])
                    print(f"   steady-state error e = {ss:+.4f} rad ({ss*57.3:+.1f} deg)")
                    print(f"   -> implied tau_g = kp*e ; with kp~15 that is {15*ss:+.2f} N.m")
                    print("   -> if e ~= tau_g/kp, gravity FF (controller/mabel/arm_control)")
                    print("      is the fix: kd is hardware-capped at 5 => kp~15 max, so PD")
                    print("      alone can NEVER remove this droop.")
            w.set_torque(False)
            print("\ntorque OFF. arm limp.")
    finally:
        hw.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
