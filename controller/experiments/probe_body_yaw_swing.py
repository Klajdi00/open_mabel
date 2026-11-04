#!/usr/bin/env python3
"""How much does a PURE ARM MOTION rotate the estimated operator body frame?

The bisection in probe_arm_crosstalk.py pinned the bimanual cross-talk on
ARM_BODY_FRAME_MAP: the arm targets are decomposed in the operator's BODY frame,
whose yaw is estimated from the head AND BOTH WRISTS, so one hand's motion turns
the frame and drags the other arm's target with it.

Here the operator's head and torso are RIGIDLY STATIC and only the hands move.
Any body-frame yaw at all is therefore error, and its magnitude is the size of
the coupling. Reported per configuration so the estimator's own rejection logic
can be checked rather than assumed.

    python3 controller/experiments/probe_body_yaw_swing.py
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import exp_vp_bimanual_latency as h                              # noqa: E402


def yaw_of(axes):
    """Body-frame yaw [rad] from the (right, up, fwd) triple, in ARKit world."""
    if axes is None:
        return np.nan
    r = axes[0]
    return float(np.arctan2(r[2], r[0]))


def run(label, traj, frames=1200):
    op = h.SyntheticVPOperator(traj)
    rig = h.Rig(frame_hz=30.0)
    ys, ts = [], []
    t = 0.0
    nxt = 0.0
    while t < 10.0:
        pose = None
        if t >= nxt:
            pose = op.frame(t)
            nxt += 1.0 / 30.0
        rig.teleop.update(pose, None, h.DT)
        rig._plant()
        if t > 2.0:
            ys.append(yaw_of(rig.eng.shoulder.body_axes()))
            ts.append(t)
        t += h.DT
    ys = np.asarray(ys, float)
    ys = ys[np.isfinite(ys)]
    if len(ys) < 10:
        print(f"{label:34s} (no body axes)")
        return
    swing = float(np.nanmax(ys) - np.nanmin(ys))
    print(f"{label:34s} yaw swing {np.degrees(swing):6.2f} deg   "
          f"std {np.degrees(np.nanstd(ys)):5.2f} deg")
    return swing


def main():
    print("\nBODY-FRAME YAW UNDER A RIGIDLY STATIC HEAD AND TORSO")
    print("(only the hands move — every degree here is coupling error)\n")
    run("both hands sweeping", lambda s, t: h.traj_bimanual(s, t))
    run("right hand only", lambda s, t: h.traj_bimanual(s, t, only="right"))
    run("left hand only", lambda s, t: h.traj_bimanual(s, t, only="left"))
    run("hands still (control)", lambda s, t: h.traj_static(s, t))
    print()


if __name__ == "__main__":
    main()
