#!/usr/bin/env python3
"""Does HAND motion move the NECK? (the erratic-neck-while-stationary report)

THE OPERATOR REPORT (2026-07-31)
  "the neck acts super erratic even when I am stationary ... I had working code
   before for the retargeting, find it and see what changed"

WHAT CHANGED (git): commit 0b7d37f5, 2026-07-28. Before it the neck datum was
the WORLD-anchored head orientation captured once at anchor time:

    na = neck_angles(head_mat, self._head0)          # a CONSTANT reference

after it, the datum is rebuilt EVERY FRAME from the body-frame estimator:

    axes = self.shoulder.body_axes()                 # estimated from the HANDS
    ref  = np.column_stack([right, up, -fwd])
    ...  neck_angles(head_mat, ref4)

The estimator is fed by the head AND BOTH WRISTS, so the neck's zero now moves
with the operator's hands. A perfectly still head can therefore command a moving
neck, which is exactly the report.

This holds the head RIGIDLY STILL and moves only the hands. Every milliradian of
neck command is error.

    python3 controller/experiments/probe_neck_jitter.py
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import exp_vp_bimanual_latency as h                              # noqa: E402

NECK = ["neck_yaw", "neck_pitch", "neck_roll"]


def _roll(side, t, amp=0.5, hz=0.4):
    """A realistic wrist that ROLLS as it moves — pronation/supination is
    continuous in real teleoperation, and it is precisely what the body-frame
    estimator's orientation cue reads."""
    a = amp * np.sin(2 * np.pi * hz * t + (0.0 if side == "right" else 1.3))
    c, sn = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -sn], [0, sn, c]])


def run(label, traj, secs=10.0, rot=None):
    op = h.SyntheticVPOperator(traj, rot_fn=rot)
    rig = h.Rig(frame_hz=30.0)
    acts = [rig.model.actuator(n).id for n in ("Neck Yaw", "Neck Pitch", "Neck Roll")]
    t, nxt = 0.0, 0.0
    rec = []
    while t < secs:
        pose = None
        if t >= nxt:
            pose = op.frame(t)
            nxt += 1.0 / 30.0
        rig.teleop.update(pose, None, h.DT)
        rig._plant()
        if t > 3.0:                       # past the anchor/settle transient
            rec.append([float(rig.data.ctrl[a]) for a in acts])
        t += h.DT
    q = np.asarray(rec, float)
    if len(q) < 50:
        print(f"{label:30s} (no data)")
        return None
    rng = q.max(axis=0) - q.min(axis=0)
    print(f"{label:30s} yaw {np.degrees(rng[0]):6.2f}  pitch {np.degrees(rng[1]):6.2f}  "
          f"roll {np.degrees(rng[2]):6.2f}   deg peak-to-peak")
    return rng


def main():
    print("\nNECK COMMAND WITH A RIGIDLY STATIONARY HEAD")
    print("(the operator's head never moves — every degree below is error)\n")
    print("-- rigid hands (translation only) --")
    run("hands still (control)", lambda s, t: h.traj_static(s, t))
    run("BOTH hands sweeping", lambda s, t: h.traj_bimanual(s, t))
    print("-- REALISTIC hands (they also roll) --")
    run("hands still, wrists rolling", lambda s, t: h.traj_static(s, t), rot=_roll)
    run("BOTH sweeping + rolling", lambda s, t: h.traj_bimanual(s, t), rot=_roll)
    run("right only + rolling", lambda s, t: h.traj_bimanual(s, t, only="right"), rot=_roll)
    print()
    # tremor case needs the noisy operator
    op = h.SyntheticVPOperator(lambda s, t: h.traj_static(s, t), noise_m=0.002)
    rig = h.Rig(frame_hz=30.0)
    acts = [rig.model.actuator(n).id for n in ("Neck Yaw", "Neck Pitch", "Neck Roll")]
    t, nxt, rec = 0.0, 0.0, []
    while t < 10.0:
        pose = None
        if t >= nxt:
            pose = op.frame(t); nxt += 1.0 / 30.0
        rig.teleop.update(pose, None, h.DT)
        rig._plant()
        if t > 3.0:
            rec.append([float(rig.data.ctrl[a]) for a in acts])
        t += h.DT
    q = np.asarray(rec, float)
    rng = q.max(axis=0) - q.min(axis=0)
    print(f"{'hands still + 2 mm tremor':30s} yaw {np.degrees(rng[0]):6.2f}  "
          f"pitch {np.degrees(rng[1]):6.2f}  roll {np.degrees(rng[2]):6.2f}   deg peak-to-peak\n")


if __name__ == "__main__":
    main()
