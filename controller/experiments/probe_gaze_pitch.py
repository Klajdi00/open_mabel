#!/usr/bin/env python3
"""Does LEANING IN tilt the robot's gaze down? (the looking-at-the-table report)

THE OPERATOR REPORT (2026-07-31)
  "when I get close to the table the neck is not tracking the human head
   orientation well — when the human looks straight, the robot is looking
   downwards at the table"

THE MECHANISM. The robot's head is mounted on the TORSO, and two independent
laws aim it while only one of them was being controlled:

  * `TorsoController.command(fwd_in) = t0 - TORSO_GAIN * fwd_in`
    (TORSO_GAIN = 2.5 rad per metre of forward lean) — the torso is the FAST
    reach extender, so approaching a table bends it;
  * `neck_angles(head_mat, ref)` — reproduces the operator's head rotation
    relative to the body datum, with no torso term anywhere in it.

The gaze is the SUM, so every radian the torso leans in for reach is a radian of
gaze the operator never asked for.

WHY THIS IS A MODEL-LEVEL PROBE. Driving it through the full stack needs the
whole-body controller's bounding-box engage rule to fire, which a synthetic
operator does not reliably trigger — so the full-stack version measured nothing
and would have been a test that always passes. The kinematics do not need the
stack: the question "how does each joint move the gaze" is answered by the MJCF
itself, exactly and without a control loop, which is also what makes the sign of
the correction safe to derive rather than guess.

    python3 controller/experiments/probe_gaze_pitch.py
"""
import os

import numpy as np
import mujoco

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
MJCF = os.path.join(_REPO, "simulation", "mabel_mujoco", "models", "mabel_full.xml")


def main():
    m = mujoco.MjModel.from_xml_path(MJCF)
    d = mujoco.MjData(m)
    hb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "head")

    def qadr(j):
        return m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]

    tq, nq = qadr("Torso"), qadr("Neck Pitch")
    rest = float(m.qpos0[tq])
    nj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "Neck Pitch")
    nlo, nhi = m.jnt_range[nj]

    def gaze():
        """Elevation of the head's forward (z) axis [deg]; + = up, 0 = level."""
        mujoco.mj_forward(m, d)
        return float(np.degrees(np.arcsin(np.clip(d.xmat[hb].reshape(3, 3)[:, 2][2],
                                                  -1, 1))))

    print("\nGAZE SENSITIVITY (from the MJCF — no control loop involved)")
    for j, adr in (("Torso", tq), ("Neck Pitch", nq)):
        mujoco.mj_resetData(m, d)
        base = gaze()
        d.qpos[adr] = 0.20
        print(f"  {j:11s} +0.20 rad -> gaze {gaze() - base:+7.2f} deg")
    print("  => equal and opposite, so the correction is exactly 1:1:")
    print("     neck_pitch += (torso - torso_rest)\n")

    print("ROBOT GAZE WITH THE OPERATOR LOOKING LEVEL")
    print("  torso [rad]   uncompensated      compensated")
    worst = 0.0
    for t in (0.0, 0.10, 0.20, 0.30, 0.40, -0.20):
        mujoco.mj_resetData(m, d); d.qpos[tq] = t
        g_un = gaze()
        mujoco.mj_resetData(m, d); d.qpos[tq] = t
        d.qpos[nq] = float(np.clip(t - rest, nlo, nhi))      # the fix
        g_c = gaze()
        worst = max(worst, abs(g_c))
        print(f"    {t:+.2f}       {g_un:+7.2f} deg        {g_c:+7.2f} deg")
    print(f"\n  worst compensated error {worst:.2f} deg — "
          f"{'PASS' if worst < 0.5 else 'FAIL'}\n")


if __name__ == "__main__":
    main()
