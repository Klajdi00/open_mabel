#!/usr/bin/env python3
"""
Step 3 (optional) -- open the interactive MuJoCo viewer on the arm model.

Useful for sanity-checking the kinematics and watching an excitation trajectory.
Requires a display.  Press Esc / close the window to exit.

Usage:
    python scripts/03_view_arm.py            # static model
    python scripts/03_view_arm.py --excite   # play a Fourier trajectory
"""
import argparse
import time

import _bootstrap  # noqa: F401
import mujoco
import mujoco.viewer
import numpy as np

from mabel_sysid.arm_model import ArmModel
from mabel_sysid import trajectory as T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excite", action="store_true",
                    help="play a Fourier excitation trajectory")
    args = ap.parse_args()

    arm = ArmModel(_bootstrap.MODEL_PATH)
    m, d = arm.model, arm.data

    if args.excite:
        t, Q, _, _ = T.fourier_trajectory(arm, duration=20.0, dt=0.02,
                                           rng=np.random.default_rng(0))
    else:
        Q = None

    with mujoco.viewer.launch_passive(m, d) as viewer:
        k = 0
        while viewer.is_running():
            if Q is not None:
                d.qpos[:] = Q[k % len(Q)]
                k += 1
                mujoco.mj_forward(m, d)
                time.sleep(0.02)
            else:
                mujoco.mj_forward(m, d)
                time.sleep(0.05)
            viewer.sync()


if __name__ == "__main__":
    main()
