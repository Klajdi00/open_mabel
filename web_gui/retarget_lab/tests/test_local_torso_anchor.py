"""A slow bend must survive local walking-anchor updates (live NOVA regression)."""
import math

import numpy as np
import pytest

from retarget_lab.server.worker import KinematicRetargeter


@pytest.fixture(scope="module")
def robot():
    kr = KinematicRetargeter("shipped_v3")
    kr.apply_settings({"robot": "mabel", "scope": "local", "ee": "gripper"})
    return kr


@pytest.mark.parametrize("motion", ["small_bend", "hip_bend", "walk", "squat"])
def test_local_torso_keeps_bend_evidence_without_following_gait(robot, motion):
    w, d = robot.eng.wbc, robot.data
    w.reset(d)
    w.base_in_loop = False
    d.ctrl[w.torso_act] = 0.0
    angles = []
    for i in range(361):
        u = min(1.0, max(0.0, (i - 60) / 240))
        theta = math.radians(20) * u
        if motion == "small_bend":
            # Similar displacement to the recorded bend that commanded zero.
            head = [0, 1.7 - .034 * u, -.044 * u]
        elif motion == "hip_bend":
            head = [0, 1.7 - .7 * (1 - math.cos(theta)), -.7 * math.sin(theta)]
        elif motion == "walk":
            head = [0, 1.7 + .02 * math.sin(i / 60 * 2 * math.pi * 1.8), -.6 * u]
        else:
            head = [0, 1.7 - .2 * u, 0]
        h = np.array(head)
        w.update_targets(d, h, h, math.pi, head[1], dt=1 / 60)
        angles.append(float(d.ctrl[w.torso_act]))
    if motion == "small_bend":
        assert min(angles) < -.02
    elif motion == "hip_bend":
        assert min(angles) < -.15
    else:
        assert max(abs(a) for a in angles) < 1e-9
