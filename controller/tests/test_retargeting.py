"""Retargeting sanity: hand retargeters produce in-range finger commands from a
synthetic skeleton, the grasp map is monotone, the wrist retargeter anchors and
produces finite palm targets, and the neck is truly DECOUPLED (it writes only
the neck actuators)."""
import numpy as np
import mujoco
import pytest

from conftest import needs_model, settle

import mabel.config as C
from mabel.hand_retargeting import make_hand_retargeter, apply_grasp, HandIKRetargeter
from mabel.whole_body_retargeting import Retargeter, NeckController
from mabel.whole_body_retargeting.calibration import synthesize_pose, POSE_HANDS
from mabel.sources import mock_frame
from mabel.types import OperatorPose


def _finger_actuator_ids(model, side):
    names = [model.actuator(i).name for i in range(model.nu)]
    keys = ("Thumb", "Index", "Middle", "Ring", "Pinky")
    return [i for i, n in enumerate(names)
            if n.startswith(side) and any(k in n for k in keys)]


@needs_model
def test_hand_retargeter_produces_in_range_commands(fresh):
    model, data = fresh()
    settle(model, data)
    rt = make_hand_retargeter(model, data)
    assert isinstance(rt, HandIKRetargeter)     # config default: keypoint IK
    pose = synthesize_pose(POSE_HANDS)[-1]     # last frame of the synthetic hold
    assert pose.left_joints is not None
    ctrl0 = data.ctrl.copy()
    rt.apply("Left", pose.left_joints, synchronous=True)
    ids = _finger_actuator_ids(model, "Left")
    assert ids, "no left finger actuators found"
    lo = model.actuator_ctrlrange[ids, 0]; hi = model.actuator_ctrlrange[ids, 1]
    vals = data.ctrl[ids]
    assert np.isfinite(vals).all()
    assert np.all(vals >= lo - 1e-9) and np.all(vals <= hi + 1e-9)
    # DECOUPLED from the body: arm/base actuators untouched by the hand solve
    others = [i for i in range(model.nu) if i not in ids]
    assert np.allclose(data.ctrl[others], ctrl0[others])


@needs_model
def test_grasp_monotone_and_in_range(fresh):
    model, data = fresh()
    ids = _finger_actuator_ids(model, "Right")
    sums = []
    for g in (0.0, 0.5, 1.0):
        apply_grasp(model, data, "Right", g)
        vals = data.ctrl[ids]
        lo = model.actuator_ctrlrange[ids, 0]; hi = model.actuator_ctrlrange[ids, 1]
        assert np.all(vals >= lo - 1e-9) and np.all(vals <= hi + 1e-9)
        sums.append(np.abs(vals).sum())
    assert sums[0] < sums[1] < sums[2], "grasp not monotone in the grip value"


@needs_model
def test_wrist_retargeter_calibrates_and_maps(fresh):
    model, data = fresh()
    settle(model, data)
    rt = Retargeter(model, data)
    pose = OperatorPose.from_avframe(mock_frame(0.0))
    assert rt.calibrate(pose)
    out = rt.update(OperatorPose.from_avframe(mock_frame(0.5)))
    assert out is not None
    for k in ("l_pos", "r_pos"):
        assert k in out and np.isfinite(out[k]).all()
    for k in ("l_mat", "r_mat"):
        R = out[k]
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-6), f"{k} not a rotation"
    assert "gaze_dir" in out and abs(np.linalg.norm(out["gaze_dir"]) - 1) < 1e-6


@needs_model
def test_neck_is_decoupled_writes_only_neck(fresh):
    model, data = fresh()
    settle(model, data)
    neck = NeckController(model, data)
    ctrl0 = data.ctrl.copy()
    target = data.body(C.HEAD_BODY).xpos + np.array([-1.0, 0.3, 0.0])
    neck.solve_and_apply(target, 0.01)
    neck_ids = [model.actuator(n).id for n in C.NECK_ACTUATORS]
    others = [i for i in range(model.nu) if i not in neck_ids]
    assert np.allclose(data.ctrl[others], ctrl0[others]), \
        "neck controller wrote non-neck actuators — decoupling broken"
    assert not np.allclose(data.ctrl[neck_ids], ctrl0[neck_ids]), \
        "neck controller did not move the neck"
