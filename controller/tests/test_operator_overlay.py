"""Operator overlay (web-CC VP-tracking debug view) — frame-exactness pins.

The overlay's guarantee is that it never invents a transform:
  * `sides[side].target/elbow/fwd/up` are BYTE-IDENTICAL to the hk the arm IK
    consumed that frame (captured by wrapping ArmRetargeter.solve);
  * `sides[side].hand[0]` (the wrist point) == the target exactly (the skeleton
    is carried rigidly around the target);
  * the mapped hand skeleton is RIGID: inter-point distances equal the raw
    ARKit skeleton's (M is a rotation — no scale, no shear);
  * `raw.*` is the pure AVP_TO_ROBOT axis relabel of the ARKit points
    (norm-preserving, head height preserved);
  * after convergence the robot's FK wrist sits near the streamed target —
    the overlay and the rig demonstrably share one world (odom).
Slow (~20 s): full engine + physics, like test_arm_elbow_retarget.
"""
import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from mabel import MabelTeleop, OperatorPose  # noqa: E402
from mabel.teleop_engine.hyperparams import AVP_TO_ROBOT  # noqa: E402

from test_arm_elbow_retarget import (SH, L_OP, X, Y, FWD, HEAD_H,  # noqa: E402
                                     _hand, _mirror, _unit)

SCENE = ("/home/mabel/open_mabel/simulation/mabel_mujoco/scenes/"
         "mabel_home.xml")


@pytest.fixture(scope="module")
def run():
    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    teleop = MabelTeleop(model, data, base_follow=False, fail_safe=True,
                         clutch=True)
    teleop.set_control_spec(method="retarget")
    eng = teleop.controller

    captured = {}
    for s in ("right", "left"):
        orig = eng.arm[s].solve
        def wrap(hk, *a, _s=s, _o=orig, **kw):
            captured[_s] = {k: (np.copy(v) if isinstance(v, np.ndarray) else v)
                            for k, v in hk.items()}
            return _o(hk, *a, **kw)
        eng.arm[s].solve = wrap

    # a mid-workspace bent pose, both hands
    el = SH["right"] + 0.56 * L_OP * _unit(FWD - 0.35 * Y + 0.10 * X)
    wr = SH["right"] + 0.8 * L_OP * FWD
    rj = _hand("right", wr, FWD, -X, el)
    lj = _hand("left", _mirror(wr), _mirror(FWD), _mirror(-X), _mirror(el))
    head = np.eye(4); head[1, 3] = HEAD_H
    pose = OperatorPose(head=head, right_joints=rj, left_joints=lj)
    pose.right_elbow = rj["forearm_arm"]
    pose.left_elbow = lj["forearm_arm"]
    for k in range(1200):
        pose.timestamp = (k + 1) / 30.0
        teleop.update(pose, dt=model.opt.timestep)
        mujoco.mj_step(model, data)
    rec = eng.last_operator_points
    return dict(model=model, data=data, eng=eng, rec=rec, captured=captured,
                raw_joints={"right": rj, "left": lj})


def test_streamed_points_are_the_ik_inputs(run):
    rec, cap = run["rec"], run["captured"]
    for s in ("right", "left"):
        d, hk = rec["sides"][s], cap[s]
        assert np.allclose(d["target"], hk["contact"], atol=1e-12)
        assert np.allclose(d["elbow"], hk["elbow"], atol=1e-12)
        assert np.allclose(d["fwd"], hk["fwd"], atol=1e-12)
        assert np.allclose(d["up"], hk["up"], atol=1e-12)


def test_hand_skeleton_rides_the_target_rigidly(run):
    rec = run["rec"]
    for s in ("right", "left"):
        d = rec["sides"][s]
        hand = np.asarray(d["hand"], float)
        # wrist point == the IK target, exactly
        assert np.allclose(hand[0], d["target"], atol=1e-12)
        # rigid map: inter-point distances match the raw ARKit skeleton
        raw = np.asarray(rec["raw"][s], float)
        n = min(len(hand), len(raw))
        for i in range(1, n):
            assert abs(np.linalg.norm(hand[i] - hand[0])
                       - np.linalg.norm(raw[i] - raw[0])) < 1e-9


def test_raw_is_a_pure_axis_relabel(run):
    rec, joints = run["rec"], run["raw_joints"]
    assert np.allclose(AVP_TO_ROBOT @ AVP_TO_ROBOT.T, np.eye(3))
    for s in ("right", "left"):
        raw = np.asarray(rec["raw"][s], float)
        for i, (name, T) in enumerate(joints[s].items()):
            if i >= len(raw):
                break
        # spot-check the wrist: relabel of the ARKit position, norm preserved
        p_ark = np.asarray(joints[s]["wrist"], float).reshape(4, 4)[:3, 3]
        assert np.allclose(raw[0], AVP_TO_ROBOT @ p_ark, atol=1e-12)
        assert abs(np.linalg.norm(raw[0]) - np.linalg.norm(p_ark)) < 1e-12
    # the head lands at its true height (ARKit +Y -> robot +Z)
    assert abs(rec["raw"]["head"][2] - HEAD_H) < 1e-9


def test_overlay_and_rig_share_one_world(run):
    model, data, eng, rec = (run["model"], run["data"], run["eng"], run["rec"])
    mujoco.mj_kinematics(model, data)
    for s in ("right", "left"):
        wrist = data.xanchor[eng.arm[s].orca_wrist_jid]
        tgt = np.asarray(rec["sides"][s]["target"], float)
        err = float(np.linalg.norm(wrist - tgt))
        assert err < 0.12, (f"{s}: overlay target {err*1000:.0f} mm from the "
                            "robot's FK wrist — frames disagree")
