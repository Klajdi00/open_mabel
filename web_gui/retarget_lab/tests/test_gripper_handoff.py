"""Physical gripper contact: true distal tips, real envelopes, bounded commands."""
from pathlib import Path

import mujoco
import numpy as np
import pytest

from mabel.teleop_engine import collision as collision_module
from mabel.teleop_engine.end_effector import (
    GRIPPER, distal_tip, gripper_collision_shapes, tracked_point,
)
from mabel.types import OperatorPose
from mabel.whole_body_controller.collision_avoidance.capsule_guard import _seg_seg
from retarget_lab.server.worker import KinematicRetargeter


def _rx(t):
    return np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])


def _ry(t):
    return np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]])


def _pose(t, contact=(0., 1.20, -.42), gap=0.):
    """Opposed pinches: closed human fingertips share the requested contact.

    Wrist flexion cancels the normal 35-degree pinch tilt, so the two physical
    grippers approach horizontally. This avoids demanding an unreachable
    upward-pointing wrist at the left arm's maximum extension.
    """
    p = OperatorPose(timestamp=t)
    p.head = np.eye(4)
    p.head[:3, 3] = [0., 1.55, 0.]
    for side, sign in (("right", 1.), ("left", -1.)):
        roll = np.eye(3) if side == "right" else np.diag([-1., -1., 1.])
        R = _ry(sign * np.pi / 2) @ roll @ _rx(np.radians(35))
        point = np.asarray(contact) + [sign * gap / 2, 0., 0.]
        wrist = point + [sign * .15, 0., 0.]
        joints = {}
        offsets = {"wrist": [0., 0., 0.], "middle_metacarpal": [0., 0., -.05],
                   "index_metacarpal": [-sign * .018, 0., -.035],
                   "little_metacarpal": [sign * .036, 0., -.035]}
        for name, offset in offsets.items():
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = R, wrist + R @ offset
            joints[name] = T
        for name in ("thumb_tip", "index_tip", "middle_tip"):
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = R, point
            joints[name] = T
        setattr(p, side + "_joints", joints)
    return p


def _tip(kr, side):
    a = kr.eng.arm[side]
    body, local = distal_tip(kr.model, side, a.palm_bid, a.R_rob)
    return kr.data.xpos[body] + kr.data.xmat[body].reshape(3, 3) @ local


def _gaps(kr):
    out = []
    for a, b in kr.eng.projector.guard.pairs:
        aa, ab = a.world(kr.data)
        ba, bb = b.world(kr.data)
        u, v = _seg_seg(aa, ab, ba, bb)
        out.append((float(np.linalg.norm(u - v) - a.radius - b.radius), a.name, b.name))
    return out


@pytest.fixture(scope="module")
def model():
    return KinematicRetargeter("shipped_v3").model


@pytest.mark.parametrize("contact,jaw", [((0., 1.20, -.42), None), ((0., 1.32, -.42), .044),
                                        ((0., 1.20, -.35), 0.)])
def test_opposed_pinches_meet_with_physical_clearance_and_bounded_commands(model, contact, jaw):
    kr = KinematicRetargeter("shipped_v5", model=model)
    kr.apply_settings({"robot": "mabel", "scope": "local", "ee": "gripper"})
    kr.eng.enable_gripper_contact = True
    kr.eng.gripper_measured = dict.fromkeys(("right", "left"), jaw)
    kr.eng.gripper_commanded = dict.fromkeys(("right", "left"), jaw)
    previous, smallest, body_smallest, tail = None, np.inf, np.inf, []
    for i in range(1500):
        kr.step(_pose(i / 60, contact), i / 60, collide=True)
        command = kr.eng.projector.q_cmd
        if command is None:
            continue
        if previous is not None:
            assert np.max(np.abs(command - previous)) <= collision_module.ARM_QP_VMAX / 60 + 1e-8
        previous = command.copy()
        for gap, a, b in _gaps(kr):
            smallest = min(smallest, gap)
            if not ("grip_finger" in a and "grip_finger" in b):
                body_smallest = min(body_smallest, gap)
        if i >= 1440:
            tail.append(float(np.linalg.norm(_tip(kr, "right") - _tip(kr, "left"))))
    assert max(tail) < .010
    assert max(tail) - min(tail) < .001   # less than 1 mm stationary variation
    assert smallest >= 0.0
    assert body_smallest >= .009
    guard = kr.eng.projector.guard
    for a, b in guard.pairs:
        if not ("grip_finger" in a.name and "grip_finger" in b.name):
            assert guard._margin_for(a, b) == guard.d_safe
            assert guard.pair_lag_scale.get(tuple(sorted((a.name, b.name))), 1.) == 1.


def test_contact_blend_releases_and_tracking_loss_disables_coupling(model):
    kr = KinematicRetargeter("shipped_v5", model=model)
    kr.apply_settings({"robot": "mabel", "scope": "local", "ee": "gripper"})
    kr.eng.enable_gripper_contact = True
    for i in range(90):
        p = _pose(i / 60)
        kr.step(p, i / 60, collide=True)
    assert kr.eng._gripper_contact_alpha > .99
    p.left_joints = None
    payload = kr.eng._payload(p)
    out = kr.eng._gripper_handoff_targets(payload, {"right": np.array([-.4, .1, 1.])}, np.eye(3))
    assert "right" in out
    assert kr.eng._gripper_handoff_separation is None
    for i in range(180):
        p = _pose(2 + i / 60, gap=.30)
        kr.step(p, 2 + i / 60, collide=True)
    assert kr.eng._gripper_contact_alpha == 0.
    assert kr.eng._gripper_handoff_separation is None
    for side, a in kr.eng.arm.items():
        _, base = tracked_point(kr.model, side, a.palm_bid, a.R_rob)
        np.testing.assert_array_equal(a._ee_off, base)


def test_moving_into_and_out_of_contact_keeps_commands_continuous(model):
    kr = KinematicRetargeter("shipped_v5", model=model)
    kr.apply_settings({"robot": "mabel", "scope": "local", "ee": "gripper"})
    previous = None
    for i in range(900):
        gap = .30 * (1. - np.clip(i / 300., 0., 1.)) if i < 600 else .30 * ((i - 600.) / 299.)
        kr.step(_pose(i / 60., gap=gap), i / 60., collide=True)
        q = kr.eng.projector.q_cmd
        if q is None:
            continue
        if previous is not None:
            assert np.max(np.abs(q - previous)) <= collision_module.ARM_QP_VMAX / 60. + 1e-8
        previous = q.copy()
        assert min(d for d, _, _ in _gaps(kr)) >= 0.


def test_lost_hand_cannot_supply_an_unpublished_collision_correction(model):
    kr = KinematicRetargeter("shipped_v5", model=model)
    kr.apply_settings({"robot": "mabel", "scope": "local", "ee": "gripper"})
    for i in range(1500):
        kr.step(_pose(i / 60.), i / 60., collide=True)
    projector = kr.eng.projector
    left = projector.q_cmd[projector.split:].copy()
    for i in range(1500, 1680):
        p = _pose(i / 60.)
        p.left_joints = None
        kr.step(p, i / 60., collide=True)
        assert projector.active_sides == frozenset({"right"})
        assert projector.handoff_separation is None
        np.testing.assert_array_equal(projector.q_cmd[projector.split:], left)
        assert min(d for d, _, _ in _gaps(kr)) >= 0.
    held = projector.q_cmd.copy()
    for i in range(1680, 1740):
        p = _pose(i / 60.)
        p.left_joints = p.right_joints = None
        kr.step(p, i / 60., collide=True)
        np.testing.assert_array_equal(projector.q_cmd, held)


def test_orca_envelopes_and_margins_are_unchanged(model):
    kr = KinematicRetargeter("shipped_v3", model=model)
    projector = kr.eng.projector
    original = list(projector.guard.pairs)
    kr.eng.set_end_effector("gripper")
    projector.enable_gripper_contact = True
    projector._end_effectors()
    assert projector._tip_points
    kr.eng.set_end_effector("orca")
    projector.handoff_alpha = 0.
    projector._end_effectors()
    assert projector.guard.pairs == original
    assert projector.guard.lag_capsules == projector._orca_lag
    assert projector.guard.pair_lag_scale == {}


def test_contact_is_opt_in_and_unknown_jaws_keep_the_full_stroke(model):
    kr = KinematicRetargeter("shipped_v3", model=model)
    projector = kr.eng.projector
    original = list(projector.guard.pairs)
    kr.eng.set_end_effector("gripper")
    assert not kr.eng.enable_gripper_contact
    projector._end_effectors()
    assert projector.guard.pairs == original
    assert not projector._tip_points
    projector.enable_gripper_contact = True
    projector._end_effectors()
    cap = projector.guard.caps["r_grip_finger_tip"]
    full_length = np.linalg.norm(cap.p1 - cap.p0)
    projector.jaw_measured = {"right": .030}
    projector.jaw_commanded = {"right": .0}
    projector._end_effectors()
    lag_length = np.linalg.norm(cap.p1 - cap.p0)
    assert lag_length < full_length
    projector.jaw_measured["right"] = .0
    projector._end_effectors()
    closed_length = np.linalg.norm(cap.p1 - cap.p0)
    assert closed_length < lag_length
    projector.jaw_measured["right"] = None
    projector._end_effectors()
    assert np.linalg.norm(cap.p1 - cap.p0) == pytest.approx(full_length)
    projector.enable_gripper_contact = False
    projector._end_effectors()
    assert projector.guard.pairs == original


def test_contact_rejects_a_finite_unconstrained_solver_fallback(model, monkeypatch):
    kr = KinematicRetargeter("shipped_v5", model=model)
    projector = kr.eng.projector
    kr.eng.set_end_effector("gripper")
    projector.enable_gripper_contact = True
    projector.handoff_alpha = 1.
    projector.handoff_separation = np.zeros(3)
    initial = kr.data.qpos[projector.qadr].copy()
    monkeypatch.setattr(collision_module, "solve_qp", lambda *args: np.full(projector.n, 1000.))
    projector.project(initial[:7], initial[8:15], 1. / 60, initial[7], initial[15])
    np.testing.assert_array_equal(projector.q_cmd, initial)
    assert projector.fallbacks == 1


def test_nearly_parallel_finger_contact_does_not_amplify_pose_roundoff(model):
    # A captured near-contact configuration previously amplified a 1e-13 rad
    # perturbation into 0.004 rad within 150 command steps. Both copies use the
    # same held posture target; only that starting roundoff perturbation differs.
    start = np.array([1.1901890206732657, -.32642037073528346, -.6311351311994063,
                      -.731937005142138, -.8070143300805861, -.19979028386073702,
                      -1.197920260721842, 0., -.7688303971833028, -.17344970539946367,
                      -.2613679384128807, .3004862252689307, -1.570796,
                      -.3589309249331013, -1.2043948734121641, 0.])
    desired_r = [1.1940083561695285, -.2663746262067093, -.6799091956283629,
                 -.7172790573739043, -.8131401543734275, -.20012485994657248, -1.2]
    desired_l = [-.7678301167794562, -.174533, -.18716201962493373,
                 .3046097855846155, -1.570796, -.35984510633286326, -1.2]
    rigs = []
    for index in range(2):
        kr = KinematicRetargeter("shipped_v5", model=model)
        kr.apply_settings({"robot": "mabel", "scope": "local", "ee": "gripper"})
        p = kr.eng.projector
        p.enable_gripper_contact = True
        p.handoff_alpha, p.handoff_separation = 1., np.zeros(3)
        p.jaw_measured = p.jaw_commanded = dict.fromkeys(("right", "left"), .044)
        p.q_cmd = start.copy()
        p.q_cmd[0] += index * 1e-13
        kr.data.qpos[p.qadr] = p.q_cmd
        mujoco.mj_forward(model, kr.data)
        rigs.append(kr)
    for _ in range(150):
        for kr in rigs:
            p = kr.eng.projector
            p.project(desired_r, desired_l, 1. / 60., wrist_r=0., wrist_l=0.)
            kr.data.qpos[p.qadr] = p.q_cmd
            mujoco.mj_kinematics(model, kr.data)
            mujoco.mj_comPos(model, kr.data)
        assert np.max(np.abs(rigs[0].eng.projector.q_cmd - rigs[1].eng.projector.q_cmd)) < 1e-10


@pytest.mark.parametrize("opening", [0., .022, .044])
def test_gripper_envelopes_cover_the_rig_meshes_over_the_jaw_stroke(opening):
    meshes = Path(__file__).resolve().parents[3] / "simulation/mabel_mujoco/models/meshes"
    def vertices(relative, offset, flip=False):
        v = np.array([[float(x) for x in line.split()[1:4]]
                      for line in (meshes / relative).read_text().splitlines() if line.startswith("v ")]) * .001
        if flip:
            v[:, 1] *= -1
        return v + offset
    chunks = [vertices(r, np.array(GRIPPER["housing_geom_pos"])) for r in GRIPPER["meshes"]["housing"]]
    chunks += [vertices(r, np.array(GRIPPER["hand_pos"]) + GRIPPER["hand_geom_pos"])
               for r in GRIPPER["meshes"]["hand"]]
    fingers = []
    for finger in GRIPPER["fingers"]:
        for jaw in (0., opening / 2., opening):
            offset = (np.array(GRIPPER["hand_pos"]) + finger["pos"] + finger["geom_pos"]
                      + jaw * np.asarray(finger["axis"]))
            fingers += [vertices(r, offset, finger["flip"]) for r in GRIPPER["meshes"]["finger"]]
    assert max(v[:, 2].max() for v in fingers) == pytest.approx(GRIPPER["tip"], abs=1e-7)
    v = np.concatenate(chunks + fingers)
    distances = []
    for a, b, radius in gripper_collision_shapes(opening).values():
        a, b = np.asarray(a), np.asarray(b)
        axis = b - a
        t = np.clip((v - a) @ axis / max(axis @ axis, 1e-15), 0., 1.)
        distances.append(np.linalg.norm(v - a - t[:, None] * axis, axis=1) - radius)
    assert np.min(distances, axis=0).max() <= 1e-6
    a, b, radius = gripper_collision_shapes(opening)["finger_tip"]
    assert max(a[2], b[2]) + radius <= GRIPPER["tip"] + .000501
