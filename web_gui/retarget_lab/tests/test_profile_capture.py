"""Offline calibration quality: finite tracking alone must never imply a profile."""
from copy import deepcopy

import numpy as np
import pytest

from mabel.teleop_engine.geometry import ShoulderEstimator
from mabel.whole_body_retargeting.calibration import body_frame
from mabel.whole_body_retargeting.profile_capture import (
    assess_operator_capture, assess_neutral_forward_capture, fit_wrist_sphere,
)


def capture(angle=60, yaw=0.0, noise=0.0005, n=45):
    rng = np.random.default_rng(3)
    a = np.deg2rad(angle)
    directions = np.array([[0, 0, 1], [0, np.sin(a), np.cos(a)],
                           [0, -np.sin(a), np.cos(a)], [np.sin(a), 0, np.cos(a)],
                           [0.4, 0.35, np.sqrt(1 - 0.4**2 - 0.35**2)]])
    head = np.eye(4)
    head[:3, :3] = [[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]]
    head[:3, 3] = [0.25, 1.65, 0.1]
    _, basis = body_frame(head)
    centers = {"l": np.array([-.19, -.22, -.10]), "r": np.array([.20, -.21, -.11])}
    lengths = {"l": .54, "r": .56}
    rows, holds = [], []
    for i, direction in enumerate(directions):
        holds.append({"name": f"hold{i}", "start_s": 2*i, "end_s": 2*i+1.2,
                      "use": "fit" if i < 4 else "validate"})
        for k in range(n):
            stamp = 1000 + 2*i + k/40
            # Rigid tracking-origin jitter preserves exact relative geometry but
            # proves fresh float32 head samples even in deterministic fixtures.
            tracked_head = head.copy()
            tracked_head[:3, 3] = (head[:3, 3] + rng.normal(0, 1e-6, 3)).astype(np.float32)
            payload = {"sequence": len(rows), "timestamp": stamp,
                       "head": {"trackingState": "normal", "transform": {"matrix": tracked_head.ravel().tolist()}}}
            for s, key, sign in (("l", "leftHand", -1), ("r", "rightHand", 1)):
                d = direction.copy()
                d[0] *= sign
                wrist = tracked_head[:3, 3] + basis @ (centers[s] + lengths[s] * d)
                elbow = wrist - basis @ (d * .25)
                anchor = np.eye(4)
                anchor[:3, 3] = wrist + rng.normal(0, noise, 3)
                elbow_local = np.eye(4)
                elbow_local[:3, 3] = elbow - anchor[:3, 3]
                payload[key] = {"isTracked": True, "anchorTransform": {"matrix": anchor.ravel().tolist()},
                                "joints": [{"joint": "wrist", "isTracked": True,
                                            "localTransform": {"matrix": np.eye(4).ravel().tolist()}},
                                           {"joint": "forearm_arm", "isTracked": True,
                                            "localTransform": {"matrix": elbow_local.ravel().tolist()}}]}
            rows.append({"rt": stamp, "p": payload})
    return rows, holds, centers, lengths


def assess(rows, holds, **kw):
    return assess_operator_capture(rows, holds, user="synthetic_test_only",
                                   operator_confirmed_straight_arms=True, **kw)


@pytest.mark.parametrize("yaw", [0, 1.3, -2.7])
def test_identifiable_capture_recovers_nonzero_shoulder_depth_and_full_reach(yaw):
    rows, holds, centers, lengths = capture(yaw=yaw)
    report = assess(rows, holds)
    assert report["errors"] == []
    assert report["status"] == "candidate_for_review"
    profile = report["candidate_profile"]
    assert profile["quality"]["review_required"] is True
    for i, side in enumerate(("l", "r")):
        np.testing.assert_allclose(profile["shoulder_offset_body"][side], centers[side], atol=.004)
        assert profile["arm_length_lr"][i] == pytest.approx(lengths[side], abs=.004)
        assert report["arms"][side]["validation"][0]["rho"] == pytest.approx(1, abs=.01)
    shoulder = ShoulderEstimator()
    shoulder.set_profile(arm_length_lr=profile["arm_length_lr"],
                         shoulder_offset_body=profile["shoulder_offset_body"])
    h = np.asarray(rows[0]["p"]["head"]["transform"]["matrix"]).reshape(4, 4)
    shoulder.update(h, None, {})
    for i, side in enumerate(("left", "right")):
        assert shoulder.arm_length(side) == profile["arm_length_lr"][i]


def test_four_30_degree_holds_are_uncertain_despite_exact_zero_residuals():
    rows, holds, _, _ = capture(angle=30, noise=0)
    report = assess(rows, holds)
    assert report["candidate_profile"] is None
    assert any("too uncertain" in s for s in report["errors"])
    for arm in report["arms"].values():
        assert max(abs(x) for x in arm["fit_residuals_m"]) < 1e-10
        assert arm["linearized_center_95_m"][2] > .08


def test_validation_is_not_used_to_make_the_fit_look_consistent():
    rows, holds, _, _ = capture()
    baseline = assess(rows, holds)
    for row in rows[-45:]:
        for key in ("leftHand", "rightHand"):
            m = row["p"][key]["anchorTransform"]["matrix"]
            m[11] += .10
    report = assess(rows, holds)
    assert report["candidate_profile"] is None
    assert any("independent straight reach" in s for s in report["errors"])
    for s in ("l", "r"):
        np.testing.assert_array_equal(report["arms"][s]["center_body_m"], baseline["arms"][s]["center_body_m"])


def test_repeated_fit_pose_cannot_substitute_for_independent_validation():
    rows, holds, _, _ = capture(noise=0)
    for row, first in zip(rows[-45:], rows[:45]):
        for key in ("leftHand", "rightHand"):
            row["p"][key] = deepcopy(first["p"][key])
    report = assess(rows, holds)
    assert report["candidate_profile"] is None
    assert any("unused arm direction" in s for s in report["errors"])


@pytest.mark.parametrize("fault", ["head_only", "untracked_wrist", "duplicate"])
def test_tracking_and_freshness_must_be_real(fault):
    rows, holds, _, _ = capture()
    for row in rows:
        p = row["p"]
        if fault == "head_only":
            p.pop("leftHand"); p.pop("rightHand")
        elif fault == "untracked_wrist":
            p["leftHand"]["joints"][0]["isTracked"] = False
        else:
            p["sequence"] = 0
    report = assess(rows, holds)
    assert report["candidate_profile"] is None
    assert any("insufficient fresh" in s for s in report["errors"])


@pytest.mark.parametrize("fault", ["head_translation", "head_glance", "wrist_motion"])
def test_moving_operator_is_rejected(fault):
    rows, holds, _, _ = capture()
    for i, row in enumerate(rows):
        if fault == "head_translation" and i >= 90:
            row["p"]["head"]["transform"]["matrix"][3] += .05
        elif fault == "head_glance" and i >= 90:
            m = np.asarray(row["p"]["head"]["transform"]["matrix"]).reshape(4, 4)
            m[:3, :3] = [[.5, 0, np.sqrt(.75)], [0, 1, 0], [-np.sqrt(.75), 0, .5]]
            row["p"]["head"]["transform"]["matrix"] = m.ravel().tolist()
        elif fault == "wrist_motion":
            row["p"]["leftHand"]["anchorTransform"]["matrix"][3] += .04*np.sin(i)
    report = assess(rows, holds)
    assert report["candidate_profile"] is None
    assert any("head moved" in s or "not held still" in s for s in report["errors"])


def test_unlabeled_or_unconfirmed_real_motion_cannot_become_a_profile():
    rows, holds, _, _ = capture()
    report = assess_operator_capture(rows, holds)
    assert report["candidate_profile"] is None
    assert any("not confirmed" in s for s in report["errors"])
    report = assess(rows, holds[:4])
    assert report["candidate_profile"] is None
    assert any("separate fully straight validation" in s for s in report["errors"])


def test_coplanar_wrist_positions_do_not_identify_a_shoulder():
    with pytest.raises(ValueError, match="coplanar"):
        fit_wrist_sphere([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])


def test_robot_and_lab_recordings_have_identical_measurements_and_sources_do_not_mix():
    rows, holds, _, _ = capture()
    robot = [{"t": r["rt"], "cid": "vp-1", "msg": {"type": "teleop_frame", "payload": r["p"]}} for r in rows]
    lab, remote = assess(rows, holds), assess(robot, holds)
    assert lab["arms"] == remote["arms"]
    mixed = robot + [{**r, "cid": "vp-2"} for r in robot]
    with pytest.raises(ValueError, match="select one"):
        assess(mixed, holds)
    chosen = assess(mixed, holds, source="vp-1")
    assert chosen["arms"] == remote["arms"]


def test_overlapping_labels_are_rejected():
    rows, holds, _, _ = capture()
    holds[1]["start_s"] = .5
    with pytest.raises(ValueError, match="disjoint"):
        assess(rows, holds)


def neutral_capture():
    rows, holds, centers, lengths = capture(noise=0)
    rows, holds = rows[:135], holds[:3]
    for i, name in enumerate(("neutral", "forward", "diagonal")):
        holds[i]["name"] = name
        holds[i]["use"] = "validate" if i == 2 else "fit"
        for row in rows[45*i:45*(i+1)]:
            h = np.array(row["p"]["head"]["transform"]["matrix"]).reshape(4, 4)
            _, b = body_frame(h)
            for s, key in (("l", "leftHand"), ("r", "rightHand")):
                upper = .30 if s == "l" else .31
                forearm = lengths[s] - upper
                if i == 0:
                    elbow = centers[s] + [0, -upper, 0]
                    wrist = elbow + [0, 0, forearm]
                else:
                    direction = np.array([0, 0, 1.]) if i == 1 else np.array([0, -1, 1.])/np.sqrt(2)
                    elbow = centers[s] + direction*upper
                    wrist = centers[s] + direction*lengths[s]
                hand = row["p"][key]
                anchor = np.eye(4)
                anchor[:3, 3] = h[:3, 3] + b @ wrist
                local = np.eye(4)
                local[:3, 3] = b @ (elbow-wrist)
                hand["anchorTransform"]["matrix"] = anchor.ravel().tolist()
                hand["joints"][1]["localTransform"]["matrix"] = local.ravel().tolist()
    return rows, holds, centers, lengths


def assess_neutral(rows, holds, **kw):
    return assess_neutral_forward_capture(rows, holds, neutral="neutral", forward="forward",
                                          operator_confirmed_poses=True, **kw)


def test_neutral_forward_and_unused_diagonal_recover_geometry_but_never_export_profile():
    rows, holds, centers, lengths = neutral_capture()
    report = assess_neutral(rows, holds)
    assert report["errors"] == []
    assert report["status"] == "consistent_under_pose_assumptions"
    assert report["candidate_profile"] is None, "pose assumptions require review, never automatic profile export"
    for i, side in enumerate(("l", "r")):
        np.testing.assert_allclose(report["review_measurements"]["shoulder_offset_body"][side], centers[side], atol=1e-8)
        assert report["review_measurements"]["arm_length_lr"][i] == pytest.approx(lengths[side])
        assert report["arms"][side]["validation"][0]["rho"] == pytest.approx(1)


@pytest.mark.parametrize("fault", ["no_confirmation", "missing_elbow", "bad_forward", "bent_diagonal", "same_validation"])
def test_neutral_forward_does_not_accept_unverified_pose_or_proxy_geometry(fault):
    rows, holds, _, _ = neutral_capture()
    if fault == "missing_elbow":
        for r in rows:
            r["p"]["rightHand"]["joints"][1]["isTracked"] = False
    elif fault == "bad_forward":
        for r in rows[45:90]:
            r["p"]["rightHand"]["anchorTransform"]["matrix"][3] += .1
    elif fault == "bent_diagonal":
        for r in rows[90:]:
            r["p"]["rightHand"]["anchorTransform"]["matrix"][11] += .1
    elif fault == "same_validation":
        for r, forward in zip(rows[90:], rows[45:90]):
            for key in ("leftHand", "rightHand"):
                r["p"][key] = deepcopy(forward["p"][key])
    if fault == "no_confirmation":
        report = assess_neutral_forward_capture(rows, holds, neutral="neutral", forward="forward")
    else:
        report = assess_neutral(rows, holds)
    assert report["status"] == "rejected"
    assert report["review_measurements"] is None
    assert report["candidate_profile"] is None
    assert report["errors"]


@pytest.mark.parametrize("mode", ["sphere", "neutral"])
def test_a_frozen_normal_head_is_not_a_valid_capture(mode):
    rows, holds, _, _ = capture() if mode == "sphere" else neutral_capture()
    frozen = deepcopy(rows[0]["p"]["head"])
    for row in rows:
        row["p"]["head"] = deepcopy(frozen)
    report = assess(rows, holds) if mode == "sphere" else assess_neutral(rows, holds)
    assert report["candidate_profile"] is None
    assert any("head transform never updated" in e for e in report["errors"])


def test_explicit_disabled_head_marker_is_honored_when_present():
    rows, holds, _, _ = capture()
    for row in rows:
        row["p"]["headEnabled"] = False
    report = assess(rows, holds)
    assert report["candidate_profile"] is None
    assert any("insufficient fresh" in e for e in report["errors"])
