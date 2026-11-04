"""Offline validation of the body/hand calibration inverse problem.

Synthesises calibration frames for operators of known dimensions (the forward
model in calibration.py), runs them back through the measurement + profile-build
pipeline, and asserts the recovered anthropometry matches ground truth. Run with::

    cd controller && python -m pytest tests/test_calibration.py -q
    # or, without pytest:
    python tests/test_calibration.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mabel import config as C
from mabel.calibration import (
    OperatorProfile, BodyCalibrator, body_frame,
    measure_tpose, measure_arms_forward, measure_hands, build_profile,
    synthesize_pose, POSE_TPOSE, POSE_FORWARD, POSE_HANDS,
)

HAND = {"palm_len": 0.090,
        "fingers": {"thumb": 0.062, "index": 0.082, "middle": 0.094,
                    "ring": 0.088, "pinky": 0.071},
        "pinch_open": 0.090, "pinch_close": 0.018}


def _profile_for(height, yaw=0.0, noise=0.0, seed=0):
    kw = dict(height=height, head_yaw=yaw, noise=noise, seed=seed)
    t = measure_tpose(synthesize_pose(POSE_TPOSE, **kw))
    f = measure_arms_forward(synthesize_pose(POSE_FORWARD, **kw))
    h = measure_hands(synthesize_pose(POSE_HANDS, hand=HAND, **kw))
    return build_profile(t, f, h, user=f"H{height}")


def test_noise_free_roundtrip():
    """Exact recovery (no noise) of arm length, shoulder width, neck height, H."""
    for H in (1.55, 1.75, 1.95):
        p = _profile_for(H)
        assert abs(p.arm_length_m - C.ANTHRO_ARM_RATIO * H) < 1e-3, p.arm_length_m
        assert abs(p.shoulder_width_m - C.ANTHRO_BIACROMIAL * H) < 1e-3
        assert abs(p.neck_height_m - C.ANTHRO_ACROMION_BELOW_EYE * H) < 1e-3
        assert abs(p.height_m - H) < 5e-3, (H, p.height_m)


def test_yaw_invariance():
    """Measurements are invariant to where the operator faces (body frame)."""
    base = _profile_for(1.75, yaw=0.0)
    for yaw in (0.6, -1.2, 2.5):
        p = _profile_for(1.75, yaw=yaw)
        assert abs(p.arm_length_m - base.arm_length_m) < 2e-3, yaw
        assert abs(p.shoulder_width_m - base.shoulder_width_m) < 2e-3, yaw


def test_finger_lengths_recovered():
    """Per-finger metric lengths match the synthetic hand within tolerance."""
    p = _profile_for(1.75)
    for side in ("left", "right"):
        fl = p.finger_len_m.get(side, {})
        for f, truth in HAND["fingers"].items():
            assert abs(fl[f] - truth) < 1e-3, (side, f, fl.get(f), truth)
        assert abs(p.pinch_open_m[side] - HAND["pinch_open"]) < 1e-6
        assert abs(p.pinch_close_m[side] - HAND["pinch_close"]) < 1e-6


def test_gamma_clamped_and_scaled():
    """gamma = robot_reach / human_reach, clamped to the safe band."""
    p = _profile_for(1.75)
    robot_reach = 0.43
    g = p.gamma(robot_reach)
    assert abs(g - robot_reach / p.arm_length_m) < 1e-6
    # A tiny robot reach would push gamma below the floor -> clamped.
    assert p.gamma(0.05) == C.RETARGET_GAMMA_MIN


def test_shoulder_world_offset():
    """Shoulder world centre sits ~neck_height below and ~half-width lateral of head."""
    p = _profile_for(1.75)
    head = np.eye(4)
    head[:3, 3] = [0.3, 1.62, -0.5]
    _, B = body_frame(head)
    for side, sign in (("l", -1.0), ("r", +1.0)):
        c = p.shoulder_world(side, head)
        b = B.T @ (c - head[:3, 3])
        assert abs(b[0] - sign * p.shoulder_width_m / 2.0) < 1e-6
        assert abs(b[1] + p.neck_height_m) < 1e-6        # neck_height below the head


def test_noisy_robustness():
    """With realistic tracker noise (3 mm), averaged measurements stay close."""
    errs = []
    for seed in range(20):
        p = _profile_for(1.75, noise=0.003, seed=seed)
        errs.append(abs(p.arm_length_m - C.ANTHRO_ARM_RATIO * 1.75))
    assert np.mean(errs) < 0.01, np.mean(errs)


def test_streaming_state_machine():
    """BodyCalibrator captures all three poses from a stitched frame stream."""
    cal = BodyCalibrator(user="stream")
    t = 0.0
    streams = [
        synthesize_pose(POSE_TPOSE, height=1.8, n=40),
        synthesize_pose(POSE_FORWARD, height=1.8, n=40),
        synthesize_pose(POSE_HANDS, height=1.8, hand=HAND, n=40),
    ]
    for stream in streams:
        for pose in stream:
            cal.observe(pose, now=t)
            t += 1.0 / 30.0
    assert cal.done, (cal.pose_idx, cal.current_pose)
    assert abs(cal.profile.arm_length_m - C.ANTHRO_ARM_RATIO * 1.8) < 5e-3


def test_streaming_discrete_pose_jumps():
    """Regression: each pose fed only until captured, then a hard jump to the next
    pose (no observed motion transient, empty buffer). The re-arm gate must release
    on a NEW distinct still pose, not on seeing the transient. (This deadlocked the
    real-time facade path before the novelty-based gate.)"""
    cal = BodyCalibrator(user="discrete")
    t = 0.0
    for pose in (POSE_TPOSE, POSE_FORWARD, POSE_HANDS):
        for fr in synthesize_pose(pose, height=1.7, hand=HAND, n=400):
            captured = cal.observe(fr, now=t)
            t += 1.0 / 60.0
            st = cal.current_pose
            if cal.done or (captured is not None):     # advance the instant it captures
                break
    assert cal.done, (cal.pose_idx, cal.current_pose)
    assert abs(cal.profile.arm_length_m - C.ANTHRO_ARM_RATIO * 1.7) < 5e-3


def test_json_roundtrip(tmp_path=None):
    """Profile serialises and reloads identically."""
    p = _profile_for(1.75)
    text = p.to_json()
    q = OperatorProfile.from_json(text)
    assert abs(q.arm_length_m - p.arm_length_m) < 1e-9
    assert q.finger_len_m["left"]["index"] == p.finger_len_m["left"]["index"]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    fails = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            fails += 1
            print(f"FAIL {fn.__name__}: {e}")
    print(f"\n{len(fns) - fails}/{len(fns)} passed")
    sys.exit(1 if fails else 0)
