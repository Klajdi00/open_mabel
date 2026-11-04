"""Measured shoulder anchors and reach use the same profile in lab and deployment."""
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from mabel.teleop_engine.engine import TeleopEngine
from mabel.teleop_engine.geometry import ShoulderEstimator
from mabel.teleop_engine.hyperparams import (
    ANTHRO_ACROMION_BELOW_EYE, ANTHRO_BIACROMIAL,
)
from mabel.whole_body_retargeting.calibration import OperatorProfile
from retarget_lab.server import config as CFG
from retarget_lab.server.worker import KinematicRetargeter


def head(yaw=0., pitch=0.):
    h = np.eye(4)
    h[:3, :3] = Rotation.from_euler("yx", [yaw, pitch], degrees=True).as_matrix()
    h[:3, 3] = [.3, 1.6, -.2]
    return h


def measured():
    # Synthetic measurements exercise the math, never an implicitly applied user profile.
    return OperatorProfile(
        user="test", arm_length_lr=(.48, .51), shoulder_width_m=.38,
        neck_height_m=.22,
        shoulder_offset_body={"l": [-.19, -.22, -.10], "r": [.19, -.22, -.11]},
        finger_len_norm={"left": {"index": [.1, .2, .3]}},
    )


def bare_engine():
    e = TeleopEngine.__new__(TeleopEngine)
    e.shoulder = ShoulderEstimator()
    e.hand = {side: SimpleNamespace(set_profile=lambda _: None) for side in ("left", "right")}
    return e


@pytest.mark.parametrize("yaw", [-2.4, 0., np.pi])
def test_measured_offsets_rotate_with_body_and_ignore_head_glances(yaw):
    e = bare_engine()
    profile = measured()
    e.apply_profile(profile)
    h = head()
    e.shoulder.update(h, yaw, {})
    right, up, fwd = e.shoulder.body_axes()
    for side, key in (("left", "l"), ("right", "r")):
        offset = profile.shoulder_offset_body[key]
        expected = h[:3, 3] + right * offset[0] + up * offset[1] + fwd * offset[2]
        np.testing.assert_allclose(e.shoulder.shoulder_world(side), expected, atol=1e-12)
    previous = {s: e.shoulder.shoulder_world(s).copy() for s in ("left", "right")}
    e.shoulder.update(head(yaw=75, pitch=-40), yaw, {})
    for side in previous:
        np.testing.assert_allclose(e.shoulder.shoulder_world(side), previous[side], atol=1e-12)


@pytest.mark.parametrize("serialized", [False, True])
def test_measured_short_arms_read_full_extension_as_one(serialized):
    profile = measured()
    e = bare_engine()
    e.apply_profile(asdict(profile) if serialized else profile)
    h = head()
    e.shoulder.update(h, np.pi, {})
    _, _, forward = e.shoulder.body_axes()
    forward = forward / np.linalg.norm(forward)
    wrists = {s: e.shoulder.shoulder_world(s) + forward * profile.arm_length_lr[i]
              for i, s in enumerate(("left", "right"))}
    e.shoulder.update(h, np.pi, wrists)
    for i, side in enumerate(("left", "right")):
        length = e.shoulder.arm_length(side)
        assert length == profile.arm_length_lr[i]
        rho = np.linalg.norm(wrists[side] - e.shoulder.shoulder_world(side)) / length
        assert rho == pytest.approx(1.)


@pytest.mark.parametrize("profile", [None, asdict(OperatorProfile())])
def test_absent_or_unmeasured_profile_keeps_original_shoulder_defaults(profile):
    e = bare_engine()
    e.apply_profile(profile)
    h = head()
    e.shoulder.update(h, np.pi, {})
    right, up, _ = e.shoulder.body_axes()
    for side, sign in (("left", -1), ("right", 1)):
        expected = (h[:3, 3] + sign * right * .5 * ANTHRO_BIACROMIAL * e.shoulder.stature
                    - up * ANTHRO_ACROMION_BELOW_EYE * e.shoulder.stature)
        np.testing.assert_array_equal(e.shoulder.shoulder_world(side), expected)


def test_unusable_measurements_do_not_poison_the_live_estimate():
    s = ShoulderEstimator()
    s.set_profile(arm_length_lr=[float("nan"), -.2], shoulder_width=float("inf"),
                  neck_drop="bad", shoulder_offset_body={"l": [0, 0, 0], "r": [0, -.2, float("nan")]})
    s.update(head(), np.pi, {})
    assert s._prof is None
    assert np.isfinite(s.shoulder_world("left")).all()
    assert np.isfinite(s.shoulder_world("right")).all()


@pytest.fixture(scope="module")
def model():
    import mujoco
    return mujoco.MjModel.from_xml_path(CFG.SCENE)


def test_lab_setting_preserves_explicit_profile_through_reset_and_can_clear_it(model):
    kr = KinematicRetargeter("shipped_v4", model=model)
    assert kr.eng.shoulder._prof is None
    profile = measured()
    settings = kr.apply_settings({"robot": "mabel", "scope": "local", "operator_profile": profile})
    expected = asdict(profile)
    assert settings["operator_profile"] == expected
    profile.shoulder_offset_body["l"][2] = 999.
    assert kr.settings["operator_profile"] == expected, "do not alias caller-owned measurements"
    kr.reset()
    assert kr.settings["operator_profile"] == expected
    assert kr.eng.shoulder.arm_length("left") == .48
    kr.eng.shoulder.update(head(), np.pi, {})
    np.testing.assert_allclose(kr.eng.shoulder._prof["offsets"]["left"], [-.19, -.22, -.10])
    kr.apply_settings({"scope": "global"})
    assert kr.eng.shoulder.arm_length("left") == .48
    kr.apply_settings({"operator_profile": None})
    assert kr.settings["operator_profile"] is None
    assert kr.eng.shoulder._prof is None
    kr.reset()
    assert kr.eng.shoulder._prof is None
