"""Pure numerics of the validated ps5 swerve IK (mabel_hw.drivers.base_ik).
These pin down the frame conventions the whole base path rests on."""
import math

import numpy as np
import pytest

from mabel_hw.drivers import base_ik


ZERO = np.zeros(3)


def test_pure_forward_equal_modules():
    # NOTE steer is in ROTATIONS [0, 1) on this wire (the Teensy consumes
    # rotations and does its own quarter-turn steer optimization onboard).
    drive, steer = base_ik.inverse_kinematics(0.5, 0.0, 0.0, ZERO)
    assert np.allclose(drive, drive[0]) and abs(drive[0]) == pytest.approx(0.5)
    assert np.allclose(steer, 0.0)        # +x -> heading 0 for every module


def test_pure_rotation_tangential():
    drive, steer = base_ik.inverse_kinematics(0.0, 0.0, 1.0, ZERO)
    # every module's speed = |omega| * its lever arm from the rotation centre
    arms = np.linalg.norm(base_ik.MODULE_POS, axis=1)
    assert np.allclose(np.abs(drive), arms, rtol=1e-6)


def test_zero_twist_keeps_prev_steer():
    prev = np.array([0.3, -0.2, 0.1])
    drive, steer = base_ik.inverse_kinematics(0.0, 0.0, 0.0, prev)
    assert np.allclose(drive, 0.0)
    assert np.allclose(steer, prev)          # no speed -> hold last heading


def test_steer_stays_in_unit_rotation():
    rng = np.random.default_rng(7)
    prev = ZERO
    for _ in range(200):
        vx, vy = rng.uniform(-1, 1, 2)
        wz = rng.uniform(-2, 2)
        drive, steer = base_ik.inverse_kinematics(vx, vy, wz, prev)
        s = np.asarray(steer)
        assert np.all((s >= 0.0) & (s < 1.0))     # rotations wire convention
        assert np.all(np.asarray(drive) >= 0.0)   # magnitude; sign lives in steer
        prev = steer


def test_heading_matches_module_velocity():
    drive, steer = base_ik.inverse_kinematics(0.0, 0.4, 0.0, ZERO)   # pure +y
    assert np.allclose(steer, 0.25)               # atan2(vy, 0)/2pi = quarter turn
    assert np.allclose(drive, 0.4)


def test_apply_calibration_offsets_and_polarity():
    drive = np.array([0.5, 0.5, 0.5])
    steer = np.array([0.0, 0.5, 0.95])
    d2, s2 = base_ik.apply_calibration(drive, steer)
    assert np.allclose(s2, (steer + base_ik.STEER_OFFSET) % 1.0)   # rotations wrap
    assert np.allclose(d2, drive * base_ik.DRIVE_DIR)
