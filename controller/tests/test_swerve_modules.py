"""Per-module swerve model (mabel/motion_model/swerve_modules.py).

Operator request (Retarget Lab, 2026-10-15): "make sure the steering motor and the
drive motor max speed and torque are considered and within the safety margin — the
steer motor can go full speed — it is not fully holonomic, the steer takes time to
settle into the right orientation". These pin:
  * the steer optimiser is the controller's (swerve.optimize_steering) exactly;
  * a sideways command after driving forward is NOT realised at once (non-holonomic);
  * no wheel is ever asked past its safe speed (desaturation keeps the direction);
  * wheel acceleration stays inside the torque-derived cap;
  * the NOVA profile's geometry is NOVA's driver geometry, in MABEL's chassis frame.
"""
import math
from pathlib import Path

import numpy as np
import pytest

from mabel.motion_model import swerve_modules as S
from mabel.motion_model.swerve_modules import PROFILES, SwerveModules, check_limits

DT = 0.01


def test_steer_optimiser_matches_the_controller():
    sw = pytest.importorskip("mabel.whole_body_controller.base_controller.swerve")
    rng = np.random.default_rng(0)
    for a, b in rng.uniform(-7, 7, (2000, 2)):
        assert np.allclose(S.optimize_steering(a, b), sw.optimize_steering(a, b))


@pytest.mark.parametrize("profile", ["mabel", "nova"])
def test_sideways_after_forward_is_not_instant(profile):
    sm = SwerveModules(profile=profile)
    for _ in range(100):
        sm.step((0.3, 0.0, 0.0), DT)
    vy = [sm.step((0.0, 0.3, 0.0), DT)[1] for _ in range(60)]
    t63 = np.argmax(np.array(vy) > 0.63 * 0.3) * DT
    assert t63 >= 0.05, f"{profile}: reached 63% in {t63 * 1e3:.0f} ms — treated as holonomic"
    assert vy[-1] == pytest.approx(0.3, abs=0.01)                     # and it does get there


def test_hardware_steer_is_slower_than_the_mjcf_servo():
    """The measured MAXSwerve quarter turn (~170 ms) vs the MJCF servo: same command,
    the hardware profile must lag the sim one."""
    def t63(profile):
        sm = SwerveModules(profile=profile)
        for _ in range(100):
            sm.step((0.3, 0.0, 0.0), DT)
        vy = np.array([sm.step((0.0, 0.3, 0.0), DT)[1] for _ in range(60)])
        return np.argmax(vy > 0.189) * DT
    assert t63("mabel") > t63("mabel_mjcf")


@pytest.mark.parametrize("profile", ["mabel", "nova"])
def test_no_wheel_past_its_safe_speed(profile):
    rng = np.random.default_rng(1)
    cmds = np.c_[rng.uniform(-1.5, 1.5, (400, 2)), rng.uniform(-3, 3, 400)]
    out = check_limits(np.repeat(cmds, 20, axis=0), DT, **PROFILES[profile])
    cap = PROFILES[profile].get("wheel_v", S.WHEEL_V_SAFE)
    assert out["peak"]["wheel_v"] <= cap + 1e-9
    assert out["margin"]["wheel_v_collapse"] > 0 if profile == "mabel" else True
    assert out["peak"]["wheel_a"] <= PROFILES[profile].get("wheel_a", S.WHEEL_A_SAFE) + 1e-6
    assert out["peak"]["steer_rate"] <= S.STEER_RATE_MAX + 1e-9


def test_desaturation_keeps_the_direction():
    sm = SwerveModules()
    for _ in range(300):
        tw = sm.step((0.8, 0.0, 2.5), DT)                              # the outer wheel would need > 1.2 m/s
    assert sm.last["desat"] < 1.0
    assert tw[2] / tw[0] == pytest.approx(2.5 / 0.8, rel=0.02)


def test_nova_geometry_is_novas_driver():
    nova = Path(__file__).resolve().parents[3] / "NOVA" / "hardware_bridge"
    if not nova.exists():
        pytest.skip("NOVA checkout not next to MABEL")
    import sys
    sys.path.insert(0, str(nova))
    from nova_hw.subsystems.base import module_positions
    pos = module_positions(0.34, 0.36)
    want = np.array([[-pos[k][0], -pos[k][1]] for k in ("fl", "fr", "b")])   # body → MABEL chassis frame
    assert np.allclose(PROFILES["nova"]["positions"], want)


def test_limit_constants():
    assert S.WHEEL_V_FREE == pytest.approx(4.81, abs=0.01)             # NEO 14T, 3 in wheel
    assert S.STEER_RATE_FREE == pytest.approx(24.8, abs=0.1)           # NEO 550 / 46.42
    assert S.WHEEL_V_SAFE <= 0.8 * S.WHEEL_V_COLLAPSE + 1e-9
    assert S.WHEEL_TAU_SAFE == pytest.approx(0.5 * S.WHEEL_TAU_MAX)
