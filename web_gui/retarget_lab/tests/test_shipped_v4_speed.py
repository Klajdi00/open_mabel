"""The v4 speed change is limited to arm servo slews; previous snapshots stay reproducible."""
import copy

import pytest

from retarget_lab.server import config as CFG
from retarget_lab.server.variants import DEFAULT_ACTIVE, VARIANTS
from retarget_lab.server.worker import KinematicRetargeter


@pytest.fixture(scope="module")
def model():
    import mujoco
    return mujoco.MjModel.from_xml_path(CFG.SCENE)


def _arm_step(kr, direction, dt=.02):
    """Saturating per-joint target step, through the actual lab servo path."""
    rows = {}
    for a, qa, lo, hi, limited in kr._servo:
        key = kr._slew_key.get(qa)
        if key is None or key[0] != "arm":
            continue
        start = (lo + hi) / 2 if limited else 0.
        kr.data.qpos[qa] = start
        kr.data.ctrl[a] = hi if direction > 0 and limited else lo if limited else start + direction
        rows[qa] = (start, key[1])
    kr._servo_ideal(dt)
    return [(i, (float(kr.data.qpos[qa]) - start) / dt) for qa, (start, i) in rows.items()]


@pytest.mark.parametrize("direction", (-1, 1))
def test_v4_uses_nova_arm_slews_without_changing_other_channels(model, monkeypatch, direction):
    monkeypatch.setattr(CFG, "SERVO", "slew")
    original = copy.deepcopy(CFG.SLEWS)
    kr = KinematicRetargeter("shipped_v4", model=model)
    kr.apply_settings({"robot": "mabel", "scope": "local"})
    assert kr.slews is not CFG.SLEWS["mabel"]
    assert kr.slews["arm"] is not CFG.SLEWS["nova"]["arm"]
    assert kr.slews["arm"] == CFG.SLEWS["nova"]["arm"]
    for channel in ("torso", "neck", "lift"):
        assert kr.slews[channel] == CFG.SLEWS["mabel"][channel]
    rates = _arm_step(kr, direction)
    assert len(rates) == 14
    for joint, rate in rates:
        assert rate == pytest.approx(direction * CFG.SLEWS["nova"]["arm"][joint])
    assert CFG.SLEWS == original


@pytest.mark.parametrize("version", ("shipped_v1", "shipped_v2", "shipped_v3"))
def test_old_snapshots_keep_legacy_speed(model, monkeypatch, version):
    monkeypatch.setattr(CFG, "SERVO", "slew")
    kr = KinematicRetargeter(version, model=model)
    kr.apply_settings({"robot": "mabel"})
    assert kr.slews is CFG.SLEWS["mabel"]
    assert "arm_speed" not in VARIANTS[version]
    for _, rate in _arm_step(kr, 1):
        assert rate == pytest.approx(.8)


def test_speed_profile_survives_switch_and_reset_without_changing_ik_guards(model, monkeypatch):
    monkeypatch.setattr(CFG, "SERVO", "slew")
    original = copy.deepcopy(CFG.SLEWS)
    old = KinematicRetargeter("shipped_v3", model=model)
    old.apply_settings({"robot": "mabel", "scope": "local"})
    old_guards = [(a.trust_rate_distal, dict(a.trust_override)) for a in old.eng.arm.values()]
    from mabel.teleop_engine import collision
    old_qp_vmax = collision.ARM_QP_VMAX
    kr = KinematicRetargeter("shipped_v4", model=model)
    for robot in ("nova", "mabel"):
        kr.apply_settings({"robot": robot, "scope": "local", "ee": "gripper"})
        assert kr.slews["arm"] == CFG.SLEWS["nova"]["arm"]
        assert kr.slews["torso"] == CFG.SLEWS[robot]["torso"]
    assert [(a.trust_rate_distal, dict(a.trust_override)) for a in kr.eng.arm.values()] == old_guards
    assert collision.ARM_QP_VMAX == old_qp_vmax
    kr.reset()
    assert kr.settings["robot"] == "mabel" and kr.settings["ee"] == "gripper"
    assert kr.slews["arm"] == CFG.SLEWS["nova"]["arm"]
    assert collision.ARM_QP_VMAX == old_qp_vmax
    for joint, rate in _arm_step(kr, 1):
        assert rate == pytest.approx(CFG.SLEWS["nova"]["arm"][joint])
    kr.slews["arm"][0] = 123.
    assert CFG.SLEWS == original
    assert old.slews is CFG.SLEWS["mabel"] and old.slews["arm"] == [.8] * 7
    assert DEFAULT_ACTIVE[0] == "deployed" and VARIANTS["deployed"]["arm_speed"] == "nova"
