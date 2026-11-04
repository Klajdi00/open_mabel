"""The DaMiao ±p_max continuous-unwrap for multi-turn arm joints (l_j4).

The encoder reports position wrapped into [-p_max, +p_max]; near the +12.5 ceiling
(where l_j4 lives) a crossing makes the raw jump ~2*p_max = 25 rad — NOT a whole
number of turns, so the 2π fold used elsewhere can never reconcile it. _to_model must
rebuild a CONTINUOUS position so the mapped angle stays smooth through the wrap.
"""
from mabel_hw.subsystems.joint_group import JointGroupSubsystem
from mabel_hw.calibration.schema import JointCalib


class _FakeDrv:
    names = ("l_j4",)


def _sub(pmax):
    cfg = {"p_max": pmax, "calib_slice": None}
    s = JointGroupSubsystem("larm", cfg, {}, driver=_FakeDrv(), store=None)
    # l_j4-style in-band window hard against the +12.5 ceiling (from mabel.json)
    s._calib = {"l_j4": JointCalib.from_dict(
        {"raw_min": 9.635, "raw_max": 12.128, "dir": -1, "model_mid": 0.448269}, "l_j4")}
    return s


def test_pmax_configured():
    assert _sub(12.5)._pmax == 12.5
    assert _sub(0.0)._pmax == 0.0


def test_in_band_pose_matches_2pi_map_no_regression():
    """At the live in-band pose the continuous map == the old to_model (no shift)."""
    s = _sub(12.5)
    jc = s._calib["l_j4"]
    m = s._to_model("l_j4", 9.82)
    assert abs(m - jc.to_model(9.82)) < 1e-9


def test_ceiling_crossing_stays_continuous():
    """Crossing +12.5 (raw jumps to -12.5) must move the model only by the tiny physical
    step, NOT by ~25 rad — the exact l_j4 'reads wildly out of range after a second' bug."""
    s = _sub(12.5)
    s._to_model("l_j4", 12.30)                    # seed near the ceiling
    m_before = s._to_model("l_j4", 12.49)         # just below +12.5
    m_after = s._to_model("l_j4", -12.49)         # wrapped just past +12.5 (physical +0.02)
    assert abs(m_after - m_before) < 0.1, f"wrap not de-wrapped: {m_before} -> {m_after}"


def test_single_turn_joint_unaffected():
    """p_max=0 (neck/single-turn): falls back to the plain 2π to_model, byte-identical."""
    s = _sub(0.0)
    jc = s._calib["l_j4"]
    for raw in (9.82, 12.1, 10.5):
        assert s._to_model("l_j4", raw) == jc.to_model(raw)
