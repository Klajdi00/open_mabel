"""Hands stack, hardware-free: the OrcaHand joint-space HAL on its mock bus, and
the HandBridgeService RPC layer driving it end-to-end (open/state/write/torque)
— the same surface the hand studio GUI uses over :8779."""
import math

import pytest

from mabel_hw.drivers.orcahand import OrcaHand
from mabel_hw.rpc.hand_service import HandBridgeService, HandBridgeError


@pytest.fixture()
def hand():
    h = OrcaHand.mock("right")
    h.connect(enable_torque=True, scan=True)
    yield h
    h.disconnect()


def test_joint_roundtrip_and_clamp(hand):
    names = hand.joint_names
    assert "index_mcp" in names and len(names) >= 16
    hand.set_joint_positions({"index_mcp": 0.5})
    pos = hand.get_joint_positions()
    assert pos["index_mcp"] == pytest.approx(0.5, abs=0.02)
    # far beyond the joint's angle range must clamp, not wrap or raise
    hand.set_joint_positions({"index_mcp": 10.0})
    pos = hand.get_joint_positions()
    j = hand.config.joint("index_mcp")
    assert pos["index_mcp"] <= j.angle_max + 0.05
    assert not math.isnan(pos["index_mcp"])


def test_state_snapshot(hand):
    st = hand.get_state()
    for field in ("position", "velocity", "effort", "temperature", "voltage"):
        assert set(getattr(st, field)) == set(hand.joint_names)


def test_rpc_service_mock_end_to_end():
    svc = HandBridgeService()
    try:
        cfg = svc.op("open", {"side": "right", "port": "mock"})
        assert cfg["side"] == "right" and cfg["joints"]
        assert svc.op("hello", {})["sides_open"] == ["right"]
        st = svc.op("state", {"side": "right"})
        assert st["raw_position"]
        svc.op("set_torque", {"side": "right", "on": True})
        mid = cfg["joints"][0]["id"]
        assert svc.op("ping", {"side": "right", "mid": mid}) is True
        svc.op("sync_write", {"side": "right", "targets": {str(mid): 2048}})
        pos = svc.op("read_positions", {"side": "right", "ids": [mid]})
        assert str(mid) in pos
        svc.op("close", {"side": "right"})
        assert svc.op("hello", {})["sides_open"] == []
    finally:
        svc.shutdown()


def test_rpc_unknown_op_and_closed_side():
    svc = HandBridgeService()
    with pytest.raises(HandBridgeError):
        svc.op("no_such_op", {})
    with pytest.raises(HandBridgeError):
        svc.op("set_torque", {"side": "left", "on": True})   # side never opened


def test_calib_sync_paths_order_and_dedup(tmp_path):
    """The save-calibration fan-out: the hand's OWN calibration_path must come
    first (it is what the live driver reloads), followed by the repo sync
    targets; no duplicates; only existing parents."""
    svc = HandBridgeService()
    try:
        svc.op("open", {"side": "right", "port": "mock"})
        oh = svc._hand("right")
        paths = svc._calib_sync_paths(oh, "right")
        assert paths, "sync list is empty"
        assert len(set(paths)) == len(paths)
        if oh.config.calibration_path is not None:
            assert paths[0] == oh.config.calibration_path.resolve()
        joined = [str(p) for p in paths]
        # the bridge mirror + the new vendored home are sync targets; the
        # pre-move path must be gone
        assert any("calibration_data/hands" in s for s in joined)
        assert any("mabel_hw/drivers/orcahand/configs" in s for s in joined)
        assert not any("/hands/orcahand/configs" in s for s in joined)
    finally:
        svc.shutdown()


# ── measured torque: Present_Current, not the PWM-duty "load" ────────────────
# The servo's `Present_Load` is a signed DRIVE DUTY (‰ of PWM). It saturates at
# ±1 on a stall at any supply voltage, so it can never become N·m — yet it was
# being published in JointState.effort, a newton-metre field, which put the hands
# on the same plot axis as the arms' true N·m. The real signal is Present_Current
# (0x45): torque = Kt · I.

def test_load_is_duty_not_torque():
    """load_to_effort stays a [-1, 1] duty, and saturates — that is its job."""
    from mabel_hw.drivers.orcahand.joints import Joint
    j = Joint(name="index_mcp", motor_id=1, invert=False, angle_min=-0.35,
              angle_max=1.66, raw_min=1000, raw_max=3000, center=2048)
    assert j.load_to_effort(1000) == pytest.approx(1.0)     # stalled: all it has
    assert j.load_to_effort(-500) == pytest.approx(-0.5)


def test_feedback_decodes_present_current_and_survives_a_short_block():
    """The 15-byte block yields a SIGNED current in amps; the classic 8-byte
    block must NOT invent one (absent ⇒ 'not measured', never a fake zero)."""
    from mabel_hw.drivers.orcahand import constants as C
    from mabel_hw.drivers.orcahand.bus import FeetechBus

    base = bytes([0x00, 0x08,      # position 2048
                  0x0A, 0x00,      # speed +10
                  0x64, 0x04,      # load: bit10 set -> -100
                  120, 31])        # 12.0 V, 31 C

    short = FeetechBus._decode_feedback(base)
    assert short["position"] == 2048 and short["load"] == -100
    assert "current" not in short, "an 8-byte read must not fabricate a current"

    ext = bytearray(base) + bytes(7)
    ext[13], ext[14] = 0x64, 0x00                    # +100 LSB
    fb = FeetechBus._decode_feedback(bytes(ext))
    assert fb["current"] == pytest.approx(100 * C.CURRENT_LSB_A)
    ext[14] = 0x80                                   # bit15 = direction
    fb = FeetechBus._decode_feedback(bytes(ext))
    assert fb["current"] == pytest.approx(-100 * C.CURRENT_LSB_A)
    # a corrupt position still nukes the whole frame, extended or not
    bad = bytearray(ext); bad[1] = 0xFF
    assert FeetechBus._decode_feedback(bytes(bad)) is None


def test_current_to_torque_follows_the_joint_direction():
    """τ = Kt·I, flipped by `invert` exactly like position/velocity — so hand
    torque signs agree with the arms (subsystems/joint_group.state: eff = dir*tau)."""
    from mabel_hw.drivers.orcahand import constants as C
    from mabel_hw.drivers.orcahand.joints import Joint
    mk = lambda inv: Joint(name="j", motor_id=1, invert=inv, angle_min=-0.35,
                           angle_max=1.66, raw_min=1000, raw_max=3000, center=2048)
    assert mk(False).current_to_torque(0.65) == pytest.approx(0.65 * C.TORQUE_NM_PER_AMP)
    assert mk(True).current_to_torque(0.65) == pytest.approx(-0.65 * C.TORQUE_NM_PER_AMP)


# ── studio master calibration: the single-source unit-gain map ───────────────
# The hand driver reads the SAME numbers the setup studio's 3D viewport uses
# (mabel.json hand/orca-<side>/calib) and applies the SAME unit-gain map
# (angle = model_mid + dir·wrap_π(raw − raw_mid)), so a real_run render can never
# disagree with the studio. These tests pin that math and the wiring.

def _studio_forward(c, raw_tick):
    import math
    two_pi = 2 * math.pi
    t2r = two_pi / 4096.0
    wrap = lambda x: (x + math.pi) % two_pi - math.pi
    rmid = 0.5 * (c["raw_min"] + c["raw_max"])
    return c["model_mid"] + c.get("dir", 1) * wrap(raw_tick * t2r - rmid)


def test_model_calib_joint_matches_studio_forward_and_inverse():
    """A Joint carrying studio calib maps EXACTLY like hand_bridge, both ways,
    and the inverse is clamped to the swept band (no tendon past an end-stop)."""
    from mabel_hw.drivers.orcahand.joints import Joint
    c = {"raw_min": 1.4879614, "raw_max": 3.508214, "dir": 1, "model_mid": 0.767945}
    j = Joint(name="index_pip", motor_id=8, invert=(c["dir"] < 0),
              angle_min=-0.3491, angle_max=1.8850, raw_min=0, raw_max=0, center=0,
              model_mid=c["model_mid"], raw_min_rad=c["raw_min"], raw_max_rad=c["raw_max"])
    assert j._has_model_calib
    for raw in range(0, 4096, 13):
        assert j.raw_to_angle(raw) == pytest.approx(_studio_forward(c, raw), abs=1e-9)
    # inverse round-trips within one tick across the reachable band …
    half = 0.5 * (c["raw_max"] - c["raw_min"])
    for k in range(51):
        tgt = c["model_mid"] - half + 2 * half * k / 50
        assert j.raw_to_angle(j.angle_to_raw(tgt)) == pytest.approx(tgt, abs=2e-3)
    # … and a target beyond the band clamps to the band edge, never overshoots.
    assert j.clamp_angle(10.0) == pytest.approx(c["model_mid"] + half, abs=1e-9)
    assert j.raw_to_angle(j.angle_to_raw(10.0)) == pytest.approx(c["model_mid"] + half, abs=2e-3)


def test_model_calib_entry_validation_rejects_bad_and_negative_span():
    """_model_joint_kwargs mirrors the studio's guard: missing keys or a
    non-positive swept span → fall back to the tick map (None), never guess."""
    from mabel_hw.drivers.orcahand.config import _model_joint_kwargs
    assert _model_joint_kwargs(None) is None
    assert _model_joint_kwargs({"raw_min": 1.0, "raw_max": 2.0}) is None  # no model_mid
    assert _model_joint_kwargs({"raw_min": 2.0, "raw_max": 1.0, "model_mid": 0.0}) is None
    ok = _model_joint_kwargs({"raw_min": 1.0, "raw_max": 2.0, "model_mid": 0.5, "dir": -1})
    assert ok == {"raw_min_rad": 1.0, "raw_max_rad": 2.0, "model_mid": 0.5, "invert": True}


def test_load_uses_studio_master_when_present(monkeypatch):
    """HandConfig.load pulls the joint map from the studio master (mabel.json)
    and reports it as the calibration source; the mock stays on the tick map."""
    import mabel_hw.drivers.orcahand.config as cfgmod
    from pathlib import Path
    fake = {"index_mcp": {"raw_min": 1.3, "raw_max": 3.3, "dir": -1, "model_mid": 0.65}}
    monkeypatch.setattr(cfgmod, "_studio_hand_calib",
                        lambda side: (fake, Path("/tmp/mabel.json")))
    cfg = cfgmod.HandConfig.load(model="right", use_model_calib=True)
    j = cfg.joint("index_mcp")
    assert j._has_model_calib and j.invert is True
    assert cfg.calibration_path == Path("/tmp/mabel.json")
    # a joint absent from the master falls back to the tick map (identity flag off)
    assert not cfg.joint("wrist")._has_model_calib
    # the mock never consults the master — deterministic tick map
    assert not OrcaHand.mock("right").config.joint("index_mcp")._has_model_calib


def test_get_state_exposes_current_and_torque():
    """HandState carries the measured signals; a servo with no current register
    leaves BOTH dicts EMPTY so a caller can tell 'no load' from 'no sensor'."""
    oh = OrcaHand.mock(model="right")
    oh.connect()
    try:
        st = oh.get_state()
        assert st.position, "mock hand reported no joints"
        # the mock bus serves a current, so torque must be derived for every joint
        assert set(st.torque) == set(st.position)
        assert set(st.current) == set(st.position)
        for n in st.torque:
            assert st.torque[n] == pytest.approx(0.0)   # unloaded mock motor
    finally:
        oh.disconnect()
