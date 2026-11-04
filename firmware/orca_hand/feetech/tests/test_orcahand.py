"""Hardware-free contract tests for the orcahand layer (run with pytest)."""

import json
import math

import pytest

from orcahand import HandConfig, Joint, OrcaHand
from orcahand.calibration import Calibration, MotorCalibration
from orcahand.constants import WRAP, apply_offset, remove_offset, wrapped_delta


# --------------------------------------------------------------- joint mapping
def make_joint(invert=False):
    return Joint(name="j", motor_id=1, angle_min=0.0, angle_max=1.0,
                 invert=invert, raw_min=1000, raw_max=3000, center=2000)


def test_angle_raw_roundtrip():
    j = make_joint()
    for a in (0.0, 0.25, 0.5, 0.75, 1.0):
        raw = j.angle_to_raw(a)
        assert j.raw_lo <= raw <= j.raw_hi
        assert j.raw_to_angle(raw) == pytest.approx(a, abs=1e-3)


def test_angle_clamped_to_range():
    j = make_joint()
    assert j.angle_to_raw(-5.0) == j.raw_min      # below min -> min tick
    assert j.angle_to_raw(99.0) == j.raw_max      # above max -> max tick


def test_inversion_swaps_endpoints():
    normal, inv = make_joint(False), make_joint(True)
    # angle_min maps to opposite tick ends under inversion
    assert normal.angle_to_raw(0.0) == inv.angle_to_raw(1.0)
    assert normal.angle_to_raw(1.0) == inv.angle_to_raw(0.0)


def test_percent_ignores_inversion():
    j = make_joint(invert=True)
    assert j.percent_to_raw(0) == j.raw_min
    assert j.percent_to_raw(100) == j.raw_max


# ----------------------------------------------------- single-turn wrap / offset
def test_wrapped_delta_shortest_arc():
    assert wrapped_delta(10) == 10
    assert wrapped_delta(-10) == -10
    # 4085 -> 10 is a +21 step across the 0/4095 seam, not -4075
    assert wrapped_delta(10 - 4085) == 21
    # 10 -> 4085 is a -21 step, not +4075
    assert wrapped_delta(4085 - 10) == -21
    assert -WRAP // 2 <= wrapped_delta(123456) < WRAP // 2


def test_apply_remove_offset_inverse():
    for off in (0, 500, 2050, 4095):
        for raw in (0, 1, 2047, 4094, 4095):
            assert remove_offset(apply_offset(raw, off), off) == raw
            assert 0 <= apply_offset(raw, off) <= 4095


def test_offset_zero_matches_legacy():
    """A joint with offset 0 must behave exactly like the pre-offset mapping."""
    j = make_joint()                              # offset defaults to 0
    assert j.offset == 0
    assert j.angle_to_raw(0.0) == 1000
    assert j.angle_to_raw(1.0) == 3000
    assert j.raw_to_angle(2000) == pytest.approx(0.5, abs=1e-3)


def test_seam_straddling_joint_is_continuous():
    """A joint whose travel crosses 0/4095 maps monotonically once offset moves
    the seam away, and every commanded tick stays inside 0-4095."""
    # Physical travel 3900 ..(wrap).. 200 -> continuous centre raw ~2, so offset
    # relocates that centre to corrected 2048.
    off = (2 - 2048) % WRAP
    j = Joint(name="w", motor_id=2, angle_min=0.0, angle_max=1.0, invert=False,
              raw_min=2048 - 198, raw_max=2048 + 198, center=2048, offset=off)
    # Sweeping the real encoder across the seam yields a strictly increasing angle.
    seq = [3900, 3950, 4000, 4090, 0, 100, 200]
    angles = [j.raw_to_angle(r) for r in seq]
    assert all(b > a for a, b in zip(angles, angles[1:])), angles
    assert j.raw_to_angle(3900) == pytest.approx(0.0, abs=1e-3)
    assert j.raw_to_angle(200) == pytest.approx(1.0, abs=1e-3)
    # Commands round-trip and never leave the single-turn window.
    for a in (0.0, 0.5, 1.0):
        r = j.angle_to_raw(a)
        assert 0 <= r <= 4095
        assert j.raw_to_angle(r) == pytest.approx(a, abs=1e-3)


def test_calibration_loads_and_propagates_offset(tmp_path):
    calib = tmp_path / "right_calibration.json"
    calib.write_text(json.dumps({
        "2": {"joint": "wrap_joint", "min": 1850, "max": 2246,
              "center": 2048, "offset": 2050, "range": 396},
        "3": {"joint": "plain_joint", "min": 1000, "max": 3000, "center": 2000},
    }))
    cal = Calibration.from_file(calib)
    assert cal.get(2).offset == 2050
    assert cal.get(3).offset == 0                 # absent -> 0 (backward compatible)
    assert cal.get(99).offset == 0                # full-span fallback

    yaml = tmp_path / "h_hand.yaml"
    yaml.write_text(
        "side: test\njoints:\n"
        "  wrap_joint: {id: 2, angle_min: 0.0, angle_max: 1.0}\n"
        "  plain_joint: {id: 3, angle_min: 0.0, angle_max: 1.0}\n")
    cfg = HandConfig.load(config_path=yaml, calibration_path=calib)
    assert cfg.joint("wrap_joint").offset == 2050
    assert cfg.joint("plain_joint").offset == 0


def test_ofs_cal_wire_packet():
    """OFSCAL must put the exact re-zero bytes on the wire: FF FF ID 04 0B lo hi
    CHK with the target tick little-endian. (The old code wrongly wrote 128 to
    register 40 and never moved the zero.)"""
    from orcahand.bus import FeetechBus

    class FakeSerial:
        is_open = True
        def __init__(self): self.written = bytearray()
        def write(self, b): self.written += b
        def reset_input_buffer(self): pass

    bus = FeetechBus(port="/dev/null")
    bus._ser = FakeSerial()
    bus.ofs_cal(7, 2048)
    # 7+4+0x0B+0x00+0x08 = 30 -> checksum ~30 & 0xFF = 0xE1
    assert bytes(bus._ser.written) == bytes([0xFF, 0xFF, 7, 4, 0x0B, 0x00, 0x08, 0xE1])


# ------------------------------------------------------------------- config
def test_packaged_models_load():
    assert set(HandConfig.available_models()) >= {"left", "right"}
    for model in ("left", "right"):
        cfg = HandConfig.load(model=model)
        assert cfg.side == model              # the loaded side is the one asked for
        assert len(cfg.joints) == 17
        assert "wrist" in cfg.joint_names
        assert len(set(cfg.motor_ids)) == 17  # unique IDs


def test_left_and_right_are_distinct():
    """Left and right are not the same config wearing different labels."""
    left = HandConfig.load(model="left")
    right = HandConfig.load(model="right")
    assert (left.side, right.side) == ("left", "right")

    # The wrist range is mirrored between the two hands.
    lw, rw = left.joint("wrist"), right.joint("wrist")
    assert lw.angle_min == pytest.approx(-rw.angle_max)
    assert lw.angle_max == pytest.approx(-rw.angle_min)

    # Same set of joints, but the joint -> motor-ID assignment differs.
    assert set(left.joint_names) == set(right.joint_names)   # same joints...
    left_map = dict(zip(left.joint_names, left.motor_ids))
    right_map = dict(zip(right.joint_names, right.motor_ids))
    assert left_map != right_map                             # ...different motors


def test_missing_calibration_falls_back(tmp_path):
    yaml = tmp_path / "h_hand.yaml"
    yaml.write_text(
        "side: test\njoints:\n  a: {id: 1, angle_min: 0.0, angle_max: 1.0}\n")
    cfg = HandConfig.load(config_path=yaml)
    j = cfg.joint("a")
    assert (j.raw_min, j.raw_max) == (0, 4095)  # full-range fallback
    assert j.offset == 0


# ----------------------------------------------------------------- mock hand
def test_mock_roundtrip():
    hand = OrcaHand.mock("right")
    hand.connect()
    try:
        hand.set_joint_positions({"index_mcp": 0.5, "thumb_cmc": 0.1})
        q = hand.get_joint_positions()
        assert q["index_mcp"] == pytest.approx(0.5, abs=5e-3)
        assert q["thumb_cmc"] == pytest.approx(0.1, abs=5e-3)
    finally:
        hand.disconnect()
    assert not hand.is_connected


def test_partial_command_only_moves_named():
    hand = OrcaHand.mock("right")
    hand.connect()
    before = hand.get_joint_positions()
    hand.set_joint_positions({"index_mcp": 0.9})
    after = hand.get_joint_positions()
    assert after["middle_mcp"] == pytest.approx(before["middle_mcp"], abs=1e-6)
    hand.disconnect()


def test_array_command_requires_full_length():
    hand = OrcaHand.mock("right")
    hand.connect()
    with pytest.raises(ValueError):
        hand.set_joint_positions([0.0, 0.1])  # too short
    hand.disconnect()


def test_named_pose_and_state():
    hand = OrcaHand.mock("left")
    hand.connect()
    hand.set_named_position("fist")
    st = hand.get_state()
    assert set(st.position) == set(hand.joint_names)
    assert all(not math.isnan(v) for v in st.position.values())
    assert hand.is_torque_enabled
    hand.disconnect()


def test_context_manager_disables_torque():
    with OrcaHand.mock("right") as hand:
        assert hand.is_connected
        assert hand.is_torque_enabled
    assert not hand.is_connected


# --------------------------------------------------------------- runnable demo
# `pytest` captures stdout, so the asserts above run silently. Run this file
# directly to get a visible, side-labelled report for BOTH hands (mock, no
# hardware needed):
#
#     python tests/test_orcahand.py
#
# To see prints while running the actual tests, use:  pytest -s -v
def _demo_side(side: str) -> None:
    tag = side.upper()
    bar = "=" * 60
    print(f"\n{bar}\n  {tag} HAND   (orcahand mock — no hardware)\n{bar}")
    with OrcaHand.mock(side) as hand:
        wrist = hand.config.joint("wrist")
        print(f"  side            : {hand.side}")
        print(f"  joints          : {len(hand.joint_names)}")
        print(f"  motor IDs       : {hand.motor_ids}")
        print(f"  wrist ROM [rad] : [{wrist.angle_min:+.3f}, {wrist.angle_max:+.3f}]")
        print(f"  named poses     : {', '.join(hand.pose_names)}")

        hand.set_named_position("fist")
        st = hand.get_state()
        print(f"  -- joint angles after 'fist' pose --")
        for name in sorted(hand.joint_names):
            print(f"     [{tag:<5}] {name:<11} {st.position[name]:+.3f} rad")
    print(f"  {tag} hand released (torque off, bus closed)")


if __name__ == "__main__":
    print("orcahand self-test demo — LEFT and RIGHT, no hardware required")
    for _side in ("left", "right"):
        _demo_side(_side)
    print("\nDone. Run the contract tests with:  pytest -v tests/test_orcahand.py")
