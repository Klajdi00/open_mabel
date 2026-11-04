"""Both ORCA hands on the Mac: side-aware adapter detection + both configs load.

The two hand adapters are identical CH343s; the HAL tells them apart ONLY by USB
serial (config.HANDS adapter_serials). Pins the 2026-10-01 state: left=0000000001
(re-pinned after the left hand moved to a new adapter), right=0000000002, and that
with BOTH plugged in each side gets its own adapter — never the other hand's.
Hermetic: pyserial's comports() is monkeypatched; no serial port is opened.
"""
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import serial.tools.list_ports as lp  # noqa: E402

from mabel_hw.config import HANDS  # noqa: E402
from mabel_hw.drivers.orcahand.ports import detect_side_port  # noqa: E402

LEFT_SN, RIGHT_SN = "0000000001", "0000000002"
LEFT_DEV, RIGHT_DEV = "/dev/cu.usbmodem00000000011", "/dev/cu.usbmodem00000000021"


def _port(dev, sn, vid=0x1A86):
    return types.SimpleNamespace(device=dev, serial_number=sn, vid=vid,
                                 description="USB Single Serial")


@pytest.fixture
def bus(monkeypatch):
    ports = []
    monkeypatch.setattr(lp, "comports", lambda: list(ports))
    return ports


def test_registry_pins_both_hands():
    reg = HANDS["adapter_serials"]
    assert reg["left"] == LEFT_SN and reg["right"] == RIGHT_SN
    assert reg["_reserved_lidar"]                       # the lidar's CP2102 is never a hand


def test_both_hands_plugged_in_each_side_gets_its_own(bus):
    bus += [_port(LEFT_DEV, LEFT_SN), _port(RIGHT_DEV, RIGHT_SN)]
    assert detect_side_port("left", HANDS["adapter_serials"]) == LEFT_DEV
    assert detect_side_port("right", HANDS["adapter_serials"]) == RIGHT_DEV


def test_plug_order_does_not_matter(bus):
    bus += [_port(RIGHT_DEV, RIGHT_SN), _port(LEFT_DEV, LEFT_SN)]
    assert detect_side_port("left", HANDS["adapter_serials"]) == LEFT_DEV
    assert detect_side_port("right", HANDS["adapter_serials"]) == RIGHT_DEV


def test_only_left_plugged_in_right_stays_idle(bus):
    bus += [_port(LEFT_DEV, LEFT_SN)]
    assert detect_side_port("left", HANDS["adapter_serials"]) == LEFT_DEV
    assert detect_side_port("right", HANDS["adapter_serials"]) is None


def test_unregistered_adapter_is_never_claimed_but_is_named(bus):
    hints = []
    bus += [_port(LEFT_DEV, LEFT_SN), _port("/dev/cu.usbmodemNEW1", "5AE6099999")]
    assert detect_side_port("left", HANDS["adapter_serials"], log=hints.append) == LEFT_DEV
    assert detect_side_port("right", HANDS["adapter_serials"], log=hints.append) is None
    assert any("5AE6099999" in h and "MABEL_HW_HANDS_RIGHT_SERIAL" in h for h in hints)


def test_lidar_cp2102_is_never_a_hand(bus):
    bus += [_port("/dev/cu.usbserial-lidar", HANDS["adapter_serials"]["_reserved_lidar"], vid=0x10C4)]
    assert detect_side_port("left", HANDS["adapter_serials"]) is None
    assert detect_side_port("right", HANDS["adapter_serials"]) is None


@pytest.mark.parametrize("side", ["left", "right"])
def test_both_hand_configs_load_with_the_studio_master(side):
    from mabel_hw.drivers.orcahand.config import HandConfig
    cfg = HandConfig.load(model=side)
    ids = sorted(j.motor_id for j in cfg.joints)
    assert ids == list(range(1, 18)), ids                  # 17 motors, ids 1..17, unique
    assert cfg.calibration_path and str(cfg.calibration_path).endswith("mabel.json"), cfg.calibration_path
    # direction comes from the studio master's per-joint `dir` (5 inverted joints on the
    # left hand, 12 on the right as calibrated), NOT from the yaml's fallback `invert`
    assert sum(1 for j in cfg.joints if j.invert) == {"left": 5, "right": 12}[side]


# ---- swappable hands: NOVA's ORCA v2 on MABEL (2026-10-05) -------------------------------------
V2_LEFT_SN, V2_RIGHT_SN = "0000000003", "0000000004"


def test_registry_knows_novas_v2_hands():
    reg = HANDS["adapter_serials"]
    assert reg["left_v2"] == V2_LEFT_SN and reg["right_v2"] == V2_RIGHT_SN


def test_v2_adapter_is_never_claimed_as_a_v1_side(bus):
    bus += [_port("/dev/cu.usbmodem00000000031", V2_LEFT_SN)]
    assert detect_side_port("left", HANDS["adapter_serials"]) is None     # registered to left_v2, not left
    assert detect_side_port("right", HANDS["adapter_serials"]) is None


def _hand_side(side, bus_ports, monkeypatch):
    """A HAL Hand with its connect stubbed out: only the side-resolution runs."""
    from mabel_hw.drivers import hands as H
    monkeypatch.setattr(H.Hand, "_connect", lambda self: None)
    h = H.Hand(side, port="auto", baudrate=1000000, adapter_serials=HANDS["adapter_serials"])
    return h, h._resolve_port()


@pytest.mark.parametrize("side,v1_sn,v2_sn", [("left", LEFT_SN, V2_LEFT_SN), ("right", RIGHT_SN, V2_RIGHT_SN)])
def test_hal_picks_the_generation_by_adapter(bus, monkeypatch, side, v1_sn, v2_sn):
    bus += [_port(f"/dev/cu.usbmodem{v2_sn}1", v2_sn)]                       # only NOVA's v2 hand on this side
    h, port = _hand_side(side, bus, monkeypatch)
    assert port == f"/dev/cu.usbmodem{v2_sn}1" and h.generation == "v2"
    bus.clear(); bus += [_port(f"/dev/cu.usbmodem{v1_sn}1", v1_sn)]          # MABEL's own v1 hand
    h, port = _hand_side(side, bus, monkeypatch)
    assert port == f"/dev/cu.usbmodem{v1_sn}1" and h.generation == "v1"
    bus += [_port(f"/dev/cu.usbmodem{v2_sn}1", v2_sn)]                       # both plugged: the robot's own wins
    h, port = _hand_side(side, bus, monkeypatch)
    assert h.generation == "v1"


@pytest.mark.parametrize("side", ["left", "right"])
def test_v2_config_loads_its_own_wiring_and_calibration(side):
    from mabel_hw.drivers.orcahand.config import HandConfig
    v1, v2 = HandConfig.load(model=side), HandConfig.load(model=f"{side}_v2")
    assert v1.generation == "v1" and v2.generation == "v2" and v2.side == side
    assert sorted(j.motor_id for j in v2.joints) == list(range(1, 18))
    assert {j.name for j in v1.joints} == {j.name for j in v2.joints}       # same names -> same URDF joints on publish
    assert any(a.motor_id != b.motor_id for a in v1.joints for b in v2.joints if a.name == b.name)   # different wiring
    assert v2.calibration_path and str(v2.calibration_path).endswith("mabel.json")
    assert all(getattr(j, "model_mid", None) is not None for j in v2.joints)  # all 17 from hand/orca-v2-<side>, no v1 tick fallback
