"""build_joint_driver — the one factory behind arms/neck/torso: backend
selection (teensy / can / mock), config plumbing, and failure modes."""
import pytest

can = pytest.importorskip("can")

from mabel_hw.drivers import build_joint_driver, MockJointGroupDriver
from mabel_hw.drivers.damiao_can import CanLink
from mabel_hw.drivers.teensy_link import TeensyJointGroup
from mabel_hw.errors import SubsystemUnavailable

from fakes.mock_arm_teensy import LAYOUTS, MockTeensy


def test_mock_backend():
    drv = build_joint_driver({"backend": "mock", "joints": ["a", "b"]}, links={})
    assert isinstance(drv, MockJointGroupDriver) and drv.names == ("a", "b")


def test_can_backend_virtual():
    drv = build_joint_driver(
        {"backend": "can", "joint_motors": "torso:vfac:1", "kp": 25.0,
         "bustype": "virtual"}, links={})
    assert isinstance(drv, CanLink) and drv.names == ("torso",)
    drv.connect()
    try:
        assert drv.connected()
        drv.set_targets({"torso": 0.2})
        drv.tick()
        assert drv.read().names == ("torso",)
    finally:
        drv.close()


def test_teensy_backend_against_mock():
    mock = MockTeensy(LAYOUTS["neck"]).start()
    try:
        cfg = {"backend": "teensy", "link": "neck_teensy",
               "joints": ["motor_L1", "motor_L2", "motor_head"],
               "kp": 12.0, "kd": 0.6}
        links = {"neck_teensy": {"ip": "127.0.0.1", "port": mock.port,
                                 "stream_hz": 200}}
        drv = build_joint_driver(cfg, links)
        assert isinstance(drv, TeensyJointGroup)
        assert drv.names == ("motor_L1", "motor_L2", "motor_head")
        drv.close()
    finally:
        mock.stop()


def test_teensy_backend_no_hardware_defers():
    """Board absent at build time: the factory returns an UNRESOLVED group
    (reads dead, writes latch) that hot-joins later via service() — it must
    NOT raise, so a Teensy that boots after the bridge needs no restart."""
    cfg = {"backend": "teensy", "link": "neck_teensy", "joints": ["motor_L1"]}
    links = {"neck_teensy": {"ip": "127.0.0.1", "port": 1, "stream_hz": 200}}
    drv = build_joint_driver(cfg, links)
    assert isinstance(drv, TeensyJointGroup)
    assert not drv.resolved and not drv.connected()
    fb = drv.read()
    assert fb.live == (False,)                    # dead, not crashed
    drv.set_targets({"motor_L1": 0.3})            # latches locally, no raise
    with pytest.raises(SubsystemUnavailable):
        drv.connect()                             # explicit connect stays loud


def test_unknown_backend_rejected():
    with pytest.raises(ValueError):
        build_joint_driver({"backend": "zigbee", "joints": []}, links={})


def test_config_defaults_are_non_energizing():
    """Safety contract: every subsystem that ships ENABLED runs a backend that
    does NOT energize on connect. Torso and the whole neck ride the arms Teensy's
    `openarm` link (nothing streams until set_torque); the neck's calibration
    lives in the Teensy-frame slice — and the legacy ARMS/BODY dicts are untouched."""
    from mabel_hw import config
    assert config.TORSO["enabled"] is True
    assert config.TORSO["backend"] == "openarm"
    assert config.NECK["backend"] == "openarm"
    assert config.NECK["enabled"] is True
    assert config.NECK["link"] == "arms_openarm"
    assert config.NECK["calib_slice"] == "neck/teensy"
    assert set(config.NECK["joints"]) == {"neck_yaw", "neck_pitch", "neck_roll"}
    assert config.NECK["wire_t_max"] == {"neck_yaw": 10.0}        # Dynamixels: none
    assert config.NECK["kp"]["neck_pitch"] > 0                     # kp>0 == "hold"
    assert config.LINKS["arms_openarm"]["ip"] == "192.168.124.11"
    # legacy dicts stay exactly as the DDS adapters expect
    assert config.ARMS["topic_command"] == "/arms/command"
    assert config.BODY["topic_command"] == "/body/command"
