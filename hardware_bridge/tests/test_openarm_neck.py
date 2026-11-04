"""The neck as ONE OpenArmGroup on the arms Teensy (backend `openarm`, 2026-09-30):
yaw (DaMiao) + pitch/roll (one 2XL430 on the Teensy's serial bus) resolved from
the manifest, read as raw radians, held/released through the MIT command, and —
the part a resting Dynamixel breaks — reported LIVE although it publishes
vel = tau = t_mos = 0 by design. Against MockOpenArmTeensy; no hardware."""
import time

import pytest

from mabel_hw.drivers import build_joint_driver
from mabel_hw.drivers.openarm_link import OpenArmLink

from fakes.mock_openarm_teensy import LAYOUT, MockOpenArmTeensy

# The real board's manifest (read 2026-09-30): arms + grippers, torso, neck_yaw on
# CAN, and the two Dynamixel axes with t_max 0 ("no MIT scale").
NECK_LAYOUT = list(LAYOUT) + [("torso", "DM10422", 2, 1, 500.0),
                              ("neck_yaw", "DM4310", 2, 2, 10.0),
                              ("neck_pitch", "2XL430-W250-T", 0, 0, 0.0),
                              ("neck_roll", "2XL430-W250-T", 0, 0, 0.0)]
NAMES = ["neck_yaw", "neck_pitch", "neck_roll"]
POSE = {"neck_yaw": 1.3022, "neck_pitch": 1.6, "neck_roll": 0.02}


def _wait(pred, timeout=5.0, dt=0.02):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if pred():
            return True
        time.sleep(dt)
    return pred()


@pytest.fixture()
def mock():
    # temp_mos=0: like the firmware's Dynamixel branch, the health poll carries NO
    # MOSFET temperature — only the rotor/servo temperature.
    m = MockOpenArmTeensy(layout=NECK_LAYOUT, temp_mos=0, temp_rotor=31).start()
    m.set_pos(**POSE)
    yield m
    m.stop()


@pytest.fixture()
def neck(mock):
    links = {"arms_openarm": {"ip": "127.0.0.1", "port": mock.port, "stream_hz": 200}}
    cfg = dict(backend="openarm", link="arms_openarm", joints=list(NAMES),
               kp={"neck_yaw": 6.0, "neck_pitch": 1.0, "neck_roll": 1.0},
               kd={"neck_yaw": 0.5, "neck_pitch": 0.0, "neck_roll": 0.0},
               wire_t_max={"neck_yaw": 10.0})
    drv = build_joint_driver(cfg, links)
    yield drv
    drv._link.close()
    OpenArmLink._registry.pop(("127.0.0.1", mock.port), None)


def _idx(mock):
    return {n: mock.names.index(n) for n in NAMES}


def test_neck_resolves_and_reads_raw_pose(neck, mock):
    assert _wait(lambda: neck.resolved), "group did not bind to the manifest"
    assert _wait(lambda: neck.read().position[1] == pytest.approx(POSE["neck_pitch"]))
    fb = neck.read()
    assert fb.names == tuple(NAMES)
    assert fb.position[0] == pytest.approx(POSE["neck_yaw"])
    assert fb.position[2] == pytest.approx(POSE["neck_roll"])
    # connect must not energize: the mock has seen no Command at all
    assert mock.cmd_count == 0


def test_resting_dynamixel_reads_live(neck, mock):
    """vel = tau = t_mos = 0 at rest — only the health poll's rotor temperature
    says the servo exists. The old 'any live signal' rule read it DEAD."""
    assert _wait(lambda: neck.resolved)
    # health arrives on the link's 2 s poll; the mock answers every request
    assert _wait(lambda: all(neck.read().live), timeout=6.0), neck.read()
    fb = neck.read()
    assert fb.velocity[1] == 0.0 and fb.effort[1] == 0.0
    assert fb.temperature[1] == pytest.approx(31.0)          # rotor temp surfaces


def test_torque_holds_measured_pose_then_releases(neck, mock):
    assert _wait(lambda: neck.resolved and neck.read().position[1] != 0.0)
    ix = _idx(mock)
    neck.set_torque(True)
    assert neck.is_torque_enabled
    assert _wait(lambda: mock.last_cmd is not None and mock.last_cmd["kp"][ix["neck_pitch"]] > 0)
    c = mock.last_cmd
    for n in NAMES:                       # hold AT the measured raw angle, kp > 0
        assert c["kp"][ix[n]] > 0.0
        assert c["pos"][ix[n]] == pytest.approx(POSE[n], abs=1e-4)
    assert c["kp"][ix["neck_yaw"]] == pytest.approx(6.0)
    assert c["kp"][ix["neck_pitch"]] == pytest.approx(1.0)      # the "hold" flag
    # a joint outside the group (l_j1) is never touched by this group's slice
    assert c["kp"][mock.names.index("l_j1")] == 0.0
    neck.set_torque(False)
    assert _wait(lambda: mock.last_cmd["kp"][ix["neck_pitch"]] == 0.0)
    assert mock.last_cmd["kp"][ix["neck_yaw"]] == 0.0           # kp 0 = release


def test_wire_t_max_guard_only_checks_the_damiao_yaw(neck, mock):
    """The guard must not refuse the whole neck because the Dynamixels advertise
    t_max 0 (by design). Yaw's manifest 10.0 == register 10.0 -> engage allowed."""
    assert _wait(lambda: neck.resolved and neck.read().position[0] != 0.0)
    assert neck._wire_t_max == {"neck_yaw": 10.0}
    neck.set_torque(True)
    assert neck.is_torque_enabled
    neck.set_torque(False)
