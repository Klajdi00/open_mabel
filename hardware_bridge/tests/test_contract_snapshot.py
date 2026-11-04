"""Freeze the bridge's external contract (topics, ports, env names, spawn strings).

tests/contract/contract.json is the single frozen copy of everything an OUTSIDE
consumer binds to. These tests assert the live code still matches it — so any
refactor that would break a client fails here first, not on the robot.
"""
import json
import os

import pytest

from conftest import BRIDGE_DIR

CONTRACT = json.load(open(os.path.join(os.path.dirname(__file__),
                                       "contract", "contract.json")))


def _cfg():
    from mabel_hw import config
    return config


# ── topics ─────────────────────────────────────────────────────────────────────

def test_base_topics():
    c = _cfg()
    t = CONTRACT["topics"]
    assert c.BASE["topic_cmd_vel"] == t["base_cmd_vel"]
    assert c.BASE["topic_estop"] == t["base_estop"]
    assert c.BASE["topic_joint_states"] == t["base_joint_states"]


def test_arms_body_topics():
    c = _cfg()
    t = CONTRACT["topics"]
    assert c.ARMS["topic_command"] == t["arms_command"]
    assert c.ARMS["topic_joint_states"] == t["arms_joint_states"]
    assert c.ARMS["topic_torque"] == t["arms_torque"]
    assert c.ARMS["topic_torque_state"] == t["arms_torque_state"]
    assert c.BODY["topic_command"] == t["body_command"]
    assert c.BODY["topic_joint_states"] == t["body_joint_states"]
    assert c.BODY["topic_torque"] == t["body_torque"]
    assert c.BODY["topic_torque_state"] == t["body_torque_state"]


def test_lift_lidar_topics():
    c = _cfg()
    t = CONTRACT["topics"]
    assert c.LIFT["topic_command"] == t["lift_command"]
    assert c.LIFT["topic_joint_states"] == t["lift_joint_states"]
    assert c.LIFT["topic_calibrate"] == t["lift_calibrate"]
    assert c.LIFT["topic_limits"] == t["lift_limits"]
    assert c.LIDAR["topic"] == t["lidar_scan"]


# ── ports ──────────────────────────────────────────────────────────────────────

def test_ports():
    c = _cfg()
    p = CONTRACT["ports"]
    assert c.BASE["port"] == p["base_teensy_udp"]
    assert c.CAMERAS["http_port"] == p["camera_mjpeg_http"]
    assert c.CAMERAS["udp"]["jpeg_port"] == p["camera_udp_jpeg"]
    assert c.CAMERAS["udp"]["h264_port"] == p["camera_udp_h264"]
    d = c.CAMERAS["dedicated"]
    assert d["head_sbs_port"] == p["camera_dedicated_head_sbs"]
    assert d["wrist_left_port"] == p["camera_dedicated_wrist_left"]
    assert d["wrist_right_port"] == p["camera_dedicated_wrist_right"]
    assert d["front_port"] == p["camera_dedicated_front"]


def test_hand_rpc_port():
    from mabel_hw.rpc.hand_server import DEFAULT_PORT, DEFAULT_HOST
    assert DEFAULT_PORT == CONTRACT["ports"]["hand_rpc_tcp"]
    assert DEFAULT_HOST == "127.0.0.1"


# ── source needles (spawn strings, lock path — things tests can't import) ─────

@pytest.mark.parametrize("needle", CONTRACT["source_needles"],
                         ids=[n["file"] + ":" + n["contains"][:24]
                              for n in CONTRACT["source_needles"]])
def test_source_needle(needle):
    path = os.path.join(BRIDGE_DIR, needle["file"])
    assert os.path.exists(path), f"contract file vanished: {needle['file']}"
    text = open(path, encoding="utf-8", errors="replace").read()
    assert needle["contains"] in text, (
        f"{needle['file']} no longer contains {needle['contains']!r} "
        f"({needle['why']}) — breaking change or move without shim")
