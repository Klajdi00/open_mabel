"""App surface: REST + the /ws/state contract + the VP node protocol. The
command surface mirrors the app wire protocol (torque / control_mode /
joint_command / external_force)."""
import json
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
os.environ.setdefault("MABEL_CTS_VP_PORT", "19391")     # keep off the real 9090

from fastapi.testclient import TestClient                # noqa: E402
from control_lab.server.app import create_app            # noqa: E402


@pytest.fixture(scope="module")
def client():
    app = create_app()
    with TestClient(app) as c:
        c.app = app
        yield c


def _post(client, **body):
    return client.post("/api/command", json=body)


def test_health_config_joints(client):
    assert client.get("/healthz").text == "ok"
    cfg = client.get("/api/config").json()
    assert cfg["methods"] == ["navigation", "joint", "wrist", "retarget"]
    assert cfg["controlTypes"] == ["impedance", "compliance"]
    assert cfg["regions"] == ["arm", "upper_body", "whole_body"]
    j = client.get("/api/joints").json()
    assert len(j["names"]) == 60 and len(j["modules"]) == 7
    assert "ranges" in j and j["ranges"]["left_arm_2"][1] > 0    # actuator ctrlrange


def test_bodies_endpoint(client):
    b = client.get("/api/bodies").json()["bodies"]
    assert "left_palm" in b and "right_palm" in b


def test_ws_contract(client):
    with client.websocket_connect("/ws/state") as ws:
        first = json.loads(ws.receive_text())
        assert first["type"] == "schema"
        assert len(first["payload"]["names"]) == 60
        assert json.loads(ws.receive_text())["type"] == "plots_history"
        msg = json.loads(ws.receive_text())
        assert msg["type"] == "state"
        p = msg["payload"]
        for key in ("t", "torque", "spec", "method", "region", "jointPositions",
                    "base", "odom", "lift", "vp", "timing", "overlay", "plots",
                    "vpNode"):
            assert key in p, key
        assert set(p["spec"]) == {"method", "type", "region", "stiffness"}
        assert len(p["jointPositions"]) > 50
        # joystick uplink round-trips into the world
        ws.send_text(json.dumps({"type": "input", "nav": {"lx": 0.5, "ly": 0, "rx": 0, "ry": 0}}))
        time.sleep(0.2)
        assert client.app.state.world.nav["lx"] == 0.5
        ws.send_text(json.dumps({"type": "input", "nav": {"lx": 0, "ly": 0, "rx": 0, "ry": 0}}))


def test_torque_and_control_mode(client):
    r = _post(client, action="torque", on=True).json()
    assert r["ok"] and r["torque"] is True
    r = _post(client, action="control_mode", method="joint", controlType="compliance",
              region="whole_body", stiffness=15).json()
    assert r["spec"]["method"] == "joint" and r["spec"]["type"] == "compliance"
    assert r["spec"]["region"] == "whole_body" and r["spec"]["stiffness"] == 15.0
    # joint_command streams a target dict (latest-wins), like the app
    world = client.app.state.world
    _post(client, action="control_mode", method="joint", controlType="impedance",
          region="arm", stiffness=30)
    time.sleep(0.2)
    q0 = world.plant.q("Left Arm 2")
    _post(client, action="joint_command", joints={"left_arm_2": q0 + 0.25})
    time.sleep(1.0)
    assert world.plant.q("Left Arm 2") - q0 > 0.1, "joint_command did not pose the arm"
    _post(client, action="torque", on=False)


def test_wrist_uplink_moves_target(client):
    world = client.app.state.world
    assert _post(client, action="torque", on=True).json()["ok"]
    assert _post(client, action="control_mode", method="wrist",
                 controlType="impedance", region="arm").json()["ok"]
    with client.websocket_connect("/ws/state") as ws:
        ws.receive_text(); ws.receive_text()          # schema + history
        for _ in range(40):
            ov = world.snapshot()["overlay"]
            if "targets" in ov and "l" in ov["targets"]:
                break
            time.sleep(0.05)
        vis0 = world.snapshot()["overlay"]["targets"]["l"]["target"]
        goal = [vis0[0], vis0[1] - 0.08, vis0[2]]
        ws.send_text(json.dumps({"type": "wrist", "side": "l", "pos": goal}))
        time.sleep(0.3)
        vis1 = world.snapshot()["overlay"]["targets"]["l"]["target"]
        err = sum((a - b) ** 2 for a, b in zip(vis1, goal)) ** 0.5
        assert err < 0.02, f"vis-target round-trip off by {err*100:.1f} cm"
        # malformed messages must not kill the socket
        ws.send_text(json.dumps({"type": "wrist", "side": "zz", "pos": [0]}))
        ws.send_text(json.dumps({"type": "joint", "id": "left_arm_1"}))  # missing value
        time.sleep(0.2)
        assert json.loads(ws.receive_text())["type"] == "state"
    _post(client, action="torque", on=False)


def test_command_matrix_and_errors(client):
    _post(client, action="torque", on=True)
    for method in ("navigation", "joint", "wrist", "retarget"):
        for ct in ("impedance", "compliance"):
            for region in ("arm", "upper_body", "whole_body"):
                assert _post(client, action="control_mode", method=method,
                             controlType=ct, region=region).json()["ok"]
    assert _post(client, action="control_mode", method="nope").status_code == 400
    assert _post(client, action="control_mode", region="warp").status_code == 400
    assert _post(client, action="nonsense").status_code == 400
    assert "kind" in _post(client, action="calibration").json()
    _post(client, action="torque", on=False)


def test_vp_headset_protocol(client):
    """The unmodified app protocol: connect, hello, stream a teleop_frame, the
    world receives an OperatorPose."""
    import websockets.sync.client as wsc
    world = client.app.state.world
    assert _post(client, action="control_mode", method="retarget").json()["ok"]
    assert _post(client, action="vp", source="headset").json()["ok"]
    time.sleep(0.5)
    port = int(os.environ["MABEL_CTS_VP_PORT"])
    eye = [1, 0, 0, 0, 0, 1, 0, 1.4, 0, 0, 1, 0, 0, 0, 0, 1]
    with wsc.connect(f"ws://127.0.0.1:{port}/teleop") as ws:
        assert json.loads(ws.recv())["type"] == "hello"
        ws.send(json.dumps({"type": "ping", "payload": {"n": 1}}))
        assert json.loads(ws.recv())["type"] == "pong"
        before = world.vp_status["frames"]
        for i in range(5):
            ws.send(json.dumps({"type": "teleop_frame", "payload": {
                "sequence": i, "timestamp": time.time(),
                "head": {"transform": eye, "trackingState": "normal"},
                "navJoystick": {"lx": 0, "ly": 0.3, "rx": 0, "ry": 0}}}))
            time.sleep(0.05)
        time.sleep(0.5)
    assert world.vp_status["frames"] > before, "headset frames not ingested"
    _post(client, action="vp", source="off")
