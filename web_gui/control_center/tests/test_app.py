"""FastAPI app integration: HTTP surface + the /ws/telemetry contract, run
end-to-end against the mock plant (no ROS required)."""
import json
import os

import pytest

os.environ["MABEL_CC_SOURCE"] = "mock"

from fastapi.testclient import TestClient          # noqa: E402

from control_center.server import app as app_mod   # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app_mod.create_app()) as c:    # context mgr fires startup
        yield c


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_config(client):
    j = client.get("/api/config").json()
    assert j["teleopPort"] == 9090
    assert j["mapPort"] == 9092
    assert j["videoPort"] == 8080
    assert j["tempWarn"] < j["tempHot"]
    # the shared (sim-twin) frontend gates sim-only UI on these
    assert j["sim"] is False
    assert j["manageSim"] is False
    assert j["scene"] is None
    assert j["ros"] is False                 # mock source in CI


def test_joints_endpoint(client):
    j = client.get("/api/joints").json()
    assert j["dof"] == 59 and len(j["names"]) == 60


def test_system_endpoint(client):
    j = client.get("/api/system").json()
    assert "services" in j and "hostname" in j


def test_recordings_endpoint(client):
    j = client.get("/api/recordings").json()
    assert "episodes" in j and "bags" in j


def test_camera_rig_endpoint(client):
    r = client.get("/api/camera-rig")
    assert r.status_code == 200
    rig = r.json()
    cams = rig if isinstance(rig, list) else rig.get("cameras", [])
    names = {c["name"] for c in cams}
    # the frusta the viewer binds to must exist in the rig manifest
    assert {"zed_left", "l_wrist_cam", "r_wrist_cam"} <= names


def test_static_index_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "MABEL · Control Center" in r.text
    assert 'src="js/app.js"' in r.text


def test_website_assets_mounted(client):
    r = client.get("/website-assets/mabel_joints.json")
    assert r.status_code == 200
    assert len(r.json()["joints"]) == 60


def test_no_cache_headers_on_shell(client):
    r = client.get("/api/health")
    assert "no-store" in r.headers.get("cache-control", "")


def test_ws_telemetry_contract(client):
    with client.websocket_connect("/ws/telemetry") as ws:
        first = json.loads(ws.receive_text())
        assert first["type"] == "schema"
        assert first["payload"]["dof"] == 59
        names = first["payload"]["names"]

        msg = json.loads(ws.receive_text())
        assert msg["type"] == "telemetry"
        p = msg["payload"]
        for key in ("p", "v", "e", "tgt", "temp"):
            assert len(p[key]) == len(names) == 60
        assert p["source"] == "mock"
        assert p["estop"] is False
        assert len(p["scan"]) == 360
        # first frame carries the ~1 Hz system + viewers sidecar
        assert "system" in p and "viewers" in p
        assert any(v for v in p["viewers"])       # we are registered

        # ping → pong round-trip
        ws.send_text(json.dumps({"type": "ping", "payload": {"t": 123.0}}))
        for _ in range(30):
            m = json.loads(ws.receive_text())
            if m["type"] == "pong":
                assert m["payload"]["t"] == 123.0
                break
        else:
            raise AssertionError("no pong received")


def test_viewer_registry_cleanup(client):
    with client.websocket_connect("/ws/telemetry") as ws:
        ws.receive_text()
        assert client.get("/api/viewers").json()
    # after disconnect the roster drains
    assert client.get("/api/viewers").json() == []


# ── ported planes: action registry, bridge proxies, physics preview ─────────

def test_actions_registry(client):
    j = client.get("/api/actions").json()
    assert j["navBackend"] == "ros"
    acts = {a["id"]: a for a in j["actions"]}
    # poses + simple nav primitives execute today (client-side)
    assert acts["pose_home"]["status"] == "available"
    assert acts["nav_fwd"]["status"] == "available"
    # policies are honestly coming_soon (no deployable checkpoint)
    assert acts["policy_coffee"]["status"] == "coming_soon"
    # 'move to goal' needs Nav2 → unavailable on the mock source
    assert acts["nav_goto"]["status"] == "coming_soon"
    lanes = {a["lane"] for a in j["actions"]}
    assert lanes == {"nav", "manip"}


def test_ws_teleop_proxy_no_bridge(client):
    """With no bridge on :9090 the proxy must fail cleanly (accept then
    close), never crash the app. (Live-bridge round-trip is exercised in the
    manual validation run.)"""
    try:
        with client.websocket_connect("/ws/teleop?client=web&sid=t3st") as ws:
            ws.receive_text()                 # closed or errors — both fine
    except Exception:
        pass
    assert client.get("/api/health").json()["ok"] is True


def test_ws_monitor_proxy_no_bridge(client):
    try:
        with client.websocket_connect("/ws/monitor") as ws:
            ws.receive_text()
    except Exception:
        pass
    assert client.get("/api/health").json()["ok"] is True


def test_preview_status_shape(client):
    j = client.get("/api/preview/status").json()
    assert j["running"] is False
    assert j["instance"] == "preview"


def test_preview_start_guarded(client, monkeypatch):
    """When mujoco/simulation_bridge are unavailable the endpoint answers
    {ok:false, error} (frontend toasts) instead of 500-ing opaquely."""
    from control_center.server import preview as pv
    monkeypatch.setattr(pv, "_IMPORT_ERROR", "forced: no mujoco", raising=True)
    r = client.post("/api/preview/start", json={})
    assert r.status_code == 500
    j = r.json()
    assert j["ok"] is False and "no mujoco" in j["error"]


def test_preview_stop_idempotent(client):
    j = client.post("/api/preview/stop").json()
    assert j["ok"] is True


def test_config_names_the_robot_and_its_hands(client):
    j = client.get("/api/config").json()
    assert j["robot"] == "MABEL"
    assert set(j["hands"]) == {"left", "right"}
    assert all(g in ("v1", "v2") for g in j["hands"].values())


def test_page_assets_revalidate(client):
    # the operator must never drive with a stale cached module
    for path in ("/", "/js/app.js", "/js/teleopctl.js", "/css/control.css"):
        r = client.get(path)
        assert r.status_code == 200, path
        cc = r.headers.get("cache-control", "")
        assert "no-cache" in cc or "no-store" in cc, (path, cc)


def test_ws_map_placeholder_closes_cleanly(client):
    # mapPort 0 deployments dial a same-origin /ws/map: accept and close, no traceback
    from starlette.websockets import WebSocketDisconnect
    with client.websocket_connect("/ws/map") as ws:
        with pytest.raises(WebSocketDisconnect) as e:
            ws.receive_text()
        assert e.value.code == 1000
