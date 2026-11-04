"""Host/service status module."""
import socket
import threading

from control_center.server import system


def test_probe_open_and_closed_port():
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    try:
        assert system.probe(port) is True
    finally:
        srv.close()
    assert system.probe(port) is False           # nothing listens now


def test_snapshot_shape():
    s = system.snapshot()
    assert s["hostname"]
    assert isinstance(s["interfaces"], list)
    assert set(s["services"]) == {"teleop", "map", "video", "studio"}
    for v in s["services"].values():
        assert isinstance(v["up"], bool) and isinstance(v["port"], int)
    assert s["mem"]["pct"] >= 0
    assert len(s["load"]) == 3
    assert s["uptime_s"] >= 0
    assert isinstance(s["thermal"], dict)        # may be empty off-Jetson


def test_snapshot_is_json_serializable():
    import json
    json.dumps(system.snapshot())
