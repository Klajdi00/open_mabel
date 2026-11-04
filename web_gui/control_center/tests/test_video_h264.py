"""The fast video path: /ws/cam/<name>/h264.

The studio's cameras used to be plain MJPEG <img> tags, which is why its video was
far slower than the iOS app's and the website teleop page's — both of which consume
the camera server's hardware-encoded H.264. The browser now prefers H.264/WebCodecs
(web/js/h264.js + camview.js) and this backend route carries it when the studio is
served over https from a single origin (same-origin mode, see web/js/config.js).

These tests stand up a FAKE camera server speaking the real wire contract (one
binary WS message == one Annex-B access unit) and drive the proxy end-to-end.
"""
import asyncio
import os
import threading

import pytest

os.environ["MABEL_CC_SOURCE"] = "mock"

websockets = pytest.importorskip("websockets")

from fastapi.testclient import TestClient            # noqa: E402

# one AU: SPS + PPS + IDR, Annex-B — the shape video_server.py emits at a keyframe
AU_KEY = (b"\x00\x00\x00\x01\x67\x42\xc0\x1e"        # SPS (nal type 7)
          b"\x00\x00\x00\x01\x68\xce\x3c\x80"        # PPS (nal type 8)
          b"\x00\x00\x00\x01\x65" + b"\xaa" * 64)    # IDR (nal type 5)
AU_DELTA = b"\x00\x00\x00\x01\x41" + b"\xbb" * 32    # non-IDR (nal type 1)


class FakeCameraServer:
    """Serves /camera/<name>/h264 exactly like hardware_bridge's video_server."""

    def __init__(self):
        self.port = None
        self.connections = 0
        self.closed = 0
        self._loop = None
        self._ready = threading.Event()

    async def _handler(self, ws, path=None):
        self.connections += 1
        try:
            while True:
                await ws.send(AU_KEY)
                await ws.send(AU_DELTA)
                await asyncio.sleep(0.02)
        except Exception:
            pass
        finally:
            self.closed += 1

    def start(self):
        def run():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

            async def main():
                async with websockets.serve(self._handler, "127.0.0.1", 0) as s:
                    self.port = s.sockets[0].getsockname()[1]
                    self._ready.set()
                    await asyncio.Future()

            try:
                self._loop.run_until_complete(main())
            except asyncio.CancelledError:
                pass

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        assert self._ready.wait(5), "fake camera server did not start"
        return self.port


@pytest.fixture()
def cam():
    s = FakeCameraServer()
    s.start()
    yield s


@pytest.fixture()
def client(cam, monkeypatch):
    # point the backend's camera-server port at the fake one
    from control_center.server import config as CFG
    monkeypatch.setattr(CFG, "VIDEO_PORT", cam.port)
    monkeypatch.setattr(CFG, "CC_REQUIRE_LOGIN", False)   # gating covered below
    from control_center.server import app as app_mod
    with TestClient(app_mod.create_app()) as c:
        yield c


def test_h264_route_is_login_gated(cam, monkeypatch):
    """The video plane is a data plane: it must sit behind the session cookie
    like every other /ws/*, not become an unauthenticated hole in the studio."""
    from control_center.server import config as CFG
    monkeypatch.setattr(CFG, "VIDEO_PORT", cam.port)
    monkeypatch.setattr(CFG, "CC_REQUIRE_LOGIN", True)
    from control_center.server import app as app_mod
    from starlette.websockets import WebSocketDisconnect
    with TestClient(app_mod.create_app()) as c:
        with pytest.raises(WebSocketDisconnect) as e:
            with c.websocket_connect("/ws/cam/main/h264"):
                pass
    assert e.value.code == 1008          # policy violation == no session
    assert cam.connections == 0, "gate must reject BEFORE dialing the camera"


def test_h264_access_units_pass_through_byte_identical(client):
    """The proxy must be binary-transparent: the browser's WebCodecs decoder sees
    the SAME access units the encoder produced, or it shows black."""
    with client.websocket_connect("/ws/cam/main/h264") as ws:
        first = ws.receive_bytes()
        second = ws.receive_bytes()
    assert first == AU_KEY, "keyframe AU altered in transit"
    assert second == AU_DELTA, "delta AU altered in transit"
    # SPS+PPS must ride in-band on the keyframe, or a late joiner never configures
    assert first.count(b"\x00\x00\x00\x01") == 3


def test_viewer_leaving_closes_the_upstream(client, cam):
    """A departed tab MUST drop its upstream subscriber. The old MJPEG pump did
    not (BaseHTTPMiddleware swallowed http.disconnect) and leaked one camera
    reader per viewer — a known fd-leak source on the robot host."""
    with client.websocket_connect("/ws/cam/main/h264") as ws:
        ws.receive_bytes()
    assert cam.connections == 1
    for _ in range(50):                      # let the unwind land
        if cam.closed == 1:
            break
        import time
        time.sleep(0.05)
    assert cam.closed == 1, "upstream H.264 subscriber leaked after the viewer left"


def test_camera_offline_closes_cleanly_so_the_browser_falls_back(monkeypatch):
    """No camera server / no /h264 route (older build) must close the socket, not
    hang: camview.js only reveals the MJPEG fallback once H.264 fails."""
    from control_center.server import config as CFG
    monkeypatch.setattr(CFG, "VIDEO_PORT", 1)        # nothing listens here
    monkeypatch.setattr(CFG, "CC_REQUIRE_LOGIN", False)
    from control_center.server import app as app_mod
    with TestClient(app_mod.create_app()) as c:
        with pytest.raises(Exception):
            with c.websocket_connect("/ws/cam/main/h264") as ws:
                ws.receive_bytes()
