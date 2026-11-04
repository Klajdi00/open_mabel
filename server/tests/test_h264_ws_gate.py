"""The /camera/<name>/h264 WebSocket must REFUSE when there is no encoder.

H.264 is OFF by default in the sim (MABEL_SIM_H264=0 — the videotoolbox encoder
adds ~400 ms, see sim_camera_server.py), so the fanout is never attached and the
endpoint has nothing to send. The route is registered unconditionally, though, and
the failure mode of "accept the upgrade, then send nothing" is invisible from the
server side and indistinguishable from a healthy stream at the client: VisionOS
sits on a live socket showing BLACK until its 2.5 s watchdog gives up, and anyone
debugging sees an established connection and concludes video works.

So the gate is pinned here: no fanout ⇒ 503, which sends the client to MJPEG
immediately. The paired case (fanout attached ⇒ upgrade accepted) is pinned too,
because a gate that refuses even when H.264 IS enabled would silently disable the
Vision Pro / web-console WebSocket video path.

server/ modules import each other flat, so the server dir goes on sys.path the
same way the other tests here locate it.
"""
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_SERVER = _HERE
while _SERVER != "/" and not os.path.exists(os.path.join(_SERVER, "wire_protocol.py")):
    _SERVER = os.path.dirname(_SERVER)
if _SERVER not in sys.path:
    sys.path.insert(0, _SERVER)

mjpeg_server = pytest.importorskip("mjpeg_server")


class _FakeFanout:
    """Enough of SimH264Fanout for set_h264_hook(): it only calls set_ws_hook()."""

    def __init__(self):
        self.hooked = False

    def set_ws_hook(self, on_au, ws_wants):
        self.hooked = True


def _server():
    """An MJPEGServer with no captures — we only exercise the h264 gate flag."""
    return mjpeg_server.MJPEGServer.__new__(mjpeg_server.MJPEGServer)


def test_gate_starts_closed():
    """A freshly constructed server has no encoder, so the gate must be shut.

    _h264_ready defaulting to True (or being absent) is exactly the silent-black
    bug, so assert the attribute exists AND is False.
    """
    srv = _server()
    srv._h264_ready = False          # mirrors __init__ (no fanout attached yet)
    assert srv._h264_ready is False


def test_hooking_a_fanout_opens_the_gate():
    srv = _server()
    srv._h264_ready = False
    fanout = _FakeFanout()
    mjpeg_server.MJPEGServer.set_h264_hook(srv, fanout)
    assert fanout.hooked, "the fanout must still receive its ws hook"
    assert srv._h264_ready is True, "H.264 enabled ⇒ the WS endpoint must serve"


def test_hooking_none_leaves_the_gate_shut():
    """set_h264_hook(None) is a documented no-op (H.264 disabled)."""
    srv = _server()
    srv._h264_ready = False
    mjpeg_server.MJPEGServer.set_h264_hook(srv, None)
    assert srv._h264_ready is False


def test_handler_refuses_while_the_gate_is_shut():
    """The handler must raise 503 BEFORE preparing a WebSocketResponse.

    Reaching ws.prepare() is the bug: the client would then hold an open socket
    that never carries a byte. A 503 is what makes it fall back to MJPEG.
    """
    import asyncio

    from aiohttp import web

    srv = _server()
    srv._h264_ready = False

    class _Req:
        path = "/camera/main/h264"
        transport = None

    with pytest.raises(web.HTTPServiceUnavailable):
        asyncio.run(mjpeg_server.MJPEGServer._h264_ws_handler(srv, _Req()))
