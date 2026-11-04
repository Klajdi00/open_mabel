"""cameras.py — the camera suite as a stream DIRECTORY, not a frame source.

Frames deliberately do NOT flow through the Python API: the per-process video
plane (MJPEG :8080, UDP JPEG :9091, H.264 :9092, dedicated :9620-9623) already
fans out to every client with proven latency, and an in-process ndarray path
would put JPEG decode back inside the bridge GIL — exactly what the
process-per-camera split fixed. A consumer that needs pixels picks a URL/port
from here and attaches like every other client.

``alive`` is a cheap cached probe of the MJPEG server (the relay process serves
it, so it doubles as "the camera plane is up").
"""
from __future__ import annotations

import threading
import time
import urllib.request
from typing import Optional, Tuple

from ..types import CameraStreamInfo, monotonic

__all__ = ["CameraSuite"]

_PROBE_TTL_S = 2.0

# name -> (its dedicated-plane config key, present-by-default)
_DEDICATED_KEYS = {
    "head_left": "head_sbs_port",
    "wrist_left": "wrist_left_port",
    "wrist_right": "wrist_right_port",
    "front": "front_port",
}
STREAMS = ("main", "front", "head_left", "wrist_left", "wrist_right")


class CameraSuite:
    def __init__(self, cfg: dict, *, host: str = "127.0.0.1", logger=None):
        self._cfg = cfg
        self._host = host
        self._log = logger or (lambda *a, **k: None)
        self._lock = threading.Lock()
        self._alive: Optional[bool] = None
        self._alive_at = 0.0

    # ── liveness (cached HTTP probe of the MJPEG server) ──────────────────────
    def _plane_alive(self) -> bool:
        now = time.monotonic()
        with self._lock:
            if self._alive is not None and now - self._alive_at < _PROBE_TTL_S:
                return self._alive
        alive = False
        try:
            url = f"http://{self._host}:{self._cfg['http_port']}/"
            with urllib.request.urlopen(url, timeout=0.3):
                alive = True
        except Exception:
            alive = False
        with self._lock:
            self._alive, self._alive_at = alive, now
        return alive

    # ── the directory ─────────────────────────────────────────────────────────
    def streams(self) -> Tuple[CameraStreamInfo, ...]:
        alive = self._plane_alive()
        udp = self._cfg.get("udp", {})
        dedicated = self._cfg.get("dedicated", {})
        jpeg = ((self._host, int(udp["jpeg_port"]))
                if udp.get("enabled") and udp.get("jpeg") else None)
        h264 = ((self._host, int(udp["h264_port"]))
                if udp.get("enabled") and udp.get("h264") else None)
        out = []
        for name in STREAMS:
            dport = None
            if dedicated.get("enabled") and name in _DEDICATED_KEYS:
                dport = int(dedicated[_DEDICATED_KEYS[name]])
            out.append(CameraStreamInfo(
                name=name, alive=alive,
                mjpeg_url=(f"http://{self._host}:{self._cfg['http_port']}"
                           f"/camera/{name}/stream.mjpg"),
                udp_jpeg=jpeg, udp_h264=h264, dedicated_port=dport))
        return tuple(out)

    def url(self, name: str, transport: str = "mjpeg") -> str:
        for info in self.streams():
            if info.name == name:
                if transport == "mjpeg":
                    return info.mjpeg_url
                if transport == "h264" and info.udp_h264:
                    return f"udp://{info.udp_h264[0]}:{info.udp_h264[1]}"
                if transport == "dedicated" and info.dedicated_port:
                    return f"udp://{self._host}:{info.dedicated_port}"
                raise KeyError(f"camera {name!r} has no {transport!r} transport")
        raise KeyError(f"unknown camera stream {name!r}; have {STREAMS}")
