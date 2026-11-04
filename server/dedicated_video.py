"""
dedicated_video.py — the 962x dedicated-port video plane (one MJPEG stream per port).

Why dedicated ports: :8080 muxes every camera through one HTTP server, so a
burst on one stream (or a slow client) contends with all of them. Each 962x
port serves exactly ONE stream — no mux contention, and a client opts in with
just a port change (the path keeps the familiar /camera/<name>/stream.mjpg
shape, so existing MJPEG clients need no new parsing).

  9620  head_sbs     — the ONLY high-res stream: one side-by-side stereo frame.
                       Real: the head cam's native hardware-stitched SBS frame.
                       Sim: zed_left|zed_right composited at native render res.
  9621  wrist_left   — standard res, a tee of the same JPEGs :8080 serves.
  9622  wrist_right  — standard res tee.
  9623  front        — standard res tee.

A stream whose source doesn't exist in this world (e.g. front on the Mac
`--real` rig) still binds its port and answers HTTP 503 with a JSON
explanation — never a refused connection.

Demand contract matches the camera servers: nothing is composited/encoded for
an unwatched port (`wants()` gates the producers), so the plane costs zero
until a client actually attaches. MJPEG delivery itself is newest-only
(PushCapture's one-slot subscriber queues).
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
import time

import cv2
import numpy as np

from config import CameraSource
from mjpeg_server import MJPEGServer, PushCapture
import data_streams

__all__ = ["DedicatedVideoPlane"]


class DedicatedVideoPlane:
    """One MJPEG listener per dedicated video port + the sim SBS compositor."""

    LABELS = {"head_sbs": "Head Stereo SBS (hi-res)",
              "wrist_left": "Wrist Left (dedicated)",
              "wrist_right": "Wrist Right (dedicated)",
              "front": "Front Base (dedicated)"}

    def __init__(self, available: dict[str, bool], log,
                 host: str = "0.0.0.0", sbs_quality: int | None = None):
        """`available[key]=False` binds the port as an HTTP-503 notice instead
        of a stream (explanation, never a refusal)."""
        self.log = log
        self.host = host
        self.ports = dict(data_streams.DEDICATED_VIDEO_PORTS)
        self.enabled = data_streams.streams_enabled()
        self._caps: dict[str, PushCapture] = {}
        self._notices: dict[str, str] = {}
        for key in self.ports:
            if available.get(key, False):
                path = f"/camera/{key}/stream.mjpg"
                self._caps[key] = PushCapture(
                    CameraSource(name=self.LABELS.get(key, key), device="dedicated", path=path))
            else:
                self._notices[key] = available.get(f"{key}_reason") if isinstance(
                    available.get(f"{key}_reason"), str) else "no source for this stream on this rig"
        # SBS compositor state (sim): latest per-eye BGR, written by the two eye
        # render threads; composed on the LEFT eye's cadence (it paces the head).
        self._eye: dict[str, np.ndarray | None] = {"left": None, "right": None}
        self._sbs_quality = int(os.getenv("MABEL_SBS_QUALITY", str(sbs_quality or 88)))

    # ── producer side (render/capture threads) ──────────────────────────────
    def wants(self, key: str) -> bool:
        cap = self._caps.get(key)
        return bool(cap is not None and cap.has_clients)

    def push(self, key: str, jpeg: bytes) -> None:
        """Tee an already-encoded JPEG onto a dedicated port (free — no encode)."""
        cap = self._caps.get(key)
        if cap is not None:
            cap.push(jpeg)

    def push_eye(self, side: str, bgr: np.ndarray) -> None:
        """Sim head compositor: store the newest eye; on a LEFT update with both
        eyes present, hstack → one hi-res SBS JPEG → :9620. Costs nothing while
        the port has no client (callers gate on wants('head_sbs') too)."""
        self._eye[side] = bgr
        if side != "left" or not self.wants("head_sbs"):
            return
        right = self._eye["right"]
        if right is None:
            return
        left = self._eye["left"]
        if left is None or left.shape[0] != right.shape[0]:
            return
        sbs = np.hstack((left, right))
        ok, buf = cv2.imencode(".jpg", sbs, [cv2.IMWRITE_JPEG_QUALITY, self._sbs_quality])
        if ok:
            self.push("head_sbs", buf.tobytes())

    # ── server side (own thread + asyncio loop, one listener per port) ──────
    def start(self) -> bool:
        if not self.enabled:
            return False
        ready = threading.Event()
        holder: dict = {}
        threading.Thread(target=self._serve_thread, args=(ready, holder),
                         name="dedicated-video", daemon=True).start()
        ready.wait(timeout=10.0)
        return not holder.get("failed", False)

    def _serve_thread(self, ready: threading.Event, holder: dict) -> None:
        from aiohttp import web

        def _notice_handler(reason: str, port: int):
            body = json.dumps({"type": "notice", "available": False,
                               "reason": reason, "port": port})

            async def handler(request):
                return web.Response(status=503, text=body,
                                    content_type="application/json")
            return handler

        async def _serve():
            loop = asyncio.get_running_loop()
            for key, port in self.ports.items():
                try:
                    cap = self._caps.get(key)
                    if cap is not None:
                        srv = MJPEGServer.from_captures({cap.source.path: cap})
                        runner = web.AppRunner(srv.app())
                        await runner.setup()
                        await web.TCPSite(runner, self.host, port).start()
                        srv.start_captures(loop)
                        self.log.info("dedicated video  :%d  %s → %s",
                                      port, self.LABELS.get(key, key), cap.source.path)
                    else:
                        app = web.Application()
                        app.router.add_route(
                            "GET", "/{tail:.*}",
                            _notice_handler(self._notices.get(key, "unavailable"), port))
                        runner = web.AppRunner(app)
                        await runner.setup()
                        await web.TCPSite(runner, self.host, port).start()
                        self.log.info("dedicated video  :%d  %s → 503 notice (%s)",
                                      port, key, self._notices.get(key, "unavailable"))
                except OSError as e:
                    self.log.warning("dedicated video port :%d unavailable (%s) — skipped",
                                     port, e)
            ready.set()
            await asyncio.Future()

        try:
            asyncio.run(_serve())
        except Exception as e:                                     # noqa: BLE001
            self.log.error("dedicated video plane failed: %s", e)
            holder["failed"] = True
            ready.set()
