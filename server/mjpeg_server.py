"""
mjpeg_server.py — Async HTTP server that streams cameras as
multipart/x-mixed-replace (MJPEG), exactly the format MJPEGVideoStream.swift
expects.

One WebcamCapture instance is created per unique device. If two CameraSource
entries share the same device index they share one capture thread (saving CPU).
Each CameraSource gets its own MJPEG path.

Supports:
  • USB cameras (OpenCV device index)
  • RTSP / HTTP streams (pass URL string as device)
  • Stereo cameras (two separate CameraSource entries)
  • Any number of cameras — just add CameraSource entries in config.py

Dependencies:
  pip install aiohttp opencv-python
"""
from __future__ import annotations

import asyncio
import logging
import socket
import threading
import time
from typing import AsyncIterator

import cv2
import numpy as np
from aiohttp import web

from config import CameraSource
from udp_video import stream_name

log = logging.getLogger("mjpeg_server")

BOUNDARY = b"frame"
CONTENT_TYPE = f"multipart/x-mixed-replace; boundary={BOUNDARY.decode()}"


def _clamp_int(v, lo, hi):
    try:
        return max(lo, min(hi, int(v))) if v is not None else None
    except (TypeError, ValueError):
        return None


def _clamp_float(v, lo, hi):
    try:
        return max(lo, min(hi, float(v))) if v is not None else None
    except (TypeError, ValueError):
        return None


def _transcode_jpeg(jpeg: bytes, q: int, scale: float | None) -> bytes:
    """Decode a JPEG, optionally downscale by `scale` (area-averaged), re-encode at
    quality `q`. Used per-client to fit a slow link. Returns the original
    bytes on any failure so the stream never breaks."""
    try:
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return jpeg
        if scale is not None and scale < 0.999:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(q)])
        return buf.tobytes() if ok else jpeg
    except Exception:
        return jpeg


# ---------------------------------------------------------------------------
# Per-device capture (shared when multiple paths use the same device)
# ---------------------------------------------------------------------------

class DeviceCapture:
    """
    Wraps one physical camera / stream. Each call to frames() gets its own
    private asyncio.Queue so multiple concurrent HTTP clients (and multiple
    MJPEG paths sharing the same device) all receive every frame independently.
    """

    def __init__(self, source: CameraSource, jpeg_quality: int = 80):
        self.source = source
        self._quality = jpeg_quality
        # Per-subscriber queues — each frames() call registers one
        self._subscribers: list[asyncio.Queue[bytes]] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._last_error: str | None = None
        # Liveness counters for /cameras.json — same contract as PushCapture.
        # MUST NOT be named `frames`: that is the async generator the MJPEG HTTP
        # handler iterates, and an int attribute of the same name shadows it.
        self._frame_count = 0
        self._last_frame_t = 0.0

    # ------------------------------------------------------------------

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        if self._running:
            return
        self._loop = loop
        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop, name=f"cam-{self.source.device}", daemon=True
        )
        self._thread.start()
        log.info("Camera %-20s  device=%s", self.source.name, self.source.device)

    def stop(self) -> None:
        self._running = False

    @property
    def last_error(self) -> str | None:
        return self._last_error

    # ------------------------------------------------------------------

    @staticmethod
    def _make_placeholder(text: str, w: int = 640, h: int = 480) -> bytes:
        """Black frame with a status label — served when a camera is unavailable."""
        import numpy as np
        img = np.zeros((h, w, 3), dtype="uint8")
        cv2.putText(img, text, (20, h // 2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 2)
        cv2.putText(img, "Waiting for camera…", (20, h // 2 + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 60), 1)
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 60])
        return buf.tobytes()

    def _capture_loop(self) -> None:
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self._quality]
        interval = 1.0 / max(1, self.source.fps)
        cap: cv2.VideoCapture | None = None
        last_frame_t = 0.0
        fail_count = 0
        MAX_LOG_FAILURES = 3       # only log the first N consecutive failures
        BACKOFF_MAX = 10.0         # cap reconnect wait at 10 s

        while self._running:
            # ---- (re)open ----
            if cap is None or not cap.isOpened():
                cap = self._open_cap()
                if cap is None:
                    backoff = min(2.0 * (fail_count + 1), BACKOFF_MAX)
                    fail_count += 1
                    # Serve a placeholder frame so the MJPEG client stays alive
                    if self._loop and self._subscribers:
                        placeholder = self._make_placeholder(self.source.name)
                        asyncio.run_coroutine_threadsafe(
                            self._put_frame(placeholder), self._loop
                        )
                    time.sleep(backoff)
                    continue
                fail_count = 0

            # ---- rate limit ----
            now = time.monotonic()
            if now - last_frame_t < interval:
                time.sleep(0.001)
                continue
            last_frame_t = now

            # ---- read ----
            ok, frame = cap.read()
            if not ok:
                fail_count += 1
                if fail_count <= MAX_LOG_FAILURES:
                    log.warning("Camera %s: read failed (attempt %d), reconnecting…",
                                self.source.name, fail_count)
                elif fail_count == MAX_LOG_FAILURES + 1:
                    log.warning("Camera %s: repeated failures — silencing until reconnected",
                                self.source.name)
                cap.release()
                cap = None
                backoff = min(0.5 * fail_count, BACKOFF_MAX)
                time.sleep(backoff)
                continue
            fail_count = 0

            # ---- resize if requested ----
            if self.source.width and self.source.height:
                frame = cv2.resize(frame, (self.source.width, self.source.height))

            # ---- encode ----
            ok2, buf = cv2.imencode(".jpg", frame, encode_params)
            if not ok2:
                continue

            self._frame_count += 1
            self._last_frame_t = time.time()
            assert self._loop is not None
            asyncio.run_coroutine_threadsafe(
                self._put_frame(buf.tobytes()), self._loop
            )

        if cap:
            cap.release()

    def _open_cap(self) -> cv2.VideoCapture | None:
        device = self.source.device
        cap = cv2.VideoCapture(device)
        if not cap.isOpened():
            self._last_error = f"Cannot open device {device!r}"
            log.error("Camera %s: %s", self.source.name, self._last_error)
            return None
        if self.source.width and self.source.height:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.source.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.source.height)
        self._last_error = None
        log.info("Camera %-20s  opened  %dx%d @ %d fps",
                 self.source.name,
                 int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                 int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                 int(cap.get(cv2.CAP_PROP_FPS) or self.source.fps))
        return cap

    async def _put_frame(self, data: bytes) -> None:
        """Deliver one JPEG to every registered subscriber queue."""
        for q in list(self._subscribers):
            try:
                q.put_nowait(data)
            except asyncio.QueueFull:
                try:
                    q.get_nowait()   # evict oldest for this subscriber
                except asyncio.QueueEmpty:
                    pass
                try:
                    q.put_nowait(data)
                except asyncio.QueueFull:
                    pass   # subscriber is too slow — skip this frame

    async def frames(self) -> AsyncIterator[bytes]:
        """Each caller gets its own queue — no frame stealing between clients."""
        # maxsize=1 = always-freshest: a slow client never accumulates a backlog
        # of stale frames (the dominant teleop latency). On overflow the producer
        # evicts the old frame and queues the new one, so we always show "now".
        q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=1)
        self._subscribers.append(q)
        try:
            while True:
                yield await q.get()
        finally:
            try:
                self._subscribers.remove(q)
            except ValueError:
                pass


# ---------------------------------------------------------------------------
# Push-based capture — frames are supplied by an external producer (e.g. the
# MuJoCo sim renderer) instead of being read from a cv2 device. Shares the same
# per-subscriber fan-out as DeviceCapture so the MJPEG HTTP handler is identical.
# ---------------------------------------------------------------------------

class PushCapture:
    """
    A camera whose JPEG frames are pushed in from another thread via push().

    Used by sim_teleop_bridge.py: the MuJoCo render loop (main thread) renders a
    named camera, encodes it to JPEG, and calls push(); the asyncio MJPEG handler
    (network thread) fans the bytes out to every connected video client
    (Vision Pro, iOS app, web browser, …).
    """

    def __init__(self, source: CameraSource):
        self.source = source
        self._subscribers: list[asyncio.Queue[bytes]] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        self._latest: bytes | None = None
        self._last_error: str | None = "no frames yet"
        # Liveness counters for /cameras.json (the module-status roster). NOTE a
        # zero frame count is NOT "broken": sim cameras render ON DEMAND, so a
        # camera nobody is watching legitimately has no frames. Only the pairing
        # (has_clients AND no frames) means something is actually wrong — the
        # roster ships the raw facts and module_status.py makes that call.
        # MUST NOT be named `frames`: that is the async generator the MJPEG HTTP
        # handler iterates, and an int attribute of the same name shadows it —
        # every stream then 200s and immediately dies with "int is not callable".
        self._frame_count = 0
        self._last_frame_t = 0.0
        # Optional tee called synchronously from push() with every JPEG —
        # udp_video.UDPVideoSender.attach() installs the low-latency UDP path
        # here without touching the HTTP/MJPEG machinery.
        self.on_push = None

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Bind the asyncio loop that owns the subscriber queues. Call once the
        network event loop is running, before push() is used."""
        self._loop = loop

    @property
    def last_error(self) -> str | None:
        return self._last_error

    @property
    def has_clients(self) -> bool:
        """True iff an MJPEG/HTTP client is currently streaming this path. Lets
        the render loop skip cameras nobody is watching (demand-driven render)."""
        return len(self._subscribers) > 0

    def push(self, jpeg: bytes) -> None:
        """Deliver one JPEG to all subscribers. Thread-safe — schedules the queue
        writes on the network loop. Safe to call before any client connects."""
        self._latest = jpeg
        self._last_error = None
        self._frame_count += 1
        self._last_frame_t = time.time()
        if self.on_push is not None:
            try:
                self.on_push(jpeg)
            except Exception:
                log.exception("on_push tee failed for %s", self.source.path)
        loop = self._loop
        if loop is None:
            return
        loop.call_soon_threadsafe(self._deliver, jpeg)

    def _deliver(self, jpeg: bytes) -> None:
        for q in list(self._subscribers):
            try:
                q.put_nowait(jpeg)
            except asyncio.QueueFull:
                try:
                    q.get_nowait()        # evict oldest for this subscriber
                except asyncio.QueueEmpty:
                    pass
                try:
                    q.put_nowait(jpeg)
                except asyncio.QueueFull:
                    pass

    async def frames(self) -> AsyncIterator[bytes]:
        # maxsize=1 = always-freshest: a slow client never accumulates a backlog
        # of stale frames (the dominant teleop latency). On overflow the producer
        # evicts the old frame and queues the new one, so we always show "now".
        q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=1)
        if self._latest is not None:
            q.put_nowait(self._latest)    # show the newest frame immediately
        self._subscribers.append(q)
        try:
            while True:
                yield await q.get()
        finally:
            try:
                self._subscribers.remove(q)
            except ValueError:
                pass


# ---------------------------------------------------------------------------
# MJPEG HTTP server (one aiohttp app, N paths)
# ---------------------------------------------------------------------------

class MJPEGServer:
    """
    One aiohttp app serving all camera paths. Cameras that share a device
    index share one DeviceCapture instance so the webcam is only opened once.

    Two construction modes:
      • MJPEGServer(sources=...)        — cv2 device / RTSP captures (webcam proxy)
      • MJPEGServer.from_captures({...}) — pre-built PushCaptures fed by the sim
    """

    def __init__(self, sources: list[CameraSource] | None = None,
                 jpeg_quality: int = 80,
                 path_captures: dict[str, "DeviceCapture | PushCapture"] | None = None):
        # Deduplicate captures by device key
        self._captures: dict = {}
        # path → capture
        self._path_map: dict[str, "DeviceCapture | PushCapture"] = {}

        if path_captures is not None:
            self._path_map = dict(path_captures)
            # Track unique capture objects so start/stop iterate each once.
            for cap in path_captures.values():
                self._captures[id(cap)] = cap
        else:
            for src in (sources or []):
                key = src.device
                if key not in self._captures:
                    self._captures[key] = DeviceCapture(src, jpeg_quality)
                self._path_map[src.path] = self._captures[key]

        # H.264-over-WebSocket: a SimH264Fanout (sim_h264.py) feeds Annex-B access
        # units here; browsers (WebCodecs), which can't receive UDP, decode the SAME
        # stream as the UDP :9092 path. name -> {asyncio.Queue}. Installed via set_h264_hook(); stays inert
        # (no routes used) until a fanout is attached and a client connects.
        self._h264_ws: dict[str, set] = {}
        self._h264_loop: asyncio.AbstractEventLoop | None = None
        # True once a fanout is attached. H.264 is OFF by default in the sim
        # (MABEL_SIM_H264=0 — videotoolbox adds ~400 ms), and without this flag the
        # route below still ACCEPTS the upgrade and then never sends a byte: the
        # client sits on a healthy-looking socket showing black instead of falling
        # back to MJPEG. Refuse the upgrade so the fallback is immediate.
        self._h264_ready = False

        self._app = web.Application()
        for path in self._path_map:
            self._app.router.add_get(path, self._stream_handler)
            name = stream_name(path)
            self._h264_ws.setdefault(name, set())
            self._app.router.add_get(f"/camera/{name}/h264", self._h264_ws_handler)
        self._app.router.add_get("/", self._index_handler)
        # Camera ROSTER (JSON) — who is actually attached to this robot, for the
        # app's module-status panel (module_status.py polls it over localhost).
        # Deliberately NOT a stream: opening an MJPEG path would SUBSCRIBE, and
        # the sim renders on demand — probing liveness that way would pin every
        # camera's renderer on forever. This route only reads counters.
        self._app.router.add_get("/cameras.json", self._roster_handler)

    def add_console(self, console_dir: str) -> bool:
        """Serve the browser operator console (website/teleop-web.html + assets)
        from this HTTP server. A page loaded from https:// cannot open ws:// to
        a LAN host (mixed content), but http://<bridge>:8080/console can — and
        the camera streams become same-origin too."""
        import os
        page = os.path.join(console_dir, "teleop-web.html")
        assets = os.path.join(console_dir, "assets")
        if not (os.path.isfile(page) and os.path.isdir(assets)):
            return False

        async def _console(_req):
            return web.FileResponse(page)
        self._app.router.add_get("/console", _console)
        self._app.router.add_static("/assets", assets)
        return True

    @classmethod
    def from_captures(cls, path_captures: dict[str, "DeviceCapture | PushCapture"]) -> "MJPEGServer":
        """Build a server that serves already-constructed captures (PushCapture
        for sim-rendered cameras). The path→capture map sets the served routes."""
        return cls(path_captures=path_captures)

    def start_captures(self, loop: asyncio.AbstractEventLoop) -> None:
        self._h264_loop = loop           # for thread-safe AU enqueue from the encoder
        for cap in self._captures.values():
            if hasattr(cap, "start"):
                cap.start(loop)          # DeviceCapture: spawn capture thread
            elif hasattr(cap, "set_loop"):
                cap.set_loop(loop)       # PushCapture: bind the network loop

    def set_h264_hook(self, fanout) -> None:
        """Wire a SimH264Fanout (sim_h264.py) to the WebSocket H.264 endpoint: the
        fanout sends each ws-profile access unit to _ws_on_au, and gates its ws
        encoder on _ws_wants (one encoder per camera, started only while a WS client is
        connected). Safe to call with fanout=None (no-op)."""
        if fanout is not None:
            fanout.set_ws_hook(self._ws_on_au, self._ws_wants)
            self._h264_ready = True

    def stop_captures(self) -> None:
        for cap in self._captures.values():
            if hasattr(cap, "stop"):
                cap.stop()

    def app(self) -> web.Application:
        return self._app

    # ------------------------------------------------------------------

    async def _roster_handler(self, request: web.Request) -> web.Response:
        """Which cameras this server has, and whether each is healthy.

        Raw facts only — the caller decides what "connected" means, because the
        answer differs by world: a sim camera with 0 frames and no viewer is
        IDLE (fine, it renders on demand), while a real USB camera that never
        produced a frame while someone was watching is BROKEN. Shipping a single
        cooked `ok` here would have to guess which case it is.
        """
        cams = []
        now = time.time()
        for path, cap in self._path_map.items():
            frames = int(getattr(cap, "_frame_count", 0))
            last = float(getattr(cap, "_last_frame_t", 0.0))
            cams.append({
                "name": stream_name(path),
                "path": path,
                "label": getattr(cap.source, "name", ""),
                "error": cap.last_error,             # None once a frame lands
                "frames": frames,
                "ageMs": None if last <= 0 else int((now - last) * 1000),
                # someone is watching NOW. DeviceCapture has no `has_clients`
                # property (only PushCapture does) — fall back to its queue list
                # rather than exploding on the REAL camera server, which builds
                # DeviceCaptures from USB devices.
                "streaming": bool(getattr(cap, "has_clients",
                                          len(getattr(cap, "_subscribers", [])) > 0)),
            })
        return web.json_response({"cameras": cams, "t": now})

    async def _index_handler(self, request: web.Request) -> web.Response:
        rows = []
        for path, cap in self._path_map.items():
            err = cap.last_error or ""
            status = f"<span style='color:red'>{err}</span>" if err else "<span style='color:green'>live</span>"
            rows.append(
                f"<tr><td><a href='{path}'>{path}</a></td>"
                f"<td>{cap.source.name}</td>"
                f"<td>{cap.source.device}</td>"
                f"<td>{status}</td></tr>"
            )
        table = "".join(rows)
        html = (
            "<h1>MABEL MJPEG Proxy</h1>"
            "<table border='1' cellpadding='6'>"
            "<tr><th>Path</th><th>Camera</th><th>Device</th><th>Status</th></tr>"
            f"{table}</table>"
        )
        return web.Response(text=html, content_type="text/html")

    async def _stream_handler(self, request: web.Request) -> web.StreamResponse:
        capture = self._path_map.get(request.path)
        if capture is None:
            raise web.HTTPNotFound()

        peer = request.transport.get_extra_info("peername") if request.transport else "?"
        if isinstance(peer, (tuple, list)) and len(peer) >= 2:
            peer = f"{peer[0]}:{peer[1]}"

        # Per-client bitrate: a client on a slow link (MJPEG rides TCP — one big
        # frame can stall the whole stream and queue) asks for a smaller frame via ?q=<10-95>&scale=<0.1-1.0>. We
        # decode the freshest full-res JPEG and re-encode it downscaled, per
        # client, so LAN viewers (no params) keep the crisp frame untouched while
        # a remote headset gets a light one that actually fits the link. The
        # render/GPU cost is unchanged — only the bytes on the wire shrink.
        q = _clamp_int(request.query.get("q"), 10, 95)
        scale = _clamp_float(request.query.get("scale"), 0.1, 1.0)
        transcode = q is not None or (scale is not None and scale < 0.999)
        log.info("stream open    %-32s ← %s%s", request.path, peer,
                 f"  [q={q} scale={scale}]" if transcode else "")

        response = web.StreamResponse(
            headers={"Content-Type": CONTENT_TYPE, "Cache-Control": "no-cache"}
        )
        sock = request.transport.get_extra_info("socket") if request.transport else None
        if sock is not None:
            try:
                # Disable Nagle: MJPEG frames are large and latency-sensitive —
                # without TCP_NODELAY the kernel can hold a frame up to ~40 ms
                # coalescing writes.
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                # Cap the kernel send buffer to ~1 frame.
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 128 * 1024)
            except OSError:
                pass
        # The dominant backlog is aiohttp's OWN (userspace) write buffer, which by
        # default grows unbounded — a client that displays slower than we produce
        # (e.g. a phone) accumulates dozens of stale JPEGs = up to ~1.5 s of lag.
        # Cap it to ~1 frame so write()/drain() applies backpressure almost
        # immediately; while the handler is paused the upstream maxsize=1 queue
        # drops intermediate frames, so the client always resumes on the FRESHEST
        # frame. Glass-to-glass stays ≈ one frame regardless of client speed.
        if request.transport is not None:
            try:
                request.transport.set_write_buffer_limits(high=131072, low=32768)
            except (AttributeError, NotImplementedError):
                pass
        await response.prepare(request)

        loop = asyncio.get_event_loop()
        q_enc = q if q is not None else 70
        try:
            async for jpeg in capture.frames():
                if transcode:
                    # Off the event loop: decode+resize+re-encode is ~3-5 ms and
                    # would otherwise stall every other client's writes.
                    jpeg = await loop.run_in_executor(
                        None, _transcode_jpeg, jpeg, q_enc, scale)
                part = (
                    b"--" + BOUNDARY + b"\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n"
                    b"\r\n" + jpeg + b"\r\n"
                )
                await response.write(part)
        except (ConnectionResetError, asyncio.CancelledError):
            pass
        finally:
            log.info("stream closed  %-32s ← %s", request.path, peer)

        return response

    # ------------------------------------------------------------------
    # H.264 over WebSocket — one binary message == one Annex-B access unit. The
    # SAME encoder access units the UDP :9092 path sends; browsers (WebCodecs)
    # decode them. A late joiner starts at the next in-band IDR (~0.5 s on the ws
    # profile). Ported from hardware_bridge/cameras/video_server.py.
    # ------------------------------------------------------------------

    async def _h264_ws_handler(self, request: web.Request) -> web.WebSocketResponse:
        name = stream_name(request.path)             # /camera/<name>/h264
        peer = request.transport.get_extra_info("peername") if request.transport else "?"
        if isinstance(peer, (tuple, list)) and len(peer) >= 2:
            peer = f"{peer[0]}:{peer[1]}"
        if not self._h264_ready:
            # No encoder attached → this socket would be silent forever. 503 tells the
            # client to use MJPEG NOW rather than staring at a black stream.
            log.info("h264 ws refused %-31s ← %s (H.264 disabled — client uses MJPEG)",
                     request.path, peer)
            raise web.HTTPServiceUnavailable(text="H.264 disabled (set MABEL_SIM_H264=1)")
        ws = web.WebSocketResponse(max_msg_size=0)
        await ws.prepare(request)
        log.info("h264 ws open   %-32s ← %s", request.path, peer)
        # Bounded per-client queue: H.264 is inter-frame, so on overflow (slow client)
        # we CLEAR and let it resync at the next IDR rather than ship a torn GOP.
        q: asyncio.Queue = asyncio.Queue(maxsize=120)
        self._h264_ws.setdefault(name, set()).add(q)

        async def _drain():
            while True:
                au = await q.get()
                await ws.send_bytes(au)

        async def _reader():
            async for _ in ws:                       # consume pings/close; exit on close
                pass

        drain = asyncio.ensure_future(_drain())
        reader = asyncio.ensure_future(_reader())
        try:
            await asyncio.wait({drain, reader}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            drain.cancel(); reader.cancel()
            self._h264_ws.get(name, set()).discard(q)
            log.info("h264 ws closed %-32s ← %s", request.path, peer)
        return ws

    def _ws_wants(self, name: str) -> bool:
        return bool(self._h264_ws.get(name))

    def _ws_on_au(self, name: str, au: bytes) -> None:
        """Called from the encoder read thread: enqueue one access unit to every WS
        client of `name` (scheduled onto the network loop; latest-GOP-wins on overflow)."""
        loop = self._h264_loop
        if loop is None:
            return
        queues = list(self._h264_ws.get(name, ()))
        for q in queues:
            loop.call_soon_threadsafe(self._enqueue_au, q, au)

    @staticmethod
    def _enqueue_au(q: "asyncio.Queue", au: bytes) -> None:
        if q.full():
            while not q.empty():                     # drop the stale GOP; resync at next IDR
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    break
        try:
            q.put_nowait(au)
        except asyncio.QueueFull:
            pass
