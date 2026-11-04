"""video_server.py — the bridge's OWN low-latency MJPEG server for the cameras it
owns.

The hardware bridge is the single capture owner. Each camera's capture loop fans
every frame out to TWO independent sinks (see wrist_cams.py):

    capture ─┬─▶ ROS  /camera/<name>/image_raw      (DDS graph: SLAM, RViz, learning)
             └─▶ this  /camera/<name>/stream.mjpg    (network: GUI, iOS, VP, web)

The network video path does NOT go through ROS — JPEG frames are pushed straight
in, so it stays fast and is unaffected by the DDS graph's reliability/latency. The
path contract (`/camera/<name>/stream.mjpg`) is IDENTICAL to the simulator's, so
every existing client (the Control Center tiles, the apps' MJPEG fallback) works
unchanged — only the host/port differs.

Self-contained: aiohttp only. JPEG encoding is the caller's job (it already holds
the BGR frame). Demand-driven + low-latency: a stream costs nothing until a client
connects; each client gets a maxsize=1 latest-wins queue and capped write buffers,
so a slow viewer drops stale frames instead of backlogging glass-to-glass latency.
"""

import asyncio
import socket
import threading
import time            # _mark_alive() / _roster() liveness timestamps


class VideoServer:
    """One aiohttp MJPEG server, N pushed streams. Thread-safe: ``push()`` is called
    from the camera capture threads; the HTTP server runs on its own asyncio loop in
    a daemon thread."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8081,
                 net_width: int = 640, jpeg_quality: int = 65, udp=None,
                 h264_width: int = 1280):
        self._host = host
        self._port = port
        # H.264 (UDP + WebSocket) runs at its OWN, larger width than the MJPEG/JPEG
        # fallback: inter-frame compression makes high-res cheap (~3-4 Mbps at 720p),
        # so browser/headset H.264 clients get a crisp HIGH-RES image at low latency
        # while the MJPEG fallback stays small (net_width) for a phone on a poor link.
        self._h264_width = int(h264_width)
        # Optional low-latency UDP sink (cameras/udp_sink.UdpVideoSink): H.264 (NVENC)
        # on :9092 + JPEG on :9091. The SAME downscaled frame feeds HTTP-MJPEG AND the
        # UDP transports; each is demand-driven and additive (MJPEG stays the fallback).
        self._udp = udp
        # Browser H.264: a WebSocket carries the SAME NVENC access units to WebCodecs
        # (browsers can't read raw UDP). One encoder feeds UDP + WS. name -> {(ws,queue)}.
        self._h264_ws: dict[str, set] = {}
        if self._udp is not None:
            self._udp.set_ws_hook(self._ws_on_au, self._ws_wants)
        # The NETWORK frame is decoupled from the camera's capture resolution: every
        # frame is downscaled so its long edge is <= net_width and JPEG-encoded at
        # jpeg_quality BEFORE it hits the wire. This is the single biggest latency
        # lever for MJPEG-over-WiFi — a 1280x720 frame is ~150 KB and a phone simply
        # cannot decode+display them as fast as we produce them, so they backlog into
        # seconds of lag; a 640-wide frame is ~15-25 KB and keeps glass-to-glass to a
        # frame or two. Tune via CAM_NET_WIDTH / CAM_NET_QUALITY.
        self._net_width = int(net_width)
        self._jpeg_quality = int(jpeg_quality)
        self._names: set[str] = set()
        self._fallback: dict[str, tuple] = {}        # name -> (fallback name, alive())
        self._latest: dict[str, bytes] = {}          # name -> last JPEG (warm start)
        self._clients: dict[str, set] = {}           # name -> {asyncio.Queue}
        # Liveness counters for /cameras.json — the roster the apps' module-status
        # panel reads (server/module_status.py). Counted on the PRODUCER side, not
        # from HTTP clients: this server is demand-driven, so an unwatched camera
        # legitimately pushes nothing, and "no frames" alone would libel a healthy
        # camera as dead. `alive` is the honest signal — the capture worker is
        # feeding us — and that is what the roster reports.
        self._frames: dict[str, int] = {}            # name -> frames produced
        self._last_t: dict[str, float] = {}          # name -> last produce time
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

    # -- setup -------------------------------------------------------------
    def register(self, *names: str) -> None:
        """Declare the stream paths to serve (``/camera/<name>/stream.mjpg``)."""
        for n in names:
            self._names.add(n)
            self._clients.setdefault(n, set())

    def set_fallback(self, name: str, fallback: str, alive=None) -> None:
        """Serve ``fallback``'s frames on ``name``'s URL while ``alive()`` is false.

        The apps hard-code '/camera/main/stream.mjpg' as the primary view; 'main'
        normally mirrors the ZED head cam, so with the ZED unplugged that URL was a
        DEAD stream (the iOS app then shows its local fallback render — looks like
        the sim). ``alive`` is a producer-liveness callable (e.g. "is the ZED open"),
        NOT a last-push timestamp: capture is demand-driven, so an unwatched live
        camera pushes nothing. Resolution happens per client CONNECT — a hot-plugged
        ZED takes effect on the client's next (re)connect."""
        self.register(name, fallback)
        self._fallback[name] = (fallback, alive or (lambda: False))

    def _resolve(self, name: str) -> str:
        fb = self._fallback.get(name)
        if fb is None:
            return name
        target, alive = fb
        try:
            return name if alive() else target
        except Exception:                            # noqa: BLE001
            return target

    def has_clients(self, name: str) -> bool:
        """True iff an MJPEG/HTTP client is currently streaming ``name``."""
        with self._lock:
            return bool(self._clients.get(name))

    def wants(self, name: str) -> bool:
        """True iff ANY sink (HTTP-MJPEG, UDP-JPEG, or UDP-H.264) is watching ``name``.
        The capture loop gates on this so an unwatched camera does no work at all."""
        if self.has_clients(name):
            return True
        return self._udp is not None and self._udp.wants(name)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, name="bridge-video",
                                        daemon=True)
        self._thread.start()

    # -- producer side (capture threads): fast, non-blocking ---------------
    def _downscale(self, bgr, target):
        """Downscale BGR so its width is <= target (area-averaged, ~1-2 ms). No-op if
        already <= target (never upscales)."""
        import cv2
        h, w = bgr.shape[:2]
        if target and w > target:
            nh = max(1, int(round(h * target / float(w))))
            bgr = cv2.resize(bgr, (target, nh), interpolation=cv2.INTER_AREA)
        return bgr

    def _encode(self, bgr) -> bytes | None:
        """JPEG-encode a (already downscaled) BGR frame at jpeg_quality."""
        import cv2
        ok, buf = cv2.imencode(".jpg", bgr,
                               [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality])
        return buf.tobytes() if ok else None

    def push(self, names, bgr, width=None) -> None:
        """Route ONE raw BGR frame to every watched sink for ``names`` (a str, or an
        iterable — the ZED feeds 'main' + 'head_left' from one frame):
          • HTTP-MJPEG + UDP-JPEG  share a SINGLE downscale+JPEG-encode per frame
          • UDP-H.264              gets the downscaled BGR fed to its NVENC encoder
        Fully demand-driven: each path runs only if a client is actually watching, so an
        unwatched camera costs nothing and an idle codec stays at ~0 CPU."""
        if isinstance(names, str):
            names = (names,)
        names = [n for n in names if n in self._names]
        if not names:
            return
        # Count BEFORE the demand-driven early-out below: a camera nobody is
        # watching still pushes frames here, and that is exactly the evidence the
        # module roster needs ("the capture worker is alive"). Counting inside the
        # want_* branches would report every unwatched camera as dead.
        self._mark_alive(names)
        udp = self._udp
        want_http = {n for n in names if self.has_clients(n)}
        want_ujpeg = {n for n in names if udp is not None and udp.wants_jpeg(n)}
        want_h264 = {n for n in names if udp is not None and udp.wants_h264(n)}
        if not (want_http or want_ujpeg or want_h264):
            return

        # JPEG (shared by HTTP-MJPEG + UDP-JPEG) — small + low-latency, encode once.
        if want_http or want_ujpeg:
            # `width` (e.g. the 2-eye 'head_stereo' frame at 2×net_width) keeps
            # each eye at the same width every other stream gets.
            jpeg = self._encode(self._downscale(bgr, int(width) if width else self._net_width))
            if jpeg is not None:
                loop = self._loop
                for n in want_http:
                    with self._lock:
                        self._latest[n] = jpeg
                    if loop is not None:
                        loop.call_soon_threadsafe(self._fanout, n, jpeg)
                for n in want_ujpeg:
                    udp.send_jpeg(n, jpeg)
        # H.264 (UDP + WebSocket) — its own higher-res frame to the NVENC encoder.
        if want_h264:
            big = self._downscale(bgr, 2 * self._h264_width if width else self._h264_width)
            for n in want_h264:
                udp.feed_h264(n, big)

    def push_jpeg(self, name: str, jpeg: bytes) -> None:
        """Route ONE pre-encoded JPEG (already net-sized, from a camera worker via
        the video pipe — see video_pipe.py) to every watched sink. The fan-out does
        NO pixel work on the JPEG path; H.264 (needs raw frames for NVENC) decodes
        the small JPEG only while an H.264 client is attached."""
        if name not in self._names:
            return
        self._mark_alive((name,))       # producer-side liveness (see push())
        udp = self._udp
        if self.has_clients(name):
            with self._lock:
                self._latest[name] = jpeg
            loop = self._loop
            if loop is not None:
                loop.call_soon_threadsafe(self._fanout, name, jpeg)
        if udp is not None and udp.wants_jpeg(name):
            udp.send_jpeg(name, jpeg)
        if udp is not None and udp.wants_h264(name):
            try:
                import cv2
                import numpy as np
                bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                if bgr is not None:
                    udp.feed_h264(name, bgr)
            except Exception:
                pass

    def _fanout(self, name: str, jpeg: bytes) -> None:
        with self._lock:
            queues = list(self._clients.get(name, ()))
        for q in queues:
            if q.full():
                try:
                    q.get_nowait()                   # evict stale frame for this client
                except Exception:                    # noqa: BLE001
                    pass
            try:
                q.put_nowait(jpeg)
            except Exception:                        # noqa: BLE001
                pass

    # -- HTTP server (own asyncio loop, daemon thread) ---------------------
    def _run(self) -> None:
        try:
            asyncio.run(self._serve())
        except OSError as e:
            print(f"[bridge-video] FAILED to bind {self._host}:{self._port} ({e}). "
                  f"Network camera streams are OFF (ROS publishing continues).",
                  flush=True)
        except Exception as e:                       # noqa: BLE001
            print(f"[bridge-video] server error: {e}", flush=True)

    async def _serve(self) -> None:
        from aiohttp import web
        self._loop = asyncio.get_running_loop()
        app = web.Application()
        for n in sorted(self._names):
            app.router.add_get(f"/camera/{n}/stream.mjpg", self._handler)
            app.router.add_get(f"/camera/{n}/h264", self._h264_ws_handler)
        app.router.add_get("/", self._index)
        app.router.add_get("/cameras.json", self._roster)   # module-status roster
        runner = web.AppRunner(app)
        await runner.setup()
        await web.TCPSite(runner, self._host, self._port).start()
        print(f"[bridge-video] MJPEG http://{self._host}:{self._port}"
              f"/camera/<name>/stream.mjpg  serving {sorted(self._names)}", flush=True)
        try:
            await asyncio.Future()                   # serve forever
        finally:
            await runner.cleanup()

    def _mark_alive(self, names) -> None:
        """A capture worker just produced a frame for these streams."""
        now = time.time()
        with self._lock:
            for n in names:
                self._frames[n] = self._frames.get(n, 0) + 1
                self._last_t[n] = now

    async def _roster(self, request):
        """/cameras.json — which cameras this robot has and whether each is alive.

        Same contract as the simulator's (server/mjpeg_server.py) so ONE client
        (module_status.py) reads both worlds. Raw facts only; the caller decides
        what "connected" means. Deliberately not a stream: opening an MJPEG path
        would subscribe, and this server is demand-driven — probing liveness that
        way would pin every camera's encoder on for as long as the poller ran.
        """
        from aiohttp import web
        now = time.time()
        cams = []
        with self._lock:
            for n in sorted(self._names):
                frames = self._frames.get(n, 0)
                last = self._last_t.get(n, 0.0)
                cams.append({
                    "name": n,
                    "path": f"/camera/{n}/stream.mjpg",
                    "label": n,
                    # A camera that has NEVER produced a frame is not connected.
                    # One that produced frames but has gone quiet is stale — the
                    # age says so, and module_status applies the TTL.
                    "error": None if frames > 0 else "no frames yet",
                    "frames": frames,
                    "ageMs": None if last <= 0 else int((now - last) * 1000),
                    "streaming": bool(self._clients.get(n)),
                })
        return web.json_response({"cameras": cams, "t": now})

    async def _index(self, request):
        from aiohttp import web
        links = "".join(f'<li><a href="/camera/{n}/stream.mjpg">{n}</a></li>'
                        for n in sorted(self._names))
        return web.Response(content_type="text/html",
                            text=f"<h1>MABEL bridge cameras</h1><ul>{links}</ul>")

    async def _handler(self, request):
        from aiohttp import web
        name = self._resolve(request.path.split("/")[2])   # /camera/<name>/stream.mjpg
        resp = web.StreamResponse(status=200, headers={
            "Content-Type": "multipart/x-mixed-replace; boundary=frame",
            "Cache-Control": "no-cache", "Connection": "close"})
        await resp.prepare(request)
        # LOW LATENCY: cap aiohttp's (otherwise unbounded) write buffer and the
        # kernel send buffer to ~1-2 (downscaled, ~20 KB) frames, and disable Nagle.
        # When the link can't keep up, write()/drain() blocks almost immediately;
        # while it's blocked the upstream maxsize=1 queue drops every intermediate
        # frame, so the client always resumes on the FRESHEST one — glass-to-glass
        # stays ~1 frame instead of backlogging into seconds of stale video.
        try:
            request.transport.set_write_buffer_limits(high=32768, low=8192)
            sock = request.transport.get_extra_info("socket")
            if sock is not None:
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 64 * 1024)
        except Exception:                            # noqa: BLE001
            pass
        q: asyncio.Queue = asyncio.Queue(maxsize=1)
        with self._lock:
            self._clients.setdefault(name, set()).add(q)
            warm = self._latest.get(name)
        if warm:                                     # show the newest frame at once
            try:
                q.put_nowait(warm)
            except Exception:                        # noqa: BLE001
                pass
        try:
            while True:
                # BOUNDED wait, not a bare q.get(): a stream with no frames (idle /
                # unplugged camera) never writes, and aiohttp only notices a gone
                # client at a write — so a bare get() parked EVERY abandoned
                # connection here FOREVER, each holding its socket fd. Clients
                # (app tiles, browsers) retry dead streams every ~10 s, so the fds
                # accumulated until accept() hit EMFILE and ALL video died
                # ("[Errno 24] Too many open files", 2026-07-08). Waking every few
                # seconds to probe the transport reaps dead connections in ≤5 s.
                try:
                    jpeg = await asyncio.wait_for(q.get(), timeout=5.0)
                except asyncio.TimeoutError:
                    tr = request.transport
                    if tr is None or tr.is_closing():
                        break
                    continue
                await resp.write(
                    b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                    + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
        except (asyncio.CancelledError, ConnectionResetError, ConnectionError):
            pass
        finally:
            with self._lock:
                self._clients.get(name, set()).discard(q)
        return resp

    # -- H.264 over WebSocket (browser WebCodecs) --------------------------
    async def _h264_ws_handler(self, request):
        """Stream raw H.264 Annex-B access units (binary frames) to a browser. The
        browser feeds them to WebCodecs VideoDecoder → <canvas>. Same NVENC encoder
        as the UDP path; one binary WS message == one access unit. A late joiner starts
        decoding at the next in-band IDR (~1 s)."""
        from aiohttp import web
        name = self._resolve(request.path.split("/")[2])   # /camera/<name>/h264
        ws = web.WebSocketResponse(max_msg_size=0)
        await ws.prepare(request)
        # Bounded per-client queue: H.264 is inter-frame, so on overflow (slow client)
        # we CLEAR and let it resync at the next IDR rather than send a torn GOP.
        q: asyncio.Queue = asyncio.Queue(maxsize=120)
        with self._lock:
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
            with self._lock:
                self._h264_ws.get(name, set()).discard(q)
        return ws

    def _ws_wants(self, name: str) -> bool:
        with self._lock:
            return bool(self._h264_ws.get(name))

    def _ws_on_au(self, name: str, au: bytes) -> None:
        """Called from the NVENC reader thread: enqueue one access unit to every WS
        client of `name` (scheduled onto the network loop; latest-GOP-wins on overflow)."""
        loop = self._loop
        if loop is None:
            return
        with self._lock:
            queues = list(self._h264_ws.get(name, ()))
        for q in queues:
            loop.call_soon_threadsafe(self._enqueue_au, q, au)

    @staticmethod
    def _enqueue_au(q: "asyncio.Queue", au: bytes) -> None:
        if q.full():
            while not q.empty():                     # drop the stale GOP; resync at next IDR
                try:
                    q.get_nowait()
                except Exception:                    # noqa: BLE001
                    break
        try:
            q.put_nowait(au)
        except Exception:                            # noqa: BLE001
            pass

    def stop(self) -> None:
        loop = self._loop
        if loop is not None:
            loop.call_soon_threadsafe(loop.stop)
