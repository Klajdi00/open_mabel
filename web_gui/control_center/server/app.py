"""
MABEL Control Center — FastAPI app.

Serves the studio frontend + the shared website rig assets, and adds the three
planes the existing robot stack lacks:
  /api/system      host + service status (networking panel)
  /api/recordings  data-on-robot-host summary (tasks / episodes / rosbags)
  /ws/telemetry    59-DOF pos/vel/effort + targets + temps + estop @15 Hz

Control itself is NOT proxied here: the browser is a first-class client of the
bridge (ws :9090/teleop) exactly like the iOS app, so remote users can drive
while this backend keeps them informed about the server they are driving.
"""
import asyncio
import json
import os
import time

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from . import auth
from . import config as CFG
from . import joints, recordings, system, telemetry


def _cookie_from_scope(scope, name):
    for k, v in scope.get("headers", []):
        if k == b"cookie":
            for part in v.decode("latin-1").split(";"):
                nv = part.strip().split("=", 1)
                if len(nv) == 2 and nv[0] == name:
                    return nv[1]
    return None


def _is_public_path(path):
    """Paths served WITHOUT a session so the login screen can load + a browser
    can authenticate. Everything else (/api/*, /ws/*) needs the session."""
    if path in ("/api/login", "/api/logout", "/api/whoami", "/api/health"):
        return True
    # static shell (HTML/CSS/JS/rig assets) is public; all data planes are gated
    return not (path.startswith("/api") or path.startswith("/ws"))


class _AuthGate:
    """ASGI gate covering BOTH http and websocket scopes with one session cookie
    (a plain http middleware can't protect WebSocket handshakes)."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if (scope["type"] not in ("http", "websocket")
                or not CFG.CC_REQUIRE_LOGIN
                or _is_public_path(scope.get("path", ""))):
            return await self.app(scope, receive, send)
        token = _cookie_from_scope(scope, auth.COOKIE)
        if auth.verify_session(token):
            return await self.app(scope, receive, send)
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008})
            return
        body = b'{"error":"login required","code":"unauthenticated"}'
        await send({"type": "http.response.start", "status": 401, "headers": [
            (b"content-type", b"application/json"),
            (b"cache-control", b"no-store")]})
        await send({"type": "http.response.body", "body": body})


class _NoCache:
    """Pure-ASGI no-store header stamp.

    MUST NOT be a @app.middleware("http") (Starlette BaseHTTPMiddleware): that
    wrapper swallows `http.disconnect` for STREAMING endpoints, so the MJPEG
    pump below never learns its viewer left and keeps reading the camera server
    forever — one leaked upstream per departed viewer, which starves the real
    streams (and is a known fd-leak source on the robot host). Wrapping `send` only keeps
    the disconnect message flowing to the endpoint.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        path = scope.get("path", "")
        # .glb/.stl/.json included: the RIG keeps its URL when its CONTENT changes, so
        # without this a browser that once loaded an old robot draws it forever. Routes
        # that set their own Cache-Control (the rig, which prefers a revalidating
        # no-cache so a 304 is possible) are left alone by the check below.
        cacheable = (path == "/" or path.startswith("/api")
                     or path.endswith((".html", ".js", ".css", ".glb", ".stl", ".json")))

        async def send_wrapper(message):
            if cacheable and message["type"] == "http.response.start":
                headers = message.setdefault("headers", [])
                if not any(k.lower() == b"cache-control" for k, _ in headers):
                    headers.append((b"cache-control", b"no-store, must-revalidate"))
            await send(message)

        await self.app(scope, receive, send_wrapper)


def create_app():
    app = FastAPI(title="MABEL Control Center")

    state = telemetry.TelemetryState()
    app.state.telemetry = state
    app.state.viewers = {}                 # ws id -> {ip, ua, since}
    app.state.source = None

    @app.on_event("startup")
    async def _start_source():
        app.state.source = await run_in_threadpool(telemetry.make_source, state)

    @app.on_event("shutdown")
    async def _stop_source():
        if app.state.source:
            app.state.source.stop()

    app.add_middleware(_NoCache)
    app.add_middleware(_AuthGate)

    # ---- app-level login (styled popup, replaces browser basic-auth) --------
    def _client_ip(request):
        fwd = request.headers.get("x-forwarded-for", "")
        return (fwd.split(",")[0].strip() if fwd
                else (request.client.host if request.client else "?"))

    @app.get("/api/whoami")
    async def whoami(request: Request):
        token = request.cookies.get(auth.COOKIE)
        user = auth.verify_session(token)
        return {"authed": bool(user) or not CFG.CC_REQUIRE_LOGIN,
                "user": user or "", "required": CFG.CC_REQUIRE_LOGIN,
                "server": "MABEL Real-Robot Control Center"}

    @app.post("/api/login")
    async def login(request: Request):
        now = time.time()
        ip = _client_ip(request)
        if auth.throttled(ip, now):
            return JSONResponse({"error": "too many attempts — wait a few minutes"},
                                status_code=429)
        try:
            body = await request.json()
        except Exception:
            body = {}
        if not auth.check_credentials(body.get("username"), body.get("passkey")):
            auth.note_fail(ip, now)
            return JSONResponse({"error": "invalid username or passkey"},
                                status_code=401)
        auth.note_ok(ip)
        token = auth.make_session(CFG.CC_USER, now)
        resp = JSONResponse({"ok": True, "user": CFG.CC_USER})
        secure = (request.headers.get("x-forwarded-proto", request.url.scheme)
                  == "https")
        resp.set_cookie(auth.COOKIE, token, max_age=auth.TTL_SECONDS,
                        httponly=True, samesite="lax", secure=secure, path="/")
        return resp

    @app.post("/api/logout")
    async def logout():
        resp = JSONResponse({"ok": True})
        resp.delete_cookie(auth.COOKIE, path="/")
        return resp

    # ---- config handshake ---------------------------------------------------
    @app.get("/api/config")
    async def get_config():
        return {
            "robotHost": CFG.ROBOT_HOST,           # "" -> use page hostname
            "teleopPort": CFG.TELEOP_PORT,
            "mapPort": CFG.MAP_PORT,
            "videoPort": CFG.VIDEO_PORT,
            "tokenGated": bool(CFG.APP_TOKEN),
            "tempWarn": CFG.TEMP_WARN,
            "tempHot": CFG.TEMP_HOT,
            "telemetrySource": state.source,
            # the shared frontend (sim twin fork) gates sim-only UI on these:
            # this is the REAL robot — no scene switching, no managed sim.
            "sim": False,
            "ros": state.source == "ros",      # SLAM/Nav2 planes exist?
            "scene": None,
            "manageSim": False,
            "robot": "MABEL",
            # what is on each hand mount now: ORCA "v1" (MABEL's own) or "v2" (NOVA's,
            # swapped in) — the rig, the HANDS chip and the hand module follow it
            "hands": {s: h["generation"] for s, h in CFG.hand_generations().items()},
        }

    @app.get("/api/health")
    async def health():
        return {"ok": True, "source": state.source}

    @app.get("/api/joints")
    async def get_joints():
        return joints.schema()

    @app.get("/api/system")
    async def get_system():
        return await run_in_threadpool(system.snapshot)

    @app.get("/api/recordings")
    async def get_recordings():
        return await run_in_threadpool(recordings.summary)

    @app.get("/api/camera-rig")
    async def get_camera_rig():
        if CFG.CAMERA_RIG_JSON and os.path.isfile(CFG.CAMERA_RIG_JSON):
            return FileResponse(CFG.CAMERA_RIG_JSON, media_type="application/json")
        return JSONResponse({"cameras": []})

    @app.get("/api/delay")
    async def delay(s: float = 5.0):
        """E2E helper: a 1×1 gif that answers after `s` seconds. Referenced by
        index.html?e2e=<s> so a headless browser's load event (and therefore
        --screenshot) waits until the app has fully booted (GLB, links)."""
        await asyncio.sleep(min(s, 300.0))
        gif = (b"GIF89a\x01\x00\x01\x00\x80\x00\x00\x00\x00\x00\xff\xff\xff!"
               b"\xf9\x04\x01\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01"
               b"\x00\x00\x02\x02D\x01\x00;")
        return Response(gif, media_type="image/gif")

    _cam_live_cache = {"t": 0.0, "v": {}}

    @app.get("/api/cameras/live")
    async def cameras_live():
        """TRUE per-stream liveness by probing the local MJPEG server for an
        actual JPEG (the 'main' slot silently falls back to the front cam, so
        stream presence ≠ camera connected — head_left is the honest head
        probe). Cached 10 s."""
        from . import video_proxy
        now = time.time()
        if now - _cam_live_cache["t"] > 10.0:
            names = ["main", "head_left", "front", "wrist_left", "wrist_right"]

            async def probe(n):
                try:
                    jpg = await run_in_threadpool(video_proxy.grab_frame, n, 1.5)
                    return n, bool(jpg)
                except Exception:
                    return n, False
            results = await asyncio.gather(*[probe(n) for n in names])
            _cam_live_cache["v"] = dict(results)
            _cam_live_cache["t"] = now
        return _cam_live_cache["v"]

    @app.get("/api/camera-info")
    async def camera_info():
        """Live per-camera intrinsics + distortion tapped off camera_info —
        overlays the baked sim rig so real lenses project correctly."""
        return state.caminfo

    @app.get("/api/cam/{name}/stream.mjpg")
    async def cam_stream(name: str, request: Request):
        """Same-origin MJPEG pass-through of the local camera server — now only
        the FALLBACK path (the browser prefers /ws/cam/<name>/h264 below). WebGL
        may only texture same-origin images, so it also backs the 3D camera
        projections when H.264 is unavailable."""
        import urllib.request

        from starlette.responses import StreamingResponse

        url = f"http://127.0.0.1:{CFG.VIDEO_PORT}/camera/{name}/stream.mjpg"
        try:
            src = await run_in_threadpool(
                lambda: urllib.request.urlopen(url, timeout=3))
        except Exception:
            return JSONResponse({"error": "camera offline"}, status_code=502)
        ctype = src.headers.get("Content-Type",
                                "multipart/x-mixed-replace; boundary=frame")

        async def pump():
            try:
                while True:
                    # Without this the pump outlives its viewer and reads the
                    # camera server forever (see _NoCache). Checked per chunk so
                    # a departed tab is reaped within one frame.
                    if await request.is_disconnected():
                        break
                    chunk = await run_in_threadpool(src.read, 32768)
                    if not chunk:
                        break
                    yield chunk
            finally:
                try:
                    src.close()
                except Exception:
                    pass

        return StreamingResponse(pump(), media_type=ctype)

    @app.websocket("/ws/cam/{name}/h264")
    async def ws_cam_h264(ws: WebSocket, name: str):
        """Same-origin proxy of the camera server's H.264 access units.

        THE fast video path when the studio is served over https from a single
        origin (e.g. `tailscale serve` or your own reverse proxy), where the
        browser cannot dial ws://<host>:8080 directly. It rides the page's own
        origin and session cookie, exactly like the MJPEG pass-through above, but ships inter-frame H.264 (~3-4 Mbps @720p) instead of a whole
        JPEG per frame (~30-60 Mbps). On LAN the browser talks to :8080 directly
        and never touches this route.

        Binary-transparent: one upstream message == one Annex-B access unit ==
        one downstream message, so the browser's WebCodecs decoder is unchanged.

        DROP-DON'T-QUEUE. The proxy must never let a slow viewer turn into latency.
        If we awaited send_bytes() straight from the upstream loop, a browser on a
        slow link would stall our read, the camera server's own 120-deep
        per-client queue would fill (~4 s of video at 30 fps), and every frame
        after that would arrive seconds late — smooth but hopelessly behind. So we
        read upstream at full speed into a SHALLOW queue and, on overflow, throw
        away the backlog and resync at the next IDR. Latency stays bounded at the
        cost of a skip. iOS/website clients talk to the camera server directly and
        never traverse this hop, which is why only the studio saw this delay.
        """
        await ws.accept()
        try:
            import websockets                      # uvicorn[standard] dependency
        except ImportError:
            # No client lib -> no H.264 proxy; the browser falls back to MJPEG.
            await ws.close(code=1011)
            return

        url = f"ws://127.0.0.1:{CFG.VIDEO_PORT}/camera/{name}/h264"
        # A few frames of slack absorbs jitter; more than that IS latency.
        q: asyncio.Queue = asyncio.Queue(maxsize=CFG.CAM_WS_QUEUE)
        try:
            async with websockets.connect(url, max_size=None, open_timeout=4,
                                          ping_interval=None,
                                          compression=None) as up:  # H.264 is already compressed

                async def pull():
                    async for au in up:
                        if isinstance(au, str):
                            continue
                        if q.full():
                            while not q.empty():   # shed the stale GOP, not the newest frame
                                try:
                                    q.get_nowait()
                                except Exception:
                                    break
                        q.put_nowait(au)

                async def push():
                    while True:
                        await ws.send_bytes(await q.get())

                async def watch_client():
                    # A browser that closes the tab never sends us anything, so
                    # this sits here purely to raise WebSocketDisconnect when it
                    # goes. Without it the pump could keep an upstream H.264
                    # subscriber alive for a viewer that is already gone.
                    while True:
                        await ws.receive()

                tasks = [asyncio.create_task(t())
                         for t in (pull, push, watch_client)]
                try:
                    await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                finally:
                    for t in tasks:
                        t.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
        except WebSocketDisconnect:
            pass                                   # viewer left: unwind, close upstream
        except Exception:
            # camera server down / no /h264 route (older build) -> MJPEG fallback
            pass
        try:
            await ws.close()
        except Exception:
            pass

    @app.get("/api/cam/{name}/frame.jpg")
    async def cam_frame(name: str):
        """Single head/wrist-cam JPEG proxied from the local MJPEG server —
        same-origin, so the POV-transition recorder can composite it."""
        from . import video_proxy
        try:
            jpg = await run_in_threadpool(video_proxy.grab_frame, name)
        except Exception:
            jpg = None
        if not jpg:
            return JSONResponse({"error": "no frame"}, status_code=404)
        return Response(jpg, media_type="image/jpeg",
                        headers={"Cache-Control": "no-store"})

    # ── e2e recording sink (dev-gated: MABEL_CC_DEV=1) ────────────────────
    @app.post("/api/dev/frame")
    async def dev_frame(request: Request, name: str = "seq", i: int = 0):
        if os.environ.get("MABEL_CC_DEV") != "1":
            return JSONResponse({"error": "dev sink disabled"}, status_code=403)
        root = os.environ.get("MABEL_CC_E2E_DIR", "/tmp/mabel_cc_e2e")
        d = os.path.join(root, name)
        os.makedirs(d, exist_ok=True)
        body = await request.body()
        with open(os.path.join(d, f"frame_{i:04d}.jpg"), "wb") as f:
            f.write(body)
        return {"ok": True, "i": i, "bytes": len(body)}

    @app.get("/api/viewers")
    async def get_viewers():
        now = time.time()
        return [{**v, "ageS": round(now - v["since"], 1)}
                for v in app.state.viewers.values()]

    # ---- telemetry fan-out ---------------------------------------------------
    @app.websocket("/ws/telemetry")
    async def ws_telemetry(ws: WebSocket):
        await ws.accept()
        vid = id(ws)
        app.state.viewers[vid] = {
            "ip": (ws.client.host if ws.client else "?"),
            "ua": ws.headers.get("user-agent", ""), "since": time.time()}
        try:
            await ws.send_text(json.dumps(
                {"type": "schema", "payload": joints.schema()}))

            async def receiver():
                # only pings come uplink; everything else is ignored
                while True:
                    msg = json.loads(await ws.receive_text())
                    if msg.get("type") == "ping":
                        await ws.send_text(json.dumps(
                            {"type": "pong",
                             "payload": {"t": msg.get("payload", {}).get("t"),
                                         "srv": time.time()}}))

            async def sender():
                period = 1.0 / max(1.0, CFG.TELEMETRY_HZ)
                n = 0
                while True:
                    snap = state.snapshot()
                    if n % 15 == 0:          # ~1 Hz sidecar: host + viewers
                        snap["system"] = await run_in_threadpool(system.snapshot)
                        now = time.time()
                        snap["viewers"] = [
                            {**v, "ageS": round(now - v["since"], 1)}
                            for v in app.state.viewers.values()]
                    await ws.send_text(json.dumps(
                        {"type": "telemetry", "payload": snap}))
                    n += 1
                    await asyncio.sleep(period)

            await asyncio.gather(receiver(), sender())
        except (WebSocketDisconnect, RuntimeError):
            pass
        except Exception:
            pass
        finally:
            app.state.viewers.pop(vid, None)

    # ---- selectable sensor streams (binary, map-protocol records) ----------
    @app.websocket("/ws/stream")
    async def ws_stream(ws: WebSocket):
        """Client sends {"cloud":["head","front"],"grid":"lidar"|"visual"};
        server pushes 0x04/0x05 records for the chosen sources as they update."""
        from . import streams
        await ws.accept()
        sel = {"cloud": [], "grid": None}
        rev = 0
        try:
            while True:
                # apply any pending selection change (non-blocking-ish poll)
                try:
                    msg = json.loads(await asyncio.wait_for(ws.receive_text(), 0.15))
                    sel["cloud"] = list(msg.get("cloud", sel["cloud"]))
                    sel["grid"] = msg.get("grid", sel["grid"])
                    for c in sel["cloud"]:
                        streams.HUB.ensure_cloud(c)
                    if sel["grid"]:
                        streams.HUB.ensure_grid(sel["grid"])
                except asyncio.TimeoutError:
                    pass
                keys = ["cloud:" + c for c in sel["cloud"]]
                if sel["grid"]:
                    keys.append("grid:" + sel["grid"])
                rev, fresh = streams.HUB.latest(keys, rev)
                for _k, blob in fresh:
                    await ws.send_bytes(blob)
        except (WebSocketDisconnect, RuntimeError):
            pass
        except Exception:
            pass

    # ── hard e-stop: kill / relaunch the hardware bring-up ────────────────
    @app.post("/api/hw/estop")
    async def hw_estop(request: Request):
        from . import hardware
        body = await request.json()
        if body.get("engage"):
            return await run_in_threadpool(hardware.engage)
        return await run_in_threadpool(hardware.release, state)

    @app.get("/api/hw/status")
    async def hw_status():
        from . import hardware
        return await run_in_threadpool(hardware.status, state)

    @app.post("/api/slam/pause")
    async def slam_pause():
        from . import streams
        try:
            results = await run_in_threadpool(streams.pause_slam)
        except Exception as e:
            return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
        return {"ok": any(results.values()), "services": results}

    @app.post("/api/slam/reset")
    async def slam_reset():
        """Reset MAPPING ONLY (nvblox 3D mesh + the derived 2D /map slice,
        and slam_toolbox when in lidar mode) — localization (cuVSLAM) is
        deliberately untouched so the robot keeps its pose."""
        from . import streams
        try:
            results = await run_in_threadpool(streams.reset_mapping)
        except Exception as e:
            return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
        return {"ok": any(results.values()), "services": results}

    # ---- same-origin bridge proxies (https / single-origin serving) ----------
    # When the page is served over https from one origin, the browser cannot
    # dial ws://<host>:9090 directly. These proxies pipe a page websocket
    # straight to the local bridge so the studio (3D viewer, teleop, network
    # panel) still works. LAN/local clients keep connecting to :9090 directly
    # (no extra hop).
    async def _bridge_proxy(ws: WebSocket, path: str):
        import websockets as wslib
        await ws.accept()
        # forward the page's query string (client=, sid= …) so the bridge's
        # device identification and the network panel's "Current window"
        # session tag survive the hop
        query = ws.url.query or ""
        url = f"ws://127.0.0.1:{CFG.TELEOP_PORT}/{path}" + (f"?{query}" if query else "")
        try:
            async with wslib.connect(url, max_size=None, ping_interval=20) as up:

                async def client_to_bridge():
                    while True:
                        msg = await ws.receive()
                        if msg.get("type") == "websocket.disconnect":
                            raise WebSocketDisconnect()
                        if msg.get("text") is not None:
                            await up.send(msg["text"])
                        elif msg.get("bytes") is not None:
                            await up.send(msg["bytes"])

                async def bridge_to_client():
                    async for m in up:
                        if isinstance(m, bytes):
                            await ws.send_bytes(m)
                        else:
                            await ws.send_text(m)

                done, pending = await asyncio.wait(
                    [asyncio.create_task(client_to_bridge()),
                     asyncio.create_task(bridge_to_client())],
                    return_when=asyncio.FIRST_EXCEPTION)
                for t in pending:
                    t.cancel()
        except (WebSocketDisconnect, RuntimeError, OSError):
            pass
        except Exception:
            pass

    @app.websocket("/ws/teleop")
    async def ws_teleop_proxy(ws: WebSocket):
        await _bridge_proxy(ws, "teleop")

    @app.websocket("/ws/monitor")
    async def ws_monitor_proxy(ws: WebSocket):
        await _bridge_proxy(ws, "monitor")

    @app.websocket("/ws/map")
    async def ws_map(ws: WebSocket):
        """Placeholder for a deployment with no map plane (MABEL_MAP_PORT=0).

        config.js falls back to a same-origin /ws/map whenever mapPort is 0. With
        no route here the handshake fell through to the StaticFiles mount at "/",
        which asserts scope["type"] == "http" — every attempt logged an ASGI
        traceback and the page reconnected forever. Accept and close cleanly
        instead: same outcome for the page, no traceback. The real robot keeps
        the bridge's :9092 map plane and never dials this."""
        await ws.accept()
        await ws.close(code=1000)

    # ---- browser MuJoCo viewer (the mirrored REAL robot, orbit render) -------
    @app.websocket("/ws/simview")
    async def ws_simview(ws: WebSocket, instance: str = ""):
        """The native MuJoCo viewer, in the browser: JPEG frames out, camera
        orbit params in ({az, el, dist, lookat, follow, w, h, fps}). On the
        real robot the gateway mirrors the encoders into the same qpos memmap,
        so this renders the MIRRORED REAL ROBOT. ?instance=preview points it
        at the physics-preview world instead. Degrades by refusing the socket
        (log) when mujoco/cv2 are absent on the host."""
        from . import simview
        ok, err = simview.available()
        if not ok:
            import logging
            logging.getLogger("cc").warning("simview refused: %s", err)
            await ws.close(code=1011)
            return
        await ws.accept()
        state_file = None
        if instance == "preview":
            from .preview import _preview_state_file
            state_file = _preview_state_file()
        q = asyncio.Queue(maxsize=2)
        session = simview.SimViewSession(asyncio.get_event_loop(), q,
                                         state_file=state_file)
        session.start()

        async def _uplink():
            while True:
                msg = await ws.receive_text()
                try:
                    session.update(json.loads(msg))
                except Exception:
                    pass

        async def _send():
            while True:
                await ws.send_bytes(await q.get())
        try:
            done, pending = await asyncio.wait(
                [asyncio.create_task(_uplink()), asyncio.create_task(_send())],
                return_when=asyncio.FIRST_EXCEPTION)
            for t in pending:
                t.cancel()
        except (WebSocketDisconnect, RuntimeError):
            pass
        except Exception:
            pass
        finally:
            session.stop()

    # ---- PHYSICS-TRUE choreography preview (isolated bridge instance) --------
    @app.post("/api/preview/start")
    async def preview_start(request: Request):
        """Spawn (or reuse) the ephemeral preview world — its own
        simulation_bridge instance, seeded from the live robot's encoder
        mirror. Never touches the live run OR the hardware (engine-level
        instance locks; the preview process has no hardware path). Answers
        {ok: false, error} when mujoco/simulation_bridge are absent here."""
        from .preview import PREVIEW
        body = {}
        try:
            body = await request.json()
        except Exception:
            pass
        try:
            st = await run_in_threadpool(
                PREVIEW.start, body.get("scene"), body.get("seed", True))
        except Exception as e:
            return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
        return {"ok": True, **st}

    @app.post("/api/preview/stop")
    async def preview_stop():
        from .preview import PREVIEW
        return await run_in_threadpool(PREVIEW.stop)

    @app.get("/api/preview/status")
    async def preview_status():
        from .preview import PREVIEW
        return PREVIEW.status()

    @app.websocket("/ws/preview")
    async def ws_preview(ws: WebSocket):
        """The preview command wire: accepts the SAME JSON envelopes the live
        executor sends (teleop_frame / joint_command / reset / estop) and maps
        them onto the preview engine; streams {type:'preview_state'} back at
        ~8 Hz so the client can track progress."""
        from .preview import PREVIEW
        await ws.accept()

        async def _uplink():
            while True:
                msg = json.loads(await ws.receive_text())
                ok = await run_in_threadpool(
                    PREVIEW.wire, msg.get("type", ""), msg.get("payload"))
                if not ok:
                    await ws.send_text(json.dumps(
                        {"type": "error",
                         "payload": {"msg": "preview not running"}}))

        async def _state():
            while True:
                snap = await run_in_threadpool(PREVIEW.snapshot)
                if snap is not None:
                    await ws.send_text(json.dumps(
                        {"type": "preview_state", "payload": snap}))
                await asyncio.sleep(0.12)
        try:
            done, pending = await asyncio.wait(
                [asyncio.create_task(_uplink()), asyncio.create_task(_state())],
                return_when=asyncio.FIRST_EXCEPTION)
            for t in pending:
                t.cancel()
        except (WebSocketDisconnect, RuntimeError):
            pass
        except Exception:
            pass

    # ---- the long-horizon action registry (timeline composer) ---------------
    @app.get("/api/actions")
    async def actions():
        """Action registry for the timeline: everything 'available' executes
        TODAY (client-side clips/poses, nav primitives over the bridge wire);
        learned policies are declared honestly as coming_soon until a
        checkpoint is actually deployable. On the REAL robot 'move to goal'
        rides the wire nav_goal (→ ROS Nav2), so it needs the ROS stack up."""
        ros = state.source == "ros"
        A = []
        for pid, title, loc in [
                ("policy_laundry", "do laundry", "laundry room"),
                ("policy_fold", "fold clothes", "bedroom"),
                ("policy_coffee", "make coffee", "kitchen"),
                ("policy_dishes", "do the dishes", "kitchen"),
                ("policy_breakfast", "cook breakfast", "kitchen"),
                ("policy_bed", "make the bed", "bedroom"),
                ("policy_boxing", "boxing (RL)", None),
                ("policy_dance_rl", "unitree-style dance (RL)", None),
                ("policy_charge", "go to charge", "dock"),
                ("policy_water", "water the plants", "living room"),
                ("policy_tidy", "tidy the table", "living room")]:
            A.append({"id": pid, "title": title, "category": "policy",
                      "lane": "manip", "status": "coming_soon",
                      "location": ({"label": loc} if loc else None),
                      "note": "no deployable checkpoint yet"})
        for nid, title in [("nav_fwd", "move forward"), ("nav_back", "move back"),
                           ("nav_left", "strafe left"), ("nav_right", "strafe right"),
                           ("nav_turn_l", "turn left"), ("nav_turn_r", "turn right"),
                           ("nav_goto", "move to goal (x, y)"),
                           ("nav_hold", "stop & hold")]:
            A.append({"id": nid, "title": title, "category": "nav",
                      "lane": "nav", "exec": "client",
                      "status": "available" if (ros or nid != "nav_goto")
                      else "coming_soon",
                      "params": ({"x": 0.0, "y": 0.0} if nid == "nav_goto"
                                 else {"duration": 2.0})})
        for sid, title in [("pose_home", "home position"), ("pose_stand", "stand tall"),
                           ("pose_bend", "bend forward"), ("pose_arms_up", "arms up"),
                           ("pose_tpose", "T-pose"), ("pose_look", "look around")]:
            A.append({"id": sid, "title": title, "category": "pose",
                      "lane": "manip", "exec": "client", "status": "available"})
        return {"actions": A, "navBackend": "ros"}

    # ---- the rig with the hands that are MOUNTED (MABEL v1 / NOVA v2 per side) -----
    # Registered BEFORE the /website-assets mount so they win. Same URLs; the file (and so the
    # ETag the page keys its fetch by) changes with the hands.
    _REVAL = {"Cache-Control": "no-cache, must-revalidate"}

    @app.api_route("/website-assets/mabel_rig.glb", methods=["GET", "HEAD"])
    async def rig_glb():
        return FileResponse(CFG.rig_files()[0], media_type="model/gltf-binary", headers=_REVAL)

    @app.get("/website-assets/mabel_joints.json")
    async def rig_joints():
        return FileResponse(CFG.rig_files()[1], media_type="application/json", headers=_REVAL)

    @app.get("/api/hands")
    async def get_hands():
        h = CFG.hand_generations()
        return {"hands": h, "native": "v1", "rig": os.path.basename(CFG.rig_files(h)[0])}

    # ---- static ---------------------------------------------------------------
    if os.path.isdir(CFG.WEBSITE_ASSETS):
        app.mount("/website-assets", StaticFiles(directory=CFG.WEBSITE_ASSETS),
                  name="website-assets")

    class _RevalidatingStatic(StaticFiles):
        """The page's own HTML, CSS and JS must REVALIDATE, never blind-cache.

        Starlette sends an ETag and Last-Modified but no Cache-Control, which lets a
        browser apply heuristic freshness and keep running a module it already has.
        For a control surface that is a hazard, not a performance detail: the
        operator drives the robot with whatever code their browser kept (a fixed
        drive-axis module was observed still running stale in an open tab). no-cache
        still allows a 304, so revalidating costs a round trip, not a download."""
        def file_response(self, *args, **kw):
            r = super().file_response(*args, **kw)
            r.headers["Cache-Control"] = "no-cache, must-revalidate"
            return r

    app.mount("/", _RevalidatingStatic(directory=CFG.WEB_DIR, html=True), name="web")
    return app


app = create_app()
