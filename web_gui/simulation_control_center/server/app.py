"""
MABEL SIMULATION Control Center — FastAPI app.

The sim twin of control_center/server/app.py: serves the same studio frontend
+ shared website rig assets, plus the planes the pure-sim stack needs:
  /api/system      host + sim-service status (networking panel)
  /api/recordings  episodes-on-disk summary
  /ws/telemetry    59-DOF pos/vel + targets + estop @15 Hz (from the sim)
  /ws/map          pointcloud + odom from the MuJoCo mirror (no ROS needed)
  /api/sim/*       sim-only controls: reset, safe-relocate reset, scene switch
  /api/scenes      the scene library (registry + thumbnails)
  /api/episodes/*  the record → "save this episode?" confirm flow

Control itself is NOT proxied here: the browser is a first-class client of the
sim bridge (ws :9090/teleop) exactly like the iOS app.
"""
import asyncio
import json
import os
import time

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import MutableHeaders

from . import config as CFG
from . import episodes, joints, recordings, scenes, simmirror, system, telemetry
from .supervisor import SUPERVISOR


def create_app():
    app = FastAPI(title="MABEL Simulation Control Center")

    state = telemetry.TelemetryState()
    app.state.telemetry = state
    app.state.viewers = {}                 # ws id -> {ip, ua, since}
    app.state.source = None
    app.state.maphub = simmirror.MirrorMapHub()
    app.state.mirror = None
    app.state.nav = None

    @app.on_event("startup")
    async def _start_source():
        # 1) the sim stack itself (unless attached / disabled)
        if os.environ.get("MABEL_SCC_AUTOSTART", "1") != "0":
            try:
                await run_in_threadpool(
                    SUPERVISOR.autostart, CFG.DEFAULT_SCENE,
                    os.environ.get("MABEL_SCC_VIEWER", "0") == "1")
            except Exception:
                pass
        # 2) telemetry (ROS → sim bridge → mock)
        app.state.source = await run_in_threadpool(telemetry.make_source, state)
        # 3) the MuJoCo mirror feeds /ws/map + the LiDAR scan on the pure-sim
        #    path (with ROS, the bridge's own :9092 map plane serves instead)
        app.state.maphub.attach_loop(asyncio.get_event_loop())
        if CFG.MAP_PORT == 0 and os.environ.get("MABEL_SCC_MIRROR", "1") != "0":
            from . import streams
            app.state.mirror = simmirror.SimMirror(
                state, app.state.maphub, stream_hub=streams.HUB).start()
            # 4) native nav + SLAM fallback (this Mac can't run ROS Nav2/SLAM)
            if os.environ.get("MABEL_SCC_NAV", "1") != "0":
                try:
                    from . import navrunner
                    app.state.nav = navrunner.NavRunner(
                        state, app.state.mirror, app.state.maphub,
                        app.state.source, simmirror.load_map_stream()).start()
                except Exception as e:
                    import logging
                    logging.getLogger("scc").warning("macnav disabled: %s", e)
        # 5) scene-picker warm-up: registry cache + downscaled thumbnails, so
        #    the first dropdown open doesn't pay a cold N+1 scan + PNG bytes
        import threading
        threading.Thread(target=scenes.warm, daemon=True).start()

    @app.on_event("shutdown")
    async def _stop_source():
        if app.state.source:
            app.state.source.stop()
        if app.state.mirror:
            app.state.mirror.stop()
        if SUPERVISOR.managed:
            await run_in_threadpool(SUPERVISOR.stop)

    # Pure-ASGI no-cache shim — deliberately NOT @app.middleware("http"):
    # BaseHTTPMiddleware re-plumbs the endpoint's receive channel, so a
    # streaming endpoint never sees http.disconnect. That made every departed
    # MJPEG viewer an immortal zombie pump (verified live: 26 leaked upstream
    # connections), starving fresh streams (black FPV) and REST calls. A raw
    # ASGI wrapper only touches response headers and leaves receive intact.
    class _NoCacheHeaders:
        def __init__(self, app):
            self.app = app

        async def __call__(self, scope, receive, send):
            if scope["type"] != "http":
                return await self.app(scope, receive, send)
            path = scope.get("path", "")
            cacheable = (path == "/" or path.endswith((".html", ".js", ".css"))
                         or path.startswith("/api"))

            async def send_nc(message):
                if cacheable and message["type"] == "http.response.start":
                    headers = MutableHeaders(scope=message)
                    if "cache-control" not in headers:
                        headers["Cache-Control"] = "no-store, must-revalidate"
                await send(message)

            await self.app(scope, receive, send_nc)

    app.add_middleware(_NoCacheHeaders)

    # ---- liveness probe -----------------------------------------------------
    # Cheap, state-free 200 for any external health check (load balancer,
    # systemd watchdog, uptime probe).
    @app.get("/healthz")
    async def healthz():
        return Response("ok", media_type="text/plain")

    # ---- config handshake ---------------------------------------------------
    @app.get("/api/config")
    async def get_config():
        scene_path = scenes.current_scene_path(SUPERVISOR.scene)
        return {
            "robotHost": CFG.ROBOT_HOST,           # "" -> use page hostname
            "teleopPort": CFG.TELEOP_PORT,
            "mapPort": CFG.MAP_PORT,               # 0 -> same-origin /ws/map
            "videoPort": CFG.VIDEO_PORT,
            "tokenGated": bool(CFG.APP_TOKEN),
            "tempWarn": CFG.TEMP_WARN,
            "tempHot": CFG.TEMP_HOT,
            "telemetrySource": state.source,
            "sim": True,
            "ros": state.source == "ros",          # SLAM/Nav2 planes exist?
            "manageSim": SUPERVISOR.managed,
            "scene": scenes.name_for_path(scene_path),
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
        if os.path.isfile(CFG.CAMERA_RIG_JSON):
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
            # SAME probe set as the real control center — the sim camera
            # server serves the identical paths (front = body_front_cam)
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
        """Same-origin MJPEG pass-through of the local camera server. WebGL
        may only texture same-origin images, so the 3D camera projections
        stream through here instead of :8080 directly."""
        import urllib.request

        from starlette.responses import StreamingResponse

        url = (f"http://{CFG.UPSTREAM_HOST}:{CFG.VIDEO_PORT}"
               f"/camera/{name}/stream.mjpg")
        # ?fetch=1 → serve the SAME bytes as an opaque octet stream: WebKit's
        # fetch() cannot read a multipart/x-mixed-replace body (its legacy
        # multipart handler kills the reader with "Load failed"), and the
        # fetch-based MJPEG painter frames by JPEG SOI/EOI markers anyway.
        fetch_mode = "fetch" in request.query_params
        try:
            src = await run_in_threadpool(
                lambda: urllib.request.urlopen(url, timeout=3))
        except Exception:
            # OFFLINE DEMO: no live camera server → loop the recorded episode's
            # frames as MJPEG so the tiles / FPV / 3D projections still show video.
            ep = getattr(state, "demo_episode", None)
            if ep is not None:
                async def demo_pump():
                    i = 0
                    while not await request.is_disconnected():
                        jpg = await run_in_threadpool(ep.cam_jpeg, name, i)
                        if jpg:
                            yield (b"--frame\r\nContent-Type: image/jpeg\r\n"
                                   b"Content-Length: " + str(len(jpg)).encode()
                                   + b"\r\n\r\n" + jpg + b"\r\n")
                        i += 1
                        await asyncio.sleep(1.0 / 12.0)
                return StreamingResponse(
                    demo_pump(),
                    media_type="application/octet-stream" if fetch_mode
                    else "multipart/x-mixed-replace; boundary=frame")
            return JSONResponse({"error": "camera offline"}, status_code=502)
        ctype = ("application/octet-stream" if fetch_mode
                 else src.headers.get("Content-Type",
                                      "multipart/x-mixed-replace; boundary=frame"))

        async def pump():
            # A viewer that goes away does NOT cancel this generator (verified:
            # uvicorn keeps streaming into the closed transport), so every
            # departed tile/FPV/projection left a zombie pump holding an
            # upstream :8080 connection and forcing the camera server to keep
            # rendering for phantom viewers — which is what starved fresh
            # streams and REST calls in long sessions. Poll the ASGI
            # disconnect state each chunk and reap ourselves.
            try:
                while not await request.is_disconnected():
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

    @app.get("/api/cam/{name}/frame.jpg")
    async def cam_frame(name: str):
        """Single head/wrist-cam JPEG proxied from the local MJPEG server —
        same-origin, so the POV-transition recorder can composite it."""
        from . import video_proxy
        try:
            jpg = await run_in_threadpool(video_proxy.grab_frame, name)
        except Exception:
            jpg = None
        if not jpg:                                  # offline demo fallback
            ep = getattr(state, "demo_episode", None)
            if ep is not None:
                jpg = await run_in_threadpool(ep.cam_jpeg, name, int(time.time() * 12))
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

    # ── e-stop: forwarded to the SIM BRIDGE over its own wire protocol ─────
    @app.post("/api/hw/estop")
    async def hw_estop(request: Request):
        from . import hardware
        body = await request.json()
        if body.get("engage"):
            return await run_in_threadpool(hardware.engage, app.state.source)
        return await run_in_threadpool(hardware.release, state, app.state.source)

    @app.get("/api/hw/status")
    async def hw_status():
        from . import hardware
        return await run_in_threadpool(hardware.status, state, app.state.source)

    @app.post("/api/slam/pause")
    async def slam_pause():
        if state.source != "ros":
            return {"ok": False,
                    "error": "no SLAM in pure-sim mode (Mac) — run the Ubuntu "
                             "ROS sim stack for mapping"}
        from . import streams
        try:
            results = await run_in_threadpool(streams.pause_slam)
        except Exception as e:
            return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
        return {"ok": any(results.values()), "services": results}

    @app.post("/api/slam/reset")
    async def slam_reset():
        """Reset MAPPING ONLY — ROS sim stack only (no SLAM in pure sim)."""
        if state.source != "ros":
            return {"ok": False,
                    "error": "no SLAM in pure-sim mode (Mac) — run the Ubuntu "
                             "ROS sim stack for mapping"}
        from . import streams
        try:
            results = await run_in_threadpool(streams.reset_mapping)
        except Exception as e:
            return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
        return {"ok": any(results.values()), "services": results}

    # ---- the sim map plane (MuJoCo mirror: pointcloud + odom) ---------------
    @app.websocket("/ws/map")
    async def ws_map(ws: WebSocket):
        await ws.accept()
        try:
            await app.state.maphub.subscribe(ws)
        except (WebSocketDisconnect, RuntimeError):
            pass
        except Exception:
            pass

    # ---- same-origin bridge proxies (https / single-origin serving) ----------
    # When the page is served over https from one origin, the browser cannot
    # dial ws://<host>:9090 directly. These proxies pipe a page websocket
    # straight to the local bridge so the studio (3D viewer, teleop, network
    # panel) still works. LAN/local clients keep connecting to :9090 directly
    # (no extra hop).
    async def _demo_teleop(ws: WebSocket, ep):
        """OFFLINE RECORDINGS: no live bridge → serve the RECORDED episode as the
        robot_state plane so the sim studio shows Connected + a posed robot (not a
        frozen 'Disconnected' viewer). This is a REPLAY of recorded joints — a
        recording, not a physics engine. One light JSON frame @15 Hz; safe with
        swap in place."""
        await ws.send_text(json.dumps({"type": "hello", "payload": {
            "server": "mabel_sim_demo", "kind": "demo",
            "name": "MABEL Sim (recorded demo)",
            "cameras": {"head": {"fovy": 83.3, "aspect": 1.3333}}}}))
        i = 0
        try:
            while True:
                names, pos, vel, eff, scan = ep.state_frame(i)
                jp = {n: pos[k] for k, n in enumerate(names)}
                await ws.send_text(json.dumps({"type": "robot_state", "payload": {
                    "jointPositions": jp,
                    "base": {"x": 0.0, "y": 0.0, "z": 0.077, "yaw": 0.0},
                    "server": "MABEL Sim (recorded demo)", "driver": {},
                    "stateAgeMs": 0}}))
                i += 1
                await asyncio.sleep(1.0 / 15.0)
        except Exception:
            pass

    async def _bridge_proxy(ws: WebSocket, path: str):
        import websockets as wslib
        await ws.accept()
        # forward the page's query string (client=, sid= …) so the bridge's
        # device identification and the network panel's "Current window"
        # session tag survive the hop
        query = ws.url.query or ""
        url = f"ws://127.0.0.1:{CFG.TELEOP_PORT}/{path}" + (f"?{query}" if query else "")
        # Offline (no live bridge): serve the RECORDING replay on /ws/teleop so the
        # sim studio shows a posed robot instead of "Disconnected". If a real bridge
        # is up, proxy it below (live sim wins over the recording).
        ep = getattr(state, "demo_episode", None)
        if ep is not None and path == "teleop":
            try:
                probe = await wslib.connect(url, max_size=None, open_timeout=1.0)
                await probe.close()
            except Exception:
                return await _demo_teleop(ws, ep)
        if ep is not None and path == "monitor":
            try:
                probe = await wslib.connect(url, max_size=None, open_timeout=1.0)
                await probe.close()
            except Exception:
                try:                                       # keep the monitor plane open (empty roster)
                    while True:
                        m = await ws.receive()
                        if m.get("type") == "websocket.disconnect":
                            break
                except Exception:
                    pass
                return
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

    # ---- browser MuJoCo viewer (full-scene orbit render) ---------------------
    @app.websocket("/ws/simview")
    async def ws_simview(ws: WebSocket, instance: str = ""):
        """The native MuJoCo viewer, in the browser: JPEG frames out, camera
        orbit params in ({az, el, dist, lookat, follow, w, h, fps}).
        ?instance=preview points it at the physics-preview world instead of
        the live robot."""
        await ws.accept()
        state_file = None
        if instance == "preview":
            from .preview import _preview_state_file
            state_file = _preview_state_file()
        q = asyncio.Queue(maxsize=2)
        session = simmirror.SimViewSession(asyncio.get_event_loop(), q,
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
                item = await q.get()
                if isinstance(item, str):        # scene metadata / cam-sync JSON
                    await ws.send_text(item)
                else:                            # JPEG frame
                    await ws.send_bytes(item)
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

    # ---- sim-only controls ---------------------------------------------------
    def _wire(msg_type, payload):
        """Send one envelope to the bridge via the SimSource (or one-shot)."""
        from . import hardware
        return hardware._send_wire(app.state.source, msg_type, payload)

    @app.get("/api/sim/status")
    async def sim_status():
        st = await run_in_threadpool(SUPERVISOR.status)
        scene_path = scenes.current_scene_path(SUPERVISOR.scene)
        st["scene"] = scene_path
        st["sceneName"] = scenes.name_for_path(scene_path)
        st["mirror"] = {
            "ok": bool(app.state.mirror and app.state.mirror.ok),
            "scene": getattr(app.state.mirror, "scene", None),
            "error": getattr(app.state.mirror, "error", None),
        }
        src = app.state.source
        st["bridgeConnected"] = bool(getattr(src, "connected", False))
        hello = getattr(src, "hello", None)
        if hello:
            st["robotName"] = hello.get("name")
        return st

    @app.get("/api/scenes")
    async def get_scenes():
        reg = await run_in_threadpool(scenes.registry)
        cur = scenes.name_for_path(scenes.current_scene_path(SUPERVISOR.scene))
        return {"scenes": [{**e, "xml": os.path.basename(e["xml"]),
                            "thumb": bool(e.get("thumb"))} for e in reg],
                "current": cur,
                "canSwitch": SUPERVISOR.can_switch(),
                "adopts": SUPERVISOR.can_switch() and not SUPERVISOR.managed}

    @app.get("/api/scenes/thumb/{name}")
    async def scene_thumb(name: str):
        e = await run_in_threadpool(scenes.by_name, name)
        if e and e.get("thumb"):
            p = await run_in_threadpool(scenes.small_thumb, e)
            mt = "image/jpeg" if p.endswith(".jpg") else "image/png"
            return FileResponse(p, media_type=mt,
                                headers={"Cache-Control": "max-age=3600"})
        return JSONResponse({"error": "no thumbnail"}, status_code=404)

    @app.post("/api/sim/scene")
    async def set_scene(request: Request):
        """Switch the WORLD: restart the managed sim stack with a new scene.
        Every client (this page included) auto-reconnects in a few seconds."""
        body = await request.json()
        e = await run_in_threadpool(scenes.by_name, str(body.get("name", "")))
        if not e:
            return JSONResponse({"ok": False, "error": "unknown scene"},
                                status_code=404)
        try:
            st = await run_in_threadpool(SUPERVISOR.restart, e["xml"])
        except Exception as ex:
            return JSONResponse({"ok": False, "error": str(ex)}, status_code=409)
        return {"ok": True, "scene": e["name"], "status": st}

    @app.post("/api/sim/reset")
    async def sim_reset(request: Request):
        """Reset the sim world. {"relocate": true} re-spawns the robot at a
        computed open, collision-free pose instead of the origin."""
        body = {}
        try:
            body = await request.json()
        except Exception:
            pass
        payload = {}
        pose = None
        if body.get("relocate"):
            scene_path = scenes.current_scene_path(SUPERVISOR.scene)
            avoid = None
            last = getattr(app.state.source, "last_state", None) or {}
            base = last.get("base") or {}
            if "x" in base and "y" in base:
                avoid = (float(base["x"]), float(base["y"]))
            pose = await run_in_threadpool(
                scenes.safe_spawn, scene_path, avoid)
            if pose is None:
                return JSONResponse(
                    {"ok": False, "error": "no open spawn point found"},
                    status_code=409)
            payload["pose"] = pose
        ok = await run_in_threadpool(_wire, "reset", payload)
        if not ok:
            return JSONResponse({"ok": False, "error": "bridge unreachable"},
                                status_code=502)
        # odometry can't integrate through a teleport — re-anchor the native
        # nav/SLAM at the placed pose
        if app.state.nav is not None:
            async def _reanchor():
                await asyncio.sleep(1.0)          # let the world settle
                await run_in_threadpool(app.state.nav.reinit)
            asyncio.create_task(_reanchor())
        return {"ok": True, "pose": pose}

    @app.post("/api/sim/viewer")
    async def sim_viewer(request: Request):
        """Toggle the desktop MuJoCo viewer (restarts the managed stack)."""
        body = await request.json()
        try:
            st = await run_in_threadpool(
                SUPERVISOR.restart, None, bool(body.get("on")))
        except Exception as ex:
            return JSONResponse({"ok": False, "error": str(ex)}, status_code=409)
        return {"ok": True, "status": st}

    # ---- PHYSICS-TRUE choreography preview (isolated bridge instance) --------
    @app.post("/api/preview/start")
    async def preview_start(request: Request):
        """Spawn (or reuse) the ephemeral preview world — its own
        simulation_bridge instance, seeded from the live robot's pose. Never
        touches the live run (engine-level instance locks)."""
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

    # ---- native Mac navigation (macnav: FastSLAM + A* + MPPI) ----------------
    @app.post("/api/nav/goal")
    async def nav_goal(request: Request):
        if app.state.nav is None:
            return JSONResponse({"ok": False, "error": "native nav not running "
                                 "(ROS mode or mirror disabled)"}, status_code=503)
        body = await request.json()
        st = await run_in_threadpool(app.state.nav.set_goal,
                                     float(body["x"]), float(body["y"]))
        return {"ok": True, **st}

    @app.post("/api/nav/cancel")
    async def nav_cancel():
        if app.state.nav is None:
            return {"ok": False, "error": "native nav not running"}
        return {"ok": True, **await run_in_threadpool(app.state.nav.cancel)}

    @app.get("/api/nav/status")
    async def nav_status():
        if app.state.nav is None:
            return {"backend": "none", "state": "unavailable"}
        return await run_in_threadpool(app.state.nav.status)

    # ---- the long-horizon action registry (timeline composer) ---------------
    @app.get("/api/actions")
    async def actions():
        """Action registry for the timeline: everything 'available' executes
        TODAY (client-side clips/poses, nav primitives via /api/nav); learned
        policies are declared honestly as coming_soon until a checkpoint is
        actually deployable."""
        nav_ok = app.state.nav is not None
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
                      "status": "available" if (nav_ok or nid != "nav_goto")
                      else "coming_soon",
                      "params": ({"x": 0.0, "y": 0.0} if nid == "nav_goto"
                                 else {"duration": 2.0})})
        for sid, title in [("pose_home", "home position"), ("pose_stand", "stand tall"),
                           ("pose_bend", "bend forward"), ("pose_arms_up", "arms up"),
                           ("pose_tpose", "T-pose"), ("pose_look", "look around")]:
            A.append({"id": sid, "title": title, "category": "pose",
                      "lane": "manip", "exec": "client", "status": "available"})
        return {"actions": A, "navBackend": "macnav" if nav_ok else "ros"}

    # ---- episode save-confirm flow -------------------------------------------
    @app.get("/api/episodes/latest")
    async def episodes_latest():
        path = await run_in_threadpool(episodes.latest_episode)
        if not path:
            return JSONResponse({"error": "no episodes"}, status_code=404)
        inf = await run_in_threadpool(episodes.info, path)
        env = scenes.name_for_path(scenes.current_scene_path(SUPERVISOR.scene))
        inf["suggestedName"] = await run_in_threadpool(
            episodes.suggest_name, env)
        inf["env"] = env
        return inf

    @app.post("/api/episodes/commit")
    async def episodes_commit(request: Request):
        body = await request.json()
        env = scenes.name_for_path(scenes.current_scene_path(SUPERVISOR.scene))
        return await run_in_threadpool(
            episodes.commit, str(body.get("path", "")),
            body.get("name"), body.get("task"), env)

    @app.post("/api/episodes/discard")
    async def episodes_discard(request: Request):
        body = await request.json()
        return await run_in_threadpool(episodes.discard,
                                       str(body.get("path", "")))

    # ---- static ---------------------------------------------------------------
    if os.path.isdir(CFG.WEBSITE_ASSETS):
        app.mount("/website-assets", StaticFiles(directory=CFG.WEBSITE_ASSETS),
                  name="website-assets")
    app.mount("/", StaticFiles(directory=CFG.WEB_DIR, html=True), name="web")
    return app


app = create_app()
