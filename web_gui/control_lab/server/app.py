"""
Control Lab backend — FastAPI app (:8115).

Follows the MABEL studio conventions (create_app() factory + module-level
`app`; schema-first /ws/state stream; pure-ASGI no-cache middleware — NOT
BaseHTTPMiddleware, which breaks streaming disconnects; statics mounted last;
/website-assets serving the shared GLB rig). Physics runs in-process
(simworld.SimWorld); the Vision Pro node (vpnode.VPNode) binds :9090 on demand.

This backend is a MINI-GATEWAY around the SAME control seam the iOS / Vision Pro
apps drive (mabel.MabelTeleop): the command surface mirrors the app wire protocol
(control_mode / joint_command / external_force / estop), so the input here is
identical to the input there.

Command surface (POST /api/command):
    {action: "torque", on: bool}                         (E-STOP edge: ON holds current pose, OFF limps)
    {action: "control_mode", method: navigation|joint|wrist|retarget,
             controlType: impedance|compliance, region: arm|upper_body|whole_body,
             stiffness: float}                            (== the app's control_mode message)
    {action: "joint_command", joints: {wire_id: value}}  (== joint_command message)
    {action: "external_force", forces: {body: [fx,fy,fz]}} (== external_force message)
    {action: "reset"} / {action: "recalibrate"}
    {action: "vp",     source: off|headset|mock}
    {action: "calibration", user?}
    {action: "hand_side", side: left|right|both}
    {action: "wrist_target", side: l|r, pos: [x,y,z]}    (wrist method)
    {action: "wrist_sync"} / {action: "wrist_human", on: bool}
    {action: "self_collision", on: bool} / {action: "sc_method", method: stop|cbf}
    {action: "push", body, force: [fx,fy,fz], seconds?}  (diagnostics: ground-truth wrench)
    {action: "traj", kind: hold|step|sine, joint?, amp?, period?}  (diagnostics: joint driver)
Joystick input arrives on the /ws/state uplink: {type:"input", nav:{lx,ly,rx,ry}};
wrist drag: {type:"wrist", side, pos}; joint slider: {type:"joint", id, value}.
"""
from __future__ import annotations

import asyncio
import json

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from . import config as CFG
from . import joints
from .simworld import SimWorld, METHODS, CONTROL_TYPES, REGIONS
from .vpnode import VPNode


class _NoCacheHeaders:
    """Pure-ASGI no-store headers on the SPA + API (the BaseHTTPMiddleware
    trap is documented in simulation_control_center/server/app.py)."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        path = scope.get("path", "")
        stamp = (path == "/" or path.endswith((".html", ".js", ".css"))
                 or path.startswith("/api"))

        async def send2(msg):
            if stamp and msg["type"] == "http.response.start":
                headers = msg.setdefault("headers", [])
                headers.append((b"cache-control", b"no-store"))
            await send(msg)

        await self.app(scope, receive, send2)


def create_app() -> FastAPI:
    app = FastAPI(title="MABEL Control Lab")
    app.add_middleware(_NoCacheHeaders)
    app.state.world = SimWorld()
    app.state.vp = VPNode(app.state.world)

    @app.on_event("startup")
    async def _startup():
        app.state.world.start()

    @app.on_event("shutdown")
    async def _shutdown():
        app.state.vp.stop()
        app.state.world.stop()

    # ── REST ────────────────────────────────────────────────────────────────
    @app.get("/healthz")
    async def healthz():
        return PlainTextResponse("ok")

    @app.get("/api/config")
    async def api_config():
        return {"methods": list(METHODS), "controlTypes": list(CONTROL_TYPES),
                "regions": list(REGIONS), "port": CFG.PORT,
                "vpPort": CFG.VP_PORT, "lanIp": CFG.lan_ip(),
                "plotHz": CFG.PLOT_HZ,
                "plotWindowS": CFG.PLOT_WINDOW_S,
                "scene": str(CFG.SCENE)}

    @app.get("/api/joints")
    async def api_joints():
        return joints.schema(app.state.world.model)

    @app.get("/api/plots")
    async def api_plots():
        return app.state.world.plot_history()

    @app.post("/api/command")
    async def api_command(cmd: dict):
        w, vp = app.state.world, app.state.vp
        action = cmd.get("action")
        try:
            if action == "torque":
                w.set_torque(cmd.get("on", True))
            elif action == "control_mode":
                # the app wire message: method | controlType | region | stiffness
                w.set_control_spec(
                    method=cmd.get("method"),
                    control_type=cmd.get("controlType", cmd.get("control_type")),
                    region=cmd.get("region"),
                    stiffness=cmd.get("stiffness"))
                # Toggling into Teleop with NO input source selected AUTO-LISTENS
                # to the vision_pro_stream.py ZMQ publisher (operator directive):
                # a SUB connect is passive/localhost, so this never conflicts with
                # the :9090 Device path — pick Device/Mock to switch explicitly.
                if cmd.get("method") == "retarget" and vp.mode == "off":
                    vp.set_source("avp")
            elif action == "joint_command":
                w.set_joint_targets(cmd.get("joints") or {})
            elif action == "external_force":
                w.apply_external_force(cmd.get("forces") or {})
            elif action == "recalibrate":
                w.recalibrate()
            elif action == "self_collision":
                w.set_self_collision(cmd.get("on", True))
            elif action == "sc_method":
                w.set_sc_method(cmd["method"])
            elif action == "wrist_human":
                w.set_wrist_human(cmd.get("on", True))
            elif action == "wrist_target":
                w.set_wrist_target(cmd["side"], cmd["pos"])
            elif action == "wrist_sync":
                w.wrist_sync()
            elif action == "hand_side":
                w.set_hand_side(cmd["side"])
            elif action == "traj":            # diagnostics: joint driver
                w.set_trajectory(cmd.get("kind", "hold"), cmd.get("joint"),
                                 cmd.get("amp"), cmd.get("period"))
            elif action == "push":            # diagnostics: ground-truth wrench
                w.apply_push(cmd["body"], cmd.get("force", [10, 0, 0]),
                             cmd.get("seconds", 1.0))
            elif action == "vp":
                vp.set_source(cmd.get("source", "off"))
            elif action == "calibration":
                return JSONResponse(vp.load_calibration(
                    cmd.get("user", CFG.DEFAULT_USER)))
            elif action == "reset":
                w.reset_robot()
            else:
                return JSONResponse({"ok": False,
                                     "error": f"unknown action {action!r}"},
                                    status_code=400)
        except (KeyError, ValueError) as e:
            return JSONResponse({"ok": False, "error": str(e)},
                                status_code=400)
        return {"ok": True, "torque": w.torque_on, "spec": dict(w.spec),
                "selfCollision": w.self_collision, "vp": vp.status()}

    @app.get("/api/bodies")
    async def api_bodies():
        """Pushable bodies for the wrench-injection dropdown (arm chain)."""
        import mujoco
        w = app.state.world
        keys = ("arm", "palm", "forearm", "elbow", "tower", "wrist")
        names = []
        for i in range(w.model.nbody):
            nm = mujoco.mj_id2name(w.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if nm and any(k in nm.lower() for k in keys):
                names.append(nm)
        return {"bodies": sorted(names)}

    # ── WS state stream (schema first, then state @ STATE_HZ) ───────────────
    @app.websocket("/ws/state")
    async def ws_state(ws: WebSocket):
        await ws.accept()
        await ws.send_text(json.dumps(
            {"type": "schema", "payload": joints.schema(app.state.world.model)}))
        # replay plot history so charts fill instantly on (re)connect
        await ws.send_text(json.dumps(
            {"type": "plots_history", "payload": app.state.world.plot_history()}))
        stop = asyncio.Event()

        async def uplink():
            try:
                while True:
                    raw = await ws.receive_text()
                    try:
                        msg = json.loads(raw)
                    except ValueError:
                        continue
                    if msg.get("type") == "input":
                        nav = msg.get("nav") or {}
                        app.state.world.nav.update(
                            {k: float(nav.get(k, 0.0))
                             for k in ("lx", "ly", "rx", "ry")})
                    elif msg.get("type") == "wrist":
                        try:
                            app.state.world.set_wrist_target(
                                msg.get("side"), msg.get("pos"))
                        except (KeyError, ValueError, TypeError):
                            pass
                    elif msg.get("type") == "joint":
                        # joint slider — a single-joint override streamed live,
                        # exactly like the app's joint_command (latest-wins dict)
                        try:
                            app.state.world.set_joint_targets(
                                {msg["id"]: float(msg["value"])})
                        except (KeyError, ValueError, TypeError):
                            pass
                    elif msg.get("type") == "ping":
                        await ws.send_text(json.dumps(
                            {"type": "pong", "payload": msg.get("payload")}))
            except WebSocketDisconnect:
                stop.set()
            except Exception:
                stop.set()

        async def downlink():
            period = 1.0 / CFG.STATE_HZ
            try:
                while not stop.is_set():
                    snap = await asyncio.get_event_loop().run_in_executor(
                        None, app.state.world.snapshot)
                    snap["vpNode"] = app.state.vp.status()
                    await ws.send_text(json.dumps(
                        {"type": "state", "payload": snap}))
                    await asyncio.sleep(period)
            except Exception:
                stop.set()

        up = asyncio.create_task(uplink())
        down = asyncio.create_task(downlink())
        await stop.wait()
        for t in (up, down):
            t.cancel()

    # ── statics LAST ────────────────────────────────────────────────────────
    if CFG.WEBSITE_ASSETS.exists():
        app.mount("/website-assets",
                  StaticFiles(directory=str(CFG.WEBSITE_ASSETS)),
                  name="website-assets")
    app.mount("/", StaticFiles(directory=str(CFG.WEB_DIR), html=True),
              name="web")
    return app


app = create_app()
