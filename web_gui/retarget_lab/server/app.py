"""
Retarget Lab backend — FastAPI (:8120) + the Vision Pro gateway (:9090).

Studio conventions (control_lab/server/app.py): create_app() factory,
module-level `app`, /ws/state push stream, pure-ASGI no-cache middleware (NOT
BaseHTTPMiddleware — it breaks streaming disconnects), statics mounted last,
/website-assets for the shared GLB rig.

Command surface (POST /api/command):
    {action: "variants", active: [vid,...]}        which retargeters run
    {action: "catalog"}                            re-read variants.py (new / edited retargeters)
    {action: "mirror", id: vid}                     whose joints the headset sees
    {action: "source", source: live|replay}
    {action: "record", on: bool, name?, notes?}
    {action: "load", id: episode_id}                (switches source to replay)
    {action: "play", on: bool} / {action: "seek", frac} / {action: "speed", value}
    {action: "loop", on: bool} / {action: "reanchor"}
    {action: "evaluate", id: episode_id, variants?: [...]}   offline, every frame
    {action: "delete", id} / {action: "rename", id, name}
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import threading

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import config as CFG
from . import episodes, evaluate
from .lab import Lab
from . import variants as V
from .worker import METRICS

log = logging.getLogger("retarget_lab")


class _NoCache:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        path = scope.get("path", "")
        stamp = path == "/" or path.endswith((".html", ".js", ".css")) or path.startswith("/api/")

        async def _send(msg):
            if stamp and msg["type"] == "http.response.start":
                h = list(msg.get("headers", []))
                h.append((b"cache-control", b"no-store"))
                msg["headers"] = h
            await send(msg)
        return await self.app(scope, receive, _send)


def _clean(o):
    """NaN/inf -> None so the browser's JSON.parse accepts it."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return o


class _NoCacheStatic(StaticFiles):
    """The studio's own files, revalidated on every load (a 304 when unchanged): an updated viewer.js must not
    sit behind the browser's heuristic cache after the lab is restarted."""

    async def get_response(self, path, scope):
        resp = await super().get_response(path, scope)
        resp.headers["Cache-Control"] = "no-cache"
        return resp


_EE_MESHES = None


def _ee_meshes():
    global _EE_MESHES
    if _EE_MESHES is None:
        import base64

        import numpy as np
        from mabel.teleop_engine.end_effector import GRIPPER
        out = {}
        for rel in GRIPPER["meshes"]["housing"] + GRIPPER["meshes"]["hand"] + GRIPPER["meshes"]["finger"]:
            v, f = [], []
            for line in open(CFG.OPENARM_MESHES / rel):
                if line.startswith("v "):
                    v.append(line.split()[1:4])
                elif line.startswith("f "):
                    ids = [int(t.split("/")[0]) - 1 for t in line.split()[1:]]
                    f += [(ids[0], ids[k], ids[k + 1]) for k in range(1, len(ids) - 1)]   # fan: n-gons too
            name = rel.rsplit("/", 1)[-1][:-4]
            out[name] = {"pos": base64.b64encode(np.asarray(v, np.float32).tobytes()).decode(),
                         "idx": base64.b64encode(np.asarray(f, np.uint32).tobytes()).decode()}
        spec = {k: GRIPPER[k] for k in ("housing_geom_pos", "hand_pos", "hand_geom_pos", "fingers", "finger_range", "colors")}
        _EE_MESHES = {"meshes": out, "spec": spec}
    return _EE_MESHES


def create_app(lab=None, start=True):
    lab = lab or Lab()
    app = FastAPI(title="MABEL Retarget Lab")
    app.state.lab = lab
    evals = {"running": None, "progress": None, "last": None, "error": None}

    @app.on_event("startup")
    async def _up():
        if start:
            lab.start()

    @app.on_event("shutdown")
    async def _down():
        lab.shutdown()

    @app.get("/api/ee_meshes")
    def ee_meshes():
        """The OpenArm v1 gripper for the viewer: each OBJ (arm-global mm, as Enactic ships it) as base64
        float32 positions + uint32 triangle indices, plus the placement spec (end_effector.GRIPPER)."""
        return JSONResponse(_ee_meshes())

    @app.get("/api/health")
    async def health():
        g = lab.gateway.status()
        return {"ok": True, "service": "retarget_lab", "gateway": g["listening"],
                "gatewayError": g["error"], "variants": lab.pool.active()}

    @app.get("/api/config")
    async def config():
        return {"variants": V.public(), "active": lab.pool.active(),
                "metrics": {k: {"label": v[0], "unit": v[1], "higher": v[2], "per_side": v[3]}
                            for k, v in METRICS.items()},
                "vpPort": CFG.VP_PORT, "lan": CFG.lan_ip(), "port": CFG.PORT}

    @app.get("/api/episodes")
    async def eps():
        return {"episodes": episodes.list_episodes()}

    @app.get("/api/evaluations")
    async def evs():
        return {"evaluations": evaluate.list_evaluations(), "status": _eval_status()}

    @app.get("/api/evaluations/{name}")
    async def ev(name: str):
        f = (CFG.EVAL_DIR / name).resolve()
        if CFG.EVAL_DIR.resolve() not in f.parents or not f.exists():
            return JSONResponse({"error": "not found"}, 404)
        if f.suffix == ".json":
            return JSONResponse(_clean(json.loads(f.read_text())))
        return FileResponse(f)

    def _eval_status():
        return {k: evals[k] for k in ("running", "progress", "error")} | {
            "last": evals["last"]}

    def _evaluate(eid, vids):
        evals.update(running=eid, progress=[0, len(vids)], error=None)
        try:
            out = evaluate.run(eid, vids, progress=lambda a, b: evals.update(progress=[a, b]),
                               settings=dict(lab.pool.settings))
            evals["last"] = out["file"]
            lab.status_msg = f"evaluation done · {out['frames']} frames × {len(vids)} variants"
        except Exception as e:
            log.exception("evaluation failed")
            evals["error"] = str(e)
        finally:
            evals["running"] = None

    @app.post("/api/command")
    async def command(cmd: dict):
        a = cmd.get("action")
        try:
            if a == "variants":
                return {"ok": True, "active": lab.set_active(list(cmd.get("active") or []))}
            if a == "catalog":
                diff = V.reload()
                lab.pool.restart(diff["changed"])
                lab.set_active([v for v in lab.pool.active() if v in V.VARIANTS])
                return {"ok": True, "variants": V.public(), **diff}
            if a == "mirror":
                lab.mirror = cmd["id"]
            elif a == "source":
                lab.set_source(cmd["source"])
            elif a == "record":
                return {"ok": True, "meta": lab.record(bool(cmd.get("on")), name=cmd.get("name", ""),
                                                      notes=cmd.get("notes", ""))}
            elif a == "load":
                meta = lab.player.load(cmd["id"])
                lab.set_source("replay")
                lab.player.play(True)
                return {"ok": True, "meta": meta}
            elif a == "play":
                lab.player.play(bool(cmd.get("on", True)))
            elif a == "seek":
                lab.player.seek(float(cmd["frac"]))
            elif a == "speed":
                lab.player.speed = float(cmd["value"])
            elif a == "loop":
                lab.player.loop = bool(cmd.get("on"))
            elif a == "reanchor":
                lab.reanchor()
            elif a == "home":
                lab.home()
            elif a == "settings":
                kw = {k: cmd[k] for k in ("scope", "robot", "ee", *CFG.CHANNELS) if cmd.get(k)}
                st = lab.pool.configure(**kw)
                if kw.get("scope"):
                    lab.physics.control("control_mode", {"scope": st["scope"]})
                return {"ok": True, "settings": st}
            elif a == "physics":
                return {"ok": True, "physics": lab.physics.enable(bool(cmd.get("on")))}
            elif a == "evaluate":
                if evals["running"]:
                    return {"ok": False, "error": "an evaluation is already running"}
                vids = list(cmd.get("variants") or lab.pool.active())
                threading.Thread(target=_evaluate, args=(cmd["id"], vids), daemon=True).start()
            elif a == "delete":
                episodes.delete(cmd["id"])
            elif a == "rename":
                return {"ok": True, "meta": episodes.rename(cmd["id"], cmd["name"])}
            else:
                return JSONResponse({"ok": False, "error": f"unknown action {a!r}"}, 400)
        except Exception as e:
            return JSONResponse({"ok": False, "error": str(e)}, 400)
        return {"ok": True}

    @app.websocket("/ws/state")
    async def ws_state(ws: WebSocket):
        await ws.accept()
        try:
            while True:
                s = lab.state()
                s["eval"] = _eval_status()
                await ws.send_text(json.dumps(_clean(s), separators=(",", ":")))
                await asyncio.sleep(1.0 / CFG.STATE_HZ)
        except (WebSocketDisconnect, RuntimeError):
            pass

    app.mount("/website-assets", StaticFiles(directory=str(CFG.WEBSITE_ASSETS)), name="assets")
    app.mount("/", _NoCacheStatic(directory=str(CFG.WEB_DIR), html=True), name="web")
    return app


app = None
