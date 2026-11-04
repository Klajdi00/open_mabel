"""
The Vision Pro gateway — the Mac pretends to be a MABEL robot on the LAN.

Binds ws://0.0.0.0:9090/teleop (the apps hard-code :9090) and advertises
`_mabel-teleop._tcp` over Bonjour, so the unmodified MABEL Vision Pro app finds
this Mac on the local Wi-Fi and connects by itself. It speaks just enough of
the gateway wire protocol (server/wire_protocol.py) to keep the app happy:

  down: hello (server="retarget_lab", kind=sim, caps=[stream_always, record],
        hands={left, right}: "v1" = MABEL's ORCA hand, "gripper" = the OpenArm v1 gripper — the
        studio's End effector setting; sent again whenever it changes, so the headset swaps its rig),
        robot_state @ ROBOT_STATE_HZ (the mirror variant's retargeted joints —
        the headset shows the robot doing what the lab computes), pong
  up:   teleop_frame (head + both hand skeletons, 60 Hz) -> Lab.on_frame
        record {active, task}  -> start/stop an episode from the headset
        recalibrate            -> re-anchor every retargeter
        control_mode {scope}   -> whole body ("global") / upper body ("local"), all variants
        everything else (control_mode, estop, ...) is acknowledged by silence:
        nothing here moves a robot.

The app keeps the link alive as long as ANY message arrives within 3.5 s; the
robot_state stream covers that. `caps: ["stream_always"]` is read by the VP
app (≥ the 2026-10-15 change) to open its tracking space and stream hands
without a robot to engage — older apps still stream once whole-body is engaged.

LAN only (no relay, no auth): this is a data-collection bench, not a robot.
"""
from __future__ import annotations

import asyncio
import atexit
import json
import logging
import shutil
import socket
import subprocess
import sys
import threading
import time

from . import config as CFG

log = logging.getLogger("retarget_lab.gateway")


def device_of(ua):
    u = (ua or "").lower()
    if "visionos" in u or "vision" in u or "mabelteleop" in u or "client=visionpro" in u:
        return "Vision Pro"
    if "iphone" in u or "ios" in u:
        return "iPhone"
    if "mozilla" in u:
        return "Web client"
    return "Device"


class Gateway:
    def __init__(self, lab):
        self.lab = lab
        self.error = None
        self.listening = False
        self.clients = {}            # ws -> meta
        self.active = None
        self._loop = None
        self._bonjour = None
        self._thread = threading.Thread(target=self._run, daemon=True, name="rtl-gateway")

    def start(self):
        self._thread.start()
        if CFG.BONJOUR:
            self._advertise()

    def _advertise(self):
        name = f"{CFG.BONJOUR_NAME} ({socket.gethostname().split('.')[0]})"
        txt = ["robot=MABEL", "kind=sim", "role=retarget_lab"]
        if sys.platform == "darwin" and shutil.which("dns-sd"):
            cmd = ["dns-sd", "-R", name, "_mabel-teleop._tcp", "local", str(CFG.VP_PORT), *txt]
        elif shutil.which("avahi-publish-service"):
            cmd = ["avahi-publish-service", name, "_mabel-teleop._tcp", str(CFG.VP_PORT), *txt]
        else:
            log.warning("no dns-sd / avahi: headset must be pointed at %s:%d by hand",
                        CFG.lan_ip(), CFG.VP_PORT)
            return
        try:
            self._bonjour = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                             stderr=subprocess.DEVNULL)
            atexit.register(self._bonjour.terminate)
            self.bonjour_name = name
        except Exception as e:
            log.warning("Bonjour failed: %s", e)

    def stop(self):
        if self._bonjour is not None:
            self._bonjour.terminate()

    def status(self):
        now = time.time()
        roster = [{"device": m["device"], "addr": m["addr"], "frames": m["frames"],
                   "hz": round(m["hz"], 1), "ageS": round(now - m["last"], 2) if m["last"] else None,
                   "hands": m["hands"], "head": m["head"], "active": ws is self.active}
                  for ws, m in list(self.clients.items())]
        return {"listening": self.listening, "port": CFG.VP_PORT, "error": self.error,
                "lan": CFG.lan_ip(), "bonjour": getattr(self, "bonjour_name", None),
                "clients": roster}

    # ── server ────────────────────────────────────────────────────────────
    def _run(self):
        try:
            asyncio.run(self._main())
        except Exception as e:                       # bind failure etc.
            self.error = self.error or str(e)
            self.listening = False
            log.error("gateway stopped: %s", e)

    async def _main(self):
        import websockets
        self._loop = asyncio.get_running_loop()
        server = None
        for attempt in range(8):                     # ride out a previous lab's slow release
            try:
                server = await websockets.serve(self._handler, CFG.HOST, CFG.VP_PORT,
                                                max_size=2 ** 22, ping_interval=None,
                                                compression=None)
                break
            except OSError as e:
                if attempt == 7:
                    self.error = (f":{CFG.VP_PORT} is busy — another gateway (sim/real "
                                  f"start_teleop.sh, Control Lab) holds it; stop it and "
                                  f"restart the lab ({e.strerror})")
                    raise
                await asyncio.sleep(0.5)
        self.listening = True
        self.error = None
        # TCP_NODELAY: a 30 Hz state stream must not be Nagle-batched
        for s in server.sockets:
            try:
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except OSError:
                pass
        await asyncio.gather(self._state_pump(), asyncio.Future())

    def _ee(self):
        return "gripper" if self.lab.pool.settings.get("ee") == "gripper" else "orca"

    def _hello(self, meta):
        hand = "gripper" if self._ee() == "gripper" else "v1"
        return {"type": "hello", "payload": {
            "server": "retarget_lab", "kind": "sim", "robot": "MABEL",
            "name": "Retarget Lab", "id": "retarget-lab",
            "you": meta["device"], "caps": ["stream_always", "record"],
            "hands": {"left": hand, "right": hand},
            "authRequired": False, "armRequired": False, "transport": "lan"}}

    async def _handler(self, ws, path=None):
        ua = ""
        try:
            ua = ws.request_headers.get("User-Agent", "") or ""
        except Exception:
            pass
        rpath = path or getattr(ws, "path", "") or ""
        peer = ws.remote_address[0] if ws.remote_address else "?"
        meta = {"device": device_of(ua + " " + rpath), "addr": peer, "frames": 0,
                "last": 0.0, "hz": 0.0, "hands": [], "head": False}
        self.clients[ws] = meta
        log.info("client connected: %s from %s", meta["device"], peer)
        try:
            await ws.send(json.dumps(self._hello(meta)))
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except (TypeError, ValueError):
                    continue
                await self._dispatch(ws, meta, msg.get("type"), msg.get("payload") or {})
        except Exception:
            pass
        finally:
            self.clients.pop(ws, None)
            if self.active is ws:
                self.active = None
            log.info("client left: %s", meta["device"])

    def _on_frame(self, ws, meta, p):
        now = time.time()
        if meta["last"]:
            dt = now - meta["last"]
            if dt > 0:
                meta["hz"] = 0.95 * meta["hz"] + 0.05 / dt
        meta["last"] = now
        meta["frames"] += 1
        h = p.get("head") or {}
        meta["head"] = h.get("trackingState", "notAvailable") == "normal"
        meta["hands"] = [s for s, k in (("left", "leftHand"), ("right", "rightHand"))
                         if (p.get(k) or {}).get("isTracked")]
        body = meta["head"] or bool(meta["hands"])
        # arbitration: the first device streaming a TRACKED body drives; a stale
        # (>1.5 s) driver is replaced.
        act = self.active
        if act is None or act not in self.clients or (
                ws is not act and now - self.clients[act]["last"] > 1.5):
            if body:
                self.active = ws
        if ws is self.active:
            self.lab.on_frame(p, now, meta["device"])


    async def _dispatch(self, ws, meta, t, p):
        """One headset message (separate so it can be tested without a socket)."""
        if t == "ping":
            await ws.send(json.dumps({"type": "pong", "payload": p}))
        elif t == "teleop_frame":
            self._on_frame(ws, meta, p)
        elif t == "record":
            if "active" in p:
                self.lab.record(bool(p["active"]), name=p.get("task") or "",
                                device=meta["device"])
        elif t == "recalibrate":
            self.lab.reanchor()
        elif t == "control_mode" and p.get("scope") in ("local", "global"):
            # the headset's Whole body / Upper body switch (RobotController.retargetScope)
            self.lab.pool.configure(scope=p["scope"])
            self.lab.physics.control("control_mode", {"scope": p["scope"]})
            log.info("scope %s from %s", p["scope"], meta["device"])

    async def _state_pump(self):
        period = 1.0 / max(CFG.ROBOT_STATE_HZ, 1.0)
        ee = self._ee()
        while True:
            await asyncio.sleep(period)
            if not self.clients:
                continue
            if self._ee() != ee:                     # End effector toggled: re-announce the hands
                ee = self._ee()
                for ws, meta in list(self.clients.items()):
                    try:
                        await ws.send(json.dumps(self._hello(meta)))
                    except Exception:
                        pass
                log.info("end effector %s: hello re-sent to %d client(s)", ee, len(self.clients))
            msg = json.dumps({"type": "robot_state", "payload": self.lab.robot_state()})
            for ws in list(self.clients):
                try:
                    await ws.send(msg)
                except Exception:
                    pass
