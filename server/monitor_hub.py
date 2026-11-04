"""monitor_hub.py — the bridge's command-monitor tap.

A single `MonitorHub` lives inside the teleop bridge's asyncio loop and watches
every inbound message from every client. It does two jobs:

  1. Derive a clean *live status* per client from the raw wire stream — teleop
     mode, control law (impedance / compliance = "gravity-comp"), body region,
     head tracking state, per-hand tracking on/off, last reset/recalibrate, and
     a measured command rate (Hz).
  2. Fan that out to any number of *monitor subscribers* — the desktop GUI
     (`monitor_gui.py`) or a browser — over the SAME WebSocket server the bridge
     already runs (path `/monitor` on :9090), so no extra port or process.

Wire contract to subscribers (newline-free JSON objects, one per ws message):

    {"kind":"snapshot", "clients":[...], "global":{...}, "log":[ev, ...]}
    {"kind":"event",    "event": ev}                       # one protocol event
    {"kind":"state",    "clients":[...], "global":{...}}   # ~5 Hz heartbeat

where an event `ev` is
    {"seq":int, "t":float, "clientId":int, "peer":str,
     "type":str, "level":"info|strong|warn|reset|rec", "text":str}

The hub keeps the noisy continuous streams (60 Hz teleop frames, joint dribble)
OUT of the event log: those drive the per-client gauges instead, and only a
*meaningful change* (mode flip, a hand gained/lost, a stick crossing the
deadzone) is logged. Discrete protocol messages (reset, control_mode, record,
hello, error) are always logged. The result is a readable scientific trace of
operator intent rather than a firehose.

All mutation happens on the bridge's event-loop thread (the ws handler and one
heartbeat task), so no locking is needed; the only cross-thread reads are the
plain dicts the bridge hands in at `tick()`.
"""
from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from typing import Any

# How long a client's discrete deflection counts as "active" for kind-guessing.
_NAV_DEADZONE = 0.04
_RATE_WINDOW_S = 1.0          # window over which we count msgs to report Hz
_LOG_MAXLEN = 400            # events kept for the snapshot backfill
_JOINT_LOG_MIN_GAP = 1.0     # throttle joint_command log lines to ~1 Hz/client


def _clean(method: str | None) -> str:
    return (method or "").strip().lower()


class _ClientStat:
    """Everything the monitor knows about one connected client, derived purely
    from the messages it has sent. Serialised to a flat dict for the wire."""

    __slots__ = (
        "id", "peer", "device", "app", "platform", "dev_src", "net", "kind",
        "browser", "os", "session",
        "connected_t", "last_seen",
        "mode", "method", "ctype", "region", "stiffness",
        "head_state", "hand_l", "hand_r", "nav",
        "last_reset_t", "last_recalib_t", "joint_n", "last_joint_log",
        "recv_times", "frame_n", "msg_n", "seq", "_nav_was", "_last_fold",
    )

    def __init__(self, cid: int, peer: str, device: dict | None = None,
                 net: dict | None = None):
        now = time.time()
        self.id = cid
        self.peer = peer
        device = device or {}
        self.device = device.get("device")          # exact label, or None if unknown
        self.app = device.get("app")
        self.platform = device.get("platform")
        self.dev_src = device.get("src", "none")     # tag | ua | none
        self.browser = device.get("browser")         # web only: Chrome / Safari / …
        self.os = device.get("os")                   # web only: macOS / Windows / Android…
        self.session = device.get("session") or ""   # per-tab id (distinguishes openings)
        self.net = net or {}                         # ip / transport / secure / host…
        self.kind = "client"             # refined to Headset / Handheld on first frame
        self.connected_t = now
        self.last_seen = now
        # control intent
        self.mode = "—"
        self.method = "—"
        self.ctype = "—"
        self.region = "—"
        self.stiffness = 0.0
        # tracking
        self.head_state = "—"
        self.hand_l = False
        self.hand_r = False
        self.nav = {"lx": 0.0, "ly": 0.0, "rx": 0.0, "ry": 0.0}
        # events
        self.last_reset_t = 0.0
        self.last_recalib_t = 0.0
        self.joint_n = 0
        self.last_joint_log = 0.0
        # throughput
        self.recv_times: deque[float] = deque(maxlen=256)
        self.frame_n = 0
        self.msg_n = 0
        self.seq = 0
        self._nav_was = False
        self._last_fold = 0.0

    def rate_hz(self, now: float) -> float:
        cutoff = now - _RATE_WINDOW_S
        n = sum(1 for t in self.recv_times if t >= cutoff)
        return n / _RATE_WINDOW_S

    def resolved_device(self) -> str:
        """Exact device label if the client identified itself, else the
        hands/rate heuristic guess (suffixed '?' to flag it's inferred)."""
        if self.device:
            return self.device
        return {"Headset": "Headset?", "Handheld": "Handheld?"}.get(self.kind, "Unknown")

    def to_dict(self, now: float, driver_id: int | None) -> dict[str, Any]:
        return {
            "id": self.id,
            "peer": self.peer,
            "device": self.resolved_device(),
            "platform": self.platform or "",
            "app": self.app or "",
            "browser": self.browser or "",
            "os": self.os or "",
            "session": self.session,
            "devSrc": self.dev_src,
            "net": self.net,
            "kind": self.kind,
            "driver": self.id == driver_id,
            "ageS": round(now - self.connected_t, 1),
            "idleS": round(now - self.last_seen, 2),
            "rateHz": round(self.rate_hz(now), 1),
            "mode": self.mode,
            "method": self.method,
            "ctype": self.ctype,
            "region": self.region,
            "stiffness": round(self.stiffness, 1),
            "headState": self.head_state,
            "handL": self.hand_l,
            "handR": self.hand_r,
            "nav": self.nav,
            "seq": self.seq,
            "frames": self.frame_n,
            "msgs": self.msg_n,
            "sinceResetS": round(now - self.last_reset_t, 1) if self.last_reset_t else None,
        }


class MonitorHub:
    """Singleton command-stream tap shared by the bridge and its subscribers."""

    def __init__(self) -> None:
        self.loop: asyncio.AbstractEventLoop | None = None
        self._subs: set = set()                    # subscriber websockets
        self._rename_cb = None                     # bridge cb(new_name) for renames
        self._clients: dict[int, _ClientStat] = {}
        self._log: deque[dict] = deque(maxlen=_LOG_MAXLEN)
        self._seq = 0
        self._actuation: dict | None = None       # latest standardized RobotCommand
        # FUSED live status — the headline state, updated per-field by the most
        # recent client to set it (last-writer-wins), INDEPENDENT of the
        # arbitration "active driver". This is what the status strip shows, so a
        # Hold→Soft toggle, a head-tracking drop, an E-stop, or a folder pick is
        # reflected the instant its message arrives — even from a client that is
        # not (yet) the driver. The per-client panel still shows each client's
        # own state; only the headline is fused.
        self._fused = {
            "mode": "—", "method": "—", "ctype": "—", "region": "—",
            "stiffness": 0.0, "intentPeer": None,
            "headState": "—", "handL": False, "handR": False,
            "nav": {"lx": 0.0, "ly": 0.0, "rx": 0.0, "ry": 0.0}, "headPeer": None,
            "estop": False, "estopPeer": None,
            "recTask": None,
        }
        self._global = {
            "online": 0, "driverId": None, "driverPeer": None,
            "recording": {"isRecording": False, "task": None},
            "status": self._fused,
            "actuation": None,
            "serverInfo": {},
            "server": "mabel_sim_bridge", "version": "",
        }

    # ── lifecycle ───────────────────────────────────────────────────────────
    def attach_loop(self, loop: asyncio.AbstractEventLoop, version: str = "") -> None:
        self.loop = loop
        if version:
            self._global["version"] = version

    def set_actuation(self, cmd: dict) -> None:
        """Record the latest STANDARDIZED robot command (the resolved actuation
        dispatched to sim/real). Called from the bridge's control loop (main
        thread) — a plain reference set, broadcast by the next heartbeat on the
        loop thread, so no cross-thread publish is needed."""
        self._actuation = cmd

    def set_server_info(self, info: dict) -> None:
        """Record the bridge's own connection info (host, OS, IPs, endpoints) so
        the monitor's Server card can show where clients should point."""
        self._global["serverInfo"] = info or {}
        if info.get("version"):
            self._global["version"] = info["version"]
        if info.get("server"):           # identity drives the sim-vs-real badge
            self._global["server"] = info["server"]

    def enabled(self) -> bool:
        return bool(self._subs)

    # ── subscriber side (monitor GUI / browser) ──────────────────────────────
    async def add_subscriber(self, ws) -> None:
        """Register a monitor ws and send it a full snapshot. A monitor is
        read-only EXCEPT it may rename the robot — {"type":"set_name","name":…} —
        which we hand to the bridge's rename handler so the new name persists and
        syncs to every client (the same as a rename from the iPhone)."""
        self._subs.add(ws)
        try:
            await ws.send(self._snapshot_json())
            async for raw in ws:
                try:
                    m = json.loads(raw)
                    if (isinstance(m, dict) and m.get("type") == "set_name"
                            and self._rename_cb):
                        nm = str(m.get("name", "")).strip()
                        if nm:
                            self._rename_cb(nm)
                except Exception:
                    pass
        except Exception:
            pass
        finally:
            self._subs.discard(ws)

    def set_rename_handler(self, cb) -> None:
        """Bridge supplies cb(new_name) — called when a monitor renames the robot."""
        self._rename_cb = cb

    def _snapshot_json(self) -> str:
        now = time.time()
        return json.dumps({
            "kind": "snapshot",
            "clients": [c.to_dict(now, self._global["driverId"])
                        for c in self._clients.values()],
            "global": dict(self._global),
            "log": list(self._log),
        }, separators=(",", ":"))

    def _publish(self, msg: dict) -> None:
        if not self._subs or self.loop is None:
            return
        data = json.dumps(msg, separators=(",", ":"))
        for ws in list(self._subs):
            self.loop.create_task(self._safe_send(ws, data))

    @staticmethod
    async def _safe_send(ws, data: str) -> None:
        try:
            await ws.send(data)
        except Exception:
            pass

    # ── event emission ────────────────────────────────────────────────────────
    def _emit(self, cid: int, peer: str, mtype: str, text: str,
              level: str = "info") -> None:
        self._seq += 1
        ev = {"seq": self._seq, "t": time.time(), "clientId": cid,
              "peer": peer, "type": mtype, "level": level, "text": text}
        self._log.append(ev)
        self._publish({"kind": "event", "event": ev})

    # ── producer side (called by the bridge ws handler, loop thread) ──────────
    def on_connect(self, cid: int, peer: str, device: dict | None = None,
                   net: dict | None = None) -> None:
        c = self._clients[cid] = _ClientStat(cid, peer, device, net)
        n = net or {}
        via = f" via {n['transport']}" if n.get("transport") else ""
        ip = n.get("ip") or peer
        self._emit(cid, peer, "hello",
                   f"{c.resolved_device()} connected{via} ({ip})", level="strong")

    def on_disconnect(self, cid: int) -> None:
        c = self._clients.pop(cid, None)
        peer = c.peer if c else "?"
        # Drop headline state this client was the source of, so the strip doesn't
        # show a vanished operator's stale head/intent/estop.
        f = self._fused
        if f.get("headPeer") == peer:
            f.update(headState="—", handL=False, handR=False, headPeer=None)
        if f.get("estopPeer") == peer:
            f.update(estop=False, estopPeer=None)
        self._emit(cid, peer, "bye", "client disconnected", level="warn")

    def record(self, cid: int, mtype: str, payload: dict) -> None:
        """Fold one decoded inbound message into the client's status, and log it
        if it is a discrete protocol event or a meaningful change.

        HOT PATH — runs on every inbound frame (≈60 Hz/client). When no monitor
        is subscribed this returns immediately, so teleop pays ZERO monitoring
        cost; per-client status simply (re)builds the moment a monitor connects."""
        if not self._subs:
            return
        c = self._clients.get(cid)
        if c is None:                              # connect race — make one
            c = self._clients[cid] = _ClientStat(cid, "?")
        now = time.time()
        c.last_seen = now
        c.msg_n += 1
        c.recv_times.append(now)
        peer = c.peer

        if mtype == "teleop_frame":
            # The display refreshes at ~5 Hz, so folding every 60 Hz frame is
            # wasted GIL time on the net thread (which the sim thread shares).
            # Fold at ~15 Hz; rate/seq still count every frame above. Discrete
            # protocol messages below are rare and always handled immediately.
            if now - c._last_fold >= 0.066:
                c._last_fold = now
                self._fold_frame(c, payload)
        elif mtype == "control_mode":
            self._fold_control(c, payload)
        elif mtype == "joint_command":
            joints = payload.get("joints", {}) or {}
            c.joint_n = len(joints)
            if now - c.last_joint_log >= _JOINT_LOG_MIN_GAP:
                c.last_joint_log = now
                self._emit(cid, peer, "joint_command",
                           f"{len(joints)} joint target(s)", level="strong")
        elif mtype == "reset":
            c.last_reset_t = now
            self._emit(cid, peer, "reset", "RESET — home the sim", level="reset")
        elif mtype == "recalibrate":
            c.last_recalib_t = now
            self._emit(cid, peer, "recalibrate", "re-anchor retarget clutch", level="strong")
        elif mtype == "record":
            if "select" in payload:
                self._fused["recTask"] = payload.get("select")
            self._emit(cid, peer, "record", self._record_text(payload), level="rec")
        elif mtype == "estop":
            active = bool(payload.get("active", payload.get("on", True)))
            self._fused["estop"] = active
            self._fused["estopPeer"] = peer if active else None
            self._emit(cid, peer, "estop",
                       "E-STOP engaged" if active else "E-stop released",
                       level="reset" if active else "strong")
        elif mtype == "external_force":
            forces = payload.get("forces", {}) or {}
            if forces:
                self._emit(cid, peer, "external_force",
                           f"push on {', '.join(list(forces)[:3])}", level="strong")
        elif mtype == "error":
            self._emit(cid, peer, "error", str(payload), level="warn")
        # ping / pong are pure liveness — counted (rate) but never logged.

    # ── folding helpers ──────────────────────────────────────────────────────
    def _fold_frame(self, c: _ClientStat, p: dict) -> None:
        c.frame_n += 1
        c.seq = int(p.get("sequence", c.seq))
        head = p.get("head") or {}
        new_head = head.get("trackingState", c.head_state) or c.head_state
        new_mode = p.get("mode", c.mode) or c.mode
        lh = p.get("leftHand")
        rh = p.get("rightHand")
        new_l = bool(lh and lh.get("isTracked", True))
        new_r = bool(rh and rh.get("isTracked", True))
        # Heuristic role only when the client did NOT identify itself (no tag/UA).
        if not c.device:
            if lh is not None or rh is not None:
                c.kind = "Headset"                 # streams skeletons → Vision Pro
            elif c.kind == "client":
                c.kind = "Handheld"                # frames w/o hands → iPhone / web
        j = p.get("navJoystick") or {}
        if j:
            c.nav = {k: round(float(j.get(k, 0.0)), 3) for k in ("lx", "ly", "rx", "ry")}
        nav_active = max(abs(c.nav[k]) for k in c.nav) > _NAV_DEADZONE

        # Log only on a meaningful change of operator intent.
        prev = (c.mode, c.hand_l, c.hand_r, c.head_state)
        c.mode, c.hand_l, c.hand_r, c.head_state = new_mode, new_l, new_r, new_head

        # Fuse into the headline. Mode/nav come from any client; head + hands
        # come ONLY from a head-tracker (Vision Pro, or anything streaming hand
        # skeletons) so a phone's identity head never masks the real one.
        f = self._fused
        f["mode"] = new_mode
        # A head source is any pose-tracking client: the Vision Pro, the iPhone /
        # iPad (active-perception head tracking — it reports on/off via the head
        # trackingState), or anything streaming hand skeletons. The Website has no
        # head sensor, so it never drives the head pill.
        is_headtracker = (c.device in ("Vision Pro", "iPhone", "iPad")
                          or lh is not None or rh is not None)
        if is_headtracker:
            f["headState"], f["handL"], f["handR"] = new_head, new_l, new_r
            f["headPeer"] = c.peer
            if j:
                f["nav"] = dict(c.nav)
        elif j:                                        # phone joystick still drives nav
            f["nav"] = dict(c.nav)

        if prev[0] != new_mode:
            self._emit(c.id, c.peer, "teleop_frame", f"mode → {new_mode}", level="strong")
        if prev[1] != new_l or prev[2] != new_r:
            hs = "+".join([s for s, on in (("L", new_l), ("R", new_r)) if on]) or "none"
            self._emit(c.id, c.peer, "teleop_frame", f"hands tracked: {hs}")
        if prev[3] != new_head:
            self._emit(c.id, c.peer, "teleop_frame", f"head tracking: {new_head}")
        # First time a stick crosses the deadzone after rest → note driving start.
        self._nav_edge(c, nav_active)

    def _nav_edge(self, c: _ClientStat, active: bool) -> None:
        if active and not c._nav_was:
            self._emit(c.id, c.peer, "teleop_frame", "joystick engaged", level="strong")
        c._nav_was = active

    def _fold_control(self, c: _ClientStat, p: dict) -> None:
        # Structured spec {method, controlType, region, stiffness}; legacy {mode}.
        if "method" in p:
            c.method = _clean(p["method"])
        if "controlType" in p:
            c.ctype = _clean(p["controlType"])
        if "region" in p:
            c.region = _clean(p["region"])
        if "mode" in p and not any(k in p for k in ("method", "controlType", "region")):
            legacy = {
                "navigation": ("navigation", "impedance", "whole_body"),
                "gravity_comp": ("wrist", "compliance", "arm"),
                "compliance": ("wrist", "compliance", "whole_body"),
                "joints": ("joint", "impedance", "arm"),
            }.get(p["mode"])
            if legacy:
                c.method, c.ctype, c.region = legacy
        if "stiffness" in p:
            try:
                c.stiffness = float(p["stiffness"])
            except (TypeError, ValueError):
                pass
        # Fuse into the headline immediately — a Hold→Soft toggle shows the
        # instant it arrives, whether or not this client is the active driver.
        f = self._fused
        f.update(method=c.method, ctype=c.ctype, region=c.region,
                 stiffness=c.stiffness, intentPeer=c.peer)
        label = f"{c.method} · {c.ctype} · {c.region}"
        self._emit(c.id, c.peer, "control_mode", label, level="strong")

    @staticmethod
    def _record_text(p: dict) -> str:
        if "select" in p:
            return f"select task ▸ {p.get('select')}"
        if p.get("active"):
            return f"REC ▶ start ▸ {p.get('task', '?')}"
        return "REC ■ stop"

    # ── heartbeat (called ~5 Hz from a bridge task, loop thread) ──────────────
    def tick(self, driver_id: int | None, recording: dict | None,
             online: int) -> None:
        if not self._subs:                         # nobody watching → do nothing
            return
        now = time.time()
        dc = self._clients.get(driver_id)
        rec = recording or {}
        # The recorder's status() uses `isRecording` + `task`; keep the selected
        # task in fused too so the folder shows the instant it's picked.
        if rec.get("task"):
            self._fused["recTask"] = rec.get("task")
        # Stamp the live actuation with WHO drove it (the arbitration driver), so
        # the monitor shows the standardized command AND its source device.
        act = self._actuation
        if act is not None and dc is not None:
            act = {**act, "source": {"device": dc.resolved_device(), "peer": dc.peer}}
        self._global.update(
            online=online,
            driverId=driver_id,
            driverPeer=dc.peer if dc else None,
            recording=rec or {"isRecording": False, "task": self._fused["recTask"]},
            status=dict(self._fused),
            actuation=act,
        )
        self._publish({
            "kind": "state",
            "clients": [c.to_dict(now, driver_id) for c in self._clients.values()],
            "global": dict(self._global),
        })


# Process-wide singleton the bridge imports.
MONITOR = MonitorHub()
