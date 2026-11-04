#!/usr/bin/env python3
"""
mabel_cockpit.py — the retro arcade cockpit run.sh / real_run.sh put in your
terminal: an 80s-game-style live dashboard of the robot, drawn with curses.

  ┌ mabel · sim ────────────────────────────────┬ robot ────────┐
  │ drive   vx vy wz asked/served · odom        │   ▛pixel▜     │
  │ joints  meas · cmd · err bars (arms/torso)  │   ▌MABEL▐     │
  │ hardware ● 7/7 LIVE · ◐ 5/7 PARTIAL · off: … ├ radar ────────┤
  │ status  host · torque · clients · ports     │  lidar sweep  │
  │ log     live tail of the stack's logs       │  lidar sweep  │
  ├ link ───────────────────────────────────────┴───────────────┤
  │ ▂▄▆█ state-stream rate history                              │
  └─────────────────────────────────────────────────────────────┘

Data sources (all read-only — the cockpit can never move the robot):
  * ws://127.0.0.1:9600  — the telemetry plane (proprio/control/status/lidar),
    spoken with a built-in ~100-line WebSocket client so this file needs
    NOTHING beyond the python3 stdlib (runs on the Mac and the Jetson alike).
  * ws://127.0.0.1:9090/teleop — READ-ONLY, for the two things the hub does not
    carry: the module roster (module_status) and per-motor health (motorHealth),
    which feed the `hardware` panel (NOVA's cockpit panel, ported 2026-10-01).
  * TCP port probes      — which services are actually listening.
  * log tails            — the launcher's per-process log files.
The ONLY writes it does: ENTER / R keystrokes forwarded into the bridge's
stdin FIFO (sim: clutch re-anchor / sim reset) — the same keys the bridge
always took on its own stdin.

Modes:
  --mode sim|real     port set, keys, and systems panel (default sim)
  --demo              synthetic robot (no server needed) — dev/screenshot/test
  --frames N --dump F render N frames headless, write the screen text to F,
                      exit (the test harness drives this under a pty)
Exit: q (or Ctrl-C). The launcher treats cockpit exit as "shut it all down".
"""
from __future__ import annotations

import argparse
import base64
import curses
import errno
import json
import math
import os
import re
import socket
import struct
import sys
import threading
import time
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sprite  # noqa: E402


# ═════════════════════════════════ tiny websocket client ═══════════════════
class WSClosed(Exception):
    pass


class TinyWS:
    """Minimal RFC6455 TEXT-frame client (stdlib only): handshake, fragmented
    receive, ping→pong, masked send. Enough to subscribe to the :9600 hub."""

    def __init__(self, host: str, port: int, path: str = "/", timeout: float = 6.0):
        self.sock = socket.create_connection((host, port), timeout=3.0)
        self.sock.settimeout(timeout)
        key = base64.b64encode(os.urandom(16)).decode()
        self.sock.sendall(
            (f"GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\n"
             "Upgrade: websocket\r\nConnection: Upgrade\r\n"
             f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n"
             ).encode())
        buf = b""
        while b"\r\n\r\n" not in buf:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise WSClosed("handshake EOF")
            buf += chunk
            if len(buf) > 65536:
                raise WSClosed("handshake overflow")
        head, _, rest = buf.partition(b"\r\n\r\n")
        if b" 101" not in head.split(b"\r\n", 1)[0]:
            raise WSClosed("no 101 upgrade")
        self._buf = bytearray(rest)

    def _read_exact(self, n: int) -> bytes:
        while len(self._buf) < n:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise WSClosed("EOF")
            self._buf += chunk
        out = bytes(self._buf[:n])
        del self._buf[:n]
        return out

    def recv_text(self) -> str:
        msg, opcode = bytearray(), None
        while True:
            b1, b2 = self._read_exact(2)
            fin, op = b1 & 0x80, b1 & 0x0F
            masked, ln = b2 & 0x80, b2 & 0x7F
            if ln == 126:
                ln = struct.unpack(">H", self._read_exact(2))[0]
            elif ln == 127:
                ln = struct.unpack(">Q", self._read_exact(8))[0]
            mask = self._read_exact(4) if masked else None
            payload = self._read_exact(ln)
            if mask:
                payload = bytes(c ^ mask[i % 4] for i, c in enumerate(payload))
            if op == 0x9:                       # ping → pong (keeps the hub happy)
                self._send_frame(0xA, payload)
                continue
            if op == 0xA:
                continue
            if op == 0x8:
                raise WSClosed("server close")
            if op in (0x1, 0x2):
                opcode, msg = op, bytearray(payload)
            elif op == 0x0:
                msg += payload
            if fin and opcode is not None:
                if opcode == 0x1:
                    return msg.decode("utf-8", "replace")
                msg, opcode = bytearray(), None     # binary frames: skip

    def _send_frame(self, op: int, payload: bytes = b"") -> None:
        mask = os.urandom(4)
        head = bytes([0x80 | op])
        ln = len(payload)
        if ln < 126:
            head += bytes([0x80 | ln])
        elif ln < 65536:
            head += bytes([0x80 | 126]) + struct.pack(">H", ln)
        else:
            head += bytes([0x80 | 127]) + struct.pack(">Q", ln)
        self.sock.sendall(head + mask + bytes(c ^ mask[i % 4] for i, c in enumerate(payload)))

    def send_text(self, s: str) -> None:
        self._send_frame(0x1, s.encode())

    def close(self) -> None:
        try:
            self._send_frame(0x8)
            self.sock.close()
        except Exception:
            pass


# ═════════════════════════════════ shared state ════════════════════════════
class Robot:
    """Everything the panels draw, filled by the telemetry thread (or DemoFeed)."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.connected = False
        self.err = ""
        self.frames: dict[str, dict] = {}      # type -> latest frame
        self.rx_t: dict[str, float] = {}       # type -> wall clock of last rx
        self.stamps: deque = deque(maxlen=200)  # wall times of NEW proprio stamps
        self._last_stamp = None
        self.hz_hist: deque = deque(maxlen=600)  # (wall, hz) sampled 2/s for the chart
        self._hz_next = 0.0
        self.trail: deque = deque(maxlen=800)     # sparse (x, y) breadcrumbs (radar)
        self.pose_hist: deque = deque(maxlen=80)  # dense (wall, x, y, yaw) (velocity)
        self.t0 = time.time()

    def ingest(self, frame: dict) -> None:
        ftype = frame.get("type")
        if not ftype:
            return
        now = time.time()
        with self.lock:
            self.frames[ftype] = frame
            self.rx_t[ftype] = now
            if ftype == "proprio":
                st = frame.get("stamp")
                if st is not None and st != self._last_stamp:
                    self._last_stamp = st
                    self.stamps.append(now)
                base = frame.get("base") or {}
                x, y = base.get("x"), base.get("y")
                if x is not None and y is not None:
                    self.pose_hist.append((now, x, y, base.get("yaw", 0.0)))
                    if not self.trail or (self.trail[-1][0] - x) ** 2 + (self.trail[-1][1] - y) ** 2 > 4e-4:
                        self.trail.append((x, y))
            if now >= self._hz_next:
                self._hz_next = now + 0.5
                self.hz_hist.append((now, self.state_hz()))

    def state_hz(self) -> float:
        now = time.time()
        return float(sum(1 for t in self.stamps if now - t <= 1.0))

    def get(self, ftype: str) -> dict:
        with self.lock:
            return self.frames.get(ftype) or {}


class TelemetryThread(threading.Thread):
    def __init__(self, robot: Robot, host: str, port: int):
        super().__init__(daemon=True)
        self.robot, self.host, self.port = robot, host, port
        self.stop_flag = False

    def run(self) -> None:
        while not self.stop_flag:
            ws = None
            try:
                ws = TinyWS(self.host, self.port, "/telemetry")
                self.robot.connected = True
                self.robot.err = ""
                while not self.stop_flag:
                    self.robot.ingest(json.loads(ws.recv_text()))
            except (WSClosed, OSError, socket.timeout, json.JSONDecodeError) as e:
                self.robot.connected = False
                self.robot.err = f"{type(e).__name__}: {e}"[:60]
            finally:
                if ws:
                    ws.close()
            for _ in range(10):                 # 1 s backoff, responsive to stop
                if self.stop_flag:
                    return
                time.sleep(0.1)


class PortProber(threading.Thread):
    def __init__(self, ports: list[tuple[str, int]]):
        super().__init__(daemon=True)
        self.ports = ports
        self.up: dict[int, bool] = {}
        self.stop_flag = False

    def run(self) -> None:
        while not self.stop_flag:
            for _, port in self.ports:
                try:
                    s = socket.create_connection(("127.0.0.1", port), timeout=0.3)
                    s.close()
                    self.up[port] = True
                except OSError:
                    self.up[port] = False
            for _ in range(20):
                if self.stop_flag:
                    return
                time.sleep(0.1)


class CameraPoller(threading.Thread):
    """Poll the video box's /cameras.json roster (~2 s): per-camera freshness,
    errors, viewers — and an fps estimate from the frame-count deltas. Pure
    read-only metadata; it never opens a stream (each MJPEG client costs the
    server a transcode thread, so probing streams would tax the robot)."""

    def __init__(self, port: int, demo: bool = False):
        super().__init__(daemon=True)
        self.port, self.demo = port, demo
        self.cams: list[dict] = []
        self.ok = False
        self._prev: dict[str, tuple[float, int]] = {}   # name -> (t, frames)
        self.stop_flag = False

    def _poll_once(self) -> None:
        import urllib.request
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/cameras.json",
            headers={"User-Agent": "mabel-cockpit"})
        with urllib.request.urlopen(req, timeout=1.5) as r:
            data = json.loads(r.read().decode())
        now = data.get("t", time.time())
        by_name = {}
        for c in data.get("cameras", []):
            name = c.get("name", "?")
            frames = int(c.get("frames") or 0)
            fps = None
            prev = self._prev.get(name)
            if prev and now > prev[0]:
                fps = max(0.0, (frames - prev[1]) / (now - prev[0]))
            self._prev[name] = (now, frames)
            by_name[name] = {"name": name, "label": c.get("label") or "",
                             "error": c.get("error"), "ageMs": c.get("ageMs"),
                             "streaming": bool(c.get("streaming")), "fps": fps}
        # the FOUR cameras the operator cares about (stereo eyes fold into
        # "front"): head/main → front · front-base cam → base · both wrists
        out = []
        for roster, shown in (("main", "front"), ("front", "base"),
                              ("wrist_left", "wrist L"), ("wrist_right", "wrist R")):
            c = by_name.get(roster)
            if c:
                out.append({**c, "name": shown})
            else:
                out.append({"name": shown, "label": "", "error": None,
                            "ageMs": None, "streaming": False, "fps": None})
        self.cams = out
        self.ok = True

    def run(self) -> None:
        while not self.stop_flag:
            if self.demo:
                t = time.time()
                self.cams = [
                    {"name": n, "label": lb, "error": None,
                     "ageMs": 30 + i * 11, "streaming": i < 3,
                     "fps": 24 + 4 * math.sin(t * 0.7 + i)}
                    for i, (n, lb) in enumerate(
                        (("front", "Head"), ("base", "Front Base"),
                         ("wrist L", "Wrist Left"), ("wrist R", "Wrist Right")))]
                self.ok = True
            else:
                try:
                    self._poll_once()
                except Exception:
                    self.ok = False
                    self.cams = []
            for _ in range(20):
                if self.stop_flag:
                    return
                time.sleep(0.1)


class TeleopCommander(threading.Thread):
    """Keyboard base/lift teleop — a REGULAR client of the one wire contract
    (ws://:9090/teleop, teleop_frame + navJoystick, exactly like the web
    console), so every command passes the controller's safe_base_step gate
    and the normal driver arbitration. Never a private side channel.

    Keys auto-repeat while held; each press refreshes a short deadman window
    and a 20 Hz sender streams the stick values until it lapses, then sends
    one zero frame — release the key and the robot stops."""

    RATE = 20.0
    HOLD_S = 0.35            # key-repeat deadman: refreshed by every press

    def __init__(self, host: str, port: int):
        super().__init__(daemon=True)
        self.host, self.port = host, port
        self.stick = {"lx": 0.0, "ly": 0.0, "rx": 0.0, "ry": 0.0}
        self.until = 0.0
        self.seq = 0
        self.connected = False
        self.stop_flag = False
        self._lock = threading.Lock()

    def nudge(self, **axes) -> None:
        with self._lock:
            self.stick.update({k: 0.0 for k in self.stick})
            self.stick.update(axes)
            self.until = time.time() + self.HOLD_S

    def active(self) -> bool:
        return time.time() < self.until

    def _frame(self, stick: dict) -> str:
        self.seq += 1
        return json.dumps({"type": "teleop_frame", "payload": {
            "sequence": self.seq, "timestamp": time.time(),
            "mode": "Base Driving", "navJoystick": dict(stick)}})

    def run(self) -> None:
        ws = None
        was_active = False
        while not self.stop_flag:
            time.sleep(1.0 / self.RATE)
            with self._lock:
                stick = dict(self.stick)
                live = time.time() < self.until
            if not live and not was_active:
                continue
            try:
                if ws is None:
                    ws = TinyWS(self.host, self.port, "/teleop", timeout=3.0)
                    self.connected = True
                if live:
                    ws.send_text(self._frame(stick))
                    was_active = True
                else:                       # key released → one stop frame
                    ws.send_text(self._frame({k: 0.0 for k in stick}))
                    was_active = False
            except (WSClosed, OSError) as e:
                self.connected = False
                was_active = False
                if ws:
                    ws.close()
                ws = None
        if ws:
            try:                            # parting stop frame, best effort
                ws.send_text(self._frame({k: 0.0 for k in self.stick}))
            except Exception:
                pass
            ws.close()


ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]|\x1b\][^\x07]*\x07|[\r\x07]")
HW_LINE_RE = re.compile(r"\[hw (\d\d:\d\d:\d\d)\]\s*(.*)")


# ═══════════════════════ hardware roster (the NOVA-style panel) ═════════════
# "Is the arm actually on the bus?" used to be answered only by a `[hw …]` line that
# thor's real_run.sh watchdog wrote into a log — nothing else drew it, and a launcher
# that does not write that line (the Mac) left the cockpit with no connection status
# at all. NOVA's cockpit answers it with a dedicated panel: every module, how many of
# its motors answer of how many, and WHICH are silent. MABEL has the same facts: the
# gateway puts the module roster (module_status) and per-motor health (/arms and
# /body motor_health) on every robot_state, so GatewayFeed reads them and the
# helpers below shape them into the rows the panel draws.

def _lan_ip() -> str:
    """This host's LAN address (what the apps type in), without sending a packet."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "—"


LAN_IP = _lan_ip()

_HAND_PARTS = ("thumb", "index", "middle", "ring", "pinky", "wrist")


def _module_of(joint: str) -> "str | None":
    """Which roster module owns a DDS/wire joint name (server/module_status.py rules)."""
    j = joint.lower().replace("_", " ")
    if "steer" in j or "drive" in j or j == "base root":
        return "base"
    if "lift" in j:
        return "lift"
    if j == "torso":
        return "torso"
    if j.startswith("neck"):
        return "neck"
    if j.startswith("left "):
        if any(p in j for p in _HAND_PARTS):
            return "leftHand"
        if "arm" in j:
            return "leftArm"
    if j.startswith("right "):
        if any(p in j for p in _HAND_PARTS):
            return "rightHand"
        if "arm" in j:
            return "rightArm"
    return None


# One grid cell per PHYSICAL module, fixed order. The neck is two cells because it is
# two transports — the yaw a DaMiao on the arms Teensy's CAN, pitch + roll one 2XL430 on
# its serial bus — and one "neck 1/3" would hide exactly which half is dead.
#   (label, roster id, joint picker, motors expected when nothing has named them,
#    whether the roster row's own "N joints" count is this row's total — not for the
#    neck, whose one roster row is split across two cells)
_MODULE_ROWS = (
    ("arm L",        "leftArm",   lambda n: n.startswith("left_arm_"),             7, True),
    ("arm R",        "rightArm",  lambda n: n.startswith("right_arm_"),            7, True),
    ("torso",        "torso",     lambda n: n == "torso",                          1, True),
    ("lift",         "lift",      lambda n: n == "lift",                           1, True),
    ("neck yaw",     "neck",      lambda n: n == "neck_yaw",                       1, False),
    ("neck pit+rol", "neck",      lambda n: n in ("neck_pitch", "neck_roll"),      2, False),
    ("hand L",       "leftHand",  lambda n: _module_of(n) == "leftHand",          17, True),
    ("hand R",       "rightHand", lambda n: _module_of(n) == "rightHand",         17, True),
    ("base",         "base",      lambda n: _module_of(n) == "base",               6, True),
)
_ROSTER_N_RE = re.compile(r"(\d+)\s+joint")
# Only J1, J2 (shoulder) and J4 (elbow) are documented; the rest keep their index.
_ARM_JOINT_NAMES = {"1": "shoulder1", "2": "shoulder2", "4": "elbow"}
# status.torque keys (the gateway's intent) per module, when per-motor health has none
_TORQUE_KEY = {"arm L": "arms", "arm R": "arms", "neck yaw": "neck", "neck pit+rol": "neck",
               "hand L": "left_hand", "hand R": "right_hand", "base": "base", "lift": "lift"}
_SENSOR_ROWS = (("camHead", "cam head"), ("camFront", "cam front"),
                ("camWristLeft", "cam wristL"), ("camWristRight", "cam wristR"),
                ("lidar", "lidar"), ("speaker", "speaker"), ("mic", "mic"))


def _short(urdf: str) -> str:
    """One motor's short name for an 'off:' list: elbow, shoulder2, J5, pitch, thumb mcp."""
    if urdf.startswith(("left_arm_", "right_arm_")):
        i = urdf.rsplit("_", 1)[-1]
        return _ARM_JOINT_NAMES.get(i, "J" + i)
    if urdf.startswith("neck_"):
        return urdf[5:]
    parts = urdf.split("_")
    if parts[0] in ("left", "right") and len(parts) >= 2:
        return " ".join(parts[1:])
    return urdf.replace("_", " ")


def _age(ms) -> str:
    """Feedback age, narrow: `4 ms` · `12 s` · `84 min`. A dead motor's age is the
    board's uptime, which is the honest way the firmware says 'never heard it'."""
    if ms is None:
        return "—"
    ms = float(ms)
    if ms < 1000:
        return f"{ms:.0f} ms"
    if ms < 60000:
        return f"{ms / 1000:.0f} s"
    return f"{ms / 60000:.0f} min"


def _module_rows(hw: dict, joints: dict, torque: dict) -> list:
    """[(label, live, n, state, colour, detail, age_text)] — state LIVE / PARTIAL / OFFLINE / —.

    Liveness is PER MOTOR where the firmware has an opinion (motorHealth.live for the
    arms, torso and neck — the DaMiao / DYNAMIXEL feedback window), else "present in the
    joint stream" (hands, base wheels). The roster row's own verdict (module_status:
    topic seen within TTL) is the fallback for a module that names no joints."""
    roster = {r.get("id"): r for r in (hw.get("roster") or []) if isinstance(r, dict)}
    health = hw.get("health") or {}
    out = []
    for label, mid, pick, expected, use_roster_n in _MODULE_ROWS:
        row = roster.get(mid)
        names = sorted(n for n in set(joints) | set(health) if pick(n))
        if row is None and not names:
            continue                                    # this robot has no such module
        m = _ROSTER_N_RE.search(str((row or {}).get("detail") or ""))
        roster_n = int(m.group(1)) if (m and use_roster_n) else 0
        # The ROSTER's verdict comes first: module_status marks a module down when its
        # topic has gone silent (TTL), and the joint mirror keeps the last value of a
        # dead module forever — "present in the stream" is not evidence after that.
        down = row is not None and not row.get("ok")
        if names and not down:
            def alive(nm):
                h = health.get(nm) or {}
                if "live" in h:
                    return bool(h["live"])
                if h.get("stale_ms") is not None:       # older HAL: no flag, but an age —
                    return float(h["stale_ms"]) < 1000   # a never-heard motor's age is the uptime
                return nm in joints
            live = [n for n in names if alive(n)]
            off = [n for n in names if n not in live]
            n = max(len(names), roster_n or expected)
            unnamed = n - len(names)
            # feedback age of the motors that ARE answering — the silent ones are named
            # on their own line, and their age is just the board's uptime
            ages = [health[nm].get("stale_ms") for nm in (live or names)
                    if nm in health and health[nm].get("stale_ms") is not None]
            age = max(ages) if ages else None
        else:                                           # jointless module: the roster decides
            n = max(len(names), roster_n or expected)
            live = list(range(n)) if ((row or {}).get("ok") and not down) else []
            off, unnamed, age = [], 0, None
        if down:
            state, colour, detail = "OFFLINE", "red", str(row.get("detail") or "not connected")
            live = []
        elif n and not live:
            state, colour, detail = "OFFLINE", "red", "no motor answering"
        elif off:
            state, colour = "PARTIAL", "yellow"
            detail = (("on: " + ", ".join(_short(x) for x in live)) if len(live) < len(off)
                      else ("off: " + ", ".join(_short(x) for x in off)))
            if unnamed:
                detail += f" +{unnamed} unnamed"
        elif unnamed:
            state, colour, detail = "PARTIAL", "yellow", f"{unnamed} silent"
        elif n:
            en = [health[nm].get("enabled") for nm in names
                  if nm in health and health[nm].get("enabled") is not None]
            tq = all(en) if en else torque.get(_TORQUE_KEY.get(label))
            state, colour = "LIVE", "green"
            detail = "torque ON" if tq else ("torque off" if tq is False else "")
        else:
            state, colour, detail = "—", "dim", "no driver"
        out.append((label, len(live), n, state, colour, detail, _age(age)))
    return out


def _sensor_rows(hw: dict) -> list:
    """[(label, live, n, state, colour, detail)] for the roster's sensor/audio rows —
    the same cell shape as a module, with the roster's own evidence ("24 fps",
    "360 rays", "no data 9s") as the rate text."""
    roster = {r.get("id"): r for r in (hw.get("roster") or []) if isinstance(r, dict)}
    out = []
    for mid, label in _SENSOR_ROWS:
        r = roster.get(mid)
        if r is None:
            continue
        ok = bool(r.get("ok"))
        detail = str(r.get("detail") or ("live" if ok else "not connected"))
        # the roster's evidence, shortened to the grid's 6–7 char rate slot
        detail = (detail.replace(" rays", " ray").replace("no data ", "dead ")
                  .replace("not connected", "down").replace("ready", "ready"))
        out.append((label, 1 if ok else 0, 1, "LIVE" if ok else "OFFLINE",
                    "green" if ok else "red", detail))
    return out


def _hw_grid(iw: int) -> tuple:
    """(cell width, label width, columns) — ONE function for the layout and the drawing."""
    wide = iw >= 90
    cell, lab = (30, 12) if wide else (26, 9)
    return cell, lab, max(1, min(4, iw // cell))


class GatewayFeed(threading.Thread):
    """READ-ONLY client of the teleop wire (ws://:9090/teleop) for the module roster and
    per-motor health. It never sends a frame. robot_state runs at ~100 Hz, so only the
    frames carrying `modules` or `motorHealth` are parsed (a substring test, not JSON)."""

    def __init__(self, robot: Robot, host: str, port: int):
        super().__init__(daemon=True)
        self.robot, self.host, self.port = robot, host, port
        self.connected = False
        self.stop_flag = False

    def run(self) -> None:
        while not self.stop_flag:
            ws = None
            roster, health = None, {}
            try:
                ws = TinyWS(self.host, self.port, "/teleop?client=cockpit", timeout=6.0)
                self.connected = True
                next_health = 0.0
                while not self.stop_flag:
                    raw = ws.recv_text()
                    if raw.startswith('{"type":"hello"'):
                        m = json.loads(raw)
                        self.robot.ingest({"type": "gateway_hello", **(m.get("payload") or {})})
                        continue
                    has_roster = '"modules"' in raw
                    if not has_roster and '"motorHealth"' not in raw:
                        continue
                    now = time.monotonic()
                    if not has_roster and now < next_health:
                        continue                # health rides every frame: 5 Hz is plenty
                    next_health = now + 0.2
                    m = json.loads(raw)
                    if m.get("type") != "robot_state":
                        continue
                    p = m.get("payload") or {}
                    if isinstance(p.get("modules"), list):
                        roster = p["modules"]
                    if isinstance(p.get("motorHealth"), dict):
                        health = p["motorHealth"]
                    self.robot.ingest({"type": "hardware", "roster": roster,
                                       "health": health, "t": time.time()})
            except (WSClosed, OSError, ValueError):
                self.connected = False
                time.sleep(1.0)
            finally:
                if ws:
                    try:
                        ws.close()
                    except Exception:
                        pass


class LogTail(threading.Thread):
    """Follow one log file into a shared ring; also captures the newest
    real_run watchdog `[hw HH:MM:SS] …` line for the systems panel."""

    def __init__(self, name: str, path: str, ring: deque, ring_lock: threading.Lock,
                 hw_slot: dict):
        super().__init__(daemon=True)
        self.name, self.path = name, path
        self.ring, self.ring_lock, self.hw_slot = ring, ring_lock, hw_slot
        self.stop_flag = False

    def run(self) -> None:
        pos = 0
        first = True
        while not self.stop_flag:
            try:
                size = os.path.getsize(self.path)
                if size < pos:
                    pos = 0                      # truncated/rotated
                if first:
                    pos = max(0, size - 8192)    # start near the end
                    first = False
                if size > pos:
                    with open(self.path, "rb") as f:
                        f.seek(pos)
                        data = f.read(min(65536, size - pos))
                        pos += len(data)
                    for raw in data.decode("utf-8", "replace").splitlines():
                        line = ANSI_RE.sub("", raw).rstrip()
                        if not line.strip():
                            continue
                        m = HW_LINE_RE.search(line)
                        if m:
                            self.hw_slot["t"], self.hw_slot["line"] = m.group(1), m.group(2)
                        with self.ring_lock:
                            self.ring.append((self.name, line[:220]))
            except OSError:
                pass
            for _ in range(4):
                if self.stop_flag:
                    return
                time.sleep(0.1)


# ═════════════════════════════════ demo feed ═══════════════════════════════
class DemoFeed(threading.Thread):
    """Synthetic robot for --demo: fills the same frames the :9600 plane sends,
    with a scripted arc — boot → idle → drive a circle → e-stop → recover —
    so every panel exercises without a server. Deterministic in t."""

    def __init__(self, robot: Robot, time_fn=None):
        super().__init__(daemon=True)
        self.robot = robot
        self.stop_flag = False
        self.time_fn = time_fn or (lambda: time.time() - self.robot.t0)

    def frame_at(self, t: float) -> list[dict]:
        if t < 1.2:
            return []
        estop = 16.0 < t < 19.0
        driving = 6.0 < t < 14.0 and not estop
        vx = 0.55 if driving else 0.0
        wz = 0.5 if driving else 0.0
        # integrate a circle
        td = min(max(t - 6.0, 0.0), 8.0)
        yaw = wz and 0.5 * td or 0.0
        x = (0.55 / 0.5) * math.sin(0.5 * td)
        y = (0.55 / 0.5) * (1 - math.cos(0.5 * td))
        joints = {}
        for side in ("left", "right"):
            sgn = 1 if side == "left" else -1
            for i in range(1, 8):
                q = 0.45 * math.sin(0.9 * t + i * 0.7) * sgn
                joints[f"{side}_arm_{i}"] = {"q": round(q, 4),
                                             "dq": round(0.4 * math.cos(0.9 * t + i), 3),
                                             "tau": round(2.2 * math.sin(t + i), 2),
                                             "temp": None}
        joints["torso"] = {"q": 0.12 * math.sin(0.3 * t), "dq": 0.0, "tau": 1.0, "temp": None}
        joints["neck_pitch"] = {"q": 0.2 * math.sin(0.5 * t), "dq": 0, "tau": 0, "temp": None}
        joints["neck_yaw"] = {"q": 0.6 * math.sin(0.25 * t), "dq": 0, "tau": 0, "temp": None}
        joints["lift"] = {"q": 0.16 + 0.15 * (1 + math.sin(0.2 * t)), "dq": 0, "tau": 0, "temp": None}
        grip = 0.5 * (1 + math.sin(0.7 * t))
        for f in ("thumb_mcp", "index_pip", "middle_pip", "ring_pip", "pinky_pip"):
            joints[f"left_{f}"] = {"q": round(grip, 3), "dq": 0, "tau": 0, "temp": None}
            joints[f"right_{f}"] = {"q": round(1 - grip, 3), "dq": 0, "tau": 0, "temp": None}
        desired = {}
        for side in ("Left", "Right"):
            for i in range(1, 8):
                m = joints[f"{side.lower()}_arm_{i}"]["q"]
                desired[f"{side} Arm {i}"] = round(m + 0.06 * math.sin(2.3 * t + i), 4)
        desired["Torso"] = joints["torso"]["q"]
        # lidar: a 6×5 m room around the origin + a pillar, seen from (x, y, yaw)
        n = 90
        ranges = []
        for i in range(n):
            a = yaw + i * 2 * math.pi / n
            dx, dy = math.cos(a), math.sin(a)
            best = 8.0
            for wall_d, nx, ny in ((3.0 - x, 1, 0), (3.0 + x, -1, 0),
                                   (2.5 - y, 0, 1), (2.5 + y, 0, -1)):
                denom = dx * nx + dy * ny
                if denom > 1e-6:
                    best = min(best, max(0.05, wall_d / denom))
            px, py = 1.5 - x, 1.0 - y      # pillar
            proj = px * dx + py * dy
            if proj > 0:
                perp2 = px * px + py * py - proj * proj
                if perp2 < 0.09:
                    best = min(best, proj - math.sqrt(0.09 - perp2))
            ranges.append(round(min(best, 8.0), 3))
        return [
            {"type": "hello", "robot": {"name": "mabel-demo", "kind": "demo",
                                        "server": "mabel-sim-demo", "version": "0"}},
            {"type": "proprio", "stamp": t, "joints": joints,
             "base": {"x": x, "y": y, "z": 0.078, "yaw": yaw,
                      "steer": [0.1] * 4, "drive": [vx * 8] * 4}},
            {"type": "control",
             "mode": {"method": "wholebody" if driving else "navigation",
                      "type": "impedance", "region": "arm", "stiffness": 0.4},
             "desired": desired,
             "baseTwist": {"vx": vx, "vy": 0.0, "wz": wz},
             "engaged": 3.0 < t and not estop, "estop": estop},
            {"type": "status", "estop": estop,
             "controlMode": {"method": "wholebody" if driving else "navigation"},
             "latency": {"cmdStalenessMs": 12.0 + 3 * math.sin(t), "stateAgeMs": 8.0},
             "battery": {"available": False, "reason": "not applicable (sim)"},
             "torque": {"left_hand": True, "right_hand": True, "arms": not estop},
             "policy": {"selected": None, "running": False},
             "animation": {"selected": None, "playing": False},
             "driver": {"clients": 2},
             "recording": {"total": 12}, "dataCollection": {"active": False}},
            {"type": "lidar", "angleMin": 0.0, "angleInc": 2 * math.pi / n,
             "rangeMax": 8.0, "stamp": t, "ranges": ranges},
            # the gateway's roster + per-motor health, as the real robot read on
            # 2026-09-30: right shoulder2 and the neck servos never answered the Teensy
            {"type": "hardware", "t": t,
             "roster": [{"id": i, "label": i, "ok": ok, "detail": d} for i, ok, d in (
                 ("base", True, "6 joints"), ("torso", True, "1 joint"),
                 ("lift", False, "not connected"), ("neck", True, "3 joints"),
                 ("leftArm", True, "7 joints"), ("rightArm", True, "7 joints"),
                 ("leftHand", True, "5 joints"), ("rightHand", True, "5 joints"),
                 ("camHead", True, "24 fps"), ("camFront", True, "24 fps"),
                 ("camWristLeft", True, "15 fps"), ("camWristRight", False, "no data 9s"),
                 ("lidar", True, "360 rays"), ("speaker", True, "ready"), ("mic", True, "live"))],
             "health": {**{f"{sd}_arm_{i}": {"live": True, "enabled": not estop, "stale_ms": 4}
                           for sd in ("left", "right") for i in range(1, 8)},
                        "right_arm_2": {"live": False, "stale_ms": 5020544},
                        "torso": {"live": True, "enabled": not estop, "stale_ms": 4},
                        "neck_yaw": {"live": True, "enabled": not estop, "stale_ms": 4},
                        "neck_pitch": {"live": False, "stale_ms": 5020544},
                        "neck_roll": {"live": False, "stale_ms": 5020544}}},
        ]

    def run(self) -> None:
        while not self.stop_flag:
            t = self.time_fn()
            frames = self.frame_at(t)
            self.robot.connected = bool(frames)
            for fr in frames:
                self.robot.ingest(fr)
            time.sleep(0.05)


# ═════════════════════════════════ the cockpit ═════════════════════════════
IOS_ARM = [f"arm_{i}" for i in range(1, 8)]
BLOCKS = " ▁▂▃▄▅▆▇█"

LOGO = [  # 5×5 pixel font, rendered in half-blocks on the boot screen
    "X...X..XXX..XXXX..XXXXX.X....",
    "XX.XX.X...X.X...X.X.....X....",
    "X.X.X.XXXXX.XXXX..XXXX..X....",
    "X...X.X...X.X...X.X.....X....",
    "X...X.X...X.XXXX..XXXXX.XXXXX",
]


class Cockpit:
    FPS = 15

    def __init__(self, args, robot: Robot, prober: PortProber,
                 cameras: CameraPoller, pilot: "TeleopCommander | None",
                 log_ring: deque, log_lock: threading.Lock, hw_slot: dict):
        self.args = args
        self.robot = robot
        self.prober = prober
        self.cameras = cameras
        self.pilot = pilot
        self.log_ring, self.log_lock, self.hw_slot = log_ring, log_lock, hw_slot
        self.pairs: dict[tuple[int, int], int] = {}
        self.next_pair = 1
        self.show_log = True
        self.log_forced = False     # user pressed `l`: log may steal joint rows
        self.log_visible = False    # whether the last layout actually drew it
        self.paused = False
        self.help_on = False
        self.toast = ("", 0.0)
        self.blink_t = 0.0
        self.quit = False
        self.has_256 = False
        self.frame_ms = deque(maxlen=120)     # render cost, reported by --dump
        self.shadow: list[list[str]] = []     # text mirror of the screen (dumps)

    # ── colors ──────────────────────────────────────────────────────────────
    def color(self, fg: int, bg: int = -1) -> int:
        if not curses.has_colors():
            return 0
        key = (fg, bg)
        if key not in self.pairs:
            if self.next_pair >= curses.COLOR_PAIRS - 1:
                return 0
            try:
                curses.init_pair(self.next_pair, fg, bg)
            except curses.error:
                return 0
            self.pairs[key] = self.next_pair
            self.next_pair += 1
        return curses.color_pair(self.pairs[key])

    def pal(self, key: str) -> int:
        table = sprite.PALETTE_256 if self.has_256 else sprite.PALETTE_8
        return self.color(table.get(key, 7))

    def theme(self, name: str) -> int:
        """Named UI colors — 256-color first, 8-color fallback."""
        c256 = {"dim": 242, "text": 251, "bright": 255, "cyan": 51, "dcyan": 30,
                "green": 46, "dgreen": 28, "yellow": 220, "orange": 208,
                "red": 196, "magenta": 201, "blue": 39}
        c8 = {"dim": 0, "text": 7, "bright": 7, "cyan": 6, "dcyan": 6,
              "green": 2, "dgreen": 2, "yellow": 3, "orange": 3,
              "red": 1, "magenta": 5, "blue": 4}
        t = c256 if self.has_256 else c8
        return self.color(t.get(name, 7))

    # ── safe drawing helpers ────────────────────────────────────────────────
    def put(self, y: int, x: int, s: str, attr: int = 0) -> None:
        h, w = self.scr.getmaxyx()
        if y < 0 or y >= h or x >= w:
            return
        if x < 0:
            s = s[-x:]
            x = 0
        s = s[: w - x]
        try:
            self.scr.addstr(y, x, s, attr)
        except curses.error:
            pass
        # shadow text buffer: curses instr() truncates multibyte rows, so the
        # --dump screenshots (and the test harness) read from this instead
        if y < len(self.shadow):
            row = self.shadow[y]
            for i, ch in enumerate(s):
                if x + i < len(row):
                    row[x + i] = ch

    def panel(self, y: int, x: int, h: int, w: int, label: str,
              rlabel: str = "", badges=()) -> tuple[int, int, int, int]:
        """Box with a microduck-style embedded label; returns inner rect. `badges` =
        [(text, colour)] drawn right after the label on the top border."""
        dim = self.theme("dim")
        self.put(y, x, "┌" + "─" * (w - 2) + "┐", dim)
        for yy in range(y + 1, y + h - 1):
            self.put(yy, x, "│", dim)
            self.put(yy, x + w - 1, "│", dim)
        self.put(y + h - 1, x, "└" + "─" * (w - 2) + "┘", dim)
        if label:
            self.put(y, x + 2, f" {label} ", self.theme("cyan") | curses.A_BOLD)
            cx = x + 4 + len(label)
            for txt, colour in badges:
                self.put(y, cx, f" {txt} ", self.theme(colour))
                cx += len(txt) + 3
        if rlabel:
            self.put(y, x + w - 3 - len(rlabel), f" {rlabel} ", dim)
        return (y + 1, x + 1, h - 2, w - 2)

    def bar(self, y: int, x: int, width: int, frac: float, attr: int) -> None:
        frac = max(0.0, min(1.0, frac))
        cells = frac * width
        full = int(cells)
        s = "█" * full
        rem = cells - full
        if rem > 0.125 and full < width:
            s += BLOCKS[max(1, int(rem * 8))] if False else "▌" if rem > 0.5 else "▎"
        self.put(y, x, s, attr)

    # ── data helpers ────────────────────────────────────────────────────────
    def joints(self) -> dict:
        return self.robot.get("proprio").get("joints") or {}

    def desired_norm(self) -> dict:
        d = self.robot.get("control").get("desired") or {}
        return {k.lower().replace(" ", "_"): v for k, v in d.items()}

    def realized_twist(self) -> tuple[float, float, float] | None:
        """Body-frame velocity from the dense odom history (finite difference
        over the last ~0.3 s). None until enough samples exist."""
        with self.robot.lock:
            pts = list(self.robot.pose_hist)
        if len(pts) < 2:
            return None
        t1, x1, y1, yaw1 = pts[-1]
        p0 = None
        for p in reversed(pts):
            if t1 - p[0] >= 0.25:
                p0 = p
                break
        if p0 is None:
            p0 = pts[0]
        t0, x0, y0, yaw0 = p0
        dt = t1 - t0
        if dt < 0.05 or time.time() - t1 > 1.5:
            return None
        dx, dy = (x1 - x0) / dt, (y1 - y0) / dt
        vx = dx * math.cos(-yaw1) - dy * math.sin(-yaw1)
        vy = dx * math.sin(-yaw1) + dy * math.cos(-yaw1)
        wz = ((yaw1 - yaw0 + math.pi) % (2 * math.pi) - math.pi) / dt
        return (vx, vy, wz)

    def is_estop(self) -> bool:
        st = self.robot.get("status")
        if "estop" in st:
            return bool(st.get("estop"))
        return bool(self.robot.get("control").get("estop"))

    # ── sprite state from telemetry ─────────────────────────────────────────
    def sprite_frame(self, t: float, hd: bool = False) -> list[list[str]]:
        j = self.joints()
        tw = self.robot.get("control").get("baseTwist") or {}
        torque = self.robot.get("status").get("torque") or {}
        base = self.robot.get("proprio").get("base") or {}

        def q(name, default=None):
            e = j.get(name)
            return (e or {}).get("q", default) if isinstance(e, dict) else default

        # the lift is TWO prismatic joints on the wire (lift_lower + lift_upper)
        if "lift_lower" in j or "lift_upper" in j:
            lift_m = (q("lift_lower") or 0.0) + (q("lift_upper") or 0.0)
        else:
            lift_m = q("lift", 0.3175) or 0.3175
        return sprite.build_frame(
            t,
            yaw=base.get("yaw", 0.0),
            # nearest-baked-stance sync: raw (a1, a2) per side, classified
            # against the exact table the baker posed the MJCF with
            arm_l=(q("left_arm_1"), q("left_arm_2")),
            arm_r=(q("right_arm_1"), q("right_arm_2")),
            lift_m=lift_m,
            speed=abs(tw.get("vx", 0.0)) + abs(tw.get("vy", 0.0)),
            turning=tw.get("wz", 0.0),
            estop=self.is_estop(),
            limp=(torque.get("arms") is False),
            engaged=bool(self.robot.get("control").get("engaged")),
            hd=hd,
        )

    # ── panels ──────────────────────────────────────────────────────────────
    def draw_sprite_panel(self, y, x, h, w, t):
        estop = self.is_estop()
        label = "robot"
        iy, ix, ih, iw = self.panel(y, x, h, w, label,
                                    "E-STOP" if estop and int(t * 3) % 2 else "")
        # pick the HD bake when the panel is big enough (fullscreen), then
        # integer-UPSCALE into whatever room is left so the robot fills the
        # panel instead of floating tiny in it
        hd = (iw >= sprite.HD_W and 2 * (ih - 2) >= sprite.HD_H)
        rows = self.sprite_frame(t, hd=hd)
        gw = max(len(r) for r in rows)
        # UNIFORM upscale only — never stretch one axis (an asymmetric 2:1
        # scale squashed the robot at certain window sizes)
        s = max(1, min(3, min(iw // gw, (2 * (ih - 2)) // len(rows))))
        if s > 1:
            rows = [[px for px in r for _ in range(s)]
                    for r in rows for _ in range(s)]
            gw = max(len(r) for r in rows)
        # decimate as a last resort on tiny panels
        shrink = 1
        while shrink < 4 and ((len(rows) // shrink + 1) // 2 > ih - 1
                              or gw // shrink > iw):
            shrink += 1
        if shrink > 1:
            rows = [r[::shrink] for r in rows[::shrink]]
            gw = max(len(r) for r in rows)
        ph = min((len(rows) + 1) // 2, ih - 1)   # last inner row = status word
        ox = ix + max(0, (iw - gw) // 2)
        oy = iy + max(0, (ih - 1 - ph) // 2)
        pal = sprite.PALETTE_256 if self.has_256 else sprite.PALETTE_8
        for ry in range(ph):
            top = rows[2 * ry]
            bot = rows[2 * ry + 1] if 2 * ry + 1 < len(rows) else ["."] * gw
            line_y = oy + ry
            for cx in range(min(gw, len(top))):
                tk = top[cx]
                bk = bot[cx] if cx < len(bot) else "."
                if tk == "." and bk == ".":
                    continue
                if tk != "." and bk != ".":
                    ch, attr = "▀", self.color(pal[tk], pal[bk])
                elif tk != ".":
                    ch, attr = "▀", self.pal(tk)
                else:
                    ch, attr = "▄", self.pal(bk)
                self.put(line_y, ox + cx, ch, attr)
        # status word under the robot
        word, attr = "", 0
        if estop:
            word, attr = "!! E-STOP !!", self.theme("red") | curses.A_BOLD
        elif not self.robot.connected:
            word, attr = "OFFLINE", self.theme("dim")
        elif self.robot.get("control").get("engaged"):
            word, attr = "OPERATOR LINKED", self.theme("green")
        elif (self.robot.get("status").get("torque") or {}).get("arms") is False:
            word, attr = "LIMP (torque off)", self.theme("yellow")
        if word:
            self.put(iy + ih - 1, ix + max(0, (iw - len(word)) // 2), word, attr)

    def draw_video(self, y, x, h, w):
        iy, ix, ih, iw = self.panel(y, x, h, w, "video", ":8080")
        text, dim = self.theme("text"), self.theme("dim")
        if not self.cameras.ok:
            self.put(iy, ix, "camera box not answering", dim)
            return
        for i, c in enumerate(self.cameras.cams[:ih]):
            age = c.get("ageMs")
            if c.get("error"):
                dot, attr, note = "✗", self.theme("red"), str(c["error"])[:14]
            elif age is not None and age < 700:
                dot, attr = "●", self.theme("green")
                note = f"{c['fps']:4.0f} fps" if c.get("fps") is not None else "live"
            elif c.get("streaming"):
                dot, attr, note = "●", self.theme("yellow"), "stale"
            else:
                dot, attr, note = "◌", dim, "idle"
            self.put(iy + i, ix, dot, attr)
            self.put(iy + i, ix + 2, f"{c['name']:<12.12}", text)
            self.put(iy + i, ix + 15, f"{c.get('label', ''):<10.10}", dim)
            self.put(iy + i, ix + 26, note[: max(0, iw - 30)],
                     attr if dot == "●" else dim)
            if c.get("streaming"):
                self.put(iy + i, ix + iw - 3, "tx", self.theme("dcyan"))

    def draw_lidar(self, y, x, h, w, t):
        li = self.robot.get("lidar")
        iy, ix, ih, iw = self.panel(y, x, h, w, "lidar",
                                    "live" if li else "odom trail")
        # SMALL top-anchored plot; the odom trail may wander the whole panel
        rmax_px = max(2, min(4, ih // 2, iw // 4))
        cy, cx = iy + rmax_px + 1, ix + iw // 2
        green, dgreen, dim = self.theme("green"), self.theme("dgreen"), self.theme("dim")
        # range rings
        for frac in (0.5, 1.0):
            r = rmax_px * frac
            for a in range(0, 360, 15):
                ry = cy + int(round(math.sin(math.radians(a)) * r))
                rx = cx + int(round(math.cos(math.radians(a)) * r * 2))
                if iy <= ry < iy + ih and ix <= rx < ix + iw:
                    self.put(ry, rx, "·", dim)
        base = self.robot.get("proprio").get("base") or {}
        yaw = base.get("yaw", 0.0)
        # odom breadcrumbs, rotated into the CURRENT base frame (forward = up)
        with self.robot.lock:
            pts = list(self.robot.trail)
        bx, by = base.get("x", 0.0), base.get("y", 0.0)
        scale = rmax_px / max(1e-6, (li.get("rangeMax", 4.0) if li else 4.0))
        for (px, py) in pts[-200:]:
            dx0, dy0 = px - bx, py - by
            fwd = dx0 * math.cos(-yaw) - dy0 * math.sin(-yaw)
            lft = dx0 * math.sin(-yaw) + dy0 * math.cos(-yaw)
            ry = cy - int(round(fwd * scale))
            rx = cx - int(round(lft * scale * 2))
            if iy <= ry < iy + ih and ix <= rx < ix + iw:
                self.put(ry, rx, "·", self.theme("yellow"))
        # lidar points (base frame: 0 = forward, CCW)
        if li:
            ranges = li.get("ranges") or []
            a0, ai = li.get("angleMin", 0.0), li.get("angleInc", 0.0)
            rmax = li.get("rangeMax", 8.0)
            sweep = (t * 1.6) % (2 * math.pi)
            for i, r in enumerate(ranges):
                if r is None or r >= rmax - 1e-6:
                    continue
                a = a0 + i * ai
                fwd, lft = math.cos(a) * r, math.sin(a) * r
                ry = cy - int(round(fwd * scale))
                rx = cx - int(round(lft * scale * 2))
                if iy <= ry < iy + ih and ix <= rx < ix + iw:
                    da = abs((a - sweep + math.pi) % (2 * math.pi) - math.pi)
                    self.put(ry, rx, "•", green if da < 0.9 else dgreen)
            # the rotating sweep ray
            for rr in range(1, rmax_px + 1):
                ry = cy - int(round(math.cos(sweep) * rr))
                rx = cx - int(round(math.sin(sweep) * rr * 2))
                if iy <= ry < iy + ih and ix <= rx < ix + iw:
                    self.put(ry, rx, "·", dgreen)
        self.put(cy, cx, "▲", self.theme("cyan") | curses.A_BOLD)

    def draw_drive(self, y, x, h, w):
        iy, ix, ih, iw = self.panel(y, x, h, w, "drive")
        tw = self.robot.get("control").get("baseTwist") or {}
        base = self.robot.get("proprio").get("base") or {}
        served = self.realized_twist()
        rvx, rvy, rwz = served if served else (None, None, None)
        text, dim, cyan = self.theme("text"), self.theme("dim"), self.theme("cyan")
        rows = [("vx fwd", tw.get("vx"), rvx, "m/s"),
                ("vy left", tw.get("vy"), rvy, "m/s"),
                ("wz turn", tw.get("wz"), rwz, "rad/s")]
        self.put(iy, ix + 11, "asked", dim)
        self.put(iy, ix + 20, "served", dim)
        for i, (name, a, r, unit) in enumerate(rows):
            self.put(iy + 1 + i, ix, name, text)
            self.put(iy + 1 + i, ix + 9, f"{(a or 0.0):+6.2f}", cyan if a else dim)
            self.put(iy + 1 + i, ix + 18, f"{r:+6.2f}" if r is not None else "     —", text if r else dim)
            self.put(iy + 1 + i, ix + 26, unit, dim)
            mag = min(1.0, abs(a or 0.0) / 1.0)
            self.bar(iy + 1 + i, ix + 33, max(2, iw - 35), mag,
                     cyan if (a or 0) >= 0 else self.theme("orange"))
        ody = iy + 4
        self.put(ody, ix, "odom", dim)
        self.put(ody, ix + 9,
                 f"x {base.get('x', 0.0):+7.2f}  y {base.get('y', 0.0):+7.2f} m   "
                 f"yaw {math.degrees(base.get('yaw', 0.0)):+7.1f}°", text)
        mode = self.robot.get("control").get("mode") or self.robot.get("status").get("controlMode") or {}
        badge = f"{mode.get('method', '—')}·{mode.get('region', '—')}"
        eng = self.robot.get("control").get("engaged")
        self.put(ody + 1, ix, "mode", dim)
        self.put(ody + 1, ix + 9, badge, self.theme("magenta"))
        self.put(ody + 1, ix + 10 + len(badge),
                 "ENGAGED" if eng else "idle",
                 self.theme("green") | curses.A_BOLD if eng else dim)
        if self.pilot is not None and self.pilot.active():
            self.put(ody + 1, ix + 20 + len(badge), "⌨ KEYS DRIVING",
                     self.theme("yellow") | curses.A_BOLD)
        drives = base.get("drive") or []
        if drives:
            spins = "".join("◍" if abs(d) > 0.3 else "◌" for d in drives[:4])
            self.put(ody + 1, ix + iw - 12, f"wheels {spins}", dim)

    FINGERS = ("thumb", "index", "middle", "ring", "pinky")

    def _finger_flex(self, side: str) -> list[float | None]:
        """Mean |q| per finger (0..~1.6 rad) from whatever finger joints the
        wire streams (…_pip/_mcp/_dip/_cmc/_abd), None when absent."""
        j = self.joints()
        out: list[float | None] = []
        for f in self.FINGERS:
            vals = [abs(e.get("q") or 0.0) for k, e in j.items()
                    if isinstance(e, dict) and k.startswith(f"{side}_{f}_")]
            out.append(sum(vals) / len(vals) if vals else None)
        return out

    def draw_joints(self, y, x, h, w):
        iy, ix, ih, iw = self.panel(y, x, h, w, "joints", "deviation ±15°")
        j = self.joints()
        des = self.desired_norm()
        text, dim = self.theme("text"), self.theme("dim")
        half = iw // 2
        bot = iy + ih

        def qv(k):
            e = j.get(k)
            return (e or {}).get("q") if isinstance(e, dict) else None

        def track(row_y, x0, width, err_deg):
            """microduck-style deviation track: dotted rail, center tick,
            colored block at the deviation position (full rail = ±15°)."""
            halfw = max(3, (width - 1) // 2)
            self.put(row_y, x0, "·" * halfw + "│" + "·" * halfw, dim)
            if err_deg is None:
                return
            col = self.theme("green") if abs(err_deg) < 3 else (
                self.theme("orange") if abs(err_deg) < 9 else self.theme("red"))
            pos = max(-halfw, min(halfw, int(round(err_deg / 15.0 * halfw))))
            self.put(row_y, x0 + halfw + pos, "█", col)

        def jrow(row_y, x0, label, ios_id, width):
            """A torque-controlled joint: meas · motion tick · err · track."""
            e = j.get(ios_id)
            meas = (e or {}).get("q") if isinstance(e, dict) else None
            dq = (e or {}).get("dq") if isinstance(e, dict) else None
            cmd = des.get(ios_id)
            self.put(row_y, x0, label, text)
            if meas is None:
                self.put(row_y, x0 + 7, "     —", dim)
                return
            self.put(row_y, x0 + 6, f"{math.degrees(meas):+7.1f}", text)
            if dq is not None and abs(dq) > 0.04:
                self.put(row_y, x0 + 14, "▲" if dq > 0 else "▼", self.theme("dcyan"))
            err = None if cmd is None else math.degrees(cmd - meas)
            if err is not None:
                col = self.theme("green") if abs(err) < 3 else (
                    self.theme("orange") if abs(err) < 9 else self.theme("red"))
                self.put(row_y, x0 + 16, f"{err:+6.1f}", col)
            track(row_y, x0 + 24, max(7, width - 25), err)

        # ── arms (torque-controlled → deviation tracks) ──────────────────────
        self.put(iy, ix, "L arm   meas      err   deviation", dim)
        self.put(iy, ix + half, "R arm   meas      err   deviation", dim)
        avail = ih - 1
        n_arm = min(7, avail if avail < 9 else 7)
        for i in range(n_arm):
            jrow(iy + 1 + i, ix, f"a{i + 1}", f"left_arm_{i + 1}", half - 2)
            jrow(iy + 1 + i, ix + half, f"a{i + 1}", f"right_arm_{i + 1}", half - 2)
        if n_arm < 7:
            self.put(iy + n_arm, ix + half - 3, "…", dim)
            return
        ry = iy + 8
        rules_on = avail >= 16               # labelled separators need the room

        def rule(label):
            nonlocal ry
            if rules_on and ry < bot:
                self.put(ry, ix, "─" * iw, dim)
                self.put(ry, ix + 2, f" {label} ", self.theme("dcyan"))
                ry += 1

        # ── torso (torque-controlled too → its own full-width track row) ─────
        rule("torso")
        if ry < bot:
            jrow(ry, ix, "torso", "torso", min(iw, half + 10))
            ry += 1
        # ── lift (position servo → travel readout + bar) ─────────────────────
        rule("lift")
        if ry < bot:
            lift = qv("lift")
            if qv("lift_lower") is not None or qv("lift_upper") is not None:
                lift = (qv("lift_lower") or 0.0) + (qv("lift_upper") or 0.0)
            self.put(ry, ix, "lift", text)
            if lift is None:
                self.put(ry, ix + 7, "—", dim)
            else:
                self.put(ry, ix + 6, f"{lift * 100:6.1f} cm ", dim)
                self.bar(ry, ix + 17, max(6, min(30, half - 20)),
                         lift / 0.64, self.theme("dcyan"))
                self.put(ry, ix + 18 + max(6, min(30, half - 20)),
                         f"of 64 cm travel", dim)
            ry += 1
        # ── neck (position servos → measured pose) ───────────────────────────
        rule("neck")
        if ry < bot:
            npt, nyw, nrl = qv("neck_pitch"), qv("neck_yaw"), qv("neck_roll")
            self.put(ry, ix, "neck", text)
            if npt is None and nyw is None:
                self.put(ry, ix + 7, "—", dim)
            else:
                self.put(ry, ix + 6,
                         f"pitch {math.degrees(npt or 0):+6.1f}°     "
                         f"yaw {math.degrees(nyw or 0):+6.1f}°     "
                         f"roll {math.degrees(nrl or 0):+6.1f}°", dim)
            ry += 1
        # ── hands: one full-width line EACH (combined only when out of rows) ─
        rule("hands")
        tor = self.robot.get("status").get("torque") or {}

        def hand_row(row_y, x0, side, key, width):
            flex = self._finger_flex(key)
            tq = tor.get(f"{key}_hand")
            self.put(row_y, x0, f"hand{side}", text)
            dot = "●" if tq else ("○" if tq is False else "◌")
            self.put(row_y, x0 + 6, dot,
                     self.theme("green") if tq else
                     (self.theme("yellow") if tq is False else dim))
            cx = x0 + 9
            step, off = (5, 2) if width >= 55 else (3, 1)
            got = False
            for name, v in zip("TIMRP", flex):
                self.put(row_y, cx, name, dim)
                if v is None:
                    self.put(row_y, cx + off, "·", dim)
                else:
                    got = True
                    lvl = max(1, min(8, 1 + int(v / 1.6 * 7)))
                    self.put(row_y, cx + off, BLOCKS[lvl], self.theme("orange"))
                cx += step
            vals = [v for v in flex if v is not None]
            room = x0 + width - 2 - (cx + 2)
            if got and vals:
                g = sum(vals) / len(vals) / 1.6
                txt = f"grip {g:4.2f}" if room >= 10 else f"g{g:4.2f}"
                self.put(row_y, cx + 2, txt[:max(0, room)], dim)
            elif not got:
                self.put(row_y, cx + 2, "no stream"[:max(0, room)], dim)

        if bot - ry >= 2:
            hand_row(ry, ix, "L", "left", iw)
            hand_row(ry + 1, ix, "R", "right", iw)
        elif bot - ry == 1:                  # tight terminal: side by side
            hand_row(ry, ix, "L", "left", half - 2)
            hand_row(ry, ix + half, "R", "right", half - 2)
        if not j:
            self.put(iy + 3, ix + 2, "no proprio yet — waiting for the robot link…", dim)

    def draw_status(self, y, x, h, w):
        iy, ix, ih, iw = self.panel(y, x, h, w, "status")
        st = self.robot.get("status")
        text, dim = self.theme("text"), self.theme("dim")
        lat = st.get("latency") or {}
        tor = st.get("torque") or {}
        row = iy

        # host: who the server is and where the apps reach it (NOVA's host/net rows, folded)
        hello = self.robot.get("hello").get("robot") or {}
        self.put(row, ix, "host", dim)
        self.put(row, ix + 8,
                 f"{hello.get('name') or '—'} · {hello.get('kind') or '—'} · "
                 f"{hello.get('server') or '—'} v{hello.get('version') or '—'} · "
                 f"lan {LAN_IP} · ws :{self.args.teleop_port} · http :{self.args.http_port}"
                 [:max(0, iw - 9)], text)
        row += 1
        if row >= iy + ih:
            return

        def torq(v):
            if v is True:
                return ("ON", self.theme("green"))
            if v is False:
                return ("off", self.theme("yellow"))
            return ("—", dim)

        # torque (hand torque lives on the joints panel's hand rows). `tor` is the
        # GATEWAY's intent; a module the hardware panel shows OFFLINE cannot be
        # energised whatever was asked, so say `offline` (red), as NOVA's cockpit does.
        hw_state = {r[0]: r[3] for r in self.hardware_rows()[0]}
        _hw_of = {"arms": ("arm L", "arm R"), "neck": ("neck yaw", "neck pit+rol"),
                  "base": ("base",), "left_hand": ("hand L",), "right_hand": ("hand R",)}
        self.put(row, ix, "torque", dim)
        cx = ix + 8
        for name, key in (("arms", "arms"), ("neck", "neck"), ("base", "base"),
                          ("handL", "left_hand"), ("handR", "right_hand")):
            if key not in tor:
                continue
            states = [hw_state.get(lbl) for lbl in _hw_of.get(key, ())]
            if states and all(st == "OFFLINE" for st in states if st is not None) and any(states):
                val, attr = "offline", self.theme("red")
            else:
                val, attr = torq(tor.get(key))
            self.put(row, cx, name, text)
            self.put(row, cx + len(name) + 1, val, attr)
            cx += len(name) + len(val) + 3
        row += 1
        if row >= iy + ih:
            return
        # link + recording on one row
        cl = (st.get("driver") or {}).get("clients")
        rec = st.get("recording") or {}
        dc = st.get("dataCollection") or {}
        rec_on = bool(rec.get("active")) or bool(dc.get("active"))
        self.put(row, ix, "link", dim)
        self.put(row, ix + 8,
                 f"clients {cl if cl is not None else '—'} · "
                 f"cmd {lat.get('cmdStalenessMs', 0) or 0:.1f}ms · "
                 f"state {lat.get('stateAgeMs', 0) or 0:.1f}ms · ", text)
        self.put(row, ix + 8 + 44, "● REC" if rec_on else f"rec idle/{rec.get('total', 0)}",
                 self.theme("red") | curses.A_BOLD if rec_on else dim)
        row += 1
        if row >= iy + ih:
            return
        # policy/animation only when one is actually selected
        pol = st.get("policy") or {}
        ani = st.get("animation") or {}
        if pol.get("selected") or ani.get("selected"):
            self.put(row, ix, "policy", dim)
            self.put(row, ix + 8,
                     f"{pol.get('selected') or '—'}{' ▶' if pol.get('running') else ''}   "
                     f"anim {ani.get('selected') or '—'}", text)
            row += 1
            if row >= iy + ih:
                return
        # ports
        self.put(row, ix, "ports", dim)
        cx = ix + 8
        for label, port in self.prober.ports:
            up = self.prober.up.get(port)
            dot = "●" if up else ("○" if up is not None else "◌")
            attr = self.theme("green") if up else self.theme("red")
            s = f"{dot}{label}"
            if cx + len(s) + 4 >= ix + iw:
                row += 1
                cx = ix + 8
                if row >= iy + ih:
                    break
            self.put(row, cx, dot, attr)
            self.put(row, cx + 1, f"{label}:{port}", dim)
            cx += len(f"{label}:{port}") + 3
        row += 1
        # real-mode systems row from the hardware watchdog's [hw] line: parse
        # the name:✓/· liveness ticks and wrap them as colored tokens
        if self.hw_slot.get("line") and row < iy + ih:
            toks = re.findall(r"([A-Za-z0-9_+-]+):([✓·])", self.hw_slot["line"])
            self.put(row, ix, "hw", dim)
            self.put(row, ix + 3, f"[{self.hw_slot.get('t')}]", dim)
            cx = ix + 14
            for name, mark in toks:
                if cx + len(name) + 2 >= ix + iw:
                    row += 1
                    cx = ix + 14
                    if row >= iy + ih:
                        break
                self.put(row, cx, name, text)
                self.put(row, cx + len(name), mark,
                         self.theme("green") if mark == "✓" else dim)
                cx += len(name) + 3

    hw_compact = False      # set per frame by draw(): small window → motors only

    def hardware_rows(self):
        """(module rows, sensor rows) from the gateway feed — shared by layout + draw.
        Compact (short window): the sensors are left to the video panel, which already
        shows them, so the roster costs the joints panel as little as possible."""
        hw = self.robot.get("hardware")
        tor = self.robot.get("status").get("torque") or {}
        return (_module_rows(hw, self.joints(), tor),
                [] if self.hw_compact else _sensor_rows(hw))

    def draw_hardware(self, y, x, h, w):
        """EVERY MODULE, ALWAYS: how many of its motors are answering, which are silent,
        and how old their feedback is. Sized to the roster and never folds away — this is
        the one thing an operator most wants to know, and it used to be on screen only
        when thor's watchdog wrote a `[hw]` log line (NOVA's cockpit panel, ported)."""
        mod_rows, sen_rows = self.hardware_rows()
        rows = [(r[0], r[1], r[2], r[3], r[4], r[6]) for r in mod_rows] + sen_rows
        fitted = [r for r in mod_rows if r[2]]
        up = sum(1 for r in fitted if r[3] == "LIVE")
        motors = sum(r[1] for r in fitted), sum(r[2] for r in fitted)
        sen_on = sum(1 for r in sen_rows if r[3] == "LIVE")
        badge = (f"● {motors[0]}/{motors[1]} motors · {up}/{len(fitted)} modules"
                 + (f" · {sen_on}/{len(sen_rows)} sensors" if sen_rows else ""),
                 "green" if fitted and up == len(fitted) else ("yellow" if up else "red"))
        iy, ix, ih, iw = self.panel(y, x, h, w, "hardware", badges=[badge] if fitted else [])
        text, dim = self.theme("text"), self.theme("dim")
        if not rows:
            self.put(iy, ix, f"no module roster yet — waiting for the gateway "
                             f"(ws :{self.args.teleop_port}/teleop)…"[:iw], dim)
            return
        cell, lab, cols = _hw_grid(iw)
        # Short on rows? Show the modules that are NOT fully live first — those change
        # what you do next — and say how many healthy ones are folded away.
        n_faults = sum(1 for r in mod_rows if r[3] == "PARTIAL")
        hidden = 0
        if -(-len(rows) // cols) > ih - n_faults:
            bad = [r for r in rows if r[3] != "LIVE"]
            good = [r for r in rows if r[3] == "LIVE"]
            room = max(cols, (ih - n_faults) * cols - 1)
            hidden = max(0, len(rows) - room)
            rows = (bad + good)[:room]
        for i, (label, live, n, state, colour, rate) in enumerate(rows):
            r, c = divmod(i, cols)
            ry, cx = iy + r, ix + c * cell
            if ry >= iy + ih:
                break
            glyph = {"LIVE": "●", "PARTIAL": "◐", "—": "·"}.get(state, "○")
            col = self.theme(colour)
            self.put(ry, cx, glyph, col)
            self.put(ry, cx + 2, label[:lab], text if state == "LIVE" else dim)
            self.put(ry, cx + 3 + lab, f"{live}/{n}".rjust(5), col)
            self.put(ry, cx + 10 + lab, str(rate)[:cell - lab - 11].rjust(min(7, cell - lab - 11)),
                     dim if rate == "—" else (col if state != "LIVE" else text))
        if hidden:
            r, c = divmod(len(rows), cols)
            self.put(iy + min(r, ih - 1), ix + c * cell, f"… +{hidden} live", dim)
        # NAME THE SILENT ONES: a count says stop; a name says which connector to check.
        fy = iy + -(-len(rows) // cols) + (1 if hidden and len(rows) % cols == 0 else 0)
        faults = [(r[0], r[1], r[2], r[4], r[5]) for r in mod_rows if r[3] == "PARTIAL"]
        if self.hw_compact and faults and fy < iy + ih:
            # one folded line on a short window
            self.put(fy, ix, "⚠", self.theme("yellow"))
            self.put(fy, ix + 2, " · ".join(f"{l} {lv}/{n} {d}" for l, lv, n, _c, d in faults)[:iw - 2], text)
            return
        for label, live, n, colour, detail in faults:
            if fy >= iy + ih:
                break
            self.put(fy, ix, "⚠", self.theme(colour))
            head = f"{label} {live}/{n}"
            self.put(fy, ix + 2, head, self.theme(colour))
            self.put(fy, ix + 4 + len(head), detail[:max(0, iw - 6 - len(head))], text)
            fy += 1

    def draw_log(self, y, x, h, w):
        iy, ix, ih, iw = self.panel(y, x, h, w, "log", "l hides")
        with self.log_lock:
            lines = list(self.log_ring)[-ih:]
        for i, (src, line) in enumerate(lines):
            low = line.lower()
            if "error" in low or "✗" in line or "traceback" in low:
                attr = self.theme("red")
            elif "warn" in low or "⚠" in line:
                attr = self.theme("yellow")
            elif "✓" in line or " ok" in low:
                attr = self.theme("green")
            else:
                attr = self.theme("dim")
            self.put(iy + i, ix, f"{src:>7} ", self.theme("dcyan"))
            self.put(iy + i, ix + 8, line[: iw - 9], attr)
        if not lines:
            self.put(iy, ix, "no log lines yet", self.theme("dim"))

    def draw_chart(self, y, x, h, w):
        iy, ix, ih, iw = self.panel(
            y, x, h, w, "link",
            f"state {self.robot.state_hz():4.0f} Hz · full bar 50 · newest right")
        with self.robot.lock:
            hist = list(self.robot.hz_hist)[-iw:]
        cyan = self.theme("cyan")
        for i, (_, hz) in enumerate(hist):
            frac = max(0.0, min(1.0, hz / 50.0))
            cells = frac * ih
            col = cyan if hz >= 25 else self.theme("red")
            cx = ix + iw - len(hist) + i
            full = int(cells)
            for r in range(full):
                self.put(iy + ih - 1 - r, cx, "█", col)
            rem = cells - full
            if rem > 1 / 8 and full < ih:
                self.put(iy + ih - 1 - full, cx, BLOCKS[int(rem * 8)], col)

    # ── boot screen ─────────────────────────────────────────────────────────
    def draw_boot(self, t):
        h, w = self.scr.getmaxyx()
        cyan, dim, text = self.theme("cyan"), self.theme("dim"), self.theme("text")
        logo_w = len(LOGO[0]) * 2
        ox = max(0, (w - logo_w) // 2)
        oy = max(1, h // 2 - 8)
        for ry in range((len(LOGO) + 1) // 2):
            top = LOGO[2 * ry]
            bot = LOGO[2 * ry + 1] if 2 * ry + 1 < len(LOGO) else "." * len(top)
            for cx, (tk, bk) in enumerate(zip(top, bot)):
                ch = "▀" if tk == "X" and bk != "X" else (
                    "▄" if tk != "X" and bk == "X" else ("█" if tk == "X" else None))
                if ch:
                    self.put(oy + ry, ox + cx * 2, ch * 2, cyan | curses.A_BOLD)
        sub = "T E L E O P   C O C K P I T"
        self.put(oy + 4, max(0, (w - len(sub)) // 2), sub, text)
        mode = "REAL ROBOT" if self.args.mode == "real" else ("DEMO" if self.args.demo else "SIMULATION")
        self.put(oy + 5, max(0, (w - len(mode)) // 2), mode, self.theme("magenta"))
        spin = "▖▘▝▗"[int(t * 6) % 4]
        wait = f"{spin} waiting for the robot link (:{self.args.telemetry_port} telemetry)"
        if int(t * 2) % 2:
            self.put(oy + 7, max(0, (w - len(wait)) // 2), wait, self.theme("yellow"))
        if self.robot.err:
            self.put(oy + 8, max(0, (w - len(self.robot.err)) // 2), self.robot.err, dim)
        # port checklist
        cy = oy + 10
        for i, (label, port) in enumerate(self.prober.ports):
            up = self.prober.up.get(port)
            mark = "✓" if up else ("…" if up is None else "·")
            attr = self.theme("green") if up else dim
            self.put(cy + i, max(0, w // 2 - 14), f"{mark} {label:<10} :{port}", attr)
        # last log lines at the bottom so a crashing bring-up is visible here
        with self.log_lock:
            lines = list(self.log_ring)[-5:]
        for i, (src, line) in enumerate(lines):
            self.put(h - 7 + i, 2, f"{src:>7}  {line}"[: w - 4], dim)
        foot = "© 2026 MABEL ROBOTICS — INSERT OPERATOR"
        self.put(h - 1, max(0, (w - len(foot)) // 2), foot,
                 dim if int(t) % 2 else text)

    def draw_help(self):
        h, w = self.scr.getmaxyx()
        lines = [
            "MABEL COCKPIT — KEYS",
            "",
            "  ARROWS    drive the base (↑↓ fwd/back · ←→ strafe)",
            "  a / d     rotate left / right",
            "  w / s     lift up / down",
            "  (hold a key to keep moving — release and the robot stops)",
            "",
            "  q         quit (stops the whole stack)",
            "  ENTER     re-anchor the teleop clutch" + ("" if self.args.mode == "sim" else " (sim only)"),
            "  r         reset the sim" + ("" if self.args.mode == "sim" else " (sim only)"),
            "  l         show/hide the log panel",
            "  p         pause/resume redraw",
            "  h or ?    this help",
            "",
            "drive keys ride the same wire contract as every client",
            "(safe_base_step + speed caps still gate all motion)",
        ]
        bw = max(len(s) for s in lines) + 6
        bh = len(lines) + 2
        y0, x0 = max(0, (h - bh) // 2), max(0, (w - bw) // 2)
        self.panel(y0, x0, bh, bw, "help")
        for i, s in enumerate(lines):
            self.put(y0 + 1 + i, x0 + 3, s,
                     self.theme("cyan") | curses.A_BOLD if i == 0 else self.theme("text"))

    # ── frame ───────────────────────────────────────────────────────────────
    def draw(self, t):
        self.scr.erase()
        h, w = self.scr.getmaxyx()
        self.shadow = [[" "] * w for _ in range(h)]
        if h < 24 or w < 96:
            msg = f"TERMINAL TOO SMALL ({w}×{h}) — need ≥ 96×24 (fullscreen it!)"
            self.put(h // 2, max(0, (w - len(msg)) // 2), msg, self.theme("yellow"))
            return
        if not self.robot.connected and not self.robot.get("proprio"):
            self.draw_boot(t)
            return

        # header strip
        name = ((self.robot.get("hello").get("robot") or {}).get("name")) or "mabel"
        server = ((self.robot.get("hello").get("robot") or {}).get("server")) or ""
        world = "real" if self.args.mode == "real" or (server and "sim" not in server) else "sim"
        up = time.time() - self.robot.t0
        hdr_l = f" {name} · {world} "
        hdr_r = " arrows drive · a/d turn · w/s lift · q quit · h help "
        estop = self.is_estop()
        self.put(0, 0, "═" * w, self.theme("red" if estop else "dim"))
        self.put(0, 2, hdr_l, self.theme("cyan") | curses.A_BOLD | curses.A_REVERSE)
        mid = f" UPTIME {int(up // 60):02d}:{int(up % 60):02d} · LINK {self.robot.state_hz():3.0f} Hz "
        mid_x = len(hdr_l) + 6
        if mid_x + len(mid) < w - len(hdr_r) - 4:
            self.put(0, mid_x, mid, self.theme("dim"))
        self.put(0, max(0, w - len(hdr_r) - 2), hdr_r, self.theme("dim"))
        if self.toast[1] > time.time():
            ts = f"▸ {self.toast[0]} ◂"
            self.put(0, max(0, (w - len(ts)) // 2), ts,
                     self.theme("yellow") | curses.A_BOLD)

        right_w = max(34, min(96, w // 3))
        left_w = w - right_w
        chart_h = 6 if h >= 38 else 5
        body_h = h - 1 - chart_h
        # ── right column: robot takes the ROOM (the hero panel — HD bake +
        #    upscaling when it fits); video fixed; lidar small at the bottom ──
        ncams = len(self.cameras.cams) if self.cameras.ok else 3
        video_h = min(8, max(3, ncams)) + 2
        lidar_h = 12 if body_h >= 34 else max(8, body_h - video_h - 14)
        robot_h = body_h - video_h - lidar_h
        # shrink the lidar when its slack is exactly what unlocks the HD sprite
        # (panel rows needed: ceil(HD_H/2) sprite + word row + borders + gate slack)
        hd_need = (sprite.HD_H + 1) // 2 + 4
        if robot_h < hd_need and body_h - video_h - 8 >= hd_need:
            lidar_h = 8
            robot_h = body_h - video_h - lidar_h
        self.draw_sprite_panel(1, left_w, robot_h, right_w, t)
        self.draw_video(1 + robot_h, left_w, video_h, right_w)
        self.draw_lidar(1 + robot_h + video_h, left_w, lidar_h, right_w, t)
        # ── left column: drive 8 · joints ≤14 · status compact · log = REST ──
        # (the log absorbs the leftover height, so no panel balloons on a tall
        #  terminal — the status stays compact by design)
        drive_h = 8
        st = self.robot.get("status")
        has_policy = bool((st.get("policy") or {}).get("selected")
                          or (st.get("animation") or {}).get("selected"))
        status_h = 7 + (1 if has_policy else 0) \
            + (2 if self.hw_slot.get("line") else 0)
        # hardware: every module, always (sized to the roster, 3–4 cells per row, plus one
        # line per degraded module naming its silent motors) — but never at the cost of
        # pushing the panels below it off the screen: on a minimum window it shows what
        # fits, faults first (see draw_hardware).
        self.hw_compact = body_h < 40                    # 34-row terminal: motors only
        _mr, _sr = self.hardware_rows()
        n_cells = len(_mr) + len(_sr)
        n_faults = sum(1 for r in _mr if r[3] == "PARTIAL")
        if self.hw_compact:
            n_faults = 1 if n_faults else 0              # faults fold onto one line
        _, _, hw_cols = _hw_grid(left_w - 2)
        want_hw = bool(self.robot.get("hardware")) or self.args.mode == "real"
        hw_h = (2 + max(1, -(-n_cells // hw_cols)) + n_faults) if want_hw else 0
        hw_h = max(0, min(hw_h, body_h - drive_h - status_h - 7))
        if self.log_forced and self.show_log:
            # the operator asked for the log (`l`): on a small window the roster yields
            # before the joints do — it is sized to be glanced at, the log to be read
            short = 4 - (body_h - drive_h - status_h - 7 - hw_h)
            if short > 0:
                hw_h = max(0, hw_h - short)
        if hw_h in (1, 2):
            hw_h = 0
        joints_h = min(19, max(7, body_h - drive_h - status_h - hw_h))
        log_h = body_h - drive_h - joints_h - status_h - hw_h if self.show_log else 0
        if self.log_forced and self.show_log and log_h < 4 and joints_h > 7:
            steal = min(4 - max(0, log_h), joints_h - 7)
            joints_h -= steal
            log_h = body_h - drive_h - joints_h - status_h - hw_h
        if 0 < log_h < 3:
            log_h = 0
            status_h = body_h - drive_h - joints_h - hw_h   # tiny remainder → status pad
        self.log_visible = log_h > 0
        yy = 1
        self.draw_drive(yy, 0, drive_h, left_w)
        yy += drive_h
        self.draw_joints(yy, 0, joints_h, left_w)
        yy += joints_h
        if hw_h:
            self.draw_hardware(yy, 0, hw_h, left_w)
            yy += hw_h
        self.draw_status(yy, 0, status_h, left_w)
        yy += status_h
        if log_h:
            self.draw_log(yy, 0, log_h, left_w)
        self.draw_chart(h - chart_h, 0, chart_h, w)
        if self.help_on:
            self.draw_help()
        if estop and int(t * 3) % 2 == 0:
            for cx in range(0, w - 8, 12):
                self.put(0, cx, " E-STOP ", self.theme("red") | curses.A_REVERSE | curses.A_BOLD)

    # ── input ───────────────────────────────────────────────────────────────
    def fifo_write(self, data: str, what: str):
        path = self.args.fifo
        if not path:
            self.toast = (f"{what}: no bridge stdin here", time.time() + 2.0)
            return
        try:
            fd = os.open(path, os.O_WRONLY | os.O_NONBLOCK)
            try:
                os.write(fd, data.encode())
            finally:
                os.close(fd)
            self.toast = (what, time.time() + 2.0)
        except OSError as e:
            hint = "bridge not listening" if e.errno in (errno.ENXIO, errno.ENOENT) else str(e)
            self.toast = (f"{what} failed: {hint}", time.time() + 3.0)

    # keyboard teleop gains (sticks are normalised [-1,1]; the controller's
    # safe_base_step + speed caps bound what actually moves)
    DRIVE_GAIN, YAW_GAIN, LIFT_GAIN = 0.7, 0.7, 0.6

    def _drive(self, **axes):
        if self.pilot is None:
            self.toast = ("no teleop link (demo)", time.time() + 1.5)
            return
        self.pilot.nudge(**axes)

    def handle_key(self, ch: int):
        if ch in (ord("q"), ord("Q"), 3):
            self.quit = True
        # stick signs MEASURED against the live adapter (2026-09-01 probe):
        # ly=+ → vx<0 and lx=+ → vy<0, so both translation axes are inverted;
        # rx=+ → wz>0 (CCW) and ry=+ → lift up are already correct.
        elif ch == curses.KEY_UP:
            self._drive(ly=-self.DRIVE_GAIN)     # forward (vx > 0)
        elif ch == curses.KEY_DOWN:
            self._drive(ly=+self.DRIVE_GAIN)
        elif ch == curses.KEY_LEFT:
            self._drive(lx=-self.DRIVE_GAIN)     # strafe left (vy > 0)
        elif ch == curses.KEY_RIGHT:
            self._drive(lx=+self.DRIVE_GAIN)
        elif ch == ord("a"):
            self._drive(rx=+self.YAW_GAIN)       # rotate left (CCW)
        elif ch == ord("d"):
            self._drive(rx=-self.YAW_GAIN)       # rotate right
        elif ch == ord("w"):
            self._drive(ry=+self.LIFT_GAIN)      # lift up
        elif ch == ord("s"):
            self._drive(ry=-self.LIFT_GAIN)      # lift down
        elif ch in (10, 13, curses.KEY_ENTER):
            self.fifo_write("\n", "CLUTCH RE-ANCHORED")
        elif ch in (ord("r"), ord("R")) and self.args.mode == "sim":
            self.fifo_write("r\n", "SIM RESET")
        elif ch == ord("l"):
            if self.log_visible:
                self.show_log = self.log_forced = False
            else:
                self.show_log = self.log_forced = True
        elif ch == ord("p"):
            self.paused = not self.paused
        elif ch in (ord("h"), ord("?")):
            self.help_on = not self.help_on
        elif ch == curses.KEY_RESIZE:
            pass

    # ── main loop ───────────────────────────────────────────────────────────
    def run(self, scr) -> None:
        self.scr = scr
        curses.curs_set(0)
        scr.nodelay(True)
        scr.keypad(True)
        if curses.has_colors():
            curses.start_color()
            try:
                curses.use_default_colors()
            except curses.error:
                pass
            self.has_256 = curses.COLORS >= 256
        n = 0
        period = 1.0 / self.FPS
        next_t = time.monotonic()
        while not self.quit:
            t = time.time() - self.robot.t0
            if not self.paused or n == 0:
                t0 = time.monotonic()
                self.draw(t)
                scr.refresh()
                self.frame_ms.append((time.monotonic() - t0) * 1000.0)
            n += 1
            if self.args.frames and n >= self.args.frames:
                break
            while True:
                ch = scr.getch()
                if ch == -1:
                    break
                self.handle_key(ch)
            next_t += period
            delay = next_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = time.monotonic()
        if self.args.dump:
            self.dump_screen(self.args.dump)

    def dump_screen(self, path: str) -> None:
        lines = ["".join(row).rstrip() for row in self.shadow]
        avg = sum(self.frame_ms) / max(1, len(self.frame_ms))
        mx = max(self.frame_ms) if self.frame_ms else 0.0
        h, w = self.scr.getmaxyx()
        with open(path, "w") as f:
            f.write("\n".join(lines))
            f.write(f"\n--- cockpit metrics: frames={len(self.frame_ms)} "
                    f"render avg={avg:.2f}ms max={mx:.2f}ms "
                    f"size={w}x{h} colors256={self.has_256} ---\n")


# ═════════════════════════════════ wiring ══════════════════════════════════
SIM_PORTS = [("teleop", 9090), ("video", 8080), ("physics", 8788),
             ("telemetry", 9600), ("slam", 9605), ("audio", 9610), ("studio", 8097)]
REAL_PORTS = [("teleop", 9090), ("video", 8080), ("telemetry", 9600),
              ("map", 9092), ("slam", 9605), ("audio", 9610), ("studio", 8094)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--mode", choices=("sim", "real"), default="sim")
    ap.add_argument("--telemetry-host", default="127.0.0.1")
    ap.add_argument("--telemetry-port", type=int, default=9600)
    ap.add_argument("--http-port", type=int, default=8080,
                    help="camera box port for the /cameras.json video roster")
    ap.add_argument("--teleop-port", type=int, default=9090,
                    help="teleop WS for the keyboard drive/lift commands")
    ap.add_argument("--fifo", default="", help="bridge stdin FIFO (ENTER/R forwarding)")
    ap.add_argument("--log", action="append", default=[],
                    metavar="NAME=PATH",
                    help="log file to tail (repeatable). NOTE: the launchers "
                         "pass these via $MABEL_COCKPIT_LOGS (comma-separated "
                         "name=path) instead — several stack log paths contain "
                         "process names (e.g. …simulation_bridge.log) that the "
                         "launchers' pkill -f sweeps match, and argv is what "
                         "pkill -f sees. Env is invisible to it.")
    ap.add_argument("--demo", action="store_true", help="synthetic robot, no server")
    ap.add_argument("--frames", type=int, default=0, help="render N frames then exit")
    ap.add_argument("--dump", default="", help="write the final screen text here")
    args = ap.parse_args()

    robot = Robot()
    log_ring: deque = deque(maxlen=500)
    log_lock = threading.Lock()
    hw_slot: dict = {}
    threads: list[threading.Thread] = []

    prober = PortProber(REAL_PORTS if args.mode == "real" else SIM_PORTS)
    threads.append(prober)
    cameras = CameraPoller(args.http_port, demo=args.demo)
    threads.append(cameras)
    pilot = None
    if not args.demo:
        pilot = TeleopCommander(args.telemetry_host, args.teleop_port)
        threads.append(pilot)
    if args.demo:
        threads.append(DemoFeed(robot))
        for port in list(prober.up):
            prober.up[port] = True
    else:
        threads.append(TelemetryThread(robot, args.telemetry_host, args.telemetry_port))
        threads.append(GatewayFeed(robot, args.telemetry_host, args.teleop_port))
    log_specs = list(args.log)
    env_logs = os.environ.get("MABEL_COCKPIT_LOGS", "")
    log_specs += [s for s in env_logs.split(",") if s.strip()]
    for spec in log_specs:
        name, _, path = spec.strip().partition("=")
        if path:
            threads.append(LogTail(name[:7], path, log_ring, log_lock, hw_slot))
    for th in threads:
        th.start()

    cockpit = Cockpit(args, robot, prober, cameras, pilot,
                      log_ring, log_lock, hw_slot)
    os.environ.setdefault("ESCDELAY", "25")
    try:
        curses.wrapper(cockpit.run)
    except KeyboardInterrupt:
        pass
    finally:
        for th in threads:
            th.stop_flag = True
    return 0


if __name__ == "__main__":
    import locale
    locale.setlocale(locale.LC_ALL, "")
    sys.exit(main())
