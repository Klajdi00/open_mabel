"""openarm_link.py — the UDP wire to the arms Teensy (openarm-dual firmware).

WIRE ONLY. This module speaks protobuf-over-UDP to the board at
192.168.124.11:5800 and knows nothing about joints, calibration, or control —
:mod:`mabel_hw.drivers.openarm` is the :class:`~mabel_hw.drivers.seam.JointGroupDriver`
view on top. Structured exactly like :mod:`mabel_hw.drivers.teensy_link`: ONE link
per (ip, port), shared by 1..N group views that write disjoint slices of a merged
command vector, streamed by one thread.

Schema (recovered from the firmware's self-describing manifest, 2026-07-11):

    tag 0x01 COMMAND  host->teensy   { f1 varint seq, f2..f6 packed f32 arrays
                                       pos/vel/kp/kd/t_ff — MANIFEST ORDER }
    tag 0x02 HEALTHRQ host->teensy   { f1 varint seq }
    tag 0x03 HELLO    host->teensy   { f1 varint stream_hz }        (keepalive)
    tag 0x10 STATE    teensy->host   { f2 varint t_us, f4 repeated { f1 f32 pos_rad,
                                       f2 f32 vel, f3 f32 tau } — MANIFEST ORDER,
                                       an all-zero joint encodes as empty }
    tag 0x11 HEALTH   teensy->host   { f1 varint seq, f2 varint t_us, f3 repeated
                                       { f2 varint t_mos_C, f3 varint t_rotor_C } }
    tag 0x12 MANIFEST teensy->host   { f1 str name ("openarm-dual"), f4 repeated
                                       JointSpec { f1 str name (l_j1..l_j7, l_gripper,
                                       r_*), f2 str motor, f3 varint bus, f4 varint id,
                                       f6 f32 p_max, f7 f32 v_max, f8 f32 t_max } }

The COMMAND schema is authoritative — mirrored from the upstream `robot.proto`
`Command` / `robot_host/client.py::setCommand`, and validated byte-for-byte by
``tests/test_openarm_link.py``. The firmware applies
``T = kp*(pos - q) + kd*(vel - dq) + t_ff`` per actuator.

THE ONE LIMIT THIS LAYER ENFORCES: ``t_ff`` is clamped to the manifest's per-joint
``t_max`` at send time (:meth:`OpenArmLink.send`) — the single choke point every
command passes through. Nothing else is guarded here; bounding ``kp*error`` (which
the FIRMWARE computes, and which this host therefore cannot clamp) is the
commander's job — see ``controller/mabel/arm_control``.

The firmware has its own command watchdog: a stale command is replaced by a
damping hold, so the stream thread must re-send at ``stream_hz`` to hold a pose.
Hand-rolled protobuf codec (no google.protobuf dependency).
"""
from __future__ import annotations

import socket
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = ["OpenArmLink", "JointSpec", "JointFeedback", "OpenArmState"]

TAG_COMMAND = 0x01
TAG_HEALTH_REQUEST = 0x02
TAG_HELLO = 0x03
TAG_STATE = 0x10
TAG_HEALTH = 0x11
TAG_MANIFEST = 0x12


# ── minimal protobuf wire codec ──────────────────────────────────────────────
def _varint(b: bytes, i: int) -> Tuple[int, int]:
    v = s = 0
    while True:
        x = b[i]
        i += 1
        v |= (x & 0x7F) << s
        if not x & 0x80:
            return v, i
        s += 7


def _enc_varint(v: int) -> bytes:
    out = bytearray()
    while True:
        b = v & 0x7F
        v >>= 7
        out.append(b | (0x80 if v else 0))
        if not v:
            return bytes(out)


def _enc_packed_floats(field_no: int, vals: Sequence[float]) -> bytes:
    """Protobuf PACKED repeated float (proto3 default): key = (field<<3)|2, a
    length varint, then N little-endian f32. Matches nanopb on the firmware AND
    google.protobuf on the reference host (upstream robot_host)."""
    body = b"".join(struct.pack("<f", float(v)) for v in vals)
    return _enc_varint((field_no << 3) | 2) + _enc_varint(len(body)) + body


def _fields(b: bytes):
    """Yield (field_no, wire_type, value) — value is int (varint), float (f32/f64)
    or bytes (len-delimited). Raises on malformed input (caller drops the packet)."""
    i = 0
    while i < len(b):
        key, i = _varint(b, i)
        f, wt = key >> 3, key & 7
        if wt == 0:
            v, i = _varint(b, i)
            yield f, wt, v
        elif wt == 5:
            (v,) = struct.unpack_from("<f", b, i)
            i += 4
            yield f, wt, v
        elif wt == 1:
            (v,) = struct.unpack_from("<d", b, i)
            i += 8
            yield f, wt, v
        elif wt == 2:
            ln, i = _varint(b, i)
            yield f, wt, b[i:i + ln]
            i += ln
        else:
            raise ValueError(f"unsupported wire type {wt}")


# ── decoded telemetry ────────────────────────────────────────────────────────
@dataclass
class JointSpec:
    """One actuator as the firmware advertises it (manifest order)."""
    name: str                 # firmware name: l_j1..l_j7, l_gripper, r_*
    motor: str = ""           # DM8009 / DM4340 / DM4310
    bus: int = 0              # right = 1, left = 2 (openarm-dual convention)
    mid: int = 0
    p_max: float = 12.5
    v_max: float = 30.0
    t_max: float = 10.0       # N·m — the rated torque this link clamps t_ff to


@dataclass
class JointFeedback:
    pos: float = 0.0          # rad, RAW motor frame (calibration is 2 layers up)
    vel: float = 0.0          # rad/s
    tau: float = 0.0          # N·m


@dataclass
class OpenArmState:
    t_us: int
    joints: List[JointFeedback]        # MANIFEST order
    recv_time: float = field(default_factory=time.monotonic)

    def age(self) -> float:
        return time.monotonic() - self.recv_time


class OpenArmLink:
    """One UDP link to one arms Teensy; shared by 1..N OpenArmGroup views.

    Command model (mirrors TeensyLink): each group writes its own slice of a
    merged full-manifest vector; one thread streams that vector at ``stream_hz``.
    Streaming starts only once some slice has been commanded — an idle link stays
    silent and the firmware holds the motors in its own damping hold.
    """

    _registry: Dict[Tuple[str, int], "OpenArmLink"] = {}
    _registry_lock = threading.Lock()

    @classmethod
    def get(cls, teensy_ip: str = "192.168.124.11", port: int = 5800,
            stream_hz: int = 100, *, logger=None) -> "OpenArmLink":
        """The process-wide link for (ip, port) — created on first request, so
        left_arm and right_arm share one socket, one manifest, one stream."""
        key = (teensy_ip, int(port))
        with cls._registry_lock:
            link = cls._registry.get(key)
            if link is None:
                link = cls._registry[key] = cls(teensy_ip, port, stream_hz,
                                                logger=logger)
            return link

    def __init__(self, teensy_ip: str = "192.168.124.11", port: int = 5800,
                 stream_hz: int = 100, *, keepalive_interval: float = 0.1,
                 health_interval: float = 2.0, logger=None):
        self.addr = (teensy_ip, int(port))
        self.teensy_ip, self.port = teensy_ip, int(port)
        self.stream_hz = max(int(stream_hz), 1)
        self.keepalive_interval = keepalive_interval
        self.health_interval = health_interval
        self._log = logger or (lambda *a, **k: None)

        self._sock: Optional[socket.socket] = None
        self._lock = threading.Lock()
        self._specs: List[JointSpec] = []           # manifest order
        self._state: Optional[OpenArmState] = None
        self._temps: List[tuple] = []              # [(t_mos, t_rotor)] manifest order
        self._seq = 0
        self._running = False
        self._threads: List[threading.Thread] = []
        self._groups: list = []
        self._was_up = False

        # merged command vector (sized when the manifest arrives)
        self._pos: List[float] = []
        self._vel: List[float] = []
        self._kp: List[float] = []
        self._kd: List[float] = []
        self._tff: List[float] = []
        self._active = False                       # any slice EVER commanded

    # ── lifecycle ────────────────────────────────────────────────────────────
    def connect(self) -> "OpenArmLink":
        """Open the socket and start rx/keepalive/stream. Does NOT energize the
        motors: nothing is streamed until a group commands a slice."""
        with self._lock:
            if self._running:
                return self
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("", 0))
        sock.settimeout(0.2)
        with self._lock:
            self._sock = sock
            self._running = True
        self._threads = [
            threading.Thread(target=self._rx_loop, name="openarm-rx", daemon=True),
            threading.Thread(target=self._keepalive_loop, name="openarm-ka",
                             daemon=True),
            threading.Thread(target=self._stream_loop, name="openarm-tx",
                             daemon=True),
        ]
        for t in self._threads:
            t.start()
        self._send_hello()
        return self

    def close(self) -> None:
        with self._lock:
            self._running = False
            sock, self._sock = self._sock, None
            self._active = False
        for t in self._threads:
            t.join(timeout=0.5)
        self._threads = []
        if sock is not None:
            sock.close()                            # firmware damps on watchdog

    def connected(self) -> bool:
        st = self.state()
        return st is not None and st.age() < 0.5

    # ── group views ──────────────────────────────────────────────────────────
    def group(self, joint_names: Sequence[str], *, kp: float = 30.0,
              kd: float = 1.0, wire_t_max=None):
        """A JointGroupDriver over the named subset of this link's actuators.
        Created UNRESOLVED when the manifest has not arrived yet — the group
        binds itself the moment the board answers (hot-join, no restart).
        wire_t_max ({joint: register T_MAX}) arms the torque-enable guard
        against a stale SD manifest limit (see OpenArmGroup)."""
        from .openarm import OpenArmGroup
        g = OpenArmGroup(self, tuple(joint_names), kp=kp, kd=kd,
                         wire_t_max=wire_t_max)
        with self._lock:
            self._groups.append(g)
        g._resolve()
        return g

    # ── shared telemetry ─────────────────────────────────────────────────────
    def specs(self) -> List[JointSpec]:
        with self._lock:
            return list(self._specs)

    def state(self) -> Optional[OpenArmState]:
        with self._lock:
            return self._state

    def temps(self) -> List[tuple]:
        with self._lock:
            return list(self._temps)

    def health_full(self) -> List[dict]:
        """Per-actuator ActuatorHealth dicts (manifest order): err nibble,
        t_mos/t_rotor, enabled, live, stale_ms. Empty before the first Health
        reply."""
        with self._lock:
            return [dict(h) for h in getattr(self, "_health", [])]

    # ── merged command path ──────────────────────────────────────────────────
    def write_slice(self, indices: Sequence[int], pos: Sequence[float],
                    vel: Sequence[float], kp: Sequence[float],
                    kd: Sequence[float], tff: Sequence[float]) -> None:
        with self._lock:
            if not self._pos:
                return                              # manifest not up yet
            for j, i in enumerate(indices):
                if i >= len(self._pos):
                    continue
                self._pos[i] = float(pos[j])
                self._vel[i] = float(vel[j])
                self._kp[i] = float(kp[j])
                self._kd[i] = float(kd[j])
                self._tff[i] = float(tff[j])
            self._active = True

    def damp_slice(self, indices: Sequence[int], kd: float = 1.0) -> None:
        """Revert a slice to a damping hold (kp=0) — the 'torque off' state.
        We keep streaming it, so an operator's torque-off takes effect NOW
        instead of riding out the firmware's command-watchdog tail."""
        with self._lock:
            if not self._pos:
                return
            for i in indices:
                if i >= len(self._pos):
                    continue
                self._kp[i] = 0.0
                self._kd[i] = float(kd)
                self._tff[i] = 0.0
                self._vel[i] = 0.0

    def send(self) -> None:
        """Ship the merged vector once (fire-and-forget). No-op until some slice
        has been commanded.

        THE t_max CLAMP LIVES HERE — the one place every command converges, so no
        caller can route around it."""
        with self._lock:
            if self._sock is None or not self._pos or not self._active:
                return
            pos, vel = list(self._pos), list(self._vel)
            kp, kd = list(self._kp), list(self._kd)
            tff = [max(-s.t_max, min(s.t_max, t))
                   for s, t in zip(self._specs, self._tff)]
        self.send_command(pos, vel, kp, kd, tff)

    def send_command(self, pos, vel, kp, kd, t_ff) -> int:
        """Encode + send ONE MIT command (tag 0x01). Arrays are manifest-indexed
        and must be the same length. ``pos`` is the RAW motor angle in rad.
        ⚠ kd MUST be > 0 whenever a position is commanded or the motor
        oscillates. Returns the seq sent."""
        n = len(pos)
        if not (len(vel) == len(kp) == len(kd) == len(t_ff) == n):
            raise ValueError("send_command: all arrays must be the same length")
        with self._lock:
            self._seq = (self._seq + 1) & 0xFFFFFFFF
            seq = self._seq
        self._send(TAG_COMMAND, self.encode_command(seq, pos, vel, kp, kd, t_ff))
        return seq

    @staticmethod
    def encode_command(seq, pos, vel, kp, kd, t_ff) -> bytes:
        """The exact Command datagram BODY (no tag) — also used for offline
        byte-for-byte validation against the reference protobuf."""
        return (b"\x08" + _enc_varint(seq & 0xFFFFFFFF)          # f1 seq
                + _enc_packed_floats(2, pos) + _enc_packed_floats(3, vel)
                + _enc_packed_floats(4, kp) + _enc_packed_floats(5, kd)
                + _enc_packed_floats(6, t_ff))

    # ── internals ────────────────────────────────────────────────────────────
    def _send(self, tag: int, body: bytes) -> None:
        sock = self._sock
        if sock is not None:
            try:
                sock.sendto(bytes([tag]) + body, self.addr)
            except OSError:
                pass

    def _send_hello(self) -> None:
        self._send(TAG_HELLO, b"\x08" + _enc_varint(self.stream_hz))

    def _send_health_request(self) -> None:
        with self._lock:
            self._seq = (self._seq + 1) & 0x7FFFFFFF
            seq = self._seq
        self._send(TAG_HEALTH_REQUEST, b"\x08" + _enc_varint(seq))

    def _stream_loop(self) -> None:
        period = 1.0 / self.stream_hz
        while self._running:
            t0 = time.monotonic()
            try:
                self.send()
            except Exception as e:                  # never let the loop die
                self._log(f"[openarm {self.teensy_ip}] send failed: {e}")
            time.sleep(max(0.0, period - (time.monotonic() - t0)))

    def _keepalive_loop(self) -> None:
        last_health = 0.0
        while self._running:
            time.sleep(self.keepalive_interval)
            try:
                self._send_hello()
                if time.monotonic() - last_health >= self.health_interval:
                    last_health = time.monotonic()
                    self._send_health_request()
            except OSError:
                break

    def _rx_loop(self) -> None:
        while self._running:
            sock = self._sock
            if sock is None:
                break
            try:
                data, _ = sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            if not data:
                continue
            tag, body = data[0], data[1:]
            try:
                if tag == TAG_STATE:
                    self._on_state(body)
                elif tag == TAG_MANIFEST:
                    self._on_manifest(body)
                elif tag == TAG_HEALTH:
                    self._on_health(body)
            except Exception:
                pass                                # malformed datagram → drop

    def _on_manifest(self, body: bytes) -> None:
        specs = []
        for f, wt, v in _fields(body):
            if f == 4 and wt == 2:
                js = JointSpec(name="")
                for sf, _swt, sv in _fields(v):
                    if sf == 1:
                        js.name = sv.decode()
                    elif sf == 2:
                        js.motor = sv.decode()
                    elif sf == 3:
                        js.bus = sv
                    elif sf == 4:
                        js.mid = sv
                    elif sf == 6:
                        js.p_max = sv
                    elif sf == 7:
                        js.v_max = sv
                    elif sf == 8:
                        js.t_max = sv
                if js.name:
                    specs.append(js)
        if not specs:
            return
        n = len(specs)
        names = [s.name for s in specs]
        # The firmware answers EVERY hello keepalive with a manifest (~10 Hz), so
        # this must be idempotent: only an actual layout CHANGE is a link reset.
        # Resetting per manifest would limp every group 10x/second and silently
        # drop every command (_push_locked no-ops while torque is off).
        with self._lock:
            changed = [s.name for s in self._specs] != names
            self._specs = specs
            if not changed:
                return
            self._pos = [0.0] * n
            self._vel = [0.0] * n
            self._kp = [0.0] * n                    # kp=0 => damping hold
            self._kd = [1.0] * n
            self._tff = [0.0] * n
            self._active = False                    # limp until re-commanded
            groups = list(self._groups)
            was_up, self._was_up = self._was_up, True
        for g in groups:                            # (re)bind name -> index
            g._on_link_reset()
        self._log(f"[openarm {self.teensy_ip}] "
                  + ("layout CHANGED — groups reset to limp until re-commanded"
                     if was_up else
                     f"up: {n} actuators ({', '.join(names)})"))

    def _on_state(self, body: bytes) -> None:
        t_us, joints = 0, []
        for f, wt, v in _fields(body):
            if f == 2 and wt == 0:
                t_us = v
            elif f == 4 and wt == 2:
                jf = JointFeedback()
                for sf, _swt, sv in _fields(v):
                    if sf == 1:
                        jf.pos = sv
                    elif sf == 2:
                        jf.vel = sv
                    elif sf == 3:
                        jf.tau = sv
                joints.append(jf)                   # empty submessage == all-zero
        st = OpenArmState(t_us=t_us, joints=joints)
        with self._lock:
            prev, self._state = self._state, st
            # SAFETY RESET on a telemetry GAP: the board stopped streaming and came
            # back (reboot / replug / cable). It restarts limp and adopts whatever we
            # stream, so a stale latched target would be a snap. The manifest can't
            # tell us this (it answers every keepalive), so the gap is the signal.
            gap = prev is not None and (st.recv_time - prev.recv_time) > 1.0
            groups = list(self._groups) if gap else []
            if gap:
                # Damp the WHOLE merged vector, not just `_active`. Clearing only
                # the flag is not enough: the moment ANY group commands again it
                # re-arms `_active` and send() ships every OTHER group's stale
                # slice too — re-enabling the RIGHT arm would drive the LEFT one
                # (kp=30) to a pre-reboot target it has since fallen away from,
                # while its API still reads torque-off. Zero the position terms
                # for every joint here; each group re-arms its own slice when the
                # operator re-enables it.
                self._kp = [0.0] * len(self._kp)
                self._tff = [0.0] * len(self._tff)
                self._vel = [0.0] * len(self._vel)
                self._active = False                # limp until re-commanded
        for g in groups:
            g._on_link_reset()
        if groups:
            self._log(f"[openarm {self.teensy_ip}] REJOINED after a telemetry gap "
                      "— groups reset to limp until re-commanded")

    def _on_health(self, body: bytes) -> None:
        # Full ActuatorHealth (2026-07-29): the proto has always carried the DM
        # error nibble + enabled + live + stale_ms, but only the temperatures
        # were parsed — so a faulted shoulder showed as an unexplained RED row
        # ("J1/J2 go red very often") with the actual fault code (0x8..0xE:
        # overpressure/undervolt/overcurrent/MOS-temp/coil-temp/comm-loss/
        # overload) thrown away on the floor.
        temps, full = [], []
        for f, wt, v in _fields(body):
            if f == 3 and wt == 2:
                h = {"err": None, "t_mos": None, "t_rotor": None,
                     "enabled": None, "live": None, "stale_ms": None}
                for sf, _swt, sv in _fields(v):
                    if sf == 1:
                        h["err"] = sv
                    elif sf == 2:
                        h["t_mos"] = sv
                    elif sf == 3:
                        h["t_rotor"] = sv
                    elif sf == 4:
                        h["enabled"] = bool(sv)
                    elif sf == 5:
                        h["live"] = bool(sv)
                    elif sf == 6:
                        h["stale_ms"] = sv
                temps.append((h["t_mos"], h["t_rotor"]))
                full.append(h)
        if temps:
            with self._lock:
                self._temps = temps
                self._health = full
