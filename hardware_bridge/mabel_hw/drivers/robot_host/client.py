# VENDORED from the upstream arm stack
# Do not edit here beyond vendoring fixes — upstream improvements land in the
# upstream repo first, then re-vendor. The wire protocol (1-byte tag + nanopb
# robot.proto) must match the arm/neck Teensy firmware.
"""RobotClient — generic UDP client for the Teensy robot DAQ/control driver.

Self-describing: on connect() the client sends Hello and the Teensy replies with
RobotInfo — the robot name, stream rates, and a per-actuator descriptor list (name,
model, bus, ids, limits, default gains). The client builds its entire surface from
that, so ONE client adapts to any robot defined by /robot.json on the Teensy SD card.
There are no hard-coded actuator counts, names, or "gripper" special cases.

  * Action = an N-actuator MIT command — pos, vel, kp, kd, t_ff (rad / rad/s / N·m),
    where N == num_actuators (learned from RobotInfo). Actuator i is info.actuators[i]
    / names[i]; address by index, or resolve a name with index_of(name).
  * State  = per-actuator position / velocity / torque (control-relevant fields).
  * Health = per-actuator DM error code, MOSFET/rotor temperatures, enabled/live.

Data-exchange policy (must match the firmware, src/host_link.h):
  * The Teensy STREAMS a State snapshot of all actuators at a fixed rate (default from
    config, e.g. 1000 Hz). getState() returns the latest received packet — no round trip.
  * Commands are sent fire-and-forget; the Teensy caches them and streams them to CAN
    at the control rate. setCommand() does not block.
  * Health is request/reply: getHealthState() sends a HealthRequest and waits.

Safety / liveness (see host_link.h):
  * Output is gated on host liveness. This client keeps the link alive with periodic
    Hello keep-alives, so **connecting ENABLES the motors** (held by a safe damping
    command until you command). Keep the robot supported on the bench.
  * While connected but NOT commanding, the firmware applies a safe damping hold
    (kp=0, kd=per-actuator default). Call close() (or stop sending) to fully disable:
    no packets -> the firmware disableAll()s and the robot goes **LIMP** within
    ~0.25 s (fail-safe — the DM comm-loss watchdog is off, so this is explicit).

Wire units are motor-side SI: position rad, velocity rad/s, torque N·m.
"""

from __future__ import annotations

import socket
import threading
import time
from dataclasses import dataclass

import numpy as np

from . import robot_pb2 as pb

# Datagram type tags (first payload byte). Must match host_link.h / robot.proto.
TAG_COMMAND = 0x01
TAG_HEALTH_REQUEST = 0x02
TAG_HELLO = 0x03
TAG_STATE = 0x10
TAG_HEALTH = 0x11
TAG_ROBOT_INFO = 0x12


@dataclass(frozen=True)
class ActuatorInfo:
    """Static description of one actuator, from the firmware's RobotInfo."""

    name: str
    model: str
    bus: int
    can_id: int
    mst_id: int
    p_max: float
    v_max: float
    t_max: float
    kp_max: float
    kd_max: float
    default_kp: float
    default_kd: float


@dataclass(frozen=True)
class RobotInfo:
    """The robot's self-description (reply to Hello). Defines num_actuators + names."""

    name: str
    stream_default_hz: int
    stream_max_hz: int
    actuators: list  # list[ActuatorInfo], index == actuator index everywhere else

    @property
    def num_actuators(self) -> int:
        return len(self.actuators)

    @property
    def names(self) -> list:
        return [a.name for a in self.actuators]


@dataclass
class RobotState:
    """Control-relevant state of all actuators (one streamed packet)."""

    seq: int                  # echoes the last Command.seq the Teensy applied
    t_us: int                 # Teensy micros() timestamp at send (not host-synced)
    live_mask: int            # bit i set => actuator i returned feedback recently
    position: np.ndarray      # (N,) rad
    velocity: np.ndarray      # (N,) rad/s
    torque: np.ndarray        # (N,) N·m
    recv_time: float          # host monotonic time when this packet was received

    def is_live(self, i: int) -> bool:
        return bool(self.live_mask & (1 << i))

    def age(self) -> float:
        """Seconds since this packet was received on the host (host-side staleness)."""
        return time.monotonic() - self.recv_time


@dataclass
class RobotHealth:
    """Full health of all actuators (one Health reply)."""

    seq: int
    t_us: int
    err: np.ndarray            # (N,) DM error nibble: 1 = enabled OK, 8..E = fault
    t_mos: np.ndarray          # (N,) deg C, driver MOSFET
    t_rotor: np.ndarray        # (N,) deg C, motor coil
    enabled: list              # [N] bool (err == 1)
    live: list                 # [N] bool (feedback fresh)
    stale_ms: np.ndarray       # (N,) age of newest feedback frame
    recv_time: float


class RobotClient:
    """High-level client for one robot (N actuators, learned at connect) over UDP."""

    def __init__(
        self,
        teensy_ip: str = "192.168.124.11",
        port: int = 5800,
        stream_hz: int = 1000,
        timeout: float = 0.2,
        keepalive_interval: float = 0.1,
        on_state=None,
        on_info=None,
        on_health=None,
    ):
        self.teensy_ip = teensy_ip
        self.port = port
        self.stream_hz = stream_hz
        self.timeout = timeout
        self.keepalive_interval = keepalive_interval

        # Optional push callbacks, fired from the RX thread the instant a packet is parsed —
        # so consumers that need minimum latency (e.g. a Zenoh bridge) are driven by hardware
        # packet ARRIVAL, not by polling getState(). Each is called with the freshly built
        # dataclass: on_state(RobotState), on_info(RobotInfo), on_health(RobotHealth). They run
        # AFTER the internal cache is updated and OUTSIDE the lock. **Keep them fast and
        # non-blocking** — a slow callback stalls the RX loop (delays later packets + RTT). For
        # heavy work, hand off to a queue. Settable after construction (plain attributes).
        self.on_state = on_state
        self.on_info = on_info
        self.on_health = on_health

        self._sock: socket.socket | None = None
        self._addr = (teensy_ip, port)
        self._seq = 0

        self._lock = threading.Lock()
        self._info: RobotInfo | None = None
        self._info_event = threading.Event()
        self._state: RobotState | None = None
        self._health: RobotHealth | None = None
        self._health_event = threading.Event()
        self._state_rx = 0  # total State packets received (for stream-rate measurement)
        self._health_rx = 0

        self._last_send = 0.0
        self._cmd_seq_send: dict[int, float] = {}  # seq -> send monotonic time (bounded)
        self._last_rtt: float | None = None

        self._running = False
        self._rx_thread: threading.Thread | None = None
        self._ka_thread: threading.Thread | None = None

    # ---- lifecycle ----------------------------------------------------------
    def connect(self, info_timeout: float = 2.0):
        """Bind the socket, register with the Teensy (Hello), start RX + keep-alive, and
        block until RobotInfo arrives so the actuator layout is known.

        Raises TimeoutError if no RobotInfo is received within info_timeout (firmware
        too old, wrong IP/port, or no link).
        """
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind(("", 0))  # ephemeral local port; Teensy learns it from packets
        self._sock.settimeout(0.2)
        self._running = True
        self._rx_thread = threading.Thread(target=self._rx_loop, daemon=True)
        self._rx_thread.start()
        self._ka_thread = threading.Thread(target=self._keepalive_loop, daemon=True)
        self._ka_thread.start()
        self._send_hello()
        if not self._info_event.wait(info_timeout):
            self.close()
            raise TimeoutError(
                f"no RobotInfo from {self.teensy_ip}:{self.port} within {info_timeout}s "
                "(firmware up? correct IP/port? link?)"
            )
        return self

    def close(self):
        """Stop threads and close the socket. The robot disables (goes limp) within ~0.25 s."""
        self._running = False
        if self._rx_thread:
            self._rx_thread.join(timeout=0.5)
        if self._ka_thread:
            self._ka_thread.join(timeout=0.5)
        if self._sock:
            self._sock.close()
            self._sock = None

    def __enter__(self):
        return self.connect()

    def __exit__(self, *exc):
        self.close()

    # ---- layout (from RobotInfo) --------------------------------------------
    @property
    def info(self) -> RobotInfo | None:
        """The robot's self-description, or None before connect()."""
        with self._lock:
            return self._info

    @property
    def num_actuators(self) -> int:
        info = self.info
        return info.num_actuators if info else 0

    @property
    def names(self) -> list:
        info = self.info
        return info.names if info else []

    def index_of(self, name: str) -> int:
        """Index of the actuator named `name`. Raises KeyError if absent."""
        names = self.names
        try:
            return names.index(name)
        except ValueError:
            raise KeyError(f"no actuator named {name!r}; have {names}") from None

    # ---- commands -----------------------------------------------------------
    def setCommand(self, pos, vel, kp, kd, t_ff):
        """Send an MIT setpoint for all actuators (fire-and-forget, applied at the
        firmware control rate). T = kp*(pos - q) + kd*(vel - dq) + t_ff.

        Each argument is a length-num_actuators sequence / numpy array, indexed like
        info.actuators / names. Units: pos rad, vel rad/s, t_ff N·m. kd MUST be > 0
        whenever commanding a position, or the actuator oscillates.
        """
        n = self.num_actuators
        if n == 0:
            raise RuntimeError("not connected; call connect() first")
        arrs = [np.asarray(a, dtype=np.float32).reshape(-1) for a in (pos, vel, kp, kd, t_ff)]
        for name, a in zip(("pos", "vel", "kp", "kd", "t_ff"), arrs):
            if a.size != n:
                raise ValueError(f"{name} must have {n} elements, got {a.size}")
            if not np.all(np.isfinite(a)):
                raise ValueError(f"{name} contains non-finite values")
        p, v, kpv, kdv, tff = arrs

        self._seq = (self._seq + 1) & 0xFFFFFFFF
        msg = pb.Command(
            seq=self._seq,
            pos=p.tolist(), vel=v.tolist(),
            kp=kpv.tolist(), kd=kdv.tolist(), t_ff=tff.tolist(),
        )
        self._send(TAG_COMMAND, msg.SerializeToString())
        now = time.monotonic()
        self._cmd_seq_send[self._seq] = now
        if len(self._cmd_seq_send) > 256:
            for k in sorted(self._cmd_seq_send)[:128]:
                self._cmd_seq_send.pop(k, None)
        return self._seq

    # ---- state --------------------------------------------------------------
    def getState(self) -> RobotState | None:
        """Return the latest streamed State (no round trip), or None if none yet."""
        with self._lock:
            return self._state

    def getHealthState(self, timeout: float | None = None) -> RobotHealth | None:
        """Request a Health snapshot and wait for the reply (falls back to cached)."""
        timeout = self.timeout if timeout is None else timeout
        self._health_event.clear()
        self._seq = (self._seq + 1) & 0xFFFFFFFF
        self._send(TAG_HEALTH_REQUEST, pb.HealthRequest(seq=self._seq).SerializeToString())
        self._health_event.wait(timeout)
        with self._lock:
            return self._health

    # ---- diagnostics --------------------------------------------------------
    def last_rtt(self) -> float | None:
        """Round-trip time (s) of the most recent command echoed back in a State."""
        return self._last_rtt

    def connected(self) -> bool:
        """True if a State packet arrived within the last ~0.25 s."""
        st = self.getState()
        return st is not None and st.age() < 0.25

    def state_rx_count(self) -> int:
        """Total State packets received since connect (for stream-rate measurement)."""
        return self._state_rx

    # ---- internals ----------------------------------------------------------
    def _send(self, tag: int, body: bytes):
        if self._sock is None:
            raise RuntimeError("not connected; call connect() first")
        self._sock.sendto(bytes([tag]) + body, self._addr)
        self._last_send = time.monotonic()

    def _send_hello(self):
        self._send(TAG_HELLO, pb.Hello(stream_hz=self.stream_hz).SerializeToString())

    def _keepalive_loop(self):
        # Keep the firmware's host link (=> enabled output + telemetry stream) alive
        # when idle. Hello refreshes the connection watchdog but NOT the command
        # watchdog, so the robot stays in the safe damping hold — telemetry without
        # commanded motion.
        while self._running:
            time.sleep(self.keepalive_interval)
            if time.monotonic() - self._last_send >= self.keepalive_interval:
                try:
                    self._send_hello()
                except OSError:
                    break

    def _rx_loop(self):
        while self._running:
            try:
                data, _ = self._sock.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:
                break
            if not data:
                continue
            tag, body = data[0], data[1:]
            if tag == TAG_STATE:
                self._on_state(body)
            elif tag == TAG_HEALTH:
                self._on_health(body)
            elif tag == TAG_ROBOT_INFO:
                self._on_robot_info(body)

    def _on_robot_info(self, body: bytes):
        msg = pb.RobotInfo()
        try:
            msg.ParseFromString(body)
        except Exception:
            return
        info = RobotInfo(
            name=msg.name,
            stream_default_hz=msg.stream_default_hz,
            stream_max_hz=msg.stream_max_hz,
            actuators=[
                ActuatorInfo(
                    name=a.name, model=a.model, bus=a.bus, can_id=a.can_id, mst_id=a.mst_id,
                    p_max=a.p_max, v_max=a.v_max, t_max=a.t_max,
                    kp_max=a.kp_max, kd_max=a.kd_max,
                    default_kp=a.default_kp, default_kd=a.default_kd,
                )
                for a in msg.actuators
            ],
        )
        with self._lock:
            self._info = info
        self._info_event.set()
        self._fire(self.on_info, info)

    def _on_state(self, body: bytes):
        msg = pb.State()
        try:
            msg.ParseFromString(body)
        except Exception:
            return
        a = list(msg.actuators)
        st = RobotState(
            seq=msg.seq,
            t_us=msg.t_us,
            live_mask=msg.live_mask,
            position=np.array([m.position for m in a], dtype=np.float32),
            velocity=np.array([m.velocity for m in a], dtype=np.float32),
            torque=np.array([m.torque for m in a], dtype=np.float32),
            recv_time=time.monotonic(),
        )
        with self._lock:
            self._state = st
            self._state_rx += 1
        # RTT: if this State echoes a command we timed, record it.
        t_send = self._cmd_seq_send.pop(msg.seq, None)
        if t_send is not None:
            self._last_rtt = st.recv_time - t_send
        # Push to a subscriber (e.g. the bridge) driven by arrival, not polling.
        self._fire(self.on_state, st)

    def _on_health(self, body: bytes):
        msg = pb.Health()
        try:
            msg.ParseFromString(body)
        except Exception:
            return
        a = list(msg.actuators)
        h = RobotHealth(
            seq=msg.seq,
            t_us=msg.t_us,
            err=np.array([m.err for m in a], dtype=np.int32),
            t_mos=np.array([m.t_mos for m in a], dtype=np.int32),
            t_rotor=np.array([m.t_rotor for m in a], dtype=np.int32),
            enabled=[m.enabled for m in a],
            live=[m.live for m in a],
            stale_ms=np.array([m.stale_ms for m in a], dtype=np.int64),
            recv_time=time.monotonic(),
        )
        with self._lock:
            self._health = h
            self._health_rx += 1
        self._health_event.set()
        self._fire(self.on_health, h)

    @staticmethod
    def _fire(cb, arg):
        """Invoke a push callback, isolating its errors so the RX loop never dies."""
        if cb is None:
            return
        try:
            cb(arg)
        except Exception:
            pass
