"""SwerveClient — UDP client for the Teensy 4.1 MAXSwerve driver.

Data-exchange policy (must match the firmware, src/host_link.h):
  * The Teensy STREAMS a State snapshot of all 3 modules at a fixed rate (default
    200 Hz). getState() returns the latest received packet — no round trip.
  * Commands are sent fire-and-forget and applied on the Teensy immediately on
    receipt. setCommand() does not block.
  * Health is request/reply: getHealthState() sends a HealthRequest and waits for
    the Health reply (low rate).

Safety / liveness (see host_link.h):
  * Heartbeat on the Teensy — which enables the motor output AND the SPARKs' status
    broadcasting — is gated on receiving host packets. This client keeps the link
    alive with periodic Hello keep-alives while connected, so telemetry streams even
    when you are not commanding. **Connecting therefore enables the controllers**
    (output stage live, but drive is forced to 0 and steer holds until you command).
    Keep wheels off the ground on the bench.
  * While connected but NOT commanding, the Teensy forces drive velocity to 0 and
    holds steer. Stop sending packets entirely (call close()) to fully disable:
    no packets -> no heartbeat -> controllers disable within ~150 ms (fail-safe).

Wire units are firmware-native: drive = controller velocity units (m/s via the
configured conversion factor), steer = rotations [0,1). Helpers below convert
radians<->rotations if your controller prefers radians.
"""

from __future__ import annotations

import math
import socket
import threading
import time
from dataclasses import dataclass

import numpy as np

from . import maxswerve_pb2 as pb

# Datagram type tags (first payload byte). Must match host_link.h / maxswerve.proto.
TAG_COMMAND = 0x01
TAG_HEALTH_REQUEST = 0x02
TAG_HELLO = 0x03
TAG_STATE = 0x10
TAG_HEALTH = 0x11

NUM_MODULES = 3


@dataclass
class SwerveState:
    """Control-relevant state of all 3 modules (one streamed packet)."""

    seq: int                      # echoes the last Command.seq the Teensy applied
    t_us: int                     # Teensy micros() timestamp at send (not host-synced)
    live_mask: int                # bit i set => module i drive&steer both live
    drive_velocity: np.ndarray    # (3,) m/s
    drive_position: np.ndarray    # (3,) rotations
    steer_angle: np.ndarray       # (3,) rotations [0,1)
    drive_torque_est: np.ndarray  # (3,) N·m (approx)
    recv_time: float              # host monotonic time when this packet was received

    def is_live(self, i: int) -> bool:
        return bool(self.live_mask & (1 << i))

    @property
    def steer_angle_rad(self) -> np.ndarray:
        return self.steer_angle * (2.0 * math.pi)

    def age(self) -> float:
        """Seconds since this packet was received on the host (host-side staleness)."""
        return time.monotonic() - self.recv_time


@dataclass
class SwerveHealth:
    """Full health of all 3 modules (one Health reply)."""

    seq: int
    t_us: int
    drive_voltage: np.ndarray      # (3,) V
    drive_current: np.ndarray      # (3,) A
    drive_temperature: np.ndarray  # (3,) deg C
    drive_duty: np.ndarray         # (3,)
    steer_voltage: np.ndarray
    steer_current: np.ndarray
    steer_temperature: np.ndarray
    steer_duty: np.ndarray
    drive_live: list               # [3] bool
    steer_live: list               # [3] bool
    bus_down: list                 # [3] bool
    drive_stale_ms: np.ndarray     # (3,)
    steer_stale_ms: np.ndarray     # (3,)
    recv_time: float


class SwerveClient:
    """High-level client for the 3-module MAXSwerve rig over UDP."""

    def __init__(
        self,
        teensy_ip: str = "192.168.124.10",
        port: int = 5800,
        stream_hz: int = 200,
        timeout: float = 0.2,
        keepalive_interval: float = 0.1,
    ):
        self.teensy_ip = teensy_ip
        self.port = port
        self.stream_hz = stream_hz
        self.timeout = timeout
        self.keepalive_interval = keepalive_interval

        self._sock: socket.socket | None = None
        self._addr = (teensy_ip, port)
        self._seq = 0

        self._lock = threading.Lock()
        self._state: SwerveState | None = None
        self._health: SwerveHealth | None = None
        self._health_event = threading.Event()
        self._state_rx = 0  # total State packets received (for stream-rate measurement)
        self._health_rx = 0

        # Serializes the whole "bump seq -> serialize -> sendto" critical section so a
        # command's handshake seq and its datagram leave the host in the SAME order, even
        # when several threads send (ROS cmd_vel + pad teleop + a health poller + the
        # keepalive). Reentrant so the public senders can hold it across _send(). Without
        # it, two senders can interleave their non-atomic seq increment and put commands
        # on the wire out of order / with a duplicated seq, and the firmware (which echoes
        # seq and ignores any command not newer than the last applied) DROPS the loser ->
        # that module holds its last setpoint = an intermittent single-wheel stall.
        self._send_lock = threading.RLock()
        self._last_send = 0.0
        # RTT bookkeeping is shared with the RX thread (_on_state pops here); its own lock.
        self._rtt_lock = threading.Lock()
        self._cmd_seq_send: dict[int, float] = {}  # seq -> send monotonic time (bounded)
        self._last_rtt: float | None = None

        self._running = False
        self._rx_thread: threading.Thread | None = None
        self._ka_thread: threading.Thread | None = None

    # ---- lifecycle ----------------------------------------------------------
    def connect(self):
        """Bind the socket, register with the Teensy (Hello), and start RX + keep-alive."""
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind(("", 0))  # ephemeral local port; Teensy learns it from packets
        self._sock.settimeout(0.2)
        self._running = True
        self._rx_thread = threading.Thread(target=self._rx_loop, daemon=True)
        self._rx_thread.start()
        self._ka_thread = threading.Thread(target=self._keepalive_loop, daemon=True)
        self._ka_thread.start()
        self._send_hello()
        return self

    def close(self):
        """Stop threads and close the socket. The Teensy disables within ~150 ms."""
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

    # ---- commands -----------------------------------------------------------
    def setCommand(self, drive, steer):
        """Send a setpoint for all 3 modules (fire-and-forget, applied immediately).

        drive, steer: length-3 sequences / numpy arrays. drive = wheel-speed setpoint
        (controller native units, m/s via conversion factor); steer = heading in
        rotations [0,1). Use rad2rot() if you work in radians.
        """
        drive = np.asarray(drive, dtype=np.float32).reshape(-1)
        steer = np.asarray(steer, dtype=np.float32).reshape(-1)
        if drive.size != NUM_MODULES or steer.size != NUM_MODULES:
            raise ValueError(f"drive and steer must each have {NUM_MODULES} elements")
        if not (np.all(np.isfinite(drive)) and np.all(np.isfinite(steer))):
            raise ValueError("drive/steer contain non-finite values")

        # Atomic seq -> wire: hold the send lock across the increment AND the sendto so
        # concurrent senders can never reorder commands or reuse a handshake seq.
        with self._send_lock:
            self._seq = (self._seq + 1) & 0xFFFFFFFF
            seq = self._seq
            msg = pb.Command(seq=seq, drive=drive.tolist(), steer=steer.tolist())
            self._send(TAG_COMMAND, msg.SerializeToString())
        # Track send time for RTT (bounded history) under its own lock — the RX thread
        # pops from this same dict in _on_state.
        now = time.monotonic()
        with self._rtt_lock:
            self._cmd_seq_send[seq] = now
            if len(self._cmd_seq_send) > 256:
                for k in sorted(self._cmd_seq_send)[:128]:
                    self._cmd_seq_send.pop(k, None)
        return seq

    # ---- state --------------------------------------------------------------
    def getState(self) -> SwerveState | None:
        """Return the latest streamed State (no round trip), or None if none yet."""
        with self._lock:
            return self._state

    def getHealthState(self, timeout: float | None = None) -> SwerveHealth | None:
        """Request a Health snapshot and wait for the reply (falls back to cached)."""
        timeout = self.timeout if timeout is None else timeout
        self._health_event.clear()
        with self._send_lock:
            self._seq = (self._seq + 1) & 0xFFFFFFFF
            seq = self._seq
            self._send(TAG_HEALTH_REQUEST, pb.HealthRequest(seq=seq).SerializeToString())
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

    # ---- helpers ------------------------------------------------------------
    @staticmethod
    def rad2rot(rad):
        return np.asarray(rad, dtype=np.float32) / (2.0 * math.pi)

    @staticmethod
    def rot2rad(rot):
        return np.asarray(rot, dtype=np.float32) * (2.0 * math.pi)

    # ---- internals ----------------------------------------------------------
    def _send(self, tag: int, body: bytes):
        if self._sock is None:
            raise RuntimeError("not connected; call connect() first")
        # One datagram leaves at a time; keeps a keepalive Hello from interleaving inside
        # a command's seq->wire critical section and keeps _last_send monotonic. Reentrant,
        # so setCommand/getHealthState may already hold it.
        with self._send_lock:
            self._sock.sendto(bytes([tag]) + body, self._addr)
            self._last_send = time.monotonic()

    def _send_hello(self):
        self._send(TAG_HELLO, pb.Hello(stream_hz=self.stream_hz).SerializeToString())

    def _keepalive_loop(self):
        # Keep the Teensy's heartbeat (=> telemetry stream) alive when idle. Sending
        # Hello refreshes the connection watchdog but NOT the command watchdog, so the
        # drive stays forced to 0 — telemetry without motion.
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

    def _on_state(self, body: bytes):
        msg = pb.State()
        try:
            msg.ParseFromString(body)
        except Exception:
            return
        mods = list(msg.modules)
        st = SwerveState(
            seq=msg.seq,
            t_us=msg.t_us,
            live_mask=msg.live_mask,
            drive_velocity=np.array([m.drive_velocity for m in mods], dtype=np.float32),
            drive_position=np.array([m.drive_position for m in mods], dtype=np.float32),
            steer_angle=np.array([m.steer_angle for m in mods], dtype=np.float32),
            drive_torque_est=np.array([m.drive_torque_est for m in mods], dtype=np.float32),
            recv_time=time.monotonic(),
        )
        with self._lock:
            self._state = st
            self._state_rx += 1
        # RTT: if this State echoes a command we timed, record it. Same lock the senders
        # use for this dict, so the bounded-history cleanup can't race this pop.
        with self._rtt_lock:
            t_send = self._cmd_seq_send.pop(msg.seq, None)
        if t_send is not None:
            self._last_rtt = st.recv_time - t_send

    def _on_health(self, body: bytes):
        msg = pb.Health()
        try:
            msg.ParseFromString(body)
        except Exception:
            return
        mods = list(msg.modules)
        h = SwerveHealth(
            seq=msg.seq,
            t_us=msg.t_us,
            drive_voltage=np.array([m.drive_voltage for m in mods], dtype=np.float32),
            drive_current=np.array([m.drive_current for m in mods], dtype=np.float32),
            drive_temperature=np.array([m.drive_temperature for m in mods], dtype=np.float32),
            drive_duty=np.array([m.drive_duty for m in mods], dtype=np.float32),
            steer_voltage=np.array([m.steer_voltage for m in mods], dtype=np.float32),
            steer_current=np.array([m.steer_current for m in mods], dtype=np.float32),
            steer_temperature=np.array([m.steer_temperature for m in mods], dtype=np.float32),
            steer_duty=np.array([m.steer_duty for m in mods], dtype=np.float32),
            drive_live=[m.drive_live for m in mods],
            steer_live=[m.steer_live for m in mods],
            bus_down=[m.bus_down for m in mods],
            drive_stale_ms=np.array([m.drive_stale_ms for m in mods], dtype=np.int64),
            steer_stale_ms=np.array([m.steer_stale_ms for m in mods], dtype=np.int64),
            recv_time=time.monotonic(),
        )
        with self._lock:
            self._health = h
            self._health_rx += 1
        self._health_event.set()
