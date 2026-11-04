"""seam.py — the ONE interface every MIT joint-group backend implements.

Streaming model (matches both real transports):
  * ``set_targets()`` LATCHES the setpoint — it is cheap and thread-safe;
  * ``tick()`` re-sends the latched command at the stream rate (this is what
    feeds the firmware watchdogs: DaMiao MIT re-send, Teensy 150 ms cmd window);
  * ``read()`` returns the latest CACHED feedback — it never blocks on the wire.

Units at this layer are RAW motor radians — calibration (raw<->model mapping,
direction, clamps) is applied above, in mabel_hw.subsystems.joint_group, so a
backend never needs to know how the robot was calibrated.

Backends: CanLink (drivers.damiao_can, socketcan MIT — today's proven path),
TeensyJointGroup (drivers.teensy_link, UDP protobuf — the target wiring), and
MockJointGroupDriver (below — deterministic, hardware-free, the tests' plant).
"""
from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

from ..types import monotonic

__all__ = ["GroupFeedback", "JointGroupDriver", "MockJointGroupDriver"]


@dataclass(frozen=True)
class GroupFeedback:
    """Raw feedback snapshot, parallel tuples in the driver's joint order."""
    names: Tuple[str, ...]
    position: Tuple[float, ...]        # rad (raw motor frame)
    velocity: Tuple[float, ...]        # rad/s
    effort: Tuple[float, ...]          # N·m
    temperature: Tuple[float, ...]     # degC (0.0 when the backend lacks it)
    live: Tuple[bool, ...]             # fresh feedback seen for this joint?
    stamp: float


class JointGroupDriver(ABC):
    """Protocol-agnostic MIT joint group (arms, neck motors, torso)."""

    #: canonical joint names, config order — fixed for the driver's lifetime
    names: Tuple[str, ...]

    # ── lifecycle ─────────────────────────────────────────────────────────────
    @abstractmethod
    def connect(self) -> None:
        """Open the transport. Raises on hard failure; the caller (subsystem)
        degrades gracefully. NOTE some backends energize on connect (TeensyLink
        keepalive = damping hold) — see the backend docstring."""

    @abstractmethod
    def close(self) -> None:
        """Release the transport; motors limp per the backend's semantics."""

    @abstractmethod
    def connected(self) -> bool: ...

    # ── write side ────────────────────────────────────────────────────────────
    @abstractmethod
    def set_targets(self, targets: Mapping[str, float]) -> None:
        """Latch new position targets (raw rad). Unknown names raise KeyError;
        joints absent from the mapping keep their previous target."""

    def command(self, position: Mapping[str, float],
                velocity: Optional[Mapping[str, float]] = None,
                kp: Optional[Mapping[str, float]] = None,
                kd: Optional[Mapping[str, float]] = None,
                effort: Optional[Mapping[str, float]] = None) -> None:
        """Full MIT command. Default: position-only (backends that support
        per-joint gains/feedforward override this)."""
        self.set_targets(position)

    @abstractmethod
    def set_torque(self, on: bool) -> None: ...

    @property
    @abstractmethod
    def is_torque_enabled(self) -> bool: ...

    # ── read side ─────────────────────────────────────────────────────────────
    @abstractmethod
    def read(self) -> GroupFeedback: ...

    @abstractmethod
    def health(self) -> Dict[str, Dict[str, float]]:
        """{joint: {"torqueNm": .., "tempC": ..}} — the motor_health contract."""

    # ── pacing hooks (called by the owning subsystem's timers) ────────────────
    def tick(self) -> None:
        """Fast path: re-send the latched command (stream rate). Default no-op
        for backends whose transport streams autonomously."""

    def service(self) -> None:
        """Slow path (~1 Hz): reconnect watchdog, health refresh. Default no-op."""

    # ── optional push ─────────────────────────────────────────────────────────
    def on_state(self, callback: Callable[[GroupFeedback], None]) -> Callable[[], None]:
        """Register a packet-driven feedback callback where the transport
        supports it (TeensyLink). Default: unsupported no-op; poll read()."""
        return lambda: None


class MockJointGroupDriver(JointGroupDriver):
    """Deterministic hardware-free backend — the plant for unit tests and
    ``MabelHardware(mock=True)``.

    Physics: each ``tick()`` moves every joint a fixed fraction ``alpha`` of the
    way to its target (first-order lag), ONLY while torque is enabled — matching
    the real MIT behavior that a torque-off motor holds still however hard you
    stream setpoints. No background threads; no wall-clock dependence: N ticks
    give bit-identical trajectories on every run.
    """

    def __init__(self, names: Sequence[str], *, alpha: float = 0.5,
                 initial: float = 0.0, temperature: float = 35.0):
        self.names = tuple(names)
        self._alpha = float(alpha)
        self._temp = float(temperature)
        self._lock = threading.Lock()
        self._connected = False
        self._torque = False
        self._pos = {n: float(initial) for n in self.names}
        self._vel = {n: 0.0 for n in self.names}
        self._targets = {n: float(initial) for n in self.names}
        self._callbacks: list = []
        self.tick_count = 0

    # lifecycle
    def connect(self) -> None:
        with self._lock:
            self._connected = True

    def close(self) -> None:
        with self._lock:
            self._connected = False
            self._torque = False

    def connected(self) -> bool:
        return self._connected

    # write
    def set_targets(self, targets: Mapping[str, float]) -> None:
        with self._lock:
            for name, value in targets.items():
                if name not in self._targets:
                    raise KeyError(f"unknown joint {name!r}; "
                                   f"this group has {', '.join(self.names)}")
                self._targets[name] = float(value)

    def set_torque(self, on: bool) -> None:
        with self._lock:
            self._torque = bool(on)

    @property
    def is_torque_enabled(self) -> bool:
        return self._torque

    # read
    def read(self) -> GroupFeedback:
        with self._lock:
            return GroupFeedback(
                names=self.names,
                position=tuple(self._pos[n] for n in self.names),
                velocity=tuple(self._vel[n] for n in self.names),
                effort=tuple(0.0 for _ in self.names),
                temperature=tuple(self._temp for _ in self.names),
                live=tuple(self._connected for _ in self.names),
                stamp=monotonic(),
            )

    def health(self) -> Dict[str, Dict[str, float]]:
        return {n: {"torqueNm": 0.0, "tempC": self._temp} for n in self.names}

    # pacing
    def tick(self) -> None:
        with self._lock:
            self.tick_count += 1
            if not (self._connected and self._torque):
                for n in self.names:
                    self._vel[n] = 0.0
                return
            for n in self.names:
                step = self._alpha * (self._targets[n] - self._pos[n])
                self._pos[n] += step
                self._vel[n] = step          # per-tick displacement as pseudo-velocity
        fb = self.read()
        for cb in list(self._callbacks):
            cb(fb)

    def on_state(self, callback):
        self._callbacks.append(callback)
        return lambda: self._callbacks.remove(callback)
