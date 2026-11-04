"""teensy_link.py — the Teensy-Ethernet MIT backend (target wiring for arms + neck).

Structure: ONE physical Teensy carries N actuators for possibly SEVERAL logical
subsystems — the openArm Teensy serves left_arm (l_j1..l_j7) AND right_arm
(r_j1..r_j7); the neck Teensy serves the 3 RSSR motors. So the connection and
the joint-group view are separate objects:

    TeensyLink        owns one robot_host.RobotClient per (ip, port); registry-
                      shared. Latches ONE full-length MIT command vector; each
                      group writes its slice; every send ships the whole vector
                      (this is the begin/commit batching — setCommand is
                      fire-and-forget, so a slice write + merged send is atomic
                      enough for the firmware, which caches the last command).
    TeensyJointGroup  a JointGroupDriver view over a subset of the link's
                      actuators, resolved BY NAME against the RobotInfo
                      self-description (missing names raise SubsystemUnavailable
                      listing what the firmware actually advertises).

SAFETY SEMANTICS (differ from the CAN backend — read before bench use):
  * link.connect() ENABLES the motors: the client's keepalive holds the
    firmware's host-liveness gate open, so every actuator sits in a safe
    damping hold (kp=0, kd=default). A robot on a bench MOVES ON CONNECT if it
    was posed against gravity. MabelHardware.open() therefore only connects
    subsystems whose config says enabled=True.
  * set_torque(False) on a group reverts its slice to the damping hold — the
    protocol has NO per-actuator disable. Fully limp = close the link
    (firmware disables all within ~0.25 s of silence).
  * The firmware's command watchdog (~150 ms) limps the robot if commands stop
    while connected-and-commanding — the owning subsystem's tick() must keep
    calling group.tick() at its stream rate, exactly like the CAN path.
"""
from __future__ import annotations

import threading
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

from ..errors import SubsystemUnavailable
from ..types import monotonic
from .seam import GroupFeedback, JointGroupDriver

__all__ = ["TeensyLink", "TeensyJointGroup"]


class TeensyLink:
    """One UDP link to one Teensy; shared by 1..N TeensyJointGroup views."""

    _registry: Dict[Tuple[str, int], "TeensyLink"] = {}
    _registry_lock = threading.Lock()

    @classmethod
    def get(cls, teensy_ip: str, port: int = 5800, stream_hz: int = 1000,
            logger=None) -> "TeensyLink":
        """The process-wide link for (ip, port) — created on first request, so
        left_arm and right_arm share one socket/handshake."""
        key = (teensy_ip, int(port))
        with cls._registry_lock:
            link = cls._registry.get(key)
            if link is None:
                link = cls._registry[key] = cls(teensy_ip, port, stream_hz,
                                                logger=logger)
            return link

    def __init__(self, teensy_ip: str, port: int = 5800, stream_hz: int = 1000,
                 *, logger=None):
        self.teensy_ip, self.port, self.stream_hz = teensy_ip, int(port), stream_hz
        self._log = logger or (lambda *a, **k: None)
        self._lock = threading.Lock()
        self._client = None                     # robot_host.RobotClient
        self._info = None                       # robot_host.RobotInfo
        self._health = None                     # cached RobotHealth
        self._groups: list = []
        self._state_cbs: list = []
        self._last_connect_try = 0.0            # rate limit for quiet retries
        self._was_up = False                    # for LEFT/REJOINED transition logs
        # merged full-length command (built after RobotInfo arrives)
        self._pos = self._vel = self._kp = self._kd = self._tff = None
        self._commanded = None                  # per-index bool: slice is active
        self._active = False                    # any slice EVER commanded: once
                                                # true we keep streaming (incl. an
                                                # all-damping vector after torque-off,
                                                # so off is immediate, not a 150 ms
                                                # watchdog tail)

    # ── lifecycle ─────────────────────────────────────────────────────────────
    def connect(self, info_timeout: float = 2.0):
        """Hello→RobotInfo handshake. ENABLES the motors (damping hold) — see
        the module docstring. Idempotent. Returns the RobotInfo.

        A (re)connect is a SAFETY RESET for every attached group: latched
        targets are cleared and torque state reverts to off (damping hold), so
        a Teensy that rebooted or was re-plugged NEVER snaps to stale targets —
        the commander must re-command (streaming teleop resumes within one
        cycle anyway)."""
        with self._lock:
            if self._client is not None and self._client.connected():
                return self._info
        from .robot_host import RobotClient    # deferred: numpy/protobuf
        try:
            client = RobotClient(teensy_ip=self.teensy_ip, port=self.port,
                                 stream_hz=self.stream_hz,
                                 on_state=self._fan_state)
            client.connect(info_timeout=info_timeout)
        except (TimeoutError, OSError) as e:
            raise SubsystemUnavailable(
                f"arm/neck Teensy at {self.teensy_ip}:{self.port} not "
                f"answering: {e}") from e
        info = client.info
        n = info.num_actuators
        with self._lock:
            old, self._client = self._client, client
            self._info = info
            self._pos = [0.0] * n
            self._vel = [0.0] * n
            self._kp = [0.0] * n                # kp=0 => damping hold
            self._kd = [a.default_kd for a in info.actuators]
            self._tff = [0.0] * n
            self._commanded = [False] * n
            self._active = False                # limp-until-recommanded
            groups = list(self._groups)
        if old is not None:
            try:
                old.close()
            except Exception:
                pass
        for g in groups:                        # safety reset + (re)resolve
            g._on_link_reset()
        rejoin = " (REJOINED — groups reset to limp until re-commanded)" \
            if self._was_up else ""
        self._was_up = True
        self._log(f"[teensy {self.teensy_ip}] up: robot {info.name!r}, "
                  f"{n} actuators @ <= {info.stream_max_hz} Hz "
                  f"(motors ENABLED in damping hold){rejoin}")
        return info

    def try_connect(self, min_interval: float = 3.0) -> bool:
        """Quiet, rate-limited reconnect for service() paths: True if the link
        is (now) up; never raises, never blocks longer than ~0.5 s."""
        if self.connected():
            return True
        import time as _time
        now = _time.monotonic()
        if now - self._last_connect_try < min_interval:
            return False
        self._last_connect_try = now
        try:
            self.connect(info_timeout=0.5)
            return True
        except SubsystemUnavailable:
            return False

    def close(self):
        with self._lock:
            client, self._client = self._client, None
            self._info = None
        if client is not None:
            client.close()                      # firmware limps in ~0.25 s

    def connected(self) -> bool:
        client = self._client
        return client is not None and client.connected()

    @property
    def info(self):
        return self._info

    # ── group views ───────────────────────────────────────────────────────────
    def group(self, joint_names: Sequence[str], *, kp: float = 30.0,
              kd: float = 1.0, strict: bool = True) -> "TeensyJointGroup":
        """A JointGroupDriver over the named subset of this link's actuators.
        With the link up, names are validated against RobotInfo now
        (strict=True raises on a mismatch, listing what the firmware
        advertises). With the link DOWN (or strict=False) the group is created
        UNRESOLVED and attaches itself on the first successful connect — this
        is how a Teensy that boots after the bridge hot-joins with no restart."""
        g = TeensyJointGroup(self, tuple(joint_names), kp=kp, kd=kd)
        with self._lock:
            self._groups.append(g)
        info = self._info
        if info is not None:
            g._resolve()
            if strict and not g.resolved:
                missing = [n for n in joint_names if n not in info.names]
                raise SubsystemUnavailable(
                    f"robot {info.name!r} does not advertise {missing}; "
                    f"firmware exposes {info.names}")
        return g

    # ── merged command path ───────────────────────────────────────────────────
    def write_slice(self, indices: Sequence[int], pos: Sequence[float],
                    vel: Sequence[float], kp: Sequence[float],
                    kd: Sequence[float], tff: Sequence[float]) -> None:
        with self._lock:
            if self._pos is None:
                raise SubsystemUnavailable("link not connected")
            for j, i in enumerate(indices):
                self._pos[i] = float(pos[j])
                self._vel[i] = float(vel[j])
                self._kp[i] = float(kp[j])
                self._kd[i] = float(kd[j])
                self._tff[i] = float(tff[j])
                self._commanded[i] = True
            self._active = True

    def damp_slice(self, indices: Sequence[int]) -> None:
        """Revert a slice to the firmware-style damping hold (torque 'off')."""
        with self._lock:
            if self._pos is None:
                return
            for i in indices:
                self._kp[i] = 0.0
                self._kd[i] = self._info.actuators[i].default_kd
                self._tff[i] = 0.0
                self._vel[i] = 0.0
                self._commanded[i] = False

    def send(self) -> None:
        """Ship the merged vector (fire-and-forget). No-op until some slice has
        EVER been commanded — an idle link stays in the firmware's own damping
        hold (connected-not-commanding), which is the safer default. Once
        active, we keep streaming even when every slice is back to damping, so
        a torque-off takes effect immediately instead of riding the firmware's
        150 ms command-watchdog tail."""
        with self._lock:
            client = self._client
            if client is None or self._pos is None or not self._active:
                return
            args = (list(self._pos), list(self._vel), list(self._kp),
                    list(self._kd), list(self._tff))
        try:
            client.setCommand(*args)
        except (RuntimeError, ValueError, OSError) as e:
            self._log(f"[teensy {self.teensy_ip}] setCommand failed: {e}")

    # ── shared telemetry ──────────────────────────────────────────────────────
    def state(self):
        client = self._client
        return client.getState() if client is not None else None

    def refresh_health(self, timeout: float = 0.2):
        client = self._client
        if client is None:
            return None
        try:
            self._health = client.getHealthState(timeout=timeout) or self._health
        except Exception:
            pass
        return self._health

    @property
    def health(self):
        return self._health

    def add_state_callback(self, cb) -> Callable[[], None]:
        self._state_cbs.append(cb)
        return lambda: self._state_cbs.remove(cb)

    def _fan_state(self, st):
        for cb in list(self._state_cbs):
            try:
                cb(st)
            except Exception:                    # a slow/broken consumer never
                pass                             # stalls the RX loop


class TeensyJointGroup(JointGroupDriver):
    """JointGroupDriver view over a subset of a TeensyLink's actuators.

    HOT-JOIN: the group may exist BEFORE its Teensy answers (unresolved: reads
    report dead, writes latch locally). service() quietly retries the link and
    resolves the name->index map from RobotInfo the moment the board appears —
    at bridge start or after a mid-run replug. Every link (re)connect resets
    the group to limp/no-targets (see TeensyLink.connect), so a rejoin never
    snaps to stale targets."""

    def __init__(self, link: TeensyLink, names: Tuple[str, ...], *,
                 kp: float = 30.0, kd: float = 1.0):
        self.names = names
        self._link = link
        self._cfg_kp, self._cfg_kd = float(kp), float(kd)
        self._idx: list = []
        self._kp: list = []
        self._kd: list = []
        self._resolved = False
        self._warned_missing = False
        self._targets = {n: None for n in names}     # raw rad; None = not commanded
        self._torque = False
        self._lock = threading.Lock()

    # ── resolution (RobotInfo -> indices/gains) ───────────────────────────────
    @property
    def resolved(self) -> bool:
        return self._resolved

    def _resolve(self) -> bool:
        """Bind names to the firmware's actuator layout. Safe to call any time;
        returns True when bound. A name mismatch logs ONCE and keeps retrying
        (the SD-card robot.json may be fixed and the board re-plugged)."""
        if self._resolved:
            return True
        info = self._link.info
        if info is None:
            return False
        missing = [n for n in self.names if n not in info.names]
        if missing:
            if not self._warned_missing:
                self._warned_missing = True
                self._link._log(
                    f"[teensy {self._link.teensy_ip}] group {self.names} "
                    f"UNRESOLVED: firmware advertises {info.names}, "
                    f"missing {missing} — will re-check on reconnect")
            return False
        with self._lock:
            self._idx = [info.names.index(n) for n in self.names]
            # config gains, clamped to the firmware's advertised maxima
            self._kp = [min(self._cfg_kp, info.actuators[i].kp_max or self._cfg_kp)
                        for i in self._idx]
            self._kd = [min(self._cfg_kd, info.actuators[i].kd_max or self._cfg_kd)
                        for i in self._idx]
            self._resolved = True
            self._warned_missing = False
        return True

    def _on_link_reset(self) -> None:
        """Link (re)connected: SAFETY RESET — drop latched targets, torque off,
        re-resolve against the (possibly new) RobotInfo."""
        with self._lock:
            self._targets = {n: None for n in self.names}
            self._torque = False
            self._resolved = False
        self._resolve()

    # lifecycle — the LINK owns the transport; the group only marks itself
    def connect(self) -> None:
        self._link.connect()
        self._resolve()

    def close(self) -> None:
        """Release this group: its slice reverts to the damping hold. The LINK
        stays up for sibling groups; closing the link itself is the owner's
        (MabelHardware/bridge) shutdown concern."""
        self.set_torque(False)

    def connected(self) -> bool:
        return self._resolved and self._link.connected()

    # write
    def set_targets(self, targets: Mapping[str, float]) -> None:
        with self._lock:
            for name, value in targets.items():
                if name not in self._targets:
                    raise KeyError(f"unknown joint {name!r}; "
                                   f"this group has {', '.join(self.names)}")
                self._targets[name] = float(value)
            if self._torque and self._resolved:
                self._push_locked()

    def command(self, position, velocity=None, kp=None, kd=None, effort=None):
        """Full MIT write for this slice (per-joint overrides where given)."""
        with self._lock:
            for name, value in position.items():
                if name not in self._targets:
                    raise KeyError(name)
                self._targets[name] = float(value)
            if not (self._torque and self._resolved):
                return
            vel = [float((velocity or {}).get(n, 0.0)) for n in self.names]
            kpv = [float((kp or {}).get(n, self._kp[j]))
                   for j, n in enumerate(self.names)]
            kdv = [float((kd or {}).get(n, self._kd[j]))
                   for j, n in enumerate(self.names)]
            tff = [float((effort or {}).get(n, 0.0)) for n in self.names]
            pos = [self._targets[n] if self._targets[n] is not None else 0.0
                   for n in self.names]
            idx = [i for j, i in enumerate(self._idx)
                   if self._targets[self.names[j]] is not None]
            sel = [j for j, n in enumerate(self.names)
                   if self._targets[n] is not None]
            self._link.write_slice(idx, [pos[j] for j in sel],
                                   [vel[j] for j in sel], [kpv[j] for j in sel],
                                   [kdv[j] for j in sel], [tff[j] for j in sel])
        self._link.send()

    def _push_locked(self):
        """Latch every commanded joint of this slice into the link vector."""
        sel = [j for j, n in enumerate(self.names)
               if self._targets[n] is not None]
        if not sel:
            return
        self._link.write_slice(
            [self._idx[j] for j in sel],
            [self._targets[self.names[j]] for j in sel],
            [0.0] * len(sel),
            [self._kp[j] for j in sel],
            [self._kd[j] for j in sel],
            [0.0] * len(sel))

    def set_torque(self, on: bool) -> None:
        with self._lock:
            self._torque = bool(on)
            if not self._resolved:
                return                           # takes effect after resolve
            if self._torque:
                self._push_locked()
            else:
                self._link.damp_slice(self._idx)
        self._link.send()

    @property
    def is_torque_enabled(self) -> bool:
        return self._torque

    # read
    def read(self) -> GroupFeedback:
        st = self._link.state() if self._resolved else None
        h = self._link.health
        if st is None:
            return GroupFeedback(names=self.names,
                                 position=(0.0,) * len(self.names),
                                 velocity=(0.0,) * len(self.names),
                                 effort=(0.0,) * len(self.names),
                                 temperature=(0.0,) * len(self.names),
                                 live=(False,) * len(self.names),
                                 stamp=monotonic())
        temp = tuple(float(h.t_mos[i]) if h is not None else 0.0
                     for i in self._idx)
        return GroupFeedback(
            names=self.names,
            position=tuple(float(st.position[i]) for i in self._idx),
            velocity=tuple(float(st.velocity[i]) for i in self._idx),
            effort=tuple(float(st.torque[i]) for i in self._idx),
            temperature=temp,
            live=tuple(st.is_live(i) for i in self._idx),
            stamp=st.recv_time,
        )

    def health(self) -> Dict[str, Dict[str, float]]:
        fb = self.read()
        return {n: {"torqueNm": round(fb.effort[j], 3),
                    "tempC": int(fb.temperature[j])}
                for j, n in enumerate(self.names)}

    # pacing
    def tick(self) -> None:
        """Re-send the merged link command — feeds the firmware's ~150 ms
        command watchdog, exactly like the CAN path's MIT re-send."""
        self._link.send()

    def service(self) -> None:
        """Slow-path watchdog: quiet rate-limited link reconnect, (re)resolve
        against RobotInfo, health refresh. This is the hot-join path — a Teensy
        that appears mid-run attaches here within ~3 s."""
        if not self._link.try_connect():
            return
        if not self._resolved:
            self._resolve()
        self._link.refresh_health()

    def on_state(self, callback: Callable[[GroupFeedback], None]):
        names = self.names

        def _slice_cb(st):
            if not self._resolved:               # pre-resolve packets: skip
                return
            idx = self._idx
            callback(GroupFeedback(
                names=names,
                position=tuple(float(st.position[i]) for i in idx),
                velocity=tuple(float(st.velocity[i]) for i in idx),
                effort=tuple(float(st.torque[i]) for i in idx),
                temperature=(0.0,) * len(idx),
                live=tuple(st.is_live(i) for i in idx),
                stamp=st.recv_time))

        return self._link.add_state_callback(_slice_cb)
