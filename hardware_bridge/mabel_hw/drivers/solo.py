"""solo.py — seam adapters for the single-motor DaMiao dongles + the multi-bus
composite that lets one subsystem span both.

Two backends exist for a single dongle-attached DaMiao motor and share one
surface (``DamiaoSerialDriver`` for the HDSC serial dongle — the torso — and
``DamiaoFdcanDriver`` for the DM-USB2FDCAN — the neck yaw). Both are
request/response drivers whose ``read()`` BLOCKS on the wire, while the
:class:`~mabel_hw.drivers.seam.JointGroupDriver` seam requires ``read()`` to be
a cache hit. :class:`DamiaoSoloJoint` bridges that: the wire is only touched
from ``tick()``/``service()`` (the bridge's pacing timers) and ``read()``
returns the cached snapshot, aging to ``live=False`` when the motor goes
silent (the HAL's publish-gate-on-live rule).

:class:`CompositeJointGroup` merges several seam drivers into one group so the
neck subsystem can span its two buses (Dynamixel pitch/roll + FDCAN yaw) while
staying a single calibrated ``JointGroupSubsystem``. Either bus may be absent:
its joints read dead while the other keeps working, and ``service()`` keeps
retrying the missing one (hot-join).

Safety contract (same as every studio driver): connect never energizes;
``set_torque(True)`` is the underlying driver's safe hold (present pose
commanded in the same breath as enable — no snap) and CLEARS the latched
target so a stale setpoint can never replay across a torque cycle; a silent
motor refuses to energize (logged, never raised into the node).
"""
from __future__ import annotations

import threading
from typing import Dict, Mapping, Optional, Sequence, Tuple

from ..errors import SubsystemUnavailable
from ..types import monotonic
from .seam import GroupFeedback, JointGroupDriver

__all__ = ["DamiaoSoloJoint", "CompositeJointGroup"]

_STALE_S = 0.7          # cached feedback older than this reads live=False
_DEAD_POLL_S = 1.0      # poll cadence while the motor is silent (each poll can
#                         block up to the driver's rx timeout — keep it rare)
_LIVE_POLL_S = 0.04     # min gap between wire touches while live
_STREAM_HZ = 50.0       # MIT streaming rate while driving (own thread)
_SETTLE_RAD = 0.002     # |target − setpoint| below this = arrived (v_ff -> 0)


class DamiaoSoloJoint(JointGroupDriver):
    """One dongle-attached DaMiao motor behind the JointGroupDriver seam.

    Command smoothing (2026-07-27, "torso feels rigid and jittery"): the MIT law
    is a raw PD with NO profile generator, so replaying a 10-20 Hz stair-step of
    position targets with v_des = 0 makes kd FIGHT every step — jitter — and the
    only way to hold gravity was brutal kp. So the adapter runs its own 50 Hz
    streaming thread: latched targets are SLEWED (``slew`` rad/s) into a smooth
    setpoint sent with matching VELOCITY feedforward (kd damps around the motion),
    plus the commander's TORQUE feedforward (gravity — lets kp stay soft). The
    same recipe already proven on the studio's yaw glide and the arms' teleop FF.
    """

    def __init__(self, drv, joint: str, slew: float = 1.0, logger=None):
        self._drv = drv
        self.names: Tuple[str, ...] = (str(joint),)
        self._slew = float(slew)                  # rad/s glide speed cap
        self._log = logger or (lambda *a, **k: None)
        self._lock = threading.Lock()
        self._target: Optional[float] = None      # latched RAW target
        self._t_ff: float = 0.0                   # latched RAW torque feedforward (N·m)
        self._setpoint: Optional[float] = None    # streamed glide position
        self._last_wire = 0.0                     # monotonic of the last wire touch
        self._st = None                           # last DamiaoSerialState-shaped snapshot
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ── lifecycle ────────────────────────────────────────────────────────────
    def connect(self) -> None:
        self._drv.connect()                       # non-energizing (both backends)
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(target=self._stream_run, daemon=True,
                                            name=f"solo-{self.names[0]}")
            self._thread.start()

    def close(self) -> None:
        self._stop.set()
        self._drv.close()

    def connected(self) -> bool:
        return bool(self._drv.connected())

    # ── write side ───────────────────────────────────────────────────────────
    def set_targets(self, targets: Mapping[str, float]) -> None:
        for name, value in targets.items():
            if name != self.names[0]:
                raise KeyError(f"unknown joint {name!r}; this group has {self.names[0]}")
            with self._lock:
                self._target = float(value)

    def command(self, position, velocity=None, kp=None, kd=None, effort=None) -> None:
        """Full MIT command: position target + torque feedforward (gravity). The
        commander's kp/kd overrides are deliberately ignored — gains are the
        config's; softness comes from t_ff + the glide, not per-frame gains."""
        if position:
            self.set_targets(position)
        if effort and self.names[0] in effort:
            with self._lock:
                self._t_ff = float(effort[self.names[0]])

    def set_torque(self, on: bool) -> None:
        with self._lock:
            self._target = None                   # never replay a stale setpoint
            self._setpoint = None
            if not on:
                self._t_ff = 0.0
        try:
            self._drv.set_torque(bool(on))
        except SubsystemUnavailable as e:
            # A silent/unpowered motor refuses to energize — degrade, don't crash
            # the node (CanLink's fire-and-forget semantics).
            self._log(f"[{self.names[0]}] torque {'on' if on else 'off'} refused: {e}")

    @property
    def is_torque_enabled(self) -> bool:
        return self._drv.is_torque_enabled is True

    # ── read side (cache hit — never touches the wire) ───────────────────────
    def read(self) -> GroupFeedback:
        st = self._st
        fresh = st is not None and st.live and (monotonic() - st.stamp) < _STALE_S
        return GroupFeedback(
            names=self.names,
            position=(float(st.position) if st is not None else 0.0,),
            velocity=(float(st.velocity) if fresh else 0.0,),
            effort=(float(st.torque) if fresh else 0.0,),
            temperature=(float(st.t_mos) if st is not None else 0.0,),
            live=(bool(fresh),),
            stamp=(st.stamp if st is not None else 0.0),
        )

    def health(self) -> Dict[str, Dict[str, float]]:
        st = self._st
        return {self.names[0]: {"torqueNm": float(st.torque) if st else 0.0,
                                "tempC": float(st.t_mos) if st else 0.0}}

    # ── pacing ───────────────────────────────────────────────────────────────
    def tick(self) -> None:
        """Feedback keepalive from the node timer: poll (0xCC, non-energizing)
        when the streamer is idle, throttled so a silent motor costs one bounded
        wire timeout per second. Driving I/O lives on the streaming thread."""
        if not self._drv.connected():
            return
        with self._lock:
            driving = self._target is not None and self.is_torque_enabled
        if driving:
            return                                # the streamer owns the wire
        now = monotonic()
        st = self._st
        alive = st is not None and st.live and (now - st.stamp) < _STALE_S
        gap = _LIVE_POLL_S if alive else _DEAD_POLL_S
        if now - self._last_wire < gap:
            return
        self._last_wire = now
        try:
            self._st = self._drv.read()
        except SubsystemUnavailable:
            pass                                  # driver drops a dead handle itself;
        #                                           service() reconnects

    # ── the 50 Hz MIT streamer (smoothness lives here) ───────────────────────
    def _glide_step(self, dt: float) -> bool:
        """One streaming step: slew the setpoint ≤ slew·dt toward the latched
        target and send ONE MIT frame with matching velocity feedforward and the
        latched torque feedforward. Returns True if a frame was sent."""
        with self._lock:
            tgt, t_ff = self._target, self._t_ff
        if tgt is None or not self.is_torque_enabled or not self._drv.connected():
            self._setpoint = None
            return False
        if self._setpoint is None:                # (re)entering motion: start FROM
            st = self._st                         # the measured pose — never a snap
            if st is None or not st.live:
                try:
                    st = self._st = self._drv.read()
                except SubsystemUnavailable:
                    return False
                if not st.live:
                    return False
            self._setpoint = float(st.position)
        err = tgt - self._setpoint
        step = max(-self._slew * dt, min(self._slew * dt, err))
        sp = self._setpoint + step
        # Arrival frame carries v_ff = 0 so kd settles the joint AT the goal;
        # while moving, v_ff = the trajectory's own rate (kd damps around it).
        v_ff = step / dt if abs(tgt - sp) > _SETTLE_RAD else 0.0
        try:
            self._st = self._drv.set_target(sp, v_ff, t_ff)
            self._last_wire = monotonic()
        except SubsystemUnavailable:
            self._setpoint = None                 # torque dropped / dongle reset —
            return False                          # re-seed from measured on resume
        self._setpoint = sp
        return True

    def _stream_run(self) -> None:
        dt = 1.0 / _STREAM_HZ
        while not self._stop.wait(dt):
            try:
                self._glide_step(dt)
            except Exception:                     # never kill the stream thread
                pass

    def service(self) -> None:
        """Slow path: reconnect a dongle that was absent/reset (hot-join)."""
        if not self._drv.connected():
            try:
                self._drv.connect()
                self._log(f"[{self.names[0]}] dongle reconnected")
            except SubsystemUnavailable:
                pass


class CompositeJointGroup(JointGroupDriver):
    """Several seam drivers presented as ONE joint group (one group per
    subsystem, several buses underneath — MABEL's neck)."""

    def __init__(self, children: Sequence[JointGroupDriver], logger=None):
        self._children = tuple(children)
        self._log = logger or (lambda *a, **k: None)
        names: list = []
        self._owner: Dict[str, JointGroupDriver] = {}
        for ch in self._children:
            for n in ch.names:
                if n in self._owner:
                    raise ValueError(f"joint {n!r} owned by two children")
                names.append(n)
                self._owner[n] = ch
        self.names = tuple(names)

    # ── lifecycle ────────────────────────────────────────────────────────────
    def connect(self) -> None:
        """Connect every bus; PARTIAL is fine (the missing one hot-joins via
        service()). Raises only when NO bus came up."""
        errs = []
        for ch in self._children:
            try:
                ch.connect()
            except SubsystemUnavailable as e:
                errs.append(f"{'/'.join(ch.names)}: {e}")
        if errs and not any(ch.connected() for ch in self._children):
            raise SubsystemUnavailable("; ".join(errs))
        for line in errs:
            self._log(f"[composite] bus down ({line}) — will hot-join")

    def close(self) -> None:
        for ch in self._children:
            try:
                ch.close()
            except Exception:
                pass

    def connected(self) -> bool:
        return any(ch.connected() for ch in self._children)

    # ── write side ───────────────────────────────────────────────────────────
    def _split(self, mapping: Optional[Mapping[str, float]]):
        out: Dict[JointGroupDriver, Dict[str, float]] = {}
        for name, value in (mapping or {}).items():
            ch = self._owner.get(name)
            if ch is None:
                raise KeyError(f"unknown joint {name!r}; "
                               f"this group has {', '.join(self.names)}")
            out.setdefault(ch, {})[name] = value
        return out

    def set_targets(self, targets: Mapping[str, float]) -> None:
        for ch, sub in self._split(targets).items():
            try:
                ch.set_targets(sub)
            except SubsystemUnavailable:
                pass                              # that bus is down; others proceed

    def command(self, position, velocity=None, kp=None, kd=None, effort=None):
        pos = self._split(position)
        vel, kpm, kdm, eff = (self._split(m) for m in (velocity, kp, kd, effort))
        for ch in {*pos, *vel, *kpm, *kdm, *eff}:
            try:
                ch.command(pos.get(ch, {}), velocity=vel.get(ch),
                           kp=kpm.get(ch), kd=kdm.get(ch), effort=eff.get(ch))
            except SubsystemUnavailable:
                pass

    def set_torque(self, on: bool) -> None:
        for ch in self._children:
            try:
                ch.set_torque(on)
            except SubsystemUnavailable as e:
                self._log(f"[composite] torque on {'/'.join(ch.names)} refused: {e}")

    @property
    def is_torque_enabled(self) -> bool:
        return any(ch.is_torque_enabled for ch in self._children)

    # ── read side ────────────────────────────────────────────────────────────
    def read(self) -> GroupFeedback:
        by: Dict[str, tuple] = {}
        stamp = 0.0
        for ch in self._children:
            try:
                fb = ch.read()
            except Exception:                     # dead bus → its joints read dead
                continue
            stamp = max(stamp, fb.stamp)
            for i, n in enumerate(fb.names):
                by[n] = (fb.position[i], fb.velocity[i], fb.effort[i],
                         fb.temperature[i], fb.live[i])
        z = (0.0, 0.0, 0.0, 0.0, False)
        rows = [by.get(n, z) for n in self.names]
        return GroupFeedback(
            names=self.names,
            position=tuple(r[0] for r in rows),
            velocity=tuple(r[1] for r in rows),
            effort=tuple(r[2] for r in rows),
            temperature=tuple(r[3] for r in rows),
            live=tuple(r[4] for r in rows),
            stamp=stamp,
        )

    def health(self) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for ch in self._children:
            try:
                out.update(ch.health())
            except Exception:
                pass
        return out

    # ── pacing ───────────────────────────────────────────────────────────────
    def tick(self) -> None:
        for ch in self._children:
            try:
                ch.tick()
            except Exception:
                pass

    def service(self) -> None:
        for ch in self._children:
            try:
                ch.service()
            except Exception:
                pass
