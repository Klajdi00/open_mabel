"""joint_group.py — calibrated MIT joint groups (arms, torso, neck motors).

Sits between a raw :class:`~mabel_hw.drivers.seam.JointGroupDriver` (any
backend) and the facade: everything ABOVE this layer speaks MODEL-frame radians
(the MJCF joint convention); everything below speaks raw motor radians. The
raw<->model map comes from the CalibrationStore slice named in the config
(``calib_slice``) and hot-swaps atomically when the store reloads — targets are
latched in MODEL frame here, so a calibration update re-rasterizes to raw on
the next write/tick without a motion glitch beyond the calibration delta.

Uncalibrated joints pass through 1:1 (raw == model) — bench robots and the
torso (no wizard slice yet) work out of the box, just uncalibrated.
"""
from __future__ import annotations

import threading
from typing import Dict, Mapping, Optional, Tuple

from ..calibration import CalibrationStore, JointCalib
from ..drivers import build_joint_driver
from ..drivers.seam import JointGroupDriver
from ..types import JointCommand, JointGroupState

__all__ = ["JointGroupSubsystem"]


class JointGroupSubsystem:
    """One calibrated MIT joint group. Thread-safe; writes are lease-gated by
    the facade (this class itself only executes)."""

    def __init__(self, name: str, cfg: dict, links: dict, *,
                 store: Optional[CalibrationStore] = None,
                 driver: Optional[JointGroupDriver] = None, logger=None):
        self.name = name
        self._cfg = cfg
        self._log = logger or (lambda *a, **k: None)
        self._driver = driver if driver is not None else \
            build_joint_driver(cfg, links, logger=logger)
        self.names: Tuple[str, ...] = tuple(self._driver.names)
        self._slice = cfg.get("calib_slice")
        # Held off each calibrated hard stop when clamping a target (see config).
        self._margin = float(cfg.get("end_stop_margin", 0.0))
        self._store = store
        self._calib: Dict[str, JointCalib] = {}
        self._targets: Dict[str, float] = {}     # MODEL frame, latched
        self._lock = threading.Lock()
        # Multi-turn DaMiao continuous-unwrap state (only used when p_max>0, i.e. the
        # arms/torso; the neck's XC330 is single-turn so p_max is absent → 2π fold only).
        # The DaMiao encoder reports position wrapped into [-p_max, +p_max]; near the
        # ±12.5 ceiling (l_j4) the 2π fold is the WRONG modulus, so we delta-unwrap.
        self._pmax = float(cfg.get("p_max", 0.0) or 0.0)
        self._cont: Dict[str, tuple] = {}        # name -> (last_raw, continuous)
        self._cont_lock = threading.Lock()       # separate: _to_model runs lock-free
        if store is not None:
            self.recalibrate(store)

    # ── calibration ───────────────────────────────────────────────────────────
    def recalibrate(self, store: CalibrationStore) -> None:
        """Atomic calibration swap (called at open and by the store watcher)."""
        calib = store.group(self._slice, self.names) if self._slice else {}
        with self._lock:
            self._calib = calib
        with self._cont_lock:                    # re-seed continuous frame to the NEW
            self._cont.clear()                   # window (live re-calibration moves rmid)
        if self._slice:
            missing = [n for n in self.names if n not in calib]
            self._log(f"[{self.name}] calibration {self._slice}: "
                      f"{len(calib)}/{len(self.names)} joints"
                      + (f" (uncalibrated: {missing})" if missing else ""))

    def _to_raw(self, name: str, model: float,
                cur: Optional[float] = None) -> float:
        """model -> raw. ALWAYS pass ``cur`` (the joint's current raw angle) when
        it is known: DaMiao encoders are multi-turn, and without ``cur`` the
        command is emitted in the calibration WINDOW's revolution rather than the
        MOTOR's — a "stay put" that reads to the motor as "rotate -356°", i.e. a
        full-torque stall against a stop (see JointCalib.to_raw).

        The clamp also holds `end_stop_margin` off each calibrated hard stop: the
        window IS the measured stop, so clamping exactly onto it parks the motor
        against the stop at full kp*error."""
        jc = self._calib.get(name)
        return (jc.to_raw(model, cur, margin=self._margin)
                if jc is not None else model)

    def _to_model(self, name: str, raw: float) -> float:
        jc = self._calib.get(name)
        if jc is None:
            return raw
        if self._pmax > 0.0:
            # Multi-turn DaMiao: the encoder reports position wrapped into [-p_max, p_max]
            # (a ±p_max ≈ 25 rad step that is NOT a whole number of turns), so the 2π fold
            # in to_model() can never reconcile a p_max wrap — l_j4 sits hard against +12.5
            # and its readback jumps to -12.5 the instant it crosses over. Rebuild a
            # CONTINUOUS raw by delta-unwrapping the modular feedback, then map with NO fold.
            # (Ported from upstream arm_bridge._continuous — the documented HAL fix.)
            return jc.model_from(self._continuous(name, raw, jc))
        return jc.to_model(raw)

    def _continuous(self, name: str, raw: float, jc: JointCalib) -> float:
        """CONTINUOUS raw from the ±p_max-modular DaMiao feedback. The frame is SEEDED via
        the calibration's own 2π unwrap (so an out-of-band boot window still lands in the
        right revolution — no regression), then each sample-to-sample DELTA is folded into
        (-p_max, p_max] and accumulated, so the signal stays continuous straight through a
        ±p_max ceiling crossing (which 2π folding cannot do). Re-seeds after recalibrate()
        clears `_cont`. Small deltas make this robust to concurrent readers."""
        span = 2.0 * self._pmax
        with self._cont_lock:
            prev = self._cont.get(name)
            if prev is None:
                seed = jc.unwrap(raw)                 # 2π-fold into the window (boot)
                self._cont[name] = (raw, seed)
                return seed
            last_raw, cont = prev
            delta = raw - last_raw
            delta -= span * round(delta / span)       # de-wrap the ±p_max ceiling step
            cont += delta
            self._cont[name] = (raw, cont)
            return cont

    def _dir(self, name: str) -> int:
        """The joint's calibrated direction (+1/-1). model = f(dir * raw), so
        RATES map by dir alone: model_vel = dir * raw_vel, and likewise torque."""
        jc = self._calib.get(name)
        return jc.dir if jc is not None else 1

    def _cur_raw(self) -> Dict[str, float]:
        """{joint: current RAW angle} from the driver's cached feedback, for the
        multi-turn fold in _to_raw. Never blocks (read() is a cache hit)."""
        try:
            fb = self._driver.read()
        except Exception:                      # a dead driver can't move anyway
            return {}
        return {n: p for n, p, live in zip(fb.names, fb.position, fb.live) if live}

    # ── lifecycle (facade-managed) ────────────────────────────────────────────
    def connect(self) -> None:
        self._driver.connect()

    def close(self) -> None:
        self._driver.close()

    def connected(self) -> bool:
        return self._driver.connected()

    # ── read side (always free) ───────────────────────────────────────────────
    def state(self) -> JointGroupState:
        fb = self._driver.read()
        return JointGroupState(
            names=fb.names,
            position=tuple(self._to_model(n, p)
                           for n, p in zip(fb.names, fb.position)),
            # Rates map by dir alone (model = model_mid + dir*(raw - raw_mid)),
            # so a dir=-1 joint's velocity/torque MUST flip with its position or
            # the reported effort points the opposite way to the motion causing
            # it. Pins the sign convention the hands share (drivers/orcahand/
            # joints.py) and the app's motorHealth renders.
            velocity=tuple(self._dir(n) * v
                           for n, v in zip(fb.names, fb.velocity)),
            effort=tuple(self._dir(n) * e
                         for n, e in zip(fb.names, fb.effort)),
            temperature=fb.temperature,
            live=fb.live,
            torque_enabled=self._driver.is_torque_enabled,
            stamp=fb.stamp,
        )

    def health(self) -> Dict[str, Dict[str, float]]:
        return self._driver.health()

    def on_state(self, callback):
        return self._driver.on_state(callback)

    # ── write side (lease-gated by the facade handle) ─────────────────────────
    def set_targets(self, targets: Mapping[str, float]) -> None:
        """MODEL-frame position targets; unknown names raise KeyError."""
        cur = self._cur_raw()
        raw = {}
        with self._lock:
            for name, value in targets.items():
                if name not in self.names:
                    raise KeyError(f"unknown joint {name!r}; "
                                   f"{self.name} has {', '.join(self.names)}")
                self._targets[name] = float(value)
                raw[name] = self._to_raw(name, float(value), cur.get(name))
        self._driver.set_targets(raw)

    def command(self, cmd: JointCommand) -> None:
        """Full MIT command, MODEL frame in / RAW frame out. Position uses the
        calibration map; velocity and effort are rates, so they map by dir alone
        (the inverse of what state() reports). Gains are frame-free."""
        cur = self._cur_raw()
        raw_pos, raw_vel, raw_eff = {}, {}, {}
        with self._lock:
            for name, value in cmd.position.items():
                if name not in self.names:
                    raise KeyError(name)
                self._targets[name] = float(value)
                raw_pos[name] = self._to_raw(name, float(value), cur.get(name))
            for name, v in cmd.velocity.items():
                raw_vel[name] = self._dir(name) * float(v)
            for name, e in cmd.effort.items():
                raw_eff[name] = self._dir(name) * float(e)
        self._driver.command(raw_pos, velocity=raw_vel, kp=cmd.kp,
                             kd=cmd.kd, effort=raw_eff)

    def set_torque(self, on: bool) -> None:
        self._driver.set_torque(on)

    @property
    def is_torque_enabled(self) -> bool:
        return self._driver.is_torque_enabled

    # ── safety (lease-FREE at the facade) ─────────────────────────────────────
    def stop(self) -> None:
        """Hold-in-place: re-target the CURRENT position (never disables — a
        torque-off under load would drop the arm; that is a deliberate,
        lease-gated action, not a stop)."""
        fb = self._driver.read()
        try:
            self._driver.set_targets(dict(zip(fb.names, fb.position)))
        except Exception:                      # a dead driver can't move anyway
            pass
        with self._lock:
            self._targets = {n: self._to_model(n, p)
                             for n, p in zip(fb.names, fb.position)}

    # ── pacing (bridge timers) ────────────────────────────────────────────────
    def tick(self) -> None:
        self._driver.tick()

    def service(self) -> None:
        self._driver.service()

