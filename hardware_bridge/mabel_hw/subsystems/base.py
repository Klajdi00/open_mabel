"""base.py — the MAXSwerve mobile base as a typed subsystem.

Wraps the proven :class:`~mabel_hw.drivers.swerve.SwerveBase` (twist -> ps5 IK
-> calibration -> Teensy) and adds the typed read side: per-module steer/drive/
torque-estimate from the streamed State, electrical health (volts/amps/temp/
duty) from the request-reply Health at a low service rate.

estop is deliberately lease-FREE at the facade: zeroing drive must never wait
on arbitration.
"""
from __future__ import annotations

import math
from typing import Optional

from ..types import BaseHealth, BaseState, SwerveModuleState, monotonic

__all__ = ["BaseSubsystem", "MODULE_NAMES"]

# maxswerve hardware order (matches SwerveState arrays)
MODULE_NAMES = ("back", "front_right", "front_left")


class BaseSubsystem:
    def __init__(self, cfg: dict, *, swerve=None, logger=None):
        self._cfg = cfg
        self._log = logger or (lambda *a, **k: None)
        if swerve is not None:                   # injected (mock / tests)
            self._base = swerve
        else:
            from ..drivers.swerve import SwerveBase
            self._base = SwerveBase(
                teensy_ip=cfg["teensy_ip"], port=cfg["port"],
                stream_hz=cfg["stream_hz"], cmd_timeout=cfg["cmd_timeout"],
                wheel_radius=cfg["wheel_radius"],
                apply_host_calib=cfg["apply_host_calib"],
                drive_gain=cfg["drive_gain"], logger=logger)
        self._estop = False
        self._health: Optional[BaseHealth] = None

    # the raw SwerveBase, for the ROS adapter's pad loop / legacy call sites
    @property
    def swerve(self):
        return self._base

    # ── lifecycle ─────────────────────────────────────────────────────────────
    def connect(self) -> None:
        pass                                     # SwerveBase connects in __init__

    def close(self) -> None:
        self._base.close()

    def connected(self) -> bool:
        return self._base.connected()

    # ── read ──────────────────────────────────────────────────────────────────
    def state(self) -> BaseState:
        sw = getattr(self._base, "sw", None)
        st = sw.getState() if sw is not None else None
        modules = ()
        stamp = monotonic()
        if st is not None:
            try:
                modules = tuple(
                    SwerveModuleState(
                        name=MODULE_NAMES[i],
                        steer_angle=float(st.steer_angle_rad[i]),
                        drive_velocity=float(st.drive_velocity[i]),
                        drive_position=float(st.drive_position[i]) * 2.0 * math.pi,
                        drive_torque_est=float(st.drive_torque_est[i]))
                    for i in range(3))
                stamp = st.recv_time
            except Exception:                     # pragma: no cover — wire quirk
                modules = ()
        return BaseState(modules=modules, estop=self._estop,
                         connected=self.connected(), health=self._health,
                         stamp=stamp)

    # ── write (lease-gated at the facade) ─────────────────────────────────────
    def set_twist(self, vx: float, vy: float, wz: float) -> None:
        """REAL frame (+x fwd, +y left, CCW yaw); pushed on the next tick and
        immediately by the caller's tick-after-write (the DDS adapter does)."""
        self._base.set_twist(vx, vy, wz)

    # ── safety (lease-FREE) ───────────────────────────────────────────────────
    def set_estop(self, on: bool) -> None:
        self._estop = bool(on)
        self._base.set_estop(self._estop)

    def stop(self) -> None:
        """Zero drive NOW (latch estop). clear via set_estop(False)."""
        self.set_estop(True)
        self._base.tick()

    # ── pacing ────────────────────────────────────────────────────────────────
    def tick(self) -> None:
        self._base.tick()

    def service(self) -> None:
        self._base.service()
        sw = getattr(self._base, "sw", None)
        if sw is None:
            return
        try:
            h = sw.getHealthState(timeout=0.15)
        except Exception:
            return
        if h is None:
            return
        try:
            self._health = BaseHealth(
                voltage=tuple(float(v) for v in list(h.drive_voltage) +
                              list(h.steer_voltage)),
                current=tuple(float(v) for v in list(h.drive_current) +
                              list(h.steer_current)),
                temperature=tuple(float(v) for v in list(h.drive_temperature) +
                                  list(h.steer_temperature)),
                duty=tuple(float(v) for v in list(h.drive_duty) +
                           list(h.steer_duty)),
                stamp=monotonic())
        except Exception:                         # health shape differs: skip
            pass
