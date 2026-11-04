"""neck.py — MABEL's neck: a SERIAL direct-drive chain, task space == motor space.

The as-built neck is three serial revolute joints — yaw (DaMiao, at the base) ->
pitch (Dynamixel) -> roll (Dynamixel, carries the camera) — so there is no linkage
to invert: the task angle of an axis IS the calibrated angle of the motor that
drives it (level head, camera looking straight forward == 0 on every axis, via the
neutral-anchored calibration). Only the wired joints exist; ``neck_yaw`` appears
once its DaMiao is connected (config ``joints``). Because there is no closed-form
model to depend on, task space is ALWAYS available (contrast the old RSSR neck,
whose pitch/roll needed the neck_studio model).

Everything below the calibration seam is the plain :class:`JointGroupSubsystem`
(raw motor radians, lease-gated writes, hold-in-place stop); this class only adds
the task-space (pitch/roll/yaw) surface the DDS body adapter and the facade's
NeckWriter speak.
"""
from __future__ import annotations

from typing import Optional

from ..calibration import CalibrationStore
from ..drivers.seam import JointGroupDriver
from ..types import NeckState
from .joint_group import JointGroupSubsystem

__all__ = ["NeckSubsystem"]

#: task axis -> the motor joint that drives it 1:1 (direct drive).
_AXES = {"pitch": "neck_pitch", "roll": "neck_roll", "yaw": "neck_yaw"}

#: fallback clamp (rad, ~80 deg) per axis when the calib slice has no ``_task``
#: block yet — keeps an uncalibrated bench neck from being driven to a hard stop.
_DEFAULT_LIMIT = 1.396


class NeckSubsystem(JointGroupSubsystem):
    """Serial direct-drive neck (pitch/roll[/yaw]). Task == motor space."""

    def __init__(self, name: str, cfg: dict, links: dict, *,
                 store: Optional[CalibrationStore] = None,
                 driver: Optional[JointGroupDriver] = None, logger=None):
        super().__init__(name, cfg, links, store=store, driver=driver,
                         logger=logger)
        self._task_limits: Optional[dict] = None
        if store is not None and self._slice:
            self._task_limits = store.task_limits(self._slice)
        self._log(f"[{self.name}] serial direct-drive neck (task==motor): "
                  f"{', '.join(self.names) or '(no joints)'}")

    @property
    def task_space_available(self) -> bool:
        return True

    def recalibrate(self, store: CalibrationStore) -> None:
        super().recalibrate(store)
        if self._slice:
            self._task_limits = store.task_limits(self._slice)

    def _clamp(self, axis: str, val: float) -> float:
        lim = self._task_limits or {}
        lo = lim.get(f"{axis}_min", -_DEFAULT_LIMIT)
        hi = lim.get(f"{axis}_max", _DEFAULT_LIMIT)
        return max(lo, min(hi, float(val)))

    # ── task-space surface ────────────────────────────────────────────────────
    def state(self) -> NeckState:                  # type: ignore[override]
        """Motor space (calibrated) plus its task-space alias. A joint not wired
        (e.g. neck_yaw before its DaMiao) reports None so the readout is honestly
        absent rather than a fabricated 0."""
        motors = super().state()
        by = dict(zip(motors.names, motors.position))
        return NeckState(motors=motors, pitch=by.get("neck_pitch"),
                         roll=by.get("neck_roll"), yaw=by.get("neck_yaw"),
                         stamp=motors.stamp)

    def set_gaze(self, pitch: float, roll: float, yaw: float) -> None:
        """Task-space target (rad) -> the motor that drives each axis 1:1, clamped
        to the calibrated ``_task`` limits. Axes whose motor is not wired are
        ignored (no InfeasiblePose — a serial chain has no feasibility wall).

        Axes whose motor has NO calibration entry are ALSO ignored: below the
        seam an uncalibrated joint passes RAW servo radians through, so a task
        "0 = level head" would drive the motor to its arbitrary encoder zero —
        the "neck snapped to a random position" on whole-body engage
        (2026-07-21, no ``neck/serial`` slice in calibration_data). Hold still
        and say why, once, instead."""
        vals = {"pitch": pitch, "roll": roll, "yaw": yaw}
        targets = {}
        uncal = []
        for axis, motor in _AXES.items():
            if motor not in self.names:
                continue
            if motor not in self._calib:
                uncal.append(motor)
                continue
            targets[motor] = self._clamp(axis, vals[axis])
        if uncal and not getattr(self, "_gaze_uncal_warned", False):
            self._gaze_uncal_warned = True
            self._log(f"[{self.name}] gaze REFUSED for uncalibrated {', '.join(uncal)} "
                      f"(no entry in slice {self._slice!r}) — Setup Studio ▸ Neck ▸ "
                      f"Set Level writes it; holding until then.")
        if targets:
            self.set_targets(targets)
