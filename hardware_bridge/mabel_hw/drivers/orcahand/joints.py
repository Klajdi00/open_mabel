"""Per-joint kinematic mapping: joint angle [rad] <-> motor position [ticks].

A :class:`Joint` couples a joint's *logical* description (name, angle range,
inversion) with its *measured* calibration (the raw tick end-stops swept during
calibration). It is the bridge between the SI world the controller/ROS speak
and the raw ticks the :class:`~orcahand.bus.FeetechBus` speaks.

The mapping is a simple affine fit between two intervals:

    angle in [angle_min, angle_max]   <->   raw in [raw_lo, raw_hi]

where (raw_lo, raw_hi) = (raw_min, raw_max), swapped if the joint is inverted
(motor turns the opposite way to the joint). Everything is clamped to the
calibrated tick range so a bad command can never drive a tendon past its
hardware end-stop.

Single-turn wrap: the encoder rolls over at 4095->0, so a joint whose travel
crosses that seam would read discontinuously. Each joint carries a software
``offset`` that relocates the seam: the *corrected* tick is
``(raw - offset) mod 4096`` and the calibrated ``raw_min/raw_max/center`` live in
that corrected frame. ``offset`` is 0 for the common case (range fits in 0-4095
without crossing the seam), in which case the mapping is identical to the legacy
raw mapping; only a seam-straddling joint gets a non-zero offset. The correction
is applied *inside* the methods below, so callers (the bus, ``OrcaHand``, ROS)
only ever deal in real raw encoder ticks and radians -- the wrap is invisible.
"""

from __future__ import annotations

import math

from dataclasses import dataclass
from typing import Optional

from . import constants as C
from .constants import TICK_TO_RAD, apply_offset, remove_offset

_TWO_PI = 2.0 * math.pi


def _wrap_pi(x: float) -> float:
    """Fold an angle into [-π, π) — the single-turn seam handler the upstream
    studio uses (hand_bridge._calibrated_proto). Keeps a joint whose swept
    window straddles the 0/2π encoder seam continuous."""
    return (x + math.pi) % _TWO_PI - math.pi


@dataclass
class Joint:
    name: str            # bare joint name, e.g. "index_mcp"
    motor_id: int        # Feetech ID on the bus
    angle_min: float     # rad at the raw MIN tick (before inversion)
    angle_max: float     # rad at the raw MAX tick (before inversion)
    invert: bool         # motor drives the joint the opposite way
    raw_min: int         # calibrated tick end-stop (corrected frame)
    raw_max: int         # calibrated tick end-stop (corrected frame)
    center: int          # calibrated neutral tick (corrected frame)
    offset: int = 0      # software wrap offset: corrected = (raw - offset) % 4096

    # ---- studio unit-gain calibration (mabel.json hand/orca-<side>/calib) ----
    # When present these OVERRIDE the legacy ROM map above: the joint angle is the
    # STUDIO's model-anchored unit-gain map (1 rad encoder = 1 rad joint), so the
    # driver renders identically to the setup studio's 3D viewport. This is the
    # single-master path — the numbers come straight from the file the studio
    # writes. See raw_to_angle / angle_to_raw. All three must be set together.
    model_mid: Optional[float] = None     # joint angle [rad] at the swept centre
    raw_min_rad: Optional[float] = None   # swept end-stop [rad] (= tick·TICK_TO_RAD)
    raw_max_rad: Optional[float] = None   # swept end-stop [rad]; raw_max_rad > raw_min_rad

    @property
    def _has_model_calib(self) -> bool:
        return (self.model_mid is not None and self.raw_min_rad is not None
                and self.raw_max_rad is not None
                and self.raw_max_rad > self.raw_min_rad)

    def _model_frame(self) -> tuple:
        """(raw-mid [rad], half-span [rad], direction ±1) for the studio map."""
        rmid = 0.5 * (self.raw_min_rad + self.raw_max_rad)
        half = 0.5 * (self.raw_max_rad - self.raw_min_rad)
        d = -1.0 if self.invert else 1.0
        return rmid, half, d

    # raw tick endpoints, swapped if this joint is mechanically inverted
    def _endpoints(self) -> tuple:
        return (self.raw_max, self.raw_min) if self.invert else (self.raw_min, self.raw_max)

    @property
    def raw_lo(self) -> int:
        return min(self.raw_min, self.raw_max)

    @property
    def raw_hi(self) -> int:
        return max(self.raw_min, self.raw_max)

    # ------------------------------------------------------------- angle <-> raw
    def angle_to_raw(self, angle: float) -> int:
        """Clamp ``angle`` [rad] to range and map to a real encoder tick."""
        if self._has_model_calib:
            # STUDIO unit-gain inverse (hand_bridge._calib_target_ticks): walk the
            # centre-anchor back, bounding the excursion to ±half-span so a goal can
            # never be commanded past a measured end-stop.
            rmid, half, d = self._model_frame()
            delta = max(-half, min(half, d * (angle - self.model_mid)))
            raw = (rmid + delta) % _TWO_PI
            return int(round(raw / TICK_TO_RAD)) % C.WRAP
        r0, r1 = self._endpoints()
        if self.angle_max == self.angle_min:
            return remove_offset(self.center, self.offset)
        t = (angle - self.angle_min) / (self.angle_max - self.angle_min)
        t = max(0.0, min(1.0, t))
        corrected = int(round(r0 + t * (r1 - r0)))
        corrected = max(self.raw_lo, min(self.raw_hi, corrected))
        return remove_offset(corrected, self.offset)

    def raw_to_angle(self, raw: int) -> float:
        if self._has_model_calib:
            # STUDIO unit-gain forward map (hand_bridge._calibrated_proto):
            # angle = model_mid + dir·wrap_π(raw_rad − raw_mid).
            rmid, _half, d = self._model_frame()
            return self.model_mid + d * _wrap_pi(raw * TICK_TO_RAD - rmid)
        r0, r1 = self._endpoints()
        if r1 == r0:
            return 0.0
        corrected = apply_offset(raw, self.offset)
        t = (corrected - r0) / (r1 - r0)
        return self.angle_min + t * (self.angle_max - self.angle_min)

    # --------------------------------------------------------------- derivatives
    def speed_to_rad_s(self, speed_ticks: float) -> float:
        v = speed_ticks * TICK_TO_RAD
        return -v if self.invert else v

    def load_to_effort(self, load: float) -> float:
        """Signed PWM duty as a fraction [-1, 1]. This is the servo's DRIVE
        EFFORT, not a torque: it saturates at ±1 on a stall regardless of supply
        voltage, so it can never be scaled to N·m. Use `current_to_torque` for a
        real torque; this stays for backwards compatibility and for spotting
        saturation (|effort| → 1 means the motor is giving all it has)."""
        e = load / 1000.0          # raw load is -1000..1000
        return -e if self.invert else e

    def current_to_torque(self, current_a: float,
                          nm_per_amp: float = C.TORQUE_NM_PER_AMP) -> float:
        """MEASURED joint torque in N·m: τ = Kt · I, in the joint's own direction.

        `current_a` is the servo's Present_Current (already in amps). The sign
        follows the same `invert` flip as position/velocity, so a torque that
        opposes a positive joint motion reads negative — consistent with the arms
        (subsystems/joint_group.state: `eff = dir * tau`) and with ROS's JointState.effort
        contract."""
        t = current_a * nm_per_amp
        return -t if self.invert else t

    # ----------------------------------------------------------------- percent
    def percent_to_raw(self, pct: float) -> int:
        """0-100 % across the calibrated range (0 = raw_min, 100 = raw_max).

        Percent ignores inversion -- it is a convenience scale over the physical
        tick range, matching the legacy firmware ``pct_to_raw`` convention.
        Returns a real encoder tick (offset re-applied).
        """
        pct = max(0.0, min(100.0, pct))
        corrected = int(round(self.raw_min + (pct / 100.0) * (self.raw_max - self.raw_min)))
        return remove_offset(corrected, self.offset)

    def raw_to_percent(self, raw: int) -> float:
        span = self.raw_max - self.raw_min
        if span == 0:
            return 0.0
        corrected = apply_offset(raw, self.offset)
        return max(0.0, min(100.0, (corrected - self.raw_min) / span * 100.0))

    def clamp_angle(self, angle: float) -> float:
        if self._has_model_calib:
            # Reachable set of the studio map is model_mid ± half-span (dir just
            # flips direction, so the bound is symmetric and dir-independent).
            _rmid, half, _d = self._model_frame()
            return max(self.model_mid - half, min(self.model_mid + half, angle))
        lo, hi = min(self.angle_min, self.angle_max), max(self.angle_min, self.angle_max)
        return max(lo, min(hi, angle))
