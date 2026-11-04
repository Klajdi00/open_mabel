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

from dataclasses import dataclass

from .constants import TICK_TO_RAD, apply_offset, remove_offset


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
        r0, r1 = self._endpoints()
        if self.angle_max == self.angle_min:
            return remove_offset(self.center, self.offset)
        t = (angle - self.angle_min) / (self.angle_max - self.angle_min)
        t = max(0.0, min(1.0, t))
        corrected = int(round(r0 + t * (r1 - r0)))
        corrected = max(self.raw_lo, min(self.raw_hi, corrected))
        return remove_offset(corrected, self.offset)

    def raw_to_angle(self, raw: int) -> float:
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
        e = load / 1000.0          # raw load is -1000..1000
        return -e if self.invert else e

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
        lo, hi = min(self.angle_min, self.angle_max), max(self.angle_min, self.angle_max)
        return max(lo, min(hi, angle))
