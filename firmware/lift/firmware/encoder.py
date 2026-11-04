# encoder.py - Absolute quadrature position from the encoder only.
#
# The interrupt handler is a plain MODULE-LEVEL function on global ints (never
# a bound method — that faults on the RP2040). On each rising edge of phase A
# it reads phase B to decide the REAL direction of travel, so the count is an
# absolute measure of shaft motion that does NOT depend on what the motor was
# commanded to do. This eliminates the slow drift you get from counting by
# commanded direction (coasting, back-driving, reversals, controller windup).
#
# A commanded-direction fallback (config.USE_QUADRATURE = False) is kept for
# the case where phase B is not wired.

from machine import Pin
import utime

import config

# --- interrupt state (module-level so the ISR stays a plain function) -------
_a = None                    # phase A pin (set in Encoder.__init__)
_b = None                    # phase B pin
_count = 0
_last_tick_ms = utime.ticks_ms()
_dir = 0                     # commanded direction, used only in fallback mode


def _isr(pin):
    global _count, _last_tick_ms
    if config.USE_QUADRATURE:
        # Real direction from phase B at the moment of the A edge.
        if _b.value():
            _count -= config.ENCODER_SIGN
        else:
            _count += config.ENCODER_SIGN
    else:
        # Fallback: trust the commanded direction (can drift).
        if _dir > 0:
            _count += config.ENCODER_SIGN
        elif _dir < 0:
            _count -= config.ENCODER_SIGN
        else:
            return
    _last_tick_ms = utime.ticks_ms()


class Encoder:
    def __init__(self):
        global _a, _b
        _a = Pin(config.ENC_A_PIN, Pin.IN, Pin.PULL_UP)
        _b = Pin(config.ENC_B_PIN, Pin.IN, Pin.PULL_UP)
        self._offset = 0.0
        # Counts -> mm. Starts at the config default; calibration measures and
        # overwrites it (and it is restored from flash on boot).
        self.mm_per_count = config.MM_PER_COUNT

        # Velocity estimator state.
        self._velocity = 0.0
        self._last_count = 0
        self._last_update_ms = utime.ticks_ms()

        _a.irq(trigger=Pin.IRQ_RISING, handler=_isr)

    # -- direction hint (only used in the fallback, non-quadrature mode) -----
    def set_direction(self, d):
        global _dir
        _dir = d

    # -- stall-detection timestamp -------------------------------------------
    @property
    def last_tick_ms(self):
        return _last_tick_ms

    @last_tick_ms.setter
    def last_tick_ms(self, value):
        global _last_tick_ms
        _last_tick_ms = value

    # -- absolute position ---------------------------------------------------
    @property
    def count(self):
        return _count

    @property
    def position_mm(self):
        return (_count - self._offset) * self.mm_per_count

    def set_position_mm(self, mm):
        """Redefine the CURRENT physical position. Use ONLY for homing/calib."""
        self._offset = _count - (mm / self.mm_per_count)

    def set_reference(self, count, mm):
        """Map a SPECIFIC count value to a height (used at a measured end stop)."""
        self._offset = count - (mm / self.mm_per_count)

    def set_scale(self, mm_per_count):
        """Set the counts->mm scale (from calibration / restored state)."""
        if mm_per_count and mm_per_count > 0:
            self.mm_per_count = mm_per_count

    def zero(self):
        self.set_position_mm(0.0)

    # -- velocity ------------------------------------------------------------
    @property
    def velocity_mm_s(self):
        return self._velocity

    def update(self, now_ms):
        """Refresh the filtered velocity estimate. Call once per control tick."""
        dt = utime.ticks_diff(now_ms, self._last_update_ms) / 1000.0
        if dt <= 0.0:
            return
        dcount = _count - self._last_count
        raw_v = (dcount * self.mm_per_count) / dt
        self._velocity += config.VEL_FILTER_ALPHA * (raw_v - self._velocity)
        self._last_count = _count
        self._last_update_ms = now_ms
