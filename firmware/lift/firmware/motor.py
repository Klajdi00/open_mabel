# motor.py - BTS7960 H-bridge driver.
#
# Exposes a single signed-effort interface so the controller never has to
# think about which PWM channel maps to which direction. effort is in
# [-1.0, +1.0]: positive drives UP (RPWM), negative drives DOWN (LPWM).

from machine import ADC, Pin, PWM
import utime

import config


class Motor:
    def __init__(self):
        self._rpwm = PWM(Pin(config.RPWM_PIN))
        self._lpwm = PWM(Pin(config.LPWM_PIN))
        self._rpwm.freq(config.MOTOR_PWM_FREQ)
        self._lpwm.freq(config.MOTOR_PWM_FREQ)

        self._r_en = Pin(config.R_EN_PIN, Pin.OUT)
        self._l_en = Pin(config.L_EN_PIN, Pin.OUT)

        self._effort = 0.0

        # Current sense on the BTS7960's IS pin. Reads the DRIVING half-bridge only,
        # so with R_IS alone this is an upward-motion measurement (see config).
        self._adc = None           # R_IS — the UP half-bridge
        self._adc_dn = None        # L_IS — the DOWN half-bridge, if wired
        self._adc_raw = 0.0        # EMA-filtered raw count, whichever channel is live
        self._adc_zero = None      # count with the bridge idle (offset)
        self._adc_zero_dn = None
        if config.CURRENT_SENSE:
            try:
                self._adc = ADC(Pin(config.CURRENT_SENSE_PIN))
            except Exception:
                self._adc = None
            if getattr(config, "CURRENT_SENSE_PIN_DOWN", None) is not None:
                try:
                    self._adc_dn = ADC(Pin(config.CURRENT_SENSE_PIN_DOWN))
                except Exception:
                    self._adc_dn = None

        self.enable()
        self.coast()

    def enable(self):
        self._r_en.value(1)
        self._l_en.value(1)

    def disable(self):
        self.coast()
        self._r_en.value(0)
        self._l_en.value(0)

    @property
    def effort(self):
        return self._effort

    # ── current sense ────────────────────────────────────────────────────────
    def sample_current(self):
        """Read and filter the IS pin. Call once per control tick.

        While the bridge is idle the reading is captured as the zero offset, so drift
        in the sense network is subtracted rather than reported as load.
        """
        # Read whichever half-bridge is actually driving: R_IS going up, L_IS going
        # down. Each keeps its own idle offset, since the two networks differ.
        down = self._effort < 0.0
        adc = self._adc_dn if (down and self._adc_dn is not None) else self._adc
        if adc is None:
            return
        raw = adc.read_u16()
        a = config.CURRENT_FILTER_ALPHA
        self._adc_raw += a * (raw - self._adc_raw)
        if self._effort == 0.0:
            # Idle: track both offsets so either channel is ready when it takes over.
            if self._adc_zero is None:
                self._adc_zero = self._adc_raw
            else:
                self._adc_zero += 0.02 * (self._adc_raw - self._adc_zero)
            if self._adc_dn is not None:
                r2 = self._adc_dn.read_u16()
                if self._adc_zero_dn is None:
                    self._adc_zero_dn = r2
                else:
                    self._adc_zero_dn += 0.02 * (r2 - self._adc_zero_dn)

    @property
    def adc_raw(self):
        return int(self._adc_raw)

    @property
    def adc_zero(self):
        return None if self._adc_zero is None else int(self._adc_zero)

    def baseline_amps(self, duty):
        """Unloaded current at this duty, linearly interpolated from the measured
        table. Returns None outside the table's range, where it would be a guess."""
        t = config.CURRENT_BASE_UP
        d = abs(duty)
        if not t or d < t[0][0] or d > t[-1][0]:
            return None
        for i in range(len(t) - 1):
            d0, a0 = t[i]
            d1, a1 = t[i + 1]
            if d0 <= d <= d1:
                if d1 == d0:
                    return a0
                return a0 + (a1 - a0) * (d - d0) / (d1 - d0)
        return None

    @property
    def load_amps(self):
        """EXTRA current beyond the unloaded baseline at the present duty.

        This is the quantity that answers "how much external load is on the column,
        and how has it changed" — a deviation, so it needs no absolute force
        calibration and is unaffected by the sense resistor's true value. Positive
        means more load than the bare column. None when it cannot be computed
        (no sense, driving DOWN where R_IS is idle, or duty outside the table).
        """
        a = self.amps
        if a is None or self._effort == 0.0:
            return None
        if self._effort < 0.0 and self._adc_dn is None:
            return None                 # R_IS alone cannot see the DOWN half-bridge
        base = self.baseline_amps(self._effort)
        if base is None:
            return None
        return a - base

    @property
    def load_newtons(self):
        """load_amps scaled to newtons, once LOAD_N_PER_A has been calibrated."""
        la = self.load_amps
        if la is None or config.LOAD_N_PER_A is None:
            return None
        return la * config.LOAD_N_PER_A

    @property
    def amps(self):
        """Load current in amps, or None if it cannot be computed.

        I_load = (V_is / R_sense) * ratio, with the idle offset removed. Returns None
        when no sense resistor value is configured — the raw count is still published
        and works as a relative load signal.
        """
        if self._adc is None or not config.CURRENT_SENSE_R:
            return None
        if self._effort < 0.0 and self._adc_dn is not None:
            zero = 0.0 if self._adc_zero_dn is None else self._adc_zero_dn
        else:
            zero = 0.0 if self._adc_zero is None else self._adc_zero
        counts = self._adc_raw - zero
        if counts < 0:
            counts = 0.0
        # DUTY CORRECTION. The IS pin follows the PWM: it mirrors the load current
        # during the ON phase and sits near zero during OFF. The ADC is not
        # synchronised to the 20 kHz carrier — the raw signal swings rail to rail
        # (measured 576..65535) — so the filtered mean is duty x current, not current.
        # Dividing by duty recovers the actual conducted current.
        #
        # This matters: without it the "current vs duty" curve is contaminated by the
        # duty itself, which is what made the first baseline table rise so steeply.
        # Corrected, the low-duty points collapse to a near-constant value, which is
        # the physically sensible picture for a weight-dominated load.
        duty = abs(self._effort)
        if duty < 0.05:
            return 0.0
        counts = counts / duty
        v = (counts / 65535.0) * config.ADC_VREF
        return (v / config.CURRENT_SENSE_R) * config.CURRENT_SENSE_RATIO

    @property
    def adc_load_counts(self):
        """Duty-corrected load in raw ADC counts — the scale-free load signal."""
        if self._adc is None:
            return None
        duty = abs(self._effort)
        if duty < 0.05:
            return 0.0
        if self._effort < 0.0 and self._adc_dn is not None:
            zero = 0.0 if self._adc_zero_dn is None else self._adc_zero_dn
        else:
            zero = 0.0 if self._adc_zero is None else self._adc_zero
        c = (self._adc_raw - zero) / duty
        return c if c > 0 else 0.0

    def coast(self):
        """Stop driving (both channels low)."""
        self._rpwm.duty_u16(0)
        self._lpwm.duty_u16(0)
        self._effort = 0.0

    def set_effort(self, effort):
        """Drive the motor. effort in [-1, 1]; +up / -down.

        Reverses include a short break-before-make dead time so we never
        shoot through the H-bridge.
        """
        # Clamp to the configured ceiling.
        if effort > config.MAX_EFFORT:
            effort = config.MAX_EFFORT
        elif effort < -config.MAX_EFFORT:
            effort = -config.MAX_EFFORT

        # Insert dead time if the sign flips while we were driving.
        if (effort > 0 and self._effort < 0) or (effort < 0 and self._effort > 0):
            self._rpwm.duty_u16(0)
            self._lpwm.duty_u16(0)
            utime.sleep_ms(config.DEADTIME_MS)

        duty = int(65535 * abs(effort))
        if effort > 0:          # UP
            self._lpwm.duty_u16(0)
            self._rpwm.duty_u16(duty)
        elif effort < 0:        # DOWN
            self._rpwm.duty_u16(0)
            self._lpwm.duty_u16(duty)
        else:
            self._rpwm.duty_u16(0)
            self._lpwm.duty_u16(0)

        self._effort = effort
