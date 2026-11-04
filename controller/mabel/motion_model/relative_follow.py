"""Relative operator-motion follower; no stored robot position error to repay.

The same follower runs in Retarget Lab and its real-robot adapter. Wheel pose
still supplies the body orientation; it never supplies a position to return to.
"""
import json
import math
import os
from pathlib import Path

import numpy as np

# A source interval at or past this is a tracking STALL: rebase, never replay what was missed. Shorter gaps are
# Wi-Fi delivery: the Vision Pro captures at 60 Hz (16.7 ms) but NOVA receives it in bursts and solves the newest
# frame only, so consecutive SOLVED frames are 51 ms apart at p50, 150 ms at p95 and 233 ms at p99 (auto takes
# 2026-10-27_13-50-19 / 14-02-06). The old 0.1 s threshold rebased 2.3-2.5 times a second there: each one zeroed the
# follower's velocity and dropped the motion in that gap, so the base command came out as pulses and under-followed.
# The interval is the CAPTURE-clock difference (engine _frame_dt), so delta/dt over such a gap is the operator's
# true velocity, not a catch-up. MABEL_FOLLOW_STALL_S restores the old rule at 0.1.
STALL_S = float(os.environ.get("MABEL_FOLLOW_STALL_S", "0.25"))


def identified_response(path, fallback=.22):
    """Use measured delay + time constant as a conservative smoothing horizon.

    Candidate gains/deadbands are deliberately not enabled as motor feedforward.
    Their units and operating range must be validated on the loaded base first.
    """
    result = {"response_s": float(fallback), "source": None, "status": "fallback"}
    try:
        source = Path(path)
        record = json.loads(source.read_text())
        parameters = [record["motors"][name]["parameters"] for name in ("fl", "fr", "b")]
        values = [(float(p["tau"]), float(p["delay"])) for p in parameters]
        if not all(math.isfinite(t) and math.isfinite(d) and t > 0 and d >= 0 for t, d in values):
            raise ValueError("invalid identified response")
        horizon = max(t + d for t, d in values)
        if not .02 <= horizon <= 1.:
            raise ValueError("identified response outside supported time range")
        # The plant already contributes this delay. Use half its dominant
        # horizon for input smoothing rather than adding the entire lag again.
        result.update(response_s=.5*horizon, plant_response_s=horizon,
                      source=str(source), status=record.get("status", "unknown"))
    except (OSError, ValueError, TypeError, KeyError):
        pass
    return result


class RelativeBaseFollow:
    """Differentiate accepted operator motion, then smoothly follow its velocity.

    A small spatial hysteresis rejects tracking jitter without a velocity
    threshold that would discard slow deliberate movement. A tracking gap
    rebases the input; missed motion never becomes a catch-up command.
    """
    def __init__(self, response_s=.22, position_band=.008, yaw_band=math.radians(1.),
                 vmax=.8, wmax=1.5):
        if not all(math.isfinite(v) and v > 0 for v in (response_s, position_band, yaw_band, vmax, wmax)):
            raise ValueError("follow parameters must be finite and positive")
        self.response_s, self.position_band, self.yaw_band = response_s, position_band, yaw_band
        self.vmax, self.wmax = vmax, wmax
        self.reset()

    def reset(self, position=None, yaw=None):
        self.position = None if position is None else np.asarray(position, float).copy()
        self.yaw = None if yaw is None else float(yaw)
        self.velocity = np.zeros(3)
        self.translation_active = False
        self.rotation_active = False

    @staticmethod
    def _wrap(angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def step(self, position, yaw, measured_yaw, dt):
        p = np.asarray(position, float)
        valid = (p.shape == (2,) and np.all(np.isfinite(p))
                 and math.isfinite(measured_yaw) and math.isfinite(dt) and dt > 0
                 and (yaw is None or math.isfinite(yaw)))
        if not valid:
            self.reset()
            return (0., 0., 0.)
        # Rebase at a stall (STALL_S) instead of replaying movement accumulated during it.
        if self.position is None or dt >= STALL_S:
            self.reset(p, yaw)
            return (0., 0., 0.)
        delta = p - self.position
        distance = float(np.linalg.norm(delta))
        accepted = delta * max(0., 1. - self.position_band / max(distance, 1e-15))
        self.position += accepted
        self.translation_active = float(np.linalg.norm(accepted)) > 1e-12
        angular = 0.
        if yaw is not None:
            if self.yaw is None:
                self.yaw = float(yaw)
            difference = self._wrap(yaw - self.yaw)
            angular = math.copysign(max(0., abs(difference) - self.yaw_band), difference)
            self.yaw += angular
        self.rotation_active = abs(angular) > 1e-12
        requested = np.r_[accepted / dt, angular / dt]
        speed = float(np.linalg.norm(requested[:2]))
        if speed > self.vmax:
            requested[:2] *= self.vmax / speed
        requested[2] = np.clip(requested[2], -self.wmax, self.wmax)
        alpha = -math.expm1(-dt / self.response_s)
        self.velocity += alpha * (requested - self.velocity)
        if np.linalg.norm(self.velocity[:2]) < 1e-4:
            self.velocity[:2] = 0.
        if abs(self.velocity[2]) < 1e-4:
            self.velocity[2] = 0.
        # Translate the world-relative operator velocity into the LIVE wheel
        # heading. Actual wheel translation is not an error-feedback target.
        c, s = math.cos(measured_yaw), math.sin(measured_yaw)
        x, y, w = self.velocity
        return (float(c*x + s*y), float(-s*x + c*y), float(w))
