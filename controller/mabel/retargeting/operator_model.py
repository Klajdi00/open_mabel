"""The operator's rigid anthropometry — the constants both estimators need.

The torso frame's ORIGIN is the mid-shoulder root (mu_o of eq. 17) and its axes
are (right, up, forward). In it the operator is three fixed vectors:

    r^{sigma_i,o}   shoulder centres,     (+-half_width, 0, 0)
    p_d^o           the DEVICE (eyes),    (0, +neck_drop, +eye_fwd)
    L_o             shoulder->wrist arm length

The local torso fit needs only their DIFFERENCE, `c_i = r^{sigma_i,o} - p_d^o`,
because a point with torso coordinates x sits at device coordinates
`R_do x + t_do`, and the device origin's own torso coordinates p_d^o pin
`t_do = -R_do p_d^o`. The translation therefore cancels and a shoulder lands at

    s_i^d = R_do (r^{sigma_i,o} - p_d^o) = R_do c_i

— a pure rotation of a known constant. That is what collapses eq. (14) from a
6-DOF SE(3) search to a 3-DOF rotation fit (see local_frame.py).

BOOTSTRAP WITHOUT A PROFILE, AND WHY IT IS DEVICE-FRAME
-------------------------------------------------------
The existing ShoulderEstimator sizes the operator from the HMD's HEIGHT — an
F_a-referenced, floor-relative quantity. Local mode may not use it (a headset in
a car has no trustworthy floor), so this model bootstraps from the running max
of |wrist - device|, which is measured entirely in F_d. At full forward
extension that distance is ~0.336 H for Winter's proportions, within a percent
of the arm-length ratio itself, so one running max sizes the whole skeleton.

A measured OperatorProfile (the app's arm-extension calibration) always wins;
the bootstrap only covers the uncalibrated case so the mode is never unusable.
"""
from __future__ import annotations

import numpy as np

# Winter (2009) segment proportions, as fractions of stature H. The first four
# mirror teleop_engine.hyperparams.ANTHRO_* exactly (same source, same values).
ANTHRO_ARM_RATIO = 0.332           # shoulder->wrist / H
ANTHRO_BIACROMIAL = 0.259          # shoulder width / H
ANTHRO_ACROMION_BELOW_EYE = 0.118  # eye line -> acromion vertical drop / H
ANTHRO_UPPER_ARM_RATIO = 0.186     # shoulder->elbow / H
ANTHRO_FOREARM_RATIO = 0.146       # elbow->wrist / H  (0.186 + 0.146 = 0.332)
# Eye line FORWARD of the acromion line / H. Not a Winter number — an
# approximation of the head's anterior offset (~8 cm at H = 1.75). It only sets
# where the fit thinks the device sits on the torso, so an error here reads as a
# small constant pitch bias, absorbed by the mode-entry alignment latch.
ANTHRO_EYE_FWD_OF_ACROMION = 0.045
# |wrist - device| at full forward extension / H — the device-frame bootstrap
# scale. Derived from the four ratios above, checked in test_retarget_modes.py.
ANTHRO_DEVICE_REACH_RATIO = 0.336

_H_MIN, _H_MAX = 1.40, 2.10        # plausible stature band [m]
_L_MIN, _L_MAX = 0.30, 0.90        # plausible shoulder->wrist band [m]
_MED_N = 5                         # samples in the spike-rejecting median


class OperatorModel:
    """Rigid operator anthropometry, measured or bootstrapped.

    Values from a calibrated OperatorProfile are FROZEN and never overwritten by
    the live bootstrap; anything the profile leaves unset keeps estimating.
    """

    def __init__(self, stature=1.75):
        self._prof = {}
        self.reset(stature)

    def reset(self, stature=1.75):
        self._stature = float(np.clip(stature, _H_MIN, _H_MAX))
        self._buf = {"left": [], "right": []}
        self._fbuf = {"left": [], "right": []}
        self._forearm = {"left": None, "right": None}
        self._reach_max = None          # running max |wrist - device| [m]
        return self

    # ── measured anthropometry ────────────────────────────────────────────────
    def apply_profile(self, profile):
        """Freeze whatever a calibration OperatorProfile actually measured.

        Accepts the package's OperatorProfile (whole_body_retargeting.
        calibration) or any object/dict exposing the same field names, so the
        engine can hand across either without a conversion shim.
        """
        if profile is None:
            return self

        def get(name):
            if isinstance(profile, dict):
                return profile.get(name)
            return getattr(profile, name, None)

        lr = get("arm_length_lr") or (0.0, 0.0)
        prof = {}
        # arm_length_lr is (left, right) — the same order calibration.py stores.
        for side, v in (("left", lr[0]), ("right", lr[1] if len(lr) > 1 else 0.0)):
            if v and float(v) > 1e-3:
                prof[f"L_{side}"] = float(np.clip(float(v), _L_MIN, _L_MAX))
        mean_L = get("arm_length_m")
        if mean_L and float(mean_L) > 1e-3:
            prof.setdefault("L_left", float(np.clip(float(mean_L), _L_MIN, _L_MAX)))
            prof.setdefault("L_right", float(np.clip(float(mean_L), _L_MIN, _L_MAX)))
        for key, field in (("upper_arm", "upper_arm_m"), ("forearm", "forearm_m")):
            v = get(field)
            if v and float(v) > 1e-3:
                prof[key] = float(v)
        w = get("shoulder_width_m")
        if w and float(w) > 1e-3:
            prof["half_width"] = 0.5 * float(w)
        d = get("neck_height_m")
        if d and float(d) > 1e-3:
            prof["neck_drop"] = float(d)
        h = get("height_m")
        if h and float(h) > 1e-3:
            self._stature = float(np.clip(float(h), _H_MIN, _H_MAX))
        self._prof = prof
        return self

    # ── live bootstrap (device frame only) ────────────────────────────────────
    def observe(self, dev):
        """Update the stature bootstrap from one DeviceMeasurements frame.

        TWO estimators, in order of preference:

        1. FOREARM LENGTH |wrist - elbow|. A rigid segment, so it is observable
           at EVERY arm configuration and its right estimator is a robust
           CENTRAL one — a running median. This is the good path.
        2. MAX |wrist - device|, the running max of a short median. Only correct
           once the operator has reached out fully, so it under-sizes them until
           they do (measured: a session that never exceeds rho = 0.6 sizes a
           1.75 m operator at 1.51 m). It is the fallback for when the skeleton
           carries no forearm joint.

        The median in (1) rejects the single-frame ARKit spikes that would
        otherwise bias the estimate; the max in (2) is a max precisely because
        a partial reach is a lower bound on the true reach, never an estimate
        of it.
        """
        for side in ("left", "right"):
            w = dev.wrist_pos(side)
            if w is None:
                continue
            e = dev.elbow_pos(side)
            if e is not None:
                f = float(np.linalg.norm(w - e))
                if 0.15 < f < 0.40:            # plausible forearm band [m]
                    buf = self._fbuf[side]
                    buf.append(f)
                    if len(buf) > _MED_N:
                        buf.pop(0)
                    if len(buf) >= 3:
                        self._forearm[side] = float(np.median(buf))
            r = float(np.linalg.norm(w))       # device is the origin of F_d
            if not (0.15 < r < 1.20):          # implausible: drop it entirely
                continue
            buf = self._buf[side]
            buf.append(r)
            if len(buf) > _MED_N:
                buf.pop(0)
            if len(buf) == _MED_N:
                med = float(np.median(buf))
                self._reach_max = med if self._reach_max is None \
                    else max(self._reach_max, med)

        if "L_left" in self._prof:             # measured profile wins outright
            return self
        f = [v for v in self._forearm.values() if v]
        if f:
            self._stature = float(np.clip(
                float(np.mean(f)) / ANTHRO_FOREARM_RATIO, _H_MIN, _H_MAX))
        elif self._reach_max is not None:
            self._stature = float(np.clip(
                self._reach_max / ANTHRO_DEVICE_REACH_RATIO, _H_MIN, _H_MAX))
        return self

    # ── queries ───────────────────────────────────────────────────────────────
    @property
    def stature(self):
        return self._stature

    def arm_length(self, side="right"):
        """L_o for one side — shoulder->wrist at full extension [m]."""
        v = self._prof.get(f"L_{side}")
        if v:
            return float(v)
        return float(np.clip(ANTHRO_ARM_RATIO * self._stature, _L_MIN, _L_MAX))

    def upper_arm(self, side="right"):
        """Shoulder->elbow [m]. A RIGID segment, which is why the local fit can
        constrain the torso from the elbow at any arm configuration, unlike the
        wrist, which only reaches L_o when the arm is straight."""
        v = self._prof.get("upper_arm")
        if v:
            return float(v)
        return float(ANTHRO_UPPER_ARM_RATIO * self._stature)

    def forearm(self, side="right"):
        """Elbow->wrist [m], measured live when the skeleton carries an elbow."""
        v = self._prof.get("forearm") or self._forearm.get(side)
        if v:
            return float(v)
        return float(ANTHRO_FOREARM_RATIO * self._stature)

    @property
    def half_width(self):
        return float(self._prof.get("half_width")
                     or 0.5 * ANTHRO_BIACROMIAL * self._stature)

    @property
    def neck_drop(self):
        return float(self._prof.get("neck_drop")
                     or ANTHRO_ACROMION_BELOW_EYE * self._stature)

    @property
    def eye_fwd(self):
        return float(ANTHRO_EYE_FWD_OF_ACROMION * self._stature)

    def shoulder_body(self, side):
        """r^{sigma_i,o} — shoulder centre in TORSO coordinates (origin at the
        mid-shoulder root, axes right/up/forward)."""
        s = 1.0 if side == "right" else -1.0
        return np.array([s * self.half_width, 0.0, 0.0])

    def device_body(self):
        """p_d^o — the device (eye) position in TORSO coordinates."""
        return np.array([0.0, self.neck_drop, self.eye_fwd])

    def shoulder_offsets_device(self):
        """{side: c_i} with c_i = r^{sigma_i,o} - p_d^o — the constants the
        local torso fit rotates (see the module docstring)."""
        p_d = self.device_body()
        return {side: self.shoulder_body(side) - p_d
                for side in ("left", "right")}
