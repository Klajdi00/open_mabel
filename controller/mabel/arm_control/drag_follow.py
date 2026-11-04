"""drag_follow.py — SOFT mode as POSITION-based drag-and-hold (operator design).

THE LAW (operator directive 2026-07-30, replacing the τ-observer admittance that
went erratic on the real arm): soft keeps a NONZERO P — and when the user pushes
the arm, the TARGET simply follows the measured position; when the push stops,
the target FREEZES and is never updated again until the next push:

    pushing   |q − q*| > band  AND  |q̇| > qd_min   →  q* chases q (rate-limited),
                                                       stopping `band` short
    released  (either gate fails)                   →  q* frozen where it is

Why this cannot go erratic: there is NO torque feedback in the loop — the only
input is the measured position/velocity, and the target only ever moves TOWARD
the measured position, never past it (it stops `band` short). The closed loop
"target chases arm" is a pure first-order lag; the arm cannot be excited by its
own command stream.

Why it cannot creep under gravity-model error (the imperfect-sysid defense,
positional edition): a residual torque sags the joint QUASI-STATICALLY — the
spring kp·band arrests it below the velocity gate, so a static joint is a plain
spring-damper hold and the target never walks. Only a real hand moving the joint
faster than `qd_min` opens the follow gate.

Feel: while dragging, resistance ≈ kp_soft·band + kd·q̇ — light and constant;
release → the arm holds EXACTLY where you left it (the target is already there).
The app's stiffness slider scales kp_soft (see the /arms/gains path), which
scales the drag resistance and the hold stiffness together.

Pure logic (no MuJoCo/ROS): the gravity FF keeps flowing separately (plant
_arm_effort), exactly as in stiff. Params LIVE in control_param.json
``arm.soft_drag`` (control_params.arm_soft_drag).
"""
from __future__ import annotations

import math

__all__ = ["DragFollow", "DRAG_DEFAULTS"]

DRAG_DEFAULTS = {
    # rad — the standing offset the target keeps behind the measured position.
    # Sets the drag resistance (kp_soft·band) AND the anti-creep authority: the
    # spring can arrest gravity-residual sag up to kp_soft·band (~10·0.06 =
    # 0.6 N·m at the soft shoulders — at the residual band's edge; raise if a
    # joint creeps).
    "band": 0.06,
    # rad/s — measured speed that counts as "being pushed". Gravity-residual
    # sag is quasi-static and stays below this; a hand drag is far above it.
    "qd_min": 0.10,
    # rad/s — how fast the target may chase the arm (rate limit on q*).
    "follow_rate": 3.0,
    # s — hysteresis: keep following this long after the gates last held
    # (prevents chatter at the gate edges mid-drag).
    "follow_hold_s": 0.15,
    "range_margin": 0.06,     # rad — never park the target on a joint stop
}


class DragFollow:
    """Per-joint drag-and-hold target follower, model frame, actuator-agnostic."""

    def __init__(self, joints, params=None, log=None):
        self.joints = tuple(joints)
        self._log = log or (lambda *a, **k: None)
        p = dict(DRAG_DEFAULTS)
        for k, v in (params or {}).items():
            if not str(k).startswith("_"):
                p[k] = v
        self.p = p
        self._tgt: dict = {}
        self._follow_t: dict = {}     # last time the follow gates held, per joint
        self._engaged = False

    @property
    def engaged(self) -> bool:
        return self._engaged

    def engage(self, q_meas: dict) -> None:
        """Seed every target at the MEASURED pose — soft entry holds in place."""
        self._tgt = {j: float(q_meas[j]) for j in self.joints
                     if q_meas.get(j) is not None and math.isfinite(q_meas[j])}
        self._follow_t.clear()
        self._engaged = True
        self._log(f"soft-drag ENGAGED: {len(self._tgt)} joints hold at measured")

    def disengage(self) -> None:
        if self._engaged:
            self._log("soft-drag disengaged")
        self._engaged = False
        self._tgt.clear()
        self._follow_t.clear()

    def update(self, q_meas: dict, qd_meas: dict, dt: float, now: float,
               limits: dict | None = None) -> dict:
        """One tick → {joint: target} for every seeded joint (a frozen target is
        still returned — soft continuously commands the held pose)."""
        if not self._engaged or dt <= 0.0:
            return {}
        dt = min(float(dt), 0.1)
        band = float(self.p["band"])
        qd_min = float(self.p["qd_min"])
        rate = float(self.p["follow_rate"])
        hold_s = float(self.p["follow_hold_s"])
        out = {}
        for j in self.joints:
            q = q_meas.get(j)
            if q is None or not math.isfinite(q):
                if j in self._tgt:
                    out[j] = self._tgt[j]        # no feedback → hold the target
                continue
            q = float(q)
            t = self._tgt.get(j)
            if t is None:                        # hot-joined joint: seed + hold
                self._tgt[j] = q
                out[j] = q
                continue
            qd = qd_meas.get(j)
            moving = (qd is not None and math.isfinite(qd) and abs(qd) >= qd_min)
            displaced = abs(q - t) > band
            if moving and displaced:
                self._follow_t[j] = now
            if now - self._follow_t.get(j, -1e18) <= hold_s:
                # chase the measured position, stopping `band` short, rate-limited
                want = q - math.copysign(band, q - t) if abs(q - t) > band else t
                step = rate * dt
                d = want - t
                t = want if abs(d) <= step else t + math.copysign(step, d)
            if limits and j in limits:
                lo, hi = limits[j]
                m = float(self.p["range_margin"])
                t = max(lo + m, min(hi - m, t))
            self._tgt[j] = t
            out[j] = t
        return out

    def state(self) -> dict:
        return {"targets": dict(self._tgt), "engaged": self._engaged}
