"""ff_schedule.py — WHEN to re-evaluate the pose-dependent gravity feedforward.

THE ARTIFACT THIS FIXES (operator report 2026-07-30: "under stiff mode, pushing
the arm feels like step-like motion"). The gateway evaluates τ_g at the MEASURED
pose but cached it on a FIXED 0.25 s period (a 2026-07-19 CPU guard: tff() is a
full mj_forward). Pushing the arm moves the pose continuously, so the feedforward
was computed for a pose up to 250 ms old: at ∂τ_g/∂q ≈ 5–10 N·m/rad and a hand
push of ~0.5 rad/s, each refresh landed as a 0.6–1.3 N·m torque STEP every
quarter second (the 10 N·m/s output slew stretched each step into a 60–130 ms
ramp — a staircase, not smoothness). Under position-holding stiff gains those
quarter-second torque detents are exactly a "step-like" feel. The JOINT-STATE
transport itself is NOT the problem — /arms/joint_states streams at 50 Hz.

THE FIX. Re-evaluate on POSE MOTION, not on a clock: as soon as any tracked
joint has moved `pose_eps` since the pose the cache was computed at (bounded
below by `min_period` for CPU), and at `max_period` regardless (drift safety).
While the arm is being pushed the FF tracks at up to 1/min_period (25 Hz —
measured 3.1 ms/eval on thor ⇒ ~8% of one core, and only WHILE moving); at rest
it costs the old 4 Hz. Each refresh step is then ≤ ∂τ/∂q·pose_eps ≈ 0.1 N·m —
below perception — and the output slew can be raised to a spike guard rather
than a staircase-stretcher.

Pure logic (no MuJoCo, no ROS) so the gateway's timing behaviour is unit-tested.
"""
from __future__ import annotations

__all__ = ["GravityFFScheduler"]


class GravityFFScheduler:
    """Decides when a pose-dependent evaluation is due.

    min_period : s   never evaluate faster than this (CPU bound; 0.04 ⇒ 25 Hz)
    max_period : s   always evaluate at least this often (drift safety)
    pose_eps   : rad re-evaluate once ANY tracked joint moved this far from the
                     pose of the CACHED evaluation (0.01 rad ≈ 0.6°)
    """

    def __init__(self, min_period: float = 0.04, max_period: float = 0.25,
                 pose_eps: float = 0.01):
        self.min_period = float(min_period)
        self.max_period = float(max_period)
        self.pose_eps = float(pose_eps)
        self._t = -1e18            # time of the cached evaluation
        self._pose: dict = {}      # pose of the cached evaluation

    def due(self, now: float, pose: dict) -> bool:
        """True if the FF should be re-evaluated for ``pose`` at time ``now``.
        Call :meth:`mark` after actually evaluating."""
        dt = now - self._t
        if dt >= self.max_period:
            return True
        if dt < self.min_period:
            return False
        for k, v in pose.items():
            ref = self._pose.get(k)
            if ref is None or abs(float(v) - ref) > self.pose_eps:
                return True
        return False

    def mark(self, now: float, pose: dict) -> None:
        """Record that an evaluation happened at ``now`` for ``pose``."""
        self._t = float(now)
        self._pose = {k: float(v) for k, v in pose.items()}
