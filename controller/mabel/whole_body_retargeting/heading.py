"""Operator BODY heading from the head alone, drift-free (the "lazy neck" model).

Why not the head+wrist estimator (body_frame_estimator.py): on the first live
Vision Pro whole-body takes (Retarget Lab, 2026-10-15) it latched 30-75 deg
offsets from where the operator actually faced, held them for tens of seconds,
and never came back — it re-seeds from its own last state after a hand-tracking
dropout and has no absolute reference, so an error made once is kept. Both
consumers inherit it: the base yaw (GLOBAL scope) and the arm map's body frame
(every scope), so the robot turned away and the arms rotated with it.

The model. The headset measures the HEAD yaw psi_h absolutely (ARKit world,
gravity-aligned, no drift). The trunk psi_b is not measured; what is known is
the neck: the head turns relative to the trunk within a comfortable range, and
a head held turned for a long time means the trunk turned. So

    e      = wrap(psi_h - psi_b)
    psi_b += sign(e) * max(0, |e| - SLACK)                      (1) the neck's range is hard
    psi_b += (1 - exp(-dt / TAU)) * g(|psi_h_dot|) * e_in        (2) a held look becomes a turn

with e_in the in-slack part of e after (1), and g = 1 while the head is still
(|psi_h_dot| below STILL_RATE, smoothed) and 0 while it moves, so a glance and
back does not rotate the trunk. Properties, each tested:
  * NO DRIFT: psi_b - psi_h is bounded by SLACK at every instant and decays to 0
    whenever the head is held — the estimate cannot keep an offset (the defect);
  * a body turn larger than SLACK is followed at once (1);
  * a look-around inside SLACK that returns within ~TAU moves psi_b by at most
    (1 - exp(-t_held / TAU)) of it, and nothing while the head is moving;
  * hands are not used, so hand dropouts cannot corrupt it.
Output convention = the engine's body_yaw: robot-frame heading of the body's
RIGHT axis, atan2(r_x, -r_y) (engine._headset_body_yaw, ShoulderEstimator._axes).
"""
import math

import numpy as np

AVP_TO_ROBOT = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

SLACK = math.radians(25.0)       # neck yaw slack; beyond it the trunk must have turned (swept, exp_heading)
TAU = 2.0                        # s, a held look becomes the body heading over ~TAU
STILL_RATE = math.radians(20.0)  # rad/s, head "still" below this (smoothed)
RATE_TAU = 0.15                  # s, head-rate smoothing


def _wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def head_heading(head_mat):
    """Head yaw in the engine's body_yaw convention (right axis, robot frame), or
    None when the head looks straight up/down (no horizontal forward)."""
    H = np.asarray(head_mat, float).reshape(4, 4)
    f = H[:3, :3] @ np.array([0.0, 0.0, -1.0])          # ARKit gaze (-z)
    n = math.hypot(f[0], f[2])
    if n < 1e-3:
        return None
    r_ark = np.array([-f[2] / n, 0.0, f[0] / n])          # horizontal right = fwd x up
    r = AVP_TO_ROBOT @ r_ark
    return math.atan2(r[0], -r[1])


class LazyHeadHeading:
    def __init__(self, slack=SLACK, tau=TAU, still_rate=STILL_RATE):
        self.slack, self.tau, self.still_rate = float(slack), float(tau), float(still_rate)
        self.reset()

    def reset(self):
        self.psi_b = None
        self._psi_h = None
        self._rate = 0.0

    def calibrate(self, head_mat):
        """Operator declares their facing: trunk := head now."""
        h = head_heading(head_mat)
        if h is not None:
            self.psi_b, self._psi_h, self._rate = h, h, 0.0

    def update(self, head_mat, dt, *, relax=True):
        """Track the neck-slack boundary; optionally relax a held gaze inward.

        Relative base teleop disables relaxation: a stationary headset must
        not generate fresh turn commands for seconds after the person stops.
        """
        h = head_heading(head_mat) if head_mat is not None else None
        if h is None:
            return self.psi_b                              # hold through a dropout
        if self.psi_b is None:
            self.psi_b, self._psi_h = h, h
            return self.psi_b
        dt = max(1e-4, float(dt))
        rate = abs(_wrap(h - self._psi_h)) / dt
        self._rate += (1.0 - math.exp(-dt / RATE_TAU)) * (rate - self._rate)
        self._psi_h = h
        e = _wrap(h - self.psi_b)
        if abs(e) > self.slack:                            # (1) beyond the neck's range
            self.psi_b = _wrap(self.psi_b + math.copysign(abs(e) - self.slack, e))
            e = math.copysign(self.slack, e)
        if relax and self._rate < self.still_rate:         # (2) a held look relaxes in
            self.psi_b = _wrap(self.psi_b + (1.0 - math.exp(-dt / self.tau)) * e)
        return self.psi_b
