"""Dexterous-planner FINGER retargeting for the TeleopEngine (keypoint IK).

This wires the dexterous_hand planner lineage in as the engine's finger solver:
teleop/planners/planner_avp.py / planner_per_finger.py (the validated reference)
→ mabel.hand_retargeting.keypoint_ik (its faithful MABEL port: canonical palm
frame, per-finger length calibration, shape + bone-direction residuals, DexPilot
pinch gating, damped Gauss–Newton with an analytic Jacobian, ~1 ms/hand — or the
reference SLSQP cost via MABEL_HAND_SOLVER=slsqp).

The adapter is SYNCHRONOUS: the engine already memoizes the heavy retarget once
per VR frame and runs both sides in its own pool, so the keypoint solve happens
in-line there instead of on keypoint_ik's background worker (which exists for
the facade's per-substep apply() path). Output joint order matches
HandOptimizer.joint_names / AnatomicalHandRetargeter exactly, so the engine's
finger-actuator write is unchanged. A one-euro filter (same params as the
anatomical path) smooths the solved targets against ARKit tracking spikes.

Falls back cleanly: engine.py selects this class only when the controller
package imports (see deps/_HAVE_KEYPOINT); otherwise the anatomical
decomposition keeps running.
"""
import os
import time

import numpy as np
import mujoco

from .hyperparams import (HAND_KP_SOLVER, HAND_ONEEURO, HAND_ONEEURO_MIN_CUTOFF,
                          HAND_ONEEURO_BETA, HAND_ONEEURO_DCUTOFF, HAND_ANAT_SMOOTH)
from .deps import HandIKRetargeter, kp_human_keypoints
from .logsetup import logger

# Named ARKit wire layout of the engine's 21-mat hand list (mirror of
# engine._NAMES_21 — kept here so the adapter can rebuild a named skeleton for
# streams that only carry the 21-array, e.g. the desktop ZMQ streamer).
_NAMES_21 = [
    "wrist",
    "thumb_knuckle", "thumb_intermediate_base", "thumb_intermediate_tip", "thumb_tip",
    "index_knuckle", "index_intermediate_base", "index_intermediate_tip", "index_tip",
    "middle_knuckle", "middle_intermediate_base", "middle_intermediate_tip", "middle_tip",
    "ring_knuckle", "ring_intermediate_base", "ring_intermediate_tip", "ring_tip",
    "little_knuckle", "little_intermediate_base", "little_intermediate_tip", "little_tip",
]


class _OneEuro:
    """Per-joint one-euro filter (identical law/params to the anatomical path)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.q = None
        self.dq = None
        self.t = None

    def __call__(self, q):
        now = time.monotonic()
        if self.q is None or self.t is None:
            self.q = q.copy()
            self.dq = np.zeros_like(q)
            self.t = now
            return self.q
        dt = now - self.t
        self.t = now
        if not HAND_ONEEURO or dt <= 0.0 or dt > 0.5:      # legacy EMA / gap frame
            a = HAND_ANAT_SMOOTH
            self.q = self.q + a * (q - self.q)
            return self.q
        two_pi = 2.0 * np.pi
        a_d = 1.0 / (1.0 + (1.0 / (two_pi * HAND_ONEEURO_DCUTOFF)) / dt)
        dq_raw = (q - self.q) / dt
        self.dq = self.dq + a_d * (dq_raw - self.dq)
        cutoff = HAND_ONEEURO_MIN_CUTOFF + HAND_ONEEURO_BETA * np.abs(self.dq)
        a = 1.0 / (1.0 + (1.0 / (two_pi * cutoff)) / dt)
        self.q = self.q + a * (q - self.q)
        return self.q


class KeypointHandRetargeter:
    """Drop-in finger solver for the TeleopEngine (same surface as
    AnatomicalHandRetargeter: joint_names / solve / reset / set_profile)."""

    def __init__(self, model, side):
        if HandIKRetargeter is None:
            raise ImportError("mabel.hand_retargeting.keypoint_ik unavailable")
        self.side = side                       # 'right' | 'left'
        self._key = side.capitalize()          # keypoint_ik hand key
        # data is only touched by keypoint_ik's own apply() (never called here);
        # a scratch keeps this adapter free of the live MjData.
        self._ik = HandIKRetargeter(model, mujoco.MjData(model),
                                    solver=HAND_KP_SOLVER)
        self._hand = self._ik.hands.get(self._key)
        if self._hand is None:
            raise ValueError(f"no ORCA hand subtree for side '{side}'")
        self.joint_names = list(self._hand["act_names"])
        self._filter = _OneEuro()
        self.last_q = None
        # Solve pacing (the reference planner solved at 20 Hz and let the
        # servos track in between): re-solving every 60–90 Hz VR frame buys no
        # accuracy (GN is warm-started) and costs ms on the retarget path.
        # 0 disables pacing (tests). The LEFT hand's phase is offset by half a
        # period so the two hands' solves land on DIFFERENT frames — halves the
        # worst-frame retarget spike (both-hands-in-one-frame).
        self._period = 1.0 / float(os.environ.get("MABEL_HAND_KP_HZ", "30"))
        self._last_solve = 0.0
        # Half-period clock offset applied after the FIRST solve, so the two
        # hands settle onto interleaved cadences.
        self._phase = 0.5 * self._period if side == "left" else 0.0

    def reset(self):
        self._hand["last_q"][:] = 0.0
        self._hand["have_solution"] = False
        self._filter.reset()
        self.last_q = None
        self._last_solve = 0.0

    def set_profile(self, finger_len_norm):
        """Freeze the operator's calibrated per-finger (canonical-normalised)
        bone lengths; None reverts to the live estimate."""
        if finger_len_norm:
            self._ik._human_lengths[self._key] = dict(finger_len_norm)
        else:
            self._ik._human_lengths.pop(self._key, None)

    @staticmethod
    def _named_from_21(h_mats):
        """21-mat list → {wire name: 4x4} for streams without a named skeleton."""
        out = {}
        for name, m in zip(_NAMES_21, h_mats):
            out[name] = np.asarray(m, float).reshape(4, 4)
        return out

    def solve(self, h_mats, joints=None):
        """h_mats: 21-element hand transform list; joints: the named ARKit
        skeleton when the wire carries one (preferred — it has the true
        landmarks; the 21-array rebuild is the fallback). Returns the solved
        joint target array aligned with self.joint_names, or the last targets
        on a degenerate frame (None before the first clean solve)."""
        now = time.monotonic()
        if self._period > 0.0 and (now - self._last_solve) < self._period:
            return self.last_q                  # paced: servos track the last solve
        if joints is None and h_mats is not None:
            joints = self._named_from_21(h_mats)
        if not joints:
            return self.last_q
        # First solve anchors this hand's cadence, phase-shifted (left half a
        # period early) so the two hands interleave instead of spiking together.
        self._last_solve = now - self._phase if self.last_q is None else now
        h_raw = kp_human_keypoints(joints, self.side)
        if h_raw is None:                       # degenerate palm: hold last
            return self.last_q
        hand = self._hand
        h_shape = self._ik._calibrate(h_raw, hand["lengths"], hand["kn0"],
                                      self._ik._human_lengths.get(self._key))
        try:
            solve = (self._ik._solve_slsqp if self._ik.solver == "slsqp"
                     else self._ik._solve_gn)
            q = solve(hand, h_shape, h_raw)
        except Exception:
            logger.exception("keypoint hand solve failed; holding last pose")
            return self.last_q
        hand["last_q"] = q                      # warm start for the next frame
        hand["have_solution"] = True
        self.last_q = self._filter(np.asarray(q, float))
        return self.last_q
