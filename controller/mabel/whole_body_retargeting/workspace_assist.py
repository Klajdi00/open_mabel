"""Mobile-base workspace assist — let the BASE free up the arms' workspace.

WHY THIS EXISTS
---------------
MABEL is a mobile manipulator and the retargeting used none of that: the base
sat still while the arms failed to reach. Measured on EgoDex (2026-08-25), the
single largest failure class is an operator working close to their own body --
holding a fixture against the chest, using chopsticks over a bowl. Through the
root-anchored GMR similarity, and onto a torso roughly three times narrower
than a human's, those poses map to targets INSIDE the robot: 0.023-0.063 m from
the torso axis against a 0.074 m shoulder half-width, i.e. between the two
shoulders. No arm configuration reaches there. Corpus-wide, operator reach
predicts wrist error at r = -0.71 -- close work costs 6x the error of extended
work.

THE SUBTLETY THAT DICTATES THE DESIGN
-------------------------------------
The GMR target is anchored at the robot's LIVE mid-shoulder,

    t = root_r + sigma * M * (w - root_h)

so the target is rigidly attached to the robot's own shoulders. Driving the
base moves ``root_r`` and therefore moves ``t`` by exactly the same amount:
**base motion alone changes reachability not at all.** Commanding the base to
"back up for more room" is a no-op under this map, and would look like a bug.

What does work is a two-part move:

  1. Offset the map's ANCHOR by ``d`` (robot frame). This gives the arm room
     immediately -- it is a pure retargeting change, one frame, no dynamics.
  2. Drive the BASE by the same ``d``, and reduce the offset by exactly as much
     as the base actually moves.

Part 2 is what makes part 1 honest. Without it the offset is a standing lie
about where the robot is, and the robot's hands sit somewhere the operator did
not put them. With it, the hands stay where they were in the WORLD while the
robot backs out from under them -- which is what a person does when their own
arms are cramped.

So ``offset`` is a DEBT: the amount of repositioning the base still owes. It
grows when the workspace is crowded and shrinks as the base pays it down.

WHAT TRIGGERS IT
----------------
Three conditions, each with a deadband, evaluated on the commanded targets:

  * too NEAR   -- min shoulder radius below ``ARM_WS_NEAR``  -> back off
  * too FAR    -- max shoulder radius above ``ARM_WS_FAR``   -> close in
  * too LATERAL-- mean lateral offset beyond ``ARM_WS_LATERAL`` -> strafe

Near and far cannot both fire meaningfully (one hand tucked while the other
extends is a real bimanual pose), so the NEAR condition wins: an unreachable
target is a hard failure, an over-extended one merely clips.
"""
from __future__ import annotations

import numpy as np

from ..teleop_engine.hyperparams import (
    ARM_WS_ASSIST, ARM_WS_NEAR, ARM_WS_FAR, ARM_WS_LATERAL,
    ARM_WS_MAX_OFFSET, ARM_WS_RATE, ARM_WS_DEADBAND, ARM_WS_BASE_GAIN,
)


class WorkspaceAssist:
    """Tracks the outstanding base-repositioning debt and the twist that pays it.

    Frames: ``offset`` is held in the CHASSIS frame (x forward-negative, y left)
    so that it means the same thing as the robot turns. It is converted to world
    only when applied to the anchor.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.offset = np.zeros(2)        # chassis-frame debt [m], (x, y)
        self.err = 0.0                   # last workspace error magnitude [m]
        self.reason = ""                 # which condition fired (diagnostics)
        self._base_prev = None           # chassis origin last frame (world)

    # ── the workspace error ──────────────────────────────────────────────────
    @staticmethod
    def _demand(targets, shoulders, R_chassis):
        """Chassis-frame offset the workspace is ASKING for, before rate limits.

        ``targets`` / ``shoulders``: {side: world xyz}. Returns (demand_xy,
        reason), expressed as the ANCHOR shift in the chassis frame.

        SIGNS ARE THE WHOLE TRICK HERE, and getting them backwards produces a
        result that helps some clips and hurts others rather than an obvious
        failure. Targets move WITH the anchor (t = root_r + ...). So to give a
        crowded arm room, the targets must move FORWARD relative to the robot,
        which is chassis -x -- the anchor shift is NEGATIVE x. The base then
        moves the OTHER way (+x, backwards) so that in the world frame the
        hands stay where the operator put them and the robot backs out from
        under them. Anchor and base are therefore always opposite in sign; see
        `base_twist`.
        """
        if not targets:
            return np.zeros(2), ""
        radii, lat = [], []
        for side, t in targets.items():
            sh = shoulders.get(side)
            if sh is None:
                continue
            v = np.asarray(t, float) - np.asarray(sh, float)
            radii.append(float(np.linalg.norm(v)))
            lat.append(float((R_chassis.T @ (np.asarray(t, float)
                                             - np.asarray(sh, float)))[1]))
        if not radii:
            return np.zeros(2), ""
        d = np.zeros(2)
        reason = ""
        near_gap = ARM_WS_NEAR - min(radii)
        far_gap = max(radii) - ARM_WS_FAR
        if near_gap > ARM_WS_DEADBAND:
            # Crowded: shift the targets FORWARD (chassis -x) into the front
            # workspace; the base will back off (+x) to make that honest.
            d[0] = -near_gap
            reason = "near"
        elif far_gap > ARM_WS_DEADBAND:
            # Over-extended: pull the targets back toward the body; the base
            # closes in. Clipping is a softer failure than an unreachable
            # target, so this only fires when NEAR is satisfied.
            d[0] = +far_gap
            reason = "far"
        mean_lat = float(np.mean(lat)) if lat else 0.0
        if abs(mean_lat) > ARM_WS_LATERAL + ARM_WS_DEADBAND:
            # Targets sit off to one side: shift them back toward centre.
            d[1] = -(mean_lat - np.sign(mean_lat) * ARM_WS_LATERAL)
            reason = (reason + "+lateral") if reason else "lateral"
        return d, reason

    # ── per-frame update ─────────────────────────────────────────────────────
    def update(self, targets, shoulders, base_pos, R_chassis, dt):
        """Advance the debt. Returns the WORLD-frame anchor offset to apply.

        ``base_pos``/``R_chassis``: chassis origin and rotation, world frame.
        """
        if not ARM_WS_ASSIST:
            return np.zeros(3)
        base_pos = np.asarray(base_pos, float)

        # PAY DOWN THE DEBT BY WHAT THE BASE ACTUALLY MOVED. This is the step
        # that keeps the offset honest: the anchor shift and the base motion
        # must never be counted twice, or the robot walks away from the
        # operator's intent.
        if self._base_prev is not None:
            moved = R_chassis.T @ (base_pos - self._base_prev)
            self.offset += moved[:2]   # base moves opposite the anchor, so ADD
        self._base_prev = base_pos.copy()

        demand, reason = self._demand(targets, shoulders, R_chassis)
        self.reason = reason
        self.err = float(np.linalg.norm(demand))

        # Move the debt toward the demand at a bounded rate — this is a
        # whole-body reposition, not a tracking channel, so it is deliberately
        # slow and never steps.
        step = ARM_WS_RATE * dt
        delta = demand - self.offset
        n = float(np.linalg.norm(delta))
        if n > step:
            delta *= step / n
        self.offset = self.offset + delta

        n = float(np.linalg.norm(self.offset))
        if n > ARM_WS_MAX_OFFSET:
            self.offset *= ARM_WS_MAX_OFFSET / n

        return R_chassis @ np.array([self.offset[0], self.offset[1], 0.0])

    def base_twist(self):
        """(vx, vy) in the CHASSIS frame that pays the debt down.

        OPPOSITE in sign to the anchor offset, and that is the point: the anchor
        moves the targets forward relative to the robot, the base moves the
        robot backward, and the two cancel in the WORLD frame so the hands stay
        where the operator put them. `update` then subtracts the motion the base
        actually achieved, so the offset decays to zero once the robot has taken
        up the position the arms needed.
        """
        if not ARM_WS_ASSIST:
            return 0.0, 0.0
        v = -self.offset * ARM_WS_BASE_GAIN
        return float(v[0]), float(v[1])

    def status(self):
        return {"offset_m": [round(float(x), 4) for x in self.offset],
                "err_m": round(self.err, 4), "reason": self.reason}
