"""The redundant body DoFs — swivel (19) and the lift/torso comfort region (20).

SWIVEL (19)
-----------
    phi_i^* = Pi_[phi^-, phi^+] ( sgn(theta_i) [ |theta_i| - theta_d ]_+ ),
    theta_d = 60 deg

The 7-DOF arm keeps one self-motion about the shoulder-wrist axis, and the
operator's PALM ROLL is the natural handle for it -- but only past a wide
deadzone. Inside +-60 deg the elbow stays wherever the tracking cost puts it, so
ordinary wrist articulation during a reach does not swing the elbow around; past
it the operator is deliberately rolling the forearm and the elbow follows.
Pi projects onto the interval that is actually feasible under the joint limits
and torso clearance, so the law can never request an unreachable swivel.

LIFT AND TORSO (20)
-------------------
    u_c = K_c ( [c - c^+]_+ - [c^- - c]_+ ),   c = (r_L^* + r_R^*) / 2

The big, slow DoFs engage only when the HAND CENTROID leaves a comfort region --
they are reach extenders, not trackers, so they must stay still while the hands
work inside the arms' comfortable envelope and move only when the task pulls the
hands out of it.

Two things worth being explicit about:

1. READING OF (20). The paper prints the two positive parts with a `+` between
   them; taken literally that drives the same direction on both sides of the
   box. Implemented here as the SIGNED deadband (a difference), which is the
   only reading that restores the hands toward the comfort region from either
   side.
2. WHY THIS IS THE LOCAL-MODE LAW. c is built from the wrist TARGETS, which are
   robot-frame quantities, so (20) needs no world reference and is immune to a
   drifting F_a. The shipped global path instead drives the lift from the
   operator's ARKit world head HEIGHT, which in a moving vehicle rides every
   bump in the floor estimate. Global keeps its head-height law by product
   decision (feel continuity); local uses this one, and mode.py cross-fades the
   two across a switch so the change of law is never a step.
"""
from __future__ import annotations

import numpy as np

SWIVEL_DEADZONE = np.deg2rad(60.0)     # theta_d
SWIVEL_LIMIT = np.deg2rad(90.0)        # default [phi^-, phi^+] when unspecified

# Comfort region for the hand centroid, as fractions of the arm reach L_b and
# expressed in ROBOT BODY axes relative to the mid arm-mount root — so it scales
# with the robot and travels with the chassis instead of living in the world.
# (forward, lateral, vertical) x (lo, hi)
COMFORT_BOX = ((0.20, 0.75), (-0.90, 0.90), (-0.75, 0.35))
K_COMFORT = 1.2          # 1/s; comfort-region overflow -> body DoF rate
TORSO_GAIN = 2.5         # rad of torso bend per m/s of forward overflow
LIFT_RATE_MAX = 0.25     # m/s; lift speed cap (the lift is speed-limited)


def swivel_target(theta, phi_lo=-SWIVEL_LIMIT, phi_hi=SWIVEL_LIMIT,
                  deadzone=SWIVEL_DEADZONE):
    """Eq. (19). `theta` is the operator's palm roll [rad]; returns the swivel
    angle [rad], deadbanded then projected onto the feasible interval."""
    if theta is None:
        return None
    t = float(theta)
    excess = max(abs(t) - float(deadzone), 0.0)
    return float(np.clip(np.sign(t) * excess, phi_lo, phi_hi))


def palm_roll(R_wrist, u_arm):
    """Operator palm roll about the shoulder->wrist axis [rad].

    R_wrist  wrist orientation in the estimate's source frame
    u_arm    unit shoulder->wrist direction in that same frame

    Measured as the angle of the palm normal about `u_arm` relative to the
    component of the frame's own up that is perpendicular to `u_arm`, so it is
    independent of where the arm is pointing.
    """
    if R_wrist is None or u_arm is None:
        return None
    R = np.asarray(R_wrist, float)
    a = np.asarray(u_arm, float)
    a = a / (np.linalg.norm(a) + 1e-12)
    normal = R[:, 1]                       # palm normal (ARKit wrist +Y)
    ref = np.array([0.0, 1.0, 0.0])
    ref = ref - a * float(ref @ a)
    n = float(np.linalg.norm(ref))
    if n < 1e-6:                           # arm along the reference: degenerate
        return None
    ref /= n
    perp = np.cross(a, ref)
    v = normal - a * float(normal @ a)
    if float(np.linalg.norm(v)) < 1e-6:
        return None
    return float(np.arctan2(float(v @ perp), float(v @ ref)))


def comfort_box(reach, box=COMFORT_BOX):
    """Absolute (lo, hi) corners of the comfort region in robot body axes,
    scaled by the arm reach L_b."""
    L = float(reach)
    lo = np.array([box[0][0] * L, box[1][0] * L, box[2][0] * L])
    hi = np.array([box[0][1] * L, box[1][1] * L, box[2][1] * L])
    return lo, hi


def comfort_drive(centroid, robot, k=K_COMFORT, box=COMFORT_BOX):
    """Eq. (20). `centroid` is the hand centroid in ROBOT WORLD coordinates;
    returns (u_c, c_body) with u_c the signed overflow rate in robot BODY axes
    (forward, lateral, vertical) and c_body the centroid in those axes."""
    if centroid is None or robot.root is None or robot.basis is None:
        return None, None
    B = np.asarray(robot.basis, float)     # columns (right, up, forward) in F_b
    rel = np.asarray(centroid, float) - np.asarray(robot.root, float)
    # Body axes in the (forward, lateral, vertical) order the box is written in.
    c_body = np.array([float(rel @ B[:, 2]),    # forward
                       float(rel @ B[:, 0]),    # right (lateral)
                       float(rel @ B[:, 1])])   # up (vertical)
    L = max(float(np.mean([v for v in robot.reach.values()] or [0.6])), 1e-3)
    lo, hi = comfort_box(L, box)
    over = np.maximum(c_body - hi, 0.0) - np.maximum(lo - c_body, 0.0)
    return float(k) * over, c_body


def lift_and_torso(centroid, robot, k=K_COMFORT, box=COMFORT_BOX,
                   torso_gain=TORSO_GAIN, lift_max=LIFT_RATE_MAX):
    """Eq. (20) split onto the two actuators MABEL actually has.

    Returns (lift_rate [m/s], torso [rad], diag). Vertical overflow drives the
    LIFT (rate-limited); forward overflow leans the TORSO, which is the fast
    reach extender and so responds first. Lateral overflow is deliberately
    dropped here: sideways reach is the BASE's job, and the base is exactly what
    local mode holds still.
    """
    u_c, c_body = comfort_drive(centroid, robot, k, box)
    if u_c is None:
        return None, None, {}
    lift_rate = float(np.clip(u_c[2], -lift_max, lift_max))
    torso = float(-torso_gain * u_c[0] / max(k, 1e-6))
    return lift_rate, torso, {"centroid_body": c_body.tolist(),
                              "overflow": u_c.tolist()}


def hand_centroid(targets):
    """c = (r_L^* + r_R^*)/2 — or the single tracked target, so a one-handed
    task still engages the body DoFs instead of silently freezing them."""
    pts = [np.asarray(v, float) for v in targets.values() if v is not None]
    if not pts:
        return None
    return np.mean(pts, axis=0)
