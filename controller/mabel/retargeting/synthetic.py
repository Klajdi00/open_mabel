"""Synthetic operator + vehicle generator — the modelling utility the tests and
experiments drive the retargeter with.

It lives in the package (like `mabel.sources.mock`) rather than in a test file
for one blunt reason: THE VEHICLE CASE CANNOT BE TESTED ANY OTHER WAY. There is
no rig that shakes a headset the way a car does while a human performs a
repeatable bimanual task, so the only way to make "local mode is invariant to a
drifting world estimate" a checkable claim rather than a hope is to generate an
operator whose ground truth is known exactly and then corrupt the world estimate
by a known transform.

THE PHYSICAL MODEL OF THE VEHICLE CASE
--------------------------------------
Riding in a car, the headset's visual features are the CABIN, so the operator is
visually stationary and tracking looks healthy. What goes wrong is inertial: the
gyro integrates the vehicle's yaw into the anchor's heading, and the gravity
reference tilts toward the resultant of gravity and the vehicle's lateral
acceleration. Both effects move the ANCHOR relative to the cabin, so every pose
reported in F_a is premultiplied by the same drift D(t):

    T_a(t) = D(t) T_cabin(t)

Because head and hands are resolved against the same anchor, D(t) cancels
exactly in T_dh = T_ad^-1 T_ah — which is the entire premise of local mode, and
which this generator lets us verify numerically instead of by argument.
"""
from __future__ import annotations

import numpy as np

from .frames import basis_from_axes
from .local_frame import _R_from_ypr
from .operator_model import (ANTHRO_ACROMION_BELOW_EYE, ANTHRO_ARM_RATIO,
                             ANTHRO_BIACROMIAL, ANTHRO_EYE_FWD_OF_ACROMION,
                             ANTHRO_FOREARM_RATIO, ANTHRO_UPPER_ARM_RATIO)
from .types import RobotFrames

# MABEL's arm geometry, as quoted from the MJCF in the engine's arm map: the
# arm mounts are 0.148 m apart (vs a human's ~0.45 m biacromial — the reason
# eq. (17) needs a shape branch at all) and the shoulder->wrist reach is 0.60 m.
ROBOT_SHOULDER_SEP = 0.148
ROBOT_ARM_REACH = 0.60
ROBOT_MOUNT_HEIGHT = 1.10


def mabel_like_robot(chassis_yaw=0.0, lift=0.0):
    """A RobotFrames snapshot with MABEL's proportions, at a given chassis yaw.

    Robot world axes are x forward, y left, z up; the basis columns follow the
    package convention (right, up, forward) = (-y, z, x) rotated by the yaw.
    """
    c, s = np.cos(chassis_yaw), np.sin(chassis_yaw)
    fwd = np.array([c, s, 0.0])
    left = np.array([-s, c, 0.0])
    up = np.array([0.0, 0.0, 1.0])
    basis = basis_from_axes(-left, up, fwd)
    root = np.array([0.0, 0.0, ROBOT_MOUNT_HEIGHT + lift])
    half = 0.5 * ROBOT_SHOULDER_SEP
    return RobotFrames(
        shoulders={"right": root - left * half, "left": root + left * half},
        root=root,
        reach={"right": ROBOT_ARM_REACH, "left": ROBOT_ARM_REACH},
        basis=basis, lift=lift)


class SyntheticOperator:
    """An operator with EXACTLY known anthropometry and pose.

    Ground truth is generated in the CABIN frame (x right, y up, -z forward —
    ARKit's convention, so no extra bookkeeping). The torso frame's origin is
    the mid-shoulder root, matching operator_model.py.
    """

    def __init__(self, stature=1.75):
        self.H = float(stature)
        self.arm = ANTHRO_ARM_RATIO * self.H
        self.half_width = 0.5 * ANTHRO_BIACROMIAL * self.H
        self.neck_drop = ANTHRO_ACROMION_BELOW_EYE * self.H
        self.eye_fwd = ANTHRO_EYE_FWD_OF_ACROMION * self.H
        self.root = np.array([0.0, 0.72 * self.H, 0.0])   # mid-shoulder height
        self.torso_yaw = 0.0
        self.neck = (0.0, 0.0, 0.0)                        # yaw, pitch, roll
        # Per-side wrist commands in TORSO coordinates, as (extension rho,
        # direction). Direction is a unit vector in the torso frame.
        self.wrist = {"right": (0.6, np.array([0.35, -0.2, 0.9])),
                      "left": (0.6, np.array([-0.35, -0.2, 0.9]))}

    # ── frame construction ───────────────────────────────────────────────────
    @property
    def torso_basis(self):
        """B_o — torso axes (right, up, forward) in the cabin frame.

        ARKit convention, which is the convention every consumer assumes:
        x right, y up, **-z forward**. Writing forward as +z here (an easy slip)
        flips the estimators' forward axis against this generator's and shows up
        as a 16 cm torso-root disagreement at a mode handoff, not as an obvious
        failure — the frames stay orthonormal and everything keeps running.
        """
        c, s = np.cos(self.torso_yaw), np.sin(self.torso_yaw)
        right = np.array([c, 0.0, -s])          # +x, rotated about up by yaw
        fwd = np.array([-s, 0.0, -c])           # -z at yaw 0 = cross(up, right)
        return basis_from_axes(right, np.array([0.0, 1.0, 0.0]), fwd)

    def shoulder(self, side):
        s = 1.0 if side == "right" else -1.0
        return self.root + self.torso_basis @ np.array([s * self.half_width, 0.0, 0.0])

    def device(self):
        """T_ad in the CABIN frame: the head sits on the torso through the neck.

        The device's own axes are the torso's, rotated by the cervical angles —
        so a pure look-around moves T_ad without moving anything else, which is
        exactly the motion a head-only body-frame estimator cannot resolve.
        """
        B = self.torso_basis
        p = self.root + B @ np.array([0.0, self.neck_drop, self.eye_fwd])
        R = B @ _R_from_ypr(*self.neck)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = p
        return T

    def hand(self, side, roll=0.0):
        """T_ah for one wrist in the CABIN frame, at the commanded extension."""
        rho, d = self.wrist[side]
        d = np.asarray(d, float)
        d = d / (np.linalg.norm(d) + 1e-12)
        p = self.shoulder(side) + self.torso_basis @ (d * rho * self.arm)
        # Wrist orientation: palm frame aligned with the reach direction, rolled
        # about it by `roll` (the handle eq. 19 reads).
        f = self.torso_basis @ d
        up = np.array([0.0, 1.0, 0.0])
        r = np.cross(f, up)
        n = float(np.linalg.norm(r))
        r = r / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])
        u = np.cross(r, f)
        c, s = np.cos(roll), np.sin(roll)
        T = np.eye(4)
        T[:3, :3] = np.column_stack([r * c + u * s, -r * s + u * c, -f])
        T[:3, 3] = p
        return T

    def elbow(self, side, swivel=0.6):
        """T for the ARKit `forearm_arm` joint in the CABIN frame.

        The elbow sits on the circle of shoulder-wrist configurations that the
        two rigid segments allow; `swivel` picks where on it (0 = straight down
        from the reach axis, positive = OUTWARD, away from the midline), which
        is the arm self-motion eq. (19) commands.

        `swivel` is mirrored by side, because real arms are chiral: elbows swing
        away from the body on each side. Using one sign for both (an easy slip)
        builds an operator whose elbows both point the same way in the torso
        frame, and the torso fit then inherits that chirality — measured as a
        50-60 deg yaw error at POSITIVE yaw only, with negative yaw unaffected.
        An asymmetric error map is a modelling bug, not an ill-conditioned one:
        a genuinely under-determined fit fails symmetrically.
        """
        swivel = float(swivel) * (1.0 if side == "right" else -1.0)
        S, W = self.shoulder(side), self.hand(side)[:3, 3]
        a = ANTHRO_UPPER_ARM_RATIO * self.H
        b = ANTHRO_FOREARM_RATIO * self.H
        v = W - S
        d = float(np.linalg.norm(v))
        T = np.eye(4)
        if d < 1e-6:
            T[:3, 3] = S
            return T
        u = v / d
        if d >= a + b:                        # straight arm: elbow on the line
            T[:3, 3] = S + u * a
            return T
        h = (a * a - b * b + d * d) / (2.0 * d)
        r = float(np.sqrt(max(a * a - h * h, 0.0)))
        down = np.array([0.0, -1.0, 0.0])
        e1 = down - u * float(down @ u)
        n = float(np.linalg.norm(e1))
        e1 = e1 / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])
        e2 = np.cross(u, e1)
        T[:3, 3] = S + u * h + r * (np.cos(swivel) * e1 + np.sin(swivel) * e2)
        return T

    def frame(self, drift=None, roll=(0.0, 0.0), swivel=(0.6, 0.6),
              elbows=True):
        """(T_ad, {side: T_ah}, {side: skeleton}) as REPORTED IN F_a, i.e. after
        the anchor drift.

        `drift` is the 4x4 D(t) premultiplying every pose. Passing None gives the
        undrifted ground truth, which is the baseline every invariance metric is
        measured against. `elbows=False` drops the forearm joint, which is how
        the E2 ablation reproduces the paper-literal wrist-only fit.
        """
        D = np.eye(4) if drift is None else np.asarray(drift, float)
        hands, skel = {}, {}
        for side, rl, sw in (("right", roll[0], swivel[0]),
                             ("left", roll[1], swivel[1])):
            T_w = D @ self.hand(side, rl)
            hands[side] = T_w
            joints = {"wrist": T_w}
            if elbows:
                joints["forearm_arm"] = D @ self.elbow(side, sw)
            skel[side] = joints
        return D @ self.device(), hands, skel

    # ── convenience posers ───────────────────────────────────────────────────
    def set_reach(self, rho, separation=0.7, height=-0.2):
        """Both arms forward at extension `rho`, `separation` apart laterally
        (in units of the direction vector's x component). Used by the E2 sweep."""
        for side, sgn in (("right", 1.0), ("left", -1.0)):
            self.wrist[side] = (float(rho),
                                np.array([sgn * separation, height, 1.0]))
        return self


class VehicleDrift:
    """D(t) — the anchor drift a vehicle induces in the headset's world estimate.

    yaw_rate    rad/s the gyro integrates into the anchor heading
    tilt_amp    rad of gravity-reference tilt at peak lateral acceleration
    tilt_hz     how fast the vehicle weaves (the tilt oscillates with it)
    drift_vel   m/s of translational wander from double-integrated accelerometer
    """

    def __init__(self, yaw_rate=0.0, tilt_amp=0.0, tilt_hz=0.15,
                 drift_vel=(0.0, 0.0, 0.0)):
        self.yaw_rate = float(yaw_rate)
        self.tilt_amp = float(tilt_amp)
        self.tilt_hz = float(tilt_hz)
        self.drift_vel = np.asarray(drift_vel, float)

    def __call__(self, t):
        t = float(t)
        yaw = self.yaw_rate * t
        tilt = self.tilt_amp * np.sin(2.0 * np.pi * self.tilt_hz * t)
        # Anchor heading about world up (ARKit +Y), plus a gravity-reference
        # tilt about the world forward axis (-Z): a lateral-acceleration roll.
        cy, sy = np.cos(yaw), np.sin(yaw)
        Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
        ct, st = np.cos(tilt), np.sin(tilt)
        Rz = np.array([[ct, -st, 0.0], [st, ct, 0.0], [0.0, 0.0, 1.0]])
        T = np.eye(4)
        T[:3, :3] = Ry @ Rz
        T[:3, 3] = self.drift_vel * t
        return T
