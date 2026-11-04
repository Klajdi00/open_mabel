"""
CoM-over-base support (HoMMI-style) for MABEL.

HoMMI ("Learning Whole-Body Mobile Manipulation from Human Demonstrations") keeps the
upper-body mass centred over the mobile base with a whole-body-IK cost

    C_com = || p_torso^xy(q+dq) - p_base^xy(q+dq) - r*^xy ||^2_Wcom
    s.t.   |p_torso^x - p_base^x - r*^x| <= bx,   |p_torso^y - p_base^y - r*^y| <= by,

i.e. it drives the base under the torso/CoM and bounds the relative XY displacement.

MABEL's base is a velocity-controlled swerve drive solved OUTSIDE the upper-body IK
(see swerve.py / wbc.py), so we realise the same intent as a small REACTIVE base
controller rather than a QP cost on dq: each tick it measures the robot's TRUE centre
of mass (the mass-weighted subtree CoM, which moves as the arms/lift/torso move) and
drives the base XY so the CoM holds its natural offset r* over the base centre, within
a permitted box (bx, by). Outside the box it commands a proportional body-frame
velocity; inside the box the base holds (so it does not chase CoM micro-jitter).

This is complementary to -- not a replacement for -- the dynamic tip-over / ZMP motion
model in config.safe_base_step:

  * config.safe_base_step bounds the base ACCELERATION so an inertial manoeuvre does
    not tip the robot, using a FIXED CoM constant (config.COM_XY / config.com_height).
  * CoMBaseSupport keeps the QUASI-STATIC CoM centred as the arms move, and exposes the
    TRUE live CoM (`robot_com`, `com_offset_base`) so the motion model can be evaluated
    against the real CoM instead of the constant (see `live_tip_usage`).

Frame convention matches LazyBase (lazy_base.py): MABEL's front is -X and its left is
-Y in the chassis frame, and the returned (vx, vy) use the motor sign convention the
swerve IK expects.
"""
import numpy as np

from .. import config as C

# Wheel bodies whose ground-contact XY define the support triangle.
SUPPORT_WHEELS = ("fl_wheel", "fr_wheel", "b_wheel")


def robot_com(model, data, root_body=C.CHASSIS_BODY):
    """The robot's mass-weighted centre of mass in world coordinates -- the subtree
    CoM rooted at the floating base, so SCENE props (tables, the workstation) are
    excluded. This is the CoM that actually governs tip-over."""
    rid = model.body(root_body).id
    return data.subtree_com[rid].copy()


def support_polygon(model, data, wheels=SUPPORT_WHEELS):
    """The support triangle as the wheel-body ground XY positions (world frame)."""
    return np.array([data.body(w).xpos[:2] for w in wheels])


def support_margin(com_xy, poly):
    """Signed distance [m] from the CoM ground projection to the nearest support
    edge: POSITIVE inside the polygon, NEGATIVE once the CoM crosses an edge (the
    robot is then statically tipping over that edge)."""
    com_xy = np.asarray(com_xy)[:2]
    ctr = poly.mean(0)
    marg = np.inf
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        e = b - a
        nrm = np.array([e[1], -e[0]])
        nrm /= (np.linalg.norm(nrm) + 1e-12)
        if nrm @ (a - ctr) < 0:           # orient outward
            nrm = -nrm
        marg = min(marg, float(nrm @ (a - com_xy)))
    return marg


def com_offset_base(model, data, root_body=C.CHASSIS_BODY):
    """The CoM offset from the base centre, resolved on the robot's (front, left)
    axes [m]: front = -X chassis axis, left = -Y, matching LazyBase."""
    base_pos = data.body(root_body).xpos
    R = data.body(root_body).xmat.reshape(3, 3)
    fwd, left = -R[:, 0], -R[:, 1]
    e = robot_com(model, data, root_body) - base_pos
    return float(e @ fwd), float(e @ left)


def angular_tip_margin(model, data, root_body=C.CHASSIS_BODY):
    """Tip margin as an ANGLE [deg]: atan2(linear support margin, CoM height). The
    angle the whole robot would have to rotate about the nearest support edge before
    the CoM passes over it -- a height-aware stability score (high lift => small)."""
    com = robot_com(model, data, root_body)
    poly = support_polygon(model, data)
    mg = support_margin(com[:2], poly)
    return float(np.degrees(np.arctan2(mg, max(com[2], 1e-3))))


def live_tip_usage(model, data, vx, vy, omega, ax, ay, aw, root_body=C.CHASSIS_BODY):
    """config.tip_usage evaluated against the TRUE live CoM (XY offset + height) instead
    of the fixed config.COM_XY / config.com_height constants. Returns the support-polygon
    usage in [0, inf): <=1 stable. Lets the same ZMP criterion track the real CoM as the
    arms/lift move, which the constant-CoM model cannot."""
    e_fwd, e_left = com_offset_base(model, data, root_body)
    # config frame: front = -X, left = -Y -> chassis-frame CoM XY.
    com_xy = np.array([-e_fwd, -e_left])
    com_z = robot_com(model, data, root_body)[2]
    a_cx, a_cy = C.com_accel(vx, vy, omega, ax, ay, aw)
    # shift the CoM-accel projection by the live CoM offset vs the model's constant.
    u = 0.0
    for nrm, d_const, sf in C._SUPPORT_EDGES:
        d_live = d_const + nrm @ (C.COM_XY - com_xy)   # edge lever to the LIVE CoM
        toward = -(nrm[0] * a_cx + nrm[1] * a_cy)
        if toward > 0.0 and d_live > 1e-4:
            u = max(u, toward / (C.GRAVITY * d_live / (sf * com_z)))
    return u


class CoMBaseSupport:
    """Reactive base centring that keeps the robot CoM over its support polygon.

    Mirrors LazyBase's structure but drives on the TRUE CoM (not the hand centroid)
    and holds the CoM's natural standing offset r* within a box (bx, by). Call
    `calibrate()` once in the home pose to capture r*; then `command(data)` returns a
    body-frame (vx, vy, omega) for the swerve IK that nudges the base under the CoM."""

    def __init__(self, model, data, root_body=C.CHASSIS_BODY,
                 box_fwd=0.03, box_left=0.03, kp=2.0, vmax=C.MAX_LIN_SPEED):
        self.model = model
        self.data = data
        self.root_body = root_body
        self.bx = float(box_fwd)
        self.by = float(box_left)
        self.kp = float(kp)
        self.vmax = float(vmax)
        self.ref_fwd = 0.0
        self.ref_left = 0.0
        self.e_fwd = 0.0          # last CoM deviation (telemetry)
        self.e_left = 0.0
        self.calibrate()

    def calibrate(self):
        """Capture the CoM's natural offset over the base in the current pose, so the
        controller corrects only DEVIATIONS (e.g. from arm extension), not the resting
        front-bias the robot is already balanced at."""
        self.ref_fwd, self.ref_left = com_offset_base(self.model, self.data, self.root_body)

    def command(self, data=None):
        d = self.data if data is None else data
        fwd, left = com_offset_base(self.model, d, self.root_body)
        self.e_fwd = fwd - self.ref_fwd
        self.e_left = left - self.ref_left
        v_fwd = self._box_p(self.e_fwd, self.bx)
        v_left = self._box_p(self.e_left, self.by)
        # LazyBase motor convention: front/left commands are negated for the swerve IK.
        vx = float(np.clip(-v_fwd, -self.vmax, self.vmax))
        vy = float(np.clip(-v_left, -self.vmax, self.vmax))
        return vx, vy, 0.0

    def _box_p(self, e, box):
        """Proportional drive only once the deviation leaves the +/- box (deadband)."""
        if e > box:
            return self.kp * (e - box)
        if e < -box:
            return self.kp * (e + box)
        return 0.0
