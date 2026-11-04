"""
Task-space whole-body reach: the mobile base FOLLOWS the palm targets.

The base is the EAGER reach DOF (operator directive, 2026-07-18): a person
steps toward out-of-reach work instead of contorting, TURNS to face work at
their side, and — once placed — STANDS STILL. BaseFollow is therefore a
hysteretic POSITION + HEADING servo on the palm-target centroid, not a
gradient follower:

  * POSITION: the base drives so the chassis sits at its HOME-pose offset from
    the target centroid (the offset the arms were designed to work at, ~75-80%
    of usable reach — the "80% rule" generalised to the plane). Earlier laws
    failed sideways: steering "toward the centroid" is mostly a FORWARD vector,
    so lateral drags produced forward-crowding hunts instead of strafing, and
    a scalar reach-band could not represent lateral misalignment at all.
  * HEADING: bearing servo on the centroid (turn to face the work), engaged by
    hysteresis and gated on a MINIMUM planar distance — the bearing is
    ill-conditioned on top of the work (the ungated retarget bearing servo
    measured a −3.3 rad/s spin/orbit limit cycle on arrival), and capped at
    YAW_CAP so turning is deliberate.
  * SNAP, DON'T FLOAT: both servos are HYSTERETIC state machines (engage above
    _ON, drive at a decisive MINIMUM speed, disengage below _OFF and command
    ZERO). A plain P-law creeps asymptotically near its deadband — the
    "always floating, never stable" feel; the hysteresis turns every move into
    approach → stop.

Errors are PLANAR (chassis pitch must not bleed target height into them), and
every twist target — follower or joystick — is routed through
`motion_model.safe_base_step` (the MANDATORY tip-over gate). The caller owns
the plant and feeds the returned realised twist to `swerve_ik`.

Division of labour in the wrist / task-space method: upper_body = the 18-DOF
WholeBodyIK with heavily-lazy Lift/Torso; whole_body = + this follower, with
the lift/torso eased to rest while the base is `active` ("you don't lean into
a thing you're walking to" — the consumer gates on `.active`).
"""
import math

import numpy as np

from .. import config as C
from ..motion_model import safe_base_step

#: Follower translation cap [m/s]. The tip-over motion model inside
#: safe_base_step assumes the HOME CoM; a wrist-mode follow can have the arms
#: forward, so the approach stays deliberately below the nav ceiling.
FOLLOW_VMAX = 0.4
#: Follower yaw-rate cap [rad/s] — turning is deliberate, not a spin.
YAW_CAP = 1.0


class BaseFollow:
    """Palm-target → tip-safe base twist, with joystick override.

    step() implements the retarget-mode arbitration rule ("operator stick beats
    the follower"): a deliberate stick deflection drives the base directly;
    otherwise the position+heading servo emits a twist that parks the chassis
    at its home offset from the target centroid, facing it. The result is
    smoothed through safe_base_step, so the realised twist is always tip-safe.
    """

    def __init__(self, model):
        import mujoco
        self.model = model
        d0 = mujoco.MjData(model)
        mujoco.mj_forward(model, d0)
        ch = model.body(C.CHASSIS_BODY).id
        palms = [model.body(n).id for n in (C.LEFT_PALM, C.RIGHT_PALM)]
        # HOME anchor: chassis→palm-centroid offset in the CHASSIS frame (planar).
        # This is the geometry the arms were designed to work at; the position
        # servo restores it relative to wherever the operator drags the targets.
        Rb = d0.xmat[ch].reshape(3, 3)
        cen = 0.5 * (d0.xpos[palms[0]] + d0.xpos[palms[1]]) - d0.xpos[ch]
        local = Rb.T @ cen
        self._anchor_local = np.array([local[0], local[1]])   # chassis frame, planar
        self._chassis = ch

        self.vx = self.vy = self.omega = 0.0    # realised (smoothed) twist
        self._moving = False                    # position-servo hysteresis state
        self._yawing = False                    # heading-servo hysteresis state
        self.err = 0.0                          # telemetry: planar position error [m]
        self.bearing = 0.0                      # telemetry: centroid bearing [rad]
        self.yaw_err = 0.0                      # telemetry: active heading error [rad]

    @property
    def active(self):
        """True while either servo is driving — consumers hold the lazy body
        DOFs (lift/torso) at rest while the base is repositioning."""
        return self._moving or self._yawing

    def reset(self):
        """Zero the smoothed twist and disengage (torque-off / respawn)."""
        self.vx = self.vy = self.omega = 0.0
        self._moving = False
        self._yawing = False

    def _follow(self, data, p_l, p_r):
        """Hysteretic position+heading servo (see module docstring)."""
        base_pos = data.xpos[self._chassis]
        base_mat = data.xmat[self._chassis].reshape(3, 3)
        fwd_vec = base_mat[:, 0].copy()          # chassis +X (robot BACK is +X;
        left_vec = base_mat[:, 1].copy()         # conventions handled below)
        fwd_vec[2] = left_vec[2] = 0.0
        nf, nl = np.linalg.norm(fwd_vec), np.linalg.norm(left_vec)
        if nf < 0.5 or nl < 0.5:
            return 0.0, 0.0, 0.0                 # not upright: never drive
        fwd_vec /= nf
        left_vec /= nl

        cen = 0.5 * (np.asarray(p_l) + np.asarray(p_r))
        # Desired chassis position: centroid minus the home anchor, the anchor
        # rotated into the CURRENT heading (so after a turn the robot re-parks
        # with its front to the work).
        p_des = cen[:2] - (self._anchor_local[0] * fwd_vec[:2]
                           + self._anchor_local[1] * left_vec[:2])
        e_w = p_des - base_pos[:2]               # world planar position error
        e_x = float(np.dot(e_w, fwd_vec[:2]))    # chassis-frame components
        e_y = float(np.dot(e_w, left_vec[:2]))
        self.err = math.hypot(e_x, e_y)

        # centroid bearing in the chassis frame (front = -X, left = -Y)
        c_rel = cen[:2] - base_pos[:2]
        c_fwd = float(np.dot(c_rel, -fwd_vec[:2]))
        c_left = float(np.dot(c_rel, -left_vec[:2]))
        c_dist = math.hypot(c_fwd, c_left)
        self.bearing = math.atan2(c_left, c_fwd)

        # HEADING error: for bimanual work, square the SHOULDERS to the target
        # PAIR axis (signed angle from the chassis right-axis to p_r−p_l,
        # wrapped ±90° — a line has 180° symmetry, take the smaller turn).
        # Steering at the CENTROID bearing is wrong for side work: both targets
        # dragged left kept the pair axis fixed, yet the bearing turn (+39°)
        # rotated the pair diagonal in the chassis frame and left the far palm
        # 200 mm short — pure strafe was the answer, and pair-alignment says
        # exactly that (ψ=0). It also follows an OPERATOR's body turn (both
        # targets orbit → the pair rotates → the robot turns with them). The
        # centroid bearing remains the fallback for a degenerate pair, and the
        # override when the work is far BEHIND (face it first).
        pair = np.asarray(p_r)[:2] - np.asarray(p_l)[:2]
        pn = float(np.linalg.norm(pair))
        right = -left_vec[:2]
        psi = math.atan2(right[0] * pair[1] - right[1] * pair[0],
                         right[0] * pair[0] + right[1] * pair[1])
        if psi > math.pi / 2:
            psi -= math.pi
        elif psi < -math.pi / 2:
            psi += math.pi
        if abs(self.bearing) > 1.2 and c_dist > C.FOLLOW_YAW_MIN_DIST:
            yaw_err = self.bearing               # work behind: come about
        elif pn > 0.15:
            yaw_err = psi                        # square up to the pair
        elif c_dist > C.FOLLOW_YAW_MIN_DIST:
            yaw_err = self.bearing
        else:
            yaw_err = 0.0                        # degenerate + on top: hold
        self.yaw_err = yaw_err

        # ── position servo (hysteretic snap-stop) ─────────────────────────────
        if self._moving:
            if self.err < C.FOLLOW_POS_OFF:
                self._moving = False
        elif self.err > C.FOLLOW_POS_ON:
            self._moving = True
        vx = vy = 0.0
        if self._moving:
            v = float(np.clip(C.FOLLOW_KP_REACH * self.err,
                              C.FOLLOW_MIN_V, FOLLOW_VMAX))
            vx = v * e_x / self.err              # chassis-frame (swerve convention)
            vy = v * e_y / self.err

        # ── heading servo (hysteretic snap-stop on yaw_err) ───────────────────
        if self._yawing:
            if abs(yaw_err) < C.FOLLOW_YAW_OFF:
                self._yawing = False
        elif abs(yaw_err) > C.FOLLOW_YAW_ON:
            self._yawing = True
        omega = 0.0
        if self._yawing:
            omega = float(np.clip(C.LAZY_KP_ANG * yaw_err, -YAW_CAP, YAW_CAP))
            if abs(omega) < C.FOLLOW_MIN_W:
                omega = math.copysign(C.FOLLOW_MIN_W, omega)

        return vx, vy, omega

    def step(self, data, dt, z_lift, stick=None, p_l=None, p_r=None):
        """One control tick → the realised body-frame twist (vx, vy, omega).

        data   : live MjData (chassis pose for the servo)
        z_lift : current lift travel [m] (tip-over model input)
        stick  : (tvx, tvy, tw) pre-scaled joystick twist target, or None.
                 Any non-zero component claims the base (follower ignored).
        p_l/p_r: world palm TARGETS (e.g. WholeBodyIK.target_l_pos/r_pos);
                 None disables the follower (stick-only, e.g. joint mode).
        """
        if stick is not None and any(abs(v) > 0.0 for v in stick):
            tvx, tvy, tw = stick
            self._moving = self._yawing = False
        elif p_l is not None and p_r is not None:
            tvx, tvy, tw = self._follow(data, p_l, p_r)
        else:
            tvx = tvy = tw = 0.0
            self._moving = self._yawing = False
        self.vx, self.vy, self.omega = safe_base_step(
            self.vx, self.vy, self.omega, tvx, tvy, tw, dt, z_lift)
        return self.vx, self.vy, self.omega
