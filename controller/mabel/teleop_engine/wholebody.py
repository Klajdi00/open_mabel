"""LAYER 2 whole-body control: torso/base/lift cascade + base + torso trackers."""
import numpy as np
import mujoco

from .hyperparams import *
from .geometry import _yaw, _wrap
from .motion_model import BaseMotionModel, LiftAwareReverseGuard
from .deps import swerve_ik, _HAVE_SWERVE
from ..motion_model.relative_follow import RelativeBaseFollow


def _soft_deadband(x, band):
    """Shift a signal toward zero by `band`, never past it. Unlike a clip this is
    CONTINUOUS at the threshold — the response starts from zero and grows, so a
    lazy DoF never steps when the operator crosses the boundary."""
    if band <= 0.0:
        return float(x)
    return float(np.sign(x) * max(0.0, abs(float(x)) - band))


class BaseTracker:
    """Operator HEAD displacement -> swerve base command, via an asymmetric, front-of-
    head BOUNDING BOX. The head displacement (Vision Pro world) is mapped to the robot
    frame by AVP_TO_ROBOT (the SAME transform the arms use). The part of the offset
    INSIDE the box is left to the torso/lift; only the OVERFLOW beyond the box drives
    the base, scaled by a per-direction sensitivity weight (forward box deep, backward
    box shallow -> backing up is the most responsive). The position mode uses
    proportional position/yaw correction. NOVA's relative mode follows accepted
    operator motion and decays to zero without returning to a world target."""

    def __init__(self, model, data):
        self.bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ROBOT_BASE_BODY)
        self.data = data
        self.head0 = None
        self.byaw0 = None
        self._dpsi = 0.0
        self._out_ang = 0.0
        self.follow_mode = "position"
        self.relative = RelativeBaseFollow(vmax=BASE_VMAX, wmax=BASE_WMAX)
        # 1:1 POSITION follow (NOVA, 2026-10-29; worker.py sets it). The base goes exactly where the operator's
        # head went (minus only what the hip-pivot lean model explains as a bend): no comfort box, no
        # backward over-weighting, no workspace-assist base drive. Closed on the mirrored wheel odometry with
        # its own gain (`kp`, sized for NOVA's ~0.3-0.5 s command->wheel dead time) and an error deadband
        # (`err_band`) so stance sway does not twitch the modules. Yaw keeps the relative follower.
        self.one_to_one = False
        self.kp = BASE_KP
        self.err_band = 0.0
        self._last_args = None
        self._last_w = 0.0

    def set_follow_mode(self, mode, response_s=.22):
        if mode not in ("position", "relative"):
            raise ValueError(f"Unknown base follow mode: {mode}")
        if self.follow_mode != mode or self.relative.response_s != response_s:
            self.follow_mode = mode
            self.relative = RelativeBaseFollow(response_s=response_s, vmax=BASE_VMAX, wmax=BASE_WMAX)
            self.head0 = None
            return True
        return False

    # Neck pivot in the headset's own frame (ARKit: +y up, +z back): 9 cm behind and 10 cm below the eyes,
    # the same pivot wb_direction_check's look_around uses.
    NECK_PIVOT = np.array([0.0, -0.10, 0.09])

    def neck_pivot(self, head_mat):
        if head_mat is None:
            return None
        M = np.asarray(head_mat, float).reshape(4, 4)
        return M[:3, 3] + M[:3, :3] @ self.NECK_PIVOT

    def calibrate(self, head_pos, body_yaw):
        self.pivot0 = None                                  # re-latched on the first 1:1 frame
        self.head0 = np.asarray(head_pos, float).copy()
        self.base0 = self.data.xpos[self.bid].copy()
        self.cyaw0 = _yaw(self.data.xmat[self.bid].reshape(3, 3))   # chassis yaw at calib
        # None-aware: headset clients may anchor before the elbow-based heading has
        # its first sample; the yaw origin is then latched LATE in command() (the old
        # `else 0.0` made the first real heading read as a large yaw error -> spin).
        self.byaw0 = float(body_yaw) if body_yaw is not None else None
        self.relative.reset(self.base0[:2], self.cyaw0)

    def head_offset(self, head_pos, body_yaw=None, anchor=None):
        """Operator head displacement from calibration, robot frame -> (forward, lateral)
        [m].  robot front = -x, robot right = +y.  In BASE_TRANSLATION_MODE "body"
        it is expressed in the operator's CURRENT body frame (rotated by the heading
        change since calibration), so a forward lean after a turn is a lean."""
        d = AVP_TO_ROBOT @ (np.asarray(head_pos, float) - (self.head0 if anchor is None else anchor))
        self._dpsi = 0.0
        self._out_ang = 0.0
        if BASE_TRANSLATION_MODE == "body" and body_yaw is not None:
            if self.byaw0 is None:
                self.byaw0 = float(body_yaw)
            # into the operator's CURRENT body frame. body_yaw is the robot-frame
            # heading of the body's right axis, = pi when the operator faces the
            # direction AVP_TO_ROBOT maps onto the robot's front (ARKit -z), so the
            # facing at ANY moment is (body_yaw - pi) away from that.
            a = _wrap(float(body_yaw) - np.pi)
            c, s_ = np.cos(a), np.sin(a)
            d = np.array([c * d[0] + s_ * d[1], -s_ * d[0] + c * d[1], d[2]])   # R(-a) d
            # and back out along the ROBOT's heading: its calibration heading plus
            # the operator's turn since (= what the yaw follow drives it to)
            self._dpsi = BASE_YAW_SIGN * _wrap(float(body_yaw) - self.byaw0)
            self._out_ang = getattr(self, "cyaw0", 0.0) + self._dpsi
        return -float(d[0]), float(d[1])

    def command(self, fwd, lat, fwd_in, lat_in, body_yaw, dt=None):
        """fwd/lat = head offset; fwd_in/lat_in = the part absorbed by the box (torso).
        The base targets only the OVERFLOW (fwd - fwd_in, lat - lat_in), scaled by the
        direction sensitivity weight (backward weighted higher)."""
        if self.head0 is None:
            return 0.0, 0.0, 0.0
        if self.one_to_one:
            self._last_args = (fwd, lat, body_yaw)
            return self._one_to_one(fwd, lat, body_yaw, dt, new_input=True)
        o_fwd = fwd - fwd_in                                # signed forward overflow
        o_lat = lat - lat_in
        if BASE_TRANSLATION_MODE == "body":
            # 1:1, the overflow rotated from the operator's body frame back to the
            # world: the base sits exactly where the operator's body went, minus
            # what the comfort box leaves to the torso/arms
            c, s_ = np.cos(self._out_ang), np.sin(self._out_ang)
            ox, oy = -o_fwd, o_lat
            target = self.base0 + np.array([c * ox - s_ * oy, s_ * ox + c * oy, 0.0])
        else:
            w_fwd = BASE_W_FWD if o_fwd >= 0.0 else BASE_W_BACK # backward more sensitive
            # desired base position (robot world): front = -x, so forward overflow -> -x
            target = self.base0 + np.array([-w_fwd * o_fwd, BASE_W_LAT * o_lat, 0.0])
        Rb = self.data.xmat[self.bid].reshape(3, 3)
        if self.follow_mode == "relative":
            if body_yaw is not None and self.byaw0 is None:
                self.byaw0 = float(body_yaw)
            goal_yaw = (self.cyaw0 + BASE_YAW_SIGN * _wrap(float(body_yaw) - self.byaw0)
                        if body_yaw is not None else None)
            return self.relative.step(target[:2], goal_yaw, _yaw(Rb),
                                      float(dt) if dt is not None else 1./60.)
        err = target - self.data.xpos[self.bid]; err[2] = 0.0
        v_body = Rb.T @ (BASE_KP * err)                     # chassis-frame velocity for swerve_ik
        # ROTATION: the operator's BODY heading (odom body_yaw) drives the base yaw.
        omega = 0.0
        if body_yaw is not None:
            if self.byaw0 is None:                          # late anchor (first sample
                self.byaw0 = float(body_yaw)                # arrived after calibrate)
            # ORIENTATION BOX: only the heading error BEYOND the deadband turns
            # the chassis, so the base holds still through ordinary sway and
            # turns when the operator actually turns. Same soft form as the
            # translation overflow, so it starts from zero rather than stepping.
            d_yaw = _wrap(float(body_yaw) - self.byaw0)
            tgt_cyaw = self.cyaw0 + BASE_YAW_SIGN * d_yaw
            # the lazy head heading is already deadbanded by the neck model:
            # a second box here only leaves a standing heading error
            box = 0.0 if HEADING_MODE == "head" else BASE_YAW_BOX
            yaw_err = _soft_deadband(_wrap(tgt_cyaw - _yaw(Rb)), box)
            omega = float(np.clip(BASE_KP_YAW * yaw_err, -BASE_WMAX, BASE_WMAX))
        return (float(np.clip(v_body[0], -BASE_VMAX, BASE_VMAX)),
                float(np.clip(v_body[1], -BASE_VMAX, BASE_VMAX)), omega)

    def _one_to_one(self, fwd, lat, body_yaw, dt, new_input):
        """1:1 position follow: target = anchor + the head's displacement (robot world, front = -x)."""
        if BASE_TRANSLATION_MODE == "body":                # as command(): body frame -> the robot's heading
            c, s_ = np.cos(self._out_ang), np.sin(self._out_ang)
            ox, oy = -float(fwd), float(lat)
            target = self.base0 + np.array([c * ox - s_ * oy, s_ * ox + c * oy, 0.0])
        else:
            target = self.base0 + np.array([-float(fwd), float(lat), 0.0])
        Rb = self.data.xmat[self.bid].reshape(3, 3)
        if new_input:                                       # yaw: the relative follower, unchanged
            if body_yaw is not None and self.byaw0 is None:
                self.byaw0 = float(body_yaw)
            goal_yaw = (self.cyaw0 + BASE_YAW_SIGN * _wrap(float(body_yaw) - self.byaw0)
                        if body_yaw is not None else None)
            self._last_w = self.relative.step(target[:2], goal_yaw, _yaw(Rb),
                                              float(dt) if dt is not None else 1./60.)[2]
        err = target - self.data.xpos[self.bid]; err[2] = 0.0
        if float(np.linalg.norm(err)) < self.err_band:
            err[:] = 0.0
        v = Rb.T @ (self.kp * err)
        s = float(np.hypot(v[0], v[1]))
        if s > BASE_VMAX:
            v *= BASE_VMAX / s
        return float(v[0]), float(v[1]), float(self._last_w)

    def recommand(self):
        """Re-close the 1:1 position loop on the CURRENT (mirrored) base pose without a new operator frame
        (NOVA receives the headset in clumps; between them the target holds and the base must not coast on
        the last frame's velocity). The yaw rate is the last frame's. None when not in 1:1 position mode."""
        if not self.one_to_one or self.head0 is None or self._last_args is None:
            return None
        fwd, lat, body_yaw = self._last_args
        return self._one_to_one(fwd, lat, body_yaw, None, new_input=False)


class TorsoController:
    """The torso is the FIRST, fast reach extender: it bends with the part of the
    operator's HEAD forward offset that is ABSORBED INSIDE the base bounding box, so the
    within-box forward reach is taken by the torso BEFORE the base ever moves. A NEGATIVE
    torso angle leans the upper body forward (extra forward/down reach)."""

    def __init__(self, model, data):
        self.data = data
        self.tj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "Torso")
        self.lo, self.hi = model.jnt_range[self.tj]
        self.t0 = float(model.qpos0[model.jnt_qposadr[self.tj]])   # rest torso angle

    def command(self, fwd_in):
        """fwd_in = forward head offset clipped to the box [m]; >0 forward -> lean fwd."""
        return float(np.clip(self.t0 - TORSO_GAIN * fwd_in, self.lo, self.hi))

    def command_angle(self, theta):
        """theta = operator trunk pitch [rad], >0 forward -> the same forward bend."""
        return float(np.clip(self.t0 - theta, self.lo, self.hi))
class WholeBodyController:
    """LAYER 2 of 3. Coordinates the redundant body DoFs that EXTEND the operator's
    reach, in priority order, using a bounding-box ENGAGE rule — the slow DoFs only
    move once the operator leaves a comfort box, so small sway / arm-only motion
    never drives the body:

        TORSO  (fast, agile)    bends with the operator's forward LEAN, first
        BASE   (sluggish)       follows the body-odom walk translation + heading,
                                ramped through the LAYER-1 BaseMotionModel
        LIFT   (speed-limited)  follows head height, rate-limited

    SE(3) wrist tracking itself is the ARM retargeter's job (Layer 3); this layer
    only POSITIONS the body so the arm stays inside its comfortable workspace.

    Swap points (so the whole-body strategy can change without touching main):
      * motion_model : a BaseMotionModel (Layer 1) — base dynamics / constraints
      * base_filter  : optional callable(data, (vx,vy,omega)) -> (vx,vy,omega),
                       applied to the base velocity EVERY physics step. This is the
                       hook for a control-barrier function (self-collision / geofence
                       / operator-safety) — return a filtered velocity, e.g. project
                       out components that would violate a barrier h(q) >= 0.
    The torso / base / lift sub-policies are themselves separate objects
    (TorsoController, BaseTracker) and can each be replaced independently."""

    def __init__(self, model, data, torso_act, lift_act,
                 motion_model=None, base_filter=None):
        self.torso = TorsoController(model, data)
        self.tracker = BaseTracker(model, data)
        self.motion_model = motion_model if motion_model is not None else BaseMotionModel()
        # Default safety filter = the lift-aware reverse guard. Pass your own
        # control-barrier callable to replace it, or `lambda d, v: v` to disable.
        self.base_filter = base_filter if base_filter is not None else LiftAwareReverseGuard(lift_act)
        self.torso_act = torso_act
        self.lift_act = lift_act
        # RETARGET SCOPE. LOCAL leaves the mobile base out of the follow loop
        # entirely (the operator drives it on the joystick); GLOBAL lets it take
        # the overflow beyond the comfort box. Nothing else differs, so a switch
        # can never change how the arms/torso/lift/neck track.
        self.base_in_loop = (RETARGET_SCOPE == "global")
        self.reset(data)

    def reset(self, data):
        """Clear anchors + velocity state; re-capture the lift origin from the
        settled pose. Call after settle() and on an external reset."""
        self.tracker.head0 = None
        self.base_anchored = False
        self.base_cmd = (0.0, 0.0, 0.0)      # realized body velocity (ramped)
        self.base_tgt = (0.0, 0.0, 0.0)      # desired body velocity (this frame)
        self.head_h0 = None
        self._lean_drop = 0.0              # head drop a lean explains (angle model)
        self.lift0 = float(data.ctrl[self.lift_act])

    def update_targets(self, data, head_pos, odom_base, body_yaw, head_height,
                       reach_demand=0.0, dt=None, follow_dt=None, head_mat=None):
        """Per-FRAME: set the torso + lift position targets and the base velocity
        TARGET (ramped to a realized velocity later, per physics step). Returns a
        small dict for telemetry."""
        info = {}
        if head_pos is None and self.tracker.follow_mode == "relative":
            # Reacquisition starts from the new input, never from motion missed
            # while the head was untracked. Also drop workspace-assist authority.
            self.tracker.relative.reset()
            self.base_tgt = (0.0, 0.0, 0.0)
        # HEAD-displacement bounding-box cascade: the part of the head's forward/lateral
        # offset INSIDE the box is absorbed by the TORSO (fast); the OVERFLOW beyond the
        # box drives the BASE (sluggish). (odom_base is unused now -- the cascade is keyed
        # to the HEAD, not the body odom.)
        if head_pos is not None and (BASE_FOLLOW or TORSO_FOLLOW):
            if not self.base_anchored:
                self.tracker.calibrate(head_pos, body_yaw)
                self.base_anchored = True
            fwd, lat = self.tracker.head_offset(head_pos, body_yaw)
            fwd_in = float(np.clip(fwd, -BASE_BOX_BACK, BASE_BOX_FWD))
            lat_in = float(np.clip(lat, -BASE_BOX_LAT, BASE_BOX_LAT))
            # 1) TORSO — absorbs the within-box forward reach (engages first),
            #    past a soft deadband so ordinary sway does not bend it.
            # LEAN MODEL (hyperparams TORSO_LEAN_MODE): "angle" reads the
            # operator's trunk pitch off a hip-pivot model instead of the
            # box-clipped head offset, and tells the base / lift what the bend
            # already explains.
            fwd_walk = fwd
            self._lean_drop = 0.0
            if not self.base_in_loop:
                # LOCAL: the frame travels with the operator. Lean comes from the head DROP (a hip
                # pivot explains it, walking does not), and the horizontal anchor is leashed to the
                # forward distance that bend explains, so a walk drags the anchor instead of reading
                # as lean (and a squat after a walk has no forward evidence left to be a bend).
                L = TORSO_LEAN_PIVOT
                drop = float(self.tracker.head0[1] - np.asarray(head_pos, float)[1])
                # The leash is geometry, not a motion command. Subtracting the
                # gait deadband here drags the anchor forward during the start
                # of a slow bend and permanently discards its forward evidence.
                # Apply the deadband only to theta below; level walking still
                # has zero bend radius and gait bob still commands zero lean.
                drop = max(drop, 0.0)
                d_bend = L * np.sin(np.arccos(np.clip(1.0 - drop / L, -1.0, 1.0)))
                dh = np.asarray(head_pos, float) - self.tracker.head0; dh[1] = 0.0
                n = float(np.linalg.norm(dh))
                if n > d_bend:
                    self.tracker.head0 = self.tracker.head0 + dh * (1.0 - d_bend / n)
                    fwd, lat = self.tracker.head_offset(head_pos, body_yaw)
                    fwd_in = float(np.clip(fwd, -BASE_BOX_BACK, BASE_BOX_FWD))
                    lat_in = float(np.clip(lat, -BASE_BOX_LAT, BASE_BOX_LAT))
                fwd_walk = fwd
            if TORSO_LEAN_MODE == "angle":
                L = TORSO_LEAN_PIVOT
                drop = float(self.tracker.head0[1] - np.asarray(head_pos, float)[1])
                th_drop = float(np.arccos(np.clip(1.0 - max(drop, 0.0) / L, -1.0, 1.0)))
                d_bend = L * np.sin(th_drop)            # forward a bend of that drop makes
                if self.base_in_loop:
                    # the base walks only on what the bend does not explain; the
                    # torso keeps the within-box offset plus the explained part
                    fwd_walk = fwd - float(np.clip(fwd, 0.0, d_bend))
                    d_lean = float(np.clip(fwd, -BASE_BOX_BACK, max(BASE_BOX_FWD, d_bend)))
                else:
                    d_lean = fwd                        # LOCAL: what is left after the leash is lean
                # two readings of one bend: the FORWARD offset (robust for small
                # leans) and the head DROP (what a deep bend is mostly made of —
                # neck and lower-spine flexion drop the head more than a rigid
                # hip pivot predicts: measured 0.23 m fwd / 0.15 m down = 22 deg
                # by the first, 38 deg by the second). The drop reading is capped
                # by the forward evidence (asin(d / TORSO_LEAN_PIVOT_MIN)), so a
                # squat — drop with no forward travel — is not a bend.
                th_fwd = float(np.arctan2(d_lean, max(L - max(drop, 0.0), 0.2 * L)))
                th_cap = float(np.arcsin(np.clip(max(d_lean, 0.0) / TORSO_LEAN_PIVOT_MIN, 0.0, 1.0)))
                if self.base_in_loop:
                    theta = max(th_fwd, min(th_drop, th_cap))
                else:                                   # LOCAL: the drop (past the gait bob) is the bend
                    th_drop = float(np.arccos(np.clip(1.0 - max(drop - TORSO_LOCAL_DROP_DEADBAND, 0.0) / L, -1.0, 1.0)))
                    theta = min(th_drop, th_cap)
                self._lean_drop = L * (1.0 - np.cos(theta))
                info["lean"] = theta
            if TORSO_FOLLOW:
                if TORSO_LEAN_MODE == "angle":
                    t_new = self.torso.command_angle(
                        TORSO_LEAN_GAIN * _soft_deadband(theta, TORSO_LEAN_DEADBAND))
                else:
                    t_new = self.torso.command(_soft_deadband(fwd_in, TORSO_DEADBAND))
                if dt is not None:                  # slew: MAX_TORSO_IK_SPEED's tip-over reason
                    t_cur = float(data.ctrl[self.torso_act])
                    step = TORSO_FOLLOW_SPEED * float(dt)
                    t_new = t_cur + float(np.clip(t_new - t_cur, -step, step))
                data.ctrl[self.torso_act] = t_new
                info["torso"] = True
            # 2) BASE — takes the overflow beyond the box (weighted; backward
            #    sensitive). GLOBAL scope only: in LOCAL the base is not in the
            #    follow loop at all and stays wherever the joystick left it.
            if BASE_FOLLOW and _HAVE_SWERVE and self.base_in_loop:
                # REACH-AWARE: scale the box overflow by how badly the ARMS need
                # the help (Haviland RA-L 2022's 1/||e|| insight, applied to a
                # cascade instead of a QP). The box only knows where the HEAD
                # is; `reach_demand` knows whether the arms are actually
                # stretched, which is the thing that should decide whether the
                # base leans in. assist = 1 leaves the pure geometric box.
                assist = 1.0 + BASE_REACH_ASSIST * float(reach_demand)
                if BASE_TRANSLATION_MODE == "body":
                    assist = 1.0      # 1:1 — reach is the workspace assist's job (bounded debt)
                fwd_w_in = float(np.clip(fwd_walk, -BASE_BOX_BACK, BASE_BOX_FWD))
                if self.tracker.one_to_one:                 # 1:1: no box, no reach scaling
                    assist, fwd_w_in, lat_in = 1.0, 0.0, 0.0
                    piv = self.tracker.neck_pivot(head_mat)
                    if piv is not None:
                        # the NECK pivot, not the eyes: a +-50 deg look swings the eyes ~7 cm sideways and the
                        # 1:1 loop drove the base 0.23 m with the operator standing still (wb_direction_check)
                        if getattr(self.tracker, "pivot0", None) is None:
                            self.tracker.pivot0 = piv
                        f_p, lat = self.tracker.head_offset(piv, body_yaw, anchor=self.tracker.pivot0)
                        fwd_walk = f_p - (fwd - fwd_walk)   # the same bend explanation as the head's
                self.base_tgt = self.tracker.command(
                    fwd_walk * assist, lat * assist, fwd_w_in, lat_in, body_yaw,
                    dt=follow_dt if follow_dt is not None else dt)
                info["base"] = True
                info["reach_demand"] = float(reach_demand)
        # 3) LIFT — speed-limited: head height, rate-limited toward target. GLOBAL scope only:
        #    upper body holds the column where it is (the operator's height is theirs, not the robot's).
        if LIFT_FOLLOW and self.base_in_loop and head_height is not None:
            if self.head_h0 is None:
                self.head_h0 = float(head_height)
            # a lean lowers the head by L(1-cos theta); the robot's own torso bend
            # reproduces that, so it is not also "the operator got shorter"
            rise = _soft_deadband(float(head_height) - self.head_h0
                                  + getattr(self, "_lean_drop", 0.0), LIFT_DEADBAND)
            lift_tgt = float(np.clip(self.lift0 + LIFT_GAIN * rise, *LIFT_RANGE))
            cur = float(data.ctrl[self.lift_act])
            step = LIFT_RATE * (lift_tgt - cur)
            if dt is not None:                      # the physical lift speed, per SECOND
                lim = LIFT_FOLLOW_SPEED * float(dt)
                step = float(np.clip(step, -lim, lim))
            data.ctrl[self.lift_act] = cur + step
            info["lift"] = lift_tgt
        return info

    @property
    def allow_base_assist(self):
        # In relative teleop, arm workspace debt must not drive a stationary
        # person around. Its arm mapping remains active; base assistance is
        # permitted only while the person is actually translating.
        # 1:1 position follow: never. Measured 2026-10-29 (NOVA take 13-03-26): with the operator standing
        # still after a step, the assist's "far+lateral" debt drove the base 0.22-0.28 m/s back the way it
        # came — the "translation goes the wrong way" report.
        if self.tracker.one_to_one:
            return False
        return self.tracker.follow_mode != "relative" or self.tracker.relative.translation_active

    def set_base_in_loop(self, on, data):
        """Switch the base in/out of the follow loop, BUMPLESSLY.

        The tracker integrates the operator's head displacement from an anchor,
        so a base that has been out of the loop while the operator walked around
        has accumulated an arbitrarily large offset. Switching it back in without
        re-anchoring would command the base to sprint to catch up on motion it
        deliberately ignored — the switch has to mean "start following from HERE",
        not "make up for everything you missed". Leaving the loop just parks the
        velocity target; the base holds where it is (the joystick owns it there).
        """
        on = bool(on)
        if on == self.base_in_loop:
            return self.base_in_loop
        self.base_in_loop = on
        self.base_tgt = (0.0, 0.0, 0.0)
        self.base_anchored = False        # re-anchor on the next frame, at the live pose
        self.tracker.head0 = None
        self.head_h0 = None               # the lift too: it follows from the live height, from HERE
        self.lift0 = float(data.ctrl[self.lift_act])
        return self.base_in_loop

    def base_step(self, data, dt, extra_wz=0.0):
        """Per PHYSICS-STEP: filter the velocity TARGET through the optional safety
        barrier (reference governor — so the bound sets the steady speed instead of
        compounding on the realized velocity), ramp the realized velocity toward it
        through the Layer-1 motion model (lift-height aware, tip-safe), then drive the
        swerve modules."""
        if not (BASE_FOLLOW and _HAVE_SWERVE):
            return
        z_lift = float(data.ctrl[self.lift_act])
        tgt = self.base_tgt
        if extra_wz:
            # additive yaw rate (the neck handoff) — NOT written back into base_tgt,
            # which persists between ticks and would integrate it
            tgt = (tgt[0], tgt[1], float(np.clip(tgt[2] + extra_wz, -BASE_WMAX, BASE_WMAX)))
        if self.base_filter is not None:
            tgt = tuple(self.base_filter(data, tgt))       # control-barrier on the reference
        self.base_cmd = self.motion_model.step(self.base_cmd, tgt, dt, z_lift)
        swerve_ik(*self.base_cmd, data)
