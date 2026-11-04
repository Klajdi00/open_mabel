"""
TeleopController — the single, shared two-mode whole-body controller for MABEL.

Both the desktop sim viewer (teleop_vision_pro.py) and the Apple Vision Pro bridge
(server/sim_teleop_bridge.py) drive this ONE class, so the control
logic can never drift between them again. Each consumer owns its own real-time
accumulator loop and calls `step(frame, joystick, dt)` once per physics substep.

Two operator modes (toggle from the headset dock or the keyboard):

  * MANIPULATION — the arms are solved as 8 DOF per side (7 arm joints + wrist;
    lift and torso are excluded from the fast IK). The wrist is tracked
    wrist->palm in SE(3); the grasp is decoupled (driven straight from the
    pinch scalar). The lift, torso and mobile base are *lazy* followers
    (bounding-box) that creep only to keep the hands in a comfortable range.

  * NAVIGATION — drone-style virtual joysticks drive the base and lift directly:
    left stick = translate (fwd/back + strafe), right stick = yaw (x) + lift (y).
    The arms hold their pose.

Active perception (gaze -> look-at -> 3-DOF neck) runs in BOTH modes, so the head
always tracks where the operator is looking.

It composes the existing primitives (WholeBodyIK, Retargeter, LazyBase, LazyLift,
LazyTorso, NeckController, swerve_ik, apply_grasp) — no control math is reinvented.
"""
import numpy as np
import mujoco

from . import config as C
from .whole_body_controller.ik import WholeBodyIK
from .whole_body_retargeting.followers import LazyBase, LazyLift, LazyTorso
from .whole_body_retargeting.neck import NeckController, look_at_point
from .whole_body_retargeting.wrist_clutch import Retargeter
from .hand_retargeting.grasp import apply_grasp
from .hand_retargeting.vector import make_hand_retargeter
from .whole_body_retargeting.arm_cosine import ArmRetargeter
from .whole_body_controller.base_controller.swerve import swerve_ik


def _approach(cur, tgt, step):
    """Move `cur` toward `tgt` by at most `step` (slew-rate limiter)."""
    return min(tgt, cur + step) if cur < tgt else max(tgt, cur - step)


class TeleopController:
    MODE_MANIP = "manipulation"
    MODE_NAV = "navigation"
    REGION_ARM = "arm"
    REGION_UPPER = "upper"
    REGION_WHOLE = "whole"

    def __init__(self, model, data, base_follow=True):
        self.model = model
        self.data = data
        self.ik = WholeBodyIK(model, data)
        self.retgt = Retargeter(model, data)
        # Per-finger ORCA hand retargeting (decoupled from the wrist SE(3)
        # mapping): keypoint IK in a canonical palm frame, ported from the
        # proven dexterous-hand planner (see hand_ik_retarget.py). Engaged
        # whenever a frame carries the full hand skeleton; pinch-grasp remains
        # the fallback for skeleton-less sources. Targets stream to the hand's
        # position-servo (PD) actuators. The solver family (keypoint-IK vs the
        # AnyTeleop-style vector optimizer) is selected by config.FINGER_RETARGET.
        self.hand_retgt = make_hand_retargeter(model, data)
        # Arm IK family (config.ARM_RETARGET): "cosine" = the ported directional
        # contact-point ArmRetargeter (parity with scripts/avp_teleop.py); "dls" = the
        # legacy clutch-relative palm SE(3) target + DLS WholeBodyIK. In cosine mode the
        # ArmRetargeter writes the arm + ORCA-wrist actuators directly; the lazy
        # base/torso/lift followers are unchanged (they track the contact target).
        self.arm = {}
        self.arm_act = {}
        self.wrist_act = {}
        if C.ARM_RETARGET == "cosine":
            for s, full, Side in (("l", "left", "Left"), ("r", "right", "Right")):
                self.arm[s] = ArmRetargeter(model, full)
                self.arm_act[s] = [model.actuator(f"{Side} Arm {i}").id for i in range(1, 8)]
                self.wrist_act[s] = model.actuator(f"{Side} Wrist").id
        self.neck = NeckController(model, data)
        self.lazy_base = LazyBase()
        self.lazy_lift = LazyLift()
        self.lazy_torso = LazyTorso()

        self.base_follow = base_follow          # lazy base on/off (manip mode)
        self.profile = None                     # active per-operator OperatorProfile
        self.mode = self.MODE_MANIP
        # Reach region (manip mode): which extra DOF the reach-assist may drive.
        #   arm   -> arms only (lift/torso/base held)
        #   upper -> + lift + torso
        #   whole -> + mobile base
        self.region = self.REGION_WHOLE
        self.vx = self.vy = self.omega = 0.0    # smoothed base twist
        self.grasp_l = self.grasp_r = 0.0

    # ── mode / calibration ───────────────────────────────────────────────────
    @property
    def calibrated(self):
        return self.retgt.calibrated

    def calibrate(self, frame):
        """(Re)anchor the clutch at the operator's current pose, no jump."""
        ok = self.retgt.calibrate(frame)
        if ok:
            self.ik.sync_to_current()
        return ok

    def apply_profile(self, profile):
        """Push a per-operator OperatorProfile (calibration.py) into the retargeting
        scale, the shoulder anchor, the per-finger hand retargeter, and the grasp
        curve. profile=None reverts every consumer to its size-agnostic default."""
        self.profile = profile
        self.retgt.set_profile(profile)
        self.hand_retgt.set_profile(profile)
        return profile

    def set_mode(self, mode):
        if mode in (self.MODE_MANIP, self.MODE_NAV) and mode != self.mode:
            self.mode = mode
            self.ik.sync_to_current()           # hand control back without a snap

    def toggle_mode(self):
        self.set_mode(self.MODE_NAV if self.mode == self.MODE_MANIP else self.MODE_MANIP)

    def set_region(self, region):
        if region in (self.REGION_ARM, self.REGION_UPPER, self.REGION_WHOLE):
            self.region = region

    # ── self-collision avoidance (delegates to the IK; OFF by default) ─────────
    @property
    def self_collision(self):
        return self.ik.self_collision

    def set_self_collision(self, on, method=None):
        """Turn self-collision avoidance on/off live. When on it STOPS the hand at
        the safety margin along the operator's intended path (no force-field
        deflection); `method` may switch the law ("stop" | "cbf")."""
        return self.ik.set_self_collision(on, method=method)

    def toggle_self_collision(self):
        return self.ik.toggle_self_collision()

    def hold_current(self):
        """Snap targets AND actuator commands to the current pose. Call after a
        sim reset (mj_resetData zeros data.ctrl, so the position actuators would
        otherwise drive the robot away from the freshly-reset configuration)."""
        self.ik.sync_to_current()
        self.vx = self.vy = self.omega = 0.0
        for name in list(C.IK_ACTUATORS) + list(C.NECK_ACTUATORS):
            joint = C.LIFT_JOINT if name == "Lift" else name
            try:
                q = float(self.data.joint(joint).qpos[0])
                aid = self.model.actuator(name).id
                lo, hi = self.model.actuator_ctrlrange[aid]
                self.data.actuator(name).ctrl[0] = float(np.clip(q, lo, hi))
            except Exception:
                pass

    def settle(self, steps=None):
        """Hold the start pose on the position servos and step physics until the
        arm reaches gravity equilibrium, then re-capture the hold target. Call once
        before the control loop so the robot does not visibly creep on launch as the
        IK target (captured at the un-settled XML pose) is corrected against sag."""
        steps = C.IK_SETTLE_STEPS if steps is None else steps
        self.hold_current()
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)
        self.hold_current()

    # ── main entry: one control substep ──────────────────────────────────────
    def step(self, frame, joystick, dt):
        """Advance the controllers one physics tick.

        frame    : AVPFrame-like (head + hand transforms) or None
        joystick : (lx, ly, rx, ry) in [-1, 1] for navigation, or None
        """
        # DRIVE (navigation) mode does NOT use the arms/hands: skip hand-retargeting
        # and arm IK entirely (the operator drives with joysticks; the arms just hold
        # their pose in _navigate). The neck still tracks the head via _active_perception
        # with no hand targets. This is the control work dropped in drive mode; physics
        # is untouched. (Manipulation runs the full retarget + whole-body IK.)
        if self.mode == self.MODE_NAV:
            self._active_perception(None, dt)
            self._navigate(joystick, dt)
        else:
            # The bridge calls step() once per physics SUBSTEP (~5× per fresh
            # frame); the retarget mapping of the SAME frame is identical, so
            # memoize on frame identity — saves ~4 redundant updates per frame.
            if frame is not None and frame is getattr(self, "_tgt_frame", None):
                targets = self._tgt_cache
            else:
                targets = self.retgt.update(frame) if self.retgt.calibrated else None
                self._tgt_frame = frame
                self._tgt_cache = targets
            self._active_perception(targets, dt)
            self._manipulate(targets, dt, frame=frame, joystick=joystick)

        self._update_markers()

    # ── manipulation ─────────────────────────────────────────────────────────
    def _manipulate(self, targets, dt, frame=None, joystick=None):
        m, d = self.model, self.data
        # The navigation joystick PERSISTS in manipulation mode (the headset's
        # unified teleop): a deliberate stick deflection drives the base (and
        # lift) directly, overriding the lazy follower below.
        stick = None
        if joystick is not None:
            lx, ly, rx, ry = (self._deadzone(v) for v in joystick)
            if any(abs(v) > 0.0 for v in (lx, ly, rx, ry)):
                stick = (lx, ly, rx, ry)
        if targets is None:
            if stick is not None:
                self._drive_base(stick[1] * C.MAX_LIN_SPEED, stick[0] * C.MAX_LIN_SPEED,
                                 stick[2] * C.MAX_ANG_SPEED, dt)
                self._drive_lift(stick[3] * C.NAV_MAX_LIFT, dt)
            else:
                self._drive_base(0.0, 0.0, 0.0, dt)  # nothing to track: hold still
            return

        # Arms + fingers. Both paths set self.ik.target_{l,r}_pos to the tracked hand
        # target so the SHARED lazy followers (below) react identically; only the arm
        # IK differs (cosine ArmRetargeter vs DLS WholeBodyIK).
        if self.arm:
            self._arms_cosine(targets, frame)
        else:
            self._arms_dls(targets, frame, dt)

        # Reach-assist scope is set by the region, and only engages while the
        # operator is actively moving the hands (idle => the body holds still):
        #   arm   -> nothing follows (lift/torso/base held)
        #   upper -> lift + torso follow
        #   whole -> + mobile base follows
        commanding = self._targets_moving(self.ik.target_l_pos, self.ik.target_r_pos)
        tvx = tvy = tw = 0.0
        if commanding and self.region in (self.REGION_UPPER, self.REGION_WHOLE):
            self._drive_lift(self.lazy_lift.command(self.ik.target_l_pos, self.ik.target_r_pos, d), dt)
            self._drive_torso(self.lazy_torso.command(self.ik.target_l_pos, self.ik.target_r_pos, d), dt)
        if commanding and self.region == self.REGION_WHOLE and self.base_follow:
            tvx, tvy, tw = self.lazy_base.command(self.ik.target_l_pos, self.ik.target_r_pos, d)
        if stick is not None:
            # Operator stick beats the lazy follower (unified teleop).
            tvx = stick[1] * C.MAX_LIN_SPEED
            tvy = stick[0] * C.MAX_LIN_SPEED
            tw = stick[2] * C.MAX_ANG_SPEED
            self._drive_lift(stick[3] * C.NAV_MAX_LIFT, dt)
        self._drive_base(tvx, tvy, tw, dt)

    # ── arm IK: cosine (avp_teleop parity) vs legacy DLS ──────────────────────
    def _fingers(self, side, Side, targets, frame):
        """Per-finger ORCA retargeting from the named skeleton, pinch-grasp fallback."""
        if f"{side}_pos" not in targets:
            return
        g = targets.get(f"grasp_{side}", 0.0)
        setattr(self, f"grasp_{side}", g)
        joints = getattr(frame, f"{Side.lower()}_joints", None) if frame is not None else None
        if joints:
            self.hand_retgt.apply(Side, joints)
        else:
            apply_grasp(self.model, self.data, Side, g)

    def _arms_cosine(self, targets, frame):
        """Cosine-directional contact-point arm IK (ported from avp_teleop.py): solve
        each arm + ORCA wrist directly and stream to the actuators. The lazy followers
        track the contact target (self.ik.target_*_pos), so the body still follows."""
        d = self.data
        for side, Side, palm in (("r", "Right", C.RIGHT_PALM), ("l", "Left", C.LEFT_PALM)):
            tgt = "target_r_pos" if side == "r" else "target_l_pos"
            hk = self.retgt.arm_hk(side, frame)
            if hk is not None:
                ar = self.arm[side]
                ar.sync(d)
                sol = ar.solve(hk)
                for i, a in enumerate(self.arm_act[side]):
                    if i < len(sol):
                        d.ctrl[a] = float(sol[i])
                d.ctrl[self.wrist_act[side]] = float(ar.ik.qpos[ar.orca_wrist_qadr])
                setattr(self.ik, tgt, np.asarray(hk["contact"], float))
            else:
                setattr(self.ik, tgt, d.body(palm).xpos.copy())   # untracked -> hold
            self._fingers(side, Side, targets, frame)

    def _arms_dls(self, targets, frame, dt):
        """Legacy clutch-relative palm SE(3) targets + DLS WholeBodyIK (8-DOF arms)."""
        m, d = self.model, self.data
        if "r_pos" in targets:
            self.ik.target_r_pos = targets["r_pos"]
            self.ik.target_r_mat = targets["r_mat"]
            self._fingers("r", "Right", targets, frame)
        else:
            self.ik.target_r_pos = d.body(C.RIGHT_PALM).xpos.copy()
        if "l_pos" in targets:
            self.ik.target_l_pos = targets["l_pos"]
            self.ik.target_l_mat = targets["l_mat"]
            self._fingers("l", "Left", targets, frame)
        else:
            self.ik.target_l_pos = d.body(C.LEFT_PALM).xpos.copy()
        # 8-DOF arms (lift + torso excluded -> handled lazily below).
        self.ik.apply(self.ik.solve(arms_only=True), dt, arms_only=True)

    def _targets_moving(self, l_pos, r_pos):
        """True if the hand targets moved more than LAZY_CMD_EPS since last tick."""
        prev = getattr(self, "_prev_targets", None)
        moved = prev is None or (np.linalg.norm(l_pos - prev[0])
                                 + np.linalg.norm(r_pos - prev[1])) > C.LAZY_CMD_EPS
        self._prev_targets = (np.array(l_pos), np.array(r_pos))
        return moved

    # ── navigation (drone joysticks) ─────────────────────────────────────────
    def _navigate(self, joystick, dt):
        self.ik.sync_to_current()               # arms hold their current pose
        lx = ly = rx = ry = 0.0
        if joystick is not None:
            lx, ly, rx, ry = (self._deadzone(v) for v in joystick)
        # Strictly mirror scripts/teleop_keyboard.py (mode 1, DIRECT NAV): the
        # target twist is stick × ceiling, then the SAME tip-safe smoother, then
        # the SAME swerve IK — so the base behaves identically to the keyboard demo.
        tvx = ly * C.MAX_LIN_SPEED              # left stick Y → forward/back
        tvy = lx * C.MAX_LIN_SPEED              # left stick X → strafe
        tw  = rx * C.MAX_ANG_SPEED              # right stick X → yaw
        z_lift = float(self.data.actuator("Lift").ctrl[0])
        self.vx, self.vy, self.omega = C.safe_base_step(
            self.vx, self.vy, self.omega, tvx, tvy, tw, dt, z_lift)
        swerve_ik(self.vx, self.vy, self.omega, self.data)
        self._drive_lift(ry * C.NAV_MAX_LIFT, dt)

    # ── active perception (both modes) ───────────────────────────────────────
    def _active_perception(self, targets, dt):
        if targets is None or "gaze_dir" not in targets:
            return
        head_p = self.data.body(C.HEAD_BODY).xpos.copy()
        lap = look_at_point(head_p, targets["gaze_dir"], self.model, self.data,
                            exclude_body=C.HEAD_BODY)
        self.neck.solve_and_apply(lap, dt)
        self._set_mocap("gaze_point", lap)

    # ── low-level drives ─────────────────────────────────────────────────────
    def _drive_base(self, tvx, tvy, tw, dt):
        self.vx = _approach(self.vx, tvx, C.MAX_LIN_ACCEL * dt)
        self.vy = _approach(self.vy, tvy, C.MAX_LIN_ACCEL * dt)
        self.omega = _approach(self.omega, tw, C.MAX_ANG_ACCEL * dt)
        swerve_ik(self.vx, self.vy, self.omega, self.data)

    def _drive_lift(self, v_lift, dt):
        self._integrate_actuator("Lift", v_lift, dt)

    def _drive_torso(self, v_torso, dt):
        self._integrate_actuator("Torso", v_torso, dt)

    def _integrate_actuator(self, name, vel, dt):
        if abs(vel) < 1e-9:
            return
        aid = self.model.actuator(name).id
        lo, hi = self.model.actuator_ctrlrange[aid]
        cur = float(self.data.actuator(name).ctrl[0])
        self.data.actuator(name).ctrl[0] = float(np.clip(cur + vel * dt, lo, hi))

    # ── visualization markers (no-op if the mocap bodies aren't in the scene) ─
    def _update_markers(self):
        self._set_mocap("target_l", self.ik.target_l_pos, self.ik.target_l_mat)
        self._set_mocap("target_r", self.ik.target_r_pos, self.ik.target_r_mat)

    def _set_mocap(self, name, pos, mat=None):
        try:
            mid = int(self.model.body(name).mocapid[0])
        except KeyError:
            return
        if mid < 0:
            return
        self.data.mocap_pos[mid] = pos
        if mat is not None:
            q = np.zeros(4)
            mujoco.mju_mat2Quat(q, np.asarray(mat).flatten())
            self.data.mocap_quat[mid] = q

    @staticmethod
    def _deadzone(v):
        return 0.0 if abs(v) < C.NAV_DEADZONE else float(v)

    # ── telemetry for the dashboard / status line ────────────────────────────
    def status(self):
        return {
            "mode": self.mode,
            "calibrated": self.calibrated,
            "self_collision": self.self_collision,
            "palm_err_l": float(np.linalg.norm(self.ik.e_pos_l)),
            "palm_err_r": float(np.linalg.norm(self.ik.e_pos_r)),
            "grasp_l": self.grasp_l,
            "grasp_r": self.grasp_r,
            "base": (self.vx, self.vy, self.omega),
            "rot_diag": dict(self.retgt.rot_diag),     # (operator_deg, palm_deg) per hand
            "e_lift": self.lazy_lift.e_up,
            "e_torso_fwd": self.lazy_torso.e_fwd,
        }
