"""TeleopEngine — the shared, DEVICE-AGNOSTIC teleop control stack (frame-paced engine).

It drives the robot for EVERY client — web, iOS, gamepad, AND the Vision Pro headset —
not just VR; the operator pose / nav joystick arrive over the same wire regardless of
device. (It is the controller the MabelTeleop facade composes as `self.controller`.)

This is the EXACT retargeting + whole-body control that scripts/avp_teleop.py runs
(HandOptimizer fingers, cosine-directional ArmRetargeter, WholeBodyController cascade,
gravity-comp, EngagementSupervisor fail-safe, double-pinch clutch), packaged so the
WebSocket bridge (server/sim_teleop_bridge.py) can drive the SAME controller the apps'
operators expect — instead of the separate mabel facade.

LATENCY / SPEED is the whole point of the split between `_retarget` and the per-substep
work in `update`:

  * The HEAVY retargeting (two SLSQP finger solves + two SLSQP arm solves + neck) runs
    ONCE PER VR FRAME — memoized on the operator frame's timestamp. The bridge calls
    `update()` once per physics SUBSTEP (~5-16x per frame); re-solving SLSQP every
    substep is what made the earlier port crawl.
  * Per SUBSTEP we only do the cheap physics: the lift-aware base velocity ramp +
    swerve, and the safety blend. Gravity compensation is recomputed once per frame
    (it persists in qfrc_applied across the substeps).

Input is the frontend-agnostic ``mabel.types.OperatorPose`` the bridge already builds
from the wire; the engine converts it to the teleop payload internally (the 21-landmark
hand array from the named ARKit skeleton, head height from the head Y, etc.). ARKit on
the headset/phone has no elbow or body odometry, so those terms simply switch off (the
torso/base still follow the HEAD displacement, the lift follows head height).

Calibration is ONE pose: extend both arms forward; the engine measures the operator's
head->wrist reach against the robot's and sets the global REACH_SCALE (see
``start_calibration`` / model_utils.human_reach + robot_forward_reach).
"""
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import mujoco

from .hyperparams import *
from .deps import MC, BodyFrameEstimator, _HAVE_BODYFRAME, _HAVE_KEYPOINT
from .hand import HandOptimizer
from .neck_handoff import NeckYawHandoff
from .hand_anatomical import AnatomicalHandRetargeter
from .arm import ArmRetargeter
from .wholebody import WholeBodyController
from .motion_model import BaseMotionModel
from .safety import EngagementSupervisor, ClutchController, ChannelPark
from .geometry import (operator_frame, neck_angles, clamp_in_front,
                       clamp_shoulder_standoff, clamp_torso_standoff,
                       ElbowYawEstimator, ShoulderEstimator, palm_axes,
                       synth_elbow, estimate_swivel)
from .model_utils import (DirectionalReach, position_actuators, gravity_dofs, human_reach,
                          robot_forward_reach, robot_arm_max_reach, symmetric_reach)
from ..dynamic_models.gravity import UpperBodyGravityComp
from .logsetup import logger


# Named ARKit skeleton -> MediaPipe-21 landmark array (the layout HandOptimizer +
# operator_frame index into: [0]=wrist, [4]=thumb tip, [8]=index tip, [12]=middle tip).
_NAMES_21 = [
    "wrist",
    "thumb_knuckle", "thumb_intermediate_base", "thumb_intermediate_tip", "thumb_tip",
    "index_knuckle", "index_intermediate_base", "index_intermediate_tip", "index_tip",
    "middle_knuckle", "middle_intermediate_base", "middle_intermediate_tip", "middle_tip",
    "ring_knuckle", "ring_intermediate_base", "ring_intermediate_tip", "ring_tip",
    "little_knuckle", "little_intermediate_base", "little_intermediate_tip", "little_tip",
]

# Stick → base/lift command magnitudes (apps' navJoystick, unified teleop).
_STICK_LIFT_RATE = 0.20      # m/s lift jog at full right-stick Y


class TeleopEngine:
    def __init__(self, model, data, lock_j7=False, arm_map=None, finger_solver=None):
        self.model = model
        self.data = data
        # THE upper-body gravity-compensation layer — the SAME object the torque
        # engine uses (mabel.dynamic_models.gravity), so the compensated DoF set,
        # gain and rate cannot differ between control spaces. `kdata` is kept as an
        # alias of its private scratch state for the tools that reach for it.
        self.gravcomp = UpperBodyGravityComp(model)
        self.gravity_comp = True          # operator switch (set_gravity_comp)
        self.whole_body_retarget = False  # False = the wrist-focused profile
        self.kdata = self.gravcomp._scratch          # gravity-comp scratch (alias)
        # HandOptimizer is kept for the canonical robot palm frame (R_rob, incl. the
        # flexion-anchored chirality fix) the ARM retargeter aligns to. The FINGERS
        # run the dexterous_hand-planner keypoint IK by default (hand_keypoint.py →
        # mabel.hand_retargeting.keypoint_ik, the planner_avp/planner_per_finger
        # port), falling back to the webcam-parity anatomical decomposition
        # (hand_anatomical.py) when the controller package is absent or
        # MABEL_FINGER_SOLVER=anatomical.
        self.opt = {s: HandOptimizer(model, s) for s in ("right", "left")}
        self.finger_solver = (finger_solver or FINGER_SOLVER)
        if self.finger_solver == "keypoint" and _HAVE_KEYPOINT:
            from .hand_keypoint import KeypointHandRetargeter
            self.hand = {s: KeypointHandRetargeter(model, s) for s in ("right", "left")}
        else:
            if self.finger_solver == "keypoint":
                logger.warning("keypoint finger solver requested but the controller "
                               "package is unavailable — falling back to anatomical")
            self.finger_solver = "anatomical"
            self.hand = {s: AnatomicalHandRetargeter(model, s) for s in ("right", "left")}
        self.arm = {s: ArmRetargeter(model, s, self.opt[s].R_rob, lock_j7)
                    for s in ("right", "left")}
        self.lock_j7 = lock_j7
        # Extension-fraction arm mapping (ARM_MAP == "extension"): live operator
        # shoulder anchors + the robot's own per-side usable reach L_rob.
        self.arm_map = (arm_map or ARM_MAP)
        self.shoulder = ShoulderEstimator()
        self.arm_reach = symmetric_reach({s: robot_arm_max_reach(model, s) for s in ("right", "left")})
        # Direction-dependent reach for the extension-exact blend. The scalar
        # above stays the sigma of the shape-preserving map; only the "full
        # extension is exact" magnitude consults this.
        self.arm_reach_dir = ({s: DirectionalReach(model, s) for s in ("right", "left")}
                              if ARM_REACH_DIRECTIONAL else None)
        # END EFFECTOR (end_effector.py): "orca" (ORCA hand v1, the default) or "gripper" (the OpenArm v1
        # gripper on J7, tracked at its base face, end_effector.tracked_point). Each keeps its own reach tables, measured to its own
        # tracked point, so "full human extension = full robot extension" holds for both.
        self.ee = "orca"
        self._reach_by_ee = {"orca": (self.arm_reach, self.arm_reach_dir)}
        from .end_effector import GripperAperture
        self.gripper_aperture = {s: GripperAperture() for s in ("right", "left")}
        # New contact geometry/precision is versioned by the caller. Historical
        # shipped variants retain their exact ORCA-envelope behavior by default.
        self.enable_gripper_contact = False
        # RELATIVE INCREMENT (Retarget Lab "rel", 2026-10-23): latch at engage, then move by the operator's change
        # (_rel_targets). Off by default: every other variant is unchanged.
        self.rel_track = False
        self._rel, self._rel_seen, self._rel_axes = {}, {}, {}
        # One finger's slide position in metres (0..0.044), not total aperture.
        # Adapters clear unavailable/stale feedback to None; unknown jaws keep
        # the conservative complete-stroke envelope.
        self.gripper_measured = {s: None for s in ("right", "left")}
        self.gripper_commanded = {s: None for s in ("right", "left")}
        # Ordinary Lab runs declare their virtual plant as the measured state.
        # Hardware adapters explicitly require fresh complete physical feedback;
        # an unavailable sample must never fall back to the virtual arm pose.
        self.arm_feedback_required = False
        self.arm_measured = {}
        self.arm_feedback_valid = False
        self.arm_feedback_reason = ""
        self.arm_feedback_age_s = None
        self.arm_feedback_compute_s = 0.0
        self.physical_feedback_refresh = None
        logger.info("arm map '%s' | robot arm reach R %.3f m / L %.3f m | fingers '%s'",
                    self.arm_map, self.arm_reach["right"], self.arm_reach["left"],
                    self.finger_solver)

        def aid(n):
            return model.actuator(n).id
        self.arm_act = {s: [aid(f"{s.capitalize()} Arm {i}") for i in range(1, 8)]
                        for s in ("right", "left")}
        self.wrist_act = {s: aid(f"{s.capitalize()} Wrist") for s in ("right", "left")}
        self.finger_act = {s: [aid(n) for n in self.hand[s].joint_names]
                           for s in ("right", "left")}
        self.neck_act = [aid(n) for n in ("Neck Yaw", "Neck Pitch", "Neck Roll")]
        self.neck_rng = [model.actuator_ctrlrange[a] for a in self.neck_act]
        self.lift_act, self.torso_act = aid("Lift"), aid("Torso")
        # The head rides the TORSO, so the neck has to know where the torso is
        # (see the gaze compensation in _retarget).
        _tj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "Torso")
        self.torso_qadr = model.jnt_qposadr[_tj]
        self.torso_rest = float(model.qpos0[self.torso_qadr])
        # Hardware adapters may supply the actual joint angle here. The
        # kinematic torso can lead the physical joint during motion, so using
        # its pose for camera compensation would tilt the real viewing axis.
        # None retains the lab's own torso pose as the feedback source.
        self.neck_torso_measured = None

        # Where each tracking channel eases to when it is switched OFF.
        #   NECK  level, looking straight ahead
        #   ARMS  the gravity-equilibrium rest (arms down at the sides) PLUS the
        #         two big DoFs the operator called part of the same channel:
        #         torso upright and the lift down
        #   HANDS fingers open
        _arm_aids, _arm_neutral = [], []
        for side in ("right", "left"):
            for i, a in enumerate(self.arm_act[side]):
                _arm_aids.append(a)
                _arm_neutral.append(NEUTRAL_ARM[f"{side}_arm_{i + 1}"])
            _arm_aids.append(self.wrist_act[side])
            _arm_neutral.append(NEUTRAL_WRIST[f"{side}_wrist"])
        _arm_aids += [self.torso_act, self.lift_act]
        _arm_neutral += [NEUTRAL_TORSO, NEUTRAL_LIFT]
        # The lift's HOME height: the model's home keyframe (the half-extended
        # default every MjData boot seeds), so a parked lift is the SAME column
        # height in every trial of an ablation — not wherever it happened to be.
        _lift_home = (float(model.key_ctrl[0][self.lift_act]) if model.nkey > 0
                      else float(data.ctrl[self.lift_act]))
        self.lift_home = _lift_home
        self.park = {
            "neck": ChannelPark(self.neck_act, [NEUTRAL_NECK] * len(self.neck_act)),
            # ABLATION channels (2026-10-16): the same retargeter runs; a channel
            # switched off has its joints parked at HOME after it — torso upright
            # at 0, lift at the home height. (The base is the scope's job.)
            "torso": ChannelPark([self.torso_act], [NEUTRAL_TORSO]),
            "lift": ChannelPark([self.lift_act], [_lift_home]),
            "arms": ChannelPark(_arm_aids, _arm_neutral),
            "hands": ChannelPark(
                [a for s in ("right", "left") for a in self.finger_act[s]],
                [NEUTRAL_GRIP for s in ("right", "left") for _ in self.finger_act[s]]),
        }

        self.wbc = WholeBodyController(model, data, torso_act=self.torso_act,
                                       lift_act=self.lift_act, motion_model=BaseMotionModel())
        self.sup = EngagementSupervisor(model, self.lift_act)
        self.clutch = ClutchController()
        # Torso-heading estimator for headset clients: only used when the pose
        # carries no body_yaw of its own (the desktop streamer's BodyOdom already
        # publishes one). Preferred estimator is the analytic base-frame estimator
        # (head + two wrists, rejects look-around / reach / gaze-fixed that fool
        # the raw elbow line); ElbowYawEstimator is the fallback when the mabel
        # controller package is absent or the wrists are missing from a frame.
        self.yaw_est = ElbowYawEstimator()
        self.body_est = BodyFrameEstimator() if _HAVE_BODYFRAME else None
        # env escape hatch: set MABEL_BODY_FRAME=0 to revert to the elbow line.
        self._use_body_frame = (
            _HAVE_BODYFRAME
            and os.environ.get("MABEL_BODY_FRAME", "1").lower()
            not in ("0", "false", "no", "off"))
        self._body_frame_fresh = False   # was body_est updated last frame?
        self._head_heading = None        # LazyHeadHeading, built on first use (HEADING_MODE "head")
        self.grav_dofs = self.gravcomp.dofs      # the canonical upper-body DoF set
        self.hold = position_actuators(model)
        self.head_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "head")
        self.base_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ROBOT_BASE_BODY)

        self.reach = REACH_SCALE
        self._world_ref = None
        self._head0 = None
        self._neck_ref = None
        self._neck_base_yaw0 = None
        self._rel = {}
        self.neck_handoff = NeckYawHandoff(NECK_YAW_HANDOFF, NECK_HANDOFF_GAIN,
                                           NECK_HANDOFF_WMAX, NECK_HANDOFF_STALL,
                                           enabled=NECK_YAW_HANDOFF_ON)
        self._head_prev = None               # origin-jump guard state
        self._head_stable_since = None
        self._last_ts = None
        self._frame_t = None
        self._frame_dt = 1.0 / 60.0   # set per frame in update(); a sane default
                                      # so _payload never crashes pre-first-frame
                                      # (calibrate_tick can run before update)
        self._prev = time.time()
        self.mode = "engaged"
        from ..whole_body_retargeting.workspace_assist import WorkspaceAssist
        self._ws_assist = WorkspaceAssist()
        self._ws_offset = np.zeros(3)
        self.nav_mode = False             # True in DRIVE: stick-only base/lift, no head-follow
        self.retarget_scope = RETARGET_SCOPE   # 'local' (default) | 'global'
        # Pending "this is straight ahead" datum (see set_forward), and WHEN it
        # was requested — the wait for a head-plus-both-wrists frame is bounded
        # by FORWARD_DEADLINE.
        self._face_forward = False
        self._face_forward_at = 0.0
        # Which operator body parts drive the robot. All on = the classic
        # whole-body behaviour; each can be switched off independently.
        self.track_neck = True
        self.track_arms = True
        self.track_hands = True
        self.track_torso = True
        self.track_lift = True
        self.self_collision = False
        self.projector = None             # ArmCollisionProjector, built on first enable
        # SAFETY DEFAULT: arm self-collision avoidance follows the shipped controller
        # config (mabel.config.SELF_COLLISION_ENABLE, ON by default) so NO entry point
        # can run teleop unguarded by omission. Callers opt OUT explicitly via
        # set_self_collision(False) (bridge: --no-self-collision).
        if MC is not None and getattr(MC, "SELF_COLLISION_ENABLE", False):
            self.set_self_collision(True)
        self._cal = None                  # 1-pose reach calibration state
        self.profile_user = None
        self.profile = None               # OperatorProfile-like dict ({user, reach, ...})
        # The right + left finger/arm SLSQP solves use DISJOINT scratch MjData, so they
        # run in parallel (MuJoCo is thread-safe across distinct MjData) — roughly halves
        # the per-frame retarget latency. Created once; reused every frame.
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="teleop-retarget")

    # ── lifecycle ─────────────────────────────────────────────────────────────
    def init_hold(self):
        """Write the rest-pose targets onto the position servos (call before settle)."""
        for a, q0 in self.hold:
            self.data.ctrl[a] = q0

    def settle_done(self):
        """After the caller settles physics under gravity comp, capture the lift origin."""
        self.wbc.reset(self.data)

    def reset(self):
        for s in ("right", "left"):
            self.opt[s].reset_optimizer()
            self.hand[s].reset()
        self.shoulder.reset()
        self.yaw_est.reset()
        self._world_ref = None
        self._head0 = None
        self._neck_ref = None
        self._neck_base_yaw0 = None
        self._rel = {}
        self.neck_handoff.reset()
        self._head_prev = None
        self._head_stable_since = None
        self._last_ts = None
        self.wbc.reset(self.data)
        self.sup.reset()
        self.clutch.reset()
        self._cal = None

    def recalibrate(self, face=False):
        """Re-anchor the world transform + base + lift on the next STABLE frame
        (no jump).

        `face` additionally re-derives the operator's BODY FACING (see
        `set_forward`). It is OFF by default and ON only for an OPERATOR-driven
        re-anchor, because the two callers want different things:

          operator "Set Forward"   YES — that is the entire point of the press.
          supervisor re-acquire    NO  — tracking coming back after a dropout
                                   must not silently redefine which way is
                                   forward. `calibrate_facing` also cold-starts
                                   the estimator's rate state, so doing it on
                                   every re-acquire measurably degrades its
                                   tracking (arm deviation under a sustained
                                   world rotation: 3.3 deg warm, 19.5 deg cold).
        """
        self._world_ref = None
        self._head0 = None
        self._neck_ref = None
        self._neck_base_yaw0 = None
        self._rel = {}
        self.neck_handoff.reset()
        self.wbc.base_anchored = False
        self.wbc.base_tgt = (0.0, 0.0, 0.0)
        # The lift's head-height origin must re-anchor too — keeping a stale
        # head_h0 across an ARKit origin jump ran the lift to the top.
        # NOTE: deliberately does NOT reset the head-stability window — the facade
        # re-issues calibrate() EVERY frame until anchored, and resetting the
        # window here would starve the anchor forever (arms would never engage).
        # The origin-jump handler resets the window itself.
        self.wbc.head_h0 = None
        # AND THE BODY FACING. This was the bug: `recalibrate` re-anchored the
        # world ref, the base and the lift, but NOT the body-frame estimator —
        # which is calibrated once at session start and integrated from there.
        # So after the operator turned (in global mode, driving the base with
        # them) and pressed reset, the arm map still believed the ORIGINAL
        # facing, and every wrist target came out rotated by however far they
        # had turned: "the robot tries to retarget to the global position of the
        # pose still, not thinking that the human already rotated" (operator,
        # 2026-07-28). Reset now means reset, for every frame the retargeter
        # holds — but only when the OPERATOR asked (see `face`).
        if face:
            self.set_forward()

    # ── TeleopController drop-in adapters ──────────────────────────────────────
    # This engine REPLACES the dated controller/mabel TeleopController inside the
    # MabelTeleop facade. These thin adapters give the facade the same surface it
    # used on the old controller, so the facade stays the single orchestrator
    # (spec routing + WBC torque modes + joint overrides) while the engine owns the
    # retarget/navigation/neck generation + its own clutch + fail-safe + grav-comp.
    MODE_NAV = "navigation"
    MODE_MANIP = "manipulation"

    def step(self, pose, joystick, dt):
        """Alias of update() — the facade calls step() on its kinematic generator."""
        return self.update(pose, joystick, dt)

    @property
    def calibrated(self):
        """True once the world frame has been anchored (first usable frame)."""
        return self._world_ref is not None

    def calibrate(self, pose=None):
        """Re-anchor the operator<->robot mapping at the CURRENT pose on the next
        frame (no jump). The engine re-derives the anchor from the live frame, so
        the pose arg is accepted for API parity but not required.

        This is the OPERATOR-driven path (the wire's `recalibrate`, and the
        first-frame anchor), so it re-derives the body facing too."""
        self.recalibrate(face=True)
        return True

    def set_mode(self, mode):
        """No-op: unlike the old two-mode controller, this engine runs retarget AND
        the nav stick in the SAME update (a deflected stick overrides the base in
        _apply_stick), so there is no NAV/MANIP switch to make."""
        self.nav_mode = (mode == self.MODE_NAV)

    # ── retarget SCOPE: is the mobile base in the follow loop? ────────────────
    SCOPE_LOCAL = "local"
    SCOPE_GLOBAL = "global"

    def set_retarget_scope(self, scope):
        """LOCAL (default) vs GLOBAL whole-body retargeting.

        LOCAL   upper body: the operator is tracked in a frame that moves WITH
                them. Arms + hands + neck follow; the torso bends only for a real
                hip lean (head drop), never for walking; the BASE and the LIFT
                hold (the joystick still drives them). Deliberately the easier
                problem — the arm IK solves against a root that cannot move under it.
        GLOBAL  whole body: tracked in the world. The base takes the overflow
                past the comfort box (translation) and past the yaw deadband
                (orientation), the lift follows head height, the torso takes the
                within-box forward reach.

        The switch is bumpless both ways: the base and the lift re-anchor at the
        live pose rather than catching up on what they ignored while out of the
        loop (2026-10-18; wholebody.update_targets, measured by
        web_gui/retarget_lab/experiments/exp_local_scope.py).

        Returns the normalised scope, so callers can echo what actually took.
        """
        sc = str(scope or "").strip().lower()
        if sc not in (self.SCOPE_LOCAL, self.SCOPE_GLOBAL):
            sc = self.SCOPE_LOCAL          # unknown/None: never grant base authority
        self.retarget_scope = sc
        self.wbc.set_base_in_loop(sc == self.SCOPE_GLOBAL, self.data)
        return sc

    _EE_REACH = {}                                  # (id(model), kind) -> reach tables: a reset keeps them

    def set_end_effector(self, kind):
        """'orca' (ORCA hand v1, default) or 'gripper' (the OpenArm v1 parallel gripper on J7). With the
        gripper the arms track its BASE FACE (end_effector.tracked_point) with the ORCA wrist held at 0, the reach
        tables are re-measured to that point (once, then cached), the finger IK is not run and
        `gripper_open[side]` (m, 0..0.088) follows the operator's hand closure (end_effector.GripperAperture).
        Returns the kind in force."""
        from .end_effector import KINDS
        kind = str(kind or "orca").lower()
        if kind not in KINDS:
            kind = "orca"
        if kind == self.ee:
            return self.ee
        if kind not in self._reach_by_ee:
            self._reach_by_ee[kind] = self.precompute_end_effector(kind)
        self.arm_reach, self.arm_reach_dir = self._reach_by_ee[kind]
        for a in self.arm.values():
            a.set_end_effector(kind)
        if kind == "gripper":
            for s in ("right", "left"):
                self.data.ctrl[self.wrist_act[s]] = 0.0
        self.ee = kind
        logger.info("end effector '%s' | robot arm reach R %.3f m / L %.3f m", kind,
                    self.arm_reach["right"], self.arm_reach["left"])
        return self.ee

    def precompute_end_effector(self, kind="gripper"):
        """Measure (once per model, then cached) the reach tables `kind` tracks with, without switching to it —
        so a later set_end_effector(kind) is instant. Safe off-thread: it only uses its own MjData."""
        from .end_effector import TRACK_OFFSET, tracked_point
        key = (id(self.model), kind, TRACK_OFFSET)
        if key not in TeleopEngine._EE_REACH:
            ee = {s: tracked_point(self.model, s, self.arm[s].palm_bid, self.arm[s].R_rob) for s in ("right", "left")}
            TeleopEngine._EE_REACH[key] = (
                symmetric_reach({s: robot_arm_max_reach(self.model, s, ee=ee[s]) for s in ("right", "left")}),
                ({s: DirectionalReach(self.model, s, ee=ee[s]) for s in ("right", "left")}
                 if ARM_REACH_DIRECTIONAL else None))
        return TeleopEngine._EE_REACH[key]

    @property
    def gripper_open(self):
        """Per side gripper opening [m] (the operator's hand closure), or None with the ORCA hand."""
        if self.ee != "gripper":
            return None
        return {s: float(g.open) for s, g in self.gripper_aperture.items()}

    def set_forward(self):
        """THE OPERATOR'S BODY DATUM. Called when they are facing the direction
        they want to be "straight ahead": the next tracked frame zeroes the
        body-frame estimator's heading there.

        This exists because estimating the operator's torso heading from a head
        and two wrists is genuinely hard — it is the dominant open issue in
        `docs/retargeting_modes.tex` — and it does not have to be guessed when
        the operator can simply declare it. One button, one instant, no filter.
        Everything downstream reads the same datum: the arm map's body frame,
        and (in global mode) the base yaw.

        Deferred to the next frame on purpose: it needs a head and two wrists,
        and the button is pressed with the hands wherever they are. That wait is
        BOUNDED (`FORWARD_DEADLINE`) — a hand out of view must not be able to
        postpone the datum indefinitely, which is what made this feel slow.
        """
        self._face_forward = True
        self._face_forward_at = time.monotonic()
        # …and RE-LATCH the neck's datum here too. Set Forward is the operator
        # declaring which way "straight ahead" is, and the neck's zero is part
        # of that declaration — otherwise the arms would re-zero to the new
        # facing while the head kept its old one.
        self._neck_ref = None
        self._neck_base_yaw0 = None
        return True

    def set_tracking(self, neck=None, arms=None, hands=None, torso=None, lift=None):
        """Enable/disable one retargeting channel at a time. A channel that is
        OFF is not computed at all, and its joints EASE to their neutral pose
        (see `ChannelPark`) rather than freezing wherever the operator left
        them — a frozen limb reads as a crash, and a frozen NECK points the
        cameras at the floor. `None` leaves a channel unchanged.

        These are orthogonal to `region` (which joints the torque engine owns):
        this is which of the operator's BODY PARTS are driving anything at all.
        """
        if neck is not None:
            self.track_neck = bool(neck)
        if arms is not None:
            self.track_arms = bool(arms)
        if hands is not None:
            self.track_hands = bool(hands)
        # TORSO / LIFT (ablation, 2026-10-16): OFF parks the joint at HOME
        # (torso 0, lift at the home-keyframe height) — the same retargeter
        # still runs, only these joints are held. Unlike the other channels
        # their home IS the ablation condition, so "neutral" = home here.
        if torso is not None:
            self.track_torso = bool(torso)
        if lift is not None:
            self.track_lift = bool(lift)
        return {"neck": self.track_neck, "arms": self.track_arms,
                "hands": self.track_hands, "torso": self.track_torso,
                "lift": self.track_lift}

    def set_gravity_comp(self, on):
        """Operator switch for the upper-body gravity feed-forward.

        Orthogonal to every other axis: it does not select a control space, a
        region or a tracking channel, it decides whether the shared layer feeds
        anything at all. OFF means the arm carries its own weight on the position
        servos' stiffness, which is what you want when you are looking for droop,
        comparing against a model, or demonstrating what the compensation buys —
        and is otherwise a worse place to be, so it defaults ON everywhere.
        """
        self.gravity_comp = self.gravcomp.set_enabled(on)
        return self.gravity_comp

    def set_whole_body_retarget(self, on, table_dir=None):
        """Switch the arms between the two retargeting profiles.

        ON  — the ORIGINAL whole-body retargeting: the robot follows the
              operator's whole arm, elbow included.
        OFF — the front-workspace tabular retargeting, which spends the arm's
              redundancy on putting the PALM where the operator points it
              instead of on matching their elbow.

        Both arms switch together (a mixed pair would be a bimanual asymmetry
        nobody asked for), and the change is in place, so the trust region walks
        the arms over continuously.
        """
        name = "whole_body" if on else "wrist"
        for side in self.arm:
            self.arm[side].set_profile(name, table_dir=table_dir)
        self.whole_body_retarget = bool(on)
        return self.whole_body_retarget

    def set_self_collision(self, on, method=None):
        """Enable/disable ARM self-collision avoidance — the SAME capsule guard the
        standalone avp_teleop uses (`ArmCollisionProjector`: a velocity-QP that projects
        the arm IK targets onto the self-collision-safe set, so a hand can't be driven
        through the torso / head / other arm — it slides along the obstacle rather than
        stopping; fast-paths to a no-op when nothing is near). Built on first enable.
        (The MabelTeleop facade flips this AND the WBC torque-domain guard together, so
        compliance / joint modes are protected by the same SelfCollisionGuard.)"""
        on = bool(on)
        if on and self.projector is None:
            from .collision import ArmCollisionProjector
            self.projector = ArmCollisionProjector(self.model, self.data, self.arm)
        self.self_collision = on
        return self.self_collision

    def apply_profile(self, profile):
        """Tolerant profile apply: accept the facade's OperatorProfile or the
        serialized profile (including legacy {reach}). Applies the reach scale
        if present, and for a full profile freezes measured anthropometry into
        the shoulder estimator (extension arm map) and the calibrated
        per-finger lengths into the keypoint finger solver. None clears measured
        anthropometry; no stored profile is selected implicitly."""
        get = (profile.get if isinstance(profile, dict) else
               lambda key, default=None: getattr(profile, key, default))
        reach = None
        if profile is not None:
            reach = get("reach")
        if reach is not None:
            self.apply_reach(reach)
        # Full body-calibration profile (facade OperatorProfile): measured arm
        # lengths / shoulder width / neck drop → shoulder estimator; measured
        # finger bone lengths → keypoint hand solver.
        full = (hasattr(profile, "arm_length_lr") or
                isinstance(profile, dict) and any(k in profile for k in (
                    "arm_length_lr", "shoulder_width_m", "neck_height_m",
                    "shoulder_offset_body", "finger_len_norm")))
        if profile is None or full:
            self.shoulder.set_profile(
                arm_length_lr=get("arm_length_lr"),
                shoulder_width=get("shoulder_width_m"),
                neck_drop=get("neck_height_m"),
                shoulder_offset_body=get("shoulder_offset_body"))
            lens = get("finger_len_norm") or {}
            for side in ("right", "left"):
                setp = getattr(self.hand[side], "set_profile", None)
                if setp is not None:
                    setp(lens.get(side))
        return profile

    def hold_current(self):
        """Snap the position-servo targets to the CURRENT joint pose (lurch-free on
        a sim reset or when the facade enters/exits a torque mode that neutralised
        the servos)."""
        for aid, _q0 in self.hold:
            jid = int(self.model.actuator_trnid[aid, 0])
            if jid < 0:
                continue
            adr = self.model.jnt_qposadr[jid]
            lo, hi = self.model.actuator_ctrlrange[aid]
            self.data.ctrl[aid] = float(np.clip(self.data.qpos[adr], lo, hi))
        # The tendon Lift is EXCLUDED from `hold` (trntype != joint), so nothing
        # above adopts its measured extension — a Stiff re-entry then replays a
        # stale ctrl and the lift SNAPS to wherever it was before Soft/limp moved
        # it. Its ctrl targets the STAGE SUM ("Sync Lift": Upper == Lower).
        try:
            aid = self.model.actuator("Lift").id
            lo, hi = self.model.actuator_ctrlrange[aid]
            q = (self.data.joint("Lift Lower").qpos[0]
                 + self.data.joint("Lift Upper").qpos[0])
            self.data.ctrl[aid] = float(np.clip(q, lo, hi))
        except Exception:
            pass
        # Re-anchor the fail-safe ease at the pose just adopted: without this the
        # supervisor's next apply() keeps blending from its STALE anchor and undoes
        # the hold (on the real robot: torque enables, then the arms ease away from
        # the measured pose toward wherever the old ease was headed).
        self.sup.rebase()

    # Base-twist + status telemetry (parity with the old controller's taps).
    # These feed RobotCommand.base → the REAL robot's /mabel_cmd, so they MUST
    # be the REALIZED, tip-safe command (wbc.base_cmd — ramped through
    # mabel.config.safe_base_step every base_step), NOT the raw stick target.
    # Exporting base_tgt here sent unshaped instant twists to the hardware:
    # full speed the moment the stick moved, a hard stop on release — the
    # "not using the safe motion model" bug. The sim was unaffected only
    # because its wheels take the shaped swerve ctrl directly.
    @property
    def vx(self):
        return float(self.wbc.base_cmd[0])

    @property
    def vy(self):
        return float(self.wbc.base_cmd[1])

    @property
    def omega(self):
        return float(self.wbc.base_cmd[2])

    def status(self):
        """Telemetry dict in the shape the bridge status line expects (parity with the
        old TeleopController.status). The cosine ArmRetargeter has no explicit palm
        error term, so those report 0.0."""
        return {
            "mode": self.mode,
            "calibrated": self.calibrated,
            "self_collision": getattr(self, "self_collision", False),
            "gravity_comp": getattr(self, "gravity_comp", True),
            "whole_body_retarget": getattr(self, "whole_body_retarget", False),
            "palm_err_l": 0.0,
            "palm_err_r": 0.0,
            "grasp_l": float(getattr(self.opt.get("left"), "grip", 0.0) or 0.0),
            "grasp_r": float(getattr(self.opt.get("right"), "grip", 0.0) or 0.0),
            "base": (self.vx, self.vy, self.omega),
        }

    # ── 1-pose reach calibration (extend arms forward) ────────────────────────
    def start_calibration(self, user="default"):
        self._cal = {"user": user, "active": True, "done": False,
                     "buf": [], "still_since": None, "reach": None}
        self.profile_user = user

    def cancel_calibration(self):
        self._cal = None

    def calibrate_tick(self, pose, now=None):
        """Step ONLY the 1-pose reach capture from an operator frame.

        Used by the facade while a WBC torque method (joint/wrist) owns the
        actuators: its branch returns before ``update`` ever runs, which
        silently starved the capture — the app's guided popup sat at
        "measuring…" forever while its local coach ring filled (2026-07-18).
        Frame-gated on the operator timestamp exactly like ``update`` so a
        250 Hz substep loop contributes one sample per VR frame."""
        if self._cal is None or not self._cal["active"] or pose is None:
            return
        ts = getattr(pose, "timestamp", None)
        if ts is not None and ts == getattr(self, "_last_cal_ts", None):
            return
        self._last_cal_ts = ts
        self._calibrate_step(self._payload(pose), time.time() if now is None else now)

    def calibration_status(self):
        if self._cal is None:
            return None
        st = {"active": self._cal["active"], "pose": "arms_forward",
              "index": 1 if self._cal["done"] else 0, "total": 1,
              "done": self._cal["done"], "reach": self._cal.get("reach")}
        # Live measurement (additive, only while measuring): the median arm
        # length in the window, its spread, and hold progress 0..1 — so the app
        # card can show the NUMBER stabilising instead of a black-box spinner.
        live = self._cal.get("live")
        if self._cal["active"] and live is not None:
            st["reachLive"] = live["reach"]
            st["spread"] = live["std"]
            st["hold"] = live["hold"]
        return st

    def apply_reach(self, reach):
        """Apply a saved reach scale directly (re-apply a profile)."""
        self.reach = float(np.clip(reach, 0.5, 3.0))
        self.profile = {"user": self.profile_user, "reach": self.reach}

    def save_profile(self, directory="profiles"):
        """Persist the measured 1-pose profile to <directory>/<user>.json."""
        if self.profile is None:
            return None
        import os
        import json
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"{self.profile['user']}.json")
        with open(path, "w") as f:
            json.dump(self.profile, f, indent=2)
        return path

    def load_profile(self, user, directory="profiles"):
        """Load + apply a saved 1-pose profile by operator name (reach scale)."""
        import os
        import json
        path = os.path.join(directory, f"{user}.json")
        if not os.path.exists(path):
            return None
        with open(path) as f:
            prof = json.load(f)
        self.profile_user = user
        self.apply_reach(float(prof.get("reach", REACH_SCALE)))
        self.profile = prof
        logger.info(f"applied saved profile [{user}]: REACH_SCALE {self.reach:.2f}")
        return prof

    def _calibrate_step(self, payload, now):
        """Accumulate the arms-forward hold and lock in the reach when it stabilises.

        SUCCESS CRITERION (redesigned 2026-07-19 — "if the arm-length number
        stabilises, that IS success"): the head→wrist reach must sit in the
        rolling window with

            n ≥ 3 samples  ·  window span ≥ CAL_MIN_SPAN  ·  std ≤ CAL_STD

        held for CAL_HOLD, then the MEDIAN of the window is captured. The old
        rule (≥5 samples inside 0.6 s + max−min ≤ 5 cm) failed BOTH ways, measured
        by tools/exp_calibration_robustness.py: below ~8 Hz it could literally
        never complete (5 samples never fit the window — a loaded app streams
        slower than that), and at 2 cm tremor it got STRICTER with rate (max−min
        grows with sample count: 60 Hz failed where 8 Hz passed). Time-span +
        standard deviation are rate-independent, and the median shrugs off single
        ARKit outliers. MIRRORED by the app's CalibrationCoach
        (CalibrationProfile.swift) — change BOTH."""
        c = self._cal
        hr = human_reach(payload)
        if hr is None or hr < 0.1:
            c["still_since"] = None
            c["live"] = None
            return
        c["buf"].append((now, hr))
        c["buf"] = [(t, v) for (t, v) in c["buf"] if t >= now - CAL_WINDOW]
        vals = [v for _, v in c["buf"]]
        med = float(np.median(vals))
        std = float(np.std(vals))
        span = c["buf"][-1][0] - c["buf"][0][0]
        still = len(vals) >= 3 and span >= CAL_MIN_SPAN and std <= CAL_STD
        if not still:
            c["still_since"] = None
            c["live"] = {"reach": med, "std": std, "hold": 0.0}
            return
        if c["still_since"] is None:
            c["still_since"] = now
        held = now - c["still_since"]
        c["live"] = {"reach": med, "std": std, "hold": min(1.0, held / CAL_HOLD)}
        if held < CAL_HOLD:
            return
        # Stable arms-forward hold -> set REACH_SCALE.
        hr_avg = med
        rr = robot_forward_reach(self.model, self.head_bid)
        self.reach = float(np.clip(rr / hr_avg, 0.5, 3.0))
        c["reach"] = self.reach
        c["done"] = True
        c["active"] = False
        self.profile = {"user": c["user"], "reach": self.reach,
                        "human_reach": hr_avg, "robot_reach": float(rr)}
        self.recalibrate()                                    # re-anchor at the new scale
        try:
            self.save_profile()
        except OSError:
            pass
        logger.info(f"1-pose calibration [{c['user']}]: human {hr_avg:.2f} m / robot "
                    f"{rr:.2f} m -> REACH_SCALE {self.reach:.2f}")

    # ── per-frame retargeting (HEAVY; runs once per VR frame) ─────────────────
    def _build_21(self, joints):
        if not joints:
            return None
        wrist = joints.get("wrist")
        if wrist is None:
            return None
        out = []
        for name in _NAMES_21:
            T = joints.get(name)
            out.append(np.asarray(T if T is not None else wrist, float).reshape(4, 4).flatten().tolist())
        return out

    def _payload(self, pose):
        """OperatorPose -> teleop payload dict (lists, so the safety/clutch truthiness
        checks behave like they do on the JSON stream)."""
        head = pose.head
        p = {}
        p["head_mat"] = np.asarray(head, float).reshape(4, 4).flatten().tolist() if head is not None else None
        for side, jk in (("right", "right_joints"), ("left", "left_joints")):
            joints = getattr(pose, jk, None)
            p[f"{side}_hand_mats"] = self._build_21(joints)
            p[f"{side}_joints"] = joints
            p[f"{side}_elbow"] = getattr(pose, f"{side}_elbow", None)
            p[f"{side}_pinch"] = getattr(pose, f"{side}_pinch", None)
        p["head_height"] = float(np.asarray(head, float).reshape(4, 4)[1, 3]) if head is not None else None
        # Torso heading for the base yaw-follow: the streamer publishes body_yaw
        # itself; headset/app clients get it derived from the two elbows (the wire
        # fills each side's elbow from ARKit's forearm_arm), same elbow-line math.
        body_yaw = getattr(pose, "body_yaw", None)
        if body_yaw is None and HEADING_MODE == "head":
            # drift-free lazy-neck head model (whole_body_retargeting/heading.py)
            if self._head_heading is None:
                from ..whole_body_retargeting.heading import LazyHeadHeading
                self._head_heading = LazyHeadHeading(HEADING_SLACK, HEADING_TAU)
            hm = p.get("head_mat")
            if hm and (self._face_forward or self._head_heading.psi_b is None):
                self._head_heading.calibrate(np.asarray(hm, float).reshape(4, 4))
                self._face_forward = False
            body_yaw = self._head_heading.update(
                np.asarray(hm, float).reshape(4, 4) if hm else None,
                self._frame_dt or (1.0 / 60.0),
                relax=self.wbc.tracker.follow_mode != "relative")
            if not ARM_TRACK_ELBOW:
                p["right_elbow"] = None
                p["left_elbow"] = None
        elif body_yaw is None:
            body_yaw = self._headset_body_yaw(p)
            # HEADSET frames: ARKit's forearm joint is a hand-anchored
            # extrapolation, so its raw POSITION must not enter the arm IK
            # (2026-07-11 "left arm twisted very weirdly"). Since 2026-07-29 the
            # IK consumes it DIRECTION-ONLY (wrist→forearm line, re-anchored at
            # the anatomical forearm length in _solve_side) — that direction IS
            # the operator's real forearm, so the elbow finally bends like the
            # human's (T palms-down → elbow forward, palms-up → elbow up, the
            # user's constraints). MABEL_ARM_TRACK_ELBOW=0 restores the old
            # null-and-synthesize behavior.
            if not ARM_TRACK_ELBOW:
                p["right_elbow"] = None
                p["left_elbow"] = None
        else:
            self._body_frame_fresh = False   # streamer owns yaw; keep est. re-seedable
        p["body_yaw"] = body_yaw
        p["odom_mat"] = None
        return p

    def _headset_body_yaw(self, p):
        """Operator base heading for a HEADSET frame (no streamer body_yaw).

        Prefers the analytic base-frame estimator (head + two wrists), which
        rejects the look-around / two-arm reach / gaze-fixed motions that rotate
        the raw elbow or wrist LINE without a real body turn. Falls back to the
        elbow-line estimator when the controller package or the wrists are
        unavailable (and it holds through hand dropouts). The output is in the
        SAME convention as the elbow estimator — an absolute robot-frame heading,
        AVP_TO_ROBOT @ (body's right), atan2(d_x, -d_y) — so BASE_YAW_SIGN and the
        downstream delta in wholebody.py are unchanged. Only the DELTA is used
        downstream, so the estimator's arbitrary calibration zero is irrelevant."""
        dt = self._frame_dt or (1.0 / 60.0)
        rh, lh = p.get("right_hand_mats"), p.get("left_hand_mats")
        if self._use_body_frame and p.get("head_mat") and rh and lh:
            H = np.asarray(p["head_mat"], float).reshape(4, 4)
            WR = np.asarray(rh[0], float).reshape(4, 4)   # [0] = wrist (see _NAMES_21)
            WL = np.asarray(lh[0], float).reshape(4, 4)
            if self._face_forward or not self.body_est.ready:
                # THE OPERATOR DECLARES THEIR FACING; the estimator does not have
                # to guess it. `calibrate_facing` zeroes the torso heading here,
                # so everything downstream (the arm map's body frame, and the
                # base yaw in global mode) is referred to THIS moment.
                self.body_est.calibrate_facing(H, WL, WR)
                self._face_forward = False
            elif not self._body_frame_fresh:
                self.body_est.seed(H, WL, WR)               # resume: no gap jump
            _, B = self.body_est.update(H, WL, WR, dt)
            self._body_frame_fresh = True
            r = AVP_TO_ROBOT @ B[:, 0]                       # body's RIGHT, robot frame
            return float(np.arctan2(r[0], -r[1]))
        # SET FORWARD MUST STILL LAND. Reaching here means the full triple was
        # not available this frame — usually a hand out of the headset's view.
        # The branch above is the ONLY consumer of `_face_forward`, so without
        # this the datum waits for a frame that may not come for a long time,
        # and the operator sees a button that did nothing. Past the deadline,
        # zero from the head alone (see `calibrate_facing_head`).
        if (self._use_body_frame and self._face_forward and p.get("head_mat")
                and time.monotonic() - self._face_forward_at >= FORWARD_DEADLINE):
            self.body_est.calibrate_facing_head(
                np.asarray(p["head_mat"], float).reshape(4, 4))
            self._face_forward = False
            self._body_frame_fresh = False
        # fallback: elbow-line estimator (holds its last heading through dropouts)
        self._body_frame_fresh = False
        return self.yaw_est.update(p.get("right_elbow"), p.get("left_elbow"), dt)

    def _wmap_head(self, p):
        """LEGACY head-anchored world map (ARM_MAP == "head"). REACH scales the
        HORIZONTAL offset only (arm-length / workspace matching); the VERTICAL
        head→wrist offset maps 1:1 so hands raised to the operator's head sit at
        the ROBOT's head height — never above it."""
        off = AVP_TO_ROBOT @ (np.asarray(p, float) - self._world_ref[0])
        off[:2] *= self.reach
        return self._world_ref[1] + off

    def _body_map_M(self):
        """The GLOBAL body-frame map: a 3×3 rotation taking operator-world
        vectors to robot-world vectors while preserving their BODY-semantic
        components — (operator right, up, fwd) ↦ (robot right, up, fwd), with
        the operator axes from the shoulder estimator (body-yaw-grounded) and
        the robot axes from the LIVE chassis pose. Both sides of the mapping
        rotate with their own body, so the operator's world heading and the
        robot's chassis heading both cancel exactly: the same body-relative
        reach always lands the same body-relative robot target. Falls back to
        the fixed AVP_TO_ROBOT world rotation (the pre-2026-07-22 behavior)
        when disabled or before the estimator has axes."""
        if not ARM_BODY_FRAME_MAP:
            return AVP_TO_ROBOT
        axes = self.shoulder.body_axes()
        if axes is None:
            return AVP_TO_ROBOT
        r_op, u_op, f_op = axes
        B_op = np.column_stack([r_op, u_op, f_op])
        # Robot body axes in the world: chassis R composed with R_BASE_ALIGN
        # (robot body x=fwd, y=left, z=up; chassis front is −x).
        W = self.data.xmat[self.base_bid].reshape(3, 3) @ R_BASE_ALIGN
        R_rob = np.column_stack([-W[:, 1], W[:, 2], W[:, 0]])   # right, up, fwd
        return R_rob @ B_op.T

    def _map_wrist_targets(self, payload):
        """Per-side operator wrist → robot-world wrist target dict, via the
        selected arm map.

        GMR map (default, 2026-07-24 — Retargeting Matters, arXiv:2510.02252):
        one ROOT-UNIFORM similarity transform for the whole upper body. Every
        keypoint is taken relative to the operator's mid-shoulder ROOT, scaled
        by the per-side arm-length ratio sigma = L_rob/L_hum, mapped through
        the body-frame rotation M and re-anchored at the ROBOT's mid-shoulder
        root:  t = r_rob + sigma·M·(w − r_hum).  The head–wrist–wrist pose
        TRIANGLE is preserved by construction (requirement: the robot's three
        tracked coordinates sit relatively where the human's do), so bimanual
        geometry needs no anti-collapse floor to stay sane. Near full
        extension (rho ≥ ARM_GMR_EXT_GATE) the target's magnitude about the
        robot's OWN shoulder is renormalized to rho·L_rob·od(rho), keeping the
        extension map's guarantee: full human extension = full robot
        extension, in every direction.

        EXTENSION map (MABEL_ARM_MAP=extension): rho = |wrist − shoulder| /
        L_arm of the OPERATOR's own arm, target = robot_shoulder + û·rho·L_rob
        — per-shoulder anchoring, extension-exact everywhere but pose-shape
        distorting (the robot's shoulders are 3× narrower than a human's).

        Both maps: a bimanual proximity correction sets the robot inter-wrist
        vector to the operator's 1:1 when the operator's wrists come close, so
        touching hands still touch. Falls back to the legacy head-anchored map
        per side until the shoulder estimator has seen a head frame."""
        raw = {}
        for side in ("right", "left"):
            h_mats = payload.get(f"{side}_hand_mats")
            if not h_mats:
                continue
            # Arm target = the operator WRIST (h_mats[0]), invariant to finger
            # flexion, NOT the middle fingertip (h_mats[12]) which retracts when
            # the hand closes and used to swing the arm. See ARM_TRACK_WRIST.
            raw[side] = np.asarray(h_mats[0 if ARM_TRACK_WRIST else 12],
                                   float).reshape(4, 4)[:3, 3]
        out = {}
        M = self._body_map_M()                     # global body-frame map (see doc)
        self._frame_M = M                          # reused by _solve_side this frame
        # Operator inter-wrist distance (None with < 2 hands): gates the synth
        # elbow off near bimanual contact (precision-dominated; the gravity-
        # swivel hint measurably costs 3 cm of hand-meet closure there).
        self._frame_dh = (float(np.linalg.norm(raw["right"] - raw["left"]))
                          if "right" in raw and "left" in raw else None)
        # GMR roots (operator mid-shoulder ↔ robot mid arm-mount), once per frame.
        root_h = self.shoulder.root_world() if self.arm_map == "gmr" else None
        root_r = (0.5 * (self.data.xpos[self.arm["right"].shoulder_bid]
                         + self.data.xpos[self.arm["left"].shoulder_bid])
                  if root_h is not None else None)
        # MOBILE-BASE WORKSPACE ASSIST: offset the map's ANCHOR so the arms get
        # room now, and let the base pay that offset off so the hands stay put
        # in the world. Moving the base WITHOUT this does nothing at all — the
        # target is anchored at root_r and would move with the robot.
        if root_r is not None and self._ws_assist is not None:
            root_r = root_r + self._ws_offset
        for side, p in ({} if self.rel_track else raw).items():
            if (self.arm_map in ("gmr", "extension")
                    and self.shoulder.ready(side)):
                s_h = self.shoulder.shoulder_world(side)
                r = p - s_h
                d = float(np.linalg.norm(r))
                if d < 1e-6:
                    continue                       # wrist AT the shoulder: hold
                L_h = max(self.shoulder.arm_length(side), 1e-6)
                rho = min(d / L_h, ARM_EXT_RHO_MAX)
                # Overdrive fades in only NEAR full extension (rho ≳ gate):
                # mid-workspace stays exactly 1:1, while at rho = 1 the target
                # sits marginally past the reach boundary so the position term
                # overrules the posture regularizers and the elbow straightens
                # fully (instead of holding a few residual degrees of bend).
                s = min(max((rho - ARM_GMR_EXT_GATE)
                            / max(1.0 - ARM_GMR_EXT_GATE, 1e-6), 0.0), 1.0)
                lam = s * s * (3.0 - 2.0 * s)      # smoothstep gate
                od = 1.0 + (ARM_EXT_OVERDRIVE - 1.0) * lam
                s_r = self.data.xpos[self.arm[side].shoulder_bid]
                if root_h is not None:
                    # GMR: pose-shape-preserving root-uniform scaled keypoint,
                    # magnitude blended to the extension-exact radius near rho=1.
                    sigma = self.arm_reach[side] / L_h
                    t_raw = root_r + sigma * (M @ (p - root_h))
                    dv = t_raw - s_r
                    dn = float(np.linalg.norm(dv))
                    if dn < 1e-6:
                        continue
                    if ARM_SHAPE_GATE:
                        # extended arm: the operator's own shoulder->wrist DIRECTION (hyperparams ARM_SHAPE_GATE)
                        g = min(max((rho - ARM_SHAPE_GATE_LO)
                                    / max(ARM_SHAPE_GATE_HI - ARM_SHAPE_GATE_LO, 1e-6), 0.0), 1.0)
                        g = g * g * (3.0 - 2.0 * g)
                        if g > 0.0:
                            u_h = M @ (r / d)
                            u = (1.0 - g) * (dv / dn) + g * (u_h / max(float(np.linalg.norm(u_h)), 1e-9))
                            un = float(np.linalg.norm(u))
                            if un > 1e-6:
                                dv = u / un * dn
                    # L_rob along THIS direction, not one forward-ray scalar
                    # (see ARM_REACH_DIRECTIONAL): otherwise full human
                    # extension under-reaches wherever the arm is longer than
                    # the scalar, and the elbow keeps a bend nothing removes.
                    L_rob = (self.arm_reach_dir[side].at(dv / dn)
                             if self.arm_reach_dir is not None else self.arm_reach[side])
                    mag = (1.0 - lam) * dn + lam * (rho * L_rob * od)
                    out[side] = s_r + (dv / dn) * mag
                else:
                    u_rob = M @ (r / d)
                    L_rob = (self.arm_reach_dir[side].at(u_rob)
                             if self.arm_reach_dir is not None else self.arm_reach[side])
                    out[side] = s_r + u_rob * (rho * L_rob * od)
            elif self._world_ref is not None:
                out[side] = self._wmap_head(p)
        if (self.arm_map in ("gmr", "extension") and ARM_BIMANUAL_BLEND
                and "right" in out and "left" in out):
            d_h = raw["right"] - raw["left"]
            dist = float(np.linalg.norm(d_h))
            if dist < ARM_BIMANUAL_NEAR:
                # NEAR-CONTACT: symmetric blend toward the operator's EXACT 1:1
                # metric separation so touching hands touch.
                w = 1.0 - (dist - ARM_BIMANUAL_FULL) / max(
                    ARM_BIMANUAL_NEAR - ARM_BIMANUAL_FULL, 1e-6)
                w = float(np.clip(w, 0.0, 1.0))
                w = w * w * (3.0 - 2.0 * w)         # smoothstep fade-in
                mid = 0.5 * (out["right"] + out["left"])
                d_rob = M @ d_h                      # 1:1 metric separation
                out["right"] = (1.0 - w) * out["right"] + w * (mid + 0.5 * d_rob)
                out["left"] = (1.0 - w) * out["left"] + w * (mid - 0.5 * d_rob)
            # ANTI-COLLAPSE FLOOR — applied at EVERY separation. The robot's
            # shoulders are 0.148 m apart vs a human's ~0.45 m, so inward-
            # angled reach directions cross the midline far sooner on the
            # robot (measured: 1.6 cm robot separation at 30 cm human — the
            # reported wrist overlap; a geometric property of the per-shoulder
            # map, not an IK defect). The floor is DELIBERATELY absolute, not
            # γ-proportional: floor = min(HAND_CLEAR, d_human), so it (a) can
            # never push a target when the natural separation is already a
            # hand-clearance apart — one-arm tasks with the other arm hanging
            # are untouched (a γ·d_human floor measurably contaminated them:
            # E1 full-reach rho rose 0.959→0.988 and circle tracking tripled
            # its error), and (b) fades to the operator's own 1:1 separation
            # near contact, so touching hands still touch. Enforced
            # symmetrically about the pair midpoint.
            floor = min(ARM_HAND_CLEAR, dist)
            sep = out["right"] - out["left"]
            d_r = float(np.linalg.norm(sep))
            if d_r < floor:
                axis = sep / d_r if d_r > 1e-6 else M @ (d_h / dist)
                mid = 0.5 * (out["right"] + out["left"])
                out["right"] = mid + 0.5 * floor * axis
                out["left"] = mid - 0.5 * floor * axis
        if self.rel_track:
            out = self._rel_targets(raw, payload, M)
        out = self._gripper_handoff_targets(payload, out, M)
        for side in out:
            out[side] = clamp_in_front(out[side], self.data, self.base_bid)
            # Then push it out of the robot's own body. Order matters: the front
            # clamp works in the chassis frame and can itself pull a target
            # inward, so the standoff has to be the LAST word on the target.
            if side in self.arm:
                sh = np.asarray(self.data.xpos[self.arm[side].shoulder_bid], float)
                other = "right" if side == "left" else "left"
                # TORSO FIRST, then the shoulder sphere. The torso clamp fixes
                # the case the shoulder clamp gets wrong (a target on the body
                # centreline), and running it first means the shoulder clamp
                # only ever sees targets that are already outside the body, so
                # its radial direction is meaningful when it fires.
                if ARM_MIN_TORSO_RADIUS > 0.0 and other in self.arm:
                    sh_o = np.asarray(self.data.xpos[self.arm[other].shoulder_bid], float)
                    axis = 0.5 * (sh + sh_o)
                    out[side] = clamp_torso_standoff(
                        out[side], axis[:2], ARM_MIN_TORSO_RADIUS, sh - sh_o)
                if ARM_MIN_SHOULDER_RADIUS > 0.0:
                    out[side] = clamp_shoulder_standoff(
                        out[side], sh, ARM_MIN_SHOULDER_RADIUS)
        # Advance the base-assist debt from THIS frame's targets; it is applied
        # to the anchor on the next one. A frame of lag is immaterial for a
        # channel that repositions the whole robot at 0.25 m/s.
        # rel: the targets carry no workspace offset, so arm debt must not drive the base (an open loop: it ran)
        if self._ws_assist is not None and out and not self.rel_track:
            shoulders = {sd: np.asarray(self.data.xpos[self.arm[sd].shoulder_bid], float)
                         for sd in out if sd in self.arm}
            if shoulders:
                self._ws_offset = self._ws_assist.update(
                    out, shoulders,
                    np.asarray(self.data.xpos[self.base_bid], float),
                    np.asarray(self.data.xmat[self.base_bid], float).reshape(3, 3),
                    self._frame_dt or (1.0 / 60.0))
        return out

    def _rel_targets(self, raw, payload, M):
        """RELATIVE INCREMENT map (rel_track): what most VR teleop does (DROID / TeleMoMa / HATO / XRoboToolkit:
        clutched world-frame deltas).

        At engage (and whenever a hand comes back after REL_RELATCH_GAP s untracked) a side latches the robot's
        tracked point t and palm frame R where they ARE (no jump). Then, every frame k,
          t_k = t_{k-1} + s M_k (w_k - w_{k-1})                       (wrist translation increment, metres 1:1)
          R_k = M_k (R_op,k R_op,k-1^T) M_k^T R_{k-1}                  (wrist rotation increment, world frame)
        with M the body-frame map (operator heading -> robot heading) and s = REL_SCALE. The hands keep their place
        in the world: walking and turning move the robot's hands only as far as the operator's wrists move, and a
        lean or squat moves the robot's body under hands that stay put. Anti-windup: t is held inside REL_REACH x the
        arm's reach from its shoulder, so reaching past it saturates and coming back moves at once. Increments
        accumulate the base's follow error; Re-anchor re-latches."""
        from .geometry import operator_frame
        d = self.data
        now = float(getattr(self, "_rt_now", None) or time.monotonic())
        out = {}
        for side, w in raw.items():
            R_op = operator_frame(payload.get(f"{side}_joints"), payload.get(f"{side}_hand_mats"))
            if R_op is None:
                continue
            arm = self.arm[side]
            st, last = self._rel.get(side), self._rel_seen.get(side)
            if st is None or last is None or now - last > REL_RELATCH_GAP:
                st = self._rel[side] = {"t": arm._ee_point(d)[0].copy(),
                                        "R": d.xmat[arm.palm_bid].reshape(3, 3) @ arm.R_rob,
                                        "w": np.asarray(w, float).copy(), "Rop": R_op.copy()}
            self._rel_seen[side] = now
            t = st["t"] + REL_SCALE * (M @ (np.asarray(w, float) - st["w"]))
            dR = M @ (R_op @ st["Rop"].T) @ M.T
            U, _, Vt = np.linalg.svd(dR @ st["R"])             # keep it a rotation (no drift off SO(3))
            R = U @ Vt
            s_r = d.xpos[arm.shoulder_bid]
            v = t - s_r
            n = float(np.linalg.norm(v))
            lim = REL_REACH * (self.arm_reach_dir[side].at(v / n) if self.arm_reach_dir is not None and n > 1e-6
                               else self.arm_reach[side])
            if n > lim:                                          # anti-windup: the latch never leaves the reach
                t = s_r + v * (lim / n)
            st.update(t=t, R=R, w=np.asarray(w, float).copy(), Rop=R_op.copy())
            out[side] = t.copy()
            self._rel_axes[side] = (R[:, 0].copy(), R[:, 2].copy())   # what the IK aligns: palm fwd, palm normal
        return out

    def _gripper_handoff_targets(self, payload, out, M):
        """Blend wrist tracking into physical-tip precision near a two-hand handoff."""
        self._gripper_handoff_separation = None
        if self.ee != "gripper" or not self.enable_gripper_contact or self.rel_track:
            self._gripper_contact_alpha = 0.0         # rel: gripper capsules in the guard, no target blending
            return out
        from .end_effector import GRIPPER, TRACK_OFFSET, grasp_axes
        contacts, offsets = {}, {}
        for side in out:
            joints = payload.get(f"{side}_joints") or {}
            try:
                tips = [np.asarray(joints[n], float).reshape(4, 4)[:3, 3]
                        for n in ("thumb_tip", "index_tip")]
                point = .5 * (tips[0] + tips[1])
                if np.all(np.isfinite(point)):
                    contacts[side] = point
            except (KeyError, ValueError, TypeError):
                pass
            R = operator_frame(joints, payload.get(f"{side}_hand_mats"))
            if R is not None:
                fwd, up = palm_axes(M @ R, side)
                offsets[side] = (GRIPPER["tip"] - TRACK_OFFSET) * grasp_axes(fwd, up)[0]
        paired = all(s in contacts and s in offsets and s in out for s in ("right", "left"))
        gate = 0.0
        coupled = False
        if paired:
            distance = float(np.linalg.norm(contacts["right"] - contacts["left"]))
            gate = float(np.clip((.15 - distance) / (.15 - .02), 0., 1.))
            gate = gate * gate * (3.0 - 2.0 * gate)
            a, b = offsets["right"], offsets["left"]
            opposition = -float(a @ b) / max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-9)
            alignment = float(np.clip((opposition - .5) / .45, 0., 1.))
            gate *= alignment * alignment * (3.0 - 2.0 * alignment)
            coupled = distance < .15 and opposition > .5
        previous = float(getattr(self, "_gripper_contact_alpha", 0.0))
        dt = float(self._frame_dt or 1.0 / 60.0)
        alpha = previous + (1.0 - np.exp(-max(dt, 0.0) / .06)) * (gate - previous)
        if alpha < 1e-5:
            alpha = 0.0
        self._gripper_contact_alpha = alpha
        if not alpha:
            return out
        if coupled:
            separation = M @ (contacts["right"] - contacts["left"])
            mid = .5 * (out["right"] + offsets["right"] + out["left"] + offsets["left"])
            desired = {"right": mid + .5 * separation, "left": mid - .5 * separation}
            self._gripper_handoff_separation = separation
        else:
            # Lost tracking disables the coupled task immediately. The remaining
            # arm's tool point/target fade together, avoiding a task-point jump.
            desired = {s: out[s] + offsets.get(s, np.zeros(3)) for s in out}
        return {s: (1.0 - alpha) * out[s] + alpha * desired[s] for s in out}

    def _gripper_task_point(self, side):
        """Interpolate the actual IK point alongside its handoff target."""
        if self.ee != "gripper":
            return
        from .end_effector import distal_tip, tracked_point
        arm = self.arm[side]
        if not self.enable_gripper_contact and not hasattr(arm, "_gripper_contact_points"):
            return
        if not hasattr(arm, "_gripper_contact_points"):
            _, base = tracked_point(self.model, side, arm.palm_bid, arm.R_rob)
            _, tip = distal_tip(self.model, side, arm.palm_bid, arm.R_rob)
            arm._gripper_contact_points = (base, tip)
        base, tip = arm._gripper_contact_points
        alpha = (float(getattr(self, "_gripper_contact_alpha", 0.0))
                 if self.enable_gripper_contact else 0.0)
        arm._ee_off = (1.0 - alpha) * base + alpha * tip

    def _solve_side(self, side, payload, contact):
        """Finger + arm IK for ONE side, using that side's OWN scratch MjData (so the
        two sides run in parallel). ``contact`` is the pre-mapped robot-world wrist
        target from _map_wrist_targets (None → the arm holds; fingers still track).
        Returns (side, q_fingers, arm_sol, orca_wrist_q) — arm_sol/wrist_q are None
        when the arm held — or None when the hand is untracked. No writes to shared
        self.data.ctrl here — the caller writes the disjoint actuators after both
        sides finish."""
        h_mats = payload.get(f"{side}_hand_mats")
        if not h_mats:
            return None
        R_op = operator_frame(payload.get(f"{side}_joints"), h_mats)
        # FINGERS: dexterous-planner keypoint IK (default) or the webcam-parity
        # anatomical decomposition. Returns None only before the first clean solve.
        _prof = os.environ.get("MABEL_RETARGET_PROF")
        _tf = time.perf_counter() if _prof else 0.0
        if self.ee == "gripper":
            # no fingers: the operator's hand closure sets the gripper opening; the ORCA joints hold
            self.gripper_aperture[side].update(payload.get(f"{side}_joints"), getattr(self, "_rt_now", None))
            q_fingers = None                        # -> held below
        else:
            q_fingers = self.hand[side].solve(h_mats, payload.get(f"{side}_joints"))
        if _prof:
            self._prof_add("fingers", time.perf_counter() - _tf)
        if q_fingers is None:
            q_fingers = self.data.ctrl[self.finger_act[side]].copy()   # hold current
        if R_op is None or contact is None:
            return side, q_fingers, None, None      # degenerate: hold the arm

        # Palm ORIENTATION goes through the same global body-frame map as the
        # position: with the fixed AVP_TO_ROBOT a turned operator's palm axes
        # mapped world-fixed and the hands twisted with the body heading.
        M = getattr(self, "_frame_M", AVP_TO_ROBOT)
        R_palm_canon = M @ R_op
        self._gripper_task_point(side)
        # Operator ELBOW: only its DIRECTION from the shoulder enters the arm
        # IK's elbow cosine (ARM_W_ELBOW, arm.py). Webcam clients TRACK it; a
        # headset does not (ARKit's forearm joint is a hand-anchor
        # extrapolation, nulled in _payload) — in extension mode a headset gets
        # a SYNTHESIZED elbow instead: the two-link gravity-swivel elbow from
        # the estimated shoulder + wrist (geometry.synth_elbow). Without any
        # elbow objective the redundant arm's configuration is weakly
        # determined for high/close wrist targets and flip-flops (measured
        # 0.51 rad/tick command steps, exp_real_retarget_chain P3); the hint
        # both stabilises it and shapes the arm like the operator's.
        elbow = None
        elbow_w_scale = 1.0
        e = payload.get(f"{side}_elbow")
        if e is not None:
            e_pos = np.asarray(e, float).reshape(4, 4)[:3, 3]
            if self.arm_map in ("gmr", "extension") and self.shoulder.ready(side):
                # DIRECTION-ONLY elbow (2026-07-29): ARKit's forearm joint is an
                # extrapolation rigidly attached to the hand, so its wrist→elbow
                # DIRECTION is the operator's true forearm line even though its
                # absolute position is not anchored at the real elbow. Re-anchor
                # at the anatomical forearm length from the wrist, EMA-smoothed
                # (per-frame hand-tracking wobble must not churn the hint), and
                # map shoulder-relative like every other keypoint. This is what
                # makes the robot elbow bend LIKE THE OPERATOR'S — palms-down T
                # → elbow forward, palms-up T → elbow up, inward flex → elbow
                # down-forward — instead of the gravity/palm-roll synthesis.
                w_h = np.asarray(payload[f"{side}_hand_mats"][0],
                                 float).reshape(4, 4)[:3, 3]
                dvec = e_pos - w_h
                nd = float(np.linalg.norm(dvec))
                if nd > 1e-6:
                    L_op = self.shoulder.arm_length(side)
                    e_rec = w_h + (dvec / nd) * ((1.0 - 0.56) * L_op)
                    if not hasattr(self, "_elbow_ema"):
                        self._elbow_ema = {}
                    ema = self._elbow_ema.get(side)
                    if ema is None:
                        ema = e_rec.copy()
                    else:
                        ema = ema + ARM_ELBOW_TRACK_EMA * (e_rec - ema)
                    self._elbow_ema[side] = ema
                    s_h_ = self.shoulder.shoulder_world(side)
                    s_r = self.data.xpos[self.arm[side].shoulder_bid]
                    elbow = s_r + M @ (ema - s_h_)
                    # BEND-PROPORTIONAL hint strength: at a near-straight arm
                    # the elbow direction is geometrically weak (the perp
                    # offset is centimetres) and a full-weight hint trades
                    # real wrist-position accuracy for it (measured: T-pose
                    # errors tripled). Scale ARM_W_ELBOW by the operator's
                    # actual elbow flexion — full weight at a proper bend,
                    # a light directional bias when straight.
                    u_up_ = ema - s_h_
                    u_fo_ = w_h - ema
                    nu, nf = np.linalg.norm(u_up_), np.linalg.norm(u_fo_)
                    if nu > 1e-6 and nf > 1e-6:
                        cosb = float(np.clip((u_up_ / nu) @ (u_fo_ / nf),
                                             -1.0, 1.0))
                        bend = float(np.arccos(cosb))       # 0 = straight
                        elbow_w_scale = float(np.clip(
                            bend / ARM_ELBOW_BEND_REF, ARM_ELBOW_W_MIN, 1.0))
            else:
                elbow = self._wmap_head(e_pos)
        elif (ARM_SYNTH_ELBOW and self.arm_map in ("gmr", "extension")
                and self.shoulder.ready(side)
                and (getattr(self, "_frame_dh", None) is None
                     or self._frame_dh > ARM_BIMANUAL_NEAR)):
            h_mats = payload.get(f"{side}_hand_mats")
            w_h = np.asarray(h_mats[0], float).reshape(4, 4)[:3, 3]
            s_h = self.shoulder.shoulder_world(side)
            # Swivel from the palm roll (R_op = operator natural palm frame),
            # EMA-smoothed: the raw per-frame roll wobbles with hand geometry
            # during fast wrist motion and churned the elbow hint (tripled the
            # circle direction RMS unfiltered). Deliberate elbow raises pass
            # in ~0.3 s; the body axes feed the torso-clearance clamp.
            # Palm roll -> elbow swivel, at ARM_SWIVEL_PALM_GAIN (0 by default:
            # a wrist flip must turn J5, not swing the elbow — see the constant).
            phi_raw = (estimate_swivel(s_h, w_h, R_op, side)
                       if ARM_SWIVEL_PALM_GAIN > 0 else None)
            if not hasattr(self, "_swivel"):
                self._swivel = {}
            phi_s = self._swivel.get(side, 0.0)
            phi_s += 0.10 * (ARM_SWIVEL_PALM_GAIN * (phi_raw or 0.0) - phi_s)
            self._swivel[side] = phi_s
            e_syn = synth_elbow(s_h, w_h, self.shoulder.arm_length(side),
                                palm_R=phi_s, side=side,
                                body_axes=self.shoulder.body_axes())
            if e_syn is not None:
                s_r = self.data.xpos[self.arm[side].shoulder_bid]
                elbow = s_r + M @ (e_syn - s_h)
        # UPPER-ARM STANDOFF (2026-10-16, operator on NOVA: "the upper arm gets way too close to the body,
        # it feels tucked — I need the upper arms slightly separated from the body"). The elbow hint copies
        # the OPERATOR's elbow direction, and a human elbow hangs against a torso about as wide as the
        # shoulders; MABEL/NOVA's shoulders are ~3x narrower relative to the torso capsule, so the same
        # direction parks the robot's upper arm on the self-collision margin (measured 10-13 mm clearance
        # at its closest on the operator's takes, i.e. riding the 10 mm barrier). Shift the hint point
        # OUTWARD, horizontally, by ARM_ELBOW_ABDUCT metres from the robot's own shoulder line: only its
        # direction from the shoulder enters the IK cost, so this is a ~10 deg abduction bias (0.15 m default) on a hanging
        # arm that fades as the elbow rises away from the body. 0 restores the plain copy. E9 in
        # web_gui/retarget_lab/docs/retarget_lab.tex.
        contact_alpha = (float(getattr(self, "_gripper_contact_alpha", 0.0))
                         if self.ee == "gripper" and self.enable_gripper_contact else 0.0)
        elbow_abduct = ARM_ELBOW_ABDUCT * (1.0 - contact_alpha)
        if elbow is not None and elbow_abduct > 0.0:
            s_me = self.data.xpos[self.arm[side].shoulder_bid]
            s_ot = self.data.xpos[self.arm["left" if side == "right" else "right"].shoulder_bid]
            out = s_me - s_ot
            out[2] = 0.0
            no = float(np.linalg.norm(out))
            if no > 1e-6:
                elbow = np.asarray(elbow, float) + (elbow_abduct / no) * out
        fwd_ax, up_ax = palm_axes(R_palm_canon, side)   # left normal → palmar
        if self.ee == "gripper":                        # point along the operator's PINCH, not the metacarpals
            from .end_effector import grasp_axes
            fwd_ax, up_ax = grasp_axes(fwd_ax, up_ax)
        if self.rel_track and side in self._rel_axes:      # rel: the operator's rotation CHANGE on the latched palm
            fwd_ax, up_ax = self._rel_axes[side]
        # hk['swivel']: the smoothed commanded elbow swivel [rad] — drives the
        # adaptive elbow weight + Arm3 reg relax (arm._w_elbow/_reg_for).
        # None for tracked-elbow (webcam) clients: their hint is ground truth
        # and the original weight balance ships unchanged.
        sw = (getattr(self, "_swivel", {}).get(side)
              if e is None and elbow is not None else None)
        # OPERATOR ELBOW FLEXION, as a cosine — the SHAPE the robot should copy.
        # Computed from the SAME reconstructed elbow the direction cue uses, so
        # it costs nothing extra and cannot disagree with it. Sent as a cosine
        # rather than an angle so the arm cost stays a smooth quadratic with no
        # arccos derivative to blow up at full extension, which is exactly where
        # this term has to work.
        elbow_angle = None
        if elbow is not None:
            hm_ = payload.get(f"{side}_hand_mats")
            if hm_ and self.shoulder.ready(side):
                s_h_ = self.shoulder.shoulder_world(side)
                w_h_ = np.asarray(hm_[0], float).reshape(4, 4)[:3, 3]
                e_h_ = (self._elbow_ema.get(side)
                        if hasattr(self, "_elbow_ema") else None)
                if e_h_ is not None:
                    ua, fa = s_h_ - e_h_, w_h_ - e_h_
                    nua, nfa = np.linalg.norm(ua), np.linalg.norm(fa)
                    if nua > 1e-6 and nfa > 1e-6:
                        elbow_angle = float(np.clip((ua / nua) @ (fa / nfa), -1.0, 1.0))
        elbow_w_scale *= 1.0 - contact_alpha
        hk = {"contact": contact, "elbow": elbow, "fwd": fwd_ax, "up": up_ax,
              "swivel": sw, "elbow_w_scale": elbow_w_scale,
              "elbow_angle": elbow_angle}
        # OPERATOR OVERLAY capture (2026-07-30, web-CC retarget debugging): the
        # EXACT quantities this side's arm IK consumes, already in robot/odom
        # world — the wrist target, the elbow hint, the palm axes, and the full
        # hand skeleton carried rigidly around the target (p = target +
        # M·(p_ark − wrist_ark), true metric hand size). Streaming THESE (and
        # never a re-derived transform) is what guarantees the web overlay is
        # frame-exact: the mapped wrist point IS the IK target, so overlay-vs-
        # robot gap on screen == the real tracking error.
        try:
            _w_ark = np.asarray(h_mats[0], float).reshape(4, 4)[:3, 3]
            _hand = [(contact + M @ (np.asarray(m, float).reshape(4, 4)[:3, 3]
                                     - _w_ark)).tolist() for m in h_mats]
            if not hasattr(self, "_op_overlay"):
                self._op_overlay = {}
            self._op_overlay[side] = {
                "target": np.asarray(contact, float).tolist(),
                "elbow": (np.asarray(elbow, float).tolist()
                          if elbow is not None else None),
                "fwd": np.asarray(fwd_ax, float).tolist(),
                "up": np.asarray(up_ax, float).tolist(),
                "hand": _hand,
            }
        except Exception:                       # overlay is debug-only — never
            pass                                # let it touch the control path
        self.arm[side].sync(self.data)                # reads self.data.qpos (read-only)
        sol = self.arm[side].solve(hk, dt=self._frame_dt)
        wrist_q = float(self.arm[side].ik.qpos[self.arm[side].orca_wrist_qadr])
        return side, q_fingers, sol, wrist_q

    def _prof_add(self, name, dt):
        self.__dict__.setdefault("_prof", {}).setdefault(name, []).append(dt * 1e3)

    def _prof_report(self):
        p = self.__dict__.get("_prof")
        if not p or len(p.get("total", ())) < 40:
            return
        parts = " | ".join(f"{k} {int(np.median(v))}/{int(np.percentile(v, 95))}"
                           for k, v in sorted(p.items(), key=lambda kv: -np.median(kv[1])))
        logger.info("[retarget-prof ms med/p95] %s", parts)
        p.clear()

    def _neck_torso_pitch(self):
        """Torso rotation for camera compensation, with optional hardware feedback."""
        angle = self.neck_torso_measured
        if angle is None or not np.isfinite(angle):
            angle = self.data.qpos[self.torso_qadr]
        return float(angle) - self.torso_rest

    def _neck_chassis_yaw(self):
        """Measured chassis turn since the neck's forward datum was captured."""
        R = self.data.xmat[self.base_bid].reshape(3, 3)
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
        if self._neck_base_yaw0 is None:
            self._neck_base_yaw0 = yaw
        # neck_angles applies a rotation matrix, which also handles yaw wrap.
        return yaw - self._neck_base_yaw0

    def _retarget(self, payload, joystick, now):
        d = self.data
        self._rt_now = now                          # the frame clock (gripper aperture low-pass)
        _prof = os.environ.get("MABEL_RETARGET_PROF")
        _pc = time.perf_counter
        _t0 = _pc() if _prof else 0.0
        # CLUTCH: double-pinch freeze; re-anchor on release.
        if self.clutch.update(payload, now) and not self.clutch.clutched:
            self.recalibrate()
        frozen = self.clutch.clutched

        head_mat = payload.get("head_mat")
        if head_mat:
            head_mat = np.asarray(head_mat, float).reshape(4, 4)
        head_pos = head_mat[:3, 3] if head_mat is not None else None

        # ── ARKit ORIGIN-JUMP guard + anchor stability ────────────────────────
        # A per-frame head teleport (origin relocalisation, e.g. the floor snap
        # right after the immersive space opens) makes EVERY anchor stale → the
        # old behaviour mapped hands ~1.6 m too high and ran the lift to the top.
        # Re-anchor on a jump, and only latch anchors after the head has streamed
        # steadily for HEAD_ANCHOR_HOLD.
        if head_pos is not None:
            if (self._head_prev is not None
                    and np.linalg.norm(head_pos - self._head_prev) > HEAD_JUMP_M):
                logger.info("head jumped %.2f m (ARKit origin relocalised) — re-anchoring",
                            float(np.linalg.norm(head_pos - self._head_prev)))
                self.recalibrate(face=True)      # the world moved: re-derive forward
                self._head_stable_since = None   # jump ⇒ demand a fresh stable window
            self._head_prev = head_pos.copy()
            if self._head_stable_since is None:
                self._head_stable_since = now
        head_stable = (head_pos is not None and self._head_stable_since is not None
                       and now - self._head_stable_since >= HEAD_ANCHOR_HOLD)
        if head_mat is not None and self._head0 is None and head_stable:
            self._head0 = head_mat.copy()            # neck reference (stable only)
        if self._world_ref is None and head_pos is not None and not frozen:
            if head_stable:
                self._world_ref = (head_pos.copy(), d.xpos[self.head_bid].copy())
                # The first post-anchor retarget solution can sit far from the
                # held pose (the origin-jump guard kept engagement alive, so no
                # reengage edge fires) — ease it in instead of stepping to it.
                self.sup.restart_ease()
            else:
                # Not anchored yet: keep the supervisor fed (hold pose, no HOME
                # ease) and let NOTHING anchor to the unsettled head this frame.
                self.sup.t_stream = self.sup.t_head = self.sup.t_hand = self.sup.t_arm = now
                head_pos = None
                payload = dict(payload)
                payload["head_height"] = None

        # A channel that is OFF is not COMPUTED, not computed-then-overwritten:
        # the arm IK and the two finger solves are the expensive part of the
        # frame, and running them to throw the answer away would be the same
        # cost for none of the behaviour.
        if not frozen and self._world_ref is not None and (self.track_arms or self.track_hands):
            # Live shoulder anchors for the extension arm map: feed the estimator
            # the head + both wrist world positions (body_yaw when the estimator/
            # streamer produced one, so a head glance never orbits the shoulders).
            if head_pos is not None:
                wrists = {}
                for side in ("right", "left"):
                    hm = payload.get(f"{side}_hand_mats")
                    wrists[side] = (np.asarray(hm[0], float).reshape(4, 4)[:3, 3]
                                    if hm else None)
                self.shoulder.update(head_mat, payload.get("body_yaw"), wrists)
            contacts = self._map_wrist_targets(payload)
            # Solve right + left INLINE (sequential) by default. With the analytic-
            # Jacobian GN arm solver each side is ~0.3-0.6 ms, and the pool's
            # dispatch + GIL handoff alone measured 5-6 ms/frame (2026-07-22) —
            # 10x the work it parallelises. (On the Jetson the pool additionally
            # thrashed the GIL against the ROS/network threads — the retarget-mode
            # collapse to 1-2 Hz.) MABEL_ARM_POOL=1 restores the pool for A/B runs.
            _ta = _pc() if _prof else 0.0
            if os.environ.get("MABEL_ARM_POOL", "0").lower() in ("1", "true", "yes"):
                results = list(self._pool.map(
                    lambda s: self._solve_side(s, payload, contacts.get(s)),
                    ("right", "left")))
            else:
                results = [self._solve_side(s, payload, contacts.get(s))
                           for s in ("right", "left")]
            if _prof:
                self._prof_add("arm_solve", _pc() - _ta)
            # Publish the operator overlay assembled by _solve_side this frame
            # (stamped with the FRAME ts so consumers dedupe on it). Sides not
            # solved this frame (hand dropout) simply age out of the dict.
            # `raw` is the UNPROCESSED VP tracking — head + both full hand
            # skeletons — through nothing but the fixed AVP_TO_ROBOT axis
            # relabel (ARKit Y-up → robot Z-up; no anchoring, no scaling, no
            # body-frame map): the exact points the Vision Pro app shows, at
            # true metric scale, floating at the ARKit session origin.
            if getattr(self, "_op_overlay", None):
                raw = {}
                try:
                    hm = payload.get("head_mat")
                    if hm:
                        raw["head"] = (AVP_TO_ROBOT @ np.asarray(
                            hm, float).reshape(4, 4)[:3, 3]).tolist()
                    for s in ("right", "left"):
                        mats = payload.get(f"{s}_hand_mats")
                        if mats:
                            raw[s] = [(AVP_TO_ROBOT @ np.asarray(
                                m, float).reshape(4, 4)[:3, 3]).tolist()
                                for m in mats]
                except Exception:
                    pass
                self.last_operator_points = {
                    "t": float(self._last_ts or 0.0),
                    "sides": {s: v for s, v in self._op_overlay.items()},
                    "raw": raw,
                }
                self._op_overlay = {}
            arm_sol, wrist_sol = {}, {}
            for res in results:
                if res is None:
                    continue
                side, q_fingers, sol, wrist_q = res
                if self.track_hands:
                    for a, q in zip(self.finger_act[side], q_fingers):
                        d.ctrl[a] = q
                if sol is None or not self.track_arms:
                    continue                       # arm held (degenerate frame / off)
                if self.lock_j7:
                    d.ctrl[self.arm_act[side][6]] = self.arm[side].j7_lock_val
                arm_sol[side] = np.asarray(sol, float)     # 7 arm joints
                wrist_sol[side] = float(wrist_q)           # ORCA wrist, projected below
            # ARM self-collision: project BOTH arms' targets onto the safe set TOGETHER
            # (the same ArmCollisionProjector / SelfCollisionGuard avp_teleop uses). An
            # untracked side holds its current qpos so the QP reasons over the true
            # two-arm geometry. Fast-paths to a no-op when nothing is near → full speed.
            if arm_sol:
                # Gate on the FLAG, not just the built projector: set_self_collision(False)
                # must actually disable the guard (previously a once-enabled projector kept
                # projecting forever because only `is not None` was checked).
                if self.self_collision and self.projector is not None:
                    _tp = _pc() if _prof else 0.0
                    # Hardware feedback is refreshed AFTER arm IK. A dedicated
                    # receiver can supply a newer complete sample here without
                    # charging the whole IK solve against its source age.
                    if self.enable_gripper_contact and self.physical_feedback_refresh is not None:
                        self.physical_feedback_refresh()
                    self.projector.enable_gripper_contact = self.enable_gripper_contact
                    self.projector.jaw_measured = self.gripper_measured
                    self.projector.jaw_commanded = self.gripper_commanded
                    self.projector.active_sides = (frozenset(arm_sol)
                                                   if self.enable_gripper_contact else None)
                    self.projector.arm_feedback_required = self.arm_feedback_required
                    self.projector.arm_measured = self.arm_measured
                    self.projector.arm_feedback_valid = self.arm_feedback_valid
                    self.projector.arm_feedback_reason = self.arm_feedback_reason
                    self.projector.arm_feedback_age_s = self.arm_feedback_age_s
                    self.projector.arm_feedback_compute_s = self.arm_feedback_compute_s
                    self.projector.torso_measured = self.neck_torso_measured
                    self.projector.handoff_alpha = float(getattr(self, "_gripper_contact_alpha", 0.0))
                    self.projector.handoff_separation = getattr(self, "_gripper_handoff_separation", None)
                    qr = arm_sol.get("right", d.qpos[self.arm["right"].qadr])
                    ql = arm_sol.get("left",  d.qpos[self.arm["left"].qadr])
                    qr, ql, wr, wl = self.projector.project(
                        qr, ql, self._frame_dt or (1.0 / 60.0),
                        wrist_r=wrist_sol.get("right"), wrist_l=wrist_sol.get("left"))
                    if "right" in arm_sol: arm_sol["right"] = qr
                    if "left" in arm_sol:  arm_sol["left"] = ql
                    # the ORCA wrist is part of the projection now — the hand
                    # capsule turns with it (see ArmCollisionProjector)
                    if "right" in wrist_sol: wrist_sol["right"] = wr
                    if "left" in wrist_sol:  wrist_sol["left"] = wl
                    if _prof:
                        self._prof_add("self_collision", _pc() - _tp)
                for side, wq in wrist_sol.items():
                    d.ctrl[self.wrist_act[side]] = float(wq)
                for side, sol in arm_sol.items():
                    for i, a in enumerate(self.arm_act[side]):
                        if i < len(sol):
                            d.ctrl[a] = float(sol[i])

        # NECK — preserve the operator's viewing attitude in the calibrated
        # forward frame. The chassis, torso and neck jointly realize that view:
        # remove actual chassis/torso rotation before solving the neck. The
        # forward datum stays latched, so arm movement cannot move the camera's
        # zero. `_head0` is the fallback before the body estimator has axes.
        neck_yaw_want = None                 # unclipped; feeds the base-yaw handoff
        if self.track_neck and not frozen and head_mat is not None:
            ref = None
            # THE NECK DATUM IS LATCHED, NOT LIVE (2026-07-31).
            #
            # Torso-relative is the right IDEA — a neck is your head turning ON
            # your body, and a world datum dumps all the anchor drift onto it,
            # which is what made pitch track wrong (2026-07-28). But taking the
            # datum from `shoulder.body_axes()` EVERY FRAME made the neck's zero
            # a live estimate derived from the HEAD AND BOTH WRISTS — so hand
            # motion, and any noise in that estimate, moves the neck. The
            # operator reported the consequence directly: "the neck acts super
            # erratic even when I am stationary", against a build that "was
            # working very well before" (before = `neck_angles(head_mat,
            # self._head0)`, a datum captured ONCE).
            #
            # A torso heading is not a per-frame quantity. Latching keeps every
            # gain of the torso-relative datum — it is still gravity-aligned, so
            # pitch and roll are still referred to the operator's real vertical
            # rather than to whatever attitude their head had at anchor time —
            # while removing the only thing that made it move on its own. The
            # operator re-zeroes it exactly when they mean to, with Set Forward,
            # which is what that control is for; `recalibrate` drops it too, so
            # an engage or a clutch release re-latches at the live pose.
            #
            # MABEL_NECK_DATUM=live restores the per-frame behaviour for A/B.
            if self._neck_ref is None:
                axes = self.shoulder.body_axes()
                if axes is not None:
                    right, up, fwd = axes
                    # "head neutral on the torso", in ARKit's camera convention
                    # (x right, y up, -z forward).
                    self._neck_ref = np.column_stack([right, up, -fwd])
                    self._neck_base_yaw0 = None
            if NECK_DATUM_LIVE:
                axes = self.shoulder.body_axes()
                ref = (np.column_stack([axes[0], axes[1], -axes[2]])
                       if axes is not None else None)
            else:
                ref = self._neck_ref
            if ref is None and self._head0 is not None:
                ref = self._head0[:3, :3]
            if ref is not None:
                ref4 = np.eye(4)
                ref4[:3, :3] = ref
                # Preserve viewing attitude as the base turns and torso leans,
                # including simultaneous yaw/roll (full rotation composition).
                na = neck_angles(head_mat, ref4, self._neck_torso_pitch(),
                                 self._neck_chassis_yaw())
                for a, rng, ang in zip(self.neck_act, self.neck_rng, na):
                    d.ctrl[a] = float(np.clip(ang, rng[0], rng[1]))
                neck_yaw_want = float(na[0])
        self.neck_handoff.observe(neck_yaw_want)

        # WHOLE-BODY targets (torso/base/lift follow the HEAD; no body odom from ARKit).
        if frozen:
            self.wbc.base_tgt = (0.0, 0.0, 0.0)
            self.sup.t_stream = self.sup.t_head = self.sup.t_hand = self.sup.t_arm = now
        elif self.nav_mode:
            # DRONE base driving: the stick is the SOLE base/lift command — NO
            # head-follow of the base/torso/lift. Releasing the stick zeroes the base
            # target so the base decelerates and HOLDS its position (it does not drift
            # back), and the lift STAYS where it was jogged. This restores the old
            # TeleopController._navigate feel now that the engine is the controller.
            self._apply_stick_drone(joystick)
            # DRONE-mode engagement: the frame STREAM is the operator-liveness
            # signal. A phone/web driver sends NO head or hands ("notAvailable"),
            # and requiring them let the EngagementSupervisor decay to HOME and
            # overwrite the stick's lift jog every frame (the lift axis read as
            # dead on the apps). Full tracking stays required for RETARGET mode;
            # here the e-stop + the stream watchdog cover safety.
            self.sup.t_head = self.sup.t_hand = self.sup.t_arm = now
            self.sup.observe(payload, d, now)
        else:
            info = (self.wbc.update_targets(d, head_pos, head_pos, payload.get("body_yaw"),
                                            payload.get("head_height"),
                                            reach_demand=self._reach_demand(payload),
                                            dt=self._frame_dt,
                                            # the base follower needs the TRUE source interval:
                                            # its stall rebase (relative_follow.STALL_S) and
                                            # delta/dt both break on a clipped one
                                            follow_dt=getattr(self, "_follow_dt", None),
                                            # 1:1 base follow tracks the NECK pivot, not the eyes
                                            head_mat=head_mat)
                    if self.track_arms else None)
            # Did head-follow set a base target THIS frame? If so a centered stick keeps
            # it (unified immersive teleop: the base follows the head). If NOT — a headless
            # driver (the web console / a phone in base-driving sends no head) — a centered
            # stick MUST command zero, else base_tgt latches at the last deflection and the
            # base runs on non-stop after release (there is nothing to overwrite it).
            self._apply_stick(joystick, base_from_head=bool(info and info.get("base")))
            # DEVICE-INDEPENDENT fail-safe: the desktop ZMQ stream sends elbows (+ both hands)
            # to prove "fully tracked", but a headset/phone over ARKit sends neither an
            # elbow nor (in base-driving) any hand. Requiring them made the supervisor
            # ease the robot to HOME constantly ("keeps resetting"). Here, as long as the
            # HEADSET IS ON (head tracked), keep the upper body ENGAGED and HOLDING its
            # pose — the arms hold during base-driving / a brief hand glitch. Only a lost
            # HEAD / stalled stream (headset removed) eases it to HOME.
            if payload.get("head_mat"):
                self.sup.t_hand = now
                self.sup.t_arm = now
            self.sup.observe(payload, d, now)

        if _prof:
            self._prof_add("total", _pc() - _t0)
            self._prof_report()

    def _reach_demand(self, payload):
        """How hard are the ARMS working, 0..1 — the signal that decides whether
        the base should lean in to help (see BASE_REACH_ASSIST).

        Taken as the larger of the two sides' EXTENSION FRACTION: the operator's
        wrist distance from their own shoulder, over their own arm length. That
        is already the quantity the extension arm map is built on, it is
        per-operator by construction (so a short-armed operator is not read as
        permanently stretched — the MINK-IK failure mode SEW-Mimic reports), and
        it needs no robot-side error signal, which a cascade does not have.

        Returns 0 when the shoulders are not yet estimated: no evidence means no
        assist, never a guess."""
        if BASE_REACH_ASSIST <= 0.0:
            return 0.0
        worst = 0.0
        for side in ("right", "left"):
            hm = payload.get(f"{side}_hand_mats")
            if not hm or not self.shoulder.ready(side):
                continue
            L = self.shoulder.arm_length(side)
            if not L or L <= 1e-3:
                continue
            w = np.asarray(hm[0], float).reshape(4, 4)[:3, 3]
            rho = float(np.linalg.norm(w - self.shoulder.shoulder_world(side)) / L)
            worst = max(worst, rho)
        if worst <= BASE_REACH_FREE:
            return 0.0
        span = max(BASE_REACH_FULL - BASE_REACH_FREE, 1e-6)
        return float(np.clip((worst - BASE_REACH_FREE) / span, 0.0, 1.0))

    def _apply_stick(self, joystick, base_from_head=True):
        """Unified teleop: a deflected nav stick drives the base (override the head
        follow) + jogs the lift. joystick = (lx, ly, rx, ry) in the controller frame.

        `base_from_head` says whether head-follow already set base_tgt this frame. When
        the stick is CENTERED we keep that head-follow target if present, but zero the
        base otherwise — a headless driver has no head-follow, so without this the base
        target would stick at the last deflection and never stop on release."""
        if joystick is None:
            if not base_from_head:
                self.wbc.base_tgt = (0.0, 0.0, 0.0)
            return
        lx, ly, rx, ry = joystick
        if any(abs(v) > 1e-3 for v in (lx, ly, rx)):
            self.wbc.base_tgt = (float(np.clip(ly * BASE_VMAX, -BASE_VMAX, BASE_VMAX)),
                                 float(np.clip(lx * BASE_VMAX, -BASE_VMAX, BASE_VMAX)),
                                 float(np.clip(rx * BASE_WMAX, -BASE_WMAX, BASE_WMAX)))
        elif not base_from_head:
            self.wbc.base_tgt = (0.0, 0.0, 0.0)          # release: no head-follow -> stop
        if abs(ry) > 1e-3:
            dt = self._frame_dt or (1.0 / 60.0)
            cur = float(self.data.ctrl[self.lift_act])
            self.data.ctrl[self.lift_act] = float(np.clip(cur + ry * _STICK_LIFT_RATE * dt, *LIFT_RANGE))

    def _apply_stick_drone(self, joystick):
        """Drone-style base + lift for NAVIGATION mode: the stick is the ONLY command.
        base_tgt is set EVERY frame from the stick (so on release it is ZERO → the base
        ramps to a stop and HOLDS its position, never drifting back via head-follow), and
        the lift jogs by rate and STAYS where left (the fail-safe live-snapshot holds it,
        and with head-follow skipped nothing resets it to head height)."""
        lx, ly, rx, ry = joystick if joystick is not None else (0.0, 0.0, 0.0, 0.0)
        self.wbc.base_tgt = (float(np.clip(ly * BASE_VMAX, -BASE_VMAX, BASE_VMAX)),
                             float(np.clip(lx * BASE_VMAX, -BASE_VMAX, BASE_VMAX)),
                             float(np.clip(rx * BASE_WMAX, -BASE_WMAX, BASE_WMAX)))
        if abs(ry) > 1e-3:
            dt = self._frame_dt or (1.0 / 60.0)
            cur = float(self.data.ctrl[self.lift_act])
            self.data.ctrl[self.lift_act] = float(np.clip(cur + ry * _STICK_LIFT_RATE * dt, *LIFT_RANGE))

    # ── main entry: one physics substep ───────────────────────────────────────
    def update(self, pose, joystick, dt):
        """Called once per physics substep by the bridge. The heavy retarget runs only
        when the operator frame is NEW (timestamp changed); everything else is cheap."""
        d = self.data
        now = time.time()
        ts = getattr(pose, "timestamp", None) if pose is not None else None
        # A NEW operator frame is detected by its TIMESTAMP alone — NOT by whether a
        # head pose is present. The nav joystick (drone base/lift drive) is processed
        # inside _retarget, so it must run for HEAD-LESS clients too: the web console
        # and the iPhone while driving stream trackingState "notAvailable" (head=None),
        # and gating on head here meant their stick was never processed → the base never
        # moved ("the web sends no commands"). The head-based parts of _retarget already
        # self-gate on head_mat / _world_ref, so arms/neck still won't retarget to a
        # bogus identity head when none is tracked.
        new_frame = (pose is not None and ts != self._last_ts)
        if new_frame:
            self._frame_dt = (now - self._frame_t) if self._frame_t else (1.0 / 60.0)
            self._follow_dt = self._frame_dt          # unclipped on this path already
            self._frame_t = now
            self._last_ts = ts
            payload = self._payload(pose)
            if self._cal is not None and self._cal["active"]:
                self._calibrate_step(payload, now)
                # MIRROR the calibration pose (2026-07-23, "when I fully
                # extend for calibration the robot should extend too"): the
                # old branch held the robot still while measuring, which (a)
                # gave the operator no visual confirmation, and (b) starved
                # the ShoulderEstimator of exactly the full-extension frames
                # its running-max arm-length self-calibration needs — the
                # extension map only reads rho = 1 after it has SEEN a full
                # reach. The measurement itself reads only the operator
                # payload, so retargeting alongside it changes nothing about
                # the captured reach; the retarget path manages the
                # engagement supervisor itself.
                self._retarget(payload, joystick, now)
            else:
                self._retarget(payload, joystick, now)

        # ── GRAVITY COMPENSATION — on its own clock, NEVER gated on the operator ──
        # This is a property of the ROBOT holding itself up, not of anyone's body
        # stream. It used to live inside `if new_frame:` (i.e. only when a VR/webcam
        # POSE arrived), which meant a headless client — Drive/navigation is
        # stick-only, `pose=None`, exactly the iPhone/web case the comment below
        # describes — never produced a single frame, `qfrc_applied` stayed 0, and
        # the arms hung on bare position servos. MEASURED: 0 gravity-comp DoFs and
        # 3.68 deg of sag in navigation vs 16 DoFs / 0.32 deg in joint mode. Worse,
        # every e-stop -> torque-on cycle then re-adopted the sagged pose as the new
        # target (resume_hold takes the MEASURED pose by design), so the arm
        # ratcheted further down on each cycle — the "it keeps dropping" report.
        #
        # The torque is recomputed inside the layer at its own rate (30 Hz — the
        # sag it fights is slow, and each recompute is a full-model mj_forward,
        # one of several concurrent callers whose OpenMP threads oversubscribe the
        # Jetson's cores). The WRITE happens every substep so a mode that shares
        # qfrc_applied cannot leave this path holding a stale vector.
        #
        # SAME LAYER as the torque engine (mabel.dynamic_models.gravity): this is
        # what makes Drive and Joint/Task compensate the SAME 22 upper-body DoFs.
        # Leaving Drive used to drop 8 of them — see the module docstring.
        self.gravcomp.apply(d, now, clear=True)

        # SAFETY fail-safe + base ramp, EVERY substep (cheap). The ease clock is
        # CLAMPED per tick: a single wall-clock stall (a lazy first-call import,
        # the self-collision QP build, GC, a network hiccup) must advance the
        # slow safety ease by at most SAFE_EASE_MAX_DT — un-clamped, a measured
        # 1.5 s stall at the P3 engage fast-forwarded the 2.5 s ease 63 % in ONE
        # tick and published a 1.2 rad arm step (exp_real_retarget_chain).
        wall_dt = now - self._prev
        self._prev = now
        engaged = self.sup.step(now, float(np.clip(wall_dt, 0.0, SAFE_EASE_MAX_DT)))
        if self.sup.reengaged_edge:
            self.recalibrate()
        # Base uses the STREAM watchdog ONLY — not the full-tracking `engaged` (which
        # needs head+hands+arms). Navigation is stick-only: a headless client (iPhone /
        # web sends head=notAvailable, no hands/arms) never satisfies `engaged`, so
        # gating the base on it zeroed every stick command. The stream watchdog keeps
        # the safety: if frames stop, t_stream goes stale and the base is zeroed within
        # SAFE_STREAM_TIMEOUT (the HAL's cmd_timeout is a second watchdog). The upper
        # body (sup.apply / alpha) stays gated by full `engaged` below — unchanged.
        base_live = (now - self.sup.t_stream) < SAFE_STREAM_TIMEOUT
        if not base_live:
            self.wbc.base_tgt = (0.0, 0.0, 0.0)
        self.sup.apply(d)
        # PARK the switched-off channels — AFTER the supervisor, because the
        # supervisor is the last writer of ctrl: it snapshots the retargeted
        # pose as `live` inside _retarget and writes it back here, so a park
        # applied before this was undone every single frame (measured: the neck
        # sat 16.7 deg off level and the arms 87.8 deg off rest, both stuck).
        # Ordering is the whole mechanism — safety first, then the operator's
        # per-channel intent on top of whatever safety produced.
        _pdt = float(np.clip(wall_dt, 1e-4, SAFE_EASE_MAX_DT))
        self.park["neck"].update(d, self.track_neck, _pdt)
        self.park["arms"].update(d, self.track_arms, _pdt)
        self.park["hands"].update(d, self.track_hands, _pdt)
        # Torso/lift parks apply to RETARGETING only: in DRIVE (nav_mode) the
        # stick owns the lift, and an ablation switch must not make the drive
        # pad's lift slider a dead control.
        self.park["torso"].update(d, self.track_torso or self.nav_mode, _pdt)
        self.park["lift"].update(d, self.track_lift or self.nav_mode, _pdt)
        # MOBILE-BASE WORKSPACE ASSIST — applied HERE, at the point of
        # consumption, not inside _retarget.
        #
        # `_retarget` is memoised on frame identity so it runs at the operator's
        # 30 Hz, while `base_step` consumes base_tgt every 100 Hz control tick
        # and `_apply_stick` rewrites base_tgt from the stick each time it does
        # run. Adding the assist in _retarget therefore reached only a fraction
        # of the ticks: measured base_tgt.x p50 0.0012 against a commanded 0.12,
        # with the intended value appearing only at the max. Added at the
        # consumption point it is present on every tick, and it is still purely
        # ADDITIVE to head-follow and the stick.
        # GATED ON base_in_loop — i.e. on GLOBAL scope. LOCAL scope deliberately
        # keeps the base out of the follow loop and leaves it to the joystick,
        # and `test_walking_does_NOT_drive_the_base_in_LOCAL_scope` pins that.
        # The ANCHOR OFFSET still applies in LOCAL (it is a retargeting choice,
        # not base motion, and it is what buys the reachability); what LOCAL
        # gives up is paying the debt off, so the hands stay displaced in the
        # world by up to ARM_WS_MAX_OFFSET. That is the honest price of reaching
        # a crowded target with the base pinned.
        if (self._ws_assist is not None and self.track_arms and not self.nav_mode
                and getattr(self.wbc, "base_in_loop", False)
                and self.wbc.allow_base_assist):
            _ax, _ay = self._ws_assist.base_twist()
            if _ax or _ay:
                _bt = tuple(self.wbc.base_tgt or (0.0, 0.0, 0.0))
                self.wbc.base_tgt = (_bt[0] + _ax, _bt[1] + _ay, _bt[2])
        # NECK-YAW HANDOFF: the chassis carries a look past the neck's range
        # (neck_handoff.py). Every scope, but never in DRIVE (the stick owns the
        # base), never clutched, and only while engaged and streaming.
        _wz = self.neck_handoff.rate(
            self._neck_chassis_yaw(), dt,
            live=(engaged and base_live and not self.nav_mode and self.track_neck
                  and not self.clutch.clutched))
        self.wbc.base_step(d, dt, extra_wz=_wz)
        self.mode = "CLUTCH (hold)" if self.clutch.clutched else self.sup.state()
