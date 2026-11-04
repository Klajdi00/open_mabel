"""MabelTeleop -- the one-call teleop facade shared by every frontend.

It composes the production stack (TeleopController: retarget + whole-body IK + swerve +
neck + self-collision) with the safety layer (tracking-loss fail-safe + double-pinch
clutch) AND the whole-body torque engine (WBCEngine: gravity-comp / compliance / joint
impedance), behind a single frontend-agnostic call:

    teleop = MabelTeleop(model, data)
    teleop.calibrate(pose)                 # operator extends arms / T-pose
    while running:
        cmd = teleop.update(pose, dt=dt)   # OperatorPose -> RobotCommand (writes data.ctrl)

Everything the apps can send maps onto a facade method, so the transport bridge is pure
glue and the control logic lives entirely in this package:

    set_control_spec(method, control_type, region, stiffness)  # control_mode message
    set_joint_targets(dict) / clear_joint_targets()            # joint_command message
    apply_external_force({body: [fx,fy,fz]})                   # external_force message
    recalibrate()                                              # recalibrate message
    reset()                                                    # reset message

`model, data` is the robot's MuJoCo kinematic state: in the sim it is the physics
MjData; on hardware it is a kinematic mirror whose qpos you set from the joint encoders
each frame (so the same FK-based retargeting/whole-body math runs identically). The
returned RobotCommand carries the actuator targets for a hardware bridge to forward.
"""
import os
import sys
import time

from . import config as C
from .whole_body_controller.base_controller.swerve import swerve_ik
from .whole_body_controller.engine import WBCEngine, _MJ_TO_IOS
from .hand_retargeting.grasp import apply_grasp
from .types import OperatorPose, RobotCommand
from .whole_body_retargeting.calibration import BodyCalibrator, OperatorProfile

# The kinematic generator (retarget + navigation + neck) is the TeleopEngine —
# it REPLACES the dated TeleopController. Since 2026-07-22 it lives INSIDE this
# package (mabel.teleop_engine, moved from simulation/mabel_mujoco/scripts/
# teleop), so sim and the real robot share one installable stack — no sys.path
# reach-out; a shim at the old path keeps legacy `import teleop` working.
from .teleop_engine import TeleopEngine

# The control SPEC: which reference generator + torque law is live. `method` selects
# the generator (navigation -> base joystick, retarget -> hand IK, joint/wrist -> WBC
# torque targets); `control_type` + `region` parameterise the torque engine; `stiffness`
# is the impedance spring K. Defaults = plain manipulation, no joystick.
# type: "pd" since 2026-08-23 — PURE per-joint PD (operator request). It is the
# same law "impedance" ran, minus the two things that stopped it being a PD: one
# scalar spring shared across joints whose inertias differ 170x, and a low-passed
# reference that only existed to stop that scalar spring stepping on a jumpy IK
# target. See whole_body_controller/engine.py CONTROL_TYPES and config.PD_WN.
_DEFAULT_SPEC = {"method": "navigation", "type": "pd", "region": "arm",
                 "scope": "local",   # mobile base OUT of the follow loop
                 "stiffness": 0.0,
                 "tracking": {"neck": True, "arms": True, "hands": True,
                              "torso": True, "lift": True},
                 # Operator switches, both defaulting to the safe/better answer:
                 # the arm is compensated, and the arms run the retargeting that
                 # measures better on wrist accuracy.
                 "gravity_comp": True,
                 "whole_body_retarget": False}
_WBC_METHODS = ("joint", "wrist")     # methods that engage the torque engine


def _is_finger_key(k):
    return (k.endswith("_grip") or k.endswith("_wrist")
            or any(f in k for f in ("thumb", "index", "middle", "ring", "pinky")))


class MabelTeleop:
    def __init__(self, model, data, base_follow=True, fail_safe=True, clutch=True):
        self.model = model
        self.data = data
        # The retarget/navigation generator: the scripts TeleopEngine (replaces the
        # old TeleopController). It owns its OWN clutch + tracking-loss fail-safe +
        # gravity comp, so the facade no longer runs a second clutch/supervisor on the
        # kinematic path (that caused the dual-backend conflicts). The facade remains the
        # orchestrator: control-spec routing, the WBC torque modes, joint-target / grip
        # overrides, external force, and body calibration.
        self.controller = TeleopEngine(model, data)
        self.wbc = WBCEngine(model, data)
        lift_act = model.actuator("Lift").id
        self.sup = None      # fail-safe now lives in the engine (self.controller)
        self.clutch = None   # clutch now lives in the engine (self.controller)
        self._prev = time.time()

        # Orchestration state the apps drive (was scattered across the bridge globals).
        self.spec = dict(_DEFAULT_SPEC)
        self.joint_targets = {}        # {ios joint id: value} direct overrides + grips
        # JOINT SPEED LIMIT — REAL-ROBOT SAFETY, OFF BY DEFAULT (sim poses instantly).
        #
        # The arm firmware computes kp*(pos - q) on ITS side, so the host cannot clamp
        # torque — only the commanded ERROR. A slider dragged to an extreme is a huge
        # instant error: at kp=30 a 3.3 rad jump demands ~99 N.m, which parks the joint
        # against a stop and cooks the driver (its over-current trip is a LATCH needing
        # a power cycle). Every joint target from every input device funnels through
        # _apply_joint_targets -> setact, so limiting THERE bounds kp*error for all of
        # them at once; measured 99 N.m -> 1.13 N.m peak, slewing at 1.5 rad/s.
        #
        # None here keeps SIM semantics exactly as they were (a direct joint target is
        # an instantaneous pose — no motors to protect, and the e2e tests rely on it).
        # The REAL dispatch turns it on: see server/sim_teleop_bridge.py, which does
        #     teleop.arm_rate_limit = JointRateLimiter(...)
        # when the target is real:*. Tune per joint with
        # JointRateLimiter(per_joint={"Left Arm 2": 0.8}) or MABEL_ARM_MAX_SPEED.
        self.arm_rate_limit = None
        self.ext_forces = {}           # {body_name: [fx,fy,fz]} operator push (world)
        self._wbc_dirty = False        # re-enter / exit the torque engine on spec change
        self._recal = False            # pending clutch re-anchor (recalibrate message)

        # SELF-COLLISION FILTER FOR CLIENT-PLANNED TARGETS (2026-07-29). Every
        # wrist-tracking client plans in its own head and ships joint targets —
        # the web control center solves wrist IK in the BROWSER, iOS/VP solve it
        # on-device — so those targets never met the self-collision guard (it
        # lived only in the server-side retarget IK + the WBC torque soft-wall).
        # The JointTargetProjector is the ONE choke point: manipulate-method
        # (joint/wrist) arm/wrist/torso targets flow through its CBF velocity
        # projection (keep tangential, remove the normal closing excess) so a
        # target inside the body makes the arm SLIDE along the torso/head/other
        # arm and settle on the boundary — identically for web, iOS and
        # visionOS. Built lazily; a failed build degrades to unfiltered targets
        # (never crash teleop for a guard).
        self.sc_targets = bool(C.SELF_COLLISION_ENABLE)
        self._target_proj = None
        self._target_proj_dead = False

        # Per-operator body/hand calibration (calibration.py). While a BodyCalibrator
        # is active the robot is held still and frames feed the 3-pose capture; the
        # resulting OperatorProfile parameterises the retargeting scale / shoulder
        # anchor / per-finger hand size (controller.apply_profile).
        self.body_cal = None
        self.profile = None
        self._cal_applied_id = None     # one-shot guard: apply each calibrator's profile once

    @property
    def calibrated(self):
        return self.controller.calibrated

    @property
    def body_calibrating(self):
        return self.body_cal is not None and self.body_cal.active

    # ── per-operator body/hand calibration ─────────────────────────────────────
    def start_body_calibration(self, user="default"):
        """Begin the guided 3-pose anthropometry capture. While it runs, update()
        holds the robot still and routes operator frames to the calibrator; the
        profile is applied + saved automatically on completion."""
        self.body_cal = BodyCalibrator(user=user)
        return self.body_cal

    def cancel_body_calibration(self):
        self.body_cal = None

    def apply_profile(self, profile):
        """Apply (and remember) a per-operator OperatorProfile across the stack."""
        self.profile = profile
        self.controller.apply_profile(profile)
        return profile

    def load_profile(self, user="default", directory=None):
        """Load a cached OperatorProfile for ``user`` from disk and apply it; returns
        the profile or None if none is cached."""
        prof = OperatorProfile.load(user, directory)
        if prof is not None:
            self.apply_profile(prof)
        return prof

    def calibration_status(self):
        """UI/telemetry: current pose prompt + progress, or None when idle."""
        if self.body_cal is None:
            return None
        return {"active": self.body_cal.active, "pose": self.body_cal.current_pose,
                "index": self.body_cal.pose_idx, "total": 3, "done": self.body_cal.done}

    # ── calibration / lifecycle ───────────────────────────────────────────────
    def calibrate(self, pose):
        """Anchor the operator<->robot mapping (call once the operator is in the
        calibration pose, e.g. arms extended forward)."""
        return self.controller.calibrate(pose)

    def recalibrate(self):
        """Request a clutch re-anchor at the operator's CURRENT pose on the next frame
        (e.g. the headset re-enabled hand tracking / recentred -- no jump)."""
        self._recal = True

    def reset(self):
        """Re-home the controller after a sim reset: hold the current (home) pose, drop
        accumulated joint commands / pushes, and clear the safety state so nothing
        springs back to a pre-reset target. The caller resets the physics (qpos)."""
        self.controller.reset()                      # engine: clears refs + re-anchors
        self.controller.hold_current()
        self.joint_targets.clear()
        if self._target_proj is not None:
            self._target_proj.reset()
        self.ext_forces.clear()                      # clear IN PLACE — the bridge binds its
                                                     # _ext_forces mailbox to this dict by reference
                                                     # (teleop.ext_forces = _ext_forces); rebinding it
                                                     # to a new {} silently orphaned the binding, so
                                                     # every push after a reset was dropped (soft mode
                                                     # appeared dead). joint_targets above already
                                                     # clears in place — match it.
        if self.wbc.active:
            self.wbc.reset_targets()
        if self.sup is not None:
            self.sup.reset()
        if self.clutch is not None:
            self.clutch.reset()
        self._recal = False

    # ── app-driven orchestration (was in the bridge) ──────────────────────────
    def set_control_spec(self, method=None, control_type=None, region=None,
                         tracking=None, scope=None,
                         stiffness=None, type=None, gravity_comp=None, whole_body_retarget=None):
        """Set the active control spec (control_mode message). Safe to call every tick
        from the active driver; it only marks the torque engine dirty when the method /
        type / region actually changes (the (re)enter/exit then happens on the control
        thread inside update). `type` is accepted as an alias for `control_type`.

        """
        control_type = control_type if control_type is not None else type
        # RETARGET SCOPE ('local' default | 'global'): is the mobile BASE in the
        # follow loop? Orthogonal to method/type/region — region scopes which
        # joints the TORQUE engine owns, scope decides whether the base follows
        # the operator or waits for the joystick. Applied straight through to
        # the engine, which normalises it and re-anchors bumplessly.
        if scope is not None:
            self.spec["scope"] = self.controller.set_retarget_scope(scope)
        old = (self.spec["method"], self.spec["type"], self.spec["region"])
        if method is not None:       self.spec["method"] = method
        if control_type is not None: self.spec["type"] = control_type
        if region is not None:       self.spec["region"] = region
        # PER-CHANNEL TRACKING: which operator body parts drive the robot. A
        # channel switched off is not computed and eases to neutral, so this is
        # a real control-law change, not a UI filter — it belongs here with the
        # other three axes rather than in a client.
        if isinstance(tracking, dict):
            setter = getattr(self.controller, "set_tracking", None)
            if setter is not None:
                self.spec["tracking"] = setter(
                    neck=tracking.get("neck"), arms=tracking.get("arms"),
                    hands=tracking.get("hands"), torso=tracking.get("torso"),
                    lift=tracking.get("lift"))
        # GRAVITY COMP and WHOLE-BODY RETARGET: two orthogonal operator switches.
        # They live here with the other axes rather than in a client because both
        # are real control-law changes — one decides whether the shared
        # feed-forward layer feeds anything, the other picks which of the two
        # retargeting profiles the arms run.
        if gravity_comp is not None:
            self.spec["gravity_comp"] = self.controller.set_gravity_comp(gravity_comp)
            # keep the torque engine's copy of the SAME layer in step — the two
            # engines share one object, but the WBC holds its own reference
            wbc_g = getattr(self.wbc, "gravcomp", None)
            if wbc_g is not None and wbc_g is not self.controller.gravcomp:
                wbc_g.set_enabled(gravity_comp)
        if whole_body_retarget is not None:
            self.spec["whole_body_retarget"] = \
                self.controller.set_whole_body_retarget(whole_body_retarget)
        if stiffness is not None:
            try:
                self.spec["stiffness"] = float(stiffness)
            except (TypeError, ValueError):
                pass
        if (self.spec["method"], self.spec["type"], self.spec["region"]) != old:
            self._wbc_dirty = True

    def set_joint_targets(self, joints):
        """Accumulate direct joint targets (joint_command message: iOS Body tab,
        future controllers). {ios joint id: value}; applied AFTER IK so they win."""
        if joints:
            self.joint_targets.update(joints)

    def _reseed_targets_from_pose(self):
        """Overwrite any STORED pose targets with the robot's measured pose —
        arms/torso via the WBC joint map, the lift from its actuator. Grips and
        finger targets are left alone (synthetic [0,1] values, not qpos)."""
        for mj, ios in _MJ_TO_IOS.items():
            if ios in self.joint_targets:
                try:
                    self.joint_targets[ios] = float(self.wbc.plant.q(mj))
                except Exception:
                    pass
        if "lift_lower" in self.joint_targets or "lift_upper" in self.joint_targets:
            try:
                lift = float(self.data.actuator("Lift").ctrl[0])
                self.joint_targets["lift_lower"] = lift / 2.0
                self.joint_targets["lift_upper"] = lift / 2.0
            except Exception:
                pass
        # The reseed rewrote the targets to "hold here" — the collision filter's
        # plan must restart from measured too, or its stale head would glide the
        # arm away from the pose the reseed just froze.
        if self._target_proj is not None:
            self._target_proj.reset()

    def clear_joint_targets(self):
        self.joint_targets.clear()
        if self._target_proj is not None:
            self._target_proj.reset()

    def apply_external_force(self, forces):
        """Operator push anywhere on the arm (external_force message): a world-frame
        wrench {body_name: [fx,fy,fz]} the compliance/whole-body engine admits."""
        self.ext_forces = dict(forces) if forces else {}

    def set_self_collision(self, on, method=None):
        """Toggle capsule self-collision avoidance across the kinematic IK, the
        torque engine, AND the client-target filter (the manipulate joint/wrist
        path) — one switch, every path."""
        self.controller.set_self_collision(on, method)
        self.wbc.set_self_collision(on)
        self.sc_targets = bool(on)
        if not on and self._target_proj is not None:
            self._target_proj.reset()

    # ── e-stop RESUME: adopt the MEASURED pose (never snap to a stale target) ──
    def _measured_ctrl(self, act_name):
        """The MEASURED angle of the joint an actuator drives, clamped to its ctrl
        range — i.e. what to command so the joint does NOT move. On the real robot
        data.qpos is the encoder mirror (ROSPlant.read_state), so this is ground
        truth; in sim it is the physics state. None if the actuator has no joint."""
        try:
            aid = self.model.actuator(act_name).id
        except Exception:
            return None
        lo, hi = self.model.actuator_ctrlrange[aid]
        if act_name == "Lift":
            # Tendon transmission: trnid[aid,0] is a TENDON id, not a joint id —
            # indexing jnt_qposadr with it reads an unrelated joint (the free
            # joint's x!) and made torque re-enable SNAP the lift. The measured
            # quantity ctrl targets is the tendon length = the stage sum.
            q = (float(self.data.joint("Lift Lower").qpos[0])
                 + float(self.data.joint("Lift Upper").qpos[0]))
            return float(min(max(q, lo), hi))
        jid = int(self.model.actuator_trnid[aid, 0])
        if jid < 0:
            return None
        adr = self.model.jnt_qposadr[jid]
        return float(min(max(float(self.data.qpos[adr]), lo), hi))

    def resume_hold(self):
        """E-STOP RELEASE / any torque re-enable: make the COMMANDED pose equal the
        MEASURED pose, so re-energising holds the joints exactly where they are.

        SAFETY — this is the whole point: while the robot is e-stopped the joints are
        limp and get moved (by hand, by gravity, by a collision), but `data.ctrl` and
        `joint_targets` still hold the pre-e-stop command. Re-enabling torque without
        this made every joint SNAP back to that stale target at full servo authority —
        a violent, dangerous lurch (the hardware's own hold-then-enable could not stop
        it: the bridge latches goal:=present, then the gateway's very next command
        frame overwrites it with the stale target).

        Reseeds BOTH command paths — the control law's servo targets (`hold_current`)
        AND the app's direct `joint_targets` overrides (which are applied AFTER the
        control law, so reseeding only the former would still snap) — then re-anchors
        the clutch so teleop resumes from the operator's current pose without a jump."""
        self.controller.hold_current()      # every position servo: ctrl := measured qpos
        if self._target_proj is not None:
            self._target_proj.reset()       # plan restarts from measured, like the targets
        j = self.joint_targets
        for key in list(j):
            if key.endswith("_grip"):
                # Synthetic [0,1] grasp, not a joint angle: replaying it would re-close
                # the hand on whatever is now between the fingers. Drop it.
                del j[key]
            elif key in ("lift_lower", "lift_upper"):
                q = self._measured_ctrl("Lift")
                if q is None:
                    del j[key]
                else:
                    j[key] = q / 2.0        # two stages share the one Lift actuator
            else:
                act = self._actuator_for(key)
                q = self._measured_ctrl(act) if act else None
                if q is None:
                    del j[key]              # unmappable → drop rather than replay a stale value
                else:
                    j[key] = q
        # TRACK THE CURRENT POSITION (operator directive): on torque enable, the
        # commanded pose MUST equal the MEASURED pose for EVERY holdable joint — not
        # only the ones the app has already touched. Without this, an arm the app
        # never sent a target for is NOT in joint_targets, so _apply_joint_targets
        # leaves it to the control law, which drives it to the IK/rest pose instead
        # of holding where it is — the "arm doesn't track current position / stays at
        # the wrong target" report. Seed arms + wrists + torso here so the robot holds
        # its measured pose until the app moves a joint (or Soft/compliance takes over).
        #
        # EXCEPT in RETARGET (whole-body teleop): there the engine is the live
        # generator for arms/wrists/torso — hold_current has already adopted the
        # measured pose and recalibrate() below re-anchors the clutch, so the engine
        # holds without seeds. Seeding here instead FROZE the whole upper body: the
        # seeds are re-applied by _apply_joint_targets AFTER the retargeter every
        # tick, overriding it forever. Invisible in old sim runs (the VP app's
        # torque/e-stop wire never reached the server, so resume_hold never fired);
        # on the real robot the ENABLE-TORQUE edge is mandatory, so whole-body
        # teleop arms were ALWAYS frozen at the enable pose.
        if self.spec["method"] != "retarget":
            for pre in ("left", "right"):
                for i in range(1, 8):
                    q = self._measured_ctrl(f"{pre.capitalize()} Arm {i}")
                    if q is not None:
                        j[f"{pre}_arm_{i}"] = q
                wq = self._measured_ctrl(f"{pre.capitalize()} Wrist")
                if wq is not None:
                    j[f"{pre}_wrist"] = wq
            tq = self._measured_ctrl("Torso")
            if tq is not None:
                j["torso"] = tq
        else:
            # Drop any stale upper-body seeds a previous mode left behind — same
            # ownership rule as the retarget-entry pops in update().
            for pre in ("left", "right"):
                for i in range(1, 8):
                    j.pop(f"{pre}_arm_{i}", None)
                j.pop(f"{pre}_wrist", None)
            j.pop("torso", None)
        if self.wbc.active:
            self.wbc.reset_targets()        # torque modes: targets := current, no spring-back
        self.recalibrate()                  # re-anchor the operator clutch (no teleop jump)

    # ── direct joint override (was bridge._apply_joint_cmd) ───────────────────
    _FINGER = {"thumb": "Thumb", "index": "Index", "middle": "Middle",
               "ring": "Ring", "pinky": "Pinky"}
    _SEG = {"mcp": "MCP", "pip": "PIP", "dip": "DIP", "abd": "Abd"}

    @classmethod
    def _actuator_for(cls, key):
        """iOS joint id -> MuJoCo position-servo actuator name (None if the id is not a
        single-joint servo: grips are synthetic, the lift is a two-key sum). ONE mapping,
        shared by the override path and the resume reseed — they cannot diverge."""
        parts = key.split("_")
        side = "Left" if parts[0] == "left" else "Right" if parts[0] == "right" else None
        if key == "torso":
            return "Torso"
        if side and len(parts) == 3 and parts[1] == "arm" and parts[2].isdigit():
            return f"{side} Arm {int(parts[2])}"
        if side and len(parts) == 2 and parts[1] == "wrist":
            return f"{side} Wrist"
        if side and len(parts) == 3 and parts[1] in cls._FINGER and parts[2] in cls._SEG:
            return f"{side} {cls._FINGER[parts[1]]} {cls._SEG[parts[2]]}"
        return None

    #: Joints whose command is speed-limited. The firmware closes kp*(pos-q) on ITS
    #: side, so the ONLY way this host can bound arm torque is to bound the commanded
    #: ERROR — and the HAL deliberately carries no slew guard (it clamps t_ff to the
    #: motor's rated t_max and position to the calibrated end-stops, nothing else).
    #: Gravity-loaded arm joints are what a slider slam actually damages; fingers/lift
    #: are small, geared, and self-limiting, so they stay direct.
    _RATE_LIMITED = ("Arm", "Torso", "Wrist")

    def _rate_limit(self, name, aid, val, dt):
        """Bound `val` to arm_rate_limit.max_speed rad/s from where the joint IS.

        Measured comes from data.qpos — on real:ros that mirror is fed by the real
        encoders, so this bounds the REAL position error (hence kp*error, hence
        torque). Limiting against the previous COMMAND instead would let command and
        joint drift arbitrarily far apart while each step still looked small — the
        torque would grow unbounded anyway, which is the whole failure this prevents.
        Returns `val` unchanged for non-limited actuators or when the joint is
        unreadable (we cannot bound an error we cannot see)."""
        lim = self.arm_rate_limit
        if lim is None or dt is None or not any(k in name for k in self._RATE_LIMITED):
            return val
        try:
            jid = int(self.model.actuator_trnid[aid][0])
            cur = float(self.data.qpos[self.model.jnt_qposadr[jid]])
        except Exception:
            return val
        return lim.step({name: val}, {name: cur}, dt=dt)[name]

    def _sc_filter(self, dt):
        """{actuator: rad} — the CBF-projected values for this tick's manipulate
        joint targets, or {} when the filter is off / inapplicable.

        Runs only for the manipulate methods ("joint"/"wrist" — the paths whose
        arm targets are CLIENT-planned and therefore unguarded), and only over
        the guarded actuators (arms + wrists + torso). The projector's plan
        glides from the measured pose toward the raw target at SC_TARGET_VMAX,
        each step projected through the capsule CBF: keep the tangential
        component, remove the normal (gap-closing) excess. Failures degrade to
        unfiltered targets — a guard must never take teleop down."""
        if (not self.sc_targets or self._target_proj_dead
                or self.spec["method"] not in _WBC_METHODS):
            if self._target_proj is not None:
                self._target_proj.reset()
            return {}
        j = self.joint_targets
        if not j:
            if self._target_proj is not None:
                self._target_proj.reset()
            return {}
        try:
            if self._target_proj is None:
                from .whole_body_controller.collision_avoidance import (
                    JointTargetProjector)
                self._target_proj = JointTargetProjector(self.model, self.data)
            tgt = {}
            for key, val in j.items():
                act = self._actuator_for(key)
                if act is not None and act in self._target_proj._qadr:
                    tgt[act] = float(val)
            if not tgt:
                self._target_proj.reset()
                return {}
            out = self._target_proj.step(tgt, dt if dt else 0.02)
            return {a: out[a] for a in tgt}
        except Exception:                              # noqa: BLE001
            self._target_proj_dead = True              # degrade loudly-once, never crash
            print("[facade] self-collision target filter failed — "
                  "manipulate targets pass through UNFILTERED", file=sys.stderr)
            return {}

    def _apply_joint_targets(self, skip=None, sc=None):
        """Override mapped actuator commands with the accumulated direct joint targets
        so ANY input device poses the robot through the same path. Runs AFTER the
        control law so it takes priority for the commanded joints; untouched joints keep
        their controller command. iOS ids -> MuJoCo actuator names, clamped to range.

        `skip` is a set of actuator names to leave alone — used in COMPLIANCE (Soft) to
        take NO position target from the app on the torque-managed arm/torso joints, so a
        dragged arm holds wherever the operator leaves it. Grips / lift / fingers (never
        torque-managed) still track in every mode.

        `sc` is the {actuator: rad} CBF-projected value map from _sc_filter — a
        guarded actuator commands the PROJECTED value (the collision-avoided plan)
        instead of the raw client target."""
        j = self.joint_targets
        if not j:
            return
        skip = skip or set()
        sc = sc or {}
        model, data = self.model, self.data
        # ONE clock tick for the whole pass — every joint is limited against the SAME
        # elapsed dt. (Ticking per joint would give the first joint the entire interval
        # and leave the other 13 with dt~=0, i.e. frozen.)
        _dt = (self.arm_rate_limit.tick()
               if self.arm_rate_limit is not None else None)

        def setact(name, val):
            if name in skip:
                return
            try:
                aid = model.actuator(name).id
            except Exception:
                return
            val = sc.get(name, val)     # CBF-projected plan beats the raw target
            lo, hi = model.actuator_ctrlrange[aid]
            val = float(min(max(val, lo), hi))
            val = self._rate_limit(name, aid, val, _dt)
            data.actuator(name).ctrl[0] = val

        # Hand open/close: a single [0,1] grip curls all fingers (0=open, 1=fist).
        for pre, Side in (("left", "Left"), ("right", "Right")):
            if (grip := j.get(f"{pre}_grip")) is not None:
                apply_grasp(model, data, Side, grip)
        if (t := j.get("torso")) is not None:
            setact("Torso", t)
        # Neck: the ONE mapped group this override was missing — neck_* keys in
        # joint_command were silently dropped, so joint-space neck control never
        # moved the robot on ANY client ("the neck joystick does not work",
        # verified by server/tools/exp_neck_stomp.py). Runs after the control
        # law like everything else; retarget entry pops stale neck targets so
        # the head-follow owns the neck in whole-body mode.
        for key, act in (("neck_yaw", "Neck Yaw"), ("neck_pitch", "Neck Pitch"),
                         ("neck_roll", "Neck Roll")):
            if (v := j.get(key)) is not None:
                setact(act, v)
        if "lift_lower" in j or "lift_upper" in j:
            setact("Lift", float(j.get("lift_lower", 0.0)) + float(j.get("lift_upper", 0.0)))
        # Every single-joint servo target (arms, wrists, torso, ORCA fingers) through the
        # ONE id->actuator map `_actuator_for`, which resume_hold reseeds against.
        for key, val in j.items():
            act = self._actuator_for(key)
            if act is not None:
                setact(act, val)

    # ── per-tick control law ──────────────────────────────────────────────────
    def update(self, pose, joystick=None, dt=None):
        """Advance one control substep: OperatorPose -> RobotCommand (also writes
        self.data.ctrl). One call runs the full stack -- spec (re)entry, clutch,
        retarget + whole-body IK OR the torque engine, fail-safe, and the direct joint
        override. The caller owns physics (mj_step) and its real-time pacing."""
        now = time.time()
        wall_dt = now - self._prev
        self._prev = now
        if dt is None:
            dt = wall_dt

        # 0a) BODY CALIBRATION owns the loop while active: hold the robot still and
        #     feed the operator frame to the 3-pose capture. On completion the profile
        #     is applied + saved EXACTLY ONCE (guarded by the calibrator's identity, so
        #     it fires whether the finishing frame lands inside or just before this
        #     branch), then the calibrator is KEPT (inactive) so calibration_status()
        #     reports done=True to the apps as one clean signal while the loop falls
        #     through to normal teleop.
        if self.body_cal is not None:
            if self.body_cal.active:
                self.body_cal.observe(pose, now)
            if (self.body_cal.done and self.body_cal.profile is not None
                    and id(self.body_cal) != self._cal_applied_id):
                self.apply_profile(self.body_cal.profile)
                try:
                    self.body_cal.profile.save()
                except OSError:
                    pass
                self._cal_applied_id = id(self.body_cal)
            if self.body_cal.active:
                self.controller.hold_current()
                swerve_ik(0.0, 0.0, 0.0, self.data)
                return RobotCommand(ctrl=self.data.ctrl.copy(), engaged=0.0,
                                    clutched=False, mode="CALIBRATING:BODY")

        # 0) (re)anchor the clutch: on an explicit request, or automatically on the
        #    first usable frame after (re)connect / reset.
        if pose is not None and getattr(pose, "head", None) is not None:
            if self._recal or not self.controller.calibrated:
                if self.controller.calibrate(pose):
                    self._recal = False

        # 1) engage / release the torque engine on a spec change (this thread owns the
        #    actuator-gain edits). Manipulation methods (joint / wrist) run the
        #    impedance/compliance law; navigation and retarget run the kinematic stack.
        if self._wbc_dirty:
            self._wbc_dirty = False
            # BUMPLESS mode switch: re-seed the persistent joint targets from the
            # MEASURED pose so the incoming law holds the robot exactly where it
            # IS. joint_targets can be minutes stale (the app's last command) —
            # wbc.enter seeds q_des to the current pose, but one tick later
            # set_targets(joint_targets) sprang the arms back to the stale dict
            # (and in NAV, _apply_joint_targets did the same to the servos): the
            # visible "arms drop when I switch drive↔joint/task". Targets may
            # only move again when the operator actually commands (slider /
            # joystick / grab-IK / whole-body stream).
            self._reseed_targets_from_pose()
            if self.spec["method"] in _WBC_METHODS:
                self.controller.hold_current()                 # freeze before torque
                self.data.qfrc_applied[:] = 0.0                # drop the engine's gravity FF; the torque law owns forces
                self.wbc.enter(self.spec["type"], self.spec["region"], q_des=self.joint_targets)
            else:
                self.wbc.exit()
                self.controller.hold_current()                 # resume position hold cleanly
            if self.spec["method"] == "retarget":
                # RETARGET OWNS THE WHOLE UPPER BODY: hand IK the fingers, the
                # head-follow the neck, the retargeter + whole-body cascade the
                # arms/torso and the LIFT (head-height follow). Drop every stale
                # direct target for those joints — _apply_joint_targets runs AFTER
                # the engine each tick, so one leftover key (a Body-tab slider, a
                # nav lift jog, a resume_hold seed) permanently freezes that limb.
                for k in [k for k in self.joint_targets if _is_finger_key(k)]:
                    self.joint_targets.pop(k, None)
                for k in ("neck_yaw", "neck_pitch", "neck_roll", "torso",
                          "lift_lower", "lift_upper"):
                    self.joint_targets.pop(k, None)
                for pre in ("left", "right"):
                    for i in range(1, 8):
                        self.joint_targets.pop(f"{pre}_arm_{i}", None)

        # 3a) TORQUE ENGINE owns the arms/base: apply the operator push, run the
        #     impedance/compliance law, still honour grips/lift from joint targets.
        if self.wbc.active:
            # The ENGINE's 1-pose reach calibration must still see operator
            # frames HERE: this branch returns without ever calling
            # controller.update, which silently starved the capture whenever a
            # WBC method was active — the app's guided popup measured locally
            # but the server never confirmed, so it never said "finished"
            # (2026-07-18). The robot is meant to hold still during the capture
            # anyway; the torque law below keeps holding while the engine
            # measures. Frame-gated inside calibrate_tick (one sample per frame).
            cal_tick = getattr(self.controller, "calibrate_tick", None)
            if cal_tick is not None:
                cal_tick(pose, now)
            self.wbc.set_stiffness(self.spec["stiffness"])
            soft = self.wbc.control_type == "compliance"
            # COMPLIANCE (Soft) is pure gravity-comp: it takes NO position target from
            # the app — q_des auto-tracks wherever the operator drags the arm (viewer /
            # real robot), so it holds there. Only IMPEDANCE (Hold) consumes the app's
            # joint targets as the set-point it springs to. This is exactly the law in
            # simulation/mabel_mujoco/scripts/compliance.py (same WBCEngine).
            # CBF-project the client-planned manipulate targets ONCE per tick;
            # both the torque engine's q_des and the servo overrides consume the
            # SAME projected plan (else the two paths would disagree about where
            # the arm is going).
            scf = self._sc_filter(dt)
            if not soft:
                if scf:
                    jt = dict(self.joint_targets)
                    for key, val in self.joint_targets.items():
                        act = self._actuator_for(key)
                        if act in scf:
                            jt[key] = scf[act]
                    self.wbc.set_targets(jt)
                else:
                    self.wbc.set_targets(self.joint_targets)
            self.wbc.apply_external(self.ext_forces)
            self.wbc.step(dt)
            # Direct overrides drive only the NON-managed joints (grips / lift / fingers)
            # in Soft; the torque-managed arm/torso joints ignore app position targets.
            self._apply_joint_targets(skip=set(self.wbc.managed) if soft else None,
                                      sc=scf)
            return RobotCommand(ctrl=self.data.ctrl.copy(), engaged=1.0, clutched=False,
                                mode=f"WBC:{self.spec['type']}·{self.spec['region']}",
                                base=(self.wbc.vx, self.wbc.vy, self.wbc.w))

        # 3b) KINEMATIC stack (navigation + retarget): the engine generator owns the
        #     WHOLE law for these methods — its own double-pinch clutch, retarget +
        #     arm/finger IK, neck, base/lift followers, gravity comp, AND the
        #     tracking-loss fail-safe (head-only-tolerant while driving). The facade
        #     just applies the direct joint/grip overrides on top.
        #     NAVIGATION = drone drive (stick-only base/lift, no head-follow: the base
        #     stops & holds on release, the lift stays where jogged); RETARGET = the
        #     immersive whole-body head-follow stack.
        if self.spec["method"] == "retarget":
            # RETARGET OWNS THE WHOLE UPPER BODY — scrub direct overrides EVERY
            # tick, not just at entry. The entry-time pop (step 1) is not enough:
            # the bridge ALIASES joint_targets to its accumulated _joint_cmd and
            # the apps re-send their WHOLE accumulated joint dict on any change,
            # so ARM keys left by a Joint-space nudge / executed wrist plan / grip
            # slider crept back in and step 3c re-applied them AFTER the engine —
            # cementing the arms at the entry pose ("whole-body tracking does
            # nothing", measured 0.1° of arm motion under a full wrist sweep,
            # 2026-07-18 — the arm half of the earlier neck-stomp bug).
            for k in _MJ_TO_IOS.values():
                self.joint_targets.pop(k, None)
            for k in ("lift_lower", "lift_upper",
                      "neck_yaw", "neck_pitch", "neck_roll"):
                self.joint_targets.pop(k, None)
            for k in [k for k in self.joint_targets if _is_finger_key(k)]:
                self.joint_targets.pop(k, None)
        self.controller.set_mode(self.controller.MODE_NAV
                                 if self.spec["method"] == "navigation"
                                 else self.controller.MODE_MANIP)
        self.controller.update(pose, joystick, dt)

        # 3b) NAV lift jog beats a STALE lift setpoint: joint_targets persist
        # (by design — setpoints hold), but a lift_lower/upper left behind by a
        # slider or a whole-body stream would stomp the Lift actuator every
        # tick and permanently deaden the right stick's lift axis. An ACTIVE
        # ry deflection evicts the stale setpoint — most recent meaningful
        # input wins, exactly like the driver arbitration.
        if (self.spec["method"] == "navigation" and joystick is not None
                and abs(joystick[3]) > 0.04):
            self.joint_targets.pop("lift_lower", None)
            self.joint_targets.pop("lift_upper", None)

        # 3c) direct joint targets (iOS Body tab / web) win over IK for their joints
        #     — manipulate-method arm/wrist/torso targets ride the CBF projection.
        self._apply_joint_targets(sc=self._sc_filter(dt))

        mode = getattr(self.controller, "mode", "engaged")
        return RobotCommand(ctrl=self.data.ctrl.copy(), engaged=1.0,
                            clutched=(mode == "CLUTCH (hold)"), mode=mode,
                            base=(self.controller.vx, self.controller.vy,
                                  self.controller.omega))
