"""
wbc_runtime.py — runtime whole-body-control engine for the teleop bridge.

Wraps the SAME controllers the standalone scripts use so the iOS / Vision Pro
apps drive the real laws over the bridge:

  * gravity_comp : arms fully back-drivable (ComplianceController, K=0); torso/
                   lift lazy; base held. == gravity_compensation.py
  * compliance   : whole-body admittance — arms yield (impedance with adjustable
                   stiffness K) AND the mobile base admits the NET external
                   wrench over all arm bodies and glides (BaseAdmittance +
                   tip-safe swerve). == whole_body_armguide.py
  * joints       : joint impedance — hold the commanded joint targets with a
                   spring K and damping D, so pushing deflects and springs back;
                   base held.

The operator's "push anywhere on the arm" is data.xfrc_applied on the grabbed
body — set from the app's drag via apply_external(), exactly the wrench the
viewer's Ctrl+drag produces.

Entering a WBC mode neutralises the position servos on the managed joints (so
they become torque-driven and back-drivable); exit() restores them so normal
position teleop (drive / manipulation) works again.
"""
import os
import time

import numpy as np
import mujoco

from .. import config as C
from ..dynamic_models.model import UpperBodyDynamics, _dof
from ..dynamic_models.gravity import UpperBodyGravityComp
from ..dynamic_models.regions import (COMPLIANT_ARM_JOINTS, LAZY_JOINTS,
                                      LAZY_TENDON_ACTUATORS)
from ..plant import SimPlant
from .admittance.base_admittance import BaseAdmittance, external_base_wrench
from .collision_avoidance.capsule_guard import SelfCollisionGuard
from .base_controller.swerve import swerve_ik

# The torque engine runs for the manipulation teleop methods (joint / wrist /
# retarget). It is parameterised on two orthogonal axes:
#   * control type — IMPEDANCE (rigid: τ = bias + K(q_des−q) − D q̇, tracks the
#     target) or COMPLIANCE (soft: K≈0, gravity-comp + damping, back-drivable, and
#     in the whole-body region the base admits the push and glides).
#   * region — ARM (arms only) | UPPER_BODY (+ torso + lift) | WHOLE_BODY (+ base).
# In ARM the lift stays on its strong position servo (precise, stable height hold).
# In UPPER_BODY / WHOLE_BODY the lift JOINS the compliant body: its servo is
# neutralised and it is gravity-compensated (K=0 always, back-drivable) so the
# operator can also push the whole upper body up/down.
# "pd" (default since 2026-08-23) — PURE PER-JOINT PD on top of the gravity
# bias: tau_j = b_j + kp_j (q*_j - q_j) - kd_j qd_j, with kp_j = M_j w_n^2 and
# kd_j = 2 zeta M_j w_n from each joint's OWN reflected inertia. It is the same
# law "impedance" runs, minus the two things that made impedance not a PD: a
# single scalar K shared by a 0.52 kg m^2 shoulder and a 0.0031 kg m^2 wrist,
# and a low-passed reference. Operator request 2026-08-23, and the repo's own
# rule (design note: "PD gains AND clamps MUST be
# per-joint"). Derivation + measurements: experiments/exp_pd_tuning.py,
# docs/arm_pd_tuning.tex.
CONTROL_TYPES = ("pd", "impedance", "compliance")
CONTROL_REGIONS = ("arm", "upper_body", "whole_body")
_REGION_TORSO = ("upper_body", "whole_body")   # regions that add the torso
_REGION_LIFT = ("upper_body", "whole_body")    # regions that free the lift (gravity-comp)
_REGION_BASE = ("whole_body",)                 # regions that engage the base

# Any body whose name contains one of these is part of the arms/hands and can
# receive an operator push (matches whole_body_armguide.arm_bodies).
_ARM_BODY_KEYS = ("arm", "wrist", "palm", "forearm", "hand", "tower",
                  "thumb", "index", "middle", "ring", "pinky")

# iOS joint id -> MuJoCo joint name, for mapping slider targets onto the
# impedance set-point. (Wrists have no iOS slider -> they stay gravity-comp.)
# WIRE name <-> MJCF actuator, for the joints the APP addresses by name.
# NOTE: this map is ALSO iterated by facade.py (stale-target eviction, retarget
# ownership), so it is not the place to enumerate "everything the WBC manages" —
# adding the wrists here perturbed the retarget dispatch. Use `_wbc_key()` for
# the WBC's own set-point bookkeeping instead.
_IOS_TO_MJ = {
    **{f"left_arm_{i}":  f"Left Arm {i}"  for i in range(1, 8)},
    **{f"right_arm_{i}": f"Right Arm {i}" for i in range(1, 8)},
    "torso": "Torso",
}
_MJ_TO_IOS = {v: k for k, v in _IOS_TO_MJ.items()}

#: Joints the TORQUE LAW must not own, even though the region dynamic model
#: still gravity-compensates them. See WBCEngine._managed_for for the full
#: rationale: across the gateway->bridge process split these are light enough
#: (M ~ 2.5e-3 kg m^2) for the delayed effort feedback to ring them up and drive
#: them through their MJCF limit. They keep their position servo instead.
_UNMANAGED_JOINTS = ("Left Wrist", "Right Wrist")

#: KEEP THE GRAVITY FEED-FORWARD ON THE JOINTS THE TORQUE LAW DOES NOT OWN.
#:
#: Operator report, 2026-07-24: "when I switch from drive to joint or task space
#: the robot arm will sag". `plant.begin()` zeroes the WHOLE qfrc_applied vector
#: and this engine then re-applied effort only to `self.managed` — so entering a
#: torque mode silently dropped the feed-forward the kinematic path (Drive,
#: Teleop) holds on the ORCA wrists, the neck, the lift, and the torso in the
#: "arm" region. Those DoFs then settled to their servos' tau_g/kp error: measured
#: 1.61 deg at the wrists on a single Drive -> Task switch.
#:
#: The compensation is now the SAME shared layer the kinematic path uses
#: (mabel.dynamic_models.gravity.UpperBodyGravityComp), applied to exactly the
#: complement of the torque-managed set — so the union is mode-invariant: 22
#: upper-body DoFs compensated in EVERY control space. This is safe on the
#: wrists precisely because it is static g(q) with no velocity term; see that
#: module's docstring for why -D*qd was not (it rang them through their limit).
#:
#: Set MABEL_UNIFIED_GRAVCOMP=0 to restore the old split behaviour — the
#: before/after arm of controller/experiments/exp_mode_switch_sag.py.
_GRAV_UNMANAGED = os.environ.get("MABEL_UNIFIED_GRAVCOMP", "1").strip().lower() \
    not in ("0", "false", "off", "no")


def _wbc_key(mj_name):
    """Set-point key for a WBC-managed joint.

    Prefers the app-facing wire name so an operator target and the engine's own
    seed land on the SAME entry; falls back to the MJCF name normalised the same
    way ("Left Wrist" -> "left_wrist") for joints the app never addresses.

    The fallback is the whole point. `enter()` used to seed only `if ios`, so a
    managed joint missing from `_MJ_TO_IOS` was silently never given a set-point
    — and the control law's `self._q_des.get(ios, q)` then re-defaulted it to the
    MEASURED angle every tick. That is a damper, not a hold: with no fixed
    reference the joint creeps. The ORCA wrists (managed by every arm region,
    absent from the map, MJCF ref +/-0.15) drifted 7-10 deg toward 0 on the first
    Drive -> Task/Joint switch after a reset, twisting both hands.
    Guarded by controller/tests/test_wbc_seeding.py.
    """
    return _MJ_TO_IOS.get(mj_name) or mj_name.lower().replace(" ", "_")


def _lift_z(model, data):
    q = data.joint(C.LIFT_JOINT).qpos[0]
    return min(C.LIFT_RANGE[1], max(C.LIFT_RANGE[0], q / 0.303 * C.LIFT_RANGE[1]))


class WBCEngine:
    def __init__(self, model, data):
        self.model, self.data = model, data
        self.dyn = UpperBodyDynamics(model, data)
        # THE upper-body gravity layer — the same component TeleopEngine holds, so
        # Drive and Joint/Task compensate the same DoFs at the same gain and rate.
        # This engine feeds the joints its torque law does NOT own (see
        # _GRAV_UNMANAGED); `_ff_joints` is that complement, refreshed in enter().
        self.gravcomp = UpperBodyGravityComp(model)
        self._ff_joints = []
        self.plant = SimPlant(model, data)
        self.base = BaseAdmittance(ang_damping=7.0, m_static=2.5)
        self.arm_bodies = [model.body(i).name for i in range(model.nbody)
                           if model.body(i).name
                           and any(k in model.body(i).name.lower() for k in _ARM_BODY_KEYS)]
        # Snapshot the position-servo gains so exit() can restore normal teleop.
        self._gain0 = model.actuator_gainprm.copy()
        self._bias0 = model.actuator_biasprm.copy()

        self.active = False
        self.control_type = "pd"
        self.region = "arm"
        # torque-driven joints (re-set in enter); wrists excluded — _managed_for
        self.managed = [j for j in COMPLIANT_ARM_JOINTS if j not in _UNMANAGED_JOINTS]
        self.stiffness = 0.0          # impedance K [Nm/rad], operator-tunable
        # PURE PD: one bandwidth + one damping ratio, turned into per-joint gains
        # by each joint's own reflected inertia. Chosen by sweep in
        # experiments/exp_pd_tuning.py (P2); see config.PD_WN / PD_ZETA.
        self.pd_wn = float(getattr(C, "PD_WN", 60.0))
        self.pd_zeta = float(getattr(C, "PD_ZETA", 1.0))
        self._pd_gains = {}           # joint -> (kp, kd), rebuilt on enter
        self._tau_lims = {}           # joint -> |forcerange|, per joint
        self.damping = float(C.WBC_DAMP_BASE)      # base joint viscous damping (Soft/back-drive feel)
        self._damp_ratio = float(C.WBC_DAMP_RATIO) # + this·√K in IMPEDANCE only (holds ζ≈const)
        self._damp_disc_margin = float(C.WBC_DAMP_DISC_MARGIN)  # β for the per-joint discrete-damping cap
        self.lazy_damping = 12.0
        self._lift_freed = False      # is the lift servo currently neutralised (Soft only)?
        self._q_des = {}              # iOS id -> radians (the DESIRED pose, as commanded)
        self._last_cmd = {}           # last command value seen per joint (change detect)
        # Reference smoothing: the impedance law tracks _q_ref (a filtered version
        # of _q_des), NOT the raw command, so a jumpy wrist-IK target does not step
        # the spring every tick and buzz the arm. See _smooth_reference + config.
        self._q_ref = {}              # iOS id -> radians (the SMOOTHED reference fed to τ)
        self._ref_tau = float(C.WBC_REF_TAU)     # s,     LPF time constant
        self._ref_slew = float(C.WBC_REF_SLEW)   # rad/s, reference slew cap
        self.vx = self.vy = self.w = 0.0

        # Self-collision avoidance (RUNTIME-TOGGLEABLE, OFF by default). When on, a
        # torque-domain "soft wall" (guard.barrier_force) is added to the managed
        # joints' efforts so a dragged arm STOPS at the safety margin instead of
        # being pushed through itself — it brakes only the closing motion, so free
        # dragging is unaffected. Built lazily on first enable (no cost when off).
        self.self_collision = bool(C.SELF_COLLISION_ENABLE)
        self.guard = SelfCollisionGuard(model, data) if self.self_collision else None

    def _refresh_ff_joints(self):
        """The upper-body joints THIS engine must gravity-compensate: everything
        the shared layer covers, minus whatever the torque law already carries.

        The torque law's own effort already contains the bias term b[dof] for
        every `managed` joint, and for the whole lift when it is freed. Everything
        else — the ORCA wrists always, the torso outside upper_body/whole_body,
        the neck always, the lift's stages when it is NOT freed — would be left on
        a bare position servo and sag, which is precisely the mode-switch defect.

        BOTH lift stages are skipped when the lift is freed, not just the
        LIFT_JOINT the torque law names. The two stages are rigidly synced by the
        MJCF "Sync Lift" joint equality (q_upper = q_lower), so the pair is one
        DoF and the single b[LIFT_JOINT] term the float applies already holds it:
        measured stationary at qvel = 0 with 293.7 N on the lower stage and
        nothing on the upper. Feeding "Lift Upper" its own 272 N on top DOUBLE-
        compensates a coupled pair and drove the lift 11 mm UP on entry to Soft
        (exp_mode_switch_sag, 2026-07-24). The float is a separately tuned,
        deliberately back-drivable branch — this layer stays out of it."""
        skip = set(self.managed)
        if self._lift_freed:
            skip.update(("Lift Lower", "Lift Upper"))
        self._ff_joints = (self.gravcomp.joints_except(skip)
                           if _GRAV_UNMANAGED else [])

    def _managed_for(self, region):
        """Joints the torque law owns for a region (arms always; + torso higher).

        The ORCA wrists are deliberately EXCLUDED — they stay on their position
        servo, which is strong enough to hold them (they remain in
        COMPLIANT_ARM_JOINTS, so the region dynamic model still accounts for
        them; only this torque law lets them go). The cost is a small, STATIC
        droop: the servo has finite stiffness and the ORCA hand is 1.74 kg, so
        the wrist settles ~0.196 rad against its +0.15 ref (2.6 deg). See the
        note in step() for why compensating that gravity is not worth it.

        WHY (2026-07-19, operator report: "the wrist rotated 180 when I switch
        from drive to task space"). The torque law is fine in-process: driving
        the raw engine with the wrists parked ON their limit holds at every
        stiffness 0.5-30 (controller/experiments/wbc_wrist_limit.py). It is the
        DEPLOYMENT that breaks it. Under `sim:bridge` — what server/run.sh runs,
        and the mirror of the real robot's split — physics lives in another
        process; the gateway computes tau from a MIRRORED qpos/qvel and ships it
        over RPC at SimBridgePlant.CMD_HZ = 120 Hz, while the bridge steps at
        500 Hz and holds that tau across its substeps. That is delayed feedback,
        and the ORCA wrist is the one joint light enough to be destabilised by
        it: effective inertia 2.5e-3 kg m^2, ~100x lighter than an arm link, so
        the -D*qd term lands far enough out of phase to inject energy. Measured
        on the live stack: the wrist rings up to -118.6 rad/s (~19 rev/s on a
        joint whose whole range is 1.57 rad), TUNNELS through its MJCF limit in a
        single substep, and parks wrapped at ~+3.85 rad (221 deg) — an
        equilibrium it never leaves. Confirmed against the ground-truth qpos
        memmap, not just telemetry.

        Excluding them is also what the wrists want anyway: they carry no app
        slider (see the _IOS_TO_MJ note above) and are never an operator target,
        so nothing is lost by leaving them on the servo that already holds them
        correctly. Guarded by controller/tests/test_wbc_wrist_stability.py.
        """
        m = [j for j in COMPLIANT_ARM_JOINTS if j not in _UNMANAGED_JOINTS]
        if region in _REGION_TORSO:
            m += list(LAZY_JOINTS)               # Torso
        return m

    # ── lifecycle ────────────────────────────────────────────────────────────
    def enter(self, control_type, region, q_des=None):
        """Engage the torque engine with a control type + region. Re-entry with a
        changed region/type reconfigures which servos are neutralised.

        The lift is neutralised (back-drivable, gravity-comp float) ONLY in the
        upper_body/whole_body regions AND under COMPLIANCE (Soft) --- the "dance
        with the robot, push the whole upper body up/down" mode. Under IMPEDANCE
        (Stiff) it stays on its strong position servo in EVERY region, so a
        commanded lift height (joint target / stick jog) drives it precisely.
        (Neutralising it in Stiff-upper made the lift slider inert --- the servo
        ignored the command.)"""
        new_managed = self._managed_for(region)
        # free the lift only in Soft + an upper/whole region
        want_lift_free = (region in _REGION_LIFT and control_type == "compliance")
        reconfig = ((not self.active) or (new_managed != self.managed)
                    or (want_lift_free != self._lift_freed))
        self.control_type = control_type
        self.region = region
        if q_des:
            self._q_des.update(q_des)
        # Seed the set-point to the CURRENT pose so the arm holds where it is now
        # (no snap on the enter / region switch); the user's targets then track in.
        for j in new_managed:
            self._q_des[_wbc_key(j)] = float(self.plant.q(j))
        if reconfig:
            # Restore everything, then neutralise only the now-managed joints'
            # position servos (ungoverned joints keep their servos).
            self.model.actuator_gainprm[:] = self._gain0
            self.model.actuator_biasprm[:] = self._bias0
            disable = list(new_managed)
            if want_lift_free:
                disable += list(LAZY_TENDON_ACTUATORS)   # the "Lift" actuator (Soft only)
            elif self._lift_freed:
                # Re-stiffening after a Soft float: ctrl is stale — hold the
                # measured height (no servo snap). The "Lift" actuator drives a
                # TENDON, so ctrl targets the tendon length = the STAGE SUM
                # (Sync Lift) — seeding it with the single LIFT_JOINT qpos (the
                # half value) slammed a raised lift to half height on every
                # switch. Same convention bug as facade._actuator_q /
                # TeleopController hold. When the servo was NEVER freed (e.g.
                # Drive → joint·impedance) ctrl already realizes the held height
                # including gravity sag — leave it alone (reseeding from the
                # sagged qpos re-settled the lift ~3 cm on every mode switch).
                for a in LAZY_TENDON_ACTUATORS:
                    try:
                        q = self.plant.q(C.LIFT_JOINT) + self.plant.q("Lift Upper")
                        aid = self.model.actuator(a).id
                        lo, hi = self.model.actuator_ctrlrange[aid]
                        self.data.actuator(a).ctrl[0] = float(min(max(q, lo), hi))
                    except (KeyError, Exception):
                        pass
            for a in disable:
                try:
                    self.plant.disable_servo(a)
                except KeyError:
                    pass
            self._lift_freed = want_lift_free
            self.vx = self.vy = self.w = 0.0
        self.managed = new_managed
        self._refresh_ff_joints()
        self.active = True

    def exit(self):
        """Restore position servos and clear applied forces."""
        if not self.active:
            return
        self.model.actuator_gainprm[:] = self._gain0
        self.model.actuator_biasprm[:] = self._bias0
        self.data.qfrc_applied[:] = 0.0
        self.data.xfrc_applied[:] = 0.0
        self._lift_freed = False
        self.active = False
        # Leaving the torque engine FORGETS its set-points. `enter()` re-seeds
        # every managed joint from the measured pose anyway, so nothing is lost —
        # but keeping them meant a set-point held here (the ORCA wrists) survived
        # into RETARGET, where the retargeter owns the wrists, and perturbed the
        # first follow step at engage past the per-tick command budget
        # (measured 0.193 rad vs the 0.15 bound, controller/tests/
        # test_real_dispatch_chain.py). The WBC must leave no residue.
        self._q_des.clear()
        self._q_ref.clear()
        self._last_cmd.clear()

    # ── self-collision toggle ─────────────────────────────────────────────────
    def set_self_collision(self, on):
        """Enable/disable the torque-domain self-collision "soft wall" at runtime.
        Built on first enable (a disabled session pays no construction cost)."""
        on = bool(on)
        if on and self.guard is None:
            self.guard = SelfCollisionGuard(self.model, self.data)
        self.self_collision = on
        return self.self_collision

    def toggle_self_collision(self):
        return self.set_self_collision(not self.self_collision)

    # ── inputs ───────────────────────────────────────────────────────────────
    def set_stiffness(self, k):
        self.stiffness = max(0.0, float(k))

    def set_pd(self, wn=None, zeta=None):
        """Retune the pure-PD law. Gains are DERIVED, never set per joint by hand.

        The two numbers are a closed-loop bandwidth and a damping ratio; each
        joint's kp and kd follow from its own inertia. That is what keeps the
        tracking lag (kd/kp = 2*zeta/w_n) identical across the arm while the
        damping ratio is correct for every joint individually — the property one
        shared scalar K can never have.
        """
        if wn is not None:
            self.pd_wn = max(1.0, float(wn))
        if zeta is not None:
            self.pd_zeta = max(0.1, float(zeta))
        self._pd_gains = {}
        return self.pd_wn, self.pd_zeta

    def _tau_limit(self, joint):
        """The joint's own actuator |forcerange| — never a uniform clamp.

        design note: a uniform 40 N.m clamp stick-slips the
        +/-7 N.m distal joints after a hard shove and leaves a standing limit
        cycle the wrench estimator reads as a phantom contact.
        """
        lim = self._tau_lims.get(joint)
        if lim is None:
            try:
                fr = self.model.actuator_forcerange[self.model.actuator(joint).id]
                lim = float(max(abs(fr[0]), abs(fr[1]))) or 40.0
            except (KeyError, Exception):
                lim = 40.0
            self._tau_lims[joint] = lim
        return lim

    def _pd_gain(self, joint):
        """(kp, kd) for one joint from its reflected inertia, cached.

        M_jj is read as 1/dof_invweight0 — the model's own diagonal inverse
        inertia, the same quantity the discrete-damping cap already uses. It is a
        property of the MODEL, so it is measured, not tuned.
        """
        g = self._pd_gains.get(joint)
        if g is None:
            dof = _dof(self.model, joint)
            m_jj = 1.0 / max(float(self.model.dof_invweight0[dof]), 1e-12)
            g = (m_jj * self.pd_wn ** 2, 2.0 * self.pd_zeta * m_jj * self.pd_wn)
            self._pd_gains[joint] = g
        return g

    def set_targets(self, q_des):
        """Update the desired pose ONLY for joints that received a NEW command.
        The bridge re-sends the whole accumulated command dict every tick, so a
        blind update would re-assert stale targets and clobber a desired pose that
        compliance has since moved (→ the arm snaps back). Tracking the last value
        per joint means only a genuine new teleop/drag input changes q_des."""
        if not q_des:
            return
        for k, v in q_des.items():
            if self._last_cmd.get(k) != v:
                self._last_cmd[k] = v
                self._q_des[k] = v

    def reset_targets(self):
        """Send every desired pose to the CURRENT pose (called after a sim reset to
        home) and forget past commands, so the controller holds home cleanly with
        no stale pull and no command re-asserting an old target."""
        self._q_des.clear()
        self._last_cmd.clear()
        self._q_ref.clear()          # re-seed the smoothed reference to home next tick
        for j in self.managed:
            k = _wbc_key(j)
            self._q_des[k] = float(self.plant.q(j))
            self._q_ref[k] = float(self.plant.q(j))

    def apply_external(self, forces):
        """Add the operator's app push {body_name: [fx,fy,fz]} (world frame) ON TOP of
        whatever external wrench is already in xfrc_applied — namely the desktop-viewer
        Ctrl+drag the bridge writes each substep BEFORE the control law. Adding (not
        overwriting) is what lets WHOLE-BODY compliance glide under a viewer drag: the
        base-wrench estimate (external_base_wrench, read in step()) then sees BOTH the
        app push and the viewer drag. It never accumulates because the bridge resets
        xfrc to the viewer perturbation every substep. (The standalone compliance.py
        reference never calls this — the viewer owns xfrc directly there.)"""
        if not forces:
            return
        for name, f in forces.items():
            try:
                bid = self.model.body(name).id
            except KeyError:
                continue
            self.data.xfrc_applied[bid, 0:3] += np.asarray(f[:3], dtype=float)

    def _smooth_reference(self, prev, target, dt):
        """Ease a (possibly jumpy) commanded joint target into the reference the
        impedance law actually springs to. Two dt-correct stages:
          1. first-order LOW-PASS (time constant self._ref_tau) — attenuates the
             high-frequency wiggle a multi-joint IK setpoint carries frame to frame;
          2. SLEW cap (self._ref_slew rad/s) — bounds how fast the reference may
             move, turning a large jump into a deliberate ramp ("slower but surely")
             instead of a spring-torque impulse.
        A slow, smooth command sits below the corner and under the cap, so it passes
        through with only a few tens of ms of lag. Returns `target` verbatim when
        smoothing is effectively disabled (tau<=0 and a huge slew) or dt is unusable."""
        if dt is None or dt <= 0.0:
            return target
        # 1) low-pass toward the command
        if self._ref_tau > 0.0:
            alpha = dt / (self._ref_tau + dt)
            lp = prev + alpha * (target - prev)
        else:
            lp = target
        # 2) slew-limit the resulting step
        max_step = self._ref_slew * dt
        step = lp - prev
        if step > max_step:
            step = max_step
        elif step < -max_step:
            step = -max_step
        return prev + step

    # ── per-tick control law ────────────────────────────────────────────────
    def step(self, dt, wrench=None):
        """Advance one tick. `wrench` optionally overrides the base-driving planar
        wrench (fx, fy, mz): pass None (default) to read it from the simulator
        (data.xfrc_applied, the viewer drag), or supply a wrench estimated from
        joint-torque feedback so the SAME engine runs the hardware path."""
        m, d = self.model, self.data
        b = self.dyn.bias()
        # IMPEDANCE springs to the target (K>0); COMPLIANCE is back-drivable (K=0,
        # gravity-comp + damping). Both cancel gravity via the bias term.
        pure_pd = self.control_type == "pd"
        K = self.stiffness if self.control_type == "impedance" else 0.0
        # Stiffness-scaled damping: hold the damping ratio ~constant as K rises so a
        # stiff arm settles cleanly instead of ringing/overshooting the wrist target.
        # Compliance (K=0) keeps exactly the base viscous damping (back-drive feel).
        D0 = self.damping + (self._damp_ratio * np.sqrt(K) if K > 0.0 else 0.0)

        # Self-collision "soft wall": a generalized braking force that opposes only
        # motion closing a monitored gap (zero elsewhere → free dragging unaffected).
        tau_sc = (self.guard.barrier_force(d.qvel)
                  if (self.self_collision and self.guard is not None)
                  else None)

        self.plant.begin()
        for j in self.managed:
            dof = _dof(m, j)
            q = self.plant.q(j)
            qd = self.plant.qd(j)
            ios = _wbc_key(j)
            qdes_cmd = self._q_des.get(ios, q)                 # default: where it is
            if pure_pd:
                # PURE PD. No reference filter: the smoothing below exists to
                # stop a scalar spring from stepping on a jumpy IK target, and a
                # per-joint PD does not need it — the retarget side already
                # rate-limits its own output (ARM_TRUST_RATE), so filtering here
                # only adds lag the operator feels.
                kp, kd = self._pd_gain(j)
                self._q_ref[ios] = qdes_cmd
                tau = b[dof] + kp * (qdes_cmd - q) - kd * qd
                if tau_sc is not None:
                    tau += tau_sc[dof]
                lim = self._tau_limit(j)
                if lim is not None:
                    tau = float(np.clip(tau, -lim, lim))
                self.plant.apply_effort(j, tau)
                continue
            if K > 0.0:
                # IMPEDANCE: track a SMOOTHED reference so a jumpy wrist-IK target
                # doesn't step the spring every tick. Seeded to the joint the first
                # time it is seen (and glued to it in compliance below), so engaging
                # impedance never springs from a stale reference.
                qdes = self._smooth_reference(self._q_ref.get(ios, q), qdes_cmd, dt)
                self._q_ref[ios] = qdes
            else:
                # COMPLIANCE (K=0): q_des is unused by τ; keep the reference glued to
                # the arm so a later switch to impedance holds where it is (no jump).
                qdes = qdes_cmd
                self._q_ref[ios] = q
            # PER-JOINT discrete-stability cap on the viscous damping. The term −D·qd is
            # integrated with a zero-order hold each control step; if D exceeds the
            # explicit limit 2·M/dt the discrete update overshoots the velocity sign and
            # the joint rings UP instead of settling. The ORCA wrist (effective inertia
            # M≈2.5e−3 kg·m²) is ~100× lighter than the arm links, so the stiffness-
            # scaled damping D0 tuned for the arm blows the wrist up at K≳15 (measured:
            # qacc 1.3e4 rad/s² at K=30, dt=4 ms). Cap D at β·M/dt with β≤1 (deadbeat at
            # β=1, well under the 2·M/dt hard limit); M per DOF from the model's diagonal
            # inverse inertia (dof_invweight0). No effect on the heavy arm/torso joints,
            # whose limit (≥40) dwarfs D0. Protects BOTH the sim and the real CAN path.
            D = D0
            if dt and dt > 0.0:
                m_jj = 1.0 / max(float(m.dof_invweight0[dof]), 1e-12)
                D = min(D0, self._damp_disc_margin * m_jj / dt)
            tau = b[dof] + K * (qdes - q) - D * qd
            if tau_sc is not None:
                tau += tau_sc[dof]
            self.plant.apply_effort(j, tau)
        # GRAVITY FEED-FORWARD ON EVERYTHING THE TORQUE LAW DOES NOT OWN — the
        # ORCA wrists, the neck, the lift's stages, the torso outside
        # upper_body/whole_body. Without this, entering a torque mode dropped
        # their compensation and they sagged to tau_g/kp: the "arm sags when I
        # switch from drive to joint/task space" report (2026-07-24).
        #
        # This resolves the dilemma the old comment here described. It wanted the
        # wrists compensated but had only two options, both bad: the full bias
        # b[dof] reintroduces Coriolis (velocity feedback on the one joint too
        # light to tolerate it — the ring-up that excluded them from the torque
        # law), and UpperBodyDynamics.gravity() zeroes/restores data.qvel around
        # its mj_rne, a mid-tick mutation of shared state measured at a 0.335 rad
        # single-tick jump on left_arm_5. The shared layer has NEITHER problem: it
        # evaluates on private scratch state at qvel = 0, so it is pure g(q) —
        # nothing shared is touched, and there is no velocity term to ring.
        if self._ff_joints:
            tau_g = self.gravcomp.tau(d, time.time())
            gain = self.gravcomp.gain
            for j in self._ff_joints:
                self.plant.apply_effort(j, gain * tau_g[self.gravcomp.dof(j)])
        # Lift: freed ONLY in Soft + upper/whole (see enter) — then it is part of the
        # compliant body: pure gravity-comp + heavy damping on its DOF (K=0, no spring)
        # so it floats and is back-drivable. Otherwise it stays on its position servo
        # (commandable) and is left untouched here. (Tendon-cascaded: drive the "Lift
        # Lower" joint DOF.)
        if self._lift_freed:
            lj = C.LIFT_JOINT
            self.plant.apply_effort(lj, b[_dof(m, lj)] - self.lazy_damping * self.plant.qd(lj))
        self.plant.commit()

        # COMPLIANCE is the human moving the arm: continuously track the desired
        # pose to where the arm ACTUALLY is. This is the handoff — when the operator
        # later switches to IMPEDANCE, q_des already equals the dragged pose, so it
        # holds there instead of springing back to the last commanded target.
        if self.control_type == "compliance":
            for j in self.managed:
                self._q_des[_wbc_key(j)] = float(self.plant.q(j))

        # Base: only the WHOLE-BODY + COMPLIANCE pairing lets the chassis admit the
        # net wrench and glide; every other config holds the base still.
        if self.region in _REGION_BASE and self.control_type == "compliance":
            fx, fy, mz = wrench if wrench is not None else external_base_wrench(m, d, self.arm_bodies)
            tvx, tvy, tw = self.base.step(fx, fy, mz, dt, gate=1.0)
            self.vx, self.vy, self.w = C.safe_base_step(
                self.vx, self.vy, self.w, tvx, tvy, tw, dt, _lift_z(m, d))
            swerve_ik(self.vx, self.vy, self.w, d)
        else:
            self.vx = self.vy = self.w = 0.0
            swerve_ik(0.0, 0.0, 0.0, d)
