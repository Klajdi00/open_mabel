"""Arm self-collision avoidance: a velocity-QP projection of the IK targets
onto the barrier-safe set, reusing the shipped capsule guard."""
import numpy as np
import mujoco

from .hyperparams import *
from .deps import SelfCollisionGuard, solve_qp


class ArmCollisionProjector:
    """Projects the natural arm-IK targets onto the self-collision-safe set so a hand
    can never be driven through the torso / head / other arm, while tracking the
    desired pose as closely as the barrier allows (the hand SLIDES along the obstacle
    rather than stopping). Reuses the shipped capsule guard (mabel.self_collision):
    a one-step velocity QP  min ½‖q̇ − q̇_des‖²  s.t.  CBF rows G q̇ ≤ h + velocity/
    limit box. All FK-based (capsules ride the bodies), so it works on the real robot.

    Modular: build it with the live (model, data) + the two ArmRetargeters and call
    project() each frame; pass None / omit to disable. Tunables in config.SC_*."""

    def __init__(self, model, data, arm):
        self.model = model
        self.data = data
        # THE BARRIER IS EVALUATED AT THE COMMANDED POSE, not the measured one.
        # The QP's decision variable is the COMMAND, so a barrier evaluated on
        # measured geometry answers a different question than the one being
        # asked: while the servo lags, the measured gap is larger than the gap
        # the command implies, the CBF reads that slack as room to keep closing,
        # and the arm penetrates as it catches up. Measured: the palm settled
        # 2.4 mm INSIDE the torso capsule on 0.14 % of frames with every
        # constraint reporting itself satisfied.
        #
        # The scratch carries the LIVE state for everything the projector does
        # not own (base, lift, torso, neck, the fingers) and the COMMANDED values
        # for the joints it does — so the guard still sees the real world, just
        # with these arms where they are being told to go.
        self.scratch = mujoco.MjData(model)
        self.guard = SelfCollisionGuard(model, self.scratch,
                                        influence=SELFCOL_INFLUENCE,
                                        alpha=SELFCOL_ALPHA)
        # Only the hand-side capsules ride the ORCA wrist, so only they need the
        # following-error pad; the forearm and upper arm track well and keep the
        # bare margin. Padding everything cost 18 mm of standoff on links that
        # never needed it.
        self.guard.lag_capsules = frozenset(
            {"l_hand", "r_hand", "l_wristmt", "r_wristmt"})
        # THE ORCA WRIST IS IN THE QP (2026-08-23). It was not, and that was a
        # hole in the guarantee rather than an optimisation: the hand capsule
        # spans wrist -> palm and therefore ROTATES with the ORCA wrist, so a
        # wrist-driven approach to the torso had no column in the QP and the
        # barrier could not push back on it at all. Measured over the front
        # workspace, closed loop: the hand reached 27 mm INSIDE the torso capsule
        # while every constrained pair held its margin exactly. Per side the
        # block is [7 arm joints, ORCA wrist], so the sides stay contiguous.
        jids = [j for s in ("right", "left")
                for j in list(arm[s].jids) + [arm[s].orca_wrist_jid]]
        self.dofs = np.array([model.jnt_dofadr[j] for j in jids])
        self.qadr = np.array([model.jnt_qposadr[j] for j in jids])
        self.lo = model.jnt_range[jids, 0].copy()
        self.hi = model.jnt_range[jids, 1].copy()
        self.n = len(jids)
        self.n_arm = len(arm["right"].jids)                # arm joints per side
        self.split = self.n_arm + 1                        # right block width
        self.min_dist = np.inf
        self.fallbacks = 0
        # COMMAND-domain integration state: the last command this projector
        # returned. The QP used to integrate from MEASURED qpos, so the
        # activation edge JUMPED the command from the tracked target down to
        # q_measured ± vmax·dt — a step the size of the servo lag (measured
        # 0.33 rad on an arm actuator when the bimanual targets skim the
        # influence boundary and the guard flaps). Integrating from the last
        # COMMAND keeps the published targets continuous through both edges;
        # the barrier rows still come from the MEASURED capsule geometry.
        self.q_cmd = None
        self.arm = arm
        self.enable_gripper_contact = False
        self.jaw_measured = {}
        self.jaw_commanded = {}
        self.active_sides = None
        self.arm_feedback_required = False
        self.arm_measured = {}
        self.arm_feedback_valid = False
        self.arm_feedback_reason = ""
        self.arm_feedback_age_s = None
        self.arm_feedback_compute_s = 0.0
        self.torso_measured = None
        self.actual = mujoco.MjData(model)
        self.physical_feedback_hold = False
        self.contact_guard_reason = ""
        self.closing_lead = 0.0
        self.closing_lead_limit = .004
        self._command_contact_margins = {}
        self._measured_contact_margins = {}
        self.handoff_alpha = 0.0
        self.handoff_separation = None
        self._ee_signature = None
        self._orca_caps = dict(self.guard.caps)
        self._orca_pairs = list(self.guard.pairs)
        self._orca_margins = dict(self.guard._pair_margin)
        self._orca_lag = self.guard.lag_capsules
        self._tip_points = {}
        self._finger_pairs = []
        self._gripper_frames = {}
        self._jaw_envelope = {}

    def _end_effectors(self):
        """Replace phantom ORCA envelopes only on arms carrying a gripper."""
        signature = tuple(self.arm[s].ee if self.enable_gripper_contact else "orca"
                          for s in ("right", "left"))
        if signature != self._ee_signature:
            from .end_effector import distal_tip, gripper_frame, gripper_collision_shapes
            from ..whole_body_controller.collision_avoidance.capsule_guard import Capsule
            guard = self.guard
            caps = dict(self._orca_caps)
            names = {s: [p + "_hand", p + "_wristmt"]
                     for s, p in (("right", "r"), ("left", "l"))}
            self._tip_points = {}
            self._gripper_frames = {}
            self._jaw_envelope = {}
            for side, p in (("right", "r"), ("left", "l")):
                arm = self.arm[side]
                if signature[("right", "left").index(side)] != "gripper":
                    continue
                T = gripper_frame(self.model, side, arm.palm_bid, arm.R_rob)
                self._gripper_frames[side] = T
                self._jaw_envelope[side] = None
                body, tip = distal_tip(self.model, side, arm.palm_bid, arm.R_rob)
                self._tip_points[side] = (body, tip)
                names[side] = []
                for kind, (a, b, radius) in gripper_collision_shapes().items():
                    name = p + "_" + {"housing": "wristmt", "palm": "hand"}.get(kind, "grip_" + kind)
                    caps[name] = Capsule(body, T[:3, 3] + T[:3, :3] @ np.asarray(a),
                                         T[:3, 3] + T[:3, :3] @ np.asarray(b), radius, name)
                    names[side].append(name)
            guard.caps = caps
            pairs = [(a.name, b.name) for a, b in self._orca_pairs]
            seen = {tuple(sorted(pair)) for pair in pairs}
            for side, other, op in (("right", "left", "l"), ("left", "right", "r")):
                if signature[("right", "left").index(side)] != "gripper":
                    continue
                for name in names[side]:
                    for target in ["torso", "head", op + "_fore"] + names[other]:
                        key = tuple(sorted((name, target)))
                        if key not in seen:
                            seen.add(key)
                            pairs.append((name, target))
            guard.pairs = [(caps[a], caps[b]) for a, b in pairs]
            guard.lag_capsules = self._orca_lag | frozenset(
                n for side in names if signature[("right", "left").index(side)] == "gripper" for n in names[side])
            self._finger_pairs = [tuple(sorted((a.name, b.name))) for a, b in guard.pairs
                                  if "grip_finger" in a.name and "grip_finger" in b.name]
            self._ee_signature = signature
        # Only intentional distal-finger contact gets a smaller pad. Housings,
        # forearms, torso and head keep their original margins and lag pads.
        alpha = (float(np.clip(self.handoff_alpha, 0., 1.))
                 if self.enable_gripper_contact and self.handoff_separation is not None else 0.)
        self.guard._pair_margin = dict(self._orca_margins)
        self.guard.pair_lag_scale = {}
        for pair in self._finger_pairs:
            self.guard.pair_lag_scale[pair] = 1.0 - .8 * alpha
            if all(name.endswith("finger_tip") for name in pair):
                self.guard._pair_margin[pair] = self.guard.d_safe * (1.0 - alpha) + .001 * alpha
        self.guard.endpoint_pairs = (frozenset(self._finger_pairs)
                                     if alpha > 1e-6 and len(self._tip_points) == 2 else frozenset())
        if self._tip_points:
            from .end_effector import GRIPPER, gripper_collision_shapes
            for side, p in (("right", "r"), ("left", "l")):
                if side not in self._tip_points:
                    continue
                values = [source.get(side) for source in (self.jaw_measured, self.jaw_commanded)]
                known = all(v is not None and np.isfinite(v) for v in values)
                opening = min(GRIPPER["finger_range"][1], max(values) + .002) if known else None
                if opening == self._jaw_envelope[side]:
                    continue
                self._jaw_envelope[side] = opening
                T = self._gripper_frames[side]
                for kind, (a, b, _) in gripper_collision_shapes(opening).items():
                    name = p + "_" + {"housing": "wristmt", "palm": "hand"}.get(kind, "grip_" + kind)
                    cap = self.guard.caps[name]
                    cap.p0 = T[:3, 3] + T[:3, :3] @ np.asarray(a)
                    cap.p1 = T[:3, 3] + T[:3, :3] @ np.asarray(b)

    def _contact_objective(self, qd_des, qd_lo, qd_hi, dt=1.0 / 60.0):
        """Relative physical-tip precision, constrained by the SAME collision QP."""
        alpha = float(np.clip(self.handoff_alpha, 0., 1.)) if self.enable_gripper_contact else 0.
        H, g = np.eye(self.n), -qd_des
        if alpha <= 1e-6 or self.handoff_separation is None or len(self._tip_points) != 2:
            return H, g
        points, jac = {}, {}
        for side, (body, local) in self._tip_points.items():
            points[side] = self.scratch.xpos[body] + self.scratch.xmat[body].reshape(3, 3) @ local
            jac[side] = self.guard._point_jac(points[side], body)[:, self.dofs]
        J = jac["right"] - jac["left"]
        error = points["right"] - points["left"] - np.asarray(self.handoff_separation, float)
        velocity = -3.0 * error
        speed = float(np.linalg.norm(velocity))
        cap = .03 + .05 * float(np.clip((np.linalg.norm(error) - .03) / .07, 0., 1.))
        if speed > cap:
            velocity *= cap / speed
        # A remote IK optimum may be radians away: dividing that by one frame
        # must not give it arbitrary priority over a feasible contact velocity.
        feasible = np.clip(qd_des, qd_lo, qd_hi)
        g = -((1.0 - alpha) * qd_des + alpha * feasible)
        weight = 5000.0 * alpha * alpha
        H += weight * J.T @ J
        g -= weight * J.T @ velocity
        # Hard clipping above has flat posture plateaus while a distant IK
        # optimum saturates. A small bounded smooth pull toward that SAME
        # optimum anchors its otherwise weakly determined orientation. Without
        # it, actual-gap feedback can amplify pose roundoff in this nullspace.
        error_q = qd_des * dt
        vmax = np.maximum(np.maximum(np.abs(qd_lo), np.abs(qd_hi)), 1e-6)
        posture = error_q / np.sqrt(1.0 + (error_q / vmax) ** 2)
        H += alpha * np.eye(self.n)
        g -= alpha * posture
        return H, g

    def _contact_step(self, step):
        """Check the finite step where near-parallel finger cores change closest points.

        The velocity barrier is a linearization. At millimetre contact clearances
        its closest segment points can switch within one otherwise valid step.
        Backtracking only shortens that same step; it never jumps the target or
        relaxes a physical margin to make the contact task feasible.
        """
        from ..whole_body_controller.collision_avoidance.capsule_guard import _seg_seg
        pairs = [(a, b) for a, b in self.guard.pairs
                 if "grip_finger" in a.name and "grip_finger" in b.name]

        def gaps():
            result = []
            for a, b in pairs:
                aa, ab = a.world(self.scratch)
                ba, bb = b.world(self.scratch)
                u, v = _seg_seg(aa, ab, ba, bb)
                result.append(float(np.linalg.norm(u - v) - a.radius - b.radius))
            return np.asarray(result)

        minimum = np.minimum(gaps(), [self._command_contact_margins.get(
            tuple(sorted((a.name, b.name))), self.guard._margin_for(a, b)) for a, b in pairs])
        scale = 1.0
        for _ in range(10):
            candidate = self.q_cmd + scale * step
            self.scratch.qpos[self.qadr] = candidate
            mujoco.mj_kinematics(self.model, self.scratch)
            # Do not let repeated held-physical updates spend a per-frame
            # nanometre tolerance as an accumulating command closure lead.
            if np.all(gaps() >= minimum - 1e-12):
                return candidate
            scale *= .5
        return self.q_cmd.copy()

    def _measured_pose(self):
        """Separate actual-arm FK; the virtual Lab state remains the planning state."""
        self.physical_feedback_hold = False
        self.contact_guard_reason = ""
        if not (self.enable_gripper_contact and self._tip_points and self.arm_feedback_required):
            return self.data
        values = []
        for side in ("right", "left"):
            for i in range(1, self.n_arm + 1):
                value = self.arm_measured.get(f"{side}_arm_{i}")
                if value is None or not np.isfinite(value):
                    self.physical_feedback_hold = True
                    self.contact_guard_reason = self.arm_feedback_reason or "physical arm feedback incomplete/non-finite"
                    return None
                values.append(float(value))
        age = self.arm_feedback_age_s
        reserve = self.arm_feedback_compute_s
        age_valid = (age is not None and np.isfinite(age) and np.isfinite(reserve)
                     and 0. <= age and 0. <= reserve and age + reserve <= .1)
        if not self.arm_feedback_valid or not age_valid or self.torso_measured is None or not np.isfinite(self.torso_measured):
            self.physical_feedback_hold = True
            self.contact_guard_reason = self.arm_feedback_reason or "physical arm/torso feedback missing or stale"
            return None
        self.actual.qpos[:] = self.data.qpos
        for index, side in enumerate(("right", "left")):
            self.actual.qpos[self.arm[side].qadr] = values[index * self.n_arm:(index + 1) * self.n_arm]
        joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "Torso")
        self.actual.qpos[self.model.jnt_qposadr[joint]] = float(self.torso_measured)
        mujoco.mj_kinematics(self.model, self.actual)
        mujoco.mj_comPos(self.model, self.actual)
        return self.actual

    def _contact_lead_margins(self, measured):
        """Bound command-side closure ahead of fresh physical finger geometry.

        Share the measured-side pad between age reserve and command lead.
        These are additional command barriers, never smaller physical margins.
        Opening/retreat stays available when the physical arms lag the planner.
        """
        self._command_contact_margins = {}
        self._measured_contact_margins = {}
        self.closing_lead = 0.0
        self.closing_lead_limit = .004
        if measured is self.data or self.handoff_separation is None or self.handoff_alpha <= 1e-6:
            return
        from ..whole_body_controller.collision_avoidance.capsule_guard import _seg_seg
        for ca, cb in self.guard.pairs:
            key = tuple(sorted((ca.name, cb.name)))
            if key not in self._finger_pairs:
                continue
            gaps = []
            for data in (self.scratch, measured):
                a0, a1 = ca.world(data)
                b0, b1 = cb.world(data)
                a, b = _seg_seg(a0, a1, b0, b1)
                gaps.append(float(np.linalg.norm(a - b) - ca.radius - cb.radius))
            pad = max(0., SELFCOL_LAG_PAD * self.guard.pair_lag_scale.get(key, 1.0))
            # One shared uncertainty budget: age consumes following allowance,
            # rather than being added on top of a separate 4 mm command lead.
            # Keep a 2 mm controller/transient guardband. This is conservative
            # engineering reserve, not a validated bound on real PD overshoot.
            age_margin = .03 * float(self.arm_feedback_age_s + self.arm_feedback_compute_s)
            reserve = .002
            measured_pad = max(pad, age_margin + reserve)
            allowance = max(0., min(.004, measured_pad - age_margin - reserve))
            self.closing_lead_limit = min(self.closing_lead_limit, allowance)
            self.closing_lead = max(self.closing_lead, gaps[1] - gaps[0])
            self._command_contact_margins[key] = max(self.guard._margin_for(ca, cb), gaps[1] - allowance)
            self._measured_contact_margins[key] = self.guard._margin_for(ca, cb) + measured_pad - pad
        if self.closing_lead > self.closing_lead_limit:
            self.contact_guard_reason = (f"waiting for measured grippers: closing lead {1000 * self.closing_lead:.1f} mm "
                                         f"exceeds {1000 * self.closing_lead_limit:g} mm")

    def project(self, q_des_r, q_des_l, dt, wrist_r=None, wrist_l=None):
        """q_des_r/l: the per-arm desired joint targets (right then left), and the
        matching ORCA wrist targets. Returns
        (q_safe_r, q_safe_l, wrist_safe_r, wrist_safe_l).

        The ORCA wrist is projected alongside the arm because the hand capsule
        turns with it — leaving it out let a wrist-driven approach walk the palm
        into the torso with no constraint able to stop it. Passing None for a
        wrist holds that joint at its current measured value.

        FAST PATH: when no monitored pair is within the influence
        distance there is no barrier to enforce, so the targets pass straight through
        (with only a bounded RECOVERY slew when the last projected command still lags
        the raw target — an active→free edge must not step either). The QP runs ONLY
        in the rare frames a hand is actually near the body/other arm.
        dt = VR frame interval [s] (so the guard's alpha [1/s] is dimensionally right)."""
        self._end_effectors()
        dt = max(float(dt), 1e-3)
        # The COMMAND step per call is bounded load-invariantly: vmax × a CLAMPED
        # dt. Using the raw frame dt let a loaded host (frame gaps > 33 ms) grow
        # the per-tick command step past the real-dispatch 0.15 rad bound; the
        # CBF rows below keep the TRUE dt (alpha·dt is the barrier's physics).
        dt_step = min(dt, 1.0 / 30.0)

        def _blk(q_arm, w, side):
            q_arm = np.asarray(q_arm, float).ravel()
            if w is None:
                w = float(self.data.qpos[self.qadr[self.split - 1
                                                  if side == "r" else 2 * self.split - 1]])
            return np.concatenate([q_arm[:self.n_arm], [float(w)]])

        q_des = np.concatenate([_blk(q_des_r, wrist_r, "r"),
                                _blk(q_des_l, wrist_l, "l")])
        if self.q_cmd is None:
            self.q_cmd = self.data.qpos[self.qadr].copy()
        measured = self._measured_pose()
        if measured is None:
            self._command_contact_margins = {}
            return self._unpack()
        inactive = []
        if self.enable_gripper_contact and self._tip_points and self.active_sides is not None:
            for index, side in enumerate(("right", "left")):
                if side not in self.active_sides:
                    block = np.arange(index * self.split, (index + 1) * self.split)
                    inactive.extend(block)
                    q_des[block] = self.q_cmd[block]
        # place the scratch at "the world as it is, these arms where they are told to be"
        self.scratch.qpos[:] = self.data.qpos
        self.scratch.qpos[self.qadr] = self.q_cmd
        mujoco.mj_kinematics(self.model, self.scratch)
        mujoco.mj_comPos(self.model, self.scratch)          # cdof for mj_jac
        self.min_dist = np.inf
        # CONSTRAIN AGAINST BOTH POSES — the command AND the measurement.
        #
        # Neither alone is enough, and each fails in the opposite direction:
        #
        #   measured only  the QP's variable is the COMMAND, so while the servo
        #                  lags, the measured gap is bigger than the gap the
        #                  command implies; the CBF reads that as room and the
        #                  arm penetrates as it catches up (measured 27 mm).
        #   command only   the command is then provably safe, but the ARM is not
        #                  the command: on a fast transit it lags by up to 1 rad
        #                  and overshoots the safe set-point (measured 18.5 mm).
        #
        # Stacking both row sets is valid — they are both linear constraints on
        # the same q̇, just linearised at different points — and it is exactly
        # the right semantics: while the arm trails, the command rows bind and
        # keep the target honest; when the arm overshoots, the measured rows bind
        # and the command retreats to pull it back out. The alternative was to
        # slow every approach (SC_STOP_DECEL 2.0 -> 0.25), which cleared the
        # penetration by 0.4 mm and cost 3.7 deg of palm accuracy — paying with
        # the quantity the operator actually asked for.
        #
        # braking=True additionally caps the approach at sqrt(2*a*slack), the
        # fastest speed from which the link can still stop in the gap that remains.
        # The MEASURED rows carry a PAD (the guard's own mechanism for exactly
        # this): the commanded velocity reaches the metal through a lagging
        # servo, so the physical link overshoots the commanded stop by its
        # following error and the measured rows must brake that much earlier.
        # It matters most for the HAND, whose capsule turns with the ORCA wrist —
        # a ~2.9 Hz joint (kp = 1 in the MJCF, and correctly so: only 0.217 kg
        # hangs off it and it sags 1.9 deg, while STIFFENING it measurably breaks
        # external-contact force estimation). A joint that soft cannot be
        # commanded out of the way quickly, so the barrier leaves room for it.
        self._contact_lead_margins(measured)
        base_margins, influence = self.guard._pair_margin, self.guard.d_inf
        if self._command_contact_margins:
            self.guard._pair_margin = {**base_margins, **self._command_contact_margins}
            self.guard.d_inf = max(influence, max(self._command_contact_margins.values()) + .001)
        try:
            Gc, hc = self.guard.rows(self.dofs, braking=True)   # at the COMMAND pose
        finally:
            self.guard._pair_margin, self.guard.d_inf = base_margins, influence
        self.guard.data = measured
        if self._measured_contact_margins:
            self.guard._pair_margin = {**base_margins, **self._measured_contact_margins}
        try:
            Gm, hm = self.guard.rows(self.dofs, braking=True,
                                     margin_pad=SELFCOL_LAG_PAD)  # at the MEASURED pose
        finally:
            self.guard._pair_margin = base_margins
            self.guard.data = self.scratch
        self.min_dist = min(self.min_dist, self.guard.min_dist)
        G = np.vstack([Gc, Gm]) if len(Gc) and len(Gm) else (Gc if len(Gc) else Gm)
        h = np.concatenate([hc, hm]) if len(hc) and len(hm) else (hc if len(hc) else hm)
        contact = (self.enable_gripper_contact and self.handoff_separation is not None and len(self._tip_points) == 2
                   and self.handoff_alpha > 1e-6)
        alpha = float(np.clip(self.handoff_alpha, 0., 1.)) if contact else 0.0
        vmax = (1.0 - alpha) * ARM_QP_VMAX + alpha * min(ARM_QP_VMAX, .5)
        step_max = vmax * dt_step
        if G.shape[0] == 0 and not contact:                 # nothing within influence -> free
            if float(np.max(np.abs(q_des - self.q_cmd))) <= step_max:
                self.q_cmd = q_des.copy()                  # caught up: exact, no clamp
            else:                                          # recovering from a projection
                self.q_cmd = self.q_cmd + np.clip(q_des - self.q_cmd,
                                                  -step_max, step_max)
            return self._unpack()
        qd_des = (q_des - self.q_cmd) / dt
        qd_hi = np.minimum((self.hi - self.q_cmd) / dt, vmax * dt_step / dt)
        qd_lo = np.maximum((self.lo - self.q_cmd) / dt, -vmax * dt_step / dt)
        qd_hi[inactive] = 0.0
        qd_lo[inactive] = 0.0
        H, g = self._contact_objective(qd_des, qd_lo, qd_hi, dt)
        qd = solve_qp(H, g, qd_lo, qd_hi, G, h)
        if self.enable_gripper_contact and self._tip_points:
            # The shared helper's last resort is a finite unconstrained solve.
            # A large contact weight must never make that fallback look safe.
            def feasible(v):
                return (np.all(np.isfinite(v))
                        and np.all(v >= qd_lo - 1e-7) and np.all(v <= qd_hi + 1e-7)
                        and (not len(G) or np.all(G @ v <= h + 1e-7)))
            if not feasible(qd):
                qd = solve_qp(np.eye(self.n), np.zeros(self.n), qd_lo, qd_hi, G, h)
                if not feasible(qd):
                    self.fallbacks += 1
                    qd = np.zeros(self.n)
        if not np.all(np.isfinite(qd)):                    # degenerate -> hold (safe)
            self.fallbacks += 1
            qd = np.zeros(self.n)
        # QP solvers may return roundoff-sized velocities on zero-width boxes.
        # An untracked side must retain the exact command that was published.
        qd[inactive] = 0.0
        self.q_cmd = (self._contact_step(qd * dt) if contact
                      else self.q_cmd + qd * dt)
        return self._unpack()

    def _unpack(self):
        """(right arm, left arm, right ORCA wrist, left ORCA wrist)."""
        q = self.q_cmd
        return (q[:self.n_arm], q[self.split:self.split + self.n_arm],
                float(q[self.n_arm]), float(q[-1]))
