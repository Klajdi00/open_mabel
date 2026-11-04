"""
Whole-body upper-body controller for MABEL.

Bimanual SE(3) differential inverse kinematics over the 18-DOF upper chain
(lift + torso + 7 arm joints + wrist, per side), solved with a weighted
damped-least-squares pseudo-inverse and a null-space posture task that keeps
the body near a comfortable "human" configuration (HoMMI's C_nominal idea).

Targets are 6-DOF palm poses (position + SO(3) matrix). The controller writes
position commands to the arm/lift/torso actuators each tick.

LIFT UNITS — the lift is a tendon-cascaded stage pair ("Sync Lift" equality:
Upper == Lower; the "Lift" actuator drives the "Total Lift" tendon = their sum).
All lift state in this class (`target_q["Lift"]`, q_rest, the velocity box) is
in JOINT units — the "Lift Lower" qpos, HALF the tendon/ctrl travel. solve()
doubles the lift Jacobian column and apply() writes ctrl = 2*target, so the
planner's linearization matches the controlled plant. Consumers that mirror the
target onto the wire send lift_lower = lift_upper = target_q["Lift"] (the
facade sums them back into the tendon ctrl).
"""
import numpy as np
import mujoco

from .. import config as C
from ..motion_model import com_support as CS
from .collision_avoidance.capsule_guard import SelfCollisionGuard, solve_qp


def _deadband(e, db):
    """Shrink a vector's magnitude by `db`, clamped at zero (soft, continuous).
    Sub-threshold tracking errors map to no command, so the arm holds still
    instead of chasing gravity sag when the operator is idle."""
    n = float(np.linalg.norm(e))
    return e * (max(0.0, n - db) / n) if n > 1e-12 else e


def _so3_error(R_cur, R_trg):
    """Orientation error as a body-frame rotation vector (sum of column crosses)."""
    return 0.5 * (
        np.cross(R_cur[:, 0], R_trg[:, 0])
        + np.cross(R_cur[:, 1], R_trg[:, 1])
        + np.cross(R_cur[:, 2], R_trg[:, 2])
    )


class WholeBodyIK:
    """Holds IK state (targets, rest posture) and solves the upper-body chain."""

    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.dofs = self._ik_dof_indices()

        # Per-DOF weights for the weighted DLS (lazier joints get larger weight).
        w = np.ones(len(self.dofs))
        w[0] = C.IK_WEIGHT_LIFT
        w[1] = C.IK_WEIGHT_TORSO
        self.W_inv = np.diag(1.0 / w)

        # Arms-only subset: drop the Lift + Torso DOFs (first two), leaving the
        # 7-DOF arm + wrist per side. Used by the teleop "ARMS ONLY" mode so the
        # lift and torso stay perfectly stationary.
        self.arm_dofs = self.dofs[2:]
        self.arm_actuators = list(C.IK_ACTUATORS[2:])
        self.W_inv_arms = np.eye(len(self.arm_dofs))

        # Cache qpos addresses + joint limits per IK actuator so the hot path reads
        # them with one vectorised gather instead of a named lookup per joint/tick.
        self._qadr = np.array([self.model.joint(self._joint_of(n)).qposadr[0]
                               for n in C.IK_ACTUATORS])
        rng = np.array([self.model.joint(self._joint_of(n)).range
                        for n in C.IK_ACTUATORS], dtype=float)
        self._jnt_range = rng             # (ndof, 2) — Lift row is the STAGE joint range
        # Sync-Lift twin stage: scratch/target poses must move BOTH prismatic
        # joints (writing only "Lift Lower" leaves the pose half-lifted).
        self._lift_qadr2 = int(self.model.joint("Lift Upper").qposadr[0])
        self._lift_jrange = tuple(self.model.joint(C.LIFT_JOINT).range)

        # Per-DOF null-space posture weights (1.0 = the historical uniform
        # pull). A consumer may shape these to make the redundancy resolve
        # human-like — e.g. the Control Lab maps config.ARM_REG_W onto them
        # (strong return on shoulder-roll Arm3 and wrist-yaw Arm7, light on
        # the forearm) so roll happens at the forearm the way a human's does.
        self.posture_w = np.ones(len(self.dofs))
        # Orientation tracking gain (default = config). Lowering it lets the
        # palm orientation yield so a dragged arm finds natural poses.
        self.kp_rot = C.IK_KP_ROT

        # Targets and rest posture, captured from the current configuration.
        self.target_q = {}
        self.q_rest = None
        self.target_l_pos = self.target_r_pos = None
        self.target_l_mat = self.target_r_mat = None
        self.sync_to_current()
        self.q_rest = np.array([self._q(n) for n in C.IK_ACTUATORS])

        # Telemetry for the dashboard.
        self.e_pos_l = np.zeros(3); self.e_rot_l = np.zeros(3)
        self.e_pos_r = np.zeros(3); self.e_rot_r = np.zeros(3)

        # Self-collision avoidance (RUNTIME-TOGGLEABLE). When on, the plain DLS
        # tracking solution is post-processed by the guard:
        #   "project" -> CBF velocity projection: remove ONLY the normal
        #                (gap-closing) excess, keep the tangential component —
        #                the hand SLIDES along the body toward the target
        #                (default since 2026-07-29, operator-requested),
        #   "stop"    -> uniform slow-down that stops ON the approach path,
        #   "cbf"     -> legacy tracking-QP (re-optimises, can deflect sideways).
        # When off, solve() returns the plain weighted-DLS exactly as before.
        # The guard is built lazily on first enable so a disabled run pays nothing.
        self.self_collision = bool(C.SELF_COLLISION_ENABLE)
        self.sc_method = C.SC_METHOD
        self.guard = SelfCollisionGuard(model, data) if self.self_collision else None
        self._sc_scratch = None                 # MjData for target-pose ("stop") eval
        # Static-stability guard (ON by default): apply() decelerates the posture
        # advance to stop AT the CoM support margin (config.COM_GUARD_*) — chasing
        # an out-of-reach target must never contort the body past static tip-over.
        self.com_guard = bool(C.COM_GUARD_ENABLE)
        self.com_margin = None                  # last planned support margin [m]
        self._lp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, C.LEFT_PALM)
        self._rp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, C.RIGHT_PALM)
        # Preallocated Jacobian scratch (reused every tick -> no per-solve alloc).
        self._jpl = np.zeros((3, model.nv)); self._jrl = np.zeros((3, model.nv))
        self._jpr = np.zeros((3, model.nv)); self._jrr = np.zeros((3, model.nv))

    # ── helpers ──────────────────────────────────────────────────────────────
    def _ik_dof_indices(self):
        idx = [self.model.joint(C.LIFT_JOINT).dofadr[0],
               self.model.joint("Torso").dofadr[0]]
        for i in range(1, 8):
            idx.append(self.model.joint(f"Left Arm {i}").dofadr[0])
        idx.append(self.model.joint("Left Wrist").dofadr[0])
        for i in range(1, 8):
            idx.append(self.model.joint(f"Right Arm {i}").dofadr[0])
        idx.append(self.model.joint("Right Wrist").dofadr[0])
        return idx

    @staticmethod
    def _joint_of(actuator_name):
        return C.LIFT_JOINT if actuator_name == "Lift" else actuator_name

    def _q(self, actuator_name):
        joint = self._joint_of(actuator_name)
        return self.data.joint(joint).qpos[0]

    def _q_vec(self, arms_only):
        """All IK joint angles via one cached gather (no per-joint named lookup)."""
        qadr = self._qadr[2:] if arms_only else self._qadr
        return self.data.qpos[qadr]

    def sync_to_current(self, freeze_posture=False):
        """Snap targets to where the palms / joints actually are right now.
        Call this whenever the base/lift is driven directly so the IK mode does
        not snap when control is handed back to it.

        freeze_posture=True ALSO retargets the null-space posture regulariser to
        the current configuration. Without it, the null-space term keeps pulling the
        arms toward the construction-time rest pose, so taking IK control back after
        the arm was moved elsewhere (e.g. the operator dragged the robot in
        compliance mode) would reconfigure the elbows even with the palm targets held.
        With q_rest == q_cur that term is zero, so NOTHING moves until the operator
        commands a palm target."""
        for name in C.IK_ACTUATORS:
            self.target_q[name] = self._q(name)
        self.target_l_pos = self.data.body(C.LEFT_PALM).xpos.copy()
        self.target_r_pos = self.data.body(C.RIGHT_PALM).xpos.copy()
        self.target_l_mat = self.data.body(C.LEFT_PALM).xmat.copy().reshape(3, 3)
        self.target_r_mat = self.data.body(C.RIGHT_PALM).xmat.copy().reshape(3, 3)
        if freeze_posture:
            self.q_rest = np.array([self._q(n) for n in C.IK_ACTUATORS])

    # ── self-collision toggle ─────────────────────────────────────────────────
    def set_self_collision(self, on, method=None):
        """Enable/disable self-collision avoidance at runtime. The guard is built
        on first enable (so a disabled session pays no construction cost). Pass
        `method` ("project" | "stop" | "cbf") to also switch the avoidance law."""
        on = bool(on)
        if method is not None:
            self.sc_method = method
        if on and self.guard is None:
            self.guard = SelfCollisionGuard(self.model, self.data)
        self.self_collision = on
        return self.self_collision

    def toggle_self_collision(self):
        return self.set_self_collision(not self.self_collision)

    # ── core solve ───────────────────────────────────────────────────────────
    def _twist(self, p_trg, R_trg, body):
        p_cur = self.data.body(body).xpos
        R_cur = self.data.body(body).xmat.reshape(3, 3)
        e_pos = _deadband(p_trg - p_cur, C.IK_POS_DEADBAND)
        e_rot = _deadband(_so3_error(R_cur, R_trg), C.IK_ROT_DEADBAND)

        v = C.IK_KP_POS * e_pos
        w = self.kp_rot * e_rot
        vn = np.linalg.norm(v)
        if vn > C.IK_VEL_CAP:
            v *= C.IK_VEL_CAP / vn
        wn = np.linalg.norm(w)
        if wn > C.IK_ANGVEL_CAP:
            w *= C.IK_ANGVEL_CAP / wn
        return np.hstack([v, w]), e_pos, e_rot

    def solve(self, arms_only=False):
        """Return dq (joint velocities) to track both palms. With arms_only the
        Lift + Torso DOFs are excluded, so only the 7-DOF arms (+ wrist) move."""
        m, d = self.model, self.data
        if arms_only:
            dofs, W_inv, actuators, q_rest = (
                self.arm_dofs, self.W_inv_arms, self.arm_actuators, self.q_rest[2:])
        else:
            dofs, W_inv, actuators, q_rest = (
                self.dofs, self.W_inv, list(C.IK_ACTUATORS), self.q_rest)

        twist_l, self.e_pos_l, self.e_rot_l = self._twist(
            self.target_l_pos, self.target_l_mat, C.LEFT_PALM)
        twist_r, self.e_pos_r, self.e_rot_r = self._twist(
            self.target_r_pos, self.target_r_mat, C.RIGHT_PALM)
        e_vel = np.hstack([twist_l, twist_r])  # 12-vector

        mujoco.mj_jacBody(m, d, self._jpl, self._jrl, self._lp_id)
        mujoco.mj_jacBody(m, d, self._jpr, self._jrr, self._rp_id)
        J = np.vstack([self._jpl, self._jrl, self._jpr, self._jrr])[:, dofs]  # 12 x ndof
        if not arms_only:
            # Sync-Lift cascade: one unit of the "Lift" (joint-space) target moves
            # BOTH stages, so the palm rises 2x what the raw "Lift Lower" column says.
            J[:, 0] *= 2.0

        lam = C.IK_DAMPING
        q_cur = self._q_vec(arms_only)
        pw = self.posture_w[2:] if arms_only else self.posture_w
        dq_post = C.IK_POSTURE_GAIN * pw * (q_rest - q_cur)

        # Plain weighted-DLS + null-space posture: the operator's tracking solution.
        JW = J @ W_inv
        J_pinv = W_inv @ J.T @ np.linalg.inv(JW @ J.T + (lam ** 2) * np.eye(12))
        dq = J_pinv @ e_vel + (np.eye(len(dofs)) - (J_pinv @ J)) @ dq_post

        if not self.self_collision or self.guard is None:
            return dq                              # default teleop, untouched

        if self.sc_method == "cbf":
            # Legacy CBF tracking-QP: re-optimise tracking subject to the barrier.
            # With W = W_inv⁻¹ the cost ½‖J q̇ − v‖² + ½λ²q̇ᵀW q̇ + ½μ‖q̇ − q̇_post‖²
            # reproduces the weighted DLS + posture when no constraint is active. It
            # slides the hand around obstacles (the "force field" feel).
            W = np.diag(1.0 / np.diag(W_inv))
            mu = C.SC_POSTURE_REG
            H = J.T @ J + (lam ** 2) * W + mu * np.eye(len(dofs))
            g = -(J.T @ e_vel + mu * dq_post)
            dq_lo, dq_hi = self._velocity_box(arms_only, q_cur)
            Gc, hc = self.guard.rows(dofs)
            return solve_qp(H, g, dq_lo, dq_hi, Gc, hc)

        if self.sc_method == "project":
            # CBF velocity projection (default): remove ONLY each active barrier's
            # normal (gap-closing) excess and keep the tangential component — the
            # hand slides along the torso/head/other arm toward the target instead
            # of stopping on the approach path. Constrained at BOTH poses (see
            # _project_both).
            return self._project_both(dq, dofs, arms_only)

        # "stop": uniformly slow the tracking solution so no monitored gap
        # closes faster than the barrier allows. Scaling preserves the motion
        # DIRECTION exactly, so the hand decelerates straight to a stop at the
        # margin along the operator's intended path — a wall, not a force field.
        # Tangential sliding and retraction are untouched (scale 1).
        return self._stop_scale(dq, dofs, arms_only) * dq

    def _project_both(self, dq, dofs, arms_only):
        """Project dq onto the CBF rows of the TARGET pose AND the MEASURED pose,
        stacked. Target-side rows keep the committed setpoints outside the
        margins (servo-lag argument, see _at_target_pose); measured-side rows
        stop the PHYSICAL corner-cut — a tangential orbit around a curved
        obstacle at the target pose is a closing motion at the lagging physical
        pose (the l_hand↔r_hand crossing penetrated 4 cm on target-only rows)."""
        Gt, ht = self._at_target_pose(
            lambda: self.guard.rows(dofs, braking=True), arms_only)
        Gp, hp = self.guard.rows(dofs, braking=True,
                                 margin_pad=C.SC_LAG_PAD)  # guard.data == live data
        if len(Gt) and len(Gp):
            rows = (np.vstack([Gt, Gp]), np.hstack([ht, hp]))
        elif len(Gt):
            rows = (Gt, ht)
        else:
            rows = (Gp, hp)
        return self.guard.project(dq, dofs, rows=rows)

    def _at_target_pose(self, fn, arms_only):
        """Run ``fn()`` with the guard's capsules evaluated at the TARGET pose.

        The arm is position-servoed to the integrated targets, so the physical
        configuration LAGS the target. Gating on the (lagging) physical pose lets
        the target run past the margin while the arm catches up → penetration.
        Evaluating the barrier where the servos will drive keeps the committed
        target outside the margin, so the physical arm settles outside it too.
        The target pose is built in a scratch MjData (kinematics + comPos only)
        so the live simulation is never disturbed."""
        if self._sc_scratch is None:
            self._sc_scratch = mujoco.MjData(self.model)
        scratch = self._sc_scratch
        scratch.qpos[:] = self.data.qpos
        actuators = self.arm_actuators if arms_only else C.IK_ACTUATORS
        for name in actuators:
            scratch.qpos[self.model.joint(self._joint_of(name)).qposadr[0]] = self.target_q[name]
        if not arms_only:
            scratch.qpos[self._lift_qadr2] = self.target_q["Lift"]   # Sync-Lift twin
        mujoco.mj_kinematics(self.model, scratch)
        mujoco.mj_comPos(self.model, scratch)
        self.guard.data = scratch               # evaluate capsules at the target pose
        try:
            return fn()
        finally:
            self.guard.data = self.data         # restore for everything else

    def _stop_scale(self, dq, dofs, arms_only):
        """Scale factor for the "stop" law, evaluated at the TARGET pose
        (see _at_target_pose for why the target pose, not the lagging physical
        one) — robust regardless of approach speed."""
        return self._at_target_pose(
            lambda: self.guard.speed_scale(dq, dofs), arms_only)

    def _velocity_box(self, arms_only, q_cur):
        """Per-DOF q̇ limits (vectorised): a velocity cap, tightened by a linear
        damper that drives the bound to zero at each joint's position limit
        (q̇ ≤ k(q_hi−q)). The lift cap is HALF the hardware travel ceiling: the
        Sync-Lift cascade moves both stages, so total travel rate = 2·q̇_joint."""
        cap = np.full(len(q_cur), C.SC_QP_VEL_CAP)
        rng = self._jnt_range[2:] if arms_only else self._jnt_range
        if not arms_only:
            cap[0] = 0.5 * C.MAX_LIFT_IK_SPEED
        k = C.SC_JOINT_LIMIT_GAIN
        hi = np.minimum(cap, k * (rng[:, 1] - q_cur))     # +inf range → cap wins
        lo = np.maximum(-cap, -k * (q_cur - rng[:, 0]))
        return lo, hi

    # ── static-stability guard (CoM stays inside the wheel support triangle) ──
    def _margin_at(self, actuators, dq=None, dt=0.0):
        """CoM support margin [m] of the pose the servos are being SENT toward:
        the integrated targets (+ optionally the dq*dt candidate step) on a
        scratch MjData; base pose and non-managed joints stay at the live state.
        Evaluating the plan (not the lagging physical pose) keeps the committed
        posture statically stable — same rationale as the self-collision law."""
        if self._sc_scratch is None:
            self._sc_scratch = mujoco.MjData(self.model)
        scratch = self._sc_scratch
        scratch.qpos[:] = self.data.qpos
        for i, name in enumerate(actuators):
            v = self.target_q[name] + (dq[i] * dt if dq is not None else 0.0)
            scratch.qpos[self.model.joint(self._joint_of(name)).qposadr[0]] = v
            if name == "Lift":
                scratch.qpos[self._lift_qadr2] = v      # Sync-Lift twin
        mujoco.mj_kinematics(self.model, scratch)
        mujoco.mj_comPos(self.model, scratch)
        com = CS.robot_com(self.model, scratch)
        return CS.support_margin(com[:2], CS.support_polygon(self.model, scratch))

    def _com_guard_project(self, dq, actuators):
        """CBF damper on the static-stability margin: near the boundary, project
        dq onto the half-space  ∇m·q̇ ≥ −γ(m − m_stop),  so the posture may only
        CONSUME margin at a bounded rate and must slide TANGENTIALLY along the
        boundary once there (γ from config.COM_GUARD_GAIN).

        A uniform stop-scale (the self-collision "stop" recipe) is wrong here:
        blocking the whole dq at the boundary while margin-raising deflections
        pass ratchets the posture into folded-arm attractors — measured on the
        bench as 0.6–0.9 m of lost reach. The projection keeps every tracking
        component that does not push the CoM out, which is the whole-body
        behaviour we actually want: the arm keeps reaching, the torso yields.

        The margin gradient is finite-differenced over the managed DOF targets;
        the FD sweep only runs inside a 2 cm approach band, so far from the
        boundary the guard costs one margin evaluation."""
        m0 = self._margin_at(actuators)
        self.com_margin = m0
        if m0 >= C.COM_GUARD_MARGIN + 0.01:
            return dq                           # outside the approach band: free
        n = len(actuators)
        g = np.zeros(n)
        eps = 1e-3
        step = np.zeros(n)
        for i in range(n):
            step[:] = 0.0
            step[i] = eps
            g[i] = (self._margin_at(actuators, step, 1.0) - m0) / eps
        gg = float(g @ g)
        if gg < 1e-12:
            return dq
        rhs = -C.COM_GUARD_GAIN * (m0 - C.COM_GUARD_MARGIN)  # >0 inside: push out
        viol = float(g @ dq) - rhs
        if viol >= 0.0:
            return dq
        return dq - (viol / gg) * g             # minimal correction onto the plane

    def apply(self, dq, dt, arms_only=False):
        """Integrate dq into the actuator position targets and command them.
        With arms_only the Lift + Torso actuators are left untouched (they hold)."""
        actuators = self.arm_actuators if arms_only else C.IK_ACTUATORS
        if self.com_guard:
            dq = self._com_guard_project(np.asarray(dq, float), list(actuators))
        # The CoM-guard projection (and the torso rate clip below) re-shape dq
        # AFTER solve()'s barrier pass, and a re-shaped velocity can point back
        # into a barrier — measured on the arms-crossing scenario as a per-tick
        # violation that integrated 4 cm INTO the capsule model. Under the
        # "project" law the LAST word on dq therefore belongs to the barrier:
        # re-project the post-CoM velocity so what actually integrates into the
        # targets satisfies the CBF rows.
        if (self.self_collision and self.guard is not None
                and self.sc_method == "project"):
            dofs = self.arm_dofs if arms_only else self.dofs
            dq = self._project_both(np.asarray(dq, float), dofs, arms_only)

        def clipped_v(i, name):
            """The velocity that will ACTUALLY integrate for this actuator after
            every clamp (torso/lift rate caps + ctrlrange saturation)."""
            v = float(dq[i])
            if name == "Torso":
                v = float(np.clip(v, -C.MAX_TORSO_IK_SPEED, C.MAX_TORSO_IK_SPEED))
            if name == "Lift":
                v = float(np.clip(v, -0.5 * C.MAX_LIFT_IK_SPEED,
                                  0.5 * C.MAX_LIFT_IK_SPEED))
                lo, hi = self._lift_jrange
            else:
                act_id = self.model.actuator(name).id
                lo, hi = self.model.actuator_ctrlrange[act_id]
            q_new = float(np.clip(self.target_q[name] + v * dt, lo, hi))
            return (q_new - self.target_q[name]) / dt

        # FINAL SAFETY GATE ("project" law). The rate caps and ctrlrange
        # saturations above/below RESHAPE the projected velocity — a joint
        # pinned on its range limit deletes the component the projection relied
        # on to keep a gap open, and what integrates then violates the barrier
        # by a fraction of a millimetre EVERY tick (measured: the hand-hand
        # crossing collapsed 2.5 cm of margin + 3 cm of capsule this way,
        # gradients verified exact against FD). So: form the velocity that will
        # ACTUALLY integrate, re-check the stacked CBF rows, and if the clips
        # broke feasibility fall back to uniformly SLOWING the whole step until
        # it is feasible again ("stop"-law semantics — direction preserved,
        # provably conservative: s=0 is always safe for h ≥ 0 rows).
        if (self.self_collision and self.guard is not None
                and self.sc_method == "project"):
            v_real = np.array([clipped_v(i, n) for i, n in enumerate(actuators)])
            Gt, ht = self._at_target_pose(
                lambda: self.guard.rows(dofs, braking=True), arms_only)
            Gp, hp = self.guard.rows(dofs, braking=True, margin_pad=C.SC_LAG_PAD)
            G = np.vstack([g for g in (Gt, Gp) if len(g)]) if (len(Gt) or len(Gp)) \
                else np.zeros((0, len(dofs)))
            h = np.hstack([x for x in (ht, hp) if len(x)]) if len(G) else np.zeros(0)
            if len(G):
                s = 1.0
                gv = G @ v_real
                for k in range(len(h)):
                    if gv[k] > h[k] + 1e-9 and gv[k] > 1e-12:
                        s = min(s, max(0.0, h[k]) / gv[k])
                if s < 1.0:
                    dq = v_real * s

        for i, name in enumerate(actuators):
            v = dq[i]
            if name == "Torso":
                # The torso swings the whole upper-body inertia — rate-cap it so
                # a far target cannot slam it (the momentum defeats the CoM guard).
                v = float(np.clip(v, -C.MAX_TORSO_IK_SPEED, C.MAX_TORSO_IK_SPEED))
            if name == "Lift":
                # JOINT units: total travel rate = 2·v (Sync-Lift cascade), so cap
                # v at half the hardware ceiling; clip the target to the STAGE joint
                # range (the tendon ctrlrange is the 2x stage sum) and command the
                # tendon actuator with the sum.
                v = float(np.clip(v, -0.5 * C.MAX_LIFT_IK_SPEED, 0.5 * C.MAX_LIFT_IK_SPEED))
                lo, hi = self._lift_jrange
                self.target_q[name] = float(np.clip(self.target_q[name] + v * dt, lo, hi))
                self.data.actuator(name).ctrl[0] = 2.0 * self.target_q[name]
                continue
            act_id = self.model.actuator(name).id
            lo, hi = self.model.actuator_ctrlrange[act_id]
            self.target_q[name] = float(np.clip(self.target_q[name] + v * dt, lo, hi))
            self.data.actuator(name).ctrl[0] = self.target_q[name]
