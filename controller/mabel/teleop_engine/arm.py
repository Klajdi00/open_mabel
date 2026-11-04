"""Decoupled per-arm cosine-similarity IK (wrist SE(3) + natural posture)."""
import numpy as np
import mujoco
from scipy.optimize import minimize

from .hyperparams import *


class ArmRetargeter:
    """Cosine-similarity arm retargeting — the SAME approach as the finger
    optimizer, lifted to the arm. On a scratch MjData it matches, scale-
    independently, the DIRECTIONS of the operator arm keypoints (head->elbow,
    head->wrist) and the palm orientation axes to the robot's (cosine), plus
    relative-distance (reach) and a per-joint regularization. The ORCA wrist
    (pitch) is a jointly-optimised 8th DoF; with lock_j7 the OpenArm J7 is fixed
    (6-DoF arm + ORCA wrist).

    TWO SOURCES OF STRUCTURE (2026-08-23). The solve above is LOCAL and
    warm-started, which is right for TRACKING and wrong for choosing which
    configuration a pose belongs to — the two are not separable on a redundant
    arm, and the palm flip is where that bites (see workspace_table.py for the
    mechanism). So the global part comes from a precomputed posture table over
    the front manipulation volume, which enters here in three ways, cheapest
    first:

      * ``self.q_ref``  — the regulariser pulls toward the table's posture
        instead of qpos0 (which on this robot is both arms straight forward, a
        posture no human holds). One interpolation, no IK.
      * ``self.w_orient`` — the palm-normal weight is gated on the table's own
        recorded feasibility, so the arm leans on a palm direction it can
        actually reach and gives it up (keeping the wrist) where it cannot.
      * ``_table_steer`` — when the table's CONFIGURATION beats the warm solve
        on the task cost by a clear margin, walk to it through the same trust
        region tracking uses. Continuous, rate-limited, no snap.

    With no table file present every one of those is a no-op and this is exactly
    the pre-2026-08-23 local solver."""

    def __init__(self, model, side, R_rob, lock_j7=False, optimize_wrist=None,
                 use_table=None, table_dir=None, j7_yaw_lim=None,
                 w_orient_fwd=None, w_orient=None, w_orient_min=None,
                 trust_rate_distal=None):
        self.model = model
        # finger-axis (spin) weight, separated from the palm-normal weight; the
        # experiment harness sweeps it, so it is an instance value not a global
        self.w_orient_fwd = (ARM_W_ORIENT_FWD if w_orient_fwd is None
                             else float(w_orient_fwd))
        # palm-NORMAL weight: a RANGE, relaxed toward the minimum where the table
        # says the palm direction is not deliverable (see _table_lookup and
        # hyperparams.ARM_W_ORIENT). Held per instance so the experiment harness
        # can reconstruct the pre-change behaviour exactly, not approximately.
        self.w_orient_max = ARM_W_ORIENT if w_orient is None else float(w_orient)
        self.w_orient_min = (ARM_W_ORIENT_MIN if w_orient_min is None
                             else float(w_orient_min))
        self.w_orient = self.w_orient_max
        self.trust_rate_distal = (ARM_TRUST_RATE_DISTAL if trust_rate_distal is None
                                  else float(trust_rate_distal))
        # {opt index: rad/s} beating the proximal / distal split for single joints (the robot's own limits; set by the
        # Retarget Lab per robot, e.g. NOVA's J3 upper-arm roll that finishes a palm flip once J5 is on its stop)
        self.trust_override = {}
        self.w_elbow = ARM_W_ELBOW
        self._j7_yaw_lim_arg = j7_yaw_lim
        self.profile = "wrist"
        self.ik = mujoco.MjData(model)
        self.side = side
        S = side.capitalize()
        self.lock_j7 = lock_j7
        if optimize_wrist is None:
            optimize_wrist = ARM_OPTIMIZE_WRIST
        self.optimize_wrist = optimize_wrist
        self.orca_wrist_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{S} Wrist")
        self.orca_wrist_qadr = model.jnt_qposadr[self.orca_wrist_jid]
        self.orca_wrist_lo, self.orca_wrist_hi = model.jnt_range[self.orca_wrist_jid]

        n_arm = 6 if lock_j7 else 7
        self.n_arm = n_arm
        # self.jids / self.qadr are the ARM joints only (what the ctrl-write + the
        # self-collision projector consume). The OPTIMISER additionally includes the
        # ORCA wrist as an 8th DoF when optimize_wrist -- it is jointly solved (not
        # hard-coded from the human flexion), which adds redundancy that unlocks the
        # natural flip up/down and lets the arm avoid twisting.
        self.jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{S} Arm {i}")
                     for i in range(1, n_arm + 1)]
        self.qadr = np.array([model.jnt_qposadr[j] for j in self.jids])
        self.opt_jids = list(self.jids) + ([self.orca_wrist_jid] if optimize_wrist else [])
        self.opt_qadr = np.array([model.jnt_qposadr[j] for j in self.opt_jids])
        self.lo = model.jnt_range[self.opt_jids, 0].copy()
        self.hi = model.jnt_range[self.opt_jids, 1].copy()
        self.q_nom = model.qpos0[self.opt_qadr].copy()
        # Wrist YAW (Arm7) is anatomically constrained -> clip its OPTIMISATION bound to
        # rest +/- ARM_J7_YAW_LIM so the solver cannot spin it to fake orientation
        # (forces the flip onto Arm5/Arm6 + the ORCA wrist). Arm7 is index 6 of opt_jids.
        if not lock_j7:
            nom7 = self.q_nom[6]
            lim = ARM_J7_YAW_LIM if j7_yaw_lim is None else float(j7_yaw_lim)
            self.lo[6] = max(self.lo[6], nom7 - lim)
            self.hi[6] = min(self.hi[6], nom7 + lim)
        self.bounds = list(zip(self.lo, self.hi))
        # per-joint regularisation: the arm entries from ARM_REG_W, plus a LIGHT weight on
        # the ORCA wrist so it can move freely to flip but returns to rest when idle.
        self.reg_w = ARM_REG_W[:n_arm].copy()
        if optimize_wrist:
            self.reg_w = np.append(self.reg_w, ARM_REG_WRIST)
        self.last_q = self.q_nom.copy()
        self._warm = False           # first solve gets the full joint range
        self._stuck = 0              # consecutive quasi-static high-error solves
        self._prev_contact = None    # last target (basin-escape staleness gate)
        self._last_dt = None          # this solve's frame interval (trust step)
        self._steer_q = None         # latched canonical basin being walked to
        # DoF columns of the optimised joints, for the analytic-Jacobian GN
        # solver (mj_jac returns nv-wide rows; slice these).
        self.opt_dofs = np.array([model.jnt_dofadr[j] for j in self.opt_jids])
        self.wrist_body = int(model.jnt_bodyid[self.orca_wrist_jid])
        # END EFFECTOR (end_effector.py): "orca" tracks the ORCA wrist anchor (unchanged); "gripper" tracks the
        # gripper base face (end_effector.tracked_point) — set_end_effector.
        self.ee = "orca"
        self._ee_body, self._ee_off, self._ee_saved = None, None, None
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))
        # J7 locked at its rest value when requested
        j7 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{S} Arm 7")
        self.j7_qadr = model.jnt_qposadr[j7]
        self.j7_lock_val = float(model.qpos0[self.j7_qadr])
        # keypoint bodies
        self.elbow_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                           "r_elbow" if side == "right" else "l_elbow")
        self.shoulder_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                              "r_arm_mount" if side == "right" else "l_arm_mount")
        self.palm_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_palm")
        self.head_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "head")
        self.chassis_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mabel_chassis")
        self.R_rob = R_rob

        # ── the front-workspace posture table (see hyperparams ARM_USE_TABLE) ──
        # Loaded lazily-but-once; absent file = the pure local solve, unchanged.
        self.table = None
        self.q_ref = self.q_nom.copy()      # posture reference the reg term pulls to
        self._tab_q = None                  # this frame's looked-up posture
        self._tab_hits = 0                  # diagnostics: how often the table steered
        self._tab_feas = 1.0                # last frame's table feasibility (diagnostic)
        if use_table is None:
            use_table = ARM_USE_TABLE
        if use_table:
            self.table = self._load_table(table_dir)

    # ── retargeting PROFILE (the app's "Whole-Body Retargeting" switch) ──────
    #
    # Two named configurations of the same solver, because the operator is
    # choosing between two different JOBS rather than tuning a number:
    #
    #   "whole_body"  the ORIGINAL retargeting — the arm follows the operator's
    #                 whole arm, elbow direction included, with the pre-
    #                 2026-08-23 constants exactly. Reproduces the operator's
    #                 POSTURE at the cost of wrist precision.
    #   "wrist"       the front-workspace tabular retargeting — the palm goes
    #                 where the operator points it and holds the direction they
    #                 asked for; the elbow is resolved by the posture table
    #                 rather than tracked. Measured over the front grid, palm
    #                 error p90 4.6 deg -> 1.4 deg and flips that never converge
    #                 15 % -> 0 %.
    #
    # Applied IN PLACE, preserving `last_q`, so a switch mid-session is
    # continuous — the trust region walks the arm from wherever it is to
    # whatever the new profile asks for, exactly as it does for a moving target.
    PROFILES = {
        "whole_body": dict(use_table=False, j7_yaw_lim=0.45, w_orient=20.0,
                           w_orient_fwd=20.0, w_elbow=30.0, trust_rate_distal=2.0),
        "wrist":      dict(use_table=True,  j7_yaw_lim=None, w_orient=None,
                           w_orient_fwd=None, w_elbow=None, trust_rate_distal=None),
    }

    def set_profile(self, name, table_dir=None):
        """Switch retargeting profile (see PROFILES). Returns the active name."""
        if name not in self.PROFILES:
            raise ValueError(f"profile must be one of {sorted(self.PROFILES)}, "
                             f"got {name!r}")
        p = self.PROFILES[name]
        self.w_orient = ARM_W_ORIENT if p["w_orient"] is None else p["w_orient"]
        self.w_orient_fwd = (ARM_W_ORIENT_FWD if p["w_orient_fwd"] is None
                             else p["w_orient_fwd"])
        self.w_elbow = ARM_W_ELBOW if p["w_elbow"] is None else p["w_elbow"]
        self.trust_rate_distal = (ARM_TRUST_RATE_DISTAL
                                  if p["trust_rate_distal"] is None
                                  else p["trust_rate_distal"])
        # Arm 7's optimisation bound is part of the profile, so recompute it from
        # the MODEL range each time rather than narrowing the previous value.
        if not self.lock_j7:
            lim = ARM_J7_YAW_LIM if p["j7_yaw_lim"] is None else p["j7_yaw_lim"]
            nom7 = self.q_nom[6]
            self.lo[6] = max(self.model.jnt_range[self.opt_jids[6], 0], nom7 - lim)
            self.hi[6] = min(self.model.jnt_range[self.opt_jids[6], 1], nom7 + lim)
            self.bounds = list(zip(self.lo, self.hi))
            self.last_q = np.clip(self.last_q, self.lo, self.hi)
        self.table = self._load_table(table_dir) if p["use_table"] else None
        if self.table is None:
            self.q_ref = self.q_nom.copy()      # no atlas -> the plain rest posture
        self._steer_q = None                    # drop any latched walk from the old profile
        self.profile = name
        return name

    def _load_table(self, table_dir=None):
        from .workspace_table import WorkspaceTable, model_fingerprint
        try:
            fp = model_fingerprint(self.model, self.side, self.R_rob, self.lock_j7,
                                   self.optimize_wrist, self._j7_yaw_lim_arg)
            return WorkspaceTable.load_for(self.side, table_dir, fingerprint=fp)
        except Exception:                       # a corrupt/old table must not break teleop
            return None

    def sync(self, data):
        """Load the live robot pose (base/lift/torso/neck current) into the scratch."""
        self.ik.qpos[:] = data.qpos
        if self.lock_j7:
            self.ik.qpos[self.j7_qadr] = self.j7_lock_val

    def set_wrist_flex(self, theta):
        """Hard-code the ORCA wrist from the human flexion. No-op when the wrist is a
        jointly-optimised DoF (the solver sets it from the orientation objective), and held at 0 with the
        gripper (it has no wrist)."""
        if self.optimize_wrist or self.ee == "gripper":
            return
        q = WRIST_FLEX_SIGN * WRIST_FLEX_GAIN * float(theta)
        self.ik.qpos[self.orca_wrist_qadr] = min(max(q, self.orca_wrist_lo), self.orca_wrist_hi)

    def _ee_point(self, d):
        """(tracked point in world, its body) on MjData d: the ORCA wrist anchor, or the gripper base face."""
        if self.ee == "orca":
            return d.xanchor[self.orca_wrist_jid], self.wrist_body
        b = self._ee_body
        return d.xpos[b] + d.xmat[b].reshape(3, 3) @ self._ee_off, b

    def set_end_effector(self, kind):
        """'orca' (default) or 'gripper' (the OpenArm v1 gripper on J7, end_effector.py): the tracked point
        becomes the gripper base face (end_effector.tracked_point), the ORCA wrist is held at 0 (the gripper
        has none; with it at 0 the palm frame is rigid on J7, so the palm-orientation objective carries over) and the ORCA posture table is
        set aside (it was built for the ORCA wrist point)."""
        from .end_effector import KINDS, tracked_point
        if kind not in KINDS:
            raise ValueError(f"end effector must be one of {KINDS}, got {kind!r}")
        if kind == self.ee:
            return self.ee
        iw = len(self.jids)                                  # the ORCA wrist's slot in the opt vector
        if kind == "gripper":
            if self._ee_body is None:
                self._ee_body, self._ee_off = tracked_point(self.model, self.side, self.palm_bid, self.R_rob)
            self._ee_saved = (self.table, self.lo.copy(), self.hi.copy())
            self.table = None
            if self.optimize_wrist:
                self.lo[iw] = self.hi[iw] = 0.0
                self.last_q[iw] = self.q_nom[iw] = self.q_ref[iw] = 0.0
            self.ik.qpos[self.orca_wrist_qadr] = 0.0
        else:
            self.table, self.lo, self.hi = self._ee_saved
            self.q_nom = self.model.qpos0[self.opt_qadr].copy()
        self.bounds = list(zip(self.lo, self.hi))
        self.ee = kind
        self._warm = False                                   # next solve: the full joint range again
        return self.ee

    def ee_pos(self):
        mujoco.mj_kinematics(self.model, self.ik)
        return self._ee_point(self.ik)[0].copy()

    def _kps(self, q):
        q = np.asarray(q, float)
        if len(q) == len(self.opt_qadr):                      # full opt vector (arm + wrist)
            self.ik.qpos[self.opt_qadr] = q
        else:                                                 # arm-only -> keep current wrist
            self.ik.qpos[self.qadr] = q
        if self.lock_j7:
            self.ik.qpos[self.j7_qadr] = self.j7_lock_val
        mujoco.mj_kinematics(self.model, self.ik)             # FORWARD KINEMATICS
        elbow = self.ik.xpos[self.elbow_bid].copy()           # world
        wrist = self._ee_point(self.ik)[0].copy()             # world (ORCA-wrist anchor / gripper base face)
        sh = self.ik.xpos[self.shoulder_bid].copy()           # world (arm mount)
        Rp = self.ik.xmat[self.palm_bid].reshape(3, 3) @ self.R_rob   # palm canonical (world)
        # Match the ROBOT WRIST to the operator wrist (ARM_TRACK_WRIST) so finger flexion
        # never moves the arm; the HAND_REACH fingertip offset is only added in the legacy
        # fingertip-contact mode (operator target = middle fingertip).
        contact = wrist if ARM_TRACK_WRIST else wrist + HAND_REACH * Rp[:, 0]
        return elbow, contact, sh, Rp[:, 0], Rp[:, 2]         # elbow, CONTACT, shoulder, fwd, up

    def _reg_for(self, hk):
        """Posture regulariser, Arm3 (shoulder roll) relaxed IN PROPORTION to
        the operator's commanded elbow swivel (hk['swivel'], rad, None for
        tracked-elbow clients): the 0.40 weight resolves swivel redundancy at
        a neutral elbow, but pins the humerus roll elbow-up needs. An
        unconditional relax measurably tripled the fast-circle direction RMS
        — the freedom is only granted when the operator is USING it."""
        sw = hk.get("swivel")
        if sw is None or len(self.reg_w) < 3:
            return self.reg_w
        reg = self.reg_w.copy()
        reg[2] *= max(ARM_REG_ELBOW_RELAX, 1.0 - abs(float(sw)))
        return reg

    def _w_elbow(self, hk):
        """Elbow-direction weight, boosted with the commanded swivel: at a
        large deliberate elbow raise the ROLLED palm orientation becomes
        unreachable (J5/J6 saturate) and its cost cancels the elbow gain —
        the two basins measured EQUAL total cost, so the arm stalled half-way.
        When the operator steers the elbow, the elbow look wins the tie.
        hk['elbow_w_scale'] (tracked-elbow clients, 2026-07-29) scales the
        weight by the operator's actual elbow FLEXION: a near-straight arm's
        elbow direction is geometrically weak and must not tax wrist-position
        accuracy."""
        sw = hk.get("swivel")
        boost = 1.0 + ARM_ELBOW_SWIVEL_BOOST * abs(float(sw or 0.0))
        return self.w_elbow * boost * float(hk.get("elbow_w_scale", 1.0))

    def _cost(self, q, hk, q_prev):
        re, rc, rsh, rfwd, rup = self._kps(q)
        # precise CONTACT-point position (hand contact pt -> operator fingertip), so
        # it tracks AND the two robot hands touch when the operator's fingertips do.
        loss = ARM_W_WRIST * float(np.sum((rc - hk["contact"]) ** 2))
        # ELBOW direction FROM THE SHOULDER -> the arm bends like the human's (this
        # is what keeps the redundant arm from twisting into weird configurations).
        if hk["elbow"] is not None:
            ed = re - rsh; hed = hk["elbow"] - rsh
            ed /= (np.linalg.norm(ed) + 1e-9); hed /= (np.linalg.norm(hed) + 1e-9)
            loss += self._w_elbow(hk) * (1.0 - float(np.dot(ed, hed)))
        # palm orientation: the NORMAL (which way the palm faces) carries the
        # weight; the finger axis inside the palm plane is a weak preference.
        loss += self.w_orient_fwd * (1.0 - float(np.dot(rfwd, hk["fwd"])))
        loss += self.w_orient * (1.0 - float(np.dot(rup, hk["up"])))
        # temporal smoothness + per-joint regularization (roll from forearm)
        loss += ARM_W_SMOOTH * float(np.sum((q - q_prev) ** 2))
        loss += float(np.sum(self._reg_for(hk) * (q - self.q_ref) ** 2))
        return loss

    def _task_cost(self, q, hk):
        """Position + orientation ONLY — the part of the cost that says whether
        the POSE was achieved. Basin comparisons must use this and not the full
        cost: smoothness always favours the incumbent configuration, so it would
        veto every genuine branch change (that is why the palm flip never
        happened)."""
        _, rc, _, rfwd, rup = self._kps(q)
        return (ARM_W_WRIST * float(np.sum((rc - hk["contact"]) ** 2))
                + self.w_orient_fwd * (1.0 - float(np.dot(rfwd, hk["fwd"])))
                + self.w_orient * (1.0 - float(np.dot(rup, hk["up"]))))

    def solve(self, hk, dt=None):
        """hk: dict with contact / elbow / fwd / up (robot world). Jointly optimises the
        arm (+ ORCA wrist if optimize_wrist) and leaves the scratch at the solution.
        Returns the ARM joint angles only (length n_arm); the ORCA wrist is read back
        from self.ik.qpos[self.orca_wrist_qadr] by the caller, unchanged in both modes.

        Two interchangeable solvers minimise the SAME cost (hyperparams weights):
        "gn" (default) — damped Gauss–Newton with MuJoCo's ANALYTIC Jacobians
        (mj_jac), ~0.5 ms/side vs SLSQP's ~7 ms of numerical finite-differencing
        (measured p50, 2026-07-22 — the retarget frame dropped 33→<15 ms);
        "slsqp" (MABEL_ARM_SOLVER=slsqp) — the scipy reference, kept for parity
        checks.

        TRUST REGION — the solve runs on the FULL joint range from the warm
        start last_q (a local method, so it stays in the CURRENT IK basin — no
        frame-to-frame hop to a distant equal-cost basin, the property the
        trust region protects), and the RESULT is then clipped to
        last_q ± ARM_TRUST_STEP. This is the 2026-07-24 fix for a pervasive
        held-pose LIMIT CYCLE (16/36 static hold poses oscillated ±0.10 rad =
        the trust step; the reported "very unstable"). The old scheme solved
        INSIDE the trust box, so when the basin's local optimum lay outside the
        box the "converged" q was a box EDGE — not a stationary point — and
        that edge flipped sides each frame (the box re-centres on last_q), a
        stable 2-cycle. Clipping the true optimum instead makes the target a
        constant on a static pose, so the arm walks to it monotonically and
        STOPS; a moving target is still rate-limited to ARM_TRUST_STEP/frame.
        The cold first solve searches the full range unclipped."""
        # THE TABLE FIRST: it sets the posture reference the regulariser pulls
        # to, so "naturally extended" is a workspace-aware statement instead of a
        # pull toward qpos0 (which on this robot is BOTH arms straight forward —
        # not a posture a human ever holds).
        self.w_orient = self.w_orient_max      # reset; the lookup relaxes it
        self._tab_q = self._table_lookup(hk)
        self.q_ref = (self.q_nom if self._tab_q is None else
                      self.q_nom + ARM_TABLE_REG * (self._tab_q - self.q_nom))

        if ARM_SOLVER == "slsqp":
            res = minimize(self._cost, self.last_q, args=(hk, self.last_q),
                           method="SLSQP", bounds=list(zip(self.lo, self.hi)),
                           options={"maxiter": ARM_MAXITER, "ftol": 1e-4})
            q = res.x
        else:
            q = self._solve_gn(hk, self.last_q, self.lo, self.hi)
        # Trust-region on the RESULT, not the search: clip the basin-local
        # optimum to a per-frame step so a moving target is rate-limited but a
        # static one converges to a true stationary point (no box-edge cycle).
        self._last_dt = dt              # _basin_escape strides by the same step
        step = self._trust_step(dt)
        if np.any(step > 0.0) and self._warm:
            q = np.clip(q, self.last_q - step, self.last_q + step)
        q = self._table_steer(q, hk)
        q = self._basin_escape(q, hk, self.lo, self.hi)
        self.last_q = q
        self._warm = True
        self.ik.qpos[self.opt_qadr] = q
        if self.lock_j7:
            self.ik.qpos[self.j7_qadr] = self.j7_lock_val
        return q[:self.n_arm]

    # ── the tabular path ──────────────────────────────────────────────────
    def _shoulder_pose(self):
        """(frame R, origin, mount R) of this arm's table frame, from the LIVE
        scratch state — so the lookup follows the lift, the torso and the base
        without the table having to know about any of them."""
        d = self.ik
        Rc = d.xmat[self.chassis_bid].reshape(3, 3)
        Rm = d.xmat[self.shoulder_bid].reshape(3, 3)
        p0 = d.xanchor[self.jids[0]].copy()          # Arm 1 anchor = the shoulder
        return (np.diag([-1.0, -1.0, 1.0]) @ Rc.T @ Rm), p0, Rm

    def _table_lookup(self, hk):
        """The table's posture for this command, UNREFINED, or None.

        Cheap on purpose: one interpolation, no IK. This is the posture the
        regulariser pulls toward every frame, and a reference does not need to
        be exact — refining it here would put ~1 ms of Gauss-Newton on the hot
        path of every frame to improve a term whose weight is 0.02.
        """
        if self.table is None:
            return None
        mujoco.mj_kinematics(self.model, self.ik)
        Rf, p0, Rm = self._shoulder_pose()
        p_s = Rf @ (Rm.T @ (np.asarray(hk["contact"], float) - p0))
        if not self.table.grid.inside(p_s, tol=0.02):
            return None                              # outside the manipulation box
        fwd = Rf @ (Rm.T @ np.asarray(hk["fwd"], float))
        nrm = Rf @ (Rm.T @ np.asarray(hk["up"], float))
        side_ax = np.cross(nrm, fwd)
        n = np.linalg.norm(side_ax)
        if n < 1e-6:
            return None
        R_s = np.column_stack([fwd, side_ax / n, nrm])
        q, feas = self.table.lookup(p_s, R_s)
        # Relax the palm-normal weight where the table says this direction is not
        # deliverable at this point: keep the WRIST, give up the palm angle,
        # rather than spending the arm's whole configuration budget on an angle
        # that is not on offer. See hyperparams.ARM_W_ORIENT.
        self.w_orient = (self.w_orient_min
                         + (self.w_orient_max - self.w_orient_min) * feas)
        self._tab_feas = feas
        return np.clip(q, self.lo, self.hi)

    def _table_steer(self, q, hk):
        """Walk to the table's configuration when it is a clearly better answer.

        This is the palm-flip fix. The warm solve cannot get there on its own:
        the path runs UPHILL — rolling Arm 5 sweeps the ORCA-wrist anchor around
        a cone, so the position residual (weight 600) fights every step of the
        roll, and the orientation residual that wants it was weighted 20 against
        it. Reweighting alone does not fix that: pushed hard enough to win, it
        drags the WRIST off the target on every pose whose palm direction the
        arm cannot reach anyway (measured in W4). Comparing on the TASK cost
        only, and walking through the SAME trust region the tracking path
        already uses, turns an unreachable branch into a continuous,
        rate-limited move instead.

        TWO-STAGE, so this costs nothing on a normal frame. The raw table
        posture is scored first (one forward-kinematics call, ~0.05 ms). It is
        deliberately coarse — sampled on grid points against a canonical finger
        direction — so during ordinary tracking it LOSES to the warm solve and
        the expensive refinement never runs. It only wins when the warm solve is
        in the wrong configuration, which is exactly when the refinement is worth
        paying for. Measured (W7): the whole tabular path adds 0.16 ms to a
        tracking solve (0.68 -> 0.84 ms) and 0.56 ms to a flipping one
        (0.75 -> 1.31 ms) — both arms tracking is 1.7 ms of a 33 ms frame.
        """
        if self._tab_q is None or self._steer_q is not None:
            return q
        c_warm = self._task_cost(q, hk)
        if self._task_cost(self._tab_q, hk) >= c_warm:
            return q                                 # not competitive: no refine
        q_tab = (self._solve_gn(hk, self._tab_q, self.lo, self.hi,
                                iters=ARM_TABLE_REFINE)
                 if ARM_TABLE_REFINE > 0 else self._tab_q)
        if self._task_cost(q_tab, hk) < ARM_TABLE_MARGIN * c_warm and \
                float(np.max(np.abs(q_tab - q))) > 0.05:
            self._steer_q = q_tab
            self._tab_hits += 1
            _s = self._trust_step(self._last_dt)
            step = np.clip(self._steer_q - self.last_q, -_s, _s)
            return np.clip(self.last_q + step, self.lo, self.hi)
        return q

    def _trust_step(self, dt):
        """The per-solve joint step, in rad — a RATE times this solve's interval,
        PER JOINT.

        `dt` is the wall time since the previous solve, i.e. the operator's frame
        interval, so the resulting slew is a rate regardless of how fast the
        client streams. Without a usable dt (the cold path) it falls back to the
        fixed step. ARM_TRUST_MAX bounds a long gap: after a stall the point of
        the trust region — no basin hops — matters more than catching up fast.

        The rate is per-joint because the joints are not alike: the proximal four
        move the whole arm and are what "slams" on a re-anchor, while Arm 5..7
        and the ORCA wrist move a hand. Capping the forearm roll at the
        shoulder's rate is what made a palm flip a 1.6 s crawl.
        """
        if ARM_TRUST_STEP <= 0.0:
            return np.zeros(len(self.lo))    # explicitly disabled
        rate = np.full(len(self.lo), ARM_TRUST_RATE)
        rate[ARM_TRUST_DISTAL_FROM:] = self.trust_rate_distal
        for j, r in getattr(self, "trust_override", {}).items():
            rate[j] = r
        if dt is None or not (dt > 0.0):
            return np.full(len(self.lo), ARM_TRUST_STEP)
        return np.minimum(rate * float(dt), ARM_TRUST_MAX)

    def _basin_escape(self, q, hk, lo, hi):
        """Escape a poor LOCAL basin the warm start walked into (a long teleop
        excursion — walking a loop, sweeping through crossed configurations —
        can strand the redundant arm in a minimum with tens of mm of residual
        position error that the trust region then faithfully preserves; the
        2026-07-22 walk-loop test measured 81 mm). When the TARGET has been
        quasi-static for ARM_ESCAPE_FRAMES frames yet the contact error stays
        above ARM_ESCAPE_ERR, re-solve COLD from the nominal posture over the
        full joint range and, if that basin is genuinely better, STEER toward
        it — the published command is the cold solution clipped into the trust
        box, so the arm walks over continuously (no snap) and converges there
        on subsequent frames. Cold solves run only while triggered (~0.6 ms)."""
        if ARM_ESCAPE_ERR <= 0.0:
            return q
        tgt = np.asarray(hk["contact"], float)
        # STATIC = the target has stayed inside a 1 cm ball of the anchor set
        # when the still-window began — CUMULATIVE, not per-frame. A slow
        # continuous sweep (2–3 mm/frame) passed a per-frame gate, latched the
        # escape mid-motion and froze tracking (exp_real_retarget_chain P6:
        # arm std 0.0033 rad under a commanded 5 cm sweep).
        if self._prev_contact is None or \
                float(np.linalg.norm(tgt - self._prev_contact)) > 0.010:
            self._prev_contact = tgt.copy()        # new anchor: window restarts
            self._stuck = 0
            self._steer_q = None                   # target moved: drop the latch
            return q
        self._stuck += 1
        # LATCHED WALK: a single trust-box step toward a better basin gets
        # pulled straight back by the incumbent minimum on the next warm solve
        # (measured: 13 one-shot steers, zero escapes). Once a better basin is
        # found, walk last_q toward it in trust-bounded joint-space steps EVERY
        # frame until arrival — only then resume warm solving, now inside the
        # canonical basin.
        if self._steer_q is not None:
            _s = self._trust_step(self._last_dt)
            step = np.clip(self._steer_q - self.last_q, -_s, _s)
            q_next = np.clip(self.last_q + step, self.lo, self.hi)
            if float(np.max(np.abs(q_next - self._steer_q))) < 1e-3:
                self._steer_q = None               # arrived
            return q_next
        if self._stuck < ARM_ESCAPE_FRAMES or self._stuck % ARM_ESCAPE_FRAMES:
            return q
        # Compare basins by TOTAL cost with the smoothness term neutralised
        # (q_prev := the candidate itself) — smoothness always favours the
        # incumbent and would veto every escape. The cold seeds are
        # DETERMINISTIC, so steering to the best whenever it is genuinely
        # better makes every quasi-static pose history-independent — the
        # property the walk-loop test asserts.
        # MULTI-SEED (2026-07-29): cold-from-nominal flows to the basin
        # NEAREST the nominal posture, which is not always the good one — the
        # right arm measured 53-100 mm of stuck mid-workspace error while the
        # mirrored left solved to 20 mm, and every escape from nominal walked
        # straight back. A second seed with the ELBOW pre-bent in the arm's
        # natural flexion direction (the dominant side of the J4 range) starts
        # GN inside the elbow-down basin family, which is where human-like
        # solutions live.
        q_bent = self.q_nom.copy()
        if len(q_bent) > 3:
            lo4, hi4 = self.lo[3], self.hi[3]
            q_bent[3] = 0.6 * (lo4 if abs(lo4) > abs(hi4) else hi4)
        cands = [self._solve_gn(hk, seed, self.lo, self.hi, iters=25)
                 for seed in (self.q_nom, q_bent)]
        q_cold = min(cands, key=lambda c: self._cost(c, hk, c))
        c_warm = self._cost(q, hk, q)
        c_cold = self._cost(q_cold, hk, q_cold)
        # Near-equality margin only guards numerical noise (the latch + the
        # static gate already prevent chatter); a wide margin left a ±10 %
        # hysteresis band of history-dependent basins (0.30 rad walk-loop
        # residual measured at 0.9).
        if c_cold < 0.98 * c_warm - 1e-4 \
                and float(np.max(np.abs(q_cold - q))) > 0.05:
            self._steer_q = q_cold
        return q

    def _residuals(self, q, hk, q_prev):
        """Stacked weighted residuals r and analytic Jacobian J (rows × n) at q,
        with Σr² equal to the SLSQP cost (1−cos terms via ½|â−b̂|²). One
        mj_kinematics + mj_comPos + three mj_jac per call — no finite differences."""
        m, d = self.model, self.ik
        d.qpos[self.opt_qadr] = q
        if self.lock_j7:
            d.qpos[self.j7_qadr] = self.j7_lock_val
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)                                # cdof for mj_jac
        n = len(q)
        cols = self.opt_dofs
        rows_r, rows_J = [], []

        # CONTACT position (the ORCA-wrist anchor in ARM_TRACK_WRIST mode; the gripper base face with the gripper).
        wrist, wbody = self._ee_point(d)
        Rp = d.xmat[self.palm_bid].reshape(3, 3) @ self.R_rob
        mujoco.mj_jac(m, d, self._jacp, self._jacr, wrist, wbody)
        Jc = self._jacp[:, cols].copy()
        contact = wrist.copy()
        if not ARM_TRACK_WRIST:
            contact = wrist + HAND_REACH * Rp[:, 0]
            mujoco.mj_jac(m, d, self._jacp, self._jacr, contact, self.palm_bid)
            Jc = self._jacp[:, cols].copy()
        sw = np.sqrt(ARM_W_WRIST)
        rows_r.append(sw * (contact - hk["contact"]))
        rows_J.append(sw * Jc)

        # ELBOW direction from the shoulder (mount is upstream of the chain, so
        # only the elbow endpoint moves with the arm DoFs).
        if hk.get("elbow") is not None:
            e = d.xpos[self.elbow_bid]
            sh = d.xpos[self.shoulder_bid]
            dv = e - sh
            dist = np.linalg.norm(dv)
            if dist > 1e-9:
                ed = dv / dist
                hed = hk["elbow"] - sh
                hed = hed / (np.linalg.norm(hed) + 1e-9)
                mujoco.mj_jac(m, d, self._jacp, None, e, self.elbow_bid)
                P = (np.eye(3) - np.outer(ed, ed)) / dist
                sw = np.sqrt(self._w_elbow(hk) / 2.0)         # 1−cos = ½|ê−ĥ|²
                rows_r.append(sw * (ed - hed))
                rows_J.append(sw * (P @ self._jacp[:, cols]))

        # PALM orientation axes: d(R·a)/dq = Jr × (R·a) column-wise.
        mujoco.mj_jac(m, d, None, self._jacr, d.xpos[self.palm_bid], self.palm_bid)
        Jr = self._jacr[:, cols]
        for axis, key, w in ((Rp[:, 0], "fwd", self.w_orient_fwd),
                             (Rp[:, 2], "up", self.w_orient)):
            sw = np.sqrt(w / 2.0)                     # 1-cos = 1/2 |a_hat - b_hat|^2
            rows_r.append(sw * (axis - hk[key]))
            rows_J.append(sw * np.cross(Jr.T, axis).T)

        # smoothness + posture regularisation
        sws = np.sqrt(ARM_W_SMOOTH)
        rows_r.append(sws * (q - q_prev))
        rows_J.append(sws * np.eye(n))
        swr = np.sqrt(self._reg_for(hk))
        rows_r.append(swr * (q - self.q_ref))
        rows_J.append(np.diag(swr))

        r = np.concatenate([np.atleast_1d(x).ravel() for x in rows_r])
        J = np.vstack(rows_J)
        return r, J

    def _solve_gn(self, hk, q_prev, lo, hi, iters=None):
        """Projected damped Gauss–Newton on the arm cost (analytic Jacobians).
        Fixed damping, every step accepted (projection = clip to bounds):
        accept/reject Levenberg stalls exactly at ACTIVE joint bounds — the
        clipped step reads as a cost increase, λ explodes and full extension
        (elbow AT its limit) dies at rho 0.79 (measured). Fixed-damping
        projected GN reaches the same optimum SLSQP does on this cost
        (verified identical q on static targets); iteration count is the only
        convergence knob, and each iteration is one analytic _residuals call
        (~0.08 ms) — no finite differences anywhere."""
        if iters is None:
            iters = ARM_GN_ITERS
        q = np.clip(np.asarray(q_prev, float).copy(), lo, hi)
        n = len(q)
        I = ARM_GN_DAMPING * np.eye(n)
        for _ in range(iters):
            r, J = self._residuals(q, hk, q_prev)
            H = J.T @ J + I
            g = J.T @ r
            dq = np.linalg.solve(H, -g)
            # ACTIVE-SET pass: when the raw step exits the trust/joint box, naive
            # clipping also kills the step's TANGENTIAL component (the clipped-GN
            # zig-zag — measured 12.4° circle-direction lag vs SLSQP's 1.3° at any
            # iteration count). Fix the box-hitting coordinates AT their bound and
            # re-solve the reduced normal equations for the free ones conditioned
            # on that displacement, so the solution slides ALONG the active face.
            q_new = np.clip(q + dq, lo, hi)
            act = (q_new <= lo + 1e-12) | (q_new >= hi - 1e-12)
            for _pass in range(3):                 # set can grow as frees clip
                if not act.any() or act.all():
                    break
                free = ~act
                dq_a = q_new[act] - q[act]
                rhs = -(g[free] + H[np.ix_(free, act)] @ dq_a)
                dq_f = np.linalg.solve(H[np.ix_(free, free)], rhs)
                q_new[free] = np.clip(q[free] + dq_f, lo[free], hi[free])
                act_new = (q_new <= lo + 1e-12) | (q_new >= hi - 1e-12)
                if (act_new == act).all():
                    break
                act = act_new
            step = float(np.max(np.abs(q_new - q)))
            q = q_new
            if step < 1e-5:
                break
        return q
