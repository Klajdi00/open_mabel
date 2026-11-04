"""Front-workspace posture table — the tabular basis for arm retargeting.

WHY A TABLE
-----------
The live arm IK (``arm.py``) is a warm-started LOCAL solve: every frame it
descends the same cost from the previous answer.  That is exactly right for
TRACKING and exactly wrong for BRANCH SELECTION, and on a 7-DoF arm chasing a
6-DoF task the two are not separable.

The failure the operator reports as "the wrist flip is extremely bad" is a
direct consequence.  Rolling the palm from down to up sweeps ``Arm 5`` (the
forearm roll) across its whole +/-90 deg range.  ``Arm 5`` is NOT collinear
with the ORCA-wrist anchor, so rolling it sweeps that anchor around a cone —
the POSITION residual (weight 600) therefore fights the roll, while the
orientation residual that wants it is weighted 20.  A 30:1 fight with a cosine
whose gradient vanishes at exactly 180 deg: the solver settles half-flipped (or
not flipped at all) with a perfect wrist position, and ``_basin_escape`` never
fires because it only ever watches POSITION error.

A local method cannot fix that, because the path to the good configuration runs
UPHILL.  So the global structure is precomputed instead: for a grid over the
robot's front manipulation volume, crossed with a family of palm rolls, solve
the pose GLOBALLY (multi-seed, full joint range, no trust region) under a
human-arm posture prior, and make the resulting field COHERENT so neighbouring
cells hold neighbouring joint vectors.  The live solver then only ever has to
track: the table tells it which configuration the pose belongs to, and the
trust region walks there continuously.

THE FRAME
---------
The table is indexed in the SHOULDER frame: origin at the arm mount, axes
forward / left / up at the nominal torso angle.  That makes it exactly
invariant to the lift, to the base pose and to the base heading (all upstream
of the mount) and it rotates with the torso, which is the correct behaviour —
the arm's reachable set rotates with the torso too.

THE POSTURE PRIOR
-----------------
A 7-DoF arm on a 6-DoF task has a 1-D null space: the elbow SWIVEL about the
shoulder-wrist axis.  Position and orientation say nothing about it, so
"natural, human-like extension" IS the choice of swivel — nothing else.  The
prior is the classical human-arm model (Korein; Kondo): the elbow hangs down
under gravity and slightly OUTWARD from the shoulder-wrist axis.  It enters the
cost as an elbow POSITION residual (analytic Jacobian) rather than an angle, so
it costs one ``mj_jac`` and no finite differences.

Elbow FLEXION is not a free choice at all — the law of cosines fixes it from
the shoulder-to-wrist distance — so "the arm extends like a human's" follows
from tracking the wrist position, provided the solver is not allowed to trade
position away for posture.  It is not: the position weight dominates.
"""
import hashlib
import os
import warnings

import numpy as np
import mujoco

__all__ = ["ShoulderFrame", "ArmChain", "PostureSolver", "WorkspaceGrid",
           "model_fingerprint", "WorkspaceTable", "natural_elbow",
           "sew_reference", "palm_target", "PALM_ROLLS", "PALM_ORDER",
           "PALM_ALL", "DEFAULT_TABLE_DIR", "table_path"]


# ── where built tables live ────────────────────────────────────────────────────
DEFAULT_TABLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tables")


def table_path(side, directory=None):
    return os.path.join(directory or DEFAULT_TABLE_DIR, f"front_workspace_{side}.npz")


# ── small rotation helpers ────────────────────────────────────────────────────
def _skew(v):
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def rot_log(R):
    """Axis-angle vector of a rotation matrix (the residual the GN solve uses).

    Branch-safe at theta -> pi: the near-antipodal case is exactly the wrist
    flip this module exists for, and the naive skew-part formula degenerates
    there (it is proportional to sin(theta), so it reports ~0 error for a
    180 deg mismatch — which is precisely how the shipped cosine cost lost the
    flip). Fall back to the symmetric part, which stays well conditioned.
    """
    R = np.asarray(R, float)
    c = 0.5 * (np.trace(R) - 1.0)
    c = float(np.clip(c, -1.0, 1.0))
    th = float(np.arccos(c))
    if th < 1e-8:
        return np.zeros(3)
    if th < np.pi - 1e-3:
        w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        return (th / (2.0 * np.sin(th))) * w
    # theta ~ pi: axis from the largest diagonal of (R + I)/2 = a a^T
    A = 0.5 * (R + np.eye(3))
    k = int(np.argmax(np.diag(A)))
    a = A[:, k] / np.sqrt(max(A[k, k], 1e-12))
    a /= (np.linalg.norm(a) + 1e-12)
    # sign is genuinely ambiguous at exactly pi; pick the branch that agrees
    # with the (tiny) skew part so the residual stays continuous through the flip
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if float(np.dot(a, w)) < 0.0:
        a = -a
    return th * a


def rot_axis(axis, ang):
    axis = np.asarray(axis, float)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    K = _skew(axis)
    return np.eye(3) + np.sin(ang) * K + (1.0 - np.cos(ang)) * (K @ K)


def orient_err(R_cur, R_tgt):
    """Angle [rad] between two frames."""
    return float(np.linalg.norm(rot_log(np.asarray(R_cur) @ np.asarray(R_tgt).T)))


# ── the palm target family the operator actually asks for ─────────────────────
#: roll of the palm about its own forward (finger) axis, measured so that
#: phi = 0 puts the PALMAR normal DOWN for a level reach. These four are the
#: poses the request names: palm down / inward / up / outward.
PALM_ROLLS = {"down": 0.0, "inward": 0.5 * np.pi, "up": np.pi, "outward": 1.5 * np.pi}

#: THE THREE DIRECTIONS THAT MATTER (operator, 2026-08-23: "just focus on the
#: palm up, down and inwards — make sure all 3 work perfectly; right now when
#: facing outwards the arms will go very erratic").
#:
#: This is a scope decision, and it is also the honest one. Palm-OUTWARD asks the
#: arm to rotate the palm away from the robot's midline, which on this arm means
#: driving the single forearm roll (Arm 5, +/-90 deg) to its end stop while the
#: wrist stays put. Measured over the front box it is the least achievable of the
#: four (76.6 % against 94.2/92.0 for down/up), and where it is not achievable
#: the solver spends the arm's whole configuration budget chasing it — which is
#: what "erratic" looks like from outside. It is also the direction with the
#: least manipulation value: a hand reaching out of the robot's own workspace.
#:
#: The table still covers the FULL roll circle (the operator can roll their palm
#: anywhere and must get something sane back); this tuple is what the system is
#: optimised and reported against.
PALM_ORDER = ("down", "inward", "up")
PALM_ALL = ("down", "inward", "up", "outward")


def palm_target(fwd, roll, side, up=(0.0, 0.0, 1.0)):
    """Palm frame [f | s | n] (columns) for a finger direction and a roll.

    ``n`` is the PALMAR normal (the direction the fingers curl toward), which is
    how both the robot frame (``HandOptimizer.R_rob``, flexion-anchored) and the
    operator frame (``geometry.palm_axes``, left normal flipped) are defined —
    so a target built here is directly comparable to both.

    roll = 0 puts the palm normal DOWN; +90 deg rolls it INWARD (toward the
    robot's midline) for either hand, so the same roll number means the same
    ANATOMICAL pose on the left and the right.
    """
    f = np.asarray(fwd, float)
    f = f / (np.linalg.norm(f) + 1e-12)
    up = np.asarray(up, float)
    # the "down" reference normal: gravity-down, orthogonalised against f
    n0 = -up + float(np.dot(up, f)) * f
    if np.linalg.norm(n0) < 1e-6:                 # f is vertical: use body-forward
        n0 = np.array([1.0, 0.0, 0.0]) - float(np.dot([1.0, 0.0, 0.0], f)) * f
    n0 /= (np.linalg.norm(n0) + 1e-12)
    inward = -1.0 if side == "left" else 1.0      # +y is left, so inward is -y on the left
    n = rot_axis(f, inward * float(roll)) @ n0
    s = np.cross(n, f)
    s /= (np.linalg.norm(s) + 1e-12)
    return np.column_stack([f, s, n])


# ── the human-arm swivel prior ────────────────────────────────────────────────
#
# A 7-DoF arm on a 6-DoF task has a 1-D null space: the elbow SWIVEL about the
# shoulder-wrist axis. Position and orientation say nothing about it, so
# "natural, human-like extension" IS the choice of swivel — nothing else. Elbow
# FLEXION is not a choice at all; the law of cosines fixes it from the
# shoulder-to-wrist distance.
#
# THE REFERENCE DIRECTION, AND WHY IT IS NOT JUST "DOWN".
#
# The swivel is measured from a reference direction perpendicular to the
# shoulder-wrist axis. The obvious choice — gravity, orthogonalised against that
# axis — is the CONVENTIONAL shoulder-elbow-wrist (SEW) angle, and it has a
# known algorithmic singularity: when the arm points along the reference
# direction the projection vanishes and the swivel is undefined. Elias and Wen
# (Mech. Mach. Theory 2024, arXiv:2307.13122) show such a singularity is
# UNAVOIDABLE for any parameterisation — it is the hairy-ball theorem — but that
# where it sits is a choice, and their STEREOGRAPHIC SEW angle places it on a
# half-line that can be put out of reach, rather than on a full line through the
# workspace.
#
# That matters here and not in the abstract: gravity-down is singular when the
# arm hangs straight down, and the bottom of MABEL's front manipulation box
# (x = 0.25 m, z = -0.40 m from the shoulder) is 58 deg from straight down. The
# prior was ill-conditioned inside the volume it exists to shape.
#
# The construction below is the single-singularity field: parallel-transport a
# reference vector from a chosen pole along the geodesic to the current arm
# direction. It is continuous everywhere except at the ANTIPODE of the pole. Put
# the pole at "straight forward" and the singularity lands at "straight
# backwards", which no manipulation pose uses.
SEW_POLE = np.array([1.0, 0.0, 0.0])      # +x forward: reaching out in front
SEW_REF = np.array([0.0, 0.0, -1.0])      # elbow hangs DOWN on a level forward reach


def sew_reference(u, pole=None, ref=None):
    """Stereographic SEW reference direction for arm axis ``u`` (unit).

    Singular only at ``u == -pole`` (straight backwards), never inside the front
    workspace. At ``u == pole`` it returns ``ref`` exactly, so a level forward
    reach still means "elbow straight down" — the convention the conventional
    SEW angle gets right and everything else was tuned against.
    """
    u = np.asarray(u, float)
    u = u / (np.linalg.norm(u) + 1e-12)
    pole = SEW_POLE if pole is None else np.asarray(pole, float)
    ref = SEW_REF if ref is None else np.asarray(ref, float)
    c = float(np.clip(np.dot(pole, u), -1.0, 1.0))
    ax = np.cross(pole, u)
    n = np.linalg.norm(ax)
    if n < 1e-9:                                  # u == +/- pole
        return ref.copy() if c > 0 else -ref.copy()
    R = rot_axis(ax / n, float(np.arctan2(n, c)))
    r = R @ ref
    r -= float(np.dot(r, u)) * u                  # numerical re-orthogonalisation
    return r / (np.linalg.norm(r) + 1e-12)


def natural_elbow(shoulder, wrist, l_upper, l_fore, side, swivel_out=0.35,
                  up=(0.0, 0.0, 1.0), stereographic=True):
    """Where a relaxed human elbow sits for this shoulder/wrist pair.

    The elbow lies on a circle whose radius the law of cosines fixes; the only
    freedom is the swivel around the shoulder-wrist axis. A relaxed human arm
    hangs the elbow DOWN and slightly OUTWARD — that, and nothing else, is what
    "the arms look naturally extended" means for a 7-DoF arm tracking a 6-DoF
    pose.

    Returns None when the pair is unreachable (outside the two-link annulus), in
    which case the caller drops the prior rather than inventing one.
    """
    S = np.asarray(shoulder, float)
    W = np.asarray(wrist, float)
    v = W - S
    d = float(np.linalg.norm(v))
    if d < 1e-6 or d > l_upper + l_fore or d < abs(l_upper - l_fore):
        return None
    u = v / d
    a = (l_upper ** 2 - l_fore ** 2 + d ** 2) / (2.0 * d)
    rho2 = l_upper ** 2 - a ** 2
    if rho2 <= 1e-9:
        return None
    rho = np.sqrt(rho2)
    C = S + a * u
    if stereographic:
        n = sew_reference(u)
    else:                                          # conventional SEW, for the A/B
        g = -np.asarray(up, float)
        n = g - float(np.dot(g, u)) * u
        nn = np.linalg.norm(n)
        if nn < 1e-6:
            return None                            # THE singularity this replaces
        n /= nn
    b = np.cross(u, n)                             # +b is the operator's LEFT
    out = 1.0 if side == "left" else -1.0          # outward = away from the midline
    psi = out * float(swivel_out)
    return C + rho * (np.cos(psi) * n + np.sin(psi) * b)


# ── the arm chain (FK + analytic Jacobians on a private scratch state) ────────
class ShoulderFrame:
    """Origin at the arm mount, axes forward / left / up at nominal torso.

    Stores the constant rotation from the MOUNT body frame into those axes, so
    a live query only needs the mount's current world pose.
    """

    def __init__(self, R_mount_to_frame):
        self.R = np.asarray(R_mount_to_frame, float)

    def point(self, p_world, mount_pos, mount_R):
        return self.R @ (mount_R.T @ (np.asarray(p_world, float) - mount_pos))

    def frame(self, R_world, mount_R):
        return self.R @ (mount_R.T @ np.asarray(R_world, float))

    def to_world_point(self, p, mount_pos, mount_R):
        return mount_R @ (self.R.T @ np.asarray(p, float)) + mount_pos

    def to_world_frame(self, R, mount_R):
        return mount_R @ (self.R.T @ np.asarray(R, float))


class ArmChain:
    """One arm's kinematics: the 7 OpenArm joints plus the ORCA wrist pitch.

    All FK runs on a PRIVATE ``MjData`` seeded from the live state, so nothing
    here can perturb the simulator or read privileged state — the same contract
    ``ArmRetargeter`` already honours.
    """

    def __init__(self, model, side, lock_j7=False, optimize_wrist=True,
                 j7_yaw_lim=None):
        from .hyperparams import ARM_J7_YAW_LIM
        from .hand import HandOptimizer

        self.model = model
        self.side = side
        S = side.capitalize()
        self.data = mujoco.MjData(model)
        self.lock_j7 = bool(lock_j7)
        self.j7_yaw_lim = j7_yaw_lim
        n_arm = 6 if lock_j7 else 7
        self.n_arm = n_arm

        def J(name):
            return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)

        def B(name):
            return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)

        self.arm_jids = [J(f"{S} Arm {i}") for i in range(1, n_arm + 1)]
        self.wrist_jid = J(f"{S} Wrist")
        self.opt_jids = list(self.arm_jids) + ([self.wrist_jid] if optimize_wrist else [])
        self.optimize_wrist = bool(optimize_wrist)
        self.qadr = np.array([model.jnt_qposadr[j] for j in self.opt_jids])
        self.dofs = np.array([model.jnt_dofadr[j] for j in self.opt_jids])
        self.n = len(self.opt_jids)

        self.lo = model.jnt_range[self.opt_jids, 0].copy()
        self.hi = model.jnt_range[self.opt_jids, 1].copy()
        self.q_nom = model.qpos0[self.qadr].copy()
        if not lock_j7:                       # anatomical wrist-yaw clamp (see hyperparams)
            lim = ARM_J7_YAW_LIM if j7_yaw_lim is None else float(j7_yaw_lim)
            self.lo[6] = max(self.lo[6], self.q_nom[6] - lim)
            self.hi[6] = min(self.hi[6], self.q_nom[6] + lim)
        j7 = J(f"{S} Arm 7")
        self.j7_qadr = model.jnt_qposadr[j7]
        self.j7_lock_val = float(model.qpos0[self.j7_qadr])

        self.mount_bid = B("l_arm_mount" if side == "left" else "r_arm_mount")
        self.shoulder_jid = self.arm_jids[0]      # Arm 1: the shoulder's first axis
        self.elbow_bid = B("l_elbow" if side == "left" else "r_elbow")
        self.palm_bid = B(f"{side}_palm")
        self.chassis_bid = B("mabel_chassis")
        self.R_rob = HandOptimizer(model, side).R_rob      # flexion-anchored palm frame

        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

        self.sync(None)
        self.l_upper = float(np.linalg.norm(self.elbow() - self.shoulder()))
        self.l_fore = float(np.linalg.norm(self.wrist() - self.elbow()))
        self.frame = self._build_frame()

    # -- state -----------------------------------------------------------------
    def sync(self, data=None):
        """Load the live robot pose (base / lift / torso / neck) into the scratch."""
        if data is not None:
            self.data.qpos[:] = data.qpos
        if self.lock_j7:
            self.data.qpos[self.j7_qadr] = self.j7_lock_val
        mujoco.mj_kinematics(self.model, self.data)
        # ORIGIN = the Arm 1 joint ANCHOR, not the mount body's origin. The two
        # coincide on the left arm but sit 6 cm apart on the right (the MJCF's
        # r_arm_mount body origin is offset), and using the body origin makes the
        # table's own frame asymmetric — every left/right comparison then reads
        # as a retargeting defect that is really a frame offset.
        self.mount_pos = self.data.xanchor[self.shoulder_jid].copy()
        self.mount_R = self.data.xmat[self.mount_bid].reshape(3, 3).copy()

    def _build_frame(self):
        """Rotation from the MOUNT frame to forward/left/up at the current torso."""
        Rc = self.data.xmat[self.chassis_bid].reshape(3, 3)
        # chassis FRONT is -x (see hyperparams.R_BASE_ALIGN), so body = chassis
        # turned 180 deg about z
        Bm = np.diag([-1.0, -1.0, 1.0])
        # R maps MOUNT-local vectors into forward/left/up. (Do not transpose: on
        # the LEFT arm this matrix happens to be symmetric, so a transposed frame
        # is silently correct there and wrong only on the right — a mirror-image
        # bug that reads as "the right arm is the broken one".)
        return ShoulderFrame(Bm @ Rc.T @ self.mount_R)

    def set_q(self, q):
        self.data.qpos[self.qadr] = q
        if self.lock_j7:
            self.data.qpos[self.j7_qadr] = self.j7_lock_val
        mujoco.mj_kinematics(self.model, self.data)

    def q(self):
        return self.data.qpos[self.qadr].copy()

    # -- FK (world) ------------------------------------------------------------
    def wrist(self):
        return self.data.xanchor[self.wrist_jid].copy()

    def elbow(self):
        return self.data.xpos[self.elbow_bid].copy()

    def shoulder(self):
        return self.data.xanchor[self.shoulder_jid].copy()

    def palm(self):
        return self.data.xmat[self.palm_bid].reshape(3, 3) @ self.R_rob

    # -- FK (shoulder frame) ---------------------------------------------------
    def wrist_s(self):
        return self.frame.point(self.wrist(), self.mount_pos, self.mount_R)

    def elbow_s(self):
        return self.frame.point(self.elbow(), self.mount_pos, self.mount_R)

    def palm_s(self):
        return self.frame.frame(self.palm(), self.mount_R)

    def pose_s(self, q):
        """(wrist, palm frame, elbow) in the shoulder frame for joints q."""
        self.set_q(q)
        return self.wrist_s(), self.palm_s(), self.elbow_s()


# ── the global posture solve ──────────────────────────────────────────────────
class PostureSolver:
    """Projected damped Gauss-Newton on [position | orientation | posture].

    Residual stack, all with ANALYTIC Jacobians (one ``mj_kinematics`` +
    ``mj_comPos`` + three ``mj_jac`` per iteration, no finite differences):

      * wrist position   w_p * (p(q) - p*)                              [3]
      * palm TILT        w_o * perp part of log(R(q) R*^T)              [3]
      * palm SPIN        w_f * part along the palm normal               [-]
      * natural elbow    w_e * (e(q) - e*)   (the swivel prior)         [3]
      * joint reg        w_reg * (q - q_ref)                            [n]
      * continuity       w_c * (q - q_seed)  (table coherence only)     [n]

    The orientation residual is the axis-angle vector, NOT a cosine: a cosine
    is stationary at 180 deg, which is exactly the palm flip, and that is how
    the flip gets lost.  Bounds are handled by the same active-set projection
    ``arm.py`` uses (a naive clip kills the step's tangential component).

    ORIENTATION IS NOT ISOTROPIC.  "Palm facing up / down / in / out" is a
    constraint on the palm NORMAL — two degrees of freedom.  The third, the
    SPIN about that normal (which way the fingers point within the palm plane),
    is a much weaker preference: a human reaching to a point with the palm up
    picks whatever finger azimuth is comfortable.  Weighting all three equally
    is what makes most of the front volume read as "unreachable": MABEL has ONE
    forearm roll (Arm 5, +/-90 deg) and a +/-45 deg wrist pitch, so demanding an
    exact finger azimuth on top of an exact palm normal over-constrains a 7-DoF
    arm that also has to hold a position.  Splitting them (w_ori on the tilt,
    w_spin on the spin) buys back the volume the operator actually asked for.
    """

    #: w_elbow = 16 is measured, and the measurement is counter-intuitive enough
    #: to write down: a table built at w_elbow = 8 SOLVES MORE OF ITS OWN CELLS
    #: (84.4 % against 82.3 %) and is WORSE everywhere it is actually used —
    #: end-to-end poses achieved 72.2 % vs 74.1 %, flip settle p90 1.03 s vs
    #: 0.71 s, joint path 4.26 vs 4.09 rad, and elbow-vs-natural 45.4 vs 37.6 mm
    #: with no elbow hint. The table is a REFERENCE, not an answer sheet: what
    #: the live solver inherits from it is posture consistency, and a table that
    #: bought extra cells by letting each one pick its own swivel has less of
    #: that to give.
    def __init__(self, chain, w_pos=600.0, w_ori=150.0, w_spin=12.0,
                 w_elbow=16.0, w_reg=None, damping=1e-2, iters=60):
        self.chain = chain
        self.w_pos = float(w_pos)
        self.w_ori = float(w_ori)
        self.w_spin = float(w_spin)
        self.w_elbow = float(w_elbow)
        n = chain.n
        if w_reg is None:
            # light everywhere; the posture prior is carried by the ELBOW term,
            # not by pinning joints. Arm 7 (wrist yaw) and the ORCA wrist keep a
            # real pull toward rest because a human barely yaws the wrist and
            # the ORCA joint should return to neutral when nothing needs it.
            w_reg = np.full(n, 0.02)
            if not chain.lock_j7 and n >= 7:
                w_reg[6] = 0.25
            if chain.optimize_wrist:
                w_reg[-1] = 0.05
        self.w_reg = np.asarray(w_reg, float)
        self.damping = float(damping)
        self.iters = int(iters)

    # -- cost / residuals ------------------------------------------------------
    def targets(self, p_star, R_star, elbow_star=None, q_ref=None, q_seed=None,
                w_cont=0.0):
        return dict(p=np.asarray(p_star, float), R=np.asarray(R_star, float),
                    e=None if elbow_star is None else np.asarray(elbow_star, float),
                    q_ref=self.chain.q_nom if q_ref is None else np.asarray(q_ref, float),
                    q_seed=None if q_seed is None else np.asarray(q_seed, float),
                    w_cont=float(w_cont))

    def _residuals(self, q, tg):
        ch = self.chain
        m, d = ch.model, ch.data
        ch.set_q(q)
        mujoco.mj_comPos(m, d)                       # cdof for mj_jac
        cols = ch.dofs
        Rf = ch.frame.R @ ch.mount_R.T               # world -> shoulder frame
        rows_r, rows_J = [], []

        # wrist position
        w = ch.wrist()
        mujoco.mj_jac(m, d, ch._jacp, None, w, int(m.jnt_bodyid[ch.wrist_jid]))
        s = np.sqrt(self.w_pos)
        rows_r.append(s * (ch.frame.point(w, ch.mount_pos, ch.mount_R) - tg["p"]))
        rows_J.append(s * (Rf @ ch._jacp[:, cols]))

        # palm orientation, split into TILT (align the palm normal) and SPIN
        # (rotate about it). Axis-angle, not a cosine: a cosine is stationary at
        # 180 deg, which is exactly the palm flip this module exists for.
        mujoco.mj_jac(m, d, None, ch._jacr, d.xpos[ch.palm_bid], ch.palm_bid)
        Jr = Rf @ ch._jacr[:, cols]
        R_cur = ch.palm_s()
        e = rot_log(R_cur @ tg["R"].T)
        nhat = tg["R"][:, 2]                          # target PALMAR normal
        P_spin = np.outer(nhat, nhat)
        P_tilt = np.eye(3) - P_spin
        s = np.sqrt(self.w_ori)
        rows_r.append(s * (P_tilt @ e))
        rows_J.append(s * (P_tilt @ Jr))
        if self.w_spin > 0.0:
            s = np.sqrt(self.w_spin)
            rows_r.append(s * (P_spin @ e))
            rows_J.append(s * (P_spin @ Jr))

        # natural elbow (the swivel prior)
        if tg["e"] is not None and self.w_elbow > 0.0:
            e = ch.elbow()
            mujoco.mj_jac(m, d, ch._jacp, None, e, ch.elbow_bid)
            s = np.sqrt(self.w_elbow)
            rows_r.append(s * (ch.frame.point(e, ch.mount_pos, ch.mount_R) - tg["e"]))
            rows_J.append(s * (Rf @ ch._jacp[:, cols]))

        # joint regularisation + optional continuity toward the seed
        s = np.sqrt(self.w_reg)
        rows_r.append(s * (q - tg["q_ref"]))
        rows_J.append(np.diag(s))
        if tg["q_seed"] is not None and tg["w_cont"] > 0.0:
            s = np.sqrt(tg["w_cont"])
            rows_r.append(s * (q - tg["q_seed"]))
            rows_J.append(s * np.eye(ch.n))

        r = np.concatenate([np.atleast_1d(x).ravel() for x in rows_r])
        return r, np.vstack(rows_J)

    def cost(self, q, tg):
        r, _ = self._residuals(q, tg)
        return float(r @ r)

    # -- solve -----------------------------------------------------------------
    def solve(self, q0, tg, iters=None):
        ch = self.chain
        lo, hi = ch.lo, ch.hi
        q = np.clip(np.asarray(q0, float).copy(), lo, hi)
        I = self.damping * np.eye(ch.n)
        for _ in range(int(self.iters if iters is None else iters)):
            r, Jm = self._residuals(q, tg)
            H = Jm.T @ Jm + I
            g = Jm.T @ r
            dq = np.linalg.solve(H, -g)
            q_new = np.clip(q + dq, lo, hi)
            act = (q_new <= lo + 1e-12) | (q_new >= hi - 1e-12)
            for _p in range(3):                       # active-set: slide along the face
                if not act.any() or act.all():
                    break
                free = ~act
                rhs = -(g[free] + H[np.ix_(free, act)] @ (q_new[act] - q[act]))
                q_new[free] = np.clip(q[free] + np.linalg.solve(H[np.ix_(free, free)], rhs),
                                      lo[free], hi[free])
                act_new = (q_new <= lo + 1e-12) | (q_new >= hi - 1e-12)
                if (act_new == act).all():
                    break
                act = act_new
            step = float(np.max(np.abs(q_new - q)))
            q = q_new
            if step < 1e-6:
                break
        return q

    def solve_multi(self, seeds, tg, iters=None):
        """Best of several cold starts — this is what makes the table GLOBAL."""
        best, best_c = None, np.inf
        for s in seeds:
            if s is None:
                continue
            q = self.solve(s, tg, iters=iters)
            c = self.cost(q, tg)
            if c < best_c:
                best, best_c = q, c
        return best, best_c

    def errors(self, q, p_star, R_star):
        """(position [m], palm-TILT [rad], palm-SPIN [rad]) error of a solution.

        Tilt and spin are reported separately for the same reason the cost
        weights them separately: "the palm faces the wrong way" (tilt) and "the
        fingers point somewhere else in the palm plane" (spin) are different
        failures, and lumping them into one angle hides which one happened.
        """
        p, R, _ = self.chain.pose_s(q)
        R_star = np.asarray(R_star, float)
        tilt = float(np.arccos(np.clip(float(np.dot(R[:, 2], R_star[:, 2])), -1.0, 1.0)))
        e = rot_log(R @ R_star.T)
        spin = abs(float(np.dot(e, R_star[:, 2])))
        return float(np.linalg.norm(p - np.asarray(p_star, float))), tilt, spin


# ── the grid ──────────────────────────────────────────────────────────────────
class WorkspaceGrid:
    """A regular box of wrist positions in the SHOULDER frame, crossed with a
    family of palm rolls about a per-cell canonical finger direction.

    The finger direction is not a table axis: it is fixed per cell to the
    natural "fingers point where the arm reaches" convention (the horizontal
    radial from the shoulder), and the live solver corrects whatever the
    operator's actual finger direction is from the seed the table hands it.
    Roll IS an axis because roll is where the branch structure lives.
    """

    def __init__(self, x, y, z, rolls=8, side="right"):
        self.x, self.y, self.z = (np.asarray(a, float) for a in (x, y, z))
        self.rolls = (np.linspace(0.0, 2.0 * np.pi, int(rolls), endpoint=False)
                      if np.isscalar(rolls) else np.asarray(rolls, float))
        self.side = side

    @property
    def shape(self):
        return (len(self.x), len(self.y), len(self.z), len(self.rolls))

    def point(self, i, j, k):
        return np.array([self.x[i], self.y[j], self.z[k]])

    def canonical_fwd(self, p):
        """Fingers point along the horizontal radial from the shoulder."""
        f = np.array([p[0], p[1], 0.0])
        n = np.linalg.norm(f)
        return np.array([1.0, 0.0, 0.0]) if n < 1e-6 else f / n

    def target(self, i, j, k, r):
        p = self.point(i, j, k)
        return p, palm_target(self.canonical_fwd(p), self.rolls[r], self.side)

    def locate(self, p):
        """Fractional grid index of a shoulder-frame point (clamped to the box)."""
        out = []
        for v, ax in zip(np.asarray(p, float), (self.x, self.y, self.z)):
            if len(ax) == 1:
                out.append(0.0)
                continue
            t = (v - ax[0]) / (ax[-1] - ax[0]) * (len(ax) - 1)
            out.append(float(np.clip(t, 0.0, len(ax) - 1)))
        return np.array(out)

    def inside(self, p, tol=1e-6):
        p = np.asarray(p, float)
        return bool(np.all(p >= np.array([self.x[0], self.y[0], self.z[0]]) - tol) and
                    np.all(p <= np.array([self.x[-1], self.y[-1], self.z[-1]]) + tol))


# ── the table ─────────────────────────────────────────────────────────────────
def model_fingerprint(model, side, R_rob, lock_j7=False, optimize_wrist=True,
                      j7_yaw_lim=None):
    """A short hash of everything about the arm the table depends on.

    The table is a DERIVED artefact of the canonical MJCF, exactly like the
    motion model's CoM constants — and it goes stale the same silent way. A
    changed joint limit, link length or rest pose invalidates every cell, and the
    failure mode is not a crash: it is a slightly wrong posture that reads as
    "the retargeting got worse again". So the table carries the fingerprint of
    the model it was built against and the loader refuses a mismatch.

    Computed from the model plus the caller's own ``R_rob`` — deliberately NOT
    from an ``ArmChain``, so the live retargeter can check a table without
    building one (an ArmChain constructs a HandOptimizer, which is ~100 ms and
    would be paid on every construction).
    """
    from .hyperparams import ARM_J7_YAW_LIM

    S = side.capitalize()

    def J(name):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)

    n_arm = 6 if lock_j7 else 7
    jids = [J(f"{S} Arm {i}") for i in range(1, n_arm + 1)]
    if optimize_wrist:
        jids = jids + [J(f"{S} Wrist")]
    qadr = [model.jnt_qposadr[j] for j in jids]
    rng = model.jnt_range[jids].copy()
    if not lock_j7:                                  # the clamp is part of the build
        lim = ARM_J7_YAW_LIM if j7_yaw_lim is None else float(j7_yaw_lim)
        nom = float(model.qpos0[qadr[6]])
        rng[6, 0] = max(rng[6, 0], nom - lim)
        rng[6, 1] = min(rng[6, 1], nom + lim)
    # link geometry + the mount frame, from one forward-kinematics pass
    d = mujoco.MjData(model)
    mujoco.mj_kinematics(model, d)
    mb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                           "l_arm_mount" if side == "left" else "r_arm_mount")
    eb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                           "l_elbow" if side == "left" else "r_elbow")
    sh = d.xanchor[jids[0]]
    el = d.xpos[eb]
    wr = d.xanchor[J(f"{S} Wrist")]
    geo = np.array([np.linalg.norm(el - sh), np.linalg.norm(wr - el)], float)
    parts = [rng.ravel(), np.asarray(model.qpos0[qadr], float).ravel(), geo,
             np.asarray(d.xmat[mb], float).ravel(), np.asarray(R_rob, float).ravel()]
    blob = np.concatenate(parts).round(6).tobytes()
    return hashlib.sha1(blob).hexdigest()[:16]


class WorkspaceTable:
    """A coherent field of arm postures over the front manipulation volume.

    ``q[i, j, k, r]`` is the joint vector (7 arm DoFs + the ORCA wrist) that
    puts the wrist at grid point (i, j, k) with the palm rolled to bin r, with
    the elbow where a relaxed human's would be.

    COHERENCE is the property that makes the table usable live, and it is not
    free: solving each cell independently gives a field full of seams, because
    equal-cost configurations sit on either side of a branch and the solver
    picks whichever the seed favoured.  ``build`` therefore alternates
    independent multi-seed solves with RELAXATION sweeps that re-solve each
    cell seeded from its already-settled neighbours and keep the new answer
    only when it is genuinely better once continuity is priced in.  The
    measured effect is reported by ``experiments/exp_front_workspace.py``.
    """

    VERSION = 1

    def __init__(self, grid, q, res_pos, res_ori, meta=None):
        self.grid = grid
        self.q = np.asarray(q, float)
        self.res_pos = np.asarray(res_pos, float)
        self.res_ori = np.asarray(res_ori, float)
        self.meta = dict(meta or {})

    # -- build ----------------------------------------------------------------
    @classmethod
    def build(cls, chain, grid, solver=None, passes=3, w_cont=8.0,
              swivel_out=0.35, progress=None):
        solver = solver or PostureSolver(chain)
        nx, ny, nz, nr = grid.shape
        n = chain.n
        q = np.zeros((nx, ny, nz, nr, n))
        res_p = np.zeros((nx, ny, nz, nr))
        res_o = np.zeros((nx, ny, nz, nr))
        sh = chain.frame.point(chain.shoulder(), chain.mount_pos, chain.mount_R)

        def cell_targets(i, j, k, r, q_seed=None, wc=0.0):
            p, R = grid.target(i, j, k, r)
            e = natural_elbow(sh, p, chain.l_upper, chain.l_fore, chain.side,
                              swivel_out=swivel_out)
            return solver.targets(p, R, elbow_star=e, q_seed=q_seed, w_cont=wc), p, R

        # pass 0 — independent multi-seed solves (the GLOBAL step)
        total = nx * ny * nz * nr
        done = 0
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    prev = None
                    for r in range(nr):
                        tg, p, R = cell_targets(i, j, k, r)
                        seeds = cls._seed_set(chain, prev, r, nr)
                        qq, _ = solver.solve_multi(seeds, tg)
                        q[i, j, k, r] = qq
                        res_p[i, j, k, r], res_o[i, j, k, r], _ = solver.errors(qq, p, R)
                        prev = qq
                        done += 1
                    if progress and (done % 200 < nr):
                        progress(done, total, "solve")

        # passes 1..N — TWO alternating relaxations, and they are different.
        #
        #   ACCURACY  re-solve each cell seeded from its settled neighbours and
        #             keep the result only if it strictly beats the incumbent on
        #             the task.  This is what pulls cells out of the poor local
        #             minima a cold seed lands in.
        #   COHERENCE break TIES toward the neighbours.  Two sides of a branch
        #             are equal-cost by construction, so an accuracy test can
        #             never choose between them and the field keeps whichever
        #             the cold seed happened to hit — that, not bad solving, is
        #             where a seam comes from.  The 2 % tolerance is the whole
        #             safety margin: a candidate that costs more than that is
        #             not a tie, and is rejected however much tidier it looks.
        for it in range(int(passes)):
            for phase in ("accuracy", "coherence"):
                changed = 0
                order = list(np.ndindex(nx, ny, nz, nr))
                if it % 2:
                    order = order[::-1]
                for (i, j, k, r) in order:
                    nb = cls._neighbours(q, i, j, k, r, nx, ny, nz, nr)
                    if not nb:
                        continue
                    tg, p, R = cell_targets(i, j, k, r)
                    nb_arr = np.asarray(nb)
                    inc = q[i, j, k, r]
                    base = solver.cost(inc, tg)

                    def spread(cand):
                        return float(np.mean(np.sum((nb_arr - cand) ** 2, axis=1)))

                    best_q, best_c, best_s = inc, base, spread(inc)
                    for seed in nb:
                        tg_c, _, _ = cell_targets(i, j, k, r, q_seed=seed, wc=w_cont)
                        cand = solver.solve(seed, tg_c, iters=25)
                        c = solver.cost(cand, tg)
                        if phase == "accuracy":
                            if c < best_c - 1e-6:
                                best_q, best_c, best_s = cand, c, spread(cand)
                        else:
                            sp = spread(cand)
                            if c <= 1.02 * base + 1e-9 and sp < best_s - 1e-6:
                                best_q, best_c, best_s = cand, c, sp
                    if float(np.max(np.abs(best_q - inc))) > 1e-6:
                        q[i, j, k, r] = best_q
                        res_p[i, j, k, r], res_o[i, j, k, r], _ = solver.errors(best_q, p, R)
                        changed += 1
                if progress:
                    progress(it + 1, passes, f"relax/{phase} ({changed} cells changed)")

        meta = dict(side=chain.side, n=n, l_upper=chain.l_upper,
                    l_fore=chain.l_fore, swivel_out=swivel_out,
                    shoulder=sh, frame_R=chain.frame.R, version=cls.VERSION,
                    fingerprint=model_fingerprint(
                        chain.model, chain.side, chain.R_rob, chain.lock_j7,
                        chain.optimize_wrist, chain.j7_yaw_lim))
        return cls(grid, q, res_p, res_o, meta)

    @staticmethod
    def _seed_set(chain, prev, r, nr):
        """Cold seeds for the global step: nominal, an elbow pre-bent into the
        arm's natural flexion direction, the mid-range posture, and the previous
        roll's answer (which keeps a roll sweep continuous by construction)."""
        seeds = [chain.q_nom.copy()]
        bent = chain.q_nom.copy()
        if len(bent) > 3:
            lo4, hi4 = chain.lo[3], chain.hi[3]
            bent[3] = 0.6 * (lo4 if abs(lo4) > abs(hi4) else hi4)
        seeds.append(bent)
        seeds.append(0.5 * (chain.lo + chain.hi))
        if prev is not None:
            seeds.append(prev.copy())
        return seeds

    @staticmethod
    def _neighbours(q, i, j, k, r, nx, ny, nz, nr):
        out = []
        for di, dj, dk in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            a, b, c = i + di, j + dj, k + dk
            if 0 <= a < nx and 0 <= b < ny and 0 <= c < nz:
                out.append(q[a, b, c, r])
        for dr in (1, -1):                      # roll is periodic
            out.append(q[i, j, k, (r + dr) % nr])
        return out

    # -- persistence ----------------------------------------------------------
    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez_compressed(
            path, q=self.q, res_pos=self.res_pos, res_ori=self.res_ori,
            x=self.grid.x, y=self.grid.y, z=self.grid.z, rolls=self.grid.rolls,
            side=self.grid.side, version=self.VERSION,
            shoulder=self.meta.get("shoulder", np.zeros(3)),
            frame_R=self.meta.get("frame_R", np.eye(3)),
            l_upper=self.meta.get("l_upper", 0.0), l_fore=self.meta.get("l_fore", 0.0),
            swivel_out=self.meta.get("swivel_out", 0.0),
            fingerprint=self.meta.get("fingerprint", ""))
        return path

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        grid = WorkspaceGrid(z["x"], z["y"], z["z"], rolls=z["rolls"],
                             side=str(z["side"]))
        meta = dict(side=str(z["side"]), shoulder=z["shoulder"], frame_R=z["frame_R"],
                    l_upper=float(z["l_upper"]), l_fore=float(z["l_fore"]),
                    swivel_out=float(z["swivel_out"]), version=int(z["version"]),
                    fingerprint=str(z["fingerprint"]) if "fingerprint" in z else "")
        return cls(grid, z["q"], z["res_pos"], z["res_ori"], meta)

    _CACHE = {}

    def matches(self, fingerprint):
        """Was this table built against the model that produced `fingerprint`?"""
        fp = self.meta.get("fingerprint", "")
        return (not fp) or (not fingerprint) or fp == fingerprint

    @classmethod
    def load_for(cls, side, directory=None, fingerprint=None):
        """Load (and cache) the built table for a side; None when absent.

        With `fingerprint`, a table built against a DIFFERENT arm model is refused
        (with a warning) rather than used — a stale table is worse than none,
        because none degrades to the pure local solve and a stale one degrades to
        a confidently wrong posture.

        Cached by path + mtime: a table is a read-only artefact shared by every
        retargeter in the process, and reloading it per construction dominated
        the grid harness (1200 poses x one 300 kB npz each).
        """
        p = table_path(side, directory)
        if not os.path.exists(p):
            return None
        key = (p, os.path.getmtime(p))
        if key not in cls._CACHE:
            cls._CACHE.clear()
            cls._CACHE[key] = cls.load(p)
        tbl = cls._CACHE[key]
        if fingerprint and not tbl.matches(fingerprint):
            warnings.warn(
                f"front-workspace table {os.path.basename(p)} was built against a "
                f"different arm model (fingerprint mismatch) — ignoring it and "
                f"falling back to the local solve. Rebuild with: python3 "
                f"controller/experiments/exp_front_workspace.py --phase W3 --build",
                RuntimeWarning, stacklevel=2)
            return None
        return tbl

    # -- query ----------------------------------------------------------------
    def roll_of(self, R_palm_s, p_s):
        """Roll of a commanded palm frame about the cell's canonical finger axis.

        Measured in the SAME convention ``palm_target`` builds: 0 = palmar
        normal down, +90 deg = inward.  The commanded finger direction is used
        as the roll axis (not the canonical one), so a pose whose fingers point
        somewhere else still lands in the right roll bin.
        """
        f = np.asarray(R_palm_s, float)[:, 0]
        n = np.asarray(R_palm_s, float)[:, 2]
        n0 = palm_target(f, 0.0, self.grid.side)[:, 2]      # the "down" normal for this f
        b = np.cross(f, n0)
        inward = -1.0 if self.grid.side == "left" else 1.0
        return float(np.arctan2(inward * float(np.dot(n, b)), float(np.dot(n, n0))))

    #: cells further than this (rad, per joint) from the nearest cell are not
    #: blended with it — they are on the other side of a branch, and averaging
    #: two branches yields a posture that is in neither.
    SEAM_TOL = 0.60

    def cells_for(self, p_s, R_palm_s):
        """The 16 bracketing cells (index, weight) for a pose."""
        t = self.grid.locate(p_s)
        nr = len(self.grid.rolls)
        rf = (self.roll_of(R_palm_s, p_s) % (2.0 * np.pi)) / (2.0 * np.pi) * nr
        r0 = int(np.floor(rf)) % nr
        wr = float(rf - np.floor(rf))
        idx0 = np.clip(np.floor(t).astype(int), 0,
                       np.maximum(np.array(self.q.shape[:3]) - 2, 0))
        f = np.clip(t - idx0, 0.0, 1.0)
        out = []
        for r, wrr in ((r0, 1.0 - wr), ((r0 + 1) % nr, wr)):
            if wrr <= 0.0:
                continue
            for di in (0, 1):
                for dj in (0, 1):
                    for dk in (0, 1):
                        w = wrr * ((f[0] if di else 1 - f[0])
                                   * (f[1] if dj else 1 - f[1])
                                   * (f[2] if dk else 1 - f[2]))
                        if w <= 1e-9:
                            continue
                        out.append(((min(idx0[0] + di, self.q.shape[0] - 1),
                                     min(idx0[1] + dj, self.q.shape[1] - 1),
                                     min(idx0[2] + dk, self.q.shape[2] - 1), r), w))
        return out

    def lookup(self, p_s, R_palm_s, seam_tol=None):
        """(posture, feasibility) in ONE pass — the live path's only entry point.

        ``query`` and ``feasibility`` each need the same 16 bracketing cells, and
        finding them costs a roll decomposition; the live solver wants both every
        frame, so it asks once.
        """
        cells = self.cells_for(p_s, R_palm_s)
        return self._blend(cells, seam_tol), self._feasibility(cells)

    def query(self, p_s, R_palm_s, seam_tol=None):
        """Posture for a shoulder-frame wrist point + palm frame.

        SEAM-AWARE interpolation.  Inside a coherent region this is ordinary
        quadrilinear blending (position x roll) and the answer varies smoothly
        as the operator moves.  At a branch boundary — where two adjacent cells
        genuinely hold opposite configurations, which on this arm is real and
        not a build artefact — blending would invent a posture that solves
        neither cell, so cells further than ``SEAM_TOL`` from the dominant one
        are dropped and the weights renormalised.  The result degrades to
        nearest-cell exactly where nearest-cell is the only honest answer.
        """
        return self._blend(self.cells_for(p_s, R_palm_s), seam_tol)

    def _blend(self, cells, seam_tol=None):
        tol = self.SEAM_TOL if seam_tol is None else float(seam_tol)
        if not cells:
            return self.q[0, 0, 0, 0].copy()
        anchor = max(cells, key=lambda c: c[1])[0]
        q0 = self.q[anchor]
        acc = np.zeros(self.q.shape[-1])
        wsum = 0.0
        for idx, w in cells:
            qc = self.q[idx]
            if float(np.max(np.abs(qc - q0))) > tol:
                continue
            acc += w * qc
            wsum += w
        return acc / wsum if wsum > 1e-9 else q0.copy()

    def feasibility(self, p_s, R_palm_s):
        """0..1: could the arm actually hold this pose?

        The table already knows — it recorded, per cell, how close its own global
        multi-seed solve got. That is exactly the information the live solver
        needs and has no cheap way to compute: over the front box about a quarter
        of the (position, palm-direction) pairs are simply not reachable with
        this arm (one +/-90 deg forearm roll, a +/-45 deg wrist pitch, and narrow
        shoulders), and on those the palm angle the operator asked for cannot be
        delivered at any weight. Pushing orientation hard there does not buy the
        palm angle, it only drags the WRIST off the point the operator put it —
        measured, 40 mm of position error either way, and worse the harder you
        push. So the live cost reads this and backs off.
        """
        return self._feasibility(self.cells_for(p_s, R_palm_s))

    def _feas_field(self):
        f = getattr(self, "_feas", None)
        if f is None:
            pos_ok = np.clip((0.040 - self.res_pos) / (0.040 - 0.015), 0.0, 1.0)
            ori_ok = np.clip((np.radians(15.0) - self.res_ori)
                             / (np.radians(15.0) - np.radians(6.0)), 0.0, 1.0)
            f = self._feas = pos_ok * ori_ok
        return f

    def _feasibility(self, cells):
        f = self._feas_field()
        acc, wsum = 0.0, 0.0
        for idx, w in cells:
            acc += w * float(f[idx])
            wsum += w
        return acc / wsum if wsum > 1e-9 else 1.0

    # -- diagnostics ----------------------------------------------------------
    def seam(self, tol=None):
        """Joint-space jump between adjacent cells [rad], per axis.

        This is what says whether blending the table is meaningful.  Report the
        DISTRIBUTION, not just the max: a handful of true branch boundaries is
        expected on a 7-DoF arm with a +/-90 deg forearm roll (palm-up and
        palm-down sit at opposite ends of that one joint), and a max of ~pi
        proves only that at least one such boundary exists.  What matters is the
        FRACTION of adjacent pairs that are seams, because that is the fraction
        of the workspace where the query has to fall back to nearest-cell.
        """
        tol = self.SEAM_TOL if tol is None else float(tol)
        out = {}
        seams = []
        for ax, name in ((0, "x"), (1, "y"), (2, "z")):
            d = np.abs(np.diff(self.q, axis=ax)).max(axis=-1)
            out[name] = float(d.max()) if d.size else 0.0
            out[f"{name}_p95"] = float(np.percentile(d, 95)) if d.size else 0.0
            seams.append(d.ravel())
        d = np.abs(self.q - np.roll(self.q, -1, axis=3)).max(axis=-1)
        out["roll"] = float(d.max()) if d.size else 0.0
        out["roll_p95"] = float(np.percentile(d, 95)) if d.size else 0.0
        seams.append(d.ravel())
        allp = np.concatenate(seams)
        out["seam_pct"] = float(np.mean(allp > tol) * 100.0)
        out["p50"] = float(np.median(allp))
        out["p95"] = float(np.percentile(allp, 95))
        return out
