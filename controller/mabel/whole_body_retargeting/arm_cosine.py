"""Cosine-directional arm retargeting (wrist SE(3) + natural posture).

A faithful port of the modular research stack's arm IK (scripts/teleop/arm.py) into
the shipped controller package, so the Vision Pro app (bridge -> MabelTeleop) and
scripts/avp_teleop.py drive the robot with the SAME arm motion. On a scratch MjData it
matches, scale-independently, the DIRECTIONS of the operator's arm/palm to the robot's
(cosine), tracks the hand CONTACT point (~middle fingertip) position, and regularizes
the per-joint posture so the redundant 7-DoF arm resolves like a human's (roll done by
the forearm, not the shoulder/last joint). Solved with SLSQP -> smooth, never erratic
at singularities (no Jacobian inverse). The ORCA wrist is a jointly-optimised 8th DoF
(natural flip) when ``ARM_OPTIMIZE_WRIST``.

All tunables live in ``config`` (``ARM_*`` / ``HAND_REACH``).
"""
import numpy as np
import mujoco
from scipy.optimize import minimize

from .. import config as C


def _natural_frame(wrist, fwd_pt, index_pt, pinky_pt):
    """Right-handed palm frame, columns (f=forward, s=index->pinky, n=palm normal),
    in whatever frame the points are given. Anatomy-derived, so chirality is automatic.
    The SAME construction is used on the operator skeleton and the robot FK, making the
    two frames directly comparable (this is the convention avp_teleop.py uses)."""
    f = np.asarray(fwd_pt, float) - np.asarray(wrist, float)
    nf = np.linalg.norm(f)
    if nf < 1e-9:
        return None
    f /= nf
    s_raw = np.asarray(pinky_pt, float) - np.asarray(index_pt, float)
    n = np.cross(f, s_raw)
    nn = np.linalg.norm(n)
    if nn < 1e-9:
        return None
    n /= nn
    s = np.cross(n, f)
    return np.column_stack([f, s, n])


def robot_palm_frame(model, side):
    """Robot natural palm frame R_rob (palm-LOCAL, constant) for side 'left'/'right',
    from the rest-pose finger-mount anatomy — the same frame the operator skeleton is
    expressed in, so the orientation cosine is apples-to-apples."""
    d0 = mujoco.MjData(model)
    mujoco.mj_kinematics(model, d0)
    palm = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_palm")
    pp = d0.xpos[palm]
    pm = d0.xmat[palm].reshape(3, 3)

    def loc(name):
        b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        return pm.T @ (d0.xpos[b] - pp)

    return _natural_frame(np.zeros(3), loc(f"{side}_middle_mp"),
                          loc(f"{side}_index_mp"), loc(f"{side}_pinky_mp"))


class ArmRetargeter:
    """Decoupled per-arm cosine-similarity IK. `side` is 'left' | 'right'."""

    def __init__(self, model, side, R_rob=None, lock_j7=False, optimize_wrist=None):
        self.model = model
        self.ik = mujoco.MjData(model)
        self.side = side
        S = side.capitalize()
        s0 = side[0]                                   # 'l' | 'r' for the body-name maps
        self.lock_j7 = lock_j7
        if optimize_wrist is None:
            optimize_wrist = C.ARM_OPTIMIZE_WRIST
        self.optimize_wrist = optimize_wrist
        self.orca_wrist_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{S} Wrist")
        self.orca_wrist_qadr = model.jnt_qposadr[self.orca_wrist_jid]
        self.orca_wrist_lo, self.orca_wrist_hi = model.jnt_range[self.orca_wrist_jid]

        n_arm = 6 if lock_j7 else 7
        self.n_arm = n_arm
        self.jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{S} Arm {i}")
                     for i in range(1, n_arm + 1)]
        self.qadr = np.array([model.jnt_qposadr[j] for j in self.jids])
        self.opt_jids = list(self.jids) + ([self.orca_wrist_jid] if optimize_wrist else [])
        self.opt_qadr = np.array([model.jnt_qposadr[j] for j in self.opt_jids])
        self.lo = model.jnt_range[self.opt_jids, 0].copy()
        self.hi = model.jnt_range[self.opt_jids, 1].copy()
        self.q_nom = model.qpos0[self.opt_qadr].copy()
        # Wrist YAW (Arm7) is anatomically constrained -> clip its OPTIMISATION bound to
        # rest +/- ARM_J7_YAW_LIM so the solver can't spin it to fake orientation.
        if not lock_j7:
            nom7 = self.q_nom[6]
            self.lo[6] = max(self.lo[6], nom7 - C.ARM_J7_YAW_LIM)
            self.hi[6] = min(self.hi[6], nom7 + C.ARM_J7_YAW_LIM)
        self.bounds = list(zip(self.lo, self.hi))
        self.reg_w = C.ARM_REG_W[:n_arm].copy()
        if optimize_wrist:
            self.reg_w = np.append(self.reg_w, C.ARM_REG_WRIST)
        self.last_q = self.q_nom.copy()
        j7 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{S} Arm 7")
        self.j7_qadr = model.jnt_qposadr[j7]
        self.j7_lock_val = float(model.qpos0[self.j7_qadr])
        # keypoint bodies
        self.elbow_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, C.ARM_ELBOW[s0])
        self.shoulder_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, C.ARM_SHOULDER[s0])
        self.palm_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_palm")
        self.R_rob = R_rob if R_rob is not None else robot_palm_frame(model, side)

    def sync(self, data):
        """Load the live robot pose (base/lift/torso/neck current) into the scratch."""
        self.ik.qpos[:] = data.qpos
        if self.lock_j7:
            self.ik.qpos[self.j7_qadr] = self.j7_lock_val

    def _kps(self, q):
        q = np.asarray(q, float)
        if len(q) == len(self.opt_qadr):                 # full opt vector (arm + wrist)
            self.ik.qpos[self.opt_qadr] = q
        else:                                            # arm-only -> keep current wrist
            self.ik.qpos[self.qadr] = q
        if self.lock_j7:
            self.ik.qpos[self.j7_qadr] = self.j7_lock_val
        mujoco.mj_kinematics(self.model, self.ik)        # FORWARD KINEMATICS
        elbow = self.ik.xpos[self.elbow_bid].copy()
        wrist = self.ik.xanchor[self.orca_wrist_jid].copy()
        sh = self.ik.xpos[self.shoulder_bid].copy()
        Rp = self.ik.xmat[self.palm_bid].reshape(3, 3) @ self.R_rob   # palm canonical (world)
        contact = wrist + C.HAND_REACH * Rp[:, 0]        # hand contact pt (~middle fingertip)
        return elbow, contact, sh, Rp[:, 0], Rp[:, 2]    # elbow, CONTACT, shoulder, fwd, up

    def _cost(self, q, hk, q_prev):
        re, rc, rsh, rfwd, rup = self._kps(q)
        loss = C.ARM_W_WRIST * float(np.sum((rc - hk["contact"]) ** 2))
        if hk.get("elbow") is not None:
            ed = re - rsh; hed = hk["elbow"] - rsh
            ed /= (np.linalg.norm(ed) + 1e-9); hed /= (np.linalg.norm(hed) + 1e-9)
            loss += C.ARM_W_ELBOW * (1.0 - float(np.dot(ed, hed)))
        loss += C.ARM_W_ORIENT * ((1.0 - float(np.dot(rfwd, hk["fwd"])))
                                  + (1.0 - float(np.dot(rup, hk["up"]))))
        loss += C.ARM_W_SMOOTH * float(np.sum((q - q_prev) ** 2))
        loss += float(np.sum(self.reg_w * (q - self.q_nom) ** 2))
        return loss

    def solve(self, hk):
        """hk: dict with contact / elbow / fwd / up (robot world). Jointly optimises the
        arm (+ ORCA wrist) and leaves the scratch at the solution. Returns the ARM joint
        angles only (length n_arm); the ORCA wrist is read from self.ik.qpos."""
        res = minimize(self._cost, self.last_q, args=(hk, self.last_q),
                       method="SLSQP", bounds=self.bounds,
                       options={"maxiter": C.ARM_MAXITER, "ftol": 1e-4})
        self.last_q = res.x
        self.ik.qpos[self.opt_qadr] = res.x
        if self.lock_j7:
            self.ik.qpos[self.j7_qadr] = self.j7_lock_val
        return res.x[:self.n_arm]
