"""Low-level model helpers: actuator/DoF maps, robot reach, scale calibration."""
import json
import numpy as np
import mujoco
import zmq

from .hyperparams import *
from .logsetup import logger
from ..dynamic_models.gravity import gravity_dofs as _gravity_dofs


def position_actuators(model):
    """Position-servo actuator ids whose transmission is a single joint, with that
    joint's rest target (qpos0). Used to hold the rest pose on launch / reset so the
    dynamic robot does not jump, and excludes the swerve wheels (gtype != joint)."""
    out = []
    for aid in range(model.nu):
        if model.actuator_trntype[aid] != mujoco.mjtTrn.mjTRN_JOINT:
            continue
        jid = model.actuator_trnid[aid, 0]
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
        if name in ("FL Steer", "FR Steer", "B Steer", "FL Drive", "FR Drive", "B Drive"):
            continue
        out.append((aid, float(model.qpos0[model.jnt_qposadr[jid]])))
    return out


def gravity_dofs(model):
    """DoF addresses of the actuated upper body (lift, torso, neck, both arms +
    ORCA wrists) — where the gravity feed-forward is applied. The floating base and
    swerve wheels are excluded (gravity there is carried by the ground contacts).

    Re-exported from mabel.dynamic_models.gravity, which is the ONE definition —
    the torque engine compensates the same set, and a second copy here is exactly
    how Drive and Joint/Task drifted apart (see that module's docstring)."""
    return _gravity_dofs(model)


def robot_forward_reach(model, head_bid, side="right"):
    """Robot URDF reach: max head->ORCA-wrist distance with the arm extended
    straight forward (front = -x via R_BASE_ALIGN). Self-contained DLS over Arm1..7;
    used to auto-scale the human."""
    d2 = mujoco.MjData(model); mujoco.mj_forward(model, d2)
    S = side.capitalize()
    jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{S} Arm {i}") for i in range(1, 8)]
    dofs = np.array([model.jnt_dofadr[j] for j in jids])
    qadr = np.array([model.jnt_qposadr[j] for j in jids])
    lo = model.jnt_range[jids, 0]; hi = model.jnt_range[jids, 1]
    wjid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{S} Wrist")
    palm_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_palm")
    rh = d2.xpos[head_bid].copy()
    base_R = d2.xmat[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ROBOT_BASE_BODY)].reshape(3, 3)
    target = rh + 2.0 * (base_R @ R_BASE_ALIGN @ np.array([1.0, 0.0, 0.0]))   # far front
    jacp = np.zeros((3, model.nv))
    for _ in range(150):
        mujoco.mj_kinematics(model, d2); mujoco.mj_comPos(model, d2)
        pt = d2.xanchor[wjid]; e = target - pt
        mujoco.mj_jac(model, d2, jacp, None, pt, palm_bid)
        J = jacp[:, dofs]
        dq = J.T @ np.linalg.solve(J @ J.T + 0.01 * np.eye(3), e)
        nrm = np.linalg.norm(dq)
        if nrm > 0.2:
            dq *= 0.2 / nrm
        d2.qpos[qadr] = np.clip(d2.qpos[qadr] + dq, lo, hi)
    mujoco.mj_kinematics(model, d2)
    return float(np.linalg.norm(d2.xanchor[wjid] - rh))


def _ee_pt(d2, wjid, ee):
    """The tracked point: the ORCA wrist anchor, or ee = (body, local point) (end_effector.wrist_centre)."""
    if ee is None:
        return d2.xanchor[wjid]
    b, off = ee
    return d2.xpos[b] + d2.xmat[b].reshape(3, 3) @ off


def symmetric_reach(reach):
    """{side: L_rob} -> the same value for both sides (the smaller), when ARM_REACH_SYMMETRIC."""
    from .hyperparams import ARM_REACH_SYMMETRIC
    if not ARM_REACH_SYMMETRIC or not reach:
        return dict(reach)
    v = min(float(x) for x in reach.values())
    return {s: v for s in reach}


def robot_arm_max_reach(model, side="right", ee=None):
    """Usable ARM reach [m]: max |arm-mount body → ORCA-wrist anchor| with the
    arm stretched as straight as its joint limits allow. Measured by the same
    self-contained DLS as robot_forward_reach, but ANCHORED AT THE SHOULDER
    (the arm mount) — this is the L_rob of the extension-fraction mapping
    (t = s_rob + û·rho·L_rob). Two far targets are tried (straight forward and
    forward-up 45°) and the larger converged reach wins, so a joint-limit
    corner on one ray cannot under-report the reach."""
    S = side.capitalize()
    jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{S} Arm {i}")
            for i in range(1, 8)]
    dofs = np.array([model.jnt_dofadr[j] for j in jids])
    qadr = np.array([model.jnt_qposadr[j] for j in jids])
    lo = model.jnt_range[jids, 0]; hi = model.jnt_range[jids, 1]
    wjid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{S} Wrist")
    palm_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_palm")
    mount_bid = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY,
        "r_arm_mount" if side == "right" else "l_arm_mount")

    best = 0.0
    jacp = np.zeros((3, model.nv))
    for ray in (np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 1.0]) / np.sqrt(2)):
        d2 = mujoco.MjData(model); mujoco.mj_forward(model, d2)
        sh = d2.xpos[mount_bid].copy()            # mount is above the arm chain: fixed
        base_R = d2.xmat[mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, ROBOT_BASE_BODY)].reshape(3, 3)
        target = sh + 2.0 * (base_R @ R_BASE_ALIGN @ ray)
        for _ in range(150):
            mujoco.mj_kinematics(model, d2); mujoco.mj_comPos(model, d2)
            pt = _ee_pt(d2, wjid, ee); e = target - pt
            mujoco.mj_jac(model, d2, jacp, None, pt, palm_bid)
            J = jacp[:, dofs]
            dq = J.T @ np.linalg.solve(J @ J.T + 0.01 * np.eye(3), e)
            nrm = np.linalg.norm(dq)
            if nrm > 0.2:
                dq *= 0.2 / nrm
            d2.qpos[qadr] = np.clip(d2.qpos[qadr] + dq, lo, hi)
        mujoco.mj_kinematics(model, d2)
        best = max(best, float(np.linalg.norm(_ee_pt(d2, wjid, ee) - sh)))
    return best


def human_reach(payload):
    """Operator head->wrist distance (mean of both arms) from a stream frame."""
    hm = payload.get("head_mat")
    if not hm:
        return None
    head = np.asarray(hm, float).reshape(4, 4)[:3, 3]
    ds = []
    for k in ("right_hand_mats", "left_hand_mats"):
        hand = payload.get(k)
        if hand:
            ds.append(np.linalg.norm(np.asarray(hand[0], float).reshape(4, 4)[:3, 3] - head))
    return float(np.mean(ds)) if ds else None


def calibrate_scale(sub, model, head_bid):
    """Interactive scale calibration: ask the operator to extend both arms straight
    forward, press Enter, then set REACH_SCALE = robot reach / human reach so the
    robot always covers the operator's workspace. Returns the scale or None."""
    print("\n\033[1;96m── SCALE CALIBRATION ──\033[0m")
    print("  Extend BOTH arms straight out in FRONT of you, then press Enter "
          "(Enter with no stream = keep default)...")
    try:
        input()
    except EOFError:
        return None
    msg = None
    try:
        while True:
            msg = sub.recv_string(flags=zmq.NOBLOCK)
    except zmq.Again:
        pass
    if msg is None:
        logger.warning("No stream frame captured — keeping default REACH_SCALE.")
        return None
    hr = human_reach(json.loads(msg))
    if hr is None or hr < 0.1:
        logger.warning("Could not measure human reach — keeping default REACH_SCALE.")
        return None
    rr = robot_forward_reach(model, head_bid)
    scale = rr / hr
    logger.info(f"human reach {hr:.2f} m | robot reach {rr:.2f} m  ->  REACH_SCALE = {scale:.2f}")
    return scale

class DirectionalReach:
    """Max |shoulder -> ORCA-wrist anchor| as a function of DIRECTION.

    `robot_arm_max_reach` returns ONE scalar, measured on two forward-ish rays.
    The extension map then asks for rho * L_rob in whatever direction the
    operator is pointing — so in directions where the arm can reach FURTHER than
    that scalar, the operator's full extension maps to a sub-maximal robot
    extension and the elbow keeps a bend that nothing will remove.

    Measured on MABEL: forward reach 0.652 m with the elbow at 162 deg (its
    kinematic best), but LATERAL reach 0.772 m at 174 deg. A T-pose therefore
    asked for 0.696 m into a direction good for 0.772 m, and the arm answered
    with 132 deg of elbow — a 41 deg shape error that is entirely an artefact of
    using one direction-independent number.

    This samples the reachable sphere once at construction (a few hundred DLS
    solves, ~1 s) and interpolates at run time by nearest-of-k on the unit
    sphere, which is cheap enough for the 30-60 Hz retarget path.
    """

    def __init__(self, model, side="right", n_dirs=64, ee=None):
        self.dirs, self.reach = [], []
        S = side.capitalize()
        jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{S} Arm {i}")
                for i in range(1, 8)]
        dofs = np.array([model.jnt_dofadr[j] for j in jids])
        qadr = np.array([model.jnt_qposadr[j] for j in jids])
        lo, hi = model.jnt_range[jids, 0], model.jnt_range[jids, 1]
        wjid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{S} Wrist")
        palm_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_palm")
        mount_bid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY,
            "r_arm_mount" if side == "right" else "l_arm_mount")
        jacp = np.zeros((3, model.nv))
        base_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ROBOT_BASE_BODY)
        # Fibonacci sphere: even coverage without a pole bias.
        ga = np.pi * (3.0 - np.sqrt(5.0))
        for i in range(n_dirs):
            z = 1.0 - 2.0 * (i + 0.5) / n_dirs
            r = np.sqrt(max(0.0, 1.0 - z * z))
            th = ga * i
            ray = np.array([r * np.cos(th), r * np.sin(th), z])
            d2 = mujoco.MjData(model); mujoco.mj_forward(model, d2)
            sh = d2.xpos[mount_bid].copy()
            base_R = d2.xmat[base_bid].reshape(3, 3)
            world_ray = base_R @ R_BASE_ALIGN @ ray
            target = sh + 2.0 * world_ray
            for _ in range(120):
                mujoco.mj_kinematics(model, d2); mujoco.mj_comPos(model, d2)
                pt = _ee_pt(d2, wjid, ee); e = target - pt
                mujoco.mj_jac(model, d2, jacp, None, pt, palm_bid)
                J = jacp[:, dofs]
                dq = J.T @ np.linalg.solve(J @ J.T + 0.01 * np.eye(3), e)
                n = np.linalg.norm(dq)
                if n > 0.2:
                    dq *= 0.2 / n
                d2.qpos[qadr] = np.clip(d2.qpos[qadr] + dq, lo, hi)
            mujoco.mj_kinematics(model, d2)
            v = _ee_pt(d2, wjid, ee) - sh
            nv = float(np.linalg.norm(v))
            if nv > 1e-3:
                self.dirs.append(v / nv)          # the direction ACHIEVED, not asked
                self.reach.append(nv)
        self.dirs = np.asarray(self.dirs, float)
        self.reach = np.asarray(self.reach, float)
        self.scalar = float(self.reach.max()) if len(self.reach) else 0.0

    def at(self, u, k=3):
        """Max reach along unit direction `u` (robot world), nearest-k blended."""
        if not len(self.reach):
            return self.scalar
        u = np.asarray(u, float)
        n = np.linalg.norm(u)
        if n < 1e-9:
            return self.scalar
        c = self.dirs @ (u / n)
        idx = np.argpartition(-c, min(k, len(c) - 1))[:k]
        w = np.clip(c[idx], 0.0, None) ** 4          # sharp: neighbours only
        if w.sum() < 1e-9:
            return float(self.reach[int(np.argmax(c))])
        return float((w * self.reach[idx]).sum() / w.sum())

