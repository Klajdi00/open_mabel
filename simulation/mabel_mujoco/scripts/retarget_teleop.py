"""
retarget_teleop.py -- SELF-CONTAINED Vision Pro -> MABEL whole-body retargeting.

A single-file, dependency-light reimplementation of the MABEL upper-body teleop:
it turns the operator's streamed head + two 6-DoF wrists + finger pinch into
coordinated robot motion and opens the MuJoCo viewer. Unlike the package version
(controller/mabel/ + scripts/retarget_teleop's thin orchestrator), this file
imports NOTHING from the project -- only numpy, mujoco, zmq and pynput -- so it
can be copied out and run on its own, or read end-to-end to understand the method.

Pipeline (all inlined below):
  operator pose ──▶ AVP→robot frame map ──▶ clutch-based RELATIVE wrist→palm SE(3)
  targets ──▶ weighted damped-least-squares whole-body IK (lift+torso+arms+wrists)
  ──▶ position servos + feed-forward gravity g(q). Head ──▶ gaze direction ──▶
  3-DoF look-at neck. Pinch ──▶ grasp curl. WHOLE region ──▶ swerve base tracks the
  operator's global translation.

Method notes (what is faithful vs simplified relative to the package):
  * Retargeting, whole-body DLS IK, swerve IK, neck look-at and the gravity
    feed-forward are ported verbatim (same math + tuning constants).
  * FINGERS use the simple pinch→grasp curl (a documented stand-in); the package's
    full DexPilot per-finger keypoint IK (hand_ik_retarget.py) is NOT inlined.
  * The base motion model is a plain accel-limited smoother; the package's
    direction-aware tip-over / ZMP model (config.safe_base_step) is NOT inlined.

Region cycle (TAB, or 1/2/3):
  1 ARM   -- arms + hands only (lift, torso, base perfectly still)
  2 UPPER -- + lift + torso join the IK (vertical / forward reach)
  3 WHOLE -- + the swerve base tracks the operator's global head translation

Sources (--source):
  avp    live JSON stream from vision_pro_stream.py over ZMQ (default)
  mock   synthetic circling wrists (no headset; smoke-test the pipeline)

Keys:  TAB cycle region · C re-anchor clutch · G toggle gravity feedforward
       BACKSPACE reset sim

Run (macOS needs mjpython for the passive viewer):
  # terminal 1:  python scripts/vision_pro_stream.py --ip <headset-ip>
  # terminal 2:
  mjpython scripts/retarget_teleop.py --source avp
  mjpython scripts/retarget_teleop.py --source mock        # no headset
"""
import os
import math
import time
import json
import argparse

import numpy as np
import mujoco

# =====================================================================
# CONFIG -- every constant the pipeline needs (mirrors controller/mabel/config.py)
# =====================================================================
_HERE = os.path.dirname(os.path.abspath(__file__))
SCENE_XML = os.path.join(_HERE, "..", "scenes", "mabel_sim.xml")

# Manipulation frames (body names in mabel_sim.xml).
CHASSIS_BODY = "mabel_chassis"
LEFT_PALM = "left_palm"
RIGHT_PALM = "right_palm"
HEAD_BODY = "head"

# Vision Pro stream + axis remap (ARKit X-right/Y-up/-Z-fwd -> robot -X-fwd/Z-up).
AVP_ZMQ_ADDR = "tcp://127.0.0.1:5556"
AVP_ZMQ_TOPIC = ""
AVP_TO_ROBOT = np.array([[0.0, 0.0, 1.0],
                         [1.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0]])
WRIST_ALIGN = np.eye(3)
RETARGET_SCALE = np.array([1.3, 1.3, 1.4])     # GMR per-axis [fwd,left,up] amplify
HEAD_FORWARD_LOCAL = np.array([0.0, 0.0, -1.0])

# Pinch -> grasp.
PINCH_OPEN_DIST = 0.08
PINCH_CLOSE_DIST = 0.02

# Upper-body IK chain (18 DOF) -- actuator names in mabel_sim.xml.
IK_ACTUATORS = (["Lift", "Torso"]
                + [f"Left Arm {i}" for i in range(1, 8)] + ["Left Wrist"]
                + [f"Right Arm {i}" for i in range(1, 8)] + ["Right Wrist"])
LIFT_JOINT = "Lift Lower"          # Lift actuator drives a tendon; read this joint
IK_KP_POS = 6.0
IK_KP_ROT = 2.0
IK_DAMPING = 0.05
IK_POSTURE_GAIN = 0.5
IK_VEL_CAP = 1.5
IK_ANGVEL_CAP = 3.0
IK_POS_DEADBAND = 0.002
IK_ROT_DEADBAND = 0.0087
IK_SETTLE_STEPS = 800
IK_WEIGHT_LIFT = 50.0
IK_WEIGHT_TORSO = 2.0
MAX_LIFT_IK_SPEED = 0.10

# Active-perception neck.
NECK_ACTUATORS = ["Neck Yaw", "Neck Pitch", "Neck Roll"]
NECK_JOINTS = ["Neck Yaw", "Neck Pitch", "Neck Roll"]
NECK_KP = 4.0
NECK_FALLBACK_DIST = 1.5
NECK_RAY_OFFSET = 0.20
NECK_MIN_HIT_DIST = 0.25

# Swerve base (3-module delta).
WHEEL_RADIUS = 0.0381
MODULE_POSITIONS = {"fl": np.array([-0.17, -0.17]),
                    "fr": np.array([-0.17, 0.17]),
                    "b":  np.array([0.15, 0.0])}
MODULE_YAW_OFFSETS = {"fl": 1.570795 + math.pi / 2,
                      "fr": 0.0 + math.pi / 2,
                      "b":  3.9269926 + math.pi / 2}
STEER_DEADBAND = 0.52
MAX_LIN_SPEED = 0.8
MAX_LIN_ACCEL = 2.0
LIFT_RANGE = (0.0, 0.635)

# Global base tracking (WHOLE region): operator head translation -> base motion.
GLOBAL_BASE_SCALE = 1.0
GLOBAL_BASE_KP = 1.5

# Real-time accumulator cap (matches the server bridge that "worked very well").
MAX_CATCHUP = float(os.environ.get("MABEL_MAX_CATCHUP", "0.05"))

# Finger flexion actuators driven by the grasp scalar (pinch stand-in for full IK).
_FLEX_SUFFIXES = ["Thumb PIP", "Thumb DIP", "Index MCP", "Index PIP",
                  "Middle MCP", "Middle PIP", "Ring MCP", "Ring PIP",
                  "Pinky MCP", "Pinky PIP"]


# =====================================================================
# Small math helpers
# =====================================================================
def _deadband(e, db):
    n = float(np.linalg.norm(e))
    return e * (max(0.0, n - db) / n) if n > 1e-12 else e


def _so3_error(R_cur, R_trg):
    return 0.5 * (np.cross(R_cur[:, 0], R_trg[:, 0])
                  + np.cross(R_cur[:, 1], R_trg[:, 1])
                  + np.cross(R_cur[:, 2], R_trg[:, 2]))


def _natural_frame(origin, i_kn, p_kn, m_kn):
    """Orthonormal palm frame: y=forward (wrist->middle knuckle), z=palm normal
    (index x pinky knuckle rays), x completes. Built identically on the human
    skeleton and the robot FK so the two are directly comparable."""
    fwd = m_kn - origin
    nf = np.linalg.norm(fwd)
    nrm = np.cross(i_kn - origin, p_kn - origin)
    nn = np.linalg.norm(nrm)
    if nf < 1e-9 or nn < 1e-9:
        return None
    fwd /= nf
    nrm /= nn
    side = np.cross(fwd, nrm)
    side /= (np.linalg.norm(side) + 1e-9)
    nrm = np.cross(side, fwd)
    return np.column_stack([side, fwd, nrm])


# =====================================================================
# Vision Pro ZMQ receiver (latest-frame-wins) + parsed frame
# =====================================================================
class AVPFrame:
    """Parsed Vision Pro frame (AVP/ARKit frame, 4x4 matrices)."""

    def __init__(self, payload):
        self.timestamp = payload.get("timestamp", 0.0)
        self.head = self._mat(payload.get("head_mat"))
        self.right = self._mats(payload.get("right_hand_mats"))
        self.left = self._mats(payload.get("left_hand_mats"))
        self.right_joints = self._joints(payload.get("right_joints"))
        self.left_joints = self._joints(payload.get("left_joints"))

    @staticmethod
    def _mat(flat):
        return None if flat is None else np.asarray(flat, float).reshape(4, 4)

    @staticmethod
    def _mats(lst):
        return None if not lst else [np.asarray(m, float).reshape(4, 4) for m in lst]

    @staticmethod
    def _joints(d):
        return None if not d else {k: np.asarray(v, float).reshape(4, 4) for k, v in d.items()}


class AVPReceiver:
    def __init__(self, addr=AVP_ZMQ_ADDR):
        import zmq
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.SUB)
        self.sock.setsockopt(zmq.CONFLATE, 1)
        self.sock.setsockopt_string(zmq.SUBSCRIBE, AVP_ZMQ_TOPIC)
        self.sock.connect(addr)
        self._zmq = zmq
        self.latest = None

    def poll(self):
        got = None
        while True:
            try:
                msg = self.sock.recv_string(flags=self._zmq.NOBLOCK)
            except self._zmq.Again:
                break
            got = msg
        if got is not None:
            self.latest = AVPFrame(json.loads(got))
        return self.latest

    def close(self):
        self.sock.close(0)


def mock_frame(t):
    """Synthetic operator: head fixed forward, both wrists circling in front, thumb
    and index tips oscillating so the pinch/grasp varies. No headset needed."""
    def T(R, p):
        M = np.eye(4)
        M[:3, :3] = R
        M[:3, 3] = p
        return M

    head = T(np.eye(3), np.array([0.0, 1.4, 0.0]))     # ARKit frame, ~head height
    out = {"timestamp": t, "head_mat": head.flatten().tolist()}
    for side, sgn in (("right", +1.0), ("left", -1.0)):
        cx = 0.20 * sgn + 0.05 * math.cos(t * 1.3)
        cy = 1.0 + 0.08 * math.sin(t * 0.9)
        cz = -0.35 + 0.06 * math.sin(t * 1.7)
        wrist = T(np.eye(3), np.array([cx, cy, cz]))
        mats = [wrist.copy() for _ in range(21)]
        gap = 0.02 + 0.03 * (0.5 + 0.5 * math.sin(t * 2.0))   # pinch open/close
        mats[4] = T(np.eye(3), np.array([cx + 0.5 * gap, cy, cz]))   # thumb tip
        mats[8] = T(np.eye(3), np.array([cx - 0.5 * gap, cy, cz]))   # index tip
        out[f"{side}_hand_mats"] = [m.flatten().tolist() for m in mats]
    return AVPFrame(out)


# =====================================================================
# Retargeter -- clutch-based RELATIVE wrist->palm SE(3) mapping
# =====================================================================
def _pinch_grasp(hand_mats):
    if hand_mats is None or len(hand_mats) < 9:
        return 0.0
    dist = np.linalg.norm(hand_mats[4][:3, 3] - hand_mats[8][:3, 3])
    g = (PINCH_OPEN_DIST - dist) / (PINCH_OPEN_DIST - PINCH_CLOSE_DIST)
    return float(np.clip(g, 0.0, 1.0))


class Retargeter:
    def __init__(self, model, data):
        self.model, self.data = model, data
        self.M = AVP_TO_ROBOT
        self.M_rot = AVP_TO_ROBOT @ WRIST_ALIGN
        self.scale = RETARGET_SCALE
        self.calibrated = False
        self.body_relative = True       # palm target rides with the chassis
        self.orient_skeleton = True      # palm orientation from hand geometry
        self._avp = {}
        self._rob = {}
        self._rfids = {}

    # natural-frame helpers (operator skeleton + robot FK) ---------------------
    def _op_frame(self, joints):
        if not joints:
            return None

        def P(n):
            Tn = joints.get(n)
            return None if Tn is None else np.asarray(Tn, float)[:3, 3]
        w, i, p, m = P("wrist"), P("index_knuckle"), P("little_knuckle"), P("middle_knuckle")
        if any(v is None for v in (w, i, p, m)):
            return None
        return _natural_frame(w, i, p, m)

    def _rob_frame_ids(self, side):
        if side not in self._rfids:
            pre = "left" if side == "l" else "right"
            palm = LEFT_PALM if side == "l" else RIGHT_PALM
            names = [palm, f"{pre}_index_pp", f"{pre}_pinky_pp", f"{pre}_middle_pp"]
            ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n) for n in names]
            self._rfids[side] = None if any(i < 0 for i in ids) else ids
        return self._rfids[side]

    def _rob_frame(self, side):
        ids = self._rob_frame_ids(side)
        if ids is None:
            return None
        x = self.data.xpos
        return _natural_frame(x[ids[0]], x[ids[1]], x[ids[2]], x[ids[3]])

    def _joints_for(self, frame, side):
        return frame.right_joints if side == "r" else frame.left_joints

    # calibration -------------------------------------------------------------
    def calibrate(self, frame):
        if frame is None or frame.head is None:
            return False
        self._avp["Rh"] = frame.head[:3, :3].copy()
        if frame.right is not None:
            self._anchor_hand("r", frame.right)
        else:
            for k in ("wr", "Rwr"):
                self._avp.pop(k, None)
        if frame.left is not None:
            self._anchor_hand("l", frame.left)
        else:
            for k in ("wl", "Rwl"):
                self._avp.pop(k, None)
        self._rob["l_pos"] = self.data.body(LEFT_PALM).xpos.copy()
        self._rob["r_pos"] = self.data.body(RIGHT_PALM).xpos.copy()
        self._rob["l_mat"] = self.data.body(LEFT_PALM).xmat.copy().reshape(3, 3)
        self._rob["r_mat"] = self.data.body(RIGHT_PALM).xmat.copy().reshape(3, 3)
        R_head = self.data.body(HEAD_BODY).xmat.reshape(3, 3)
        self._rob["gaze"] = R_head @ HEAD_FORWARD_LOCAL
        self._rob["Rbase0"] = self.data.body(CHASSIS_BODY).xmat.reshape(3, 3).copy()
        for side in ("r", "l"):
            self._rob.pop(f"{side}_K", None)
            F_op = self._op_frame(self._joints_for(frame, side))
            F_rob = self._rob_frame(side)
            if F_op is not None and F_rob is not None:
                palm = LEFT_PALM if side == "l" else RIGHT_PALM
                R_palm = self.data.body(palm).xmat.reshape(3, 3)
                self._rob[f"{side}_K"] = R_palm.T @ F_rob
        self.calibrated = True
        return True

    def _anchor_hand(self, side, mats):
        self._avp[f"w{side}"] = mats[0][:3, 3].copy()
        self._avp[f"Rw{side}"] = mats[0][:3, :3].copy()
        palm = LEFT_PALM if side == "l" else RIGHT_PALM
        self._rob[f"{side}_pos"] = self.data.body(palm).xpos.copy()
        self._rob[f"{side}_mat"] = self.data.body(palm).xmat.copy().reshape(3, 3)
        R_base = self.data.body(CHASSIS_BODY).xmat.reshape(3, 3)
        p_base = self.data.body(CHASSIS_BODY).xpos
        self._rob[f"{side}_pos_body"] = R_base.T @ (self._rob[f"{side}_pos"] - p_base)

    # per-frame update --------------------------------------------------------
    def _map_pos_side(self, side, w_now):
        delta = self.scale * (self.M @ (w_now - self._avp[f"w{side}"]))
        if not self.body_relative:
            return self._rob[f"{side}_pos"] + delta
        R_base = self.data.body(CHASSIS_BODY).xmat.reshape(3, 3)
        p_base = self.data.body(CHASSIS_BODY).xpos
        return p_base + R_base @ (self._rob[f"{side}_pos_body"] + delta)

    def _map_rot(self, Rw_now, Rw0, R0):
        R_rel = Rw_now @ Rw0.T
        return (self.M_rot @ R_rel @ self.M_rot.T) @ R0

    def _map_rot_side(self, side, Rw_now, frame):
        if self.orient_skeleton and f"{side}_K" in self._rob:
            F_op = self._op_frame(self._joints_for(frame, side))
            if F_op is not None:
                return F_op @ self._rob[f"{side}_K"].T
        return self._map_rot(Rw_now, self._avp[f"Rw{side}"], self._rob[f"{side}_mat"])

    def update(self, frame):
        if not self.calibrated or frame is None:
            return None
        out = {}
        if frame.right is not None and "wr" not in self._avp:
            self._anchor_hand("r", frame.right)
        if frame.left is not None and "wl" not in self._avp:
            self._anchor_hand("l", frame.left)
        if frame.right is not None and "wr" in self._avp:
            out["r_pos"] = self._map_pos_side("r", frame.right[0][:3, 3])
            out["r_mat"] = self._map_rot_side("r", frame.right[0][:3, :3], frame)
            out["grasp_r"] = _pinch_grasp(frame.right)
        if frame.left is not None and "wl" in self._avp:
            out["l_pos"] = self._map_pos_side("l", frame.left[0][:3, 3])
            out["l_mat"] = self._map_rot_side("l", frame.left[0][:3, :3], frame)
            out["grasp_l"] = _pinch_grasp(frame.left)
        if frame.head is not None and "Rh" in self._avp:
            R_head_rel = frame.head[:3, :3] @ self._avp["Rh"].T
            R_rel_robot = self.M @ R_head_rel @ self.M.T
            R_base = self.data.body(CHASSIS_BODY).xmat.reshape(3, 3)
            dR_base = R_base @ self._rob["Rbase0"].T
            gaze = R_rel_robot @ dR_base @ self._rob["gaze"]
            out["gaze_dir"] = gaze / (np.linalg.norm(gaze) + 1e-9)
        return out


# =====================================================================
# Whole-body IK -- weighted damped-least-squares over the 18-DOF upper chain
# =====================================================================
class WholeBodyIK:
    def __init__(self, model, data):
        self.model, self.data = model, data
        self.dofs = self._ik_dof_indices()
        w = np.ones(len(self.dofs))
        w[0] = IK_WEIGHT_LIFT
        w[1] = IK_WEIGHT_TORSO
        self.W_inv = np.diag(1.0 / w)
        self.arm_dofs = self.dofs[2:]
        self.arm_actuators = list(IK_ACTUATORS[2:])
        self.W_inv_arms = np.eye(len(self.arm_dofs))
        self.target_q = {}
        self.target_l_pos = self.target_r_pos = None
        self.target_l_mat = self.target_r_mat = None
        self.sync_to_current()
        self.q_rest = np.array([self._q(n) for n in IK_ACTUATORS])
        self.e_pos_l = np.zeros(3); self.e_pos_r = np.zeros(3)
        self._lp = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, LEFT_PALM)
        self._rp = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, RIGHT_PALM)
        self._jpl = np.zeros((3, model.nv)); self._jrl = np.zeros((3, model.nv))
        self._jpr = np.zeros((3, model.nv)); self._jrr = np.zeros((3, model.nv))

    def _ik_dof_indices(self):
        idx = [self.model.joint(LIFT_JOINT).dofadr[0],
               self.model.joint("Torso").dofadr[0]]
        for i in range(1, 8):
            idx.append(self.model.joint(f"Left Arm {i}").dofadr[0])
        idx.append(self.model.joint("Left Wrist").dofadr[0])
        for i in range(1, 8):
            idx.append(self.model.joint(f"Right Arm {i}").dofadr[0])
        idx.append(self.model.joint("Right Wrist").dofadr[0])
        return idx

    @staticmethod
    def _joint_of(name):
        return LIFT_JOINT if name == "Lift" else name

    def _q(self, name):
        return self.data.joint(self._joint_of(name)).qpos[0]

    def _q_vec(self, arms_only):
        names = IK_ACTUATORS[2:] if arms_only else IK_ACTUATORS
        return np.array([self._q(n) for n in names])

    def sync_to_current(self):
        for name in IK_ACTUATORS:
            self.target_q[name] = self._q(name)
        self.target_l_pos = self.data.body(LEFT_PALM).xpos.copy()
        self.target_r_pos = self.data.body(RIGHT_PALM).xpos.copy()
        self.target_l_mat = self.data.body(LEFT_PALM).xmat.copy().reshape(3, 3)
        self.target_r_mat = self.data.body(RIGHT_PALM).xmat.copy().reshape(3, 3)

    def _twist(self, p_trg, R_trg, body):
        p_cur = self.data.body(body).xpos
        R_cur = self.data.body(body).xmat.reshape(3, 3)
        e_pos = _deadband(p_trg - p_cur, IK_POS_DEADBAND)
        e_rot = _deadband(_so3_error(R_cur, R_trg), IK_ROT_DEADBAND)
        v = IK_KP_POS * e_pos
        wv = IK_KP_ROT * e_rot
        vn = np.linalg.norm(v)
        if vn > IK_VEL_CAP:
            v *= IK_VEL_CAP / vn
        wn = np.linalg.norm(wv)
        if wn > IK_ANGVEL_CAP:
            wv *= IK_ANGVEL_CAP / wn
        return np.hstack([v, wv]), e_pos

    def solve(self, arms_only=False):
        m, d = self.model, self.data
        if arms_only:
            dofs, W_inv, q_rest = self.arm_dofs, self.W_inv_arms, self.q_rest[2:]
        else:
            dofs, W_inv, q_rest = self.dofs, self.W_inv, self.q_rest
        twist_l, self.e_pos_l = self._twist(self.target_l_pos, self.target_l_mat, LEFT_PALM)
        twist_r, self.e_pos_r = self._twist(self.target_r_pos, self.target_r_mat, RIGHT_PALM)
        e_vel = np.hstack([twist_l, twist_r])
        mujoco.mj_jacBody(m, d, self._jpl, self._jrl, self._lp)
        mujoco.mj_jacBody(m, d, self._jpr, self._jrr, self._rp)
        J = np.vstack([self._jpl, self._jrl, self._jpr, self._jrr])[:, dofs]
        lam = IK_DAMPING
        q_cur = self._q_vec(arms_only)
        dq_post = IK_POSTURE_GAIN * (q_rest - q_cur)
        JW = J @ W_inv
        J_pinv = W_inv @ J.T @ np.linalg.inv(JW @ J.T + (lam ** 2) * np.eye(12))
        return J_pinv @ e_vel + (np.eye(len(dofs)) - (J_pinv @ J)) @ dq_post

    def apply(self, dq, dt, arms_only=False):
        names = self.arm_actuators if arms_only else IK_ACTUATORS
        for i, name in enumerate(names):
            v = dq[i]
            if name == "Lift":
                v = float(np.clip(v, -MAX_LIFT_IK_SPEED, MAX_LIFT_IK_SPEED))
            act_id = self.model.actuator(name).id
            lo, hi = self.model.actuator_ctrlrange[act_id]
            self.target_q[name] = float(np.clip(self.target_q[name] + v * dt, lo, hi))
            self.data.actuator(name).ctrl[0] = self.target_q[name]


# =====================================================================
# Swerve IK (3-module delta base)
# =====================================================================
def _optimize_steering(target_angle, current_angle):
    target_angle = current_angle + ((target_angle - current_angle + math.pi)
                                    % (2 * math.pi) - math.pi)
    if target_angle - current_angle > math.pi / 2:
        return target_angle - math.pi, -1.0
    if target_angle - current_angle < -math.pi / 2:
        return target_angle + math.pi, -1.0
    return target_angle, 1.0


def swerve_ik(vx, vy, omega, data):
    for key, r in MODULE_POSITIONS.items():
        v_ix = vx - omega * r[1]
        v_iy = vy + omega * r[0]
        v_mag = math.hypot(v_ix, v_iy)
        prefix = key.upper()
        cur = data.joint(f"{prefix} Steer").qpos[0]
        if v_mag > 1e-3:
            local = math.atan2(v_iy, v_ix) - MODULE_YAW_OFFSETS[key]
            theta, dir_mult = _optimize_steering(local, cur)
            err = abs(theta - cur)
            drive_scale = 0.0 if err > STEER_DEADBAND else math.cos(err)
            data.actuator(f"{prefix} Steer").ctrl[0] = theta
            data.actuator(f"{prefix} Drive").ctrl[0] = (v_mag / WHEEL_RADIUS) * dir_mult * drive_scale
        else:
            data.actuator(f"{prefix} Steer").ctrl[0] = cur
            data.actuator(f"{prefix} Drive").ctrl[0] = 0.0


# =====================================================================
# Active-perception neck (3-DOF look-at)
# =====================================================================
def look_at_point(origin, forward_dir, model, data, exclude_body=None):
    origin = np.asarray(origin, float)
    d = np.asarray(forward_dir, float)
    n = np.linalg.norm(d)
    if n < 1e-9:
        return origin
    d = d / n
    bodyexclude = -1
    if exclude_body is not None:
        bodyexclude = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, exclude_body)
    pnt = origin + d * NECK_RAY_OFFSET
    geomid = np.zeros(1, dtype=np.int32)
    dist = mujoco.mj_ray(model, data, pnt, d, None, 1, bodyexclude, geomid)
    if dist >= 0.0 and geomid[0] != -1 and dist >= NECK_MIN_HIT_DIST:
        return pnt + d * dist
    return origin + d * NECK_FALLBACK_DIST


class NeckController:
    def __init__(self, model, data):
        self.model, self.data = model, data
        self.neck_dofs = [model.joint(j).dofadr[0] for j in NECK_JOINTS]
        self.head_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, HEAD_BODY)
        self.target_q = {n: data.actuator(n).ctrl[0] for n in NECK_ACTUATORS}

    def gaze_origin(self):
        return self.data.body(HEAD_BODY).xpos.copy()

    def solve_and_apply(self, target_point, dt):
        m, d = self.model, self.data
        p = self.data.body(HEAD_BODY).xpos
        R = self.data.body(HEAD_BODY).xmat.reshape(3, 3)
        f_cur = R @ HEAD_FORWARD_LOCAL
        f_cur /= (np.linalg.norm(f_cur) + 1e-9)
        d_des = np.asarray(target_point, float) - p
        nd = np.linalg.norm(d_des)
        if nd < 1e-6:
            return
        d_des /= nd
        w = NECK_KP * np.cross(f_cur, d_des)
        jacp = np.zeros((3, m.nv)); jacr = np.zeros((3, m.nv))
        mujoco.mj_jacBody(m, d, jacp, jacr, self.head_id)
        Jr = jacr[:, self.neck_dofs]
        lam = 0.05
        dq = Jr.T @ np.linalg.inv(Jr @ Jr.T + (lam ** 2) * np.eye(3)) @ w
        for i, name in enumerate(NECK_ACTUATORS):
            act_id = m.actuator(name).id
            lo, hi = m.actuator_ctrlrange[act_id]
            self.target_q[name] = float(np.clip(self.target_q[name] + dq[i] * dt, lo, hi))
            d.actuator(name).ctrl[0] = self.target_q[name]


# =====================================================================
# Grasp (pinch -> finger curl) -- the simple stand-in for full finger IK
# =====================================================================
def apply_grasp(model, data, side, g):
    g = float(np.clip(g, 0.0, 1.0))
    for suffix in _FLEX_SUFFIXES:
        name = f"{side} {suffix}"
        try:
            act_id = model.actuator(name).id
        except KeyError:
            continue
        lo, hi = model.actuator_ctrlrange[act_id]
        data.actuator(name).ctrl[0] = lo + g * (hi - lo)


# =====================================================================
# Feed-forward gravity compensation over the upper-body chain
# =====================================================================
class GravityFeedforward:
    FF_JOINTS = (["Lift Lower", "Lift Upper", "Torso"]
                 + [f"Left Arm {i}" for i in range(1, 8)] + ["Left Wrist"]
                 + [f"Right Arm {i}" for i in range(1, 8)] + ["Right Wrist"])

    def __init__(self, model, data, gain=1.0):
        self.model, self.data = model, data
        self.dofs = np.array([model.joint(j).dofadr[0] for j in self.FF_JOINTS])
        self.gain = float(gain)
        self.enabled = True
        self._g = np.zeros(model.nv)

    def gravity(self):
        """g(q): generalized gravity force (bias forces evaluated at zero velocity)."""
        qvel = self.data.qvel.copy()
        self.data.qvel[:] = 0.0
        mujoco.mj_rne(self.model, self.data, 0, self._g)   # flg_acc=0, qd=0 -> g(q)
        self.data.qvel[:] = qvel
        return self._g

    def apply(self):
        self.data.qfrc_applied[:] = 0.0
        if not self.enabled:
            return
        g = self.gravity()
        self.data.qfrc_applied[self.dofs] = self.gain * g[self.dofs]


# =====================================================================
# Global base tracker (WHOLE region): operator translation -> swerve base
# =====================================================================
class GlobalBaseTracker:
    """Mirror the operator's GLOBAL head translation onto the swerve base with a
    simple accel-limited P-controller. (The package's tip-over/ZMP-safe smoother is
    intentionally not inlined here -- this is a plain demo follower.)"""

    def __init__(self, model, data):
        self.model, self.data = model, data
        self.calibrated = False
        self.vx = self.vy = 0.0

    def calibrate(self, frame):
        if frame is None or frame.head is None:
            return False
        self.head0 = frame.head[:3, 3].copy()
        self.base0 = self.data.body(CHASSIS_BODY).xpos.copy()
        self.R0 = self.data.body(CHASSIS_BODY).xmat.reshape(3, 3).copy()
        self.calibrated = True
        return True

    def step(self, frame, dt):
        if not self.calibrated or frame is None or frame.head is None:
            self._decel(dt)
            swerve_ik(self.vx, self.vy, 0.0, self.data)
            return
        d_op = AVP_TO_ROBOT @ (frame.head[:3, 3] - self.head0)
        target_world = self.base0 + self.R0 @ np.array(
            [GLOBAL_BASE_SCALE * d_op[0], GLOBAL_BASE_SCALE * d_op[1], 0.0])
        err = target_world - self.data.body(CHASSIS_BODY).xpos
        err[2] = 0.0
        v_world = GLOBAL_BASE_KP * err
        R_now = self.data.body(CHASSIS_BODY).xmat.reshape(3, 3)
        v_body = R_now.T @ v_world                     # (fwd, left, _)
        tvx = float(np.clip(v_body[0], -MAX_LIN_SPEED, MAX_LIN_SPEED))
        tvy = float(np.clip(v_body[1], -MAX_LIN_SPEED, MAX_LIN_SPEED))
        self.vx = _approach(self.vx, tvx, MAX_LIN_ACCEL * dt)
        self.vy = _approach(self.vy, tvy, MAX_LIN_ACCEL * dt)
        swerve_ik(self.vx, self.vy, 0.0, self.data)

    def _decel(self, dt):
        self.vx = _approach(self.vx, 0.0, MAX_LIN_ACCEL * dt)
        self.vy = _approach(self.vy, 0.0, MAX_LIN_ACCEL * dt)


def _approach(cur, tgt, step):
    if cur < tgt:
        return min(tgt, cur + step)
    return max(tgt, cur - step)


# =====================================================================
# Region engine: ties retarget + IK + grasp + neck + base together
# =====================================================================
REGION_ARM, REGION_UPPER, REGION_WHOLE = "arm", "upper", "whole"
REGIONS = [REGION_ARM, REGION_UPPER, REGION_WHOLE]
REGION_LABEL = {REGION_ARM: "1 ARM  (arms + hands)",
                REGION_UPPER: "2 UPPER  (+ lift + torso)",
                REGION_WHOLE: "3 WHOLE  (+ base global tracking)"}


class RetargetTeleop:
    """The whole control stack for one robot, region-switchable."""

    def __init__(self, model, data):
        self.model, self.data = model, data
        self.retgt = Retargeter(model, data)
        self.ik = WholeBodyIK(model, data)
        self.neck = NeckController(model, data)
        self.tracker = GlobalBaseTracker(model, data)
        self.region = REGION_ARM
        self.grasp_l = self.grasp_r = 0.0
        self._tgt = None

    @property
    def calibrated(self):
        return self.retgt.calibrated

    def calibrate(self, frame):
        ok = self.retgt.calibrate(frame)
        if ok:
            self.ik.sync_to_current()
            self.tracker.calibrate(frame)
        return ok

    def hold_current(self):
        self.ik.sync_to_current()

    def step(self, frame, dt):
        targets = self.retgt.update(frame) if self.retgt.calibrated else None
        self._tgt = targets
        if targets is not None:
            if "r_pos" in targets:
                self.ik.target_r_pos = targets["r_pos"]
                self.ik.target_r_mat = targets["r_mat"]
                self.grasp_r = targets.get("grasp_r", self.grasp_r)
            if "l_pos" in targets:
                self.ik.target_l_pos = targets["l_pos"]
                self.ik.target_l_mat = targets["l_mat"]
                self.grasp_l = targets.get("grasp_l", self.grasp_l)

        # Whole-body IK. ARM region keeps the lift + torso perfectly still.
        arms_only = (self.region == REGION_ARM)
        dq = self.ik.solve(arms_only=arms_only)
        self.ik.apply(dq, dt, arms_only=arms_only)

        # Fingers (pinch-grasp stand-in).
        apply_grasp(self.model, self.data, "Left", self.grasp_l)
        apply_grasp(self.model, self.data, "Right", self.grasp_r)

        # Active perception: aim the neck at the operator's gaze look-at point.
        if targets is not None and "gaze_dir" in targets:
            pt = look_at_point(self.neck.gaze_origin(), targets["gaze_dir"],
                               self.model, self.data, exclude_body=HEAD_BODY)
            self.neck.solve_and_apply(pt, dt)

        # WHOLE region: the swerve base tracks the operator's global translation.
        if self.region == REGION_WHOLE:
            self.tracker.step(frame, dt)
        else:
            swerve_ik(0.0, 0.0, 0.0, self.data)


# =====================================================================
# Keyboard + main loop
# =====================================================================
region_idx = 0
events = {"calibrate": False, "reset": False, "toggle_ff": False}


def _make_listener():
    from pynput import keyboard

    def on_press(key):
        global region_idx
        try:
            ch = key.char.lower()
            if ch == "c":
                events["calibrate"] = True
            elif ch == "g":
                events["toggle_ff"] = True
            elif ch in "123":
                region_idx = int(ch) - 1
        except AttributeError:
            if key == keyboard.Key.tab:
                region_idx = (region_idx + 1) % len(REGIONS)
            elif key == keyboard.Key.backspace:
                events["reset"] = True
    return keyboard.Listener(on_press=on_press)


def settle(ctrl, ff, steps=IK_SETTLE_STEPS):
    """Hold the start pose on the servos + gravity feedforward until equilibrium,
    then recapture the hold so it does not creep on launch."""
    ctrl.hold_current()
    for _ in range(steps):
        ff.apply()
        mujoco.mj_step(ctrl.model, ctrl.data)
    ctrl.hold_current()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=("avp", "mock"), default="avp")
    ap.add_argument("--addr", default=AVP_ZMQ_ADDR, help="AVP ZMQ publisher address")
    ap.add_argument("--scene", default=SCENE_XML, help="MJCF scene to load")
    ap.add_argument("--ff-gain", type=float, default=1.0, help="gravity FF scale (0 = pure PD)")
    ap.add_argument("--timestep", type=float, default=0.004, help="physics timestep [s]")
    args = ap.parse_args()

    import mujoco.viewer

    model = mujoco.MjModel.from_xml_path(args.scene)
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_SENSOR     # skip lidar raycasts
    model.opt.timestep = args.timestep
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)

    ctrl = RetargetTeleop(model, data)
    ff = GravityFeedforward(model, data, gain=args.ff_gain)
    settle(ctrl, ff)

    recv = AVPReceiver(args.addr) if args.source == "avp" else None
    _make_listener().start()

    B, Y, Cc, E = "\033[1m", "\033[93m", "\033[96m", "\033[0m"
    print(f"\n{B}{Cc}MABEL retargeting (self-contained) — Vision Pro → whole-body{E}"
          f"  (source: {args.source})")
    print(f"  {Y}REGION{E} TAB cycle · 1 ARM  2 UPPER  3 WHOLE")
    print(f"  {Y}CLUTCH{E} C re-anchor · {Y}GRAVITY{E} G toggle FF · BACKSPACE reset\n")

    last_ui = 0.0
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 3.0
        viewer.cam.lookat = np.array([0.0, 0.0, 0.8])
        t0 = time.time()
        prev = time.time()
        acc = 0.0
        while viewer.is_running():
            frame = (mock_frame(time.time() - t0) if args.source == "mock" else recv.poll())

            # Auto-anchor on the first valid frame (no manual clutch needed).
            if frame is not None and not ctrl.calibrated:
                ctrl.calibrate(frame)
            if events["calibrate"]:
                events["calibrate"] = False
                if frame is not None:
                    ctrl.calibrate(frame)
            if events["toggle_ff"]:
                events["toggle_ff"] = False
                ff.enabled = not ff.enabled
            if events["reset"]:
                events["reset"] = False
                mujoco.mj_resetData(model, data)
                mujoco.mj_forward(model, data)
                settle(ctrl, ff)
                ctrl.retgt.calibrated = False
                ctrl.tracker.calibrated = False
                t0 = time.time() - data.time
                prev = time.time(); acc = 0.0

            ctrl.region = REGIONS[region_idx]

            # Real-time accumulator: advance physics by REAL elapsed wall-time,
            # capped, so the sim chases the freshest hand frame (low latency) and
            # runs in slight slow-motion if the model can't keep up. IK runs per
            # substep, exactly like the server bridge.
            now = time.time()
            acc += min(now - prev, MAX_CATCHUP)
            prev = now
            while acc >= model.opt.timestep:
                ff.apply()
                ctrl.step(frame, model.opt.timestep)
                mujoco.mj_step(model, data)
                acc -= model.opt.timestep

            viewer.sync()
            if acc < model.opt.timestep:
                time.sleep(0.002)

            if time.time() - last_ui > 0.2:
                last_ui = time.time()
                el = float(np.linalg.norm(ctrl.ik.e_pos_l)) * 1000.0
                er = float(np.linalg.norm(ctrl.ik.e_pos_r)) * 1000.0
                stream = "ok" if frame is not None else "waiting"
                anch = "anchored" if ctrl.calibrated else "auto-anchoring"
                print(f"\r[{REGION_LABEL[ctrl.region]:28s}] stream {stream:8s} "
                      f"{anch:14s} palm err L/R {el:5.1f}/{er:5.1f} mm  "
                      f"grasp {ctrl.grasp_l:.2f}/{ctrl.grasp_r:.2f}  "
                      f"FF {'on ' if ff.enabled else 'off'}   ", end="", flush=True)

    if recv is not None:
        recv.close()


if __name__ == "__main__":
    main()
