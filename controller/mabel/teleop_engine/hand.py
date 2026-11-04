"""Decoupled per-hand finger retargeting (cosine + relative-distance SLSQP)."""
import time
import numpy as np
import mujoco
from scipy.optimize import minimize

from .hyperparams import *
from .hyperparams import _GROUP
from .geometry import _natural_frame


class HandOptimizer:
    def __init__(self, model, side):
        """side: 'right' or 'left'."""
        self.model = model
        self.ik_data = mujoco.MjData(model)
        self.side = side
        s = side

        self.palm_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{s}_palm")
        if self.palm_bid == -1:
            raise ValueError(f"Palm body '{s}_palm' not found in {XML_FILE}")

        # 21-landmark index -> ORCA finger body (knuckle / proximal / tip per finger).
        body_mapping = {
            1: f"{s}_thumb_mp",  2: f"{s}_thumb_pp",  3: f"{s}_thumb_ip",  4: f"{s}_thumb_dp",
            5: f"{s}_index_mp",  6: f"{s}_index_pp",  8: f"{s}_index_ip",
            9: f"{s}_middle_mp", 10: f"{s}_middle_pp", 12: f"{s}_middle_ip",
            13: f"{s}_ring_mp",  14: f"{s}_ring_pp",  16: f"{s}_ring_ip",
            17: f"{s}_pinky_mp", 18: f"{s}_pinky_pp", 20: f"{s}_pinky_ip",
        }
        self.body_ids = {}
        for idx, name in body_mapping.items():
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid != -1:
                self.body_ids[idx] = bid

        # 16 finger DOFs/hand (the 17th, "<Side> Wrist", is owned by the arm IK).
        S = side.capitalize()
        joint_names = [f"{S} Thumb MCP", f"{S} Thumb Abd", f"{S} Thumb PIP", f"{S} Thumb DIP"]
        for finger in ("Index", "Middle", "Ring", "Pinky"):
            joint_names += [f"{S} {finger} Abd", f"{S} {finger} MCP", f"{S} {finger} PIP"]
        self.joint_names = joint_names
        self.joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joint_names]
        self.joint_ids = [j for j in self.joint_ids if j != -1]
        self.q_indices = [model.jnt_qposadr[j] for j in self.joint_ids]
        self.bounds = [tuple(model.jnt_range[j]) for j in self.joint_ids]
        self.last_q = np.zeros(len(self.joint_ids))

        # Robot natural palm frame, expressed in palm-LOCAL coords (constant — the
        # finger mounts are rigid in the palm). Built from the same anatomy the
        # operator frame uses, so the two are directly comparable.
        d0 = mujoco.MjData(model)
        mujoco.mj_kinematics(model, d0)
        pp = d0.xpos[self.palm_bid]
        pm = d0.xmat[self.palm_bid].reshape(3, 3)
        def loc(name):
            b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            return pm.T @ (d0.xpos[b] - pp)
        self.R_rob = _natural_frame(np.zeros(3), loc(f"{s}_middle_mp"),
                                    loc(f"{s}_index_mp"), loc(f"{s}_pinky_mp"))
        # ── flexion-anchored chirality (measured fix, 2026-07-02) ────────────
        # _natural_frame carries chirality via the index→pinky handedness. A
        # TRUE human hand always curls toward +n̂ of its own natural frame, but
        # the ORCA "{side}" palm geometry is MIRRORED (a left-geometry right
        # hand), so its index→pinky handedness flips n̂ — every human curl then
        # maps to a robot EXTENSION ("fingers do not pitch"), and the old
        # client-side palm reflection that papered over this broke two-hand
        # fingertip contact (heart gap 329 mm vs 76 mm). Anchor n̂ to physics
        # instead of topology: flex all MCP/PIPs and demand the fingertips move
        # toward +n̂; if they don't, flip ŝ and n̂ (still right-handed, det +1).
        # --- precomputed index arrays for the VECTORISED cost (hot path) ----------
        # The optimiser evaluates the cost ~16x per SLSQP gradient; doing it as a
        # per-body Python loop (the old robot_local_points) dominated the frame time.
        # These let _robot_arr / cost_function run as a few batched numpy ops instead.
        self._qadr = np.asarray(self.q_indices, dtype=np.intp)
        self._idx = sorted(self.body_ids.keys())                 # landmark order (ascending)
        self._bid = np.asarray([self.body_ids[i] for i in self._idx], dtype=np.intp)
        self._tip = np.asarray([i in TIP_INDICES for i in self._idx])
        self._pad = np.array([0.0, 0.0, 0.022])                  # fingertip-pad extension
        self._grp_scale = np.asarray([SCALES[_GROUP[i]] for i in self._idx])[:, None]
        _pos = {idx: k for k, idx in enumerate(self._idx)}        # landmark -> array row
        self._rel_a = np.asarray([_pos[a] for a, b in RELATIVE_PAIRS
                                  if a in _pos and b in _pos], dtype=np.intp)
        self._rel_b = np.asarray([_pos[b] for a, b in RELATIVE_PAIRS
                                  if a in _pos and b in _pos], dtype=np.intp)
        self._anchor_normal_to_flexion(model)

    def _anchor_normal_to_flexion(self, model):
        d = mujoco.MjData(model)
        tips = [self.body_ids[i] for i in (8, 12, 16, 20) if i in self.body_ids]
        if not tips:
            return
        def tip_pts(q):
            d.qpos[:] = 0.0
            for qa, v in zip(self.q_indices, q):
                d.qpos[qa] = v
            mujoco.mj_kinematics(model, d)
            palm_pos = d.xpos[self.palm_bid]
            palm_R = d.xmat[self.palm_bid].reshape(3, 3)
            return (d.xpos[tips] - palm_pos) @ palm_R    # palm-local (N,3)
        q0 = np.zeros(len(self.joint_ids))
        qf = q0.copy()
        for k, name in enumerate(self.joint_names):
            if "MCP" in name or "PIP" in name:
                lo, hi = self.bounds[k]
                qf[k] = lo + 0.7 * (hi - lo)             # toward the flex end
        delta = (tip_pts(qf) - tip_pts(q0)) @ self.R_rob  # in the natural frame
        if float(np.mean(delta[:, 2])) < 0.0:             # curl must go +n̂
            self.R_rob = self.R_rob @ np.diag([1.0, -1.0, -1.0])

    def reset_optimizer(self):
        self.last_q = np.zeros(len(self.joint_ids))

    def robot_local_points(self, q):
        """ORCA finger-body positions in the canonical palm frame for joints q, as an
        (N,3) array aligned with ``self._idx``. Vectorised (no per-body Python loop):
        gather all tracked bodies at once and transform them in a single batched op."""
        self.ik_data.qpos[self._qadr] = q
        mujoco.mj_kinematics(self.model, self.ik_data)
        palm_pos = self.ik_data.xpos[self.palm_bid]
        palm_R = self.ik_data.xmat[self.palm_bid].reshape(3, 3)
        gp = self.ik_data.xpos[self._bid].copy()                      # (N,3) world
        if self._tip.any():                                          # extend tips to the pad
            xmats = self.ik_data.xmat[self._bid].reshape(-1, 3, 3)
            gp[self._tip] += xmats[self._tip] @ self._pad
        # R_rob.T @ palm_R.T @ (gp - palm_pos), batched over all rows
        return ((gp - palm_pos) @ palm_R) @ self.R_rob

    def cost_function(self, q, human_arr, q_prev):
        r = self.robot_local_points(q)
        rf, hf = r.ravel(), human_arr.ravel()
        cos = float(rf @ hf) / ((np.linalg.norm(rf) + 1e-6) * (np.linalg.norm(hf) + 1e-6))
        loss = W_COSINE * (1.0 - cos)
        if self._rel_a.size:                                         # pinch geometry (thumb<->tips)
            dr = r[self._rel_b] - r[self._rel_a]
            dh = human_arr[self._rel_b] - human_arr[self._rel_a]
            loss += W_RELATIVE * float(np.sum((dr - dh) ** 2))
        loss += W_SMOOTH * float(np.sum((q - q_prev) ** 2))
        loss += W_REG * float(np.sum(q ** 2))
        return loss

    def solve(self, h_mats, R_op):
        """h_mats: list of 21 4x4 (ARKit). R_op: operator natural palm frame (3x3,
        ARKit world). Returns (q, solve_ms, human_arr) with the operator keypoints in
        the canonical palm frame (scaled); the 3rd value is for debug/overlay."""
        t0 = time.perf_counter()
        H = np.asarray(h_mats, float).reshape(-1, 4, 4)
        wrist = H[0, :3, 3]
        # operator finger keypoints in the canonical palm frame, scaled per finger group:
        # R_op.T @ (p - wrist) for every tracked landmark, batched, then * per-group scale.
        human_arr = ((H[self._idx, :3, 3] - wrist) @ R_op) * self._grp_scale
        res = minimize(self.cost_function, self.last_q, args=(human_arr, self.last_q),
                       method="SLSQP", bounds=self.bounds, options={"maxiter": 8, "ftol": 1e-3})
        self.last_q = res.x
        return res.x, (time.perf_counter() - t0) * 1e3, human_arr
