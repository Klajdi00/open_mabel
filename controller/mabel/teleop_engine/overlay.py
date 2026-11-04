"""Tracking overlay: operator ghost hands + arm balls + gaze ray."""
import numpy as np
import mujoco

from .hyperparams import *
from .hyperparams import _GROUP


class Overlay:
    R_RGBA = (0.10, 0.90, 0.30, 1.0)     # right hand ghost
    L_RGBA = (1.00, 0.45, 0.20, 1.0)     # left hand ghost

    def __init__(self, model):
        self.model = model

    def _sphere(self, scn, pos, r, rgba):
        if scn.ngeom >= scn.maxgeom:
            return
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([r, r, r]),
                            np.asarray(pos, float), np.eye(3).flatten(), np.array(rgba, float))
        scn.ngeom += 1

    def _bone(self, scn, p0, p1, w, rgba):
        if scn.ngeom >= scn.maxgeom:
            return
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                            np.zeros(3), np.eye(3).flatten(), np.array(rgba, float))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, w,
                             np.asarray(p0, float), np.asarray(p1, float))
        scn.ngeom += 1

    def hand(self, scn, opt, data, h_mats, R_op, rgba):
        """Operator hand placed ON the robot palm, in the canonical frame, so the
        ghost overlays the robot fingers. Green/orange spheres + bones = operator;
        red spheres = the (scaled) solver fingertip targets."""
        H = [np.asarray(m, float).reshape(4, 4) for m in h_mats]
        wrist = H[0][:3, 3]
        palm_pos = data.xpos[opt.palm_bid].copy()
        palm_mat = data.xmat[opt.palm_bid].reshape(3, 3).copy()
        def to_world(v_nat):
            return palm_pos + palm_mat @ (opt.R_rob @ v_nat)
        pts = [to_world(R_op.T @ (H[i][:3, 3] - wrist)) for i in range(min(21, len(H)))]
        faint = (rgba[0], rgba[1], rgba[2], 0.5)
        for a, b in HAND_LINKS:
            if a < len(pts) and b < len(pts):
                self._bone(scn, pts[a], pts[b], 0.004, faint)
        for i, p in enumerate(pts):
            self._sphere(scn, p, 0.009 if i in TIP_INDICES else 0.006, rgba)
        for idx in opt.body_ids:                          # red solver targets (scaled)
            if idx in TIP_INDICES and idx < len(H):
                v = R_op.T @ (H[idx][:3, 3] - wrist) * SCALES[_GROUP[idx]]
                self._sphere(scn, to_world(v), 0.007, (1.0, 0.1, 0.1, 0.7))

    def arm_ball(self, scn, target_pos, rgba):
        self._sphere(scn, target_pos, 0.03, (rgba[0], rgba[1], rgba[2], 0.6))

    def gaze(self, scn, origin, direction):
        end = np.asarray(origin, float) + 1.2 * np.asarray(direction, float)
        self._bone(scn, origin, end, 0.004, (0.4, 0.7, 1.0, 0.6))
        self._sphere(scn, end, 0.025, (0.4, 0.7, 1.0, 0.9))

    def _triad(self, scn, pos, R, length=0.1):
        for ax, col in ((0, (1, 0.2, 0.2, 1)), (1, (0.2, 1, 0.2, 1)), (2, (0.3, 0.5, 1, 1))):
            self._bone(scn, pos, np.asarray(pos, float) + length * R[:, ax], 0.005, col)

    def tracking(self, scn, payload, world_ref, reach):
        """Overlay the operator's EXACT tracked HEAD + HAND skeletons, mapped into the
        robot world by the FIXED Vision-Pro-world -> robot-world transform the arms
        use (so the overlay is in the VP WORLD frame and shows global walk / turn).
        world_ref = (avp_head0, robot_head0); reach = the live REACH_SCALE."""
        def M(p):                                            # VP world -> robot world (fixed)
            return world_ref[1] + reach * (AVP_TO_ROBOT @ (np.asarray(p, float) - world_ref[0]))
        hm = payload.get("head_mat")
        head_w = M(np.asarray(hm, float).reshape(4, 4)[:3, 3]) if hm else world_ref[1]
        self._sphere(scn, head_w, 0.05, (0.4, 0.7, 1.0, 0.5))    # operator head (world-mapped)
        # hand skeletons at their true world-mapped positions
        for mk, rgba in (("right_hand_mats", self.R_RGBA), ("left_hand_mats", self.L_RGBA)):
            h_mats = payload.get(mk)
            if not h_mats:
                continue
            pts = [M(np.asarray(m, float).reshape(4, 4)[:3, 3]) for m in h_mats[:21]]
            faint = (rgba[0], rgba[1], rgba[2], 0.5)
            self._bone(scn, head_w, pts[0], 0.003, faint)        # head <-> wrist link
            for a, b in HAND_LINKS:
                if a < len(pts) and b < len(pts):
                    self._bone(scn, pts[a], pts[b], 0.004, faint)
            for i, p in enumerate(pts):
                r = 0.013 if i == 0 else (0.009 if i in TIP_INDICES else 0.006)
                self._sphere(scn, p, r, rgba)
