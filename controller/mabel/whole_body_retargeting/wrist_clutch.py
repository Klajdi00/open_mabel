"""
Vision Pro -> MABEL retargeting.

A clutch-based *relative* end-effector mapping (UMI-style), so it is robust to
the absolute frame mismatch between ARKit and the robot:

  * On `calibrate()` we snapshot the operator's wrist/head poses and the robot's
    current palm/head poses. From then on the targets equal the robot's current
    state plus the operator's motion *since* calibration, remapped into the robot
    frame and scaled (GMR non-uniform local scaling) to cover MABEL's workspace.
  * Head motion is reduced to a gaze direction that feeds the look-at-point neck
    controller (HoMMI active-perception relaxation) — never a raw 6-DOF copy.

Re-press the clutch any time to re-anchor without a jump.
"""
import numpy as np
import mujoco

from .. import config as C
from .arm_cosine import _natural_frame


def _clamp_in_front(p_world, data, base_bid):
    """Clamp a world target so it stays IN FRONT of the chassis front line (robot
    front = -x at chassis-x = ARM_WRIST_FRONT): the contact target can't be driven
    behind the robot and twist the arm to the back. Ported from teleop/geometry.py."""
    o = data.xpos[base_bid]
    R = data.xmat[base_bid].reshape(3, 3)
    loc = R.T @ (np.asarray(p_world, float) - o)
    if loc[0] > C.ARM_WRIST_FRONT:
        loc[0] = C.ARM_WRIST_FRONT
        return o + R @ loc
    return np.asarray(p_world, float)


def _pinch_grasp(hand_mats, open_d=None, close_d=None):
    """Map thumb-index pinch distance to a grasp scalar in [0, 1]. ``open_d`` /
    ``close_d`` default to the global config spans but can be overridden with the
    operator's CALIBRATED pinch open/close gap (so a small and a large hand both
    reach grasp=0/1 at their own comfortable extremes)."""
    if hand_mats is None or len(hand_mats) < 9:
        return 0.0
    open_d = C.PINCH_OPEN_DIST if open_d is None else open_d
    close_d = C.PINCH_CLOSE_DIST if close_d is None else close_d
    thumb_tip = hand_mats[4][:3, 3]
    index_tip = hand_mats[8][:3, 3]
    dist = np.linalg.norm(thumb_tip - index_tip)
    g = (open_d - dist) / max(open_d - close_d, 1e-4)
    return float(np.clip(g, 0.0, 1.0))


class Retargeter:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.M = C.AVP_TO_ROBOT
        # Effective rotation remap = world axis remap composed with the wrist-frame
        # alignment, so operator wrist rotations land on the palm with the right
        # axes. WRIST_ALIGN defaults to identity (no change); set it in config.py
        # if the headset's ARKit wrist axes are offset from the palm.
        self.M_rot = C.AVP_TO_ROBOT @ C.WRIST_ALIGN
        # Workspace gain. Default = the fixed per-axis RETARGET_SCALE; once a per-user
        # OperatorProfile is set (set_profile), this becomes the scalar gamma =
        # robot_arm_reach / human_arm_reach so reach maps 1:1 regardless of operator
        # size (ACE-style; arXiv 2408.11805). See calibration.py.
        self.scale = C.RETARGET_SCALE
        self.profile = None
        self.gamma = None
        self.anchor = C.RETARGET_ANCHOR     # "clutch" (default) | "shoulder"
        self._head = None                   # latest head pose (for the shoulder anchor)
        self.pinch_open = {}                # per-side calibrated pinch span ('l'/'r')
        self.pinch_close = {}
        # Fixed Vision-Pro-world -> robot-world transform for the cosine arm path
        # (avp_teleop.py parity): (operator_head, robot_head) captured at calibration,
        # so the contact-point target tracks GLOBAL translation (walking), not just
        # head-relative motion. Re-anchored on every clutch/recalibrate.
        self._world_ref = None
        self._base_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, C.CHASSIS_BODY)
        self.calibrated = False
        # body_relative=False (default): the palm target is anchored in the WORLD
        # frame, so when the base drives the target stays put. True: the target is
        # anchored in the robot BODY (chassis) frame, so it rides with the base —
        # the operator's wrist offset is held relative to the moving robot body
        # (what a mobile-base teleop wants). The apps/bridge keep the world default.
        self.body_relative = False
        # orient_skeleton=True: derive the palm ORIENTATION from the operator's
        # hand GEOMETRY (a fingers-forward / palm-normal "natural frame" built
        # from the streamed knuckles) aligned to the robot's own natural palm
        # frame from FK — so the robot hand matches the operator's hand absolutely,
        # regardless of the ARKit-wrist vs ORCA-palm axis conventions (no hand-
        # specific R_offset to guess). False keeps the legacy relative-rotation
        # clutch (apps/bridge default). Needs the full named skeleton.
        self.orient_skeleton = False
        self.rot_diag = {}     # per-hand (operator_angle, palm_angle) deg, for tuning
        # Anchors captured at calibration.
        self._avp = {}     # 'wl','wr' positions; 'Rwl','Rwr','Rh' rotations
        self._rob = {}     # 'l_pos','r_pos','l_mat','r_mat','gaze' (robot frame)

    # ── per-operator profile / workspace scale ────────────────────────────────
    def robot_arm_reach(self):
        """Rest-pose robot arm reach [m] = mean shoulder-body -> palm-body distance.
        Computed live from the model so it tracks the XML, not a hard-coded number."""
        reaches = []
        for sh, palm in ((C.LEFT_SHOULDER, C.LEFT_PALM),
                         (C.RIGHT_SHOULDER, C.RIGHT_PALM)):
            try:
                ps = self.data.body(sh).xpos
                pp = self.data.body(palm).xpos
                reaches.append(float(np.linalg.norm(pp - ps)))
            except KeyError:
                continue
        return float(np.mean(reaches)) if reaches else 0.0

    def set_profile(self, profile, anchor=None):
        """Apply a per-operator OperatorProfile: derive the workspace scale gamma from
        the operator's measured arm length vs the robot's reach, and (optionally)
        select the arm anchor. Pass profile=None to revert to the fixed default scale."""
        self.profile = profile
        if anchor is not None:
            self.anchor = anchor
        if profile is None:
            self.gamma = None
            self.scale = C.RETARGET_SCALE
            self.pinch_open, self.pinch_close = {}, {}
            return None
        L = profile.arm_length_m
        reach = self.robot_arm_reach()
        # Absolute 1:1 reach match (used by the shoulder-anchored mode).
        self.gamma = profile.gamma(reach) if reach > 0 else 1.0
        # Clutch mode: keep the tuned per-axis amplification, corrected to the
        # operator's arm length relative to the reference operator.
        corr = (float(np.clip(C.ANTHRO_ARM_REF / L, C.RETARGET_GAMMA_MIN,
                              C.RETARGET_GAMMA_MAX)) if L > 1e-3 else 1.0)
        self.scale = C.RETARGET_SCALE * corr
        # Per-operator pinch spans (left/right -> l/r) for the grasp curve.
        self.pinch_open = {s: profile.pinch_open_m[k]
                           for s, k in (("l", "left"), ("r", "right"))
                           if k in profile.pinch_open_m}
        self.pinch_close = {s: profile.pinch_close_m[k]
                            for s, k in (("l", "left"), ("r", "right"))
                            if k in profile.pinch_close_m}
        return self.gamma

    def _grasp(self, mats, side):
        """Per-side grasp scalar using the operator's calibrated pinch span."""
        return _pinch_grasp(mats, self.pinch_open.get(side), self.pinch_close.get(side))

    def _shoulder_anchored(self, side, w_now):
        """Absolute shoulder-anchored palm target (RETARGET_ANCHOR == 'shoulder'):
            p_ee = c_sh_robot + gamma * AVP_TO_ROBOT @ (p_wrist - c_sh_human)
        Anchors the operator's wrist offset from their OWN (calibrated) shoulder onto
        the robot's shoulder, so reach is anatomically faithful across operators.
        Returns None (caller falls back to the clutch mapping) if the inputs are
        missing/degenerate."""
        if self.profile is None or self._head is None:
            return None
        ssh = C.LEFT_SHOULDER if side == "l" else C.RIGHT_SHOULDER
        try:
            c_sh_robot = self.data.body(ssh).xpos.copy()
        except KeyError:
            return None
        c_sh_human = self.profile.shoulder_world(side, self._head)
        g = self.gamma if self.gamma is not None else 1.0
        return c_sh_robot + g * (self.M @ (w_now - c_sh_human))

    # ── natural palm frame (fingers-forward / palm-normal), geometry-derived ──
    @staticmethod
    def _natural_frame(origin, i_kn, p_kn, m_kn):
        """Orthonormal frame from a palm: ŷ=forward (wrist→middle knuckle),
        ẑ=palm normal (index×pinky knuckle rays), x̂ completes. Built identically
        on the human skeleton and the robot FK, so the two are directly comparable
        (the same idea as hand_ik_retarget's canonical frame, for the WRIST)."""
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
        nrm = np.cross(side, fwd)               # re-orthogonalise
        return np.column_stack([side, fwd, nrm])

    def _op_frame(self, joints):
        """Operator natural palm frame from the streamed named skeleton."""
        if not joints:
            return None
        def P(n):
            T = joints.get(n)
            return None if T is None else np.asarray(T, float)[:3, 3]
        w, i, p, m = P("wrist"), P("index_knuckle"), P("little_knuckle"), P("middle_knuckle")
        if any(v is None for v in (w, i, p, m)):
            return None
        return self._natural_frame(w, i, p, m)

    def _rob_frame(self, side):
        """Robot natural palm frame from FK knuckle (proximal-phalanx) body poses.
        Body ids are cached (MuJoCo name lookups are ~150 µs each)."""
        ids = self._rob_frame_ids(side)
        if ids is None:
            return None
        xpos = self.data.xpos
        return self._natural_frame(xpos[ids[0]], xpos[ids[1]], xpos[ids[2]], xpos[ids[3]])

    def _rob_frame_ids(self, side):
        cache = getattr(self, "_rfids", None)
        if cache is None:
            cache = self._rfids = {}
        if side not in cache:
            pre = "left" if side == "l" else "right"
            palm = C.LEFT_PALM if side == "l" else C.RIGHT_PALM
            names = [palm, f"{pre}_index_pp", f"{pre}_pinky_pp", f"{pre}_middle_pp"]
            ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n) for n in names]
            cache[side] = None if any(i < 0 for i in ids) else ids
        return cache[side]

    def _joints_for(self, frame, side):
        return frame.right_joints if side == "r" else frame.left_joints

    # ── calibration ──────────────────────────────────────────────────────────
    def calibrate(self, frame):
        """Snapshot operator + robot state to re-anchor the relative mapping.

        Hands absent from this frame have their anchors DROPPED (not kept
        stale): `update()` late-anchors each hand the moment it first appears,
        so a headset that enables hand streaming after connecting — or loses a
        hand mid-session — re-anchors that hand with no jump. Without this,
        the first (hands-free) frame auto-calibrated head-only and the wrists
        never tracked."""
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

        self._rob["l_pos"] = self.data.body(C.LEFT_PALM).xpos.copy()
        self._rob["r_pos"] = self.data.body(C.RIGHT_PALM).xpos.copy()
        self._rob["l_mat"] = self.data.body(C.LEFT_PALM).xmat.copy().reshape(3, 3)
        self._rob["r_mat"] = self.data.body(C.RIGHT_PALM).xmat.copy().reshape(3, 3)
        R_head = self.data.body(C.HEAD_BODY).xmat.reshape(3, 3)
        self._rob["gaze"] = R_head @ C.HEAD_FORWARD_LOCAL
        # Chassis orientation at calibration. The gaze is held relative to THIS
        # frame, so when the base drives/turns the neck stays body-relative
        # instead of locking onto a world-fixed direction.
        self._rob["Rbase0"] = self.data.body(C.CHASSIS_BODY).xmat.reshape(3, 3).copy()
        # Geometry-derived orientation alignment K per hand: K maps the palm-BODY
        # frame to the robot's natural palm frame (F_rob = R_palm_body · K). The
        # target palm orientation is then R_palm_target = F_op · Kᵀ, which forces
        # the robot's natural frame to equal the operator's (F_rob = F_op) — an
        # ABSOLUTE orientation match that self-aligns here, no R_offset to guess.
        for side in ("r", "l"):
            self._rob.pop(f"{side}_K", None)
            F_op = self._op_frame(self._joints_for(frame, side))
            F_rob = self._rob_frame(side)
            if F_op is not None and F_rob is not None:
                palm = C.LEFT_PALM if side == "l" else C.RIGHT_PALM
                R_palm = self.data.body(palm).xmat.reshape(3, 3)
                self._rob[f"{side}_K"] = R_palm.T @ F_rob
        # Fixed world transform for the cosine arm path: operator head <-> robot head.
        self._world_ref = (frame.head[:3, 3].copy(),
                           self.data.body(C.HEAD_BODY).xpos.copy())
        self.calibrated = True
        return True

    # ── cosine arm retargeting keypoints (avp_teleop.py parity) ───────────────
    def arm_hk(self, side, frame):
        """Operator arm keypoints in the robot WORLD for the cosine ArmRetargeter:
        ``{contact, elbow, fwd, up}``. The CONTACT point (~middle fingertip) is mapped
        through the fixed world transform so the robot hands meet when the operator's
        fingertips do; the palm forward/normal axes give the orientation cosine. Returns
        None (caller holds the arm) when the world ref or the named skeleton is missing.
        side: 'l' | 'r'."""
        if self._world_ref is None or frame is None:
            return None
        joints = self._joints_for(frame, side)
        if not joints:
            return None

        def P(n):
            T = joints.get(n)
            return None if T is None else np.asarray(T, float).reshape(4, 4)[:3, 3]

        w, mid_kn, idx_kn, pky_kn, mid_tip = (P("wrist"), P("middle_knuckle"),
                                              P("index_knuckle"), P("little_knuckle"),
                                              P("middle_tip"))
        if any(v is None for v in (w, mid_kn, idx_kn, pky_kn, mid_tip)):
            return None
        R_op = _natural_frame(w, mid_kn, idx_kn, pky_kn)
        if R_op is None:
            return None

        op_head, rob_head = self._world_ref
        reach = float(self.gamma) if self.gamma is not None else C.REACH_SCALE

        def Wmap(p):                                  # VP world point -> robot world
            return rob_head + reach * (self.M @ (np.asarray(p, float) - op_head))

        contact = Wmap(mid_tip)
        if self._base_bid >= 0:
            contact = _clamp_in_front(contact, self.data, self._base_bid)
        R_palm_canon = self.M @ R_op                  # operator hand frame, robot world
        elbow = None
        e = getattr(frame, "right_elbow" if side == "r" else "left_elbow", None)
        if e is not None:
            elbow = Wmap(np.asarray(e, float).reshape(4, 4)[:3, 3])
        return {"contact": contact, "elbow": elbow,
                "fwd": R_palm_canon[:, 0], "up": R_palm_canon[:, 2]}

    def _anchor_hand(self, side, mats):
        """Anchor one hand's clutch: operator wrist pose now + the robot's
        CURRENT palm pose (so the mapping starts exactly where the robot is).
        side: 'l' | 'r'; mats: the hand's transform list (index 0 = wrist)."""
        self._avp[f"w{side}"] = mats[0][:3, 3].copy()
        self._avp[f"Rw{side}"] = mats[0][:3, :3].copy()
        palm = C.LEFT_PALM if side == "l" else C.RIGHT_PALM
        self._rob[f"{side}_pos"] = self.data.body(palm).xpos.copy()
        self._rob[f"{side}_mat"] = self.data.body(palm).xmat.copy().reshape(3, 3)
        # Palm anchor expressed in the chassis frame AT ANCHOR TIME, for
        # body-relative tracking (frame-invariant offset of the palm from the body).
        R_base = self.data.body(C.CHASSIS_BODY).xmat.reshape(3, 3)
        p_base = self.data.body(C.CHASSIS_BODY).xpos
        self._rob[f"{side}_pos_body"] = R_base.T @ (self._rob[f"{side}_pos"] - p_base)

    # ── per-frame update ─────────────────────────────────────────────────────
    def _map_pos(self, w_now, w0, r0):
        delta_robot = self.M @ (w_now - w0)
        return r0 + self.scale * delta_robot

    def _map_pos_side(self, side, w_now):
        """Operator wrist position → robot palm target. The operator's displacement
        since the anchor is scaled into robot axes; in WORLD mode it is added to the
        world palm anchor (legacy), in BODY mode it is added to the palm's body-frame
        anchor and then carried by the CURRENT chassis pose, so the target rides with
        the robot body as the base drives/turns."""
        # Shoulder-anchored absolute mode (behind the RETARGET_ANCHOR flag): map the
        # operator's wrist offset from their calibrated shoulder onto the robot's
        # shoulder. Falls back to the clutch mapping below when unavailable.
        if self.anchor == "shoulder":
            p = self._shoulder_anchored(side, w_now)
            if p is not None:
                return p
        delta = self.scale * (self.M @ (w_now - self._avp[f"w{side}"]))
        if not self.body_relative:
            return self._rob[f"{side}_pos"] + delta
        R_base = self.data.body(C.CHASSIS_BODY).xmat.reshape(3, 3)
        p_base = self.data.body(C.CHASSIS_BODY).xpos
        target_body = self._rob[f"{side}_pos_body"] + delta
        return p_base + R_base @ target_body

    @staticmethod
    def _rot_angle(R):
        """Rotation magnitude of a 3x3 in degrees (for the 1:1 tracking diag)."""
        return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))

    def _map_rot(self, Rw_now, Rw0, R0, tag=None):
        R_rel = Rw_now @ Rw0.T                       # operator rotation since calib
        R_rel_robot = self.M_rot @ R_rel @ self.M_rot.T   # in robot axes (+ align)
        if tag is not None:                          # angle is conjugation-invariant
            self.rot_diag[tag] = (self._rot_angle(R_rel), self._rot_angle(R_rel_robot))
        return R_rel_robot @ R0

    def _map_rot_side(self, side, Rw_now, frame):
        """Palm orientation target. In skeleton mode (orient_skeleton + named
        joints + a calibrated K) the palm BODY orientation is F_op·Kᵀ, which makes
        the robot's natural palm frame equal the operator's natural palm frame
        (absolute geometric match). Otherwise it falls back to the legacy
        relative-rotation clutch about the calibration palm pose."""
        tag = side.upper()
        if self.orient_skeleton and f"{side}_K" in self._rob:
            F_op = self._op_frame(self._joints_for(frame, side))
            if F_op is not None:
                return F_op @ self._rob[f"{side}_K"].T
        return self._map_rot(Rw_now, self._avp[f"Rw{side}"], self._rob[f"{side}_mat"], tag=tag)

    def update(self, frame):
        """Return a dict of targets, or None until calibrated / no data.

        keys: l_pos,l_mat,r_pos,r_mat (only for hands present), gaze_dir,
              grasp_l, grasp_r
        """
        if not self.calibrated or frame is None:
            return None
        self._head = frame.head             # latest head pose for the shoulder anchor
        out = {}

        # Late-anchor hands that were absent at calibrate() time (e.g. body
        # tracking toggled on after connecting, or a hand re-acquired).
        if frame.right is not None and "wr" not in self._avp:
            self._anchor_hand("r", frame.right)
        if frame.left is not None and "wl" not in self._avp:
            self._anchor_hand("l", frame.left)

        if frame.right is not None and "wr" in self._avp:
            wr = frame.right[0][:3, 3]
            Rwr = frame.right[0][:3, :3]
            out["r_pos"] = self._map_pos_side("r", wr)
            out["r_mat"] = self._map_rot_side("r", Rwr, frame)
            out["grasp_r"] = self._grasp(frame.right, "r")

        if frame.left is not None and "wl" in self._avp:
            wl = frame.left[0][:3, 3]
            Rwl = frame.left[0][:3, :3]
            out["l_pos"] = self._map_pos_side("l", wl)
            out["l_mat"] = self._map_rot_side("l", Rwl, frame)
            out["grasp_l"] = self._grasp(frame.left, "l")

        # Gaze direction: rotate the calibration gaze by the operator's relative
        # head rotation since calibration, AND by how far the chassis has turned
        # since calibration — so the gaze is local to the body. When the base
        # drives/turns with the phone held still, the neutral gaze rotates with
        # the body (head stays centred on the body); when the phone moves, the
        # operator's rotation is applied on top, in the body's current frame.
        # With dR_base = I (not driving) this is identical to the old mapping.
        if frame.head is not None and "Rh" in self._avp:
            R_head_rel = frame.head[:3, :3] @ self._avp["Rh"].T
            R_rel_robot = self.M @ R_head_rel @ self.M.T
            R_base = self.data.body(C.CHASSIS_BODY).xmat.reshape(3, 3)
            dR_base = R_base @ self._rob["Rbase0"].T     # chassis rotation since calib
            gaze = R_rel_robot @ dR_base @ self._rob["gaze"]
            out["gaze_dir"] = gaze / (np.linalg.norm(gaze) + 1e-9)

        return out
