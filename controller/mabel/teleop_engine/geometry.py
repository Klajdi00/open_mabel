"""Frame / geometry helpers: palm-natural frames, operator frame, neck angles,
wrist flexion, yaw/wrap utilities, and the operator BodyFrame."""
import numpy as np
import mujoco

from .hyperparams import *


def quat_from_mat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).flatten())
    return q


def _natural_frame(wrist, fwd_pt, index_pt, pinky_pt):
    """Right-handed palm frame (columns f=forward, s=index->pinky, n=palm normal)
    in whatever frame the points are given. Derived from anatomy, so it carries
    the correct chirality per hand automatically — no left/right special case."""
    f = np.asarray(fwd_pt, float) - wrist
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


def operator_frame(joints, h_mats):
    """Operator natural palm frame (3x3, ARKit world). Prefers the named skeleton
    (metacarpals), falls back to the 21-array knuckles when joints are absent."""
    if joints:
        def P(n):
            v = joints.get(n)
            return None if v is None else np.asarray(v, float).reshape(4, 4)[:3, 3]
        w, mid, idx, pky = (P("wrist"), P("middle_metacarpal"),
                            P("index_metacarpal"), P("little_metacarpal"))
        if all(v is not None for v in (w, mid, idx, pky)):
            return _natural_frame(w, mid, idx, pky)
    H = [np.asarray(m, float).reshape(4, 4) for m in h_mats]
    if len(H) >= 18:                                      # knuckles: mid=9, idx=5, pinky=17
        return _natural_frame(H[0][:3, 3], H[9][:3, 3], H[5][:3, 3], H[17][:3, 3])
    return None


def palm_axes(R_palm_canon, side):
    """(fwd, up) target axes for the ARM orientation objective, anatomically
    NORMALIZED per hand: the LEFT operator normal is flipped to PALMAR.
    _natural_frame's normal is chirality-mirrored on the human side (palmar on
    the right hand, dorsal on the left), while the robot R_rob normal is
    flexion-anchored (= the physical curl direction = palmar) on BOTH sides --
    so the left flip is REQUIRED for the pairing.

    MEASURED ground truth (server/tools/exp_arm_symmetry.py -- mirrored-hands
    probe + FK palm-mirror check where PALMAR is anchored to the finger-CURL
    direction, i.e. physics, never an index->pinky cross product): with the
    flip the palms mirror physically; WITHOUT it the left palm solves
    PHYSICALLY INVERTED (palm-normal mirror error 178.6 deg; operator: "left
    wrist flipped 180"). Two forensic warnings, both paid for on 2026-07-11:
    (1) joint-space |q| symmetry CANNOT detect this defect (the inversion
    hides in the arm_5/arm_7/ORCA-wrist redundancy split); (2) an FK metric
    whose palm normal is built anatomically from mount POSITIONS inherits the
    left ORCA palm's MIRRORED geometry and reports the physically-flipped palm
    as "symmetric" -- always anchor the metric's normal to flexion/curl. The
    "left arm twisted very weirdly" defect was the wire-derived forearm_arm
    elbows entering the arm IK (now yaw-only, see engine._payload), NOT this
    flip -- reverting the flip traded the twist for a 180-degree palm flip."""
    up = R_palm_canon[:, 2]
    if side == "left":
        up = -up
    return R_palm_canon[:, 0], up


def neck_angles(head_mat, head0, torso_pitch=0.0, chassis_yaw=0.0):
    """Head attitude in the calibrated forward frame -> neck yaw/pitch/roll.

    Express the head in the reference frame BEFORE converting ARKit axes.
    A world-frame rotation delta conjugates a nod by the room heading: at
    90 degrees it becomes roll, and at 180 degrees it reverses pitch.
    """
    dR = AVP_TO_ROBOT @ (head0[:3, :3].T @ head_mat[:3, :3]) @ AVP_TO_ROBOT.T
    # The viewing attitude is relative to the chassis heading at the datum.
    # Remove the chassis's measured turn first, so a base turn transfers yaw
    # from the neck to the base instead of adding another turn to the camera.
    c, s = np.cos(chassis_yaw), np.sin(chassis_yaw)
    dR = np.array([[c, s, 0.], [-s, c, 0.], [0., 0., 1.]]) @ dR
    # Torso precedes neck yaw in the physical chain. Remove its rotation
    # before decomposing the neck; adding pitch afterwards only works when
    # the operator is looking straight ahead.
    c, s = np.cos(torso_pitch), np.sin(torso_pitch)
    dR = np.array([[c, 0., -s], [0., 1., 0.], [s, 0., c]]) @ dR
    yaw = np.arctan2(dR[1, 0], dR[0, 0])
    pitch = -np.arcsin(np.clip(dR[2, 0], -1.0, 1.0))
    roll = np.arctan2(dR[2, 1], dR[2, 2])
    return (NECK_GAIN * NECK_YAW_SIGN * yaw,
            NECK_GAIN * NECK_PITCH_SIGN * pitch,
            NECK_GAIN * NECK_ROLL_SIGN * roll)


def wrist_flexion(R_op, wrist_pos, elbow_pos):
    """Operator 1-DoF wrist flexion [rad]: signed angle between the forearm
    (elbow->wrist) and the hand forward (R_op[:,0]) about the hand's flexion axis
    (R_op[:,1] side). 0 = straight wrist; maps to the single ORCA wrist joint."""
    fa = np.asarray(wrist_pos, float) - np.asarray(elbow_pos, float)
    n = np.linalg.norm(fa)
    if n < 1e-6:
        return 0.0
    fa /= n
    fh, side = R_op[:, 0], R_op[:, 1]
    return float(np.arctan2(np.dot(np.cross(fa, fh), side), np.dot(fa, fh)))


def _yaw(R):
    return float(np.arctan2(R[1, 0], R[0, 0]))


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class BodyFrame:
    """The operator's base_link = the streamer's TORSO odom frame (`odom_mat`),
    which the Vision Pro stream computes directly:

      * origin = the neck pivot dropped to the floor, low-passed -> walking
        translates it, so the wrist expressed in this frame reflects only arm
        motion (no world-frame floating).
      * heading = the FORWARD normal of the two-elbow line, low-passed -> turning
        the TORSO rotates the frame (arm targets turn with you), but turning only
        the HEAD does not (head-yaw-decoupled), so looking around never swings the
        arms. This is why we use the streamer's body odom instead of head yaw.

    The odom frame's own axes are forward / left / up, so expressing a point in it
    (`to_base`) yields robot-base-ordered coordinates (x=fwd, y=left, z=up)
    directly — no separate axis-swap matrix is needed."""

    def __init__(self):
        self.R = None
        self.p = None

    def update(self, odom_mat):
        """Latch the streamer's odom frame (call every frame it is present)."""
        M = np.asarray(odom_mat, float).reshape(4, 4)
        self.R = M[:3, :3]
        self.p = M[:3, 3]

    def to_base(self, p_world, origin):
        """World point, relative to `origin` (e.g. the head), expressed in the body
        heading frame (x=fwd, y=left, z=up)."""
        return self.R.T @ (np.asarray(p_world, float) - np.asarray(origin, float))

    def rot_to_base(self, R_world):
        return self.R.T @ np.asarray(R_world, float)


class ElbowYawEstimator:
    """Operator TORSO heading from the two elbows — the desktop streamer's
    BodyOdom.heading (elbow_normal_yaw + low-pass), ported into the engine so
    HEADSET clients get base yaw-follow too (the app wire derives per-side elbows
    from ARKit's `forearm_arm`; the streamer publishes body_yaw itself, so this
    only runs when the pose has no body_yaw).

    Both elbows are mapped through AVP_TO_ROBOT into the z-up robot frame and the
    heading is the FORWARD normal of the elbow line on the floor plane — the same
    atan2(d_x, -d_y) the streamer uses (d = right - left, pointing to the body's
    right), CCW-positive about up, matching the chassis yaw convention. Turning
    the HEAD does not move the elbows, so looking around never yaws the base.
    Holds the last heading through hand-tracking dropouts."""

    def __init__(self, tau=BODY_YAW_TAU):
        self.tau = float(tau)
        self.yaw = None

    def reset(self):
        self.yaw = None

    def update(self, r_elbow, l_elbow, dt):
        """4x4 elbow transforms (ARKit world) or None -> smoothed heading [rad]
        (or None until both elbows have been seen once)."""
        if r_elbow is None or l_elbow is None:
            return self.yaw                                   # hold through dropout
        pr = AVP_TO_ROBOT @ np.asarray(r_elbow, float).reshape(4, 4)[:3, 3]
        pl = AVP_TO_ROBOT @ np.asarray(l_elbow, float).reshape(4, 4)[:3, 3]
        d = pr[:2] - pl[:2]
        if np.linalg.norm(d) < 1e-3:
            return self.yaw
        yaw = float(np.arctan2(d[0], -d[1]))
        if self.yaw is None:
            self.yaw = yaw
        else:
            a = 1.0 - np.exp(-max(dt, 0.0) / max(self.tau, 1e-3))
            self.yaw = float(self.yaw + a * _wrap(yaw - self.yaw))
        return self.yaw


class ShoulderEstimator:
    """Live operator SHOULDER centres + per-side arm length from the sparse
    headset stream (head + two wrists) — the anchors of the extension-fraction
    arm mapping (hyperparams.ARM_MAP == "extension").

    * Body frame: yaw-only, gravity-aligned. The yaw comes from the engine's
      body-yaw estimate (BodyFrameEstimator / elbow line) when available, so
      LOOKING AROUND does not orbit the estimated shoulders about the neck;
      the head-heading is the fallback before a body yaw exists.
    * Shoulder offsets from the head: measured (OperatorProfile) when set,
      else the Winter ratios of a stature estimated from the HMD height
      (running max, so crouching never shrinks the skeleton).
    * Arm length L: measured (profile) when set, else a live running estimate —
      the running max of the MEDIAN of the last 5 |wrist − shoulder| samples
      (the median rejects single-frame ARKit spikes that would permanently
      inflate a raw max), seeded from the stature ratio and clamped to a
      plausible human band. Reaching fully therefore reads rho = 1 after the
      first full extension even with no calibration at all.
    """

    _L_MIN, _L_MAX = 0.30, 0.90        # m; plausible shoulder→wrist band
    _H_MIN, _H_MAX = 1.40, 2.10        # m; plausible stature band
    _MED_N = 5                          # samples in the spike-rejecting median

    def __init__(self):
        self.reset()

    def reset(self):
        self.stature = None                       # running-max stature estimate [m]
        self._sh = {}                             # side -> latest shoulder world pos
        self._L_est = {"right": None, "left": None}
        self._buf = {"right": [], "left": []}     # recent |wrist−shoulder| samples
        self._prof = None                         # dict: L_lr / half_width / drop
        self._body_axes = None                    # latest (right, up, fwd), ARKit world
        self._head_anchor_f = None                # low-passed head anchor (see update)

    def set_profile(self, arm_length_lr=None, shoulder_width=None, neck_drop=None,
                    shoulder_offset_body=None):
        """Freeze measured anthropometry (facade OperatorProfile). Any argument
        left None keeps the live estimate for that quantity. Shoulder offsets
        are [right, up, forward] in the yaw-only body frame, never head attitude.
        The profile dataclass's all-zero offsets mean unmeasured."""
        def positive(value):
            try:
                value = float(value)
                return value if np.isfinite(value) and value > 0.0 else None
            except (TypeError, ValueError):
                return None
        lr = arm_length_lr if isinstance(arm_length_lr, (list, tuple, np.ndarray)) else ()
        lengths = [positive(lr[i]) if i < len(lr) else None for i in range(2)]
        width, drop = positive(shoulder_width), positive(neck_drop)
        offsets = {}
        if isinstance(shoulder_offset_body, dict):
            for side, key in (("left", "l"), ("right", "r")):
                value = shoulder_offset_body.get(key, shoulder_offset_body.get(side))
                try:
                    off = np.asarray(value, float)
                    if off.shape == (3,) and np.isfinite(off).all() and np.any(off):
                        offsets[side] = off.copy()
                except (TypeError, ValueError):
                    pass
        self._prof = {
            "L": {"left": lengths[0], "right": lengths[1]},
            "half_width": width / 2.0 if width is not None else None,
            "drop": drop,
            "offsets": offsets,
        } if any(lengths) or width or drop or offsets else None

    # ── frame construction ────────────────────────────────────────────────────
    @staticmethod
    def _axes(head_mat, body_yaw):
        """Operator body axes (right, up, fwd) in the ARKit world. body_yaw is
        the engine's absolute robot-frame heading (atan2(r_x, −r_y) of the
        body-right axis, see engine._headset_body_yaw); inverted here so the
        frame turns with the BODY, not with a head glance."""
        up = np.array([0.0, 1.0, 0.0])
        if body_yaw is not None:
            r_rob = np.array([np.sin(body_yaw), -np.cos(body_yaw), 0.0])
            right = AVP_TO_ROBOT.T @ r_rob
        else:
            R = head_mat[:3, :3]
            fwd = -R[:, 2]                       # ARKit camera looks along −Z
            fwd = fwd - up * float(fwd @ up)
            n = np.linalg.norm(fwd)
            if n < 1e-6:                          # looking straight up/down
                fwd = -R[:, 1]
                fwd = fwd - up * float(fwd @ up)
                n = np.linalg.norm(fwd)
            fwd /= (n + 1e-9)
            right = np.cross(fwd, up)
            right /= (np.linalg.norm(right) + 1e-9)
        fwd = np.cross(up, right)
        return right, up, fwd

    # ── per-frame update ──────────────────────────────────────────────────────
    def update(self, head_mat, body_yaw, wrists):
        """head_mat: 4x4 ARKit head pose. body_yaw: engine heading estimate or
        None. wrists: {'right': xyz|None, 'left': xyz|None} (ARKit world)."""
        # LOW-PASS THE HEAD POSITION THAT ANCHORS THE SHOULDERS.
        #
        # The body frame above is yaw-only precisely so that LOOKING AROUND does
        # not orbit the estimated shoulders — but the shoulder POSITION below is
        # a rigid offset from head_pos, so head TRANSLATION was passed through
        # 1:1 and landed on every wrist target. Measured on EgoDex
        # `wipe_kitchen_surfaces/ep3`: the operator's left wrist moves 2 mm/s
        # while its commanded target moves 26.6 mm/s, and freezing the head
        # alone removes 66 % of that (freezing the other hand removes only 14 %,
        # so this is not bimanual cross-talk). A real torso translation is slow
        # and survives the filter; head bob, lean and glance do not.
        raw_head_pos = head_mat[:3, 3]
        if ARM_HEAD_ANCHOR_LP >= 1.0:
            head_pos = raw_head_pos
        else:
            f = getattr(self, "_head_anchor_f", None)
            self._head_anchor_f = (raw_head_pos.copy() if f is None
                                   else f + ARM_HEAD_ANCHOR_LP * (raw_head_pos - f))
            head_pos = self._head_anchor_f
        # Stature from the HMD eye height — running max so a crouch/lean never
        # shrinks the skeleton constants mid-session.
        h = float(np.clip(head_pos[1] / ANTHRO_EYE_HEIGHT_RATIO,
                          self._H_MIN, self._H_MAX))
        self.stature = h if self.stature is None else max(self.stature, h)

        right, up, fwd = self._axes(head_mat, body_yaw)
        self._body_axes = (right.copy(), up.copy(), fwd.copy())
        halfw = (self._prof or {}).get("half_width") or \
            0.5 * ANTHRO_BIACROMIAL * self.stature
        drop = (self._prof or {}).get("drop") or \
            ANTHRO_ACROMION_BELOW_EYE * self.stature
        self._sh["right"] = head_pos + right * halfw - up * drop
        self._sh["left"] = head_pos - right * halfw - up * drop
        for side, offset in (self._prof or {}).get("offsets", {}).items():
            self._sh[side] = head_pos + right * offset[0] + up * offset[1] + fwd * offset[2]

        for side in ("right", "left"):
            w = wrists.get(side)
            if w is None:
                continue
            d = float(np.linalg.norm(np.asarray(w, float) - self._sh[side]))
            if not (self._L_MIN * 0.5 < d < self._L_MAX * 1.5):
                continue                          # wild sample: ignore entirely
            buf = self._buf[side]
            buf.append(d)
            if len(buf) > self._MED_N:
                buf.pop(0)
            if len(buf) == self._MED_N:
                med = float(np.clip(np.median(buf), self._L_MIN, self._L_MAX))
                cur = self._L_est[side]
                self._L_est[side] = med if cur is None else max(cur, med)

    # ── queries ───────────────────────────────────────────────────────────────
    def ready(self, side):
        return side in self._sh

    def body_axes(self):
        """Latest operator body axes ``(right, up, fwd)`` in the ARKit world
        (yaw-only, gravity-aligned — from the body-yaw estimate when present,
        else the head heading), or None before the first update. These are the
        decomposition axes of the body-frame global map (engine
        `_map_wrist_targets`)."""
        return self._body_axes

    def shoulder_world(self, side):
        return self._sh.get(side)

    def root_world(self):
        """The operator ROOT of the GMR scaled-keypoint map: the mid-shoulder
        point (ARKit world), or None before the first update. Head-position-
        derived, so leaning translates it (the robot torso/base follow absorbs
        that), but a head GLANCE never rotates it (yaw-only body frame)."""
        r, l = self._sh.get("right"), self._sh.get("left")
        if r is None or l is None:
            return None
        return 0.5 * (r + l)

    def arm_length(self, side):
        """Best current shoulder→wrist arm length [m] for this side.

        Order: measured profile > max(anthro prior, live running max). The
        anthro prior FLOORS the live estimate — early sub-maximal reaches must
        not shrink L (that would over-read rho and over-extend the robot at
        mid-workspace); a longer-than-prior arm is adopted after its first
        full extension. A shorter-than-prior arm needs the measured profile to
        read rho = 1 (documented limitation)."""
        prof_L = (self._prof or {}).get("L", {}).get(side)
        if prof_L:
            return float(np.clip(prof_L, self._L_MIN, self._L_MAX))
        h = self.stature if self.stature is not None else 1.75
        prior = float(np.clip(ANTHRO_ARM_RATIO * h, self._L_MIN, self._L_MAX))
        if self._L_est[side] is not None:
            return max(prior, self._L_est[side])
        return prior


def estimate_swivel(shoulder, wrist, palm_R, side="right"):
    """Raw elbow-swivel estimate φ [rad] from the palm's roll about the arm
    axis (see synth_elbow step 2), or None when ill-defined. The ENGINE
    low-passes this before feeding it to synth_elbow — the raw per-frame roll
    wobbles with hand geometry during fast wrist motion (measured: unfiltered
    it tripled the circle-tracking direction RMS)."""
    sh = np.asarray(shoulder, float)
    d = np.asarray(wrist, float) - sh
    dist = np.linalg.norm(d)
    if dist < 1e-6 or palm_R is None:
        return None
    dhat = d / dist
    down = np.array([0.0, -1.0, 0.0])
    r0 = down - dhat * float(down @ dhat)
    n = np.linalg.norm(r0)
    if n < 1e-6:
        return None
    r0 = r0 / n
    n_p = np.asarray(palm_R, float)[:, 2]
    if side == "left":
        n_p = -n_p                                   # mirrored normal → palmar
    proj = n_p - dhat * float(n_p @ dhat)
    if np.linalg.norm(proj) < 0.2:                   # palm faces along the arm
        return None
    p_hat = proj / np.linalg.norm(proj)
    theta = float(np.arctan2(np.cross(r0, p_hat) @ dhat, r0 @ p_hat))
    mag = max(0.0, abs(theta) - SWIVEL_DEADBAND)
    return float(np.clip(np.sign(theta) * mag, -SWIVEL_MAX_ABS, SWIVEL_MAX_ABS))


def synth_elbow(shoulder, wrist, arm_len, palm_R=None, side="right",
                body_axes=None, upper_frac=0.56):
    """Anatomically-plausible operator ELBOW position (ARKit world) from the
    shoulder + wrist + (optionally) the HAND ORIENTATION. A two-link arm
    (upper = upper_frac·L) whose flexion follows the law of cosines and whose
    SWIVEL angle φ about the shoulder→wrist axis is estimated in three parts:

      1. BASE — gravity (φ = 0 points the elbow down: the relaxed human
         swivel), previously the whole model.
      2. PALM-ROLL FOLLOW (2026-07-22, "when I push my elbow up the sim does
         not follow") — ARKit tracks no elbow, but raising the elbow rotates
         the humerus, and the whole distal arm rolls with it: the palm's roll
         about the arm axis carries the swivel. Pronation/supination can
         also roll the palm WITHOUT moving the elbow, by roughly
         ±SWIVEL_DEADBAND — so the measured roll θ (palm normal about the arm
         axis, relative to the gravity reference) maps through a deadband,
             φ = clip(θ − sign(θ)·SWIVEL_DEADBAND, SWIVEL_MIN, SWIVEL_MAX),
         the VR-arm-IK trick: wrist-only data leaves the swivel formally
         under-determined, and the deadband absorbs the pronation slack.
      3. TORSO CLEARANCE — the swivel is clamped so the elbow's lateral
         (body-frame) offset never crosses INWARD past the shoulder by more
         than SWIVEL_LAT_MIN·upper: an inward-swung elbow on a cross-body
         reach pressed the (previously unguarded) upper arm into the chest.

    ``palm_R``: the operator natural palm frame (columns f/s/n, ARKit world;
    the raw chirality-mirrored normal — this function flips the LEFT side to
    palmar itself). ``body_axes``: (right, up, fwd) operator body axes for
    the clearance clamp. Both optional — omitted, the model degrades to the
    original gravity swivel. Degenerates smoothly: at full extension the
    elbow lands on the shoulder→wrist line regardless of φ."""
    sh = np.asarray(shoulder, float)
    d = np.asarray(wrist, float) - sh
    dist = np.linalg.norm(d)
    if dist < 1e-6:
        return None
    upper = upper_frac * float(arm_len)
    fore = (1.0 - upper_frac) * float(arm_len)
    dist = min(dist, upper + fore - 1e-9)
    dhat = d / (np.linalg.norm(d) + 1e-12)
    cos_a = np.clip((upper**2 + dist**2 - fore**2) / (2.0 * upper * dist),
                    -1.0, 1.0)
    sin_a = np.sqrt(max(0.0, 1.0 - cos_a**2))
    down = np.array([0.0, -1.0, 0.0])
    r0 = down - dhat * float(down @ dhat)           # gravity swivel reference
    n = np.linalg.norm(r0)
    if n < 1e-6:                                     # arm straight down: swivel
        r0 = np.array([0.0, 0.0, 1.0]) - dhat * dhat[2]    # falls back to +Z
        n = np.linalg.norm(r0)
    r0 = r0 / (n + 1e-12)
    t0 = np.cross(dhat, r0)                          # completes the swivel basis

    phi = 0.0
    if palm_R is not None:
        if np.isscalar(palm_R) or np.asarray(palm_R).ndim == 0:
            phi = float(palm_R)                      # pre-estimated (smoothed) φ
        else:
            n_p = np.asarray(palm_R, float)[:, 2]
            if side == "left":
                n_p = -n_p                           # mirrored normal → palmar
            proj = n_p - dhat * float(n_p @ dhat)
            if np.linalg.norm(proj) > 0.2:           # roll ill-defined when the
                p_hat = proj / np.linalg.norm(proj)  # palm faces along the arm
                theta = float(np.arctan2(np.cross(r0, p_hat) @ dhat, r0 @ p_hat))
                mag = max(0.0, abs(theta) - SWIVEL_DEADBAND)
                phi = float(np.clip(np.sign(theta) * mag,
                                    -SWIVEL_MAX_ABS, SWIVEL_MAX_ABS))

    def elbow_at(p):
        w = np.cos(p) * r0 + np.sin(p) * t0
        return sh + dhat * (upper * cos_a) + w * (upper * sin_a)

    e = elbow_at(phi)
    if body_axes is not None and sin_a > 1e-6:
        r_b = np.asarray(body_axes[0], float)        # operator body RIGHT
        s_lat = 1.0 if side == "right" else -1.0     # outward = +right / −right
        lat_min = -SWIVEL_LAT_MIN * upper            # inward-crossing bound
        if s_lat * float((e - sh) @ r_b) < lat_min:
            # EXACT boundary solve (continuous — a coarse stepped walk made the
            # hint JUMP as a sweeping target crossed the bound, tripling the
            # circle-tracking direction RMS): the lateral coordinate is
            #   λ(φ) = P cosφ + Q sinφ + R,
            # solve λ = λ_min and take the root nearest the desired φ.
            P = s_lat * upper * sin_a * float(r0 @ r_b)
            Q = s_lat * upper * sin_a * float(t0 @ r_b)
            R = s_lat * upper * cos_a * float(dhat @ r_b)
            amp = float(np.hypot(P, Q))
            if amp < 1e-9 or lat_min - R > amp:
                e = elbow_at(phi + s_lat * np.pi)    # nothing satisfies: outward
            else:
                base = float(np.arctan2(Q, P))
                dphi = float(np.arccos(np.clip((lat_min - R) / amp, -1.0, 1.0)))
                cands = [base + dphi, base - dphi]
                best = min(cands, key=lambda c: abs(
                    (c - phi + np.pi) % (2 * np.pi) - np.pi))
                e = elbow_at(best)
    return e


def clamp_torso_standoff(p_world, axis_xy, r_min, side_dir):
    """Push a wrist target out of the robot's TORSO, horizontally.

    THE CASE THE SHOULDER-RADIAL CLAMP GETS WRONG (measured 2026-08-25 on
    EgoDex `screw_unscrew_allen_fixture/ep003`). When the operator holds a hand
    against their own chest, the mapped target lands on MABEL's body CENTRELINE
    at shoulder height -- 0.023-0.063 m from the torso axis, against a shoulder
    half-width of 0.074 m, i.e. between the two shoulders and inside the chest.

    A shoulder-radial push (`clamp_shoulder_standoff`) is the wrong correction
    there: from a point that close to the shoulder the radial direction is
    almost arbitrary, so preserving it and extending the distance drives the
    target further along a bearing the arm cannot follow. Measured, the angle
    between the target direction and the achievable direction is 67 deg on that
    clip against 14 deg on one the radial clamp helps -- and the radial clamp
    made it 44 mm WORSE while helping the other by 52 mm.

    Pushing away from the torso AXIS instead moves an on-centreline target
    sideways, to the side of the body where that arm can actually reach it.
    Only the horizontal component is changed; height is the operator's and is
    left alone.

    ``side_dir`` is that arm's outward lateral unit vector (world, horizontal),
    used when the target sits so close to the axis that its own radial
    direction carries no information.
    """
    p = np.asarray(p_world, float).copy()
    v = p[:2] - np.asarray(axis_xy, float)
    d = float(np.linalg.norm(v))
    if d >= r_min:
        return p
    if d < 0.02:                       # on the axis: no usable direction, use the arm's own side
        u = np.asarray(side_dir, float)[:2]
        n = np.linalg.norm(u)
        u = u / n if n > 1e-9 else np.array([0.0, 1.0])
    else:
        u = v / d
    p[:2] = np.asarray(axis_xy, float) + u * r_min
    return p


def clamp_shoulder_standoff(p_world, shoulder, r_min):
    """Push a wrist target that has been mapped INSIDE the robot's own body out
    to a minimum radius about that arm's shoulder.

    THE DEFECT THIS FIXES (measured 2026-08-25 on EgoDex `use_chopsticks/ep3`).
    A human working close to their own chest -- chopsticks over a bowl, a watch
    strap, a shirt button -- puts their wrists 0.21-0.24 m from their head. That
    is ordinary, and the operator data is clean. But the GMR map is
    root-anchored and scale-normalised, and MABEL's shoulders are roughly three
    times narrower than a human's, so the same body-relative pose lands
    **inside the robot's torso**: 82 % of that clip's left-wrist targets fell
    within 0.16 m of the torso axis, at 0.055-0.117 m from the shoulder. The arm
    cannot go there -- it reached 0.226 m at best -- so it folded against the
    chest and sat 173 mm short, which is what an operator sees as "the hand is
    in the wrong place".

    Sampling 20 000 random in-limit arm poses puts only ~1 % of them inside
    0.15 m of the shoulder: the deep-fold region is a thin sliver of
    configuration space that a warm-started local IK will not find, and the
    front-workspace posture table does not cover it either (its box starts at
    0.25 m). Rather than ask the solver to chase a pose that is unreachable in
    practice, the TARGET is corrected -- radially, so the direction the operator
    indicated is preserved exactly and only the distance changes.

    Returns the point unchanged when it is already outside ``r_min``.
    """
    p = np.asarray(p_world, float)
    v = p - np.asarray(shoulder, float)
    d = float(np.linalg.norm(v))
    if d >= r_min:
        return p
    if d < 1e-6:                       # degenerate: no direction to preserve
        return p
    return np.asarray(shoulder, float) + v * (r_min / d)


def clamp_in_front(p_world, data, base_bid):
    """Clamp a world-frame target so it stays IN FRONT of the body's front vertical
    line (the chassis frame, robot front = -x): the wrist/contact target can never be
    driven behind the robot and twist the arm to the back. The line is at chassis-x =
    ARM_WRIST_FRONT; a target behind it (x > ARM_WRIST_FRONT) is pulled onto the line,
    so the base must back up to bring the operator's intent forward of it."""
    o = data.xpos[base_bid]
    R = data.xmat[base_bid].reshape(3, 3)
    loc = R.T @ (np.asarray(p_world, float) - o)        # target in chassis frame
    if loc[0] > ARM_WRIST_FRONT:                         # behind the front line
        loc[0] = ARM_WRIST_FRONT
        return o + R @ loc
    return np.asarray(p_world, float)
