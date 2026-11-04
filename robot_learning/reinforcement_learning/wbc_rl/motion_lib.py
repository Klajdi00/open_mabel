"""motion_lib.py — human motion -> the three Vision Pro frames NOVA tracks, plus privileged body labels.

Sources (both converted to one Z-up world, x forward of the clip's first head heading, 50 Hz):
  EgoDex (Apple Vision Pro, 30 Hz): camera = head, palm frame from wrist + knuckles, ARKit upper body.
  LAFAN1 (Ubisoft mocap, 30 Hz):    walk / obstacles / aiming / multipleActions / dance — walking + turning.
  NOVA takes (an operator's Vision Pro, Retarget Lab episodes/<id>/frames.jsonl.gz, the raw teleop_frame wire):
                                    head + both palms exactly as the robot receives them; no body (shoulders from the
                                    head as at deploy, no elbows / lean): --avp, -> data/avp_takes.npz

Per frame we store
  OBSERVED (what the Vision Pro gives at teleop time, the policy's only inputs):
    head  eye position + frame (x forward, y left, z up)
    lw/rw ORCA-wrist-joint position + anatomical palm frame (x wrist->middle knuckle, y pinky->index, z = x × y)
  PRIVILEGED (critic + human-likeness rewards only; never needed on the robot):
    torso_yaw   heading of the shoulder line (body twist)           -> base yaw
    torso_xy    shoulder midpoint on the floor                       -> base position
    lean        forward lean of hip->shoulders from vertical (rad)  -> torso pitch
    neck        head relative to the torso frame, (yaw, pitch, roll) -> neck joints
    le/re       elbows                                               -> arm posture (no odd twisting)

The retarget map is RETARGET below and is the same at deployment: head 1:1, wrists 1:1 relative to the
head plus WRIST_DZ (robot eye->shoulder drop minus the human's). Nothing here is robot-specific beyond that.

    python motion_lib.py                  # build data/motions.npz from what is downloaded
    python motion_lib.py --stats          # print corpus statistics
    python motion_lib.py --avp <episodes_dir>   # your own Retarget Lab takes -> data/avp_takes.npz
"""
import argparse, glob, os, re, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data", "raw")
OUT = os.path.join(HERE, "data", "motions.npz")
FPS = 50.0
YUP_TO_ZUP = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

# Retarget constants (shared with deploy). Robot: eye 0.245 m above the shoulder line (FK, rest pose, head
# mesh front face); human: measured over EgoDex by `--stats` (eye->shoulder 0.24-0.26 m). Same within 1 cm.
WRIST_DZ = 0.0
# The live Vision Pro stream has no shoulders: training and deploy both place them from the head (eye) pose,
# in the head's yaw frame (x forward, y left, z up), the EgoDex median (1,500 clips; left/right averaged so NOVA
# stays mirror-symmetric). vs ARKit's inferred shoulders: 4 cm p50, 10 cm p90 (~1 / 2 cm on the scaled wrist).
SHOULDER_IN_HEAD = (-0.215, 0.178, -0.178)                    # left; right mirrors y


# NOVA's shoulder centre (arm_1 / arm_2 axes) from its eye point, same frame, arms / torso / neck at zero (robot.FK).
ROBOT_SHOULDER_IN_EYE = (-0.08, 0.165, -0.24)
ROBOT_REACH = 0.63               # m, shoulder -> wrist the operator's full reach maps to: NOVA's straight arm is 0.657,
                                 # 0.649 inside the elbow's end-stop margin; 0.63 keeps it reachable, the elbow just off straight
NEAR_FACE = (0.20, 0.45)        # m from the eye: hands closer than 0.20 m map eye-relative, beyond 0.45 m shoulder-relative


def retarget_hand(head_p, head_yaw, w, arm_k, side, xp=np):
    """Operator wrist (…,3) -> where NOVA's wrist should be, in the operator's world (the anchor maps it on).
    Landmark-preserving: about the shoulders, the operator's reach maps onto NOVA's (arm_k = ROBOT_REACH / operator
    reach), from NOVA's OWN shoulder (fully extended -> fully extended, straight up -> straight up); near the face,
    eye-relative and unscaled (hands in front of the eyes -> in front of NOVA's eyes), smoothly blended between."""
    c, s = xp.cos(head_yaw), xp.sin(head_yaw)
    d = w - head_p
    x = xp.stack([c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1], d[..., 2]], -1)   # head-yaw frame
    sg = 1.0 if side == "l" else -1.0
    sh = xp.stack([0 * c + SHOULDER_IN_HEAD[0], 0 * c + sg * SHOULDER_IN_HEAD[1], 0 * c + SHOULDER_IN_HEAD[2]], -1)
    sr = xp.stack([0 * c + ROBOT_SHOULDER_IN_EYE[0], 0 * c + sg * ROBOT_SHOULDER_IN_EYE[1], 0 * c + ROBOT_SHOULDER_IN_EYE[2]], -1)
    k = arm_k if not hasattr(arm_k, "shape") or arm_k.ndim == 0 else arm_k[..., None] if arm_k.shape[-1:] != (1,) else arm_k
    y_s = sr + k * (x - sh)
    r = (x * x).sum(-1) ** 0.5
    b = ((NEAR_FACE[1] - r) / (NEAR_FACE[1] - NEAR_FACE[0])).clip(0, 1) if xp is np else ((NEAR_FACE[1] - r) / (NEAR_FACE[1] - NEAR_FACE[0])).clamp(0, 1)
    b = (b * b * (3 - 2 * b))[..., None]
    y = b * x + (1 - b) * y_s
    return head_p + xp.stack([c * y[..., 0] - s * y[..., 1], s * y[..., 0] + c * y[..., 1], y[..., 2]], -1)


def shoulders_from_head(head_p, head_yaw, xp=np):
    """(…,3) eye position, (…) yaw -> left, right shoulder estimates. xp = numpy or torch."""
    c, s = xp.cos(head_yaw), xp.sin(head_yaw)
    x, y, z = SHOULDER_IN_HEAD
    out = []
    for yy in (y, -y):
        out.append(head_p + xp.stack([c * x - s * yy, s * x + c * yy, 0 * c + z], -1))
    return out[0], out[1]
LAFAN_KEEP = ("walk", "obstacles", "aiming", "multipleActions", "dance")
MAX_HEAD_SPEED = 0.6      # m/s horizontal walking speed (0.5 s average) a stretch may reach after slowing
MIN_RATE = 0.6            # a mocap take is slowed at most to 0.6x; faster stretches (jogs) are cut out
MIN_HEAD_Z_DROP = -0.45   # m below the clip's start height: deeper crouches are cut (lift + torso can't follow)
MIN_SEG_S = 2.0         # EgoDex episodes are short; the env chains clips with a blend


# ---------------------------------------------------------------- rotations (numpy, batch)
def _norm(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def frame_xy(x, y):
    """Rotation matrices with columns x, y' (y orthogonalized), z = x × y'."""
    x = _norm(x); y = _norm(y - (y * x).sum(-1, keepdims=True) * x); z = np.cross(x, y)
    return np.stack([x, y, z], axis=-1)


def mat_to_quat(R):
    """(...,3,3) -> (...,4) wxyz, w >= 0."""
    m = R.reshape(-1, 3, 3); q = np.zeros((m.shape[0], 4))
    t = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    for i in range(m.shape[0]):
        a = m[i]
        if t[i] > 0:
            s = np.sqrt(t[i] + 1.0) * 2; q[i] = [0.25 * s, (a[2, 1] - a[1, 2]) / s, (a[0, 2] - a[2, 0]) / s, (a[1, 0] - a[0, 1]) / s]
        elif a[0, 0] > a[1, 1] and a[0, 0] > a[2, 2]:
            s = np.sqrt(1.0 + a[0, 0] - a[1, 1] - a[2, 2]) * 2; q[i] = [(a[2, 1] - a[1, 2]) / s, 0.25 * s, (a[0, 1] + a[1, 0]) / s, (a[0, 2] + a[2, 0]) / s]
        elif a[1, 1] > a[2, 2]:
            s = np.sqrt(1.0 + a[1, 1] - a[0, 0] - a[2, 2]) * 2; q[i] = [(a[0, 2] - a[2, 0]) / s, (a[0, 1] + a[1, 0]) / s, 0.25 * s, (a[1, 2] + a[2, 1]) / s]
        else:
            s = np.sqrt(1.0 + a[2, 2] - a[0, 0] - a[1, 1]) * 2; q[i] = [(a[1, 0] - a[0, 1]) / s, (a[0, 2] + a[2, 0]) / s, (a[1, 2] + a[2, 1]) / s, 0.25 * s]
    q *= np.where(q[:, :1] < 0, -1.0, 1.0)
    return q.reshape(R.shape[:-2] + (4,))


def quat_to_mat(q):
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                     2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                     2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1).reshape(q.shape[:-1] + (3, 3))


def slerp_resample(q, t_src, t_dst):
    idx = np.clip(np.searchsorted(t_src, t_dst) - 1, 0, len(t_src) - 2)
    a = (t_dst - t_src[idx]) / (t_src[idx + 1] - t_src[idx])
    q0, q1 = q[idx], q[idx + 1].copy()
    d = (q0 * q1).sum(-1); q1[d < 0] *= -1; d = np.abs(d)
    th = np.arccos(np.clip(d, -1, 1)); s = np.sin(th)
    w0 = np.where(s > 1e-6, np.sin((1 - a) * th) / np.maximum(s, 1e-9), 1 - a)
    w1 = np.where(s > 1e-6, np.sin(a * th) / np.maximum(s, 1e-9), a)
    return _norm(w0[:, None] * q0 + w1[:, None] * q1)


def lerp_resample(x, t_src, t_dst):
    return np.stack([np.interp(t_dst, t_src, x[:, k]) for k in range(x.shape[1])], -1)


def yaw_of(R):
    return np.arctan2(R[..., 1, 0], R[..., 0, 0])


def euler_zyx(R):
    yaw = np.arctan2(R[..., 1, 0], R[..., 0, 0])
    pitch = -np.arcsin(np.clip(R[..., 2, 0], -1, 1))
    roll = np.arctan2(R[..., 2, 1], R[..., 2, 2])
    return np.stack([yaw, pitch, roll], -1)


# ---------------------------------------------------------------- canonical human frame set
def canonical(head_p, head_R, lw_p, lw_R, rw_p, rw_R, le, re, ls, rs, hip):
    """Everything (T,...) in Z-up world -> the stored per-frame dict (still in the source world)."""
    sm = 0.5 * (ls + rs)
    up = _norm(sm - hip)
    left = rs * 0 + (ls - rs)
    torso_R = frame_xy(np.cross(_norm(left), up), left)          # x forward, y left, z ~ along the spine
    # yaw of the body: heading of the shoulder line in the floor plane
    fwd = np.cross(_norm(left * np.array([1, 1, 0])), np.array([0, 0, 1.0]))   # left × up = forward
    torso_yaw = np.arctan2(fwd[:, 1], fwd[:, 0])
    upright_R = np.stack([np.stack([np.cos(torso_yaw), np.sin(torso_yaw), 0 * torso_yaw], -1),
                          np.stack([-np.sin(torso_yaw), np.cos(torso_yaw), 0 * torso_yaw], -1),
                          np.stack([0 * torso_yaw, 0 * torso_yaw, 1 + 0 * torso_yaw], -1)], -1)
    v = sm - hip
    lean = np.arctan2((v * fwd).sum(-1), v[:, 2])                # + = forward
    neck = euler_zyx(np.einsum("tji,tjk->tik", torso_R, head_R))  # head in torso frame
    return dict(head_p=head_p, head_q=mat_to_quat(head_R), lw_p=lw_p, lw_q=mat_to_quat(lw_R),
                rw_p=rw_p, rw_q=mat_to_quat(rw_R), le=le, re=re, ls=ls, rs=rs, sm=sm,
                torso_yaw=np.unwrap(torso_yaw), lean=lean, neck=neck, eye_to_shoulder=head_p[:, 2] - sm[:, 2])


# ---------------------------------------------------------------- EgoDex
def load_egodex(path):
    import h5py
    with h5py.File(path, "r") as f:
        T = lambda k: f["transforms/" + k][:].astype(np.float64)
        C = lambda k: f["confidences/" + k][:] if "confidences/" + k in f else np.ones(f["transforms/camera"].shape[0])
        cam = T("camera")
        tr = lambda M: np.einsum("ij,tjk->tik", YUP_TO_ZUP, M)
        P = lambda k: tr(T(k)[:, :3, 3:])[..., 0]
        conf = np.minimum.reduce([C(k) for k in ("leftHand", "rightHand", "leftIndexFingerKnuckle", "rightIndexFingerKnuckle",
                                                  "leftLittleFingerKnuckle", "rightLittleFingerKnuckle")])
        camR = tr(cam[:, :3, :3])
        head_R = np.stack([camR[:, :, 2], -camR[:, :, 0], -camR[:, :, 1]], -1)     # EgoDex camera is OpenCV-style: x right, y down, z forward
        out = {}
        for s, side in (("l", "left"), ("r", "right")):
            w, mid, idx, lit = P(side + "Hand"), P(side + "MiddleFingerKnuckle"), P(side + "IndexFingerKnuckle"), P(side + "LittleFingerKnuckle")
            out[s] = (w, frame_xy(mid - w, idx - lit), P(side + "Forearm"), P(side + "Arm"))
        attrs = dict(f.attrs)
        d = canonical(P("camera"), head_R, out["l"][0], out["l"][1], out["r"][0], out["r"][1],
                      out["l"][2], out["r"][2], out["l"][3], out["r"][3], P("hip"))
    d["conf"] = conf
    d["meta"] = f"egodex:{attrs.get('task', '')}:{attrs.get('environment', '')}"
    return d, 30.0


# ---------------------------------------------------------------- LAFAN1 (BVH)
def _euler_bvh(order, ang):
    R = np.broadcast_to(np.eye(3), ang.shape[:-1] + (3, 3)).copy()
    for i, ax in enumerate(order):
        a = np.deg2rad(ang[..., i]); c, s = np.cos(a), np.sin(a); M = np.zeros(ang.shape[:-1] + (3, 3))
        if ax == "X": M[..., 0, 0] = 1; M[..., 1, 1] = c; M[..., 1, 2] = -s; M[..., 2, 1] = s; M[..., 2, 2] = c
        if ax == "Y": M[..., 1, 1] = 1; M[..., 0, 0] = c; M[..., 0, 2] = s; M[..., 2, 0] = -s; M[..., 2, 2] = c
        if ax == "Z": M[..., 2, 2] = 1; M[..., 0, 0] = c; M[..., 0, 1] = -s; M[..., 1, 0] = s; M[..., 1, 1] = c
        R = R @ M
    return R


def load_bvh(path):
    lines = open(path).read().split("\n")
    names, parents, offsets, chans, stack = [], [], [], [], []
    i = 0
    while not lines[i].startswith("MOTION"):
        t = lines[i].split()
        if t and t[0] in ("ROOT", "JOINT"):
            names.append(t[1]); parents.append(stack[-1] if stack else -1)
        elif t and t[0] == "End":
            names.append(None); parents.append(stack[-1])
        elif t and t[0] == "{":
            stack.append(len(names) - 1)
        elif t and t[0] == "}":
            stack.pop()
        elif t and t[0] == "OFFSET":
            offsets.append([float(x) for x in t[1:4]])
        elif t and t[0] == "CHANNELS":
            chans.append((len(names) - 1, t[2:]))
        i += 1
    dt = float(lines[i + 2].split()[-1])
    data = np.array([[float(x) for x in l.split()] for l in lines[i + 3:] if l.strip()])
    J = len(names); Tn = data.shape[0]
    Rl = np.broadcast_to(np.eye(3), (Tn, J, 3, 3)).copy(); pos = np.zeros((Tn, 3)); c = 0
    for j, ch in chans:
        rot = [x[0] for x in ch if x.endswith("rotation")]
        vals = data[:, c:c + len(ch)]
        if any(x.endswith("position") for x in ch):
            pos = vals[:, :3]; vals = vals[:, 3:]
        Rl[:, j] = _euler_bvh(rot, vals); c += len(ch)
    Rg = np.zeros_like(Rl); Pg = np.zeros((Tn, J, 3)); off = np.array(offsets)
    for j in range(J):
        p = parents[j]
        if p < 0:
            Rg[:, j] = Rl[:, j]; Pg[:, j] = pos
        else:
            Rg[:, j] = Rg[:, p] @ Rl[:, j]; Pg[:, j] = Pg[:, p] + Rg[:, p] @ off[j]
    idx = {n: k for k, n in enumerate(names) if n}
    return idx, Rg, Pg / 100.0, 1.0 / dt


def _best_axis(Rg, target):
    """The signed local axis of Rg (T,3,3) best aligned with target (T,3) on average -> (T,3) world axis."""
    best, score = None, -2
    for k in range(3):
        for sgn in (1, -1):
            a = sgn * Rg[:, :, k]; sc = (a * target).sum(-1).mean()
            if sc > score: best, score = a, sc
    return best


def load_lafan(path):
    idx, Rg, Pg, fps = load_bvh(path)
    Pg = np.einsum("ij,tkj->tki", YUP_TO_ZUP, Pg); Rg = np.einsum("ij,tkjl->tkil", YUP_TO_ZUP, Rg)
    G = lambda n: Pg[:, idx[n]]
    hip, ls, rs = G("Hips"), G("LeftArm"), G("RightArm")
    left = _norm((ls - rs) * np.array([1, 1, 0])); fwd = np.cross(left, np.array([0, 0, 1.0]))
    up = np.broadcast_to(np.array([0, 0, 1.0]), fwd.shape)
    Rh = Rg[:, idx["Head"]]
    hx = _best_axis(Rh, fwd); hz = _best_axis(Rh, up)
    head_R = frame_xy(hx, np.cross(hz, hx))
    # eye point: 10 cm forward / 8 cm up of the head joint, in the head frame (the head joint is near the skull base)
    head_p = G("Head") + head_R @ np.array([0.10, 0.0, 0.08])
    hands = {}
    for s, side in (("l", "Left"), ("r", "Right")):
        Rw = Rg[:, idx[side + "Hand"]]
        fa = _norm(G(side + "Hand") - G(side + "ForeArm"))
        x = _best_axis(Rw, fa)                                  # along the hand ~ along the forearm
        # thumb side: for a hanging arm the thumb points forward; pick the remaining local axis closest to fwd
        y = _best_axis(Rw, fwd - (fwd * x).sum(-1, keepdims=True) * x)
        hands[s] = (G(side + "Hand"), frame_xy(x, y), G(side + "ForeArm"), G(side + "Arm"))
    d = canonical(head_p, head_R, hands["l"][0], hands["l"][1], hands["r"][0], hands["r"][1],
                  hands["l"][2], hands["r"][2], hands["l"][3], hands["r"][3], hip)
    d["conf"] = np.ones(len(hip))
    d["meta"] = "lafan:" + os.path.basename(path)[:-4]
    return d, fps


# ---------------------------------------------------------------- resample, clean, segment
KEYS_P = ("head_p", "lw_p", "rw_p", "le", "re", "ls", "rs", "sm")
KEYS_Q = ("head_q", "lw_q", "rw_q")
KEYS_S = ("torso_yaw", "lean", "eye_to_shoulder")


# ---------------------------------------------------------------- operator Vision Pro takes (Retarget Lab episodes)
AVP_OUT = os.path.join(HERE, "data", "avp_takes.npz")


def wire_positions(hand: dict) -> dict:
    """A teleop_frame HandPose dict -> {ARKit joint name: world xyz} (anchor @ local), EVERY joint whatever its
    isTracked says: ARKit gives an occluded joint an estimated pose and the flag is confidence, not validity."""
    A = np.asarray(hand.get("anchorTransform", {}).get("matrix") or np.eye(4).ravel(), dtype=float).reshape(4, 4)
    out = {}
    for j in hand.get("joints") or []:
        m = (j.get("localTransform") or {}).get("matrix")
        if m and len(m) == 16:
            out[j["joint"]] = (A @ np.asarray(m, dtype=float).reshape(4, 4))[:3, 3]
    return out


def _mat(m):
    M = np.asarray(m, dtype=float).reshape(4, 4)
    return M if np.allclose(M[3], [0, 0, 0, 1]) else M.T          # row-major with the translation in [:3, 3]


def load_avp_take(path):
    """episodes/<id>/frames.jsonl.gz ({"rt", "p": teleop_frame}) -> canonical dict on the payload clock (irregular t).
    Frames with a hand untracked get conf 0 (segments() cuts there). Body labels are what deploy knows: shoulders
    from the head (SHOULDER_IN_HEAD), elbows unknown (set to the shoulder-wrist midpoint, never scored: meta avp:)."""
    import gzip, json
    rows = [json.loads(l) for l in gzip.open(os.path.join(path, "frames.jsonl.gz"), "rt") if l.strip()]
    T, H, W, C = [], [], {"l": [], "r": []}, []
    for r in rows:
        p = r.get("p", r)
        if "head" not in p: continue
        M = _mat(p["head"]["transform"]["matrix"]); Rd = YUP_TO_ZUP @ M[:3, :3]
        hp, hR = YUP_TO_ZUP @ M[:3, 3], np.stack([-Rd[:, 2], -Rd[:, 0], Rd[:, 1]], -1)     # = deploy.head_from_avp
        ok = True; pw = {}
        for s, k in (("l", "leftHand"), ("r", "rightHand")):
            h = p.get(k) or {}
            P = {n: YUP_TO_ZUP @ v for n, v in wire_positions(h).items()} if h.get("isTracked") else {}
            if not all(n in P for n in ("wrist", "middle_knuckle", "index_knuckle", "little_knuckle")):
                ok = False; pw[s] = (np.full(3, np.nan), np.eye(3)); continue
            w = P["wrist"]
            pw[s] = (w, frame_xy((P["middle_knuckle"] - w)[None], (P["index_knuckle"] - P["little_knuckle"])[None])[0])   # = deploy.palm_from_avp
        T.append(float(p.get("timestamp", r.get("rt", 0.0)))); H.append((hp, hR)); C.append(1.0 if ok else 0.0)
        for s in "lr": W[s].append(pw[s])
    t = np.array(T); t -= t[0]
    keep = np.concatenate([[True], np.diff(t) > 1e-4])                  # duplicate / out-of-order stamps
    st = lambda L, i: np.stack([x[i] for x in L])[keep]
    head_p, head_R = st(H, 0), st(H, 1)
    lw_p, lw_R, rw_p, rw_R = st(W["l"], 0), st(W["l"], 1), st(W["r"], 0), st(W["r"], 1)
    for a in (lw_p, rw_p):                                             # untracked frames: hold (conf 0 cuts them anyway)
        bad = ~np.isfinite(a).all(1)
        if bad.all(): a[:] = head_p
        else: a[bad] = np.array([np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), a[~bad, k]) for k in range(3)]).T
    ls, rs = shoulders_from_head(head_p, yaw_of(head_R))
    sm = 0.5 * (ls + rs); hip = sm - np.array([0.0, 0.0, 0.5])
    d = canonical(head_p, head_R, lw_p, lw_R, rw_p, rw_R, 0.5 * (ls + lw_p), 0.5 * (rs + rw_p), ls, rs, hip)
    d["conf"] = np.array(C)[keep]
    d["meta"] = "avp:" + os.path.basename(os.path.normpath(path))
    return d, t[keep]


def build_avp(src, out=AVP_OUT):
    """Every Retarget Lab episode under src -> data/avp_takes.npz (motions.npz layout). Every 5th take (sorted by id)
    is held out: meta avp_eval:, never sampled in training (sonic.py eval --clips avp)."""
    clips, metas = [], []
    eps = sorted(d for d in glob.glob(os.path.join(src, "*")) if os.path.exists(os.path.join(d, "frames.jsonl.gz")))
    for i, e in enumerate(eps):
        try:
            d, t = load_avp_take(e)
        except Exception as ex:
            print("skip", e, ex); continue
        if len(t) < 10: continue
        tt = np.arange(0, t[-1], 1.0 / FPS)
        o = {k: lerp_resample(d[k], t, tt) for k in KEYS_P}
        o.update({k: slerp_resample(d[k], t, tt) for k in KEYS_Q})
        o.update({k: np.interp(tt, t, d[k]) for k in KEYS_S})
        o["neck"] = lerp_resample(np.unwrap(d["neck"], axis=0), t, tt)
        o["conf"] = np.interp(tt, t, d["conf"])
        held = i % 5 == 4
        n0 = len(clips)
        for a, b in segments(o, int(MIN_SEG_S * FPS)):
            clips.append({k: v[a:b].copy() for k, v in o.items()}); metas.append(d["meta"].replace("avp:", "avp_eval:" if held else "avp:"))
        print(f"{os.path.basename(e)}: {t[-1]:.0f} s, {len(clips) - n0} segments{' (eval)' if held else ''}", flush=True)
    assert clips, f"no usable takes under {src}"
    lens = np.array([len(c["head_p"]) for c in clips])
    arr = {k: np.concatenate([c[k] for c in clips]).astype(np.float32) for k in clips[0]}
    arr["wrist_dz"] = np.float32(WRIST_DZ)
    np.savez_compressed(out, starts=np.concatenate([[0], np.cumsum(lens)[:-1]]), lens=lens, meta=np.array(metas), **arr)
    print(f"wrote {out}: {len(clips)} segments, {lens.sum() / FPS / 60:.1f} min ({sum(m.startswith('avp_eval') for m in metas)} eval)")


def to_50hz(d, fps, rate=1.0):
    """Resample to FPS; rate < 1 slows the motion down (time stretch)."""
    n = len(d["head_p"]); t_src = np.arange(n) / fps / rate
    t = np.arange(0, t_src[-1], 1.0 / FPS)
    o = {k: lerp_resample(d[k], t_src, t) for k in KEYS_P}
    o.update({k: slerp_resample(d[k], t_src, t) for k in KEYS_Q})
    o.update({k: np.interp(t, t_src, d[k]) for k in KEYS_S})
    o["neck"] = lerp_resample(np.unwrap(d["neck"], axis=0), t_src, t)
    o["conf"] = np.interp(t, t_src, d["conf"])
    return o


def segments(o, min_len, vmax=None):
    """Valid contiguous stretches: tracked, finite, no deep crouch, no teleporting wrists, walking <= vmax."""
    ok = np.isfinite(np.concatenate([o[k].reshape(len(o["conf"]), -1) for k in KEYS_P + KEYS_Q], 1)).all(1)
    ok &= o["conf"] > 0.3
    z0 = np.median(o["head_p"][: int(FPS), 2])
    ok &= o["head_p"][:, 2] - z0 > MIN_HEAD_Z_DROP
    for k in ("lw_p", "rw_p", "head_p"):
        v = np.linalg.norm(np.diff(o[k], axis=0), axis=-1) * FPS
        ok[1:] &= v < 4.0                                       # > 4 m/s is a tracking glitch, not a human
    ok &= np.linalg.norm(o["lw_p"] - o["head_p"], axis=-1) < 1.0
    ok &= np.linalg.norm(o["rw_p"] - o["head_p"], axis=-1) < 1.0
    if vmax is not None:
        ok &= np.append(walk_speed(o), 0.0) < vmax
    out, s = [], None
    for i, g in enumerate(np.append(ok, False)):
        if g and s is None: s = i
        if not g and s is not None:
            if i - s >= min_len: out.append((s, i))
            s = None
    return out


def walk_speed(o):
    """(T-1,) horizontal head speed averaged over 0.5 s: the walking speed, not the head bob."""
    v = np.diff(o["head_p"][:, :2], axis=0) * FPS
    k = int(0.5 * FPS)
    v = np.stack([np.convolve(v[:, i], np.ones(k) / k, mode="same") for i in range(2)], -1)
    return np.linalg.norm(v, axis=-1)


def build(max_egodex=0, out=OUT):
    clips, metas = [], []
    files = sorted(glob.glob(os.path.join(RAW, "egodex", "**", "*.hdf5"), recursive=True))
    if max_egodex: files = files[:max_egodex]
    for i, f in enumerate(files):
        try:
            d, fps = load_egodex(f)
        except Exception as e:                                   # truncated download etc.
            print("skip", f, e); continue
        o = to_50hz(d, fps)
        for a, b in segments(o, int(MIN_SEG_S * FPS)):
            c = {k: v[a:b].copy() for k, v in o.items()}
            # ARKit infers the shoulders of a head-mounted user: its torso heading sits right of the gaze in ~90 % of
            # EgoDex clips (median +7 deg; LAFAN mocap: 0). Re-centre the heading label on the clip's median gaze.
            bias = np.median(c["neck"][:, 0])
            c["torso_yaw"] += bias; c["neck"][:, 0] -= bias
            clips.append(c); metas.append(d["meta"])
        if i % 500 == 0: print(f"egodex {i}/{len(files)}  clips {len(clips)}", flush=True)
    for f in sorted(glob.glob(os.path.join(RAW, "lafan1", "*.bvh"))):
        if not os.path.basename(f).startswith(LAFAN_KEEP): continue
        d, fps = load_lafan(f)
        o1 = to_50hz(d, fps)
        rate = float(np.clip(MAX_HEAD_SPEED / max(np.percentile(walk_speed(o1), 60), 1e-3), MIN_RATE, 1.0))
        o = to_50hz(d, fps, rate)
        for a, b in segments(o, int(MIN_SEG_S * FPS), vmax=MAX_HEAD_SPEED * 1.15):
            # long mocap takes are cut into 20 s clips so sampling is uniform over content
            for c in range(a, b - int(MIN_SEG_S * FPS) + 1, int(20 * FPS)):
                e = min(b, c + int(20 * FPS))
                if e - c >= MIN_SEG_S * FPS:
                    clips.append({k: v[c:e] for k, v in o.items()}); metas.append(d["meta"] + f"@x{rate:.2f}")
        print(f"lafan {os.path.basename(f)} rate x{rate:.2f}", flush=True)
    keys = list(clips[0].keys())
    lens = np.array([len(c["head_p"]) for c in clips])
    arr = {k: np.concatenate([c[k] for c in clips]).astype(np.float32) for k in keys}
    arr["wrist_dz"] = np.float32(WRIST_DZ)
    np.savez_compressed(out, starts=np.concatenate([[0], np.cumsum(lens)[:-1]]), lens=lens, meta=np.array(metas), **arr)
    print(f"wrote {out}: {len(clips)} clips, {lens.sum() / FPS / 3600:.2f} h")


def stats():
    z = np.load(OUT)
    src = np.array([m.split(":")[0] for m in z["meta"]])
    for s in np.unique(src):
        m = src == s; print(f"{s}: {m.sum()} clips, {z['lens'][m].sum() / FPS / 60:.1f} min")
    e2s = z["eye_to_shoulder"]; print("eye->shoulder (m) median %.3f  p10 %.3f  p90 %.3f" % tuple(np.percentile(e2s, [50, 10, 90])))
    for k in ("lean", "torso_yaw"):
        print(k, "range p1/p99", np.percentile(z[k], [1, 99]).round(3))
    print("neck yaw/pitch/roll p1/p99 (deg)", np.degrees(np.percentile(z["neck"], [1, 99], axis=0)).round(1))
    print("wrist-head dist p50/p99", np.percentile(np.linalg.norm(z["lw_p"] - z["head_p"], axis=-1), [50, 99]).round(3))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--max-egodex", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--avp", default=None, help="Retarget Lab episodes/ folder: build data/avp_takes.npz")
    a = ap.parse_args()
    if a.avp: build_avp(os.path.expanduser(a.avp), a.out or AVP_OUT)
    elif a.stats: stats()
    else: build(a.max_egodex, a.out or OUT)
