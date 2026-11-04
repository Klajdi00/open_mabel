"""
hand_ik_retarget.py — keypoint-IK Vision Pro → ORCA hand retargeting.

A port of the WORKING dexterous-hand teleop planner
(~/Desktop/dexterous_hand/teleop/planners/planner_per_finger.py /
planner_avp.py) onto MABEL's ORCA hands, replacing the v1 anatomical
angle-extraction mapping (hand_retarget.py), which did not transfer to the
real hand. The approach is DexPilot-style fingertip IK:

  1. Both skeletons are expressed in a CANONICAL PALM FRAME built the same
     way on each side (origin → mid-knuckle = ŷ, palm normal ẑ from the
     index/pinky knuckle rays, x̂ completing) and normalized by the palm
     length. Identical construction on the human landmarks and on the
     robot's FK makes the mapping scale-free and chirality-free.
     Additionally each HUMAN finger is length-scaled about its knuckle so
     its (normalized) bone length matches the robot finger's — so a straight
     human finger maps to a straight ORCA finger, not a curled one.
  2. The 16 finger joints per hand are solved by damped GAUSS-NEWTON with
     an ANALYTIC Jacobian (∂p/∂θ = ω̂ × (p − c) per hinge), minimizing
        W_SHAPE  · Σ‖robot_kp − human_kp‖²                 (shape matching)
      + W_PINCH  · Σ‖(thumb−tip)_robot − (thumb−tip)_human‖²   (boosted when
                   the human pinch distance is small — the DexPilot trick)
      + W_SMOOTH · ‖q − q_prev‖²  +  W_REG · ‖q‖²
     bounded by joint ranges (projected), warm-started, 4 iterations.
  3. Each iteration costs ONE analytic mini-FK of the 17-body hand subtree
     plus a 16×16 normal-equation solve (~0.5 ms total) — versus ~40 ms for
     SLSQP's numerical Jacobian, which starved the physics loop's GIL.
  4. Targets stream to the hand's POSITION-SERVO actuators — a simple PD per
     joint (MuJoCo position actuator: τ = kp(q*−q) − kd·q̇). No extra
     filtering: the smoothness term + warm start regularize; the PD tracks.
     The same q* targets feed the real hand's PD loop unchanged.

Keypoint correspondence (per finger; thumb one joint deeper):

      human (ARKit wire name)        robot (hand subtree)
      ------------------------       --------------------------------
      <f>_knuckle            ↔       {side}_<f>_pp origin   (MCP)
      <f>_intermediate_base  ↔       {side}_<f>_ip origin   (PIP)
      <f>_tip                ↔       ip extrapolated tip

Solving runs on a BACKGROUND worker thread (one per retargeter, both hands)
paced at SOLVE_PERIOD per hand, so the physics loop never blocks on SLSQP;
`apply()` just deposits the newest skeleton and streams the latest solved
targets to the servos. mj_step releases the GIL, so solver and physics
overlap.
"""
import logging
import os
import threading
import time

import numpy as np
import mujoco
from scipy.optimize import minimize

# ── tuning (mirrors the reference planner) ───────────────────────────────────
W_SHAPE = 15.0          # normalized keypoint position matching
W_DIR = 30.0            # per-bone DIRECTION matching (kn→pip, pip→tip): drives
                        # flexion angles even where the position manifolds of
                        # human and ORCA fingers cannot coincide (measured:
                        # rescues deep curls / cold starts the position rows
                        # leave jammed at a bound corner). 2026-07 sweep:
                        # DIR 30 / SHAPE 15 = cold curl 0.89, fist 0.87,
                        # pinch 2.6 mm, 8.8 ms — dominant on every axis
W_PINCH_LOW = 0.0       # thumb↔fingertip relative vectors while NOT pinching —
                        # 2026-07 benchmark: any always-on pull toward the thumb
                        # costs ~0.07 of fist flexion; DexPilot gates it instead
W_PINCH_HIGH = 50.0     # … boosted when the human pinches THAT pair
PINCH_DIST_THRESH = 0.55   # normalized thumb–tip distance that counts as pinching
POWER_GRASP_PAIRS = 3   # ≥ this many "pinching" pairs = a fist, not a pinch —
                        # drop the pinch rows and let shape+direction close it
W_SMOOTH = 1.0
W_REG = 0.05
GN_ITERS = 4            # cold start (no prior solution)
# Warm-started iterations: with the warm start + smoothness term, 2 iterations
# track ARKit-rate finger motion within the solver's own noise floor (validated
# by exp_retarget_fidelity's hand battery: curl/pinch/mirror metrics unchanged)
# at half the per-solve cost — the pure-Python mini-FK is the whole bill.
GN_ITERS_WARM = int(os.environ.get("MABEL_HAND_GN_ITERS_WARM", "2"))
GN_DAMPING = 1e-2
SOLVE_PERIOD = 1.0 / 20.0   # per-hand solve rate (s); servos track in between

# Alternative solver="slsqp": the dexterous_hand planner_avp HandOptimizer cost
# (cosine-similarity shape term + thumb↔fingertip relative-pinch term + smoothness
# + regularization), minimized with SLSQP in the canonical palm frame. Same
# decoupled keypoint-IK family as the GN path, just the reference's objective.
SLSQP_W_COSINE = 20.0       # 1 − cosine(stacked robot kp, stacked human kp)
SLSQP_W_RELATIVE = 150.0    # ‖(thumb−finger)_robot − (thumb−finger)_human‖²  (pinch)
SLSQP_W_SMOOTH = 0.5        # ‖q − q_prev‖²  (warm-started, temporal smoothness)
SLSQP_W_REG = 0.02          # ‖q‖²
SLSQP_ITERS = 5             # maxiter (reference uses 5; warm start carries the rest)

# Fingertip extension beyond the last body, as a fraction of the last phalanx.
TIP_EXTRAPOLATE = 0.85

_FINGERS = ("index", "middle", "ring", "pinky")
_ALL_FINGERS = ("thumb",) + _FINGERS
_FINGER_NAME = {"thumb": "Thumb", "index": "Index", "middle": "Middle",
                "ring": "Ring", "pinky": "Pinky"}
# ARKit wire names use "little" for the pinky.
_WIRE_FINGER = {"index": "index", "middle": "middle", "ring": "ring", "pinky": "little"}


# ── small SE(3) helpers ──────────────────────────────────────────────────────

def _quat_mat(q):
    """MuJoCo quaternion [w,x,y,z] → 3×3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _axis_mat(axis, angle):
    """Rodrigues rotation about a unit axis."""
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    C = 1 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


def _canonical(pts, origin, i_mcp, p_mcp):
    """The reference planner's canonical palm frame + normalization."""
    mid = (i_mcp + p_mcp) / 2.0
    vy = mid - origin
    scale = np.linalg.norm(vy)
    if scale < 1e-6:
        scale = 1.0
    y_ax = vy / scale
    z_ax = np.cross(i_mcp - origin, p_mcp - origin)
    z_ax /= (np.linalg.norm(z_ax) + 1e-9)
    x_ax = np.cross(y_ax, z_ax)
    x_ax /= (np.linalg.norm(x_ax) + 1e-9)
    R = np.column_stack([x_ax, y_ax, z_ax])
    return {k: ((p - origin) @ R) / scale for k, p in pts.items()}


def human_keypoints(joints, chirality):
    """ARKit wire joints {name: 4x4 world} → canonical normalized keypoints
    {(finger, level): xyz} with level ∈ {kn, pip, tip}; None if too sparse."""
    def P(name):
        T = joints.get(name)
        return None if T is None else np.asarray(T, dtype=float)[:3, 3]

    wrist = P("wrist")
    i_kn = P("index_knuckle")
    p_kn = P("little_knuckle")
    if wrist is None or i_kn is None or p_kn is None:
        return None

    raw = {}
    for f in _FINGERS:
        w = _WIRE_FINGER[f]
        for level, name in (("kn", f"{w}_knuckle"),
                            ("pip", f"{w}_intermediate_base"),
                            ("tip", f"{w}_tip")):
            p = P(name)
            if p is not None:
                raw[(f, level)] = p
    for level, name in (("kn", "thumb_intermediate_base"),
                        ("pip", "thumb_intermediate_tip"),
                        ("tip", "thumb_tip")):
        p = P(name)
        if p is not None:
            raw[("thumb", level)] = p
    if len(raw) < 6:
        return None
    return _canonical(raw, wrist, i_kn, p_kn)


# ── analytic FK of one hand subtree ──────────────────────────────────────────

class _HandFK:
    """Precomputed palm-relative FK for the ORCA hand chain. Every finger
    body carries exactly one hinge; transforms compose as
        T_body = T_parent · (pos, quat) · Rot(anchor, axis, q)."""

    def __init__(self, model, palm_bid):
        self.bodies = []                 # topological: dicts with parent index
        self.index_of = {}               # body name → index in self.bodies
        ids = {palm_bid: -1}             # mj body id → local index (-1 = palm)

        # Gather the palm's hand descendants in mj order (parents precede kids).
        for bid in range(palm_bid + 1, model.nbody):
            parent = model.body_parentid[bid]
            if parent not in ids:
                continue
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
            if not any(f in name for f in _ALL_FINGERS):
                continue
            joints = []
            for j in range(model.njnt):
                if model.jnt_bodyid[j] == bid:
                    joints.append({
                        "axis": model.jnt_axis[j].copy(),
                        "anchor": model.jnt_pos[j].copy(),
                        # MuJoCo hinges rotate by (qpos − ref); the compiler
                        # stores each joint's XML `ref` in qpos0.
                        "ref": float(model.qpos0[model.jnt_qposadr[j]]),
                        "id": j,
                    })
            ids[bid] = len(self.bodies)
            self.bodies.append({
                "name": name,
                "parent": ids[parent],
                "pos": model.body_pos[bid].copy(),
                "R": _quat_mat(model.body_quat[bid]),
                "joints": joints,
            })
            self.index_of[name] = ids[bid]

    def ancestors(self, body_name):
        """Solver q-indices of every joint between the palm and this body
        (inclusive) — the Jacobian columns that move it."""
        out = []
        i = self.index_of.get(body_name)
        while i is not None and i >= 0:
            b = self.bodies[i]
            out.extend(j["qi"] for j in b["joints"] if j.get("qi") is not None)
            i = b["parent"] if b["parent"] >= 0 else None
        return out

    def remap(self, joint_ids):
        """Point each hinge at its index within the solver's q vector."""
        lookup = {jid: i for i, jid in enumerate(joint_ids)}
        for b in self.bodies:
            for j in b["joints"]:
                j["qi"] = lookup.get(j["id"])

    def frames(self, q, jout=None):
        """{body name: (R, p)} palm-frame pose for a joint vector q. The
        rotation matters: the fingertip extends along the LAST body's bone
        axis, which is exactly what the distal joints rotate — extrapolating
        from origins alone would make the tip insensitive to PIP/DIP.

        If `jout` is a dict it receives, per solver q-index, the joint's
        WORLD (palm-frame) rotation axis and centre — everything the analytic
        Jacobian needs: ∂p/∂θ = ω̂ × (p − c)."""
        Rs = [None] * len(self.bodies)
        ps = [None] * len(self.bodies)
        out = {}
        for i, b in enumerate(self.bodies):
            if b["parent"] < 0:
                Rp, pp = np.eye(3), np.zeros(3)
            else:
                Rp, pp = Rs[b["parent"]], ps[b["parent"]]
            R = Rp @ b["R"]
            p = pp + Rp @ b["pos"]
            for j in b["joints"]:
                qi = j.get("qi")
                if qi is None:
                    continue
                Rj = _axis_mat(j["axis"], q[qi] - j["ref"])
                # rotate about the anchor: origin shifts, frame rotates
                p = p + R @ (j["anchor"] - Rj @ j["anchor"])
                R = R @ Rj
                if jout is not None:
                    # axis is Rj's eigenvector → R@axis is the world axis;
                    # rotation centre c = p + R@anchor (joint-invariant).
                    jout[qi] = (R @ j["axis"], p + R @ j["anchor"])
            Rs[i], ps[i] = R, p
            out[b["name"]] = (R, p)
        return out

    def positions(self, q):
        """{body name: palm-frame origin} (origins only, for tests)."""
        return {k: v[1] for k, v in self.frames(q).items()}


class HandIKRetargeter:
    """Per-hand keypoint-IK solver + PD target streamer for the ORCA hands."""

    def __init__(self, model, data, solver="gn"):
        self.model = model
        self.data = data
        self.solver = solver          # "gn" (default) | "slsqp" (reference cost)
        self.hands = {}
        # Per-operator calibrated finger lengths (canonical-normalised), keyed
        # "Left"/"Right" -> {finger: norm_len}. Empty = estimate live per frame
        # (legacy). Set via set_profile() from an OperatorProfile so each operator's
        # hand size is measured once and frozen instead of re-estimated noisily.
        self._human_lengths = {}
        for side in ("Left", "Right"):
            h = self._index_hand(side)
            if h is not None:
                self.hands[side] = h

    def set_profile(self, profile):
        """Freeze per-finger human bone lengths from an OperatorProfile (measured
        during calibration) so _calibrate uses a fixed per-user scale per finger
        instead of a noisy live estimate. profile=None reverts to the live estimate."""
        self._human_lengths = {}
        if profile is None:
            return
        for side, key in (("Left", "left"), ("Right", "right")):
            lens = profile.finger_len_norm.get(key)
            if lens:
                self._human_lengths[side] = dict(lens)

    def reset(self):
        """Clear solver history so the fingers don't snap to a stale pose after a
        sim reset (mirrors the reference HandOptimizer.reset_optimizer)."""
        for hand in self.hands.values():
            hand["last_q"][:] = 0.0
            hand["have_solution"] = False
            hand["written_gen"] = -1

    # ── model indexing ───────────────────────────────────────────────────────
    def _index_hand(self, side):
        m = self.model
        pre = side.lower()
        palm = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{pre}_palm")
        if palm < 0:
            return None

        joints, act_names, bounds, joint_ids, act_ids = [], [], [], [], []
        for f, F in _FINGER_NAME.items():
            segs = ("MCP", "Abd", "PIP", "DIP") if f == "thumb" else ("Abd", "MCP", "PIP")
            for seg in segs:
                name = f"{side} {F} {seg}"
                jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
                aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                if jid < 0 or aid < 0:
                    continue
                joint_ids.append(jid)
                act_names.append(name)
                act_ids.append(aid)
                bounds.append(tuple(m.jnt_range[jid]))
        if not joint_ids:
            return None

        fk = _HandFK(m, palm)
        fk.remap(joint_ids)

        # Distal segment lengths (metres): the bone the tip extends along is
        # the last body's own offset from its parent (fingers: ip; thumb: dp).
        tiplen = {}
        for f in _FINGERS:
            i = fk.index_of.get(f"{pre}_{f}_ip")
            if i is not None:
                tiplen[f] = float(np.linalg.norm(fk.bodies[i]["pos"]))
        i = fk.index_of.get(f"{pre}_thumb_dp")
        if i is not None:
            tiplen["thumb"] = float(np.linalg.norm(fk.bodies[i]["pos"]))

        hand = {
            "fk": fk, "pre": pre, "tiplen": tiplen,
            "act_names": act_names, "act_ids": np.array(act_ids), "bounds": bounds,
            "last_q": np.zeros(len(joint_ids)),
            "last_solve": 0.0,
            "have_solution": False,
            "q_gen": 0,                  # bumped by the worker per solution
            "written_gen": -1,           # last generation streamed to ctrl
            "last_write": 0.0,           # periodic re-write (post-reset safety)
        }
        # Calibration anchors: the robot's canonical knuckle positions and
        # finger bone lengths at q = 0. Human fingers are scaled to these
        # lengths AND translated so their knuckles coincide with the robot's —
        # palm-shape differences then cancel instead of biasing the flexion.
        kp0 = self._robot_keypoints(hand, np.zeros(len(joint_ids)))
        hand["lengths"] = {
            f: float(np.linalg.norm(kp0[(f, "pip")] - kp0[(f, "kn")])
                     + np.linalg.norm(kp0[(f, "tip")] - kp0[(f, "pip")]))
            for f in _ALL_FINGERS if (f, "tip") in kp0
        }
        hand["kn0"] = {f: kp0[(f, "kn")] for f in _ALL_FINGERS if (f, "kn") in kp0}
        # Jacobian sparsity: which solver joints move each keypoint.
        anc = {}
        for f in _FINGERS:
            anc[(f, "kn")] = fk.ancestors(f"{pre}_{f}_pp")
            anc[(f, "pip")] = fk.ancestors(f"{pre}_{f}_ip")
            anc[(f, "tip")] = anc[(f, "pip")]
        anc[("thumb", "kn")] = fk.ancestors(f"{pre}_thumb_pp")
        anc[("thumb", "pip")] = fk.ancestors(f"{pre}_thumb_ip")
        anc[("thumb", "tip")] = fk.ancestors(f"{pre}_thumb_dp")
        hand["kp_anc"] = anc
        hand["lo"] = np.array([b[0] for b in bounds])
        hand["hi"] = np.array([b[1] for b in bounds])
        hand["pending"] = None           # newest human keypoints (worker input)
        hand["lock"] = threading.Lock()
        return hand

    # ── robot canonical skeleton (analytic mini-FK) ──────────────────────────
    def _robot_keypoints(self, hand, q):
        frames = hand["fk"].frames(q)
        pre = hand["pre"]
        tiplen = hand["tiplen"]
        raw = {}
        for f in _FINGERS:
            kn = frames.get(f"{pre}_{f}_pp")
            pip = frames.get(f"{pre}_{f}_ip")
            if kn is None or pip is None:
                continue
            R_ip, p_ip = pip
            raw[(f, "kn")] = kn[1]
            raw[(f, "pip")] = p_ip
            # Tip extends along the ip body's bone axis (local +z), so it
            # ROTATES with the PIP joint — the solver can feel it.
            raw[(f, "tip")] = p_ip + R_ip @ np.array(
                [0.0, 0.0, tiplen.get(f, 0.03) * TIP_EXTRAPOLATE])
        t_kn = frames.get(f"{pre}_thumb_pp")
        t_pip = frames.get(f"{pre}_thumb_ip")
        t_dp = frames.get(f"{pre}_thumb_dp")
        if t_kn is not None and t_pip is not None:
            raw[("thumb", "kn")] = t_kn[1]
            raw[("thumb", "pip")] = t_pip[1]
            if t_dp is not None:
                R_dp, p_dp = t_dp
                raw[("thumb", "tip")] = p_dp + R_dp @ np.array(
                    [0.0, 0.0, tiplen.get("thumb", 0.03) * TIP_EXTRAPOLATE])
        return _canonical(raw, np.zeros(3),
                          frames[f"{pre}_index_mp"][1], frames[f"{pre}_pinky_mp"][1])

    def _kp_and_jac(self, hand, q):
        """Canonical keypoints AND their analytic Jacobians (3×n each) at q.
        The canonical frame (R, scale) is frozen at the current q — its own
        q-dependence (through the abd joints) is weak, and Gauss-Newton only
        needs a descent direction."""
        jout = {}
        frames = hand["fk"].frames(q, jout)
        pre = hand["pre"]
        tiplen = hand["tiplen"]
        n = len(q)

        world = {}
        for f in _FINGERS:
            kn = frames.get(f"{pre}_{f}_pp")
            pip = frames.get(f"{pre}_{f}_ip")
            if kn is None or pip is None:
                continue
            R_ip, p_ip = pip
            world[(f, "kn")] = kn[1]
            world[(f, "pip")] = p_ip
            world[(f, "tip")] = p_ip + R_ip @ np.array(
                [0.0, 0.0, tiplen.get(f, 0.03) * TIP_EXTRAPOLATE])
        t_kn = frames.get(f"{pre}_thumb_pp")
        t_pip = frames.get(f"{pre}_thumb_ip")
        t_dp = frames.get(f"{pre}_thumb_dp")
        if t_kn is not None and t_pip is not None:
            world[("thumb", "kn")] = t_kn[1]
            world[("thumb", "pip")] = t_pip[1]
            if t_dp is not None:
                R_dp, p_dp = t_dp
                world[("thumb", "tip")] = p_dp + R_dp @ np.array(
                    [0.0, 0.0, tiplen.get("thumb", 0.03) * TIP_EXTRAPOLATE])

        # Frozen canonical transform.
        i_mp = frames[f"{pre}_index_mp"][1]
        p_mp = frames[f"{pre}_pinky_mp"][1]
        mid = (i_mp + p_mp) / 2.0
        scale = np.linalg.norm(mid)
        if scale < 1e-6:
            scale = 1.0
        y_ax = mid / scale
        z_ax = np.cross(i_mp, p_mp)
        z_ax /= (np.linalg.norm(z_ax) + 1e-9)
        x_ax = np.cross(y_ax, z_ax)
        x_ax /= (np.linalg.norm(x_ax) + 1e-9)
        Rc = np.column_stack([x_ax, y_ax, z_ax])

        kp, jac = {}, {}
        for key, p in world.items():
            kp[key] = (p @ Rc) / scale
            J = np.zeros((3, n))
            for qi in hand["kp_anc"].get(key, ()):
                axis, centre = jout[qi]
                J[:, qi] = np.cross(axis, p - centre)
            jac[key] = (Rc.T @ J) / scale
        return kp, jac

    @staticmethod
    def _calibrate(h_kp, lengths, kn0, human_lengths=None):
        """Per finger: scale about the knuckle so the canonical bone length
        matches the robot's, then translate the whole finger so the human
        knuckle lands ON the robot's rest knuckle. A straight human finger
        then maps to a straight ORCA finger regardless of palm shape.

        ``human_lengths`` (optional {finger: norm_len}) supplies the operator's
        CALIBRATED per-finger length so the scale ratio is fixed per user; without
        it the human length is estimated from the live frame (legacy, noisier)."""
        out = dict(h_kp)
        for f, l_robot in lengths.items():
            kn, pip, tip = h_kp.get((f, "kn")), h_kp.get((f, "pip")), h_kp.get((f, "tip"))
            if kn is None or pip is None or tip is None:
                continue
            l_human = None
            if human_lengths is not None:
                l_human = human_lengths.get(f)
            if l_human is None:
                l_human = float(np.linalg.norm(pip - kn) + np.linalg.norm(tip - pip))
            if l_human < 1e-6:
                continue
            r = l_robot / l_human
            anchor = kn0.get(f, kn)
            out[(f, "kn")] = anchor
            out[(f, "pip")] = anchor + (pip - kn) * r
            out[(f, "tip")] = anchor + (tip - kn) * r
        return out

    # ── background solver ────────────────────────────────────────────────────
    def _ensure_worker(self):
        if getattr(self, "_worker", None) is None:
            self._worker = threading.Thread(target=self._solve_loop, daemon=True,
                                            name="hand-ik-retarget")
            self._worker.start()

    def _solve_loop(self):
        log = logging.getLogger("hand_ik")
        while True:
            worked = False
            for hand in self.hands.values():
                with hand["lock"]:
                    pending = hand["pending"]
                    hand["pending"] = None
                if pending is None:
                    continue
                try:
                    h_shape, h_pinch = pending
                    solve = (self._solve_slsqp if self.solver == "slsqp"
                             else self._solve_gn)
                    hand["last_q"] = solve(hand, h_shape, h_pinch)
                    hand["have_solution"] = True
                    hand["q_gen"] += 1          # mark a fresh solution for apply()
                except Exception:
                    # A dead worker must never silently freeze the fingers.
                    log.exception("hand IK solve failed; continuing")
                worked = True
            time.sleep(0.001 if worked else 0.01)

    def _solve_once(self, hand, h_shape, h_pinch):
        """Synchronous solve (tests / offline tools)."""
        solve = self._solve_slsqp if self.solver == "slsqp" else self._solve_gn
        hand["last_q"] = solve(hand, h_shape, h_pinch)
        hand["have_solution"] = True
        hand["q_gen"] += 1

    def _solve_slsqp(self, hand, h_shape, h_pinch):
        """Reference (planner_avp) HandOptimizer cost, minimized with SLSQP over
        the hand's joints in the canonical palm frame:

            loss = W_COSINE·(1 − cos⟨robot_kp, human_kp⟩)            (shape)
                 + W_RELATIVE·Σ_f ‖(thumb_tip − f_tip)_r − (…)_h‖²   (pinch)
                 + W_SMOOTH·‖q − q_prev‖² + W_REG·‖q‖²

        robot_kp = self._robot_keypoints(hand, q) (analytic mini-FK, canonical),
        human_kp = the calibrated human keypoints. Warm-started from last_q,
        bounded by the joint ranges. ~5 SLSQP iters, exactly like the reference."""
        q0 = np.asarray(hand["last_q"], float)

        def cost(q):
            r = self._robot_keypoints(hand, q)
            loss = 0.0
            common = [k for k in r if k in h_shape]
            if common:
                rv = np.concatenate([r[k] for k in common])
                hv = np.concatenate([h_shape[k] for k in common])
                cos = float(rv @ hv / ((np.linalg.norm(rv) + 1e-6)
                                       * (np.linalg.norm(hv) + 1e-6)))
                loss += SLSQP_W_COSINE * (1.0 - cos)
            t_r, t_h = r.get(("thumb", "tip")), h_shape.get(("thumb", "tip"))
            if t_r is not None and t_h is not None:
                for f in _FINGERS:
                    f_r, f_h = r.get((f, "tip")), h_shape.get((f, "tip"))
                    if f_r is not None and f_h is not None:
                        loss += SLSQP_W_RELATIVE * float(
                            np.sum(((f_r - t_r) - (f_h - t_h)) ** 2))
            loss += SLSQP_W_SMOOTH * float(np.sum((q - q0) ** 2))
            loss += SLSQP_W_REG * float(np.sum(q ** 2))
            return loss

        res = minimize(cost, q0, method="SLSQP", bounds=hand["bounds"],
                       options={"maxiter": SLSQP_ITERS, "ftol": 1e-3})
        return np.clip(res.x, hand["lo"], hand["hi"])

    def _solve_gn(self, hand, h_shape, h_pinch):
        """Damped Gauss-Newton on the stacked weighted residuals: shape
        keypoints, per-bone direction matching, gated DexPilot pinch vectors,
        smoothness, regularization. Bounds by projection — plus a CORNER
        ESCAPE: projected GN can jam a finger with MCP at one bound and PIP at
        the other (cold starts / deep curls); when the primary solve ends in
        that corner with a poor residual, re-solve once from mid-range and
        keep the better solution (measured: rescues cold-start full curls)."""
        warm = hand.get("have_solution", False)
        iters = GN_ITERS_WARM if warm else GN_ITERS
        q_best = self._solve_gn_from(hand, hand["last_q"].copy(),
                                     h_shape, h_pinch, iters=iters)
        # The corner escape is a COLD-START rescue (projected GN jammed at a
        # bound corner before any solution exists); warm solves inherit an
        # un-jammed pose, and the full mid-range re-solve would double the
        # per-frame cost for nothing.
        if not warm and self._corner_jammed(hand, q_best):
            q_mid = 0.5 * (hand["lo"] + hand["hi"])
            q_alt = self._solve_gn_from(hand, q_mid, h_shape, h_pinch,
                                        iters=GN_ITERS)
            if (self._shape_cost(hand, q_alt, h_shape)
                    < self._shape_cost(hand, q_best, h_shape)):
                q_best = q_alt
        return q_best

    def _corner_jammed(self, hand, q, tol=1e-6):
        """True when any finger ends with one flexion joint at a bound's floor
        and its partner at the ceiling — the projected-GN corner signature."""
        at_lo = np.abs(q - hand["lo"]) < tol
        at_hi = np.abs(q - hand["hi"]) < tol
        names = hand["act_names"]
        for f in ("Index", "Middle", "Ring", "Pinky", "Thumb"):
            idx = [i for i, nm in enumerate(names)
                   if f in nm and ("MCP" in nm or "PIP" in nm)]
            if len(idx) == 2:
                a, b = idx
                if (at_lo[a] and at_hi[b]) or (at_hi[a] and at_lo[b]):
                    return True
        return False

    def _shape_cost(self, hand, q, h_shape):
        r = self._robot_keypoints(hand, q)
        return float(sum(np.sum((r[k] - h_shape[k]) ** 2)
                         for k in r if k in h_shape))

    def _solve_gn_from(self, hand, q, h_shape, h_pinch, iters=GN_ITERS):
        q_prev = hand["last_q"]
        n = len(q)
        sw_shape = np.sqrt(W_SHAPE)
        sw_dir = np.sqrt(W_DIR)
        sw_smooth = np.sqrt(W_SMOOTH)
        sw_reg = np.sqrt(W_REG)

        # pinch gating (DexPilot): boost a thumb–finger pair only when THAT
        # pair pinches; when ≥ POWER_GRASP_PAIRS pairs read "pinching" the
        # operator is closing a fist — drop the pinch rows entirely so they
        # cannot fight full flexion (measured +0.07 fist flexion).
        h_thumb0 = h_pinch.get(("thumb", "tip"))
        pinching = []
        if h_thumb0 is not None:
            for f in _FINGERS:
                h_tip = h_pinch.get((f, "tip"))
                if h_tip is not None and \
                        np.linalg.norm(h_thumb0 - h_tip) < PINCH_DIST_THRESH:
                    pinching.append(f)
        power_grasp = len(pinching) >= POWER_GRASP_PAIRS

        for _ in range(iters):
            kp, jac = self._kp_and_jac(hand, q)
            rows_r, rows_J = [], []
            for key, p in kp.items():
                h = h_shape.get(key)
                if h is None:
                    continue
                rows_r.append(sw_shape * (p - h))
                rows_J.append(sw_shape * jac[key])
            # bone-direction rows: unit(kn→pip) and unit(pip→tip) per finger.
            # J_u = (I − u uᵀ)/|d| · (J_b − J_a)
            for f in _ALL_FINGERS:
                for a, b in (((f, "kn"), (f, "pip")), ((f, "pip"), (f, "tip"))):
                    pa, pb = kp.get(a), kp.get(b)
                    ha, hb = h_shape.get(a), h_shape.get(b)
                    if pa is None or pb is None or ha is None or hb is None:
                        continue
                    d_r = pb - pa
                    d_h = hb - ha
                    L_r = np.linalg.norm(d_r)
                    L_h = np.linalg.norm(d_h)
                    if L_r < 1e-6 or L_h < 1e-6:
                        continue
                    u_r, u_h = d_r / L_r, d_h / L_h
                    P = (np.eye(3) - np.outer(u_r, u_r)) / L_r
                    rows_r.append(sw_dir * (u_r - u_h))
                    rows_J.append(sw_dir * (P @ (jac[b] - jac[a])))
            h_thumb = h_thumb0
            r_thumb = kp.get(("thumb", "tip"))
            if h_thumb is not None and r_thumb is not None and not power_grasp:
                for f in pinching if W_PINCH_LOW == 0 else _FINGERS:
                    h_tip, r_tip = h_pinch.get((f, "tip")), kp.get((f, "tip"))
                    if h_tip is None or r_tip is None:
                        continue
                    h_rel = h_thumb - h_tip
                    w = np.sqrt(W_PINCH_HIGH if f in pinching else W_PINCH_LOW)
                    if w == 0.0:
                        continue
                    rows_r.append(w * ((r_thumb - r_tip) - h_rel))
                    rows_J.append(w * (jac[("thumb", "tip")] - jac[(f, "tip")]))
            rows_r.append(sw_smooth * (q - q_prev))
            rows_J.append(sw_smooth * np.eye(n))
            rows_r.append(sw_reg * q)
            rows_J.append(sw_reg * np.eye(n))

            r = np.concatenate([np.atleast_1d(x).ravel() for x in rows_r])
            J = np.vstack([x if x.ndim == 2 else x[None, :] for x in rows_J])
            H = J.T @ J + GN_DAMPING * np.eye(n)
            dq = np.linalg.solve(H, -(J.T @ r))
            q = np.clip(q + dq, hand["lo"], hand["hi"])
        return q

    # ── public API ───────────────────────────────────────────────────────────
    def apply(self, side, joints, chirality=None, synchronous=False):
        """side: 'Left' | 'Right'; joints: {wire_name: 4x4 world}. Deposits
        the newest skeleton for the background solver (paced at SOLVE_PERIOD)
        and streams the latest solved targets to the hand's position-servo
        (PD) actuators. Returns the target dict or None when nothing has been
        solved yet. `synchronous=True` solves inline (tests)."""
        hand = self.hands.get(side)
        if hand is None:
            return None

        now = time.monotonic()
        if now - hand["last_solve"] >= SOLVE_PERIOD:
            h_raw = human_keypoints(joints, chirality or side.lower())
            if h_raw is not None:
                h_shape = self._calibrate(h_raw, hand["lengths"], hand["kn0"],
                                          self._human_lengths.get(side))
                hand["last_solve"] = now
                if synchronous:
                    self._solve_once(hand, h_shape, h_raw)
                else:
                    with hand["lock"]:
                        hand["pending"] = (h_shape, h_raw)
                    self._ensure_worker()

        if not hand["have_solution"]:
            return None
        # apply() runs every physics SUBSTEP, but ctrl persists across
        # mj_step — write only when the worker produced a NEW solution (plus
        # a 5 Hz keep-alive so a sim reset that zeroes ctrl recovers). With
        # cached actuator ids this is one vectorized assignment, not 16 name
        # lookups per substep.
        if hand["written_gen"] != hand["q_gen"] or now - hand["last_write"] > 0.2:
            q = np.clip(hand["last_q"],
                        [b[0] for b in hand["bounds"]],
                        [b[1] for b in hand["bounds"]])
            self.data.ctrl[hand["act_ids"]] = q
            hand["written_gen"] = hand["q_gen"]
            hand["last_write"] = now
        return dict(zip(hand["act_names"], hand["last_q"]))
