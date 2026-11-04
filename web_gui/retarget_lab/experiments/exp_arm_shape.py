"""Straight-arm shape test: does the robot's arm take the operator's arm SHAPE when the operator holds both
arms fully extended (forward, sideways, up, down, diagonals)?

A synthetic operator (an EgoDex pose, head fixed) swings both arms, mirrored, RIGIDLY about the shoulder
(hand position AND orientation, like a real straight arm) from rest to shoulder + 0.99 L_h * u over 2 s
and holds 1 s. Per pose and side, at the end of the hold:
  dir err   angle between the robot's shoulder->palm and the human's shoulder->wrist direction [deg]
  tgt err   the same angle for the map's TARGET (map error, before IK) [deg]
  ext       |palm - shoulder| / the robot's reach in that direction (1 = fully extended)
  bend      equivalent elbow bend [deg] from the joint anchors (Arm 2 -> Arm 4 -> Wrist), 0 = as straight as
            this arm gets (the elbow sweep's longest shoulder->wrist distance; body origins are offset)
  miss      |target - palm| [cm]

    cd web_gui && MABEL_RTL_ROBOT=nova python -m retarget_lab.experiments.exp_arm_shape
    EXP_VARIANT=shipped_v1 ... runs a lab variant instead of "ours" (one per process: variants pin module knobs).
"""
import gzip
import json
import os

import mujoco
import numpy as np

from retarget_lab.server.worker import KinematicRetargeter, pose_from_frame
from retarget_lab.tests.test_wholebody import CLIPS, _shift

DIRS = {                                   # human body frame: (fwd, out, up); "out" = away from the midline
    "forward": (1, 0, 0), "sideways": (0, 1, 0), "up": (0, 0, 1), "down": (0, 0, -1),
    "fwd+up 45": (1, 0, 1), "fwd+out 45": (1, 1, 0), "out+up 45": (0, 1, 1), "fwd+in 30": (1, -0.577, 0),
}


def _rigid(mat16, R, c, scale_to=None):
    """Rotate one flat 4x4 (either storage order) rigidly by R about point c."""
    M = np.asarray(mat16, float).reshape(4, 4)
    row = np.allclose(M[3], [0, 0, 0, 1])
    T = M if row else M.T
    T = T.copy()
    T[:3, :3] = R @ T[:3, :3]
    T[:3, 3] = c + R @ (T[:3, 3] - c)
    return (T if row else T.T).reshape(-1).tolist()


def _rot_between(a, b, frac=1.0):
    """Rotation taking unit a toward unit b by `frac` of the angle (Rodrigues)."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    k = np.cross(a, b); s = np.linalg.norm(k); c = float(np.clip(a @ b, -1, 1))
    if s < 1e-9:
        if c > 0:
            return np.eye(3)
        k = np.cross(a, [1.0, 0, 0] if abs(a[0]) < 0.9 else [0, 1.0, 0]); s = np.linalg.norm(k)
    k = k / s; th = frac * np.arctan2(s, c)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def _move_side(q, side, R, c, w_after, w_before):
    """Swing this side's hand (all joints) rigidly by R about the shoulder c, then translate so the wrist
    lands on w_after (stretches a bent rest arm straight); the elbow sits at the arm's midpoint."""
    for k in (f"{side}_hand_mats",):
        if q.get(k):
            q[k] = [_rigid(m, R, c) for m in q[k]]
    k = f"{side}_joints"
    if q.get(k):
        q[k] = {n: _rigid(m, R, c) for n, m in q[k].items()}
    dp = w_after - (c + R @ (w_before - c))
    for k in (f"{side}_hand_mats",):
        if q.get(k):
            q[k] = [_shift(m, dp) for m in q[k]]
    k = f"{side}_joints"
    if q.get(k):
        q[k] = {n: _shift(m, dp) for n, m in q[k].items()}
    k = f"{side}_elbow"
    if q.get(k):
        M = np.asarray(q[k], float).reshape(4, 4)
        cur = M[:3, 3] if np.allclose(M[3], [0, 0, 0, 1]) else M[3, :3]
        q[k] = _shift(q[k], 0.5 * (c + w_after) - cur)
    return q


def _wrist(p, side):
    M = np.asarray(p[f"{side}_hand_mats"][0], float).reshape(4, 4)
    return M[:3, 3].copy() if np.allclose(M[3], [0, 0, 0, 1]) else M[3, :3].copy()


def _anchors(m, d, S):
    J = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
    return d.xanchor[J(f"{S} Arm 2")].copy(), d.xanchor[J(f"{S} Arm 4")].copy(), d.xanchor[J(f"{S} Wrist")].copy()


def _gamma(s, e, w):
    a, b, c = np.linalg.norm(e - s), np.linalg.norm(w - e), np.linalg.norm(w - s)
    return float(np.degrees(np.arccos(np.clip((a * a + b * b - c * c) / (2 * a * b), -1, 1))))


_STRAIGHT = {}


def straightest(m, S):
    """The elbow's most open angle (law of cosines on the anchors) over joints 3-5 (the elbow's offset makes
    the upper-arm / forearm rolls matter too)."""
    if S not in _STRAIGHT:
        d = mujoco.MjData(m)
        js = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{S} Arm {i}") for i in (3, 4, 5)]
        grids = [np.linspace(*m.jnt_range[j], n) for j, n in zip(js, (25, 121, 25))]
        best = 0.0
        for q3 in grids[0]:
            for q4 in grids[1]:
                for q5 in grids[2]:
                    d.qpos[:] = m.qpos0
                    for j, v in zip(js, (q3, q4, q5)):
                        d.qpos[m.jnt_qposadr[j]] = v
                    mujoco.mj_kinematics(m, d)
                    best = max(best, _gamma(*_anchors(m, d, S)))
        _STRAIGHT[S] = best
    return _STRAIGHT[S]


def run_pose(name, u_body, settings=None, n_ramp=120, n_hold=60, tweak=None):
    p0 = json.load(gzip.open(CLIPS[0]))["payloads"][0]
    kr = KinematicRetargeter(os.environ.get("EXP_VARIANT", "ours")); kr.apply_settings({"scope": "local", "robot": "nova", **(settings or {})})
    eng = kr.eng
    if tweak is not None:                                       # ablations: tweak(eng) edits the arm solvers
        tweak(eng)
    for i in range(60):                                         # calibrate at rest
        kr.step(pose_from_frame({"kind": "avp", "p": p0}), i / 60.0)
    right, up, fwd = eng.shoulder.body_axes()
    goal, w0 = {}, {}
    for side in ("right", "left"):
        f, o, z = u_body
        out = right if side == "right" else -right
        u = f * fwd + o * out + z * up
        u = u / np.linalg.norm(u)
        sh, L = eng.shoulder.shoulder_world(side), eng.shoulder.arm_length(side)
        goal[side] = (sh, L, u)
        w0[side] = _wrist(p0, side)
    t = 60
    for i in range(n_ramp + n_hold):
        a = min(1.0, (i + 1) / n_ramp)
        a = a * a * (3 - 2 * a)
        q = json.loads(json.dumps(p0))
        for side, (sh, L, u) in goal.items():
            R = _rot_between(w0[side] - sh, u, a)                  # swing about the shoulder
            r0 = np.linalg.norm(w0[side] - sh)
            w = sh + (R @ ((w0[side] - sh) / r0)) * ((1 - a) * r0 + a * 0.99 * L)
            _move_side(q, side, R, sh, w, w0[side])
        kr.step(pose_from_frame({"kind": "avp", "p": q}), (t + i) / 60.0)
    d = kr.data
    bR = d.xmat[eng.base_bid].reshape(3, 3)
    A = np.diag([-1.0, -1.0, 1.0]) @ bR.T                       # world -> robot frame (x fwd, y left, z up)
    rows = {}
    for side, (sh_h, L, u) in goal.items():
        arm = eng.arm[side]
        s, e, w = (np.asarray(d.xpos[b], float) for b in (arm.shoulder_bid, arm.elbow_bid, arm.palm_bid))
        # human direction in the robot frame: fwd -> +x, left -> +y, up -> +z
        uh = np.array([u @ fwd, -(u @ right), u @ up])
        ur = A @ (w - s); ur /= np.linalg.norm(ur)
        tg = kr._tgt.get(side)
        ut = A @ (tg - s); ut /= np.linalg.norm(ut)
        reach = eng.arm_reach_dir[side].at((w - s) / np.linalg.norm(w - s)) if eng.arm_reach_dir is not None else eng.arm_reach[side]
        S = side.capitalize()
        bend = straightest(kr.model, S) - _gamma(*_anchors(kr.model, d, S))
        ang = lambda x, y: float(np.degrees(np.arccos(np.clip(x @ y, -1, 1))))
        rows[side] = dict(dir=ang(ur, uh), tgt=ang(ut, uh), ext=float(np.linalg.norm(w - s) / reach),
                          bend=float(bend), miss=float(np.linalg.norm(tg - w) * 100))
    return rows


def main(settings=None, label="", tweak=None):
    print(f"{label}{'pose':12s} {'side':5s} {'dir err':>8s} {'tgt err':>8s} {'ext':>6s} {'bend':>6s} {'miss cm':>8s}")
    res = {}
    for name, u in DIRS.items():
        rows = run_pose(name, u, settings, tweak=tweak)
        res[name] = rows
        for side, r in rows.items():
            print(f"{label}{name:12s} {side:5s} {r['dir']:8.1f} {r['tgt']:8.1f} {r['ext']:6.2f} {r['bend']:6.1f} {r['miss']:8.1f}")
    return res


if __name__ == "__main__":
    main()
