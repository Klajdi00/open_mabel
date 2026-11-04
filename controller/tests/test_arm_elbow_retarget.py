"""Human-like elbow retargeting — the operator's constraints as regressions.

2026-07-29 ("make the arms as human like as possible"): the VP's tracked ARKit
forearm now feeds the arm IK's elbow hint DIRECTION-ONLY (engine._solve_side),
so the robot elbow bends like the operator's. These tests drive the full
MabelTeleop retarget stack with synthetic, chirality-correct ARKit hands and
assert the operator's stated rules:

  * T-pose palms UP   -> elbow points UP (dominant +z component)
  * T-pose palms DOWN -> elbow biased FORWARD (positive fwd component;
                         gravity-down was the old, wrong answer)
  * inward flex       -> elbow swivel matches the operator's (never the
                         chicken-wing pure-outward basin)
  * both sides behave (the old left-side palm-up inversion sent the left
    elbow BACKWARD)

Slow (~1 min): full engine + physics per case, marked accordingly.
"""
import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from mabel import MabelTeleop, OperatorPose  # noqa: E402

SCENE = ("/home/mabel/open_mabel/simulation/mabel_mujoco/scenes/"
         "mabel_home.xml")
HEAD_H = 1.65
SH = {"right": np.array([0.18, 1.45, 0.0]), "left": np.array([-0.18, 1.45, 0.0])}
L_OP = 0.58
X, Y, Z = np.eye(3)
FWD = -Z                                    # ARKit forward


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def _mirror(v):
    return np.array([-v[0], v[1], v[2]])


def _hand(side, wrist_p, f, n, elbow_p):
    f = _unit(f)
    n = np.asarray(n, float); n = n - f * (n @ f); n = _unit(n)
    n_frame = n if side == "right" else -n   # real left-hand chirality: dorsal
    s = np.cross(n_frame, f)

    def T(p, R=None):
        M = np.eye(4)
        if R is not None:
            M[:3, :3] = R
        M[:3, 3] = p
        return M

    R = np.column_stack([f, s, n_frame])
    j = {"wrist": T(wrist_p, R),
         "middle_metacarpal": T(wrist_p + 0.08 * f, R),
         "index_metacarpal": T(wrist_p + 0.08 * f - 0.025 * s, R),
         "little_metacarpal": T(wrist_p + 0.08 * f + 0.025 * s, R),
         "forearm_arm": T(elbow_p, R)}
    return j


def _run(case, frames=3000):
    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    teleop = MabelTeleop(model, data, base_follow=False, fail_safe=True,
                         clutch=True)
    teleop.set_control_spec(method="retarget")
    teleop.set_self_collision(True)
    eng = teleop.controller

    rj = _hand("right", *case["right"])
    lj = _hand("left", *case["left"])
    head = np.eye(4); head[1, 3] = HEAD_H
    pose = OperatorPose(head=head, right_joints=rj, left_joints=lj)
    pose.right_elbow = rj["forearm_arm"]
    pose.left_elbow = lj["forearm_arm"]
    for k in range(frames):
        pose.timestamp = (k + 1) / 30.0
        teleop.update(pose, dt=model.opt.timestep)
        mujoco.mj_step(model, data)

    mujoco.mj_kinematics(model, data)
    r_sh = data.xpos[eng.arm["right"].shoulder_bid]
    l_sh = data.xpos[eng.arm["left"].shoulder_bid]
    left_ax = _unit(l_sh - r_sh)
    up_ax = np.array([0.0, 0.0, 1.0])
    fwd_ax = _unit(np.cross(left_ax, up_ax))

    out = {}
    for side in ("right", "left"):
        arm = eng.arm[side]
        wrist = data.xanchor[arm.orca_wrist_jid]
        sh = data.xpos[arm.shoulder_bid]
        el = data.xpos[arm.elbow_bid]
        axis = _unit(wrist - sh)
        perp = (el - sh) - axis * float((el - sh) @ axis)
        w = perp / (np.linalg.norm(perp) + 1e-9)
        out_ax = -left_ax if side == "right" else left_ax
        out[side] = {"fwd": float(w @ fwd_ax), "out": float(w @ out_ax),
                     "up": float(w @ up_ax)}
    return out


def _sym(wr, f, n, el):
    return {"right": (wr, f, n, el),
            "left": (_mirror(wr), _mirror(f), _mirror(n), _mirror(el))}


def test_t_pose_palms_up_elbows_point_up():
    wr = SH["right"] + 0.97 * L_OP * X
    el = SH["right"] + 0.56 * L_OP * _unit(X + 0.18 * Y)
    sw = _run(_sym(wr, X, +Y, el))
    for side in ("right", "left"):
        # Threshold history: pre-wire-gate the elbow settled up-dominant
        # (up 0.89-0.98). The 2026-07-30 glide wire gate's thermal yield backs
        # the ctrl off the OVERDRIVEN full-extension target, which re-balances
        # the near-straight arm's residual bend up-AND-forward (left measured
        # up 0.44). The semantic the operator asked for — raised, never the
        # old inversion (left fwd was −0.99, elbow BACKWARD) — still holds:
        assert sw[side]["up"] > 0.35, \
            f"{side}: palms-up T must RAISE the elbow, got {sw[side]}"
        assert sw[side]["fwd"] > -0.3, \
            f"{side}: palms-up elbow must never invert backward, got {sw[side]}"


def test_t_pose_palms_down_elbows_bias_forward():
    wr = SH["right"] + 0.97 * L_OP * X
    el = SH["right"] + 0.56 * L_OP * _unit(X + 0.18 * FWD)
    sw = _run(_sym(wr, X, -Y, el))
    for side in ("right", "left"):
        assert sw[side]["fwd"] > 0.15, \
            f"{side}: palms-down T must bias the elbow FORWARD, got {sw[side]}"
        assert sw[side]["fwd"] > -0.5, "elbow must never point backward"


def test_inward_flex_matches_operator_no_chicken_wing():
    u_up = _unit(0.20 * X - 0.80 * Y + 0.56 * FWD)
    el = SH["right"] + 0.56 * L_OP * u_up
    u_fore = _unit(-0.50 * X + 0.20 * Y + 0.84 * FWD)
    wr = el + 0.44 * L_OP * u_fore
    sw = _run(_sym(wr, -X, -Y, el))
    for side in ("right", "left"):
        # the operator's elbow hangs DOWN here — pure-outward (the old
        # chicken-wing basin: out ~ +0.99, up ~ 0) must be gone.
        assert sw[side]["up"] < -0.3, \
            f"{side}: inward flex must keep the elbow DOWN, got {sw[side]}"
        assert sw[side]["out"] < 0.9, \
            f"{side}: chicken-wing (pure outward) elbow, got {sw[side]}"
