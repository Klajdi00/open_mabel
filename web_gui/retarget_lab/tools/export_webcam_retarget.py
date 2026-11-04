#!/usr/bin/env python3
"""Export the data NOVA's deployed arm retargeter runs on, for the website's
webcam demo.

The public website (website/, double-blind: it never names NOVA) maps the
operator onto the robot in website/assets/deployed-retarget.js, a port of the
retargeter NOVA runs on the robot:

    ../NOVA/controller/arm_retarget.py   NOVA_RETARGETER=lab (its default)
    ../NOVA/controller/lab_retarget.py   -> web_gui/retarget_lab/server/worker.py
                                            KinematicRetargeter("ours"),
                                            robot "nova", scope "local"
    -> controller/mabel/teleop_engine/    TeleopEngine: GMR arm map, elbow
                                            standoff, posture atlas, GN arm IK

The browser cannot load the MJCF or the engine, so everything the port needs
that is MEASURED rather than written down — the kinematic tree of the torso and
both arms at the home keyframe, the robot palm frames, the directional reach
samples, the front-workspace posture tables, and the engine's hyperparameters
as the lab's "ours" variant configures them — is read from the live engine
here and written to

    website/assets/data/deployed-retarget.json

    python3 web_gui/retarget_lab/tools/export_webcam_retarget.py           # rewrite it
    python3 web_gui/retarget_lab/tools/export_webcam_retarget.py --check   # fail if it drifted

Needs MuJoCo; reads NOVA's neck constants from the sibling checkout when present.
Validated against NOVA's own runtime by tools/webcam_parity.mjs. Like the
store's catalog generator, it lives in this (private) repository and only its
output is published; website/scripts/run_all.sh runs --check when it is here.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))     # MABEL/
SITE = os.path.join(REPO, "website")
DST = os.path.join(SITE, "assets", "data", "deployed-retarget.json")

os.environ.setdefault("MABEL_RTL_ROBOT", "nova")
os.environ.setdefault("MABEL_LOG_LEVEL", "WARNING")
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
for p in (os.path.join(REPO, "controller"), os.path.join(REPO, "web_gui")):
    if p not in sys.path:
        sys.path.insert(0, p)


def r(x, nd=12):
    """Round for a stable, diffable file (12 significant digits is far below
    anything the solve can resolve)."""
    if isinstance(x, (list, tuple)):
        return [r(v, nd) for v in x]
    return float(f"{float(x):.{nd}g}")


def build():
    import numpy as np
    import mujoco
    from retarget_lab.server.worker import KinematicRetargeter
    from retarget_lab.server import config as RCFG
    import mabel.teleop_engine.hyperparams as HP
    import mabel.teleop_engine.arm as AM
    import mabel.teleop_engine.wholebody as WB
    import mabel.teleop_engine.engine as EN
    import mabel.whole_body_retargeting.heading as HD
    import mabel.whole_body_retargeting.workspace_assist as WA
    from mabel.teleop_engine.geometry import ShoulderEstimator

    # NOVA's deployment: lab_retarget.LabRetargeter builds exactly this
    # (KinematicRetargeter(VARIANT="ours") + apply_settings robot nova / local).
    kr = KinematicRetargeter("ours")
    kr.apply_settings({"robot": "nova", "scope": "local"})
    m, d, e = kr.model, kr.data, kr.eng
    B = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
    J = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
    A = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)

    # the lift as NOVA holds it: the home height mirrored into both stages
    for q in kr._lift_q:
        d.qpos[q] = kr.lift_home / len(kr._lift_q)
    mujoco.mj_kinematics(m, d)

    root = B("body_mount")                    # parent of the Torso body: fixed in local scope
    keep = ["torso", "r_arm_mount", "l_arm_mount"]
    for s in ("r", "l"):
        keep += [f"{s}_shoulder_1", f"{s}_shoulder_2", f"{s}_shoulder_3", f"{s}_elbow",
                 f"{s}_forearm", f"{s}_wrist", f"{s}_hand_mount"]
    keep += ["right_tower", "right_palm", "left_tower", "left_palm",
             "head_mount", "neck_1", "neck_2", "head"]     # the head capsule rides the neck
    ids = [B(n) for n in keep]
    assert min(ids) >= 0, keep
    index = {root: -1}
    bodies = []
    for n, b in zip(keep, ids):
        par = int(m.body_parentid[b])
        assert par in index, (n, "parent not exported yet")
        index[b] = len(bodies)
        jn = int(m.body_jntnum[b])
        assert jn <= 1, (n, "more than one joint")
        joint = None
        if jn:
            j = int(m.body_jntadr[b])
            assert m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE, n
            qa = int(m.jnt_qposadr[j])
            joint = {"name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j),
                     "axis": r(m.jnt_axis[j].tolist()), "pos": r(m.jnt_pos[j].tolist()),
                     "qpos0": r(m.qpos0[qa]), "range": r(m.jnt_range[j].tolist()),
                     "limited": bool(m.jnt_limited[j])}
        bodies.append({"name": n, "parent": index[par], "pos": r(m.body_pos[b].tolist()),
                       "quat": r(m.body_quat[b].tolist()), "joint": joint})

    def pose(b):
        return {"pos": r(d.xpos[b].tolist()), "R": r(d.xmat[b].reshape(3, 3).tolist())}

    joints = ["Torso"] + [f"{S} Arm {i}" for S in ("Right", "Left") for i in range(1, 8)] \
        + ["Right Wrist", "Left Wrist", "Neck Yaw", "Neck Pitch", "Neck Roll"]
    home = {n: r(d.qpos[m.jnt_qposadr[J(n)]]) for n in joints}

    # forward-kinematics fixtures: the port must reproduce MuJoCo to round-off
    rng = np.random.default_rng(7)
    fk = []
    probe = ["r_arm_mount", "l_arm_mount", "r_elbow", "l_elbow", "right_palm", "left_palm", "head"]
    for _ in range(4):
        q = {}
        for n in joints:
            lo, hi = m.jnt_range[J(n)]
            q[n] = float(rng.uniform(lo, hi))
            d.qpos[m.jnt_qposadr[J(n)]] = q[n]
        mujoco.mj_kinematics(m, d)
        fk.append({"q": {k: r(v) for k, v in q.items()},
                   "xpos": {p: r(d.xpos[B(p)].tolist()) for p in probe},
                   "palmR": {p: r(d.xmat[B(p)].reshape(3, 3).tolist()) for p in ("right_palm", "left_palm")},
                   "wrist_anchor": {S: r(d.xanchor[J(f"{S} Wrist")].tolist()) for S in ("Right", "Left")}})
    for n in joints:
        d.qpos[m.jnt_qposadr[J(n)]] = home[n]
    mujoco.mj_kinematics(m, d)

    sides = {}
    for side in ("right", "left"):
        a = e.arm[side]
        tbl = a.table
        rd = e.arm_reach_dir[side] if e.arm_reach_dir is not None else None
        out = {"R_rob": r(np.asarray(a.R_rob).tolist()), "lo": r(a.lo.tolist()), "hi": r(a.hi.tolist()),
               "q_nom": r(a.q_nom.tolist()), "reg_w": r(a.reg_w.tolist()),
               "w_orient_max": a.w_orient_max, "w_orient_min": a.w_orient_min,
               "w_orient_fwd": a.w_orient_fwd, "w_elbow": a.w_elbow,
               "trust_rate_distal": a.trust_rate_distal,
               "reach": r(e.arm_reach[side]),
               "reach_dir": None if rd is None else {"dirs": r(rd.dirs.tolist()),
                                                    "reach": r(rd.reach.tolist()),
                                                    "scalar": r(rd.scalar)},
               "table": None}
        if tbl is not None:
            g = tbl.grid
            out["table"] = {"x": r(g.x.tolist()), "y": r(g.y.tolist()), "z": r(g.z.tolist()),
                            "rolls": r(g.rolls.tolist()), "shape": list(tbl.q.shape),
                            "seam_tol": tbl.SEAM_TOL,
                            # 8 significant digits: 1e-8 rad, under the solver's own tolerance
                            "q": [float(f"{v:.8g}") for v in tbl.q.ravel().tolist()],
                            "feas": [float(f"{v:.8g}") for v in tbl._feas_field().ravel().tolist()]}
        sides[side] = out

    hp_names = [
        "AVP_TO_ROBOT", "R_BASE_ALIGN", "HEAD_JUMP_M", "HEAD_ANCHOR_HOLD",
        "ARM_EXT_RHO_MAX", "ARM_GMR_EXT_GATE", "ARM_EXT_OVERDRIVE", "ARM_BIMANUAL_BLEND",
        "ARM_BIMANUAL_NEAR", "ARM_BIMANUAL_FULL", "ARM_HAND_CLEAR", "ARM_WRIST_FRONT",
        "ARM_BODY_FRAME_MAP", "ARM_TRACK_WRIST", "ARM_TRACK_ELBOW", "ARM_ELBOW_TRACK_EMA",
        "ARM_ELBOW_BEND_REF", "ARM_ELBOW_W_MIN", "ARM_SYNTH_ELBOW", "ARM_SWIVEL_PALM_GAIN",
        "SWIVEL_LAT_MIN", "ARM_W_WRIST", "ARM_W_SMOOTH", "ARM_ELBOW_SWIVEL_BOOST",
        "ARM_REG_ELBOW_RELAX", "ARM_TABLE_REG", "ARM_TABLE_MARGIN", "ARM_TABLE_REFINE",
        "ARM_GN_ITERS", "ARM_GN_DAMPING", "ARM_ESCAPE_ERR", "ARM_ESCAPE_FRAMES",
        "ARM_TRUST_STEP", "ARM_TRUST_DISTAL_FROM", "ARM_TRUST_MAX", "ARM_QP_VMAX",
        "ANTHRO_ARM_RATIO", "ANTHRO_BIACROMIAL", "ANTHRO_ACROMION_BELOW_EYE",
        "ANTHRO_EYE_HEIGHT_RATIO", "HEADING_SLACK", "HEADING_TAU", "HEADING_MODE",
        "TORSO_FOLLOW", "TORSO_FOLLOW_SPEED", "TORSO_LEAN_PIVOT", "TORSO_LEAN_PIVOT_MIN",
        "TORSO_LEAN_GAIN", "TORSO_LEAN_DEADBAND", "BASE_FOLLOW", "BASE_BOX_BACK", "BASE_BOX_FWD",
        "BASE_BOX_LAT", "TORSO_LOCAL_DROP_DEADBAND",
    ]
    hp = {}
    for n in hp_names:
        v = getattr(HP, n)
        hp[n] = r(np.asarray(v).tolist()) if isinstance(v, np.ndarray) else (
            v if isinstance(v, (bool, str)) else r(v))
    # the ones the variant / lab rebind at runtime are read where they are bound
    hp.update({
        "ARM_MAP": e.arm_map, "ARM_TRUST_RATE": AM.ARM_TRUST_RATE,
        "ARM_MIN_SHOULDER_RADIUS": EN.ARM_MIN_SHOULDER_RADIUS,
        "ARM_MIN_TORSO_RADIUS": EN.ARM_MIN_TORSO_RADIUS,
        "ARM_ELBOW_ABDUCT": EN.ARM_ELBOW_ABDUCT,
        "ARM_HEAD_ANCHOR_LP": __import__("mabel.teleop_engine.geometry", fromlist=["x"]).ARM_HEAD_ANCHOR_LP,
        "ARM_REACH_DIRECTIONAL": e.arm_reach_dir is not None,
        "ARM_WS_ASSIST": WA.ARM_WS_ASSIST, "ARM_WS_NEAR": WA.ARM_WS_NEAR, "ARM_WS_FAR": WA.ARM_WS_FAR,
        "ARM_WS_LATERAL": WA.ARM_WS_LATERAL, "ARM_WS_MAX_OFFSET": WA.ARM_WS_MAX_OFFSET,
        "ARM_WS_RATE": WA.ARM_WS_RATE, "ARM_WS_DEADBAND": WA.ARM_WS_DEADBAND,
        "TORSO_LEAN_MODE": WB.TORSO_LEAN_MODE, "BASE_TRANSLATION_MODE": WB.BASE_TRANSLATION_MODE,
        "HEADING_MODE": EN.HEADING_MODE,
        "HEADING_STILL_RATE": HD.STILL_RATE, "HEADING_RATE_TAU": HD.RATE_TAU,
        "SHOULDER_L_MIN": ShoulderEstimator._L_MIN, "SHOULDER_L_MAX": ShoulderEstimator._L_MAX,
        "SHOULDER_H_MIN": ShoulderEstimator._H_MIN, "SHOULDER_H_MAX": ShoulderEstimator._H_MAX,
        "SHOULDER_MED_N": ShoulderEstimator._MED_N,
        "SELF_COLLISION": bool(e.self_collision),
    })
    for k in ("ARM_MAP", "HEADING_MODE", "TORSO_LEAN_MODE", "BASE_TRANSLATION_MODE"):
        hp[k] = str(hp[k])
    assert hp["ARM_MAP"] == "gmr" and hp["HEADING_MODE"] == "head" and hp["TORSO_LEAN_MODE"] == "angle"
    assert not e.wbc.base_in_loop, "NOVA's default is upper-body scope (NOVA_BODY_FOLLOW=0)"

    # the self-collision projector (collision.py ArmCollisionProjector over the
    # capsule guard): capsules in their anchor body's frame, the monitored pairs,
    # the barrier constants, and the 16 joints it owns (both arms + ORCA wrists)
    P = e.projector
    g = P.guard
    BN = lambda b: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
    selfcol = None if P is None else {
        "caps": {k: {"body": BN(c.body), "p0": r(c.p0.tolist()), "p1": r(c.p1.tolist()),
                     "radius": r(c.radius)} for k, c in g.caps.items()},
        "pairs": [[a.name, b.name] for a, b in g.pairs],
        "pair_margin": {f"{a}|{b}": v for (a, b), v in g._pair_margin.items()},
        "d_inf": r(g.d_inf), "d_safe": r(g.d_safe), "alpha": r(g.alpha), "stop_decel": r(g.stop_decel),
        "lag_capsules": sorted(g.lag_capsules), "lag_pad": r(HP.SELFCOL_LAG_PAD),
        "joints": [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
                   for s in ("right", "left") for j in list(e.arm[s].jids) + [e.arm[s].orca_wrist_jid]],
        "lo": r(P.lo.tolist()), "hi": r(P.hi.tolist()), "vmax": r(HP.ARM_QP_VMAX)}

    tj = J("Torso")
    neck_acts = [A(n) for n in ("Neck Yaw", "Neck Pitch", "Neck Roll")]
    # The robot's own windows ON the engine (worker._robot_limits, robot "nova"): the neck
    # law clips to them, the torso law clips to them, and the servo clamps both joints there.
    servo_lim = {}
    for a, qa, lo, hi, on in kr._servo:
        jn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(m.actuator_trnid[a, 0]))
        if jn in ("Torso", "Neck Yaw", "Neck Pitch", "Neck Roll"):
            servo_lim[jn] = {"range": r([lo, hi]), "limited": bool(on)}

    return {
        "about": "generated from the MJCF and controller/mabel/teleop_engine at the "
                 "Retarget Lab's deploy preset (variant 'ours', upper-body scope) by "
                 "web_gui/retarget_lab/tools/export_webcam_retarget.py. Do not edit.",
        "scene": os.path.relpath(RCFG.SCENE, REPO),
        "root": {"name": "body_mount", **pose(root)},
        "chassis": pose(B("mabel_chassis")),
        "bodies": bodies,
        "home": home,
        "lift_home": r(kr.lift_home),
        "torso": {"qpos0": r(m.qpos0[m.jnt_qposadr[tj]]),
                  "range": r([float(e.wbc.torso.lo), float(e.wbc.torso.hi)])},
        "neck_range": [r(np.asarray(v, float).tolist()) for v in e.neck_rng],
        "servo_limits": servo_lim,
        # the ENGINE's own neck law (engine._retarget): it moves the head capsule
        "engine_neck": {"signs": [HP.NECK_YAW_SIGN, HP.NECK_PITCH_SIGN, HP.NECK_ROLL_SIGN],
                        "gain": HP.NECK_GAIN, "datum_live": bool(HP.NECK_DATUM_LIVE),
                        "torso_rest": r(e.torso_rest)},
        "selfcol": selfcol,
        "sides": sides,
        "hp": hp,
        "deploy": {"torso_cmd_range": list(RCFG.NOVA_TORSO_RANGE), "slews": RCFG.SLEWS["nova"]},
        "fk_fixtures": fk,
    }


def main():
    out = json.dumps(build(), separators=(",", ":"), sort_keys=True) + "\n"
    if "--check" in sys.argv:
        have = open(DST).read() if os.path.exists(DST) else ""
        if have != out:
            print("website/assets/data/deployed-retarget.json has DRIFTED from the engine/MJCF — "
                  "run web_gui/retarget_lab/tools/export_webcam_retarget.py")
            print("RESULT: FAIL")
            sys.exit(1)
        print("deployed-retarget.json matches the engine and the MJCF")
        print("RESULT: PASS")
        return
    os.makedirs(os.path.dirname(DST), exist_ok=True)
    with open(DST, "w") as f:
        f.write(out)
    print(f"wrote {os.path.relpath(DST, REPO)} ({len(out) / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
