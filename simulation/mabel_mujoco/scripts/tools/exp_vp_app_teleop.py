"""Vision Pro APP-path teleop validation — the wire→facade→engine pipeline.

Validates the 2026-07 rewiring of the app (WebSocket) teleop path onto the
documented scripts controller, with five experiments:

  E1  webcam-JS parity     the Python anatomical hand retargeter (hand_anatomical.py)
                           against the ACTUAL control-center JS (bodyteleop.decomposeHand
                           + handmodel ranges) executed in node, on random hand shapes.
  E2  wrist orientation    palm roll/pitch sweeps through the FULL app pipeline
                           (teleop_frame → wire → MabelTeleop facade → TeleopEngine →
                           MuJoCo); measures robot palm fwd/up axis error.
  E3  finger transfer      commanded anatomical MCP/PIP curls → recovered ORCA joint
                           targets (identity within the PIP+½DIP coupling).
  E4  base motion model    the engine's base chain (LiftAwareReverseGuard →
                           BaseMotionModel → swerve) against the keyboard teleop's
                           direct C.safe_base_step call — bit-exact parity forward,
                           and the designed reverse-guard divergence with the lift up.
  E5  body-yaw follow      elbow-line heading (derived from the app's forearm joints)
                           → chassis yaw follow; plus the head-box torso/base cascade.

Reproducible: fixed seed, headless matplotlib, vector PDFs into docs/figures/.
Run from anywhere:  python3 simulation/mabel_mujoco/scripts/tools/exp_vp_app_teleop.py
"""
import os
import sys
import json
import subprocess

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", "..", "..", ".."))
SCRIPTS = os.path.normpath(os.path.join(HERE, ".."))
FIG = os.path.normpath(os.path.join(HERE, "..", "..", "docs", "figures"))
JS_DIR = os.path.join(ROOT, "control_center", "web", "js")
sys.path.insert(0, SCRIPTS)
sys.path.insert(0, os.path.join(ROOT, "controller"))

import mujoco
from mabel import config as C
from mabel.facade import MabelTeleop
from mabel.types import OperatorPose
from teleop.hand_anatomical import retarget_hand, HAND_RANGES
from teleop.hyperparams import AVP_TO_ROBOT, BASE_BOX_FWD
from teleop.geometry import _natural_frame

RNG = np.random.default_rng(0)
os.makedirs(FIG, exist_ok=True)

SCENE = os.path.join(SCRIPTS, C.SCENE_XML)


# ── synthetic operator (app wire schema) ─────────────────────────────────────
# Canonical RIGHT hand in palm coords [x̂p ŷp ẑp]: ŷp = wrist→middle-knuckle,
# ẑp = dorsal, x̂p = ŷp×ẑp (little-finger side). Thumb sits at −x̂p.
_FINGERS = ("index", "middle", "ring", "little")
_KN_X = {"index": -0.030, "middle": -0.010, "ring": 0.010, "little": 0.030}
_SEG_L = (0.042, 0.028, 0.022)          # proximal / middle / distal


def _rot(a, th):
    a = np.asarray(a, float) / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def make_hand_points(side, curls=None, thumb=(0.15, 0.1, 0.1)):
    """21 MediaPipe-layout points in PALM coords (wrist at origin). curls =
    {finger: (mcp, pip, dip)} [rad]; thumb = (opposition, mp, ip) [rad]."""
    curls = curls or {}
    mir = -1.0 if side == "left" else 1.0
    xhat = np.array([mir, 0.0, 0.0])
    yhat = np.array([0.0, 1.0, 0.0])
    zhat = np.array([0.0, 0.0, 1.0])      # dorsal
    pts = [np.zeros(3)]                   # 0: wrist
    # thumb: metacarpal from a CMC base near the wrist, diving with opposition
    opp, t_mp, t_ip = thumb
    cmc = -0.025 * xhat + 0.020 * yhat - 0.005 * zhat
    tdir = (-0.80 * xhat + 0.60 * yhat)
    tdir /= np.linalg.norm(tdir)
    tdir = _rot(np.cross(tdir, -zhat), 0.0) @ tdir            # in-plane baseline
    tdir = _rot(np.cross(zhat, tdir), 0.0) @ tdir
    tdir = _rot(np.cross(tdir, zhat) / np.linalg.norm(np.cross(tdir, zhat)), 0.0) @ tdir
    tdir = _rot(np.cross(zhat, tdir), 0.0) @ tdir
    # opposition = rotate the metacarpal below the palm plane (toward −ẑ)
    tax = np.cross(zhat, tdir); tax /= np.linalg.norm(tax)
    tdir_o = _rot(tax, -opp) @ tdir if opp else tdir
    kn_t = cmc + 0.045 * tdir_o
    a_t = np.cross(tdir_o, -zhat); a_t /= np.linalg.norm(a_t)
    d_mp = _rot(a_t, t_mp) @ tdir_o
    ip_t = kn_t + 0.035 * d_mp
    d_ip = _rot(a_t, t_ip) @ d_mp
    tip_t = ip_t + 0.030 * d_ip
    pts += [cmc, kn_t, ip_t, tip_t]                            # 1-4 thumb chain
    for f in _FINGERS:
        kn = _KN_X[f] * mir * xhat + 0.095 * yhat
        c1, c2, c3 = curls.get(f, (0.0, 0.0, 0.0))
        a = -xhat * mir                                        # flexion axis (ŷ→−ẑ)
        d1 = _rot(a, c1) @ yhat
        d2 = _rot(a, c1 + c2) @ yhat
        d3 = _rot(a, c1 + c2 + c3) @ yhat
        p1 = kn + _SEG_L[0] * d1
        p2 = p1 + _SEG_L[1] * d2
        p3 = p2 + _SEG_L[2] * d3
        pts += [kn, p1, p2, p3]                                # kn / pip / dip / tip
    return np.asarray(pts)


_N21 = [
    "wrist",
    "thumb_knuckle", "thumb_intermediate_base", "thumb_intermediate_tip", "thumb_tip",
    "index_knuckle", "index_intermediate_base", "index_intermediate_tip", "index_tip",
    "middle_knuckle", "middle_intermediate_base", "middle_intermediate_tip", "middle_tip",
    "ring_knuckle", "ring_intermediate_base", "ring_intermediate_tip", "ring_tip",
    "little_knuckle", "little_intermediate_base", "little_intermediate_tip", "little_tip",
]


def _mat16(m):
    return {"matrix": np.asarray(m, float).flatten().tolist()}


def _T(p, R=None):
    m = np.eye(4)
    if R is not None:
        m[:3, :3] = R
    m[:3, 3] = np.asarray(p, float)
    return m


def hand_payload(side, wrist_p, R_hand, curls=None, thumb=(0.15, 0.1, 0.1),
                 elbow_off=(0.0, -0.15, 0.30)):
    """App-schema hand dict: palm-coord skeleton placed at wrist_p with palm
    orientation R_hand (palm coords → ARKit world), plus metacarpals + forearm."""
    pts = make_hand_points(side, curls, thumb)
    W = np.asarray(wrist_p, float)
    world = [W + R_hand @ p for p in pts]
    anchor = _T(W, R_hand)
    Ai = np.linalg.inv(anchor)
    joints = [{"joint": n, "localTransform": _mat16(Ai @ _T(pw)), "isTracked": True}
              for n, pw in zip(_N21, world)]
    mir = -1.0 if side == "left" else 1.0
    for f in _FINGERS:                                        # metacarpals (palm frame)
        pm = W + R_hand @ np.array([_KN_X[f] * mir * 0.7, 0.045, 0.0])
        joints.append({"joint": f"{f}_metacarpal", "localTransform": _mat16(Ai @ _T(pm)),
                       "isTracked": True})
    pe = W + R_hand @ np.asarray(elbow_off, float)            # forearm (elbow end)
    joints.append({"joint": "forearm_arm", "localTransform": _mat16(Ai @ _T(pe)),
                   "isTracked": True})
    return {"chirality": side, "isTracked": True,
            "anchorTransform": _mat16(anchor), "joints": joints}


# Neutral palm orientation in ARKit world (operator faces −Z): fingers forward
# (ŷp→−Z), dorsal up (ẑp→+Y), thumb side inward.
def palm_R(side, roll=0.0, pitch=0.0, yaw=0.0):
    base = np.column_stack([np.array([1.0, 0, 0]) * (1 if side == "right" else 1),
                            np.array([0.0, 0, -1.0]),
                            np.array([0.0, 1.0, 0])])   # x̂p→+X, ŷp→−Z, ẑp→+Y
    R = _rot([0, 1, 0], yaw) @ _rot([1, 0, 0], pitch) @ _rot([0, 0, 1], roll)
    return R @ base


def frame(t, right=None, left=None, head_p=(0.0, 1.6, 0.0), joystick=None):
    f = {"sequence": int(t * 1000), "timestamp": float(t), "mode": "Arms & Hands",
         "navJoystick": joystick or {"lx": 0.0, "ly": 0.0, "rx": 0.0, "ry": 0.0},
         "head": {"transform": _mat16(_T(head_p)), "trackingState": "normal"}}
    if right is not None:
        f["rightHand"] = right
    if left is not None:
        f["leftHand"] = left
    return f


def fresh_robot():
    model = mujoco.MjModel.from_xml_path(SCENE)
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_SENSOR
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    teleop = MabelTeleop(model, data)
    teleop.set_control_spec(method="retarget", type="impedance", region="wholeBody")
    return model, data, teleop


def run(teleop, model, data, frames, hz=30.0, substeps=None):
    """Feed app frames at hz through the facade, stepping physics in between."""
    dt = model.opt.timestep
    substeps = substeps or max(1, int(round(1.0 / hz / dt)))
    cmd = None
    for fp in frames:
        pose = OperatorPose.from_teleop_frame(fp)
        for _ in range(substeps):
            cmd = teleop.update(pose, None, dt)
            mujoco.mj_step(model, data)
    return cmd


# ── E1: parity with the ACTUAL control-center JS ─────────────────────────────
_NODE_SCRIPT = r"""
import { decomposeHand } from '%(js)s/bodyteleop.js';
import { radToPct, pctToRad } from '%(js)s/handmodel.js';
import fs from 'fs';
const cases = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const out = [];
for (const c of cases) {
  const dec = decomposeHand(c.shape, c.chirality);
  if (!dec) { out.push(null); continue; }
  const pose = {};
  const fk = { index: 'index', middle: 'middle', ring: 'ring', little: 'pinky' };
  for (const [f, k] of Object.entries(fk)) {
    const A = dec.fingers[f];
    pose[`${k}_mcp`] = pctToRad(`${k}_mcp`, radToPct(`${k}_mcp`, A.mcp));
    pose[`${k}_pip`] = pctToRad(`${k}_pip`, radToPct(`${k}_pip`, A.pip + 0.5 * Math.max(0, A.dip)));
    pose[`${k}_abd`] = pctToRad(`${k}_abd`, 50);
  }
  const T = dec.thumb;
  const opp = Math.asin(Math.max(-1, Math.min(1, -T.dir[2])));
  pose.thumb_mcp = pctToRad('thumb_mcp', radToPct('thumb_mcp', opp));
  pose.thumb_dip = pctToRad('thumb_dip', radToPct('thumb_dip', T.mp));
  pose.thumb_pip = pctToRad('thumb_pip', radToPct('thumb_pip', T.ip));
  pose.thumb_abd = pctToRad('thumb_abd', 50);
  out.push(pose);
}
fs.writeFileSync(process.argv[3], JSON.stringify(out));
"""


def e1_js_parity(n=200):
    print("\nE1  webcam-JS parity (node) ".ljust(60, "─"))
    cases, py = [], []
    for _ in range(n):
        side = "right" if RNG.random() < 0.5 else "left"
        curls = {f: tuple(RNG.uniform([0, 0, 0], [1.4, 1.6, 1.0])) for f in _FINGERS}
        thumb = tuple(RNG.uniform([-0.2, -0.2, -0.2], [0.9, 1.1, 1.2]))
        pts = make_hand_points(side, curls, thumb)
        R = palm_R(side, *RNG.uniform(-0.8, 0.8, 3))
        world = (R @ pts.T).T + RNG.uniform(-0.3, 0.3, 3)
        shape = world - world[0]
        # χ convention: the JS labels handedness on the MediaPipe (selfie) convention,
        # which is MIRRORED vs true ARKit chirality → pass the OPPOSITE label to JS.
        cases.append({"shape": shape.tolist(),
                      "chirality": "left" if side == "right" else "right"})
        py.append(retarget_hand(world, side))
    scratch = os.environ.get("TMPDIR", "/tmp")
    cs, rs = os.path.join(scratch, "e1_cases.json"), os.path.join(scratch, "e1_js.json")
    js = os.path.join(scratch, "e1_runner.mjs")
    with open(cs, "w") as f:
        json.dump(cases, f)
    with open(js, "w") as f:
        f.write(_NODE_SCRIPT % {"js": JS_DIR})
    subprocess.run(["node", js, cs, rs], check=True)
    with open(rs) as f:
        jsr = json.load(f)
    worst, n_deg = 0.0, 0
    for p, j in zip(py, jsr):
        if p is None or j is None:
            n_deg += int((p is None) != (j is None))
            continue
        for k, v in j.items():
            worst = max(worst, abs(p[k] - v))
    print(f"  {n} random hands (both sides, random pose/curl/orientation)")
    print(f"  max |Δjoint| Python port vs control-center JS : {worst:.3e} rad")
    print(f"  degenerate-palm disagreements                 : {n_deg}")
    assert worst < 1e-9 and n_deg == 0, "JS parity FAILED"
    print("  PASS — the app path runs the webcam GUI's exact hand mapping")
    return worst


# ── E2: wrist ORIENTATION tracking through the full pipeline ────────────────
def e2_wrist_orientation():
    print("\nE2  wrist orientation tracking (full app pipeline) ".ljust(60, "─"))
    results = {}
    for axis, lim in (("roll", np.deg2rad(60)), ("pitch", np.deg2rad(45))):
        model, data, teleop = fresh_robot()
        eng = teleop.controller
        palm_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_palm")
        sweep = np.linspace(-lim, lim, 9)
        errs_f, errs_u, wq = [], [], []
        t = 0.0
        # settle at neutral first (anchors world_ref + fail-safe engage)
        neutral = hand_payload("right", [0.25, 1.30, -0.40], palm_R("right"))
        run(teleop, model, data, [frame(t + i / 30.0, right=neutral) for i in range(45)])
        t += 2.0
        for ang in sweep:
            kw = {axis: float(ang)}
            R_hand = palm_R("right", **kw)
            hp = hand_payload("right", [0.25, 1.30, -0.40], R_hand, curls=None)
            run(teleop, model, data,
                [frame(t + i / 30.0, right=hp) for i in range(60)])   # 2 s hold
            t += 3.0
            # commanded palm axes (robot world) — same construction as the engine
            pts = make_hand_points("right")
            world = [np.asarray([0.25, 1.30, -0.40]) + R_hand @ p for p in pts]
            R_op = _natural_frame(world[0],
                                  0.5 * (world[9] + world[0]),  # ~middle metacarpal dir
                                  world[5], world[17])
            Rc = AVP_TO_ROBOT @ R_op
            Rp = data.xmat[palm_bid].reshape(3, 3) @ eng.opt["right"].R_rob
            ef = np.degrees(np.arccos(np.clip(np.dot(Rp[:, 0], Rc[:, 0]), -1, 1)))
            eu = np.degrees(np.arccos(np.clip(np.dot(Rp[:, 2], Rc[:, 2]), -1, 1)))
            errs_f.append(ef); errs_u.append(eu)
            wq.append(float(data.ctrl[eng.wrist_act["right"]]))
        results[axis] = (np.degrees(sweep), np.array(errs_f), np.array(errs_u), np.array(wq))
        print(f"  {axis:5s} sweep ±{np.degrees(lim):.0f}°: palm-fwd err "
              f"mean {np.mean(errs_f):5.1f}° max {np.max(errs_f):5.1f}° | "
              f"palm-up err mean {np.mean(errs_u):5.1f}° max {np.max(errs_u):5.1f}° | "
              f"ORCA wrist span {np.ptp(wq):.2f} rad")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, (axis, (deg, ef, eu, wq)) in zip(axes, results.items()):
        ax.plot(deg, ef, "o-", label="palm fwd axis err")
        ax.plot(deg, eu, "s-", label="palm up axis err")
        ax.set_xlabel(f"operator palm {axis} [deg]")
        ax.set_ylabel("robot palm axis error [deg]")
        ax.set_title(f"wrist orientation tracking — {axis}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "vp_wrist_orientation.pdf"))
    plt.close(fig)
    return results


# ── E3: finger curl transfer (commanded anatomical → ORCA targets) ──────────
def e3_finger_transfer():
    print("\nE3  finger curl transfer ".ljust(60, "─"))
    curls = np.linspace(0.0, 1.3, 25)
    mcp_out, pip_out = [], []
    for c in curls:
        pts = make_hand_points("right", {f: (c, 0.7 * c, 0.4 * c) for f in _FINGERS})
        R = palm_R("right")
        world = (R @ pts.T).T + np.array([0.25, 1.3, -0.4])
        pose = retarget_hand(world, "right")
        mcp_out.append(pose["index_mcp"])
        pip_out.append(pose["index_pip"])
    mcp_out, pip_out = np.array(mcp_out), np.array(pip_out)
    exp_pip = np.clip(0.7 * curls + 0.5 * np.clip(0.4 * curls, 0, None),
                      *HAND_RANGES["index_pip"])
    err_m = np.max(np.abs(mcp_out - np.clip(curls, *HAND_RANGES["index_mcp"])))
    err_p = np.max(np.abs(pip_out - exp_pip))
    mono = np.all(np.diff(mcp_out) >= -1e-9) and np.all(np.diff(pip_out) >= -1e-9)
    print(f"  MCP identity err max {err_m:.4f} rad | PIP(+½DIP) err max {err_p:.4f} rad"
          f" | monotonic: {mono}")
    assert err_m < 0.02 and err_p < 0.02 and mono, "finger transfer FAILED"
    print("  PASS — commanded curls transfer 1:1 through the anatomical decomposition")
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot(curls, mcp_out, "o-", label="ORCA index MCP")
    ax.plot(curls, pip_out, "s-", label="ORCA index PIP (couples +½DIP)")
    ax.plot(curls, curls, "k--", lw=0.8, label="identity")
    ax.set_xlabel("commanded anatomical MCP curl [rad]")
    ax.set_ylabel("retargeted ORCA joint target [rad]")
    ax.set_title("webcam-parity finger transfer")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "vp_finger_transfer.pdf"))
    plt.close(fig)


# ── E4: base motion model = keyboard teleop's safe_base_step ─────────────────
def e4_motion_model():
    print("\nE4  base motion model parity (keyboard teleop chain) ".ljust(60, "─"))
    model, data, teleop = fresh_robot()
    eng = teleop.controller
    dt = model.opt.timestep
    cases = {
        "forward, lift down": ((-0.6, 0.0, 0.0), 0.0),
        "forward, lift up":   ((-0.6, 0.0, 0.0), 0.6),
        "reverse, lift down": ((+0.6, 0.0, 0.0), 0.0),
        "reverse, lift up":   ((+0.6, 0.0, 0.0), 0.6),
    }
    T = 2.0
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    parity_max = 0.0
    for name, (tgt, lift) in cases.items():
        # engine chain (exactly what the app path runs per substep)
        data.ctrl[eng.lift_act] = lift
        eng.wbc.base_cmd = (0.0, 0.0, 0.0)
        eng.wbc.base_tgt = tgt
        v_eng = []
        for _ in range(int(T / dt)):
            eng.wbc.base_step(data, dt)
            v_eng.append(eng.wbc.base_cmd[0])
        # keyboard-teleop reference: LiftAwareReverseGuard + C.safe_base_step
        v = (0.0, 0.0, 0.0)
        v_ref = []
        for _ in range(int(T / dt)):
            tx = tgt[0]
            if tx > 0.0:                                    # the engine's reverse guard
                tx *= float(np.clip(1.0 - lift / 0.635, 0.0, 1.0))
            v = C.safe_base_step(*v, tx, tgt[1], tgt[2], dt, lift)
            v_ref.append(v[0])
        v_eng, v_ref = np.array(v_eng), np.array(v_ref)
        parity_max = max(parity_max, float(np.max(np.abs(v_eng - v_ref))))
        tgrid = np.arange(len(v_eng)) * dt
        ax.plot(tgrid, v_eng, label=f"{name} (engine)")
        ax.plot(tgrid, v_ref, "k--", lw=0.7)
        print(f"  {name:20s} steady vx {v_eng[-1]:+.3f} m/s | engine-vs-reference "
              f"max Δ {np.max(np.abs(v_eng - v_ref)):.2e}")
    assert parity_max < 1e-12, "motion-model parity FAILED"
    print("  PASS — app base path is bit-exact with the keyboard teleop motion model")
    ax.set_xlabel("time [s]"); ax.set_ylabel("realized vx [m/s]")
    ax.set_title("app base path vs keyboard-teleop safe_base_step (dashed)")
    ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "vp_base_motionmodel.pdf"))
    plt.close(fig)


# ── E5: body-yaw follow + head-box cascade ───────────────────────────────────
def e5_body_yaw():
    print("\nE5  body-yaw follow (elbow line from app forearm joints) ".ljust(60, "─"))
    model, data, teleop = fresh_robot()
    base_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mabel_chassis")

    def hands(yaw):
        """Both hands + elbows rigidly rotated about the head by operator yaw."""
        Ry = _rot([0, 1, 0], yaw)
        out = {}
        for side, x in (("right", 0.25), ("left", -0.25)):
            w = Ry @ np.array([x, 1.30, -0.40])
            out[side] = hand_payload(side, w, Ry @ palm_R(side),
                                     elbow_off=(0.0, -0.15, 0.30))
        return out

    def yaw_of(R):
        return float(np.arctan2(R[1, 0], R[0, 0]))

    t = 0.0
    h = hands(0.0)
    run(teleop, model, data, [frame(t + i / 30.0, right=h["right"], left=h["left"])
                              for i in range(60)])
    t += 3.0
    yaw0 = yaw_of(data.xmat[base_bid].reshape(3, 3))
    tgt = np.deg2rad(30.0)
    traj = []
    h = hands(tgt)
    for i in range(150):                                     # 5 s follow
        run(teleop, model, data, [frame(t, right=h["right"], left=h["left"])])
        t += 1 / 30.0
        traj.append(np.degrees(yaw_of(data.xmat[base_bid].reshape(3, 3)) - yaw0))
    traj = np.array(traj)
    print(f"  operator torso yaw +30° → chassis yaw {traj[-1]:+.1f}° "
          f"(err {abs(traj[-1] - 30):.1f}°)")
    assert abs(traj[-1] - 30.0) < 6.0, "body-yaw follow FAILED"
    print("  PASS — base yaw follows the operator's torso heading")
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.plot(np.arange(len(traj)) / 30.0, traj, label="chassis yaw")
    ax.axhline(30, color="k", ls="--", lw=0.8, label="operator torso yaw")
    ax.set_xlabel("time [s]"); ax.set_ylabel("yaw [deg]")
    ax.set_title("base yaw-follow from the app's elbow line")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "vp_body_yaw.pdf"))
    plt.close(fig)

    # head-box cascade: within-box lean = torso only; overflow drives the base.
    # The head must move like a REAL head (ramped over ~1 s) — an instantaneous
    # teleport is (correctly) classified as an ARKit origin jump and re-anchored.
    print("\nE5b head-box cascade ".ljust(60, "─"))
    model, data, teleop = fresh_robot()
    h = hands(0.0)
    t = 0.0
    run(teleop, model, data, [frame(t + i / 30.0, right=h["right"], left=h["left"])
                              for i in range(45)])
    t += 2.0
    p0 = data.xpos[base_bid].copy()
    for dz, label, expect_base in ((-0.6 * BASE_BOX_FWD, "within-box lean", False),
                                   (-(BASE_BOX_FWD + 0.25), "overflow lean", True)):
        model2, data2, teleop2 = fresh_robot()
        run(teleop2, model2, data2, [frame(i / 30.0, right=h["right"], left=h["left"])
                                     for i in range(45)])
        b0 = data2.xpos[mujoco.mj_name2id(model2, mujoco.mjtObj.mjOBJ_BODY,
                                          "mabel_chassis")].copy()
        run(teleop2, model2, data2,
            [frame(2.0 + i / 30.0, right=h["right"], left=h["left"],
                   head_p=(0.0, 1.6, dz * min(1.0, (i / 30.0) / 1.0)))
             for i in range(120)])
        b1 = data2.xpos[mujoco.mj_name2id(model2, mujoco.mjtObj.mjOBJ_BODY,
                                          "mabel_chassis")].copy()
        moved = float(np.linalg.norm((b1 - b0)[:2]))
        ok = (moved > 0.05) == expect_base
        print(f"  {label:16s} head Δz={dz:+.2f} m → base moved {moved:.3f} m "
              f"({'base engaged' if moved > 0.05 else 'torso absorbed'}) "
              f"{'PASS' if ok else 'FAIL'}")
        assert ok, "head-box cascade FAILED"


# ── E6: ARKit origin-jump guard + head-relative height contract ─────────────
def e6_origin_jump():
    print("\nE6  origin-jump guard + head-relative height ".ljust(60, "─"))

    def hands_at(head_y, fwd=-0.40, dy=-0.40, x=0.25):
        return {s: hand_payload(s, [sx, head_y + dy, fwd], palm_R(s))
                for s, sx in (("right", x), ("left", -x))}

    # A: the ARKit floor-snap (head teleports ~1.6 m right after engage) must
    # re-anchor — NOT run the lift to the top or map the hands sky-high.
    model, data, teleop = fresh_robot()
    eng = teleop.controller
    head_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "head")
    palm_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_palm")
    h = hands_at(0.05)
    run(teleop, model, data, [frame(i / 30.0, right=h["right"], left=h["left"],
                                    head_p=(0.0, 0.05, 0.0)) for i in range(30)])
    h = hands_at(1.65)
    run(teleop, model, data, [frame(1.0 + i / 30.0, right=h["right"], left=h["left"],
                                    head_p=(0.0, 1.65, 0.0)) for i in range(120)])
    lift = float(data.ctrl[eng.lift_act])
    palm_z, head_z = float(data.xpos[palm_bid][2]), float(data.xpos[head_bid][2])
    print(f"  origin jump: lift {lift:.3f} m | palm z {palm_z:.2f} vs head z {head_z:.2f}")
    assert lift < 0.20 and palm_z < head_z - 0.15, "origin-jump guard FAILED"

    # B: hands raised to the operator's head level must land at the ROBOT's head
    # height (vertical maps 1:1; reach scales horizontal only).
    for reach in (1.2, 2.5):
        model, data, teleop = fresh_robot()
        eng = teleop.controller
        eng.apply_reach(reach)
        head_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "head")
        palm_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_palm")
        h = hands_at(1.60, fwd=-0.40)
        run(teleop, model, data, [frame(i / 30.0, right=h["right"], left=h["left"])
                                  for i in range(60)])
        h = hands_at(1.60, fwd=-0.35, dy=0.0, x=0.20)
        run(teleop, model, data, [frame(2.0 + i / 30.0, right=h["right"], left=h["left"])
                                  for i in range(90)])
        dz = abs(float(data.xpos[palm_bid][2]) - float(data.xpos[head_bid][2]))
        print(f"  hands-at-head (reach {reach}): palm-to-head Δz {dz*100:.0f} cm")
        assert dz < 0.25, "head-relative height contract FAILED"
    print("  PASS — origin jumps re-anchor; hands at head level = robot head height")


if __name__ == "__main__":
    print("Vision Pro APP-path teleop validation — wire → facade → TeleopEngine")
    print(f"scene: {SCENE}")
    e1_js_parity()
    e3_finger_transfer()
    e4_motion_model()
    e2_wrist_orientation()
    e5_body_yaw()
    e6_origin_jump()
    print("\nAll experiments passed. Figures → simulation/mabel_mujoco/docs/figures/"
          "vp_*.pdf")
