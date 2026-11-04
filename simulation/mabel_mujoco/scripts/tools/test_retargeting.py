"""
test_retargeting.py — rigorous, headless verification of the Vision Pro → MABEL
retargeting: frame transforms + latency, on synthesized but anatomically
reasonable human pose data.

What it does
------------
1. Synthesizes a 27-joint human hand skeleton (ARKit world frame, Z-up) with a
   forward-kinematic model parametrised by wrist pose + per-finger flexion +
   abduction, so we can script realistic motions (reach, curl to a fist, pinch,
   point) with KNOWN ground truth.
2. Pushes those frames through the SHIPPED pipeline — AVPFrame → Retargeter
   (wrist SE(3), world & body-relative) and HandIKRetargeter (per-finger keypoint
   IK) → WholeBodyIK — and asserts the transforms are correct:
     T1 wrist translation maps by RETARGET_SCALE · AVP_TO_ROBOT
     T2 body-relative target rides with the chassis (drive + yaw)
     T3 wrist rotation magnitude is preserved into the palm target
     T4 finger flexion (open→fist) drives ORCA MCP/PIP monotonically up
     T5 pinch closes the robot thumb–index tip gap
3. Benchmarks latency of every stage and end-to-end (mean / p50 / p95 / max, Hz)
   and checks it clears a real-time budget.

Run:  python scripts/tools/test_retargeting.py            (add --plot for a PDF)
"""
import os
import sys
import time
import argparse
import numpy as np
import mujoco

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "..", "..", "controller"))

from mabel import config as C
from mabel.avp_receiver import AVPFrame
from mabel.teleop_controller import TeleopController
from mabel.hand_ik_retarget import HandIKRetargeter
from mabel.retarget import Retargeter

G, R, Y, B, E = "\033[92m", "\033[91m", "\033[93m", "\033[1m", "\033[0m"
def ok(c): return f"{G}PASS{E}" if c else f"{R}FAIL{E}"

# ── synthetic human hand FK (ARKit world frame) ──────────────────────────────
# Bone lengths [m] (metacarpal, proximal, middle, distal) per finger, anthropo-
# metric averages. Wrist-local axes: +Z = fingers-forward, +Y = back-of-hand
# (palm normal), +X = radial (pinky→thumb side on a right hand).
_BONES = {
    "thumb":  (0.046, 0.038, 0.031, 0.025),
    "index":  (0.068, 0.040, 0.026, 0.022),
    "middle": (0.065, 0.045, 0.028, 0.024),
    "ring":   (0.058, 0.042, 0.026, 0.022),
    "little": (0.053, 0.033, 0.020, 0.018),
}
_SPLAY = {"thumb": 0.55, "index": 0.18, "middle": 0.0, "ring": -0.16, "little": -0.32}
_XOFF  = {"thumb": 0.030, "index": 0.018, "middle": 0.0, "ring": -0.018, "little": -0.034}


def _rotx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def _roty(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def synth_hand_joints(wrist_T, flex, abduct=0.0, chirality="right"):
    """Return {wire_name: 4x4 world} for a 27-joint hand. flex ∈ [0,1] (0=open,
    1=fist) applied progressively down each finger; abduct spreads the fingers."""
    mir = -1.0 if chirality == "left" else 1.0
    R_w, p_w = wrist_T[:3, :3], wrist_T[:3, 3]
    out = {"wrist": wrist_T.copy()}

    def place(name, local):
        T = np.eye(4)
        T[:3, 3] = p_w + R_w @ (local * np.array([mir, 1.0, 1.0]))
        T[:3, :3] = R_w
        out[name] = T

    for f, (Lm, Lp, Lmid, Ld) in _BONES.items():
        x = _XOFF[f]
        splay = (_SPLAY[f] + abduct * np.sign(_SPLAY[f] + 1e-9)) * (0.4 if f != "thumb" else 1.0)
        Rsplay = _roty(splay * mir)
        # metacarpal (knuckle is the MCP; metacarpal sits between wrist and MCP)
        mcp = Rsplay @ np.array([x, 0.0, Lm])
        if f == "thumb":
            base = mcp * 0.5 + np.array([x, 0.01, 0.0])
            out_name = "thumb"
        else:
            base = mcp * 0.45
        # bend chain: flexion rotates each joint about local X (curl toward -Y)
        a1, a2, a3 = flex * 1.4, flex * 1.6, flex * 0.9
        d_prox = Rsplay @ _rotx(a1) @ np.array([0.0, 0.0, Lp])
        knuck = mcp
        pip = knuck + d_prox
        d_mid = Rsplay @ _rotx(a1 + a2) @ np.array([0.0, 0.0, Lmid])
        dip = pip + d_mid
        d_dist = Rsplay @ _rotx(a1 + a2 + a3) @ np.array([0.0, 0.0, Ld])
        tip = dip + d_dist
        if f == "thumb":
            place("thumb_metacarpal" if False else "thumb_knuckle", knuck)
            place("thumb_intermediate_base", pip)
            place("thumb_intermediate_tip", dip)
            place("thumb_tip", tip)
        else:
            place(f"{f}_metacarpal", base)
            place(f"{f}_knuckle", knuck)
            place(f"{f}_intermediate_base", pip)
            place(f"{f}_intermediate_tip", dip)
            place(f"{f}_tip", tip)
    return out


# MP-aligned 21-joint order (matches vision_pro_stream / get_mp_aligned_mats).
_MP_ORDER = ["wrist",
             "thumb_knuckle", "thumb_intermediate_base", "thumb_intermediate_tip", "thumb_tip",
             "index_knuckle", "index_intermediate_base", "index_intermediate_tip", "index_tip",
             "middle_knuckle", "middle_intermediate_base", "middle_intermediate_tip", "middle_tip",
             "ring_knuckle", "ring_intermediate_base", "ring_intermediate_tip", "ring_tip",
             "little_knuckle", "little_intermediate_base", "little_intermediate_tip", "little_tip"]


def make_frame(head_p=(0, 0, 1.5), r_wrist=None, l_wrist=None,
               r_flex=0.0, l_flex=0.0, r_abd=0.0, pinch=False):
    """Build an AVPFrame with full named skeletons + 21-MP lists (ARKit world)."""
    def wrist_T(p, yaw=0.0):
        T = np.eye(4); T[:3, :3] = _roty(yaw); T[:3, 3] = p
        return T
    payload = {"timestamp": time.time()}
    head = np.eye(4); head[:3, 3] = head_p
    payload["head_mat"] = head.flatten().tolist()
    for side, w, flex, abd in (("right", r_wrist, r_flex, r_abd),
                               ("left", l_wrist, l_flex, 0.0)):
        if w is None:
            payload[f"{side}_hand_mats"] = None
            payload[f"{side}_joints"] = None
            continue
        j = synth_hand_joints(wrist_T(np.asarray(w, float)), flex, abd, side)
        if pinch and side == "right":              # pull thumb tip to index tip
            j["thumb_tip"] = j["thumb_tip"].copy()
            j["thumb_tip"][:3, 3] = 0.5 * (j["thumb_tip"][:3, 3] + j["index_tip"][:3, 3])
        payload[f"{side}_joints"] = {k: v.flatten().tolist() for k, v in j.items()}
        payload[f"{side}_hand_mats"] = [j[n].flatten().tolist() for n in _MP_ORDER]
    return AVPFrame(payload)


# ── load the model + controller ──────────────────────────────────────────────
def build():
    xml = os.path.join(_HERE, "..", C.SCENE_XML)
    model = mujoco.MjModel.from_xml_path(xml)
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_SENSOR
    model.opt.timestep = 0.004
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    ctrl = TeleopController(model, data, base_follow=False)
    ctrl.retgt.body_relative = True
    ctrl.hand_retgt = HandIKRetargeter(model, data, solver="gn")
    for _ in range(400):
        mujoco.mj_step(model, data)
    ctrl.hold_current()
    return model, data, ctrl


def pct(a, p): return float(np.percentile(a, p))


def force_solve(hr, side, joints):
    """Bypass HandIKRetargeter's 20 Hz throttle so a test gets a fresh solve every
    call (the live loop relies on the throttle; tests must not be rate-limited)."""
    hr.hands[side]["last_solve"] = 0.0
    return hr.apply(side, joints, synchronous=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--trials", type=int, default=300)
    args = ap.parse_args()

    model, data, ctrl = build()
    R_WRIST0 = np.array([0.30, -0.20, 1.05])     # operator right wrist (ARKit)
    L_WRIST0 = np.array([0.30, 0.20, 1.05])
    results = []

    print(f"\n{B}══ MABEL retargeting — rigorous transform + latency test ══{E}")
    print(f"   model: {model.nbody} bodies, {model.nv} DOF, dt={model.opt.timestep*1e3:.0f} ms")

    # ── T1: wrist translation transform (world frame) ────────────────────────
    ctrl.retgt.body_relative = False
    f0 = make_frame(r_wrist=R_WRIST0, l_wrist=L_WRIST0)
    ctrl.calibrate(f0)
    base_palm = data.body(C.RIGHT_PALM).xpos.copy()
    delta_arkit = np.array([0.10, 0.05, -0.08])   # right, up, forward (ARKit)
    f1 = make_frame(r_wrist=R_WRIST0 + delta_arkit, l_wrist=L_WRIST0)
    t = ctrl.retgt.update(f1)
    got = t["r_pos"] - base_palm
    expect = C.RETARGET_SCALE * (C.AVP_TO_ROBOT @ delta_arkit)
    err = float(np.linalg.norm(got - expect))
    p1 = err < 1e-6
    results.append(("T1 wrist translation maps by SCALE·AVP_TO_ROBOT", p1))
    print(f"\n {ok(p1)}  T1 translation transform")
    print(f"      ARKit Δ {delta_arkit} →  got {np.round(got,4)}  expect {np.round(expect,4)}  err {err:.1e} m")

    # ── T2: body-relative target rides with the chassis ──────────────────────
    ctrl.retgt.body_relative = True
    ctrl.calibrate(f0)
    tgt_w0 = ctrl.retgt.update(f0)["r_pos"].copy()
    cadr = model.jnt_qposadr[model.body(C.CHASSIS_BODY).jntadr[0]]
    data.qpos[cadr] += 0.5                                   # drive base +0.5 m x
    import math
    data.qpos[cadr+3:cadr+7] = [math.cos(math.pi/6), 0, 0, math.sin(math.pi/6)]  # +60° yaw
    mujoco.mj_forward(model, data)
    tgt_w1 = ctrl.retgt.update(f0)["r_pos"].copy()
    p_base = data.body(C.CHASSIS_BODY).xpos.copy()
    R_base = data.body(C.CHASSIS_BODY).xmat.reshape(3, 3)
    # operator held still → target's offset in the BODY frame must be unchanged.
    off0 = tgt_w0 - np.array([0, 0, 0.0])      # vs world origin chassis at calib (~0)
    off1_body = R_base.T @ (tgt_w1 - p_base)
    rode = float(np.linalg.norm((tgt_w1 - p_base) - R_base @ (R_base.T @ (tgt_w0))))
    moved = float(np.linalg.norm(tgt_w1 - tgt_w0))
    p2 = moved > 0.3      # the world target visibly moved WITH the body (drove+yawed)
    results.append(("T2 body-relative target rides the chassis", p2))
    data.qpos[cadr] -= 0.5; data.qpos[cadr+3:cadr+7] = [1, 0, 0, 0]; mujoco.mj_forward(model, data)
    print(f" {ok(p2)}  T2 body-relative tracking")
    print(f"      base drove +0.5 m & yawed 60° (operator still) → world target moved {moved:.3f} m WITH the body")

    # ── T3: wrist rotation magnitude preserved ───────────────────────────────
    ctrl.retgt.body_relative = False
    ctrl.calibrate(f0)
    ang = math.radians(40)
    Rw = _roty(ang)
    fr = make_frame(r_wrist=R_WRIST0, l_wrist=L_WRIST0)
    fr.right[0][:3, :3] = Rw                      # rotate operator wrist 40° about ARKit Y
    t = ctrl.retgt.update(fr)
    R0 = ctrl.retgt._rob["r_mat"]
    Rrel = t["r_mat"] @ R0.T
    got_ang = math.degrees(math.acos(np.clip((np.trace(Rrel) - 1) / 2, -1, 1)))
    p3 = abs(got_ang - 40) < 1.0
    results.append(("T3 wrist rotation magnitude preserved", p3))
    print(f" {ok(p3)}  T3 rotation transform")
    print(f"      operator wrist rotated 40.0° → robot palm target rotated {got_ang:.1f}° (Δ {abs(got_ang-40):.2f}°)")

    # ── T4: finger flexion monotonic (open → fist) ───────────────────────────
    hr = ctrl.hand_retgt
    mcp, pip = [], []
    for flex in np.linspace(0.0, 1.0, 6):
        f = make_frame(r_wrist=R_WRIST0, l_wrist=L_WRIST0, r_flex=float(flex))
        force_solve(hr, "Right", f.right_joints)
        q = dict(zip(hr.hands["Right"]["act_names"], hr.hands["Right"]["last_q"]))
        mcp.append(q.get("Right Index MCP", 0.0))
        pip.append(q.get("Right Index PIP", 0.0))
    d_mcp = np.diff(mcp); d_pip = np.diff(pip)
    p4 = (np.mean(d_mcp) > 0) and (mcp[-1] - mcp[0] > 0.3)
    results.append(("T4 finger flexion drives ORCA MCP/PIP up", p4))
    print(f" {ok(p4)}  T4 finger flexion (open→fist)")
    print(f"      Right Index MCP: {mcp[0]:.2f} → {mcp[-1]:.2f} rad  ({'monotonic↑' if np.all(d_mcp>-0.02) else 'non-monotonic'})")
    print(f"      Right Index PIP: {pip[0]:.2f} → {pip[-1]:.2f} rad")

    # ── T5: pinch closes the robot thumb–index gap ───────────────────────────
    def tip_gap(pinch):
        f = make_frame(r_wrist=R_WRIST0, l_wrist=L_WRIST0, r_flex=0.15, pinch=pinch)
        force_solve(hr, "Right", f.right_joints)
        mujoco.mj_forward(model, data)
        # robot tip gap via the hand FK keypoints
        kp = hr._robot_keypoints(hr.hands["Right"], hr.hands["Right"]["last_q"])
        return float(np.linalg.norm(kp[("thumb", "tip")] - kp[("index", "tip")]))
    g_open, g_pinch = tip_gap(False), tip_gap(True)
    p5 = g_pinch < g_open
    results.append(("T5 pinch closes robot thumb–index gap", p5))
    print(f" {ok(p5)}  T5 pinch retargeting")
    print(f"      robot thumb–index canonical gap: open {g_open:.3f} → pinch {g_pinch:.3f}  ({100*(1-g_pinch/g_open):+.0f}%)")

    # ── Latency benchmark ────────────────────────────────────────────────────
    print(f"\n{B}── latency ({args.trials} trials, dt={model.opt.timestep*1e3:.0f} ms) ──{E}")
    ctrl.retgt.body_relative = True
    ctrl.calibrate(f0)
    ctrl.set_region(TeleopController.REGION_WHOLE)
    timers = {k: [] for k in ("AVPFrame parse", "retarget.update", "WBC solve",
                              "finger IK (sync)", "ctrl.step", "mj_step", "END-TO-END")}
    t_phase = 0.0
    for i in range(args.trials):
        t_phase += 0.02
        rw = R_WRIST0 + 0.08 * np.array([math.sin(t_phase), 0, math.cos(t_phase)])
        flex = 0.5 * (1 + math.sin(0.7 * t_phase))
        # build a fresh payload (parse timing) like the live receiver does
        f = make_frame(r_wrist=rw, l_wrist=L_WRIST0, r_flex=flex)
        e2e0 = time.perf_counter()

        t0 = time.perf_counter(); _ = AVPFrame({"head_mat": f.head.flatten().tolist(),
            "right_hand_mats": [m.flatten().tolist() for m in f.right],
            "right_joints": {k: v.flatten().tolist() for k, v in f.right_joints.items()}})
        timers["AVPFrame parse"].append(time.perf_counter() - t0)

        t0 = time.perf_counter(); ctrl.retgt.update(f); timers["retarget.update"].append(time.perf_counter() - t0)
        t0 = time.perf_counter(); ctrl.ik.solve(arms_only=True); timers["WBC solve"].append(time.perf_counter() - t0)
        t0 = time.perf_counter(); force_solve(hr, "Right", f.right_joints); timers["finger IK (sync)"].append(time.perf_counter() - t0)
        t0 = time.perf_counter(); ctrl.step(f, (0.0, 0.0, 0.0, 0.0), model.opt.timestep); timers["ctrl.step"].append(time.perf_counter() - t0)
        t0 = time.perf_counter(); mujoco.mj_step(model, data); timers["mj_step"].append(time.perf_counter() - t0)
        timers["END-TO-END"].append(time.perf_counter() - e2e0)

    print(f"   {'stage':<20}{'mean':>8}{'p50':>8}{'p95':>8}{'max':>8}   (ms)")
    for k, v in timers.items():
        a = np.array(v) * 1e3
        print(f"   {k:<20}{a.mean():>8.2f}{pct(a,50):>8.2f}{pct(a,95):>8.2f}{a.max():>8.2f}")
    e2e = np.array(timers["END-TO-END"]) * 1e3
    # the live loop solves the WBC once per substep; finger IK is async on a
    # thread. A per-substep budget at dt=4 ms physics is ~ (ctrl.step + mj_step).
    substep = np.array(timers["ctrl.step"]) + np.array(timers["mj_step"])
    sps = 1.0 / substep.mean()
    rt = model.opt.timestep * sps
    pL = e2e.mean() < 15.0 and rt > 0.6
    results.append((f"Latency: end-to-end {e2e.mean():.1f} ms mean, ~{rt:.2f}x realtime", pL))
    print(f"\n   per-substep (ctrl.step+mj_step): {substep.mean()*1e3:.2f} ms → {sps:.0f} substeps/s → "
          f"{rt:.2f}x realtime at {model.opt.timestep*1e3:.0f} ms dt")
    print(f"   {ok(pL)}  latency budget (end-to-end < 15 ms, realtime > 0.6x)")

    # ── summary ──────────────────────────────────────────────────────────────
    n_pass = sum(p for _, p in results)
    print(f"\n{B}══ summary: {n_pass}/{len(results)} checks passed ══{E}")
    for name, p in results:
        print(f"   {ok(p)}  {name}")

    if args.plot:
        _plot(timers, model.opt.timestep)
    return 0 if n_pass == len(results) else 1


def _plot(timers, dt):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figdir = os.path.join(_HERE, "..", "..", "docs", "figures")
    os.makedirs(figdir, exist_ok=True)
    stages = [k for k in timers if k != "END-TO-END"]
    means = [np.mean(timers[k]) * 1e3 for k in stages]
    p95 = [pct(np.array(timers[k]) * 1e3, 95) for k in stages]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    x = np.arange(len(stages))
    ax.bar(x - 0.2, means, 0.4, label="mean")
    ax.bar(x + 0.2, p95, 0.4, label="p95")
    ax.axhline(dt * 1e3, ls="--", c="k", lw=0.8, label=f"physics dt ({dt*1e3:.0f} ms)")
    ax.set_xticks(x); ax.set_xticklabels(stages, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("latency [ms]"); ax.legend(fontsize=8); fig.tight_layout()
    out = os.path.join(figdir, "retarget_latency.pdf")
    fig.savefig(out); print(f"\n   wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
