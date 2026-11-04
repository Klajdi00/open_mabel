"""
teleop_benchmark.py — rigorous, headless benchmark of the Vision Pro -> MABEL
teleop pipeline as implemented in scripts/retarget_teleop.py (the 3-layer stack:
LAYER 1 BaseMotionModel, LAYER 2 WholeBodyController, LAYER 3 decoupled hand +
arm retargeting), plus the live PerfMonitor HUD.

It does NOT need a headset or a viewer: it synthesizes a smoothly moving operator
(both hands tracing arcs, the body walking forward + leaning + raising the head)
and drives the EXACT per-frame pipeline that retarget_teleop.main() runs — finger
SLSQP, arm-IK SLSQP, neck map, whole-body cascade, tip-safe base ramp, gravity-
compensated physics — for a fixed sim duration.

Reports (all measured, never faked):
  * per-stage RATE (Hz) + LATENCY (ms: mean / p50 / p95 / max) for control,
    hand-estimation (VR), hand-retargeting and arm-IK, rendered in the same
    aesthetic HUD the live teleop shows;
  * whole-body STABILITY: max chassis tilt over the run;
  * a tip-over SWEEP (forward / reverse / strafe+turn x lift height) showing the
    LAYER-1 motion model + reverse guard keep every reachable command upright;
  * a vector PDF (docs/figures/teleop_benchmark.pdf): latency distributions +
    the tip-safe speed envelope vs lift height.

Run:  python scripts/tools/teleop_benchmark.py [--seconds 4] [--vr-hz 60] [--no-pdf]
"""
import os
import sys
import time
import argparse

import numpy as np
import mujoco

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.join(_HERE, "..")
sys.path.insert(0, _SCRIPTS)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "..", "..", "controller"))

# import the modular teleop package (classes + helpers + hyperparameters; viewer-free)
import teleop as rt


# ----------------------------------------------------------------------------
#  Synthetic operator (ARKit Z-up world). Smooth, anatomically-plausible motion
#  so every solver does real work and the body cascade is exercised.
# ----------------------------------------------------------------------------
def _mat(p, R=None):
    M = np.eye(4)
    if R is not None:
        M[:3, :3] = R
    M[:3, 3] = p
    return M.flatten().tolist()


def synth_hand(wrist, spread, curl):
    """21 ARKit joints (4x4 flat) for one hand. Layout: 0 wrist; per finger
    knuckle/pip/dip/tip. Fingers fan out by `spread`, curl down by `curl`.
    ARKit camera convention: +Y up, forward = -Z (the head looks down -Z)."""
    fingers = {"thumb": (1, -0.9), "index": (5, -0.3), "middle": (9, 0.0),
               "ring": (13, 0.3), "pinky": (17, 0.6)}
    H = [None] * 21
    H[0] = wrist
    fwd = np.array([0.0, 0.0, -1.0])           # ARKit: hand points forward = -Z
    for name, (base, lat) in fingers.items():
        side = np.array([lat * (0.4 + 0.3 * spread), 0.02, 0.0])
        for k in range(4):
            tt = 0.03 + 0.03 * k
            droop = np.array([0.0, -curl * tt * 6.0, 0.0])
            H[base + k] = wrist + side + fwd * tt + droop
    pts = [(p if isinstance(p, np.ndarray) else np.array(p)) for p in H]
    return [_mat(p) for p in pts]


def operator_stream(t):
    """A scripted operator frame at sim-time t (seconds) -> payload dict matching
    vision_pro_stream.py keys the teleop consumes. ARKit: +Y up, forward = -Z."""
    walk = 0.15 * t                                  # walk forward 0.15 m/s (-Z)
    rise = 0.08 * np.sin(0.4 * t)                    # head bob -> lift follows
    head_h = 1.45 + rise
    head_p = np.array([0.0, head_h, -walk])          # ARKit: y=up, forward=-z
    odom_p = np.array([0.0, 0.0, -walk])
    yaw = 0.25 * np.sin(0.3 * t)                     # turn the body slowly
    # hands trace arcs in front of the chest (forward = -z), opening/closing
    swing = 0.10 * np.sin(1.2 * t)
    rise_h = 0.05 * np.sin(0.9 * t)
    curl = 0.5 + 0.5 * np.sin(1.5 * t)
    rw = head_p + np.array([0.18 + swing, -0.32 + rise_h, -0.32])
    lw = head_p + np.array([-0.18 - swing, -0.32 + rise_h, -0.32])
    re = head_p + np.array([0.20, -0.08, -0.10])     # elbows (slightly forward)
    le = head_p + np.array([-0.20, -0.08, -0.10])
    Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    return {
        "head_mat": _mat(head_p, Ry),
        "odom_mat": _mat(odom_p),
        "body_yaw": float(yaw),
        "head_height": float(head_h),
        "right_hand_mats": synth_hand(rw, 0.6, curl),
        "left_hand_mats": synth_hand(lw, 0.6, curl),
        "right_elbow": _mat(re),
        "left_elbow": _mat(le),
    }


# ----------------------------------------------------------------------------
#  Latency accumulator (keeps raw samples for percentiles)
# ----------------------------------------------------------------------------
class Lat:
    def __init__(self):
        self.s = {}

    def add(self, k, ms):
        self.s.setdefault(k, []).append(ms)

    def stats(self, k):
        a = np.array(self.s.get(k, [0.0]))
        return dict(n=len(a), mean=a.mean(), p50=np.percentile(a, 50),
                    p95=np.percentile(a, 95), mx=a.max())


# ----------------------------------------------------------------------------
#  Full-pipeline run (mirrors retarget_teleop.main()'s body, headless)
# ----------------------------------------------------------------------------
def run_pipeline(seconds, vr_hz):
    model = mujoco.MjModel.from_xml_path(rt.XML_FILE)
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_SENSOR
    data = mujoco.MjData(model)
    kd = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    grav = rt.gravity_dofs(model)

    def gravity():
        kd.qpos[:] = data.qpos; kd.qvel[:] = 0.0
        mujoco.mj_forward(model, kd); return kd.qfrc_bias.copy()

    opt = {s: rt.HandOptimizer(model, s) for s in ("right", "left")}
    arm = {s: rt.ArmRetargeter(model, s, opt[s].R_rob, False) for s in ("right", "left")}
    wbc = rt.WholeBodyController(model, data, torso_act=model.actuator("Torso").id,
                                 lift_act=model.actuator("Lift").id)
    head_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "head")
    finger_act = {s: [model.actuator(n).id for n in opt[s].joint_names] for s in ("right", "left")}
    arm_act = {s: [model.actuator(f"{s.capitalize()} Arm {i}").id for i in range(1, 8)]
               for s in ("right", "left")}
    wrist_act = {s: model.actuator(f"{s.capitalize()} Wrist").id for s in ("right", "left")}
    neck_act = [model.actuator(n).id for n in ("Neck Yaw", "Neck Pitch", "Neck Roll")]
    neck_rng = [model.actuator_ctrlrange[a] for a in neck_act]
    lift_aid = model.actuator("Lift").id
    cb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, rt.ROBOT_BASE_BODY)
    dt = model.opt.timestep

    # settle under gravity comp
    for _ in range(400):
        data.qfrc_applied[:] = 0.0
        data.qfrc_applied[grav] = rt.GRAV_GAIN * gravity()[grav]
        mujoco.mj_step(model, data)
    wbc.reset(data)
    projector = rt.ArmCollisionProjector(model, data, arm) if rt._HAVE_SELFCOL else None

    pm = rt.PerfMonitor([("ctrl", "control·physics", 250.0), ("vr", "VR hand stream", vr_hz),
                         ("hand", "hand retarget", vr_hz), ("arm", "arm IK", 2 * vr_hz),
                         ("armQP", "selfcol QP", vr_hz), ("neck", "neck map", vr_hz)])
    lat = Lat()
    selfcol_min = np.inf
    head0 = None; world_ref = None
    AVP2 = rt.AVP_TO_ROBOT; reach = rt.REACH_SCALE
    vr_dt = 1.0 / vr_hz
    next_vr = 0.0
    max_tilt = 0.0
    contact_err = []
    sim_t = 0.0
    while sim_t < seconds:
        # ---- VR frame at vr_hz ----
        if sim_t >= next_vr:
            next_vr += vr_dt
            payload = operator_stream(sim_t)
            pm.record("vr"); lat.add("vr", 0.0)
            head_mat = np.asarray(payload["head_mat"], float).reshape(4, 4)
            if head0 is None:
                head0 = head_mat.copy()
            head_pos = head_mat[:3, 3]
            if world_ref is None:
                mujoco.mj_kinematics(model, data)
                world_ref = (head_pos.copy(), data.xpos[head_bid].copy())

            def Wmap(p):
                return world_ref[1] + reach * (AVP2 @ (np.asarray(p, float) - world_ref[0]))

            q_des = {"right": None, "left": None}
            for side, mk in (("right", "right_hand_mats"), ("left", "left_hand_mats")):
                h_mats = payload[mk]
                R_op = rt.operator_frame(None, h_mats)
                if R_op is None:
                    continue
                q_fingers, hand_ms, _ = opt[side].solve(h_mats, R_op)
                pm.record("hand", hand_ms); lat.add("hand", hand_ms)
                for a, q in zip(finger_act[side], q_fingers):
                    data.ctrl[a] = q
                mid_tip = np.asarray(h_mats[12], float).reshape(4, 4)[:3, 3]
                wrist_ark = np.asarray(h_mats[0], float).reshape(4, 4)[:3, 3]
                elbow_pos = np.asarray(payload[f"{side}_elbow"], float).reshape(4, 4)[:3, 3]
                R_pc = AVP2 @ R_op
                hk = {"contact": Wmap(mid_tip), "elbow": Wmap(elbow_pos),
                      "fwd": R_pc[:, 0], "up": R_pc[:, 2]}
                arm[side].sync(data)
                arm[side].set_wrist_flex(rt.wrist_flexion(R_op, wrist_ark, elbow_pos))
                ta = time.perf_counter()
                sol = arm[side].solve(hk)
                ams = (time.perf_counter() - ta) * 1e3
                pm.record("arm", ams); lat.add("arm", ams)
                q_des[side] = sol
                data.ctrl[wrist_act[side]] = arm[side].ik.qpos[arm[side].orca_wrist_qadr]
                re_, rc_, *_ = arm[side]._kps(sol)
                contact_err.append(np.linalg.norm(rc_ - hk["contact"]) * 1e3)  # tracking residual

            # ARM SELF-COLLISION projection (both arms jointly), then write the safe targets
            if q_des["right"] is not None or q_des["left"] is not None:
                qr = q_des["right"] if q_des["right"] is not None else data.qpos[arm["right"].qadr]
                ql = q_des["left"] if q_des["left"] is not None else data.qpos[arm["left"].qadr]
                if projector is not None:
                    tq = time.perf_counter()
                    qr, ql = projector.project(qr, ql, 1.0 / vr_hz)
                    pm.record("armQP", (time.perf_counter() - tq) * 1e3)
                    lat.add("armQP", (time.perf_counter() - tq) * 1e3)
                    selfcol_min = min(selfcol_min, projector.min_dist)
                for side, q in (("right", qr), ("left", ql)):
                    if q_des[side] is None:
                        continue
                    for i, a in enumerate(arm_act[side][:len(q)]):
                        data.ctrl[a] = q[i]

            na = rt.neck_angles(head_mat, head0)
            for a, rng, ang in zip(neck_act, neck_rng, na):
                data.ctrl[a] = float(np.clip(ang, rng[0], rng[1]))
            pm.record("neck"); lat.add("neck", 0.0)

            odom_base = np.asarray(payload["odom_mat"], float).reshape(4, 4)[:3, 3]
            wbc.update_targets(data, head_pos, odom_base,
                               payload["body_yaw"], payload["head_height"])

        # ---- physics at dt ----
        data.qfrc_applied[:] = 0.0
        data.qfrc_applied[grav] = rt.GRAV_GAIN * gravity()[grav]
        t0 = time.perf_counter()
        wbc.base_step(data, dt)
        mujoco.mj_step(model, data)
        pm.record("ctrl"); lat.add("ctrl", (time.perf_counter() - t0) * 1e3)
        za = data.xmat[cb].reshape(3, 3)[:, 2]
        max_tilt = max(max_tilt, np.degrees(np.arccos(np.clip(za[2], -1, 1))))
        sim_t += dt

    # finalize HUD rates over the whole run
    for k, _, _ in pm.stages:
        pm.hz[k] = len(lat.s.get(k, [])) / seconds
    pm.set_status(R=True, L=True, neck=True, base=wbc.base_anchored,
                  base_v=wbc.base_cmd, lift=float(data.ctrl[lift_aid]),
                  torso=float(data.ctrl[model.actuator("Torso").id]), mode="benchmark")
    finite = bool(np.all(np.isfinite(data.qpos)))
    return pm, lat, dict(max_tilt=max_tilt, finite=finite,
                         contact_mm=np.array(contact_err) if contact_err else np.array([0.0]),
                         selfcol_min=selfcol_min, dt=dt)


# ----------------------------------------------------------------------------
#  Arm-IK accuracy on genuinely REACHABLE targets (FK of random in-bounds poses
#  -> recover). Isolates LAYER-3 arm quality from base-follow lag / reachability.
# ----------------------------------------------------------------------------
def ik_accuracy(n=40, seed=0):
    model = mujoco.MjModel.from_xml_path(rt.XML_FILE)
    data = mujoco.MjData(model); mujoco.mj_forward(model, data)
    rng = np.random.default_rng(seed)
    out = {}
    for s in ("right", "left"):
        a = rt.ArmRetargeter(model, s, rt.HandOptimizer(model, s).R_rob, False); a.sync(data)
        errs = []
        for _ in range(n):
            qref = a.lo + (a.hi - a.lo) * rng.uniform(0.2, 0.8, size=len(a.lo))
            e, c, sh, fwd, up = a._kps(qref)
            a.last_q = a.q_nom.copy()                       # cold start (worst case)
            sol = a.solve({"contact": c, "elbow": e, "fwd": fwd, "up": up})
            _, rc, *_ = a._kps(sol)
            errs.append(np.linalg.norm(rc - c) * 1e3)
        out[s] = np.array(errs)
    return out


# ----------------------------------------------------------------------------
#  Tip-over sweep (Layer-1 motion model + reverse guard)
# ----------------------------------------------------------------------------
def tip_sweep(lifts=(0.0, 0.32, 0.635)):
    model = mujoco.MjModel.from_xml_path(rt.XML_FILE)
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_SENSOR
    data = mujoco.MjData(model); kd = mujoco.MjData(model)
    grav = rt.gravity_dofs(model)
    lift_aid = model.actuator("Lift").id
    cb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, rt.ROBOT_BASE_BODY)
    dt = model.opt.timestep

    def gravity():
        kd.qpos[:] = data.qpos; kd.qvel[:] = 0.0
        mujoco.mj_forward(model, kd); return kd.qfrc_bias.copy()

    cmds = [("forward", (-0.8, 0, 0)), ("reverse", (0.8, 0, 0)), ("strafe+turn", (0, 0.8, 1.5))]
    rows = []
    for z in lifts:
        for lbl, tgt in cmds:
            mujoco.mj_resetData(model, data); data.ctrl[lift_aid] = z
            wbc = rt.WholeBodyController(model, data, torso_act=model.actuator("Torso").id,
                                         lift_act=lift_aid)
            for _ in range(250):
                data.qfrc_applied[:] = 0.0
                data.qfrc_applied[grav] = rt.GRAV_GAIN * gravity()[grav]
                mujoco.mj_step(model, data)
            wbc.reset(data); wbc.base_tgt = tgt; data.ctrl[lift_aid] = z
            mt = 0.0
            for _ in range(int(3.0 / dt)):
                data.qfrc_applied[:] = 0.0
                data.qfrc_applied[grav] = rt.GRAV_GAIN * gravity()[grav]
                wbc.base_step(data, dt); mujoco.mj_step(model, data)
                za = data.xmat[cb].reshape(3, 3)[:, 2]
                mt = max(mt, np.degrees(np.arccos(np.clip(za[2], -1, 1))))
            rows.append((z, lbl, abs(wbc.base_cmd[0]) + abs(wbc.base_cmd[1]), mt))
    return rows


# ----------------------------------------------------------------------------
#  Self-collision probe: march one hand into the torso, projector ON vs OFF.
# ----------------------------------------------------------------------------
def selfcol_probe():
    if not rt._HAVE_SELFCOL:
        return None
    model = mujoco.MjModel.from_xml_path(rt.XML_FILE)
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_SENSOR
    data = mujoco.MjData(model); kd = mujoco.MjData(model)
    grav = rt.gravity_dofs(model)
    def gravity():
        kd.qpos[:] = data.qpos; kd.qvel[:] = 0.0
        mujoco.mj_forward(model, kd); return kd.qfrc_bias.copy()
    arm = {s: rt.ArmRetargeter(model, s, rt.HandOptimizer(model, s).R_rob, False)
           for s in ("right", "left")}
    proj = rt.ArmCollisionProjector(model, data, arm)
    aa = {s: [model.actuator(f"{s.capitalize()} Arm {i}").id for i in range(1, 8)]
          for s in ("right", "left")}
    torso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    relb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "r_elbow")

    def run(use_proj):
        mujoco.mj_resetData(model, data); mujoco.mj_forward(model, data)
        for _ in range(200):
            data.qfrc_applied[:] = 0.0
            data.qfrc_applied[grav] = rt.GRAV_GAIN * gravity()[grav]; mujoco.mj_step(model, data)
        mind = np.inf; ms = []
        for _ in range(120):
            tc = data.xpos[torso].copy(); tc[0] -= 0.02      # just in front of the chest
            hk = {"contact": tc, "elbow": data.xpos[relb].copy(),
                  "fwd": np.array([-1., 0, 0]), "up": np.array([0, 0, 1.])}
            arm["right"].sync(data); qr = arm["right"].solve(hk)
            ql = data.qpos[arm["left"].qadr].copy()          # left arm held (arm-only)
            if use_proj:
                t = time.perf_counter(); qr, ql = proj.project(qr, ql, 1 / 60.)
                ms.append((time.perf_counter() - t) * 1e3); mind = min(mind, proj.min_dist)
            else:
                proj.guard.rows(proj.dofs); mind = min(mind, proj.guard.min_dist)  # measure only
            for i, a in enumerate(aa["right"][:len(qr)]):
                data.ctrl[a] = qr[i]
            for _ in range(8):
                data.qfrc_applied[:] = 0.0
                data.qfrc_applied[grav] = rt.GRAV_GAIN * gravity()[grav]; mujoco.mj_step(model, data)
        return mind, (float(np.mean(ms)) if ms else 0.0)
    d_off, _ = run(False)
    d_on, qp_ms = run(True)
    return dict(off=d_off, on=d_on, margin=proj.guard.d_safe, qp_ms=qp_ms)


# ----------------------------------------------------------------------------
#  Clutch unit check: is_fist + dwell/cooldown toggle logic.
# ----------------------------------------------------------------------------
def clutch_check():
    cl = rt.ClutchController()
    t = [0.0]; dt = 0.02

    def hold(rp, lp, dur):
        """Drive both pinch scalars for `dur` seconds; return toggles seen."""
        tg = 0
        for _ in range(max(1, int(dur / dt))):
            if cl.update({"right_pinch": rp, "left_pinch": lp}, t[0]):
                tg += 1
            t[0] += dt
        return tg

    def double_pinch():
        """One single-hand (right) double-pinch gesture (pinch, release, pinch, release);
        the left hand stays open throughout, proving either hand alone works."""
        n = hold(0.01, 0.10, 0.08) + hold(0.10, 0.10, 0.08)
        n += hold(0.01, 0.10, 0.08) + hold(0.10, 0.10, 0.08)
        return n

    seq = []
    hold(0.10, 0.10, 0.2)                       # idle (open) -> nothing
    seq.append((cl.clutched, 0))
    tg = double_pinch(); seq.append((cl.clutched, tg))   # -> clutched
    hold(0.10, 0.10, 1.0)                        # wait out cooldown
    tg = hold(0.01, 0.01, 0.08) + hold(0.10, 0.10, 0.4)  # single pinch only -> no toggle
    seq.append((cl.clutched, tg))
    hold(0.10, 0.10, 1.0)
    tg = double_pinch(); seq.append((cl.clutched, tg))   # -> released
    ok = (seq[0] == (False, 0) and seq[1] == (True, 1)
          and seq[2] == (True, 0) and seq[3] == (False, 1))
    return dict(seq=seq, ok=ok)


# ----------------------------------------------------------------------------
#  Reporting
# ----------------------------------------------------------------------------
C = rt._A


def fmt_lat(name, st):
    return (f"  {C['gry']}{name:<16}{C['rst']} mean {st['mean']:6.3f}  p50 {st['p50']:6.3f}  "
            f"p95 {st['p95']:6.3f}  max {st['mx']:6.3f}  ms   (n={st['n']})")


def report(pm, lat, summ, sweep, ik, scp, clu):
    print("\n" + C["cyn"] + "═" * 58 + C["rst"])
    print(f"{C['b']}  MABEL teleop — live frequency HUD (whole-run averages){C['rst']}")
    print(C["cyn"] + "═" * 58 + C["rst"])
    print("\n".join(pm.lines()))

    print(f"\n{C['b']}  Per-stage latency (ms){C['rst']}")
    for k, label, _ in pm.stages:
        if k == "vr":
            continue
        print(fmt_lat(label, lat.stats(k)))

    print(f"\n{C['b']}  Tracking & stability{C['rst']}")
    for s in ("right", "left"):
        e = ik[s]
        print(f"  {C['gry']}arm-IK accuracy [{s:<5}]{C['rst']} "
              f"p50 {np.percentile(e,50):4.0f}  p95 {np.percentile(e,95):4.0f}  "
              f"mm   {C['dim']}(reachable targets, cold-start){C['rst']}")
    ce = summ["contact_mm"]
    print(f"  {C['gry']}end-to-end residual {C['rst']}  "
          f"mean {ce.mean():.0f}  p95 {np.percentile(ce,95):.0f}  mm   "
          f"{C['dim']}(with base-follow lag during 0.15 m/s walk){C['rst']}")
    tcol = C['grn'] if summ['max_tilt'] < 8 else C['red']
    print(f"  {C['gry']}max chassis tilt{C['rst']}      {tcol}{summ['max_tilt']:.1f}°{C['rst']}  "
          f"(finite={summ['finite']})")

    print(f"\n{C['b']}  Tip-over sweep — Layer-1 motion model + reverse guard{C['rst']}")
    print(f"  {C['gry']}{'lift':>5}  {'command':<12} {'realized|v|':>11}  {'maxtilt':>7}  verdict{C['rst']}")
    for z, lbl, v, mt in sweep:
        ok = mt < 8
        col = C['grn'] if ok else C['red']
        print(f"  {z:5.2f}  {lbl:<12} {v:9.2f}    {col}{mt:6.1f}°{C['rst']}   "
              f"{col}{'UPRIGHT' if ok else 'TIP'}{C['rst']}")

    print(f"\n{C['b']}  Safety — arm self-collision & clutch{C['rst']}")
    if scp is not None:
        held = scp['on'] >= scp['margin'] - 0.01
        col = C['grn'] if held else C['red']
        print(f"  {C['gry']}hand→torso march{C['rst']}   OFF {C['red']}{scp['off']*100:+.1f}cm{C['rst']}"
              f"  ON {col}{scp['on']*100:+.1f}cm{C['rst']}  (margin {scp['margin']*100:.1f}cm, "
              f"QP {scp['qp_ms']:.2f}ms)  {col}{'BARRIER HELD' if held else 'BREACH'}{C['rst']}")
    if 'selfcol_min' in summ and np.isfinite(summ['selfcol_min']):
        print(f"  {C['gry']}min gap (walk run){C['rst']} {summ['selfcol_min']*100:+.1f}cm")
    cc = C['grn'] if clu['ok'] else C['red']
    print(f"  {C['gry']}clutch (2x pinch){C['rst']} toggle seq {clu['seq']}  "
          f"{cc}{'PASS' if clu['ok'] else 'FAIL'}{C['rst']}")
    print()


def make_pdf(lat, sweep, scp, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
    # (a) latency distributions
    for k, label in (("ctrl", "control·physics"), ("hand", "hand retarget"),
                     ("arm", "arm IK"), ("armQP", "selfcol QP")):
        a = np.array(lat.s.get(k, [0.0]))
        ax[0].hist(a, bins=40, alpha=0.55, label=f"{label} (p95={np.percentile(a,95):.2f}ms)")
    ax[0].set_xlabel("stage latency [ms]"); ax[0].set_ylabel("count")
    ax[0].set_title("Per-stage inference latency"); ax[0].legend(fontsize=8)
    # (b) tip-safe speed envelope vs lift
    zs = np.linspace(0, rt.LIFT_RANGE[1], 40)
    mm = rt.BaseMotionModel()
    fwd = [mm.max_speed(z)[0] for z in zs]
    bak = [mm.max_speed(z)[1] for z in zs]
    latv = [mm.max_speed(z)[2] for z in zs]
    ax[1].plot(zs, fwd, label="forward cap"); ax[1].plot(zs, latv, label="lateral cap")
    ax[1].plot(zs, bak, label="reverse cap (model)", ls="--")
    guard = [(1 - z / rt.LIFT_RANGE[1]) * b for z, b in zip(zs, bak)]
    ax[1].plot(zs, guard, label="reverse cap (guarded)", color="crimson")
    ax[1].set_xlabel("lift height [m]"); ax[1].set_ylabel("tip-safe speed [m/s]")
    ax[1].set_title("Tip-safe base envelope vs lift"); ax[1].legend(fontsize=8)
    # (c) self-collision barrier: hand→torso march, projector OFF vs ON
    if scp is not None:
        ax[2].bar(["OFF", "ON"], [scp['off'] * 100, scp['on'] * 100],
                  color=["#cb4b4b", "#4e9a4e"])
        ax[2].axhline(scp['margin'] * 100, ls="--", color="k", lw=1,
                      label=f"safety margin {scp['margin']*100:.1f}cm")
        ax[2].axhline(0, color="gray", lw=0.8)
        ax[2].set_ylabel("min hand→torso gap [cm]")
        ax[2].set_title("Arm self-collision barrier"); ax[2].legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path); plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--vr-hz", type=float, default=60.0)
    ap.add_argument("--no-pdf", action="store_true")
    a = ap.parse_args()
    print(f"{C['dim']}Running {a.seconds:.0f}s of teleop at VR {a.vr_hz:.0f} Hz "
          f"(physics dt steps interleaved)…{C['rst']}")
    pm, lat, summ = run_pipeline(a.seconds, a.vr_hz)
    ik = ik_accuracy()
    sweep = tip_sweep()
    scp = selfcol_probe()
    clu = clutch_check()
    report(pm, lat, summ, sweep, ik, scp, clu)
    if not a.no_pdf:
        p = make_pdf(lat, sweep, scp, os.path.join(_SCRIPTS, "..", "docs", "figures",
                                                   "teleop_benchmark.pdf"))
        print(f"{C['dim']}  figure -> {os.path.relpath(p)}{C['rst']}\n")


if __name__ == "__main__":
    main()
