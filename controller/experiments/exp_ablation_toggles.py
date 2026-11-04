#!/usr/bin/env python3
"""exp_ablation_toggles.py — do the whole-body ablation switches do exactly what they say?

The Vision Pro exposes four switches for ablation studies (2026-10-16): FREEZE NECK,
FREEZE TORSO and UPPER BODY | WHOLE BODY (upper = lift + mobile base frozen). They are
NOT different retargeters: the same engine runs, and a frozen part is PARKED at home
after it (engine.ChannelPark — neck level & forward, torso 0 rad, lift at the home-
keyframe height; the base is taken out of the follow loop by LOCAL scope).

This drives the REAL facade (mabel.MabelTeleop → TeleopEngine → MuJoCo physics, the
same stack the gateway runs) with a scripted operator who does everything at once —
turns and nods the head, leans forward, crouches and rises, walks forward, turns the
body, and traces a bimanual circle — and measures, per condition:

  neck      range of the 3 neck joints  (frozen ⇒ ≈0, at home 0)
  torso     range of the torso joint    (frozen ⇒ ≈0, at 0)
  lift      range of the lift joint     (frozen ⇒ ≈0, at home)
  base      planar displacement         (upper body ⇒ ≈0)
  arms      range of the right-arm joints — must stay ACTIVE in every condition
            (the "same retargeter" claim: freezing the body must not stop the arms)

plus a mid-run OFF→ON→OFF sweep for bumplessness (peak joint speed at each switch).

    python3 controller/experiments/exp_ablation_toggles.py \\
        --fig controller/docs/figures/ablation_toggles.pdf
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import mujoco  # noqa: E402

from mabel.facade import MabelTeleop  # noqa: E402
from mabel.types import OperatorPose  # noqa: E402
from mabel.retargeting.synthetic import SyntheticOperator  # noqa: E402

REPO = os.path.dirname(os.path.dirname(HERE))
SCENE = os.path.join(REPO, "simulation", "mabel_mujoco", "scenes", "mabel_home.xml")
FRAME_HZ = 30.0          # operator frame cadence (the headset streams ~30-60 Hz)

CONDITIONS = {
    #  name               neck   torso  lift   scope
    "whole body":       (True,  True,  True,  "global"),
    "freeze neck":      (False, True,  True,  "global"),
    "freeze torso":     (True,  False, True,  "global"),
    "upper body":       (True,  True,  False, "local"),
    "all frozen":       (False, False, False, "local"),
}


def scale_link_timeouts():
    """This loop runs well below real time, while the EngagementSupervisor stamps
    freshness from the WALL clock — so 30 Hz frames straddle the 0.40 s stream
    timeout, the robot drops out of `engaged` and eases toward HOME, and a frozen
    joint would look frozen for the WRONG reason. There is no link to lose in an
    offline run, so scale the two link-loss timeouts ×20 (controller the root README,
    exp_egodex_retarget.py) in EVERY namespace that bound them at import. Ease
    times are untouched — they shape the motion being measured."""
    import mabel.config as C
    import mabel.teleop_engine.safety as S
    import mabel.teleop_engine.engine as E
    for mod in (C, S, E):
        for k, v in (("SAFE_STREAM_TIMEOUT", 0.40), ("SAFE_SIGNAL_TIMEOUT", 0.50)):
            if hasattr(mod, k) and getattr(mod, k) < v * 2:
                setattr(mod, k, v * 20.0)


STILL_BODY = False      # control condition: head + hands move, the body does not


def operator_at(op, root0, t):
    """Everything at once. Returns (head 4x4, skeleton dict) in ARKit cabin frame."""
    # head: look around (yaw) and nod (pitch)
    op.neck = (0.45 * np.sin(2 * np.pi * 0.20 * t), 0.25 * np.sin(2 * np.pi * 0.13 * t), 0.0)
    # body: turn, lean forward (root forward + down), crouch/rise, and STEP forward
    # and back. The step OSCILLATES (out to 0.30 m, past the 0.18 m comfort box,
    # and back): a continuous walk would pin the head at the box edge, saturate
    # the torso bend and make "freeze torso" pass trivially (measured: −17.4°
    # with 0.4° of range). Inside the box the torso takes the offset; past it
    # the base takes the overflow — so both are genuinely driven.
    if STILL_BODY:
        op.torso_yaw = 0.0
        op.root = root0.copy()
    else:
        op.torso_yaw = 0.30 * np.sin(2 * np.pi * 0.08 * t)
        lean = 0.5 - 0.5 * np.cos(2 * np.pi * 0.17 * t)            # 0..1
        rise = 0.10 * np.sin(2 * np.pi * 0.07 * t)
        step = 0.30 * (0.5 - 0.5 * np.cos(2 * np.pi * 0.12 * t))   # 0..0.30 m, 8.3 s period
        op.root = root0 + np.array([0.0, rise - 0.06 * lean, -0.12 * lean - step])
    # hands: a bimanual circle in front of the torso (keeps the arm IK busy)
    a = 2 * np.pi * 0.25 * t
    for side, s in (("right", 1.0), ("left", -1.0)):
        op.wrist[side] = (0.72, np.array([s * 0.35 + 0.15 * np.cos(a),
                                          -0.2 + 0.15 * np.sin(a), 0.9]))
    T_ad, hands, skel = op.frame()
    # A REAL hand, not a bare wrist: the arm solver needs the operator's PALM
    # frame (geometry.operator_frame: wrist + index/middle/little metacarpals)
    # and HOLDS the arm without one — a first version of this experiment fed
    # bare wrists and the arms never moved in ANY condition. Same construction
    # as tests/test_arm_elbow_retarget.py: fingers along the forearm, palm down,
    # true left-hand chirality.
    for side in ("right", "left"):
        w = hands[side][:3, 3]
        e = skel[side]["forearm_arm"][:3, 3]
        f = w - e; f = f / np.linalg.norm(f)
        n = np.array([0.0, -1.0, 0.0]); n = n - f * (n @ f); n = n / np.linalg.norm(n)
        nf = n if side == "right" else -n
        sv = np.cross(nf, f)
        R = np.column_stack([f, sv, nf])

        def T(pp):
            M = np.eye(4); M[:3, :3] = R; M[:3, 3] = pp
            return M
        skel[side] = {"wrist": T(w),
                      "middle_metacarpal": T(w + 0.08 * f),
                      "index_metacarpal": T(w + 0.08 * f - 0.025 * sv),
                      "little_metacarpal": T(w + 0.08 * f + 0.025 * sv),
                      "forearm_arm": skel[side]["forearm_arm"]}
        hands[side] = T(w)
    return T_ad, hands, skel


def pose_from(T_ad, hands, skel, t):
    """Build the OperatorPose the way the live gateway does from an app frame
    (mabel.sources.wire): the ARM retargeter reads `right_hand`/`left_hand` (a
    21-joint list, [0] = wrist — the engine's `*_hand_mats`), the elbow
    objective reads the forearm joint. Feeding only the named skeleton left the
    arms with NO input at all (they never moved), which a first version of this
    experiment silently did."""
    return OperatorPose(head=T_ad,
                        right_hand=[hands["right"]] * 21, left_hand=[hands["left"]] * 21,
                        right_joints=skel["right"], left_joints=skel["left"],
                        right_elbow=skel["right"].get("forearm_arm"),
                        left_elbow=skel["left"].get("forearm_arm"),
                        timestamp=t)


def joint_qadr(model, name):
    return model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]


def act_joint_qadr(model, act_name):
    a = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
    return model.jnt_qposadr[model.actuator_trnid[a, 0]]


def run(cond, seconds=14.0, schedule=None, label=None):
    """One physics run. `cond` = (neck, torso, lift, scope) applied after a 2 s
    all-on warm-up (so every part has first been DRIVEN away from home — a freeze
    that never had to move anything would prove nothing). `schedule` = list of
    (t, cond) switches instead, for the bumpless sweep."""
    scale_link_timeouts()
    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)
    if model.nkey > 0:                       # every MjData boot seeds keyframe 0
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    for _ in range(400):
        mujoco.mj_step(model, data)
    tele = MabelTeleop(model, data)
    eng = tele.controller

    def apply(c):
        n, to, li, sc = c
        tele.set_control_spec(method="retarget", scope=sc,
                              tracking={"neck": n, "torso": to, "lift": li,
                                        "arms": True, "hands": True})
    apply((True, True, True, "global"))
    sched = list(schedule or [(2.0, cond)])

    q_neck = [act_joint_qadr(model, n) for n in ("Neck Yaw", "Neck Pitch", "Neck Roll")]
    q_torso = act_joint_qadr(model, "Torso")
    # The lift is a cascaded TENDON (controller the root README): read the actuator's
    # own length (= ctrl units, the same units as the home target), never a
    # joint — its trnid is a tendon id, and reading it as a joint id is garbage.
    a_lift = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "Lift")
    q_arm = [joint_qadr(model, f"Right Arm {i}") for i in range(1, 8)]
    q_base = model.jnt_qposadr[0]            # free joint: x y z qw qx qy qz
    a_neck = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
              for n in ("Neck Yaw", "Neck Pitch", "Neck Roll")]
    a_torso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "Torso")
    op = SyntheticOperator()
    root0 = op.root.copy()
    dt = model.opt.timestep
    per_frame = max(1, int(round(1.0 / (FRAME_HZ * dt))))
    rec = []
    pose = None
    steps = int(seconds / dt)
    for i in range(steps):
        t = i * dt
        if sched and t >= sched[0][0]:
            apply(sched.pop(0)[1])
        if i % per_frame == 0:
            pose = pose_from(*operator_at(op, root0, t), t)
        if i == per_frame:
            # The anchor hold is on the WALL clock (HEAD_ANCHOR_HOLD): wait it out
            # once, or the whole retarget path silently no-ops (see test_retarget_modes).
            time.sleep(0.45)
        tele.update(pose, dt=dt)
        mujoco.mj_step(model, data)
        if i % 10 == 0:
            qb = data.qpos[q_base:q_base + 7]
            rec.append({"t": t,
                        "neck": [float(data.qpos[a]) for a in q_neck],
                        "torso": float(data.qpos[q_torso]),
                        "lift": float(data.actuator_length[a_lift]),
                        "base": [float(qb[0]), float(qb[1])],
                        "arm": [float(data.qpos[a]) for a in q_arm],
                        "mode": str(getattr(eng, "mode", "")),
                        # COMMANDS + WALL time: the controller's rate limits and
                        # park eases run on the WALL clock and this loop runs ~10×
                        # slower than real time, so bumplessness is judged on the
                        # commanded rate per WALL second — what a real-time robot
                        # would execute — not on sim-time joint speed (inflated ~10×).
                        "wall": time.time(),
                        "c_neck": [float(data.ctrl[a]) for a in a_neck],
                        "c_torso": float(data.ctrl[a_torso]),
                        "c_lift": float(data.ctrl[a_lift])})
    return {"label": label, "rec": rec, "lift_home": float(eng.lift_home),
            "park_time": float(eng.park["neck"].seconds)}


def summarize(r, t0, t1):
    """Ranges over [t0, t1] (after the park has settled)."""
    allR = [x for x in r["rec"] if t0 <= x["t"] <= t1]
    R = [x for x in allR if x.get("mode", "engaged") == "engaged"]   # score ENGAGED frames only
    neck = np.array([x["neck"] for x in R]); torso = np.array([x["torso"] for x in R])
    lift = np.array([x["lift"] for x in R]); base = np.array([x["base"] for x in R])
    arm = np.array([x["arm"] for x in R])
    return {
        "neck_range_deg": float(np.degrees(np.ptp(neck, axis=0).max())),
        "neck_mean_abs_deg": float(np.degrees(np.abs(neck).mean())),
        "torso_range_deg": float(np.degrees(np.ptp(torso))),
        "torso_mean_deg": float(np.degrees(torso.mean())),
        "lift_range_mm": float(1000 * np.ptp(lift)),
        "lift_mean_minus_home_mm": float(1000 * (lift.mean() - r["lift_home"])),
        "base_travel_m": float(np.linalg.norm(base - base[0], axis=1).max()),
        "arm_range_deg": float(np.degrees(np.ptp(arm, axis=0).max())),
        "engaged_frac": len(R) / max(1, len(allR)),
        # COMMANDS (what the switch controls): a frozen part's command must sit
        # EXACTLY at home; the realized joint above also carries the servos'
        # compliance under the arms' reaction torques (sub-degree / ~1 mm).
        "cmd_neck_range_deg": float(np.degrees(np.ptp(np.array([x["c_neck"] for x in R]), axis=0).max())),
        "cmd_neck_max_abs_deg": float(np.degrees(np.abs(np.array([x["c_neck"] for x in R])).max())),
        "cmd_torso_range_deg": float(np.degrees(np.ptp([x["c_torso"] for x in R]))),
        "cmd_torso_max_abs_deg": float(np.degrees(np.abs([x["c_torso"] for x in R]).max())),
        "cmd_lift_range_mm": float(1000 * np.ptp([x["c_lift"] for x in R])),
        "cmd_lift_max_dev_mm": float(1000 * np.abs(np.array([x["c_lift"] for x in R]) - r["lift_home"]).max()),
    }


def peak_speed(r, t_switch, joint, window=3.0, avg=None):
    """Peak COMMANDED rate of a joint group after a switch, per WALL second
    (rad/s or m/s), averaged over ≥0.3 s of wall time. The followers apply ONE
    rate-limited step per OPERATOR FRAME, sized by that frame's wall interval; in
    this slowed-down loop a short window can catch a whole frame's step at once
    and read up to 3× the design speed (measured on the lift: 0.089 m/s in 50 ms
    windows vs 0.030 m/s = LIFT_FOLLOW_SPEED over the glide). The lift therefore
    averages over 1 s: a CPU stall (measured 2.3 s between samples while another
    build ran) makes ONE frame-quantized lift step and a 0.3 s window reads it as
    0.067 m/s; over 1 s the same run reads 0.0365 m/s. Neck/torso keep 0.3 s, so
    a genuine command STEP (what this check exists to catch) still shows."""
    R = [x for x in r["rec"] if t_switch - 0.05 <= x["t"] <= t_switch + window]
    key = "c_" + joint
    v = []
    j = 0
    for i in range(len(R)):
        span = avg if avg is not None else (1.0 if joint == "lift" else 0.30)
        while j < len(R) and R[j]["wall"] - R[i]["wall"] < span:
            j += 1
        if j >= len(R):
            break
        qa, qb = np.atleast_1d(R[i][key]), np.atleast_1d(R[j][key])
        v.append(float(np.max(np.abs(qb - qa)) / (R[j]["wall"] - R[i]["wall"])))
    return max(v) if v else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=14.0)
    ap.add_argument("--fig", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    results = {}
    t0 = 2.0 + 3.5                 # switch at 2 s + park 2.5 s + 1 s margin
    print(f"{'condition':<14} {'neck rng':>9} {'neck|μ|':>8} {'torso rng':>10} {'torso μ':>8} "
          f"{'lift rng':>9} {'lift-home':>10} {'base trav':>10} {'arm rng':>8} {'engaged':>8}")
    for name, cond in CONDITIONS.items():
        r = run(cond, seconds=a.seconds, label=name)
        s = summarize(r, t0, a.seconds)
        results[name] = {"cond": cond, "summary": s, "rec": r["rec"], "lift_home": r["lift_home"]}
        print(f"{name:<14} {s['neck_range_deg']:8.2f}° {s['neck_mean_abs_deg']:7.2f}° "
              f"{s['torso_range_deg']:9.2f}° {s['torso_mean_deg']:7.2f}° "
              f"{s['lift_range_mm']:7.1f}mm {s['lift_mean_minus_home_mm']:8.1f}mm "
              f"{s['base_travel_m']:9.3f}m {s['arm_range_deg']:7.1f}° {100*s['engaged_frac']:6.0f}%")

    # Bumpless sweep: whole → all frozen at 4 s → whole at 9 s.
    sweep = run(None, seconds=14.0, label="sweep",
                schedule=[(4.0, CONDITIONS["all frozen"]), (9.0, CONDITIONS["whole body"])])
    bump = {f"{j}@{ts:g}s": peak_speed(sweep, ts, j) for ts in (4.0, 9.0)
            for j in ("neck", "torso", "lift")}
    print("\nbumpless sweep (peak joint speed in the 3 s after each switch):")
    for k, v in bump.items():
        unit = "m/s" if k.startswith("lift") else "rad/s"
        print(f"  {k:<12} {v:.3f} {unit}")
    results["sweep"] = {"rec": sweep["rec"], "bump": bump}

    # CONTROL for the arms: with the BODY still (head + hands only), freezing the
    # neck/torso/lift must leave the arms tracking exactly as in whole body. (With
    # the body moving, a frozen torso/base legitimately pushes a leaning, stepping
    # operator's hands out of reach and the arm saturates — that is the ablation's
    # physical consequence, not a dead retargeter.)
    global STILL_BODY
    STILL_BODY = True
    still = {}
    for name in ("whole body", "all frozen"):
        r = run(CONDITIONS[name], seconds=a.seconds, label=name + " (still body)")
        still[name] = summarize(r, t0, a.seconds)["arm_range_deg"]
        print(f"still body, {name:<11}: arm range {still[name]:.1f}°")
    STILL_BODY = False
    results["still_body_arm_range"] = still

    # ── verdicts ──────────────────────────────────────────────────────────
    S = {k: v["summary"] for k, v in results.items() if "summary" in v}
    checks = [
        ("whole body: neck, torso, lift, base all MOVE",
         S["whole body"]["neck_range_deg"] > 10 and S["whole body"]["torso_range_deg"] > 2
         and S["whole body"]["lift_range_mm"] > 10 and S["whole body"]["base_travel_m"] > 0.2),
        ("freeze neck: COMMAND exactly home (|cmd| < 0.01°); realized within servo "
         "tolerance (range < 1°, |μ| < 1°)",
         S["freeze neck"]["cmd_neck_max_abs_deg"] < 0.01
         and S["freeze neck"]["neck_range_deg"] < 1 and S["freeze neck"]["neck_mean_abs_deg"] < 1),
        ("freeze torso: COMMAND exactly 0 (|cmd| < 0.01°); realized within servo tolerance "
         "(range < 1°, |μ| < 0.5°)",
         S["freeze torso"]["cmd_torso_max_abs_deg"] < 0.01
         and S["freeze torso"]["torso_range_deg"] < 1 and abs(S["freeze torso"]["torso_mean_deg"]) < 0.5),
        ("upper body: lift COMMAND exactly home (< 0.01 mm); realized lift within 3 mm, base "
         "within 2 cm",
         S["upper body"]["cmd_lift_max_dev_mm"] < 0.01
         and S["upper body"]["lift_range_mm"] < 3 and abs(S["upper body"]["lift_mean_minus_home_mm"]) < 3
         and S["upper body"]["base_travel_m"] < 0.02),
        ("freeze neck leaves torso/lift/base driven",
         S["freeze neck"]["torso_range_deg"] > 2 and S["freeze neck"]["lift_range_mm"] > 10
         and S["freeze neck"]["base_travel_m"] > 0.2),
        ("upper body leaves neck/torso driven",
         S["upper body"]["neck_range_deg"] > 10 and S["upper body"]["torso_range_deg"] > 2),
        # NOT a pass/fail on arm range: that max-over-joints statistic varies
        # ±30° between IDENTICAL runs of this wall-clocked loop (3 repeats each,
        # still body: 56/109/59° vs 48/31/93°), so it cannot resolve "same
        # retargeter". That is proven exactly by tests/test_ablation_same_
        # retargeter.py (parks write only their own actuators; one tick from an
        # identical state → bit-identical arm commands). Here: the arms MOVE.
        ("arms are retargeted in both (still body, range > 10°; equality: see "
         "tests/test_ablation_same_retargeter.py)",
         still["whole body"] > 10 and still["all frozen"] > 10),
        ("bumpless: commanded neck/torso ≤ 1.0 rad/s, lift ≤ LIFT_FOLLOW_SPEED×1.5 at every switch",
         all(v <= (0.045 if k.startswith("lift") else 1.0) for k, v in bump.items())),
    ]
    print()
    ok = True
    for name, passed in checks:
        ok &= bool(passed)
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    print("\nVERDICT:", "all ablation switches behave as labelled" if ok else "FAILURES above")

    if a.json:
        json.dump(results, open(a.json, "w"))
    if a.fig:
        plot(results, a.fig)
    return 0 if ok else 1


def plot(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [("neck", "neck yaw (deg)", lambda x: np.degrees(x["neck"][0])),
            ("torso", "torso (deg)", lambda x: np.degrees(x["torso"])),
            ("lift", "lift (mm)", lambda x: 1000 * x["lift"]),
            ("base", "base x (m)", lambda x: x["base"][0]),
            ("arm", "R arm j1 (deg)", lambda x: np.degrees(x["arm"][0]))]
    fig, axes = plt.subplots(len(rows), 1, figsize=(7.2, 8.4), sharex=True)
    for name in CONDITIONS:
        R = results[name]["rec"]
        t = [x["t"] for x in R]
        for ax, (_, lab, f) in zip(axes, rows):
            ax.plot(t, [f(x) for x in R], lw=1.2, label=name)
    for ax, (_, lab, _) in zip(axes, rows):
        ax.set_ylabel(lab, fontsize=8)
        ax.axvline(2.0, color="0.5", ls=":", lw=1)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, ncol=3, loc="upper right")
    axes[-1].set_xlabel("time (s) — switch applied at 2 s (dotted); park eases over 2.5 s")
    fig.tight_layout()
    fig.savefig(path)
    print(f"wrote {path}")


if __name__ == "__main__":
    sys.exit(main())
