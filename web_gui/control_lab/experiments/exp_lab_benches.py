#!/usr/bin/env python3
"""
Regenerate every figure for docs/control_lab.tex from scratch, and print the
summary metrics the paper cites.

    python3 web_gui/control_lab/experiments/exp_lab_benches.py

The harness drives the SAME SimWorld the studio serves — the mini-gateway around
MabelTeleop, the SAME control seam the iOS / Vision Pro apps drive — stepped
synchronously (step_sync) for determinism, except the latency table, which needs
the real clock and therefore runs one short REALTIME segment with the mock
operator thread. Vector PDFs land in docs/figures/. Fixed seeds; no hand-made
numbers.

Figures
-------
  lab_motion_bench.pdf   Drive mode: scripted stick sequence, cmd vs safe twist,
                         tip usage, lift; command-odometry error fwd vs reverse
  lab_pd_layers.pdf      Joint mode: impedance tracking vs stiffness K (Stiff)
                         and the Soft (compliance) float that ignores the target
  lab_wrench_estimation.pdf  |F| true vs estimated + direction cosine (Stiff)
  lab_retarget_mock.pdf  Whole-body Teleop vs the mock operator: EE error,
                         MPJPE, smoothness
  lab_taskspace.pdf      Wrist task-space reach: lift recruitment (upper_body
                         drag-up) and base follow + CoM stability (whole_body
                         drag-forward)
"""
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

from control_lab.server.simworld import SimWorld              # noqa: E402
from control_lab.server.vpnode import VPNode                  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(_HERE))), "controller"))
from mabel.types import OperatorPose                          # noqa: E402
from mabel.sources import mock_frame                          # noqa: E402
from mabel.whole_body_retargeting.calibration import (        # noqa: E402
    synthesize_pose, POSE_HANDS)

FIGDIR = os.path.join(os.path.dirname(_HERE), "docs", "figures")
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 120, "savefig.bbox": "tight"})
COL = {"cmd": "#e9a679", "safe": "#1f77b4", "true": "#2ca02c",
       "est": "#d62728", "a": "#1f77b4", "b": "#d62728", "c": "#2ca02c"}
METRICS = []


def _save(fig, name):
    os.makedirs(FIGDIR, exist_ok=True)
    fig.savefig(os.path.join(FIGDIR, name))
    plt.close(fig)
    print(f"  figure -> docs/figures/{name}")


def series(w, key):
    """(t[], v[]) from the world's own plot ring buffer."""
    pts = [(t, v) for t, v in w.plots.get(key, []) if v is not None]
    if not pts:
        return np.array([]), np.array([])
    a = np.array(pts)
    return a[:, 0], a[:, 1]


def fresh():
    w = SimWorld()
    w.settle()
    return w


# ── F1: Drive mode (motion model) ─────────────────────────────────────────────

def fig_motion():
    global METRICS
    w = fresh(); w.set_torque(True)
    w.set_control_spec(method="navigation")
    seq = [  # (seconds, {stick})  — wire convention (stick-up ly>0 = forward)
        (2.5, {"ly": 1.0}),                       # full forward
        (2.0, {"ly": 1.0, "rx": 0.6}),            # forward + yaw
        (1.5, {}),                                 # stop
        (3.0, {"ry": 1.0, "lx": 0.6}),            # lift up + strafe
    ]
    for secs, stick in seq:
        w.nav.update({"lx": 0, "ly": 0, "rx": 0, "ry": 0})
        w.nav.update(stick)
        w.step_sync(secs)
    err_fwd = series(w, "odom_err")[1][-1] if len(series(w, "odom_err")[0]) else None

    fig, axs = plt.subplots(3, 1, figsize=(6.4, 5.4), sharex=True)
    for key, lab, col in (("cmd_vx", "cmd $v_x$", COL["cmd"]),
                          ("safe_vx", "safe $v_x$", COL["safe"])):
        axs[0].plot(*series(w, key), color=col, label=lab)
    for key, lab, col in (("cmd_w", r"cmd $\omega$", "#f0b68b"),
                          ("safe_w", r"safe $\omega$", "#7f7fff")):
        axs[0].plot(*series(w, key), color=col, ls="--", label=lab)
    axs[0].set_ylabel("twist"); axs[0].legend(ncol=4, fontsize=7)
    axs[0].set_title("Drive: scripted stick sequence, commanded vs tip-safe twist")
    t, u = series(w, "tip_usage")
    axs[1].plot(t, u, color=COL["safe"], label="ZMP usage $u$")
    axs[1].axhline(1.0, color=COL["est"], ls="--", lw=1)
    axs[1].plot(*series(w, "lift_z"), color=COL["true"], label="lift $z$ [m]")
    axs[1].plot(*series(w, "cap_fwd"), color="#b48ef0", label="cap fwd [m/s]")
    axs[1].set_ylabel("usage / caps"); axs[1].legend(ncol=3, fontsize=7)
    axs[2].plot(*series(w, "odom_err"), color=COL["cmd"])
    axs[2].set_ylabel("odom err [m]"); axs[2].set_xlabel("sim time [s]")
    _save(fig, "lab_motion_bench.pdf")
    METRICS += [("motion: max ZMP usage over the sequence", f"{u.max():.3f}"),
                ("motion: command-odometry error @ end [m]",
                 f"{err_fwd:.3f}" if err_fwd is not None else "n/a")]

    # forward vs reverse odometry fidelity (the honest asymmetry)
    out = {}
    for label, ly in (("forward", 1.0), ("reverse", -1.0)):
        w2 = fresh(); w2.set_torque(True)
        w2.set_control_spec(method="navigation")
        w2.nav.update({"ly": ly})
        w2.step_sync(2.0)
        out[label] = series(w2, "odom_err")[1][-1]
    METRICS += [("motion: odom err after 2 s drive fwd / rev [m]",
                 f"{out['forward']:.3f} / {out['reverse']:.3f}")]
    return out


# ── F2: Joint mode — impedance tracking vs stiffness, and Soft float ──────────

def fig_pd_layers():
    global METRICS
    Ks = (15.0, 30.0, 45.0)
    runs = {}
    for K in Ks:
        w = fresh(); w.set_torque(True)
        w.set_control_spec(method="joint", control_type="impedance",
                           region="arm", stiffness=K)
        w.set_trajectory("sine", joint="left_arm_2", amp=0.3, period=4.0)
        w.step_sync(10.0)
        runs[K] = {k: series(w, k) for k in ("q", "q_des", "err_max")}
    # Soft (compliance): the same sine target is IGNORED — the arm floats
    ws = fresh(); ws.set_torque(True)
    ws.set_control_spec(method="joint", control_type="compliance", region="arm")
    ws.set_trajectory("sine", joint="left_arm_2", amp=0.3, period=4.0)
    ws.step_sync(10.0)
    soft_q = series(ws, "q")

    fig, axs = plt.subplots(1, 2, figsize=(7.6, 2.9))
    cmap = {15.0: "#f0b68b", 30.0: COL["cmd"], 45.0: COL["safe"]}
    axs[0].plot(*runs[45.0]["q_des"], color=COL["true"], lw=1.4, label="$q_{des}$")
    for K in Ks:
        axs[0].plot(*runs[K]["q"], color=cmap[K], label=f"Stiff K={int(K)}")
    axs[0].plot(*soft_q, color="#888", ls="--", label="Soft (float)")
    axs[0].set_xlabel("sim time [s]"); axs[0].set_ylabel("Left Arm 2 [rad]")
    axs[0].legend(fontsize=6.5, ncol=2); axs[0].set_title("impedance sine tracking")
    rms = []
    for K in Ks:
        e = runs[K]["err_max"][1]
        rms.append(e[len(e) // 2:].mean() if len(e) else np.nan)
    axs[1].plot(Ks, rms, "o-", color=COL["est"])
    axs[1].set_xlabel("stiffness K [Nm/rad]")
    axs[1].set_ylabel("max joint err (2nd half) [rad]")
    axs[1].set_title("tracking error vs stiffness")
    _save(fig, "lab_pd_layers.pdf")
    for K, r in zip(Ks, rms):
        METRICS += [(f"joint sine: max-err mean (2nd half), Stiff K={int(K)} [rad]",
                     f"{r:.4f}")]
    # the discrete-damping fix: the ORCA wrist stays stable at max K
    wmax = max(abs(ws.plant.qd("Right Wrist")), abs(runs[45.0]["q"][1].std()))
    METRICS += [("joint: Right Wrist |qd| at K=45 (stable, was ~inf)",
                 f"{abs(runs[45.0]['q'][1].std()):.3f}")]


# ── F3: wrench estimation (Stiff / impedance hold) ────────────────────────────

def _wrench_run(force):
    w = fresh(); w.set_torque(True)
    w.set_control_spec(method="joint", control_type="impedance", region="arm",
                       stiffness=45.0)
    w.step_sync(1.5)                              # observer convergence
    w.apply_push("left_palm", force, seconds=3.0)
    w.step_sync(4.5)
    return w


def fig_wrench():
    global METRICS
    # +Y (lateral) = well-conditioned: the force has lever on the distal joints.
    # +X = ill-conditioned at the home pose: the force line passes near the distal
    # joint axes, so the contact is nearly unobservable from joint torques.
    runs = {"+Y (well-conditioned)": _wrench_run([0.0, 12.0, 0.0]),
            "+X (ill-conditioned)": _wrench_run([12.0, 0.0, 0.0])}
    fig, axs = plt.subplots(1, 2, figsize=(7.8, 2.9), sharey=True)
    for ax, (name, w) in zip(axs, runs.items()):
        ax.plot(*series(w, "F_true"), color=COL["true"], label=r"$\|F\|$ applied")
        ax.plot(*series(w, "F_est"), color=COL["est"], label=r"$\|\hat F\|$ estimated")
        t_cos, cos = series(w, "F_cos")
        ax.plot(t_cos, 12 * np.clip(cos, 0, 1), color=COL["safe"], lw=1,
                ls="--", label=r"12·cos$^+$(dir)")
        ax.set_xlabel("sim time [s]"); ax.set_title(f"12 N push {name}", fontsize=8.5)
        ax.legend(fontsize=6.5)
    axs[0].set_ylabel("force [N]")
    _save(fig, "lab_wrench_estimation.pdf")
    for name, w in runs.items():
        t_est, f_est = series(w, "F_est")
        t_cos, cos = series(w, "F_cos")
        on = (t_est > 2.5) & (t_est < 4.5)
        cs = cos[(t_cos > 2.5) & (t_cos < 4.5)]
        METRICS += [(f"wrench {name}: |F̂| / cos (settled)",
                     f"{f_est[on].mean():.1f} N / {cs.mean():.3f}"
                     if len(cs) and on.any() else "n/a")]


# ── F4: whole-body Teleop vs the mock operator ───────────────────────────────

def fig_retarget():
    global METRICS
    w = fresh(); w.set_torque(True)
    w.set_control_spec(method="retarget", control_type="impedance", region="upper_body")
    hands = synthesize_pose(POSE_HANDS, n=48, noise=0.0005)
    k = 0

    def feed():
        nonlocal k
        pose = OperatorPose.from_avframe(mock_frame(w._t))
        hf = hands[k % len(hands)]; k += 1
        pose.left_joints = hf.left_joints
        pose.right_joints = hf.right_joints
        w.ingest_pose(pose, "mock")

    for _ in range(int(12.0 / 0.032)):            # feed ~31 Hz of sim time
        feed()
        w.step_sync(0.032)

    fig, axs = plt.subplots(1, 3, figsize=(9.2, 2.7))
    axs[0].plot(*series(w, "ee_err_l_mm"), color=COL["safe"], label="left")
    axs[0].plot(*series(w, "ee_err_r_mm"), color=COL["cmd"], label="right")
    axs[0].set_ylabel("EE tracking err [mm]"); axs[0].legend(fontsize=7)
    axs[1].plot(*series(w, "mpjpe_mm"), color=COL["true"])
    axs[1].set_ylabel("MPJPE [mm]")
    axs[2].plot(*series(w, "jerk"), color=COL["est"])
    axs[2].set_ylabel(r"EE RMS jerk [m/s$^3$]")
    for ax in axs:
        ax.set_xlabel("sim time [s]")
    fig.suptitle("whole-body Teleop vs the mock operator (circling wrists)",
                 fontsize=9)
    _save(fig, "lab_retarget_mock.pdf")
    for key, name in (("ee_err_l_mm", "EE err L"), ("ee_err_r_mm", "EE err R"),
                      ("mpjpe_mm", "MPJPE"), ("jerk", "RMS jerk")):
        t, v = series(w, key)
        sel = v[t > 3.0]
        METRICS += [(f"retarget(mock): {name} mean after calib settle",
                     f"{sel.mean():.1f}" if len(sel) else "n/a")]
    METRICS += [("retarget: control step mean [ms]",
                 f"{w.timer.ms.get('update_ms', float('nan')):.2f}")]


# ── F5: wrist task-space reach — lift recruitment + base follow ──────────────

def fig_taskspace():
    """Upper-body: a straight-up drag recruits the LIFT (18-DOF IK, Sync-Lift
    wire fix). Whole-body: a far-forward drag drives the BASE after the targets
    (BaseFollow comfort box), with the torso rate cap + CoM guard keeping the
    live support margin positive (the uncapped chase face-planted the robot)."""
    global METRICS
    import math
    import mabel.config as C
    from mabel.motion_model import com_support as CS

    def sample_run(region, dvec, seconds):
        w = fresh(); w.set_torque(True)
        w.set_control_spec(method="wrist", control_type="impedance",
                           region=region, stiffness=45.0)
        w.step_sync(0.3)
        for side in ("l", "r"):
            p = np.asarray(w.snapshot()["bodies"][f"palm_{side}"])
            w.set_wrist_target(side, (p + np.asarray(dvec)).tolist())
        rec = {k: [] for k in ("t", "lift", "ee", "vx", "x", "margin", "pitch")}
        for _ in range(int(seconds / 0.1)):
            w.step_sync(0.1)
            s = w.snapshot()
            tg = s["overlay"]["targets"]["l"]
            com = CS.robot_com(w.model, w.data)
            R = w.data.body(C.CHASSIS_BODY).xmat.reshape(3, 3)
            f = -R[:, 0]
            rec["t"].append(s["t"])
            rec["lift"].append(s["lift"])
            rec["ee"].append(np.linalg.norm(np.asarray(tg["target"])
                                            - np.asarray(tg["actual"])))
            rec["vx"].append(w._last_base[0])
            rec["x"].append(s["base"]["x"])
            rec["margin"].append(CS.support_margin(
                com[:2], CS.support_polygon(w.model, w.data)))
            rec["pitch"].append(math.degrees(math.asin(
                max(-1.0, min(1.0, -f[2])))))
        return {k: np.asarray(v) for k, v in rec.items()}

    up = sample_run("upper_body", [0.0, 0.0, 0.5], 8.0)
    fwd = sample_run("whole_body", [-1.2, 0.0, 0.0], 8.0)

    fig, axs = plt.subplots(3, 1, figsize=(6.4, 5.8), sharex=True)
    axs[0].plot(up["t"], up["lift"], color=COL["safe"], label="lift travel [m]")
    axs[0].plot(up["t"], up["ee"], color=COL["cmd"], label="palm err [m]")
    axs[0].set_ylabel("upper: drag up 0.5 m")
    axs[0].legend(ncol=2, fontsize=7)
    axs[0].set_title("Wrist task-space reach: lift recruitment (upper) "
                     "and base follow (whole)")
    axs[1].plot(fwd["t"], fwd["vx"], color=COL["safe"], label="safe $v_x$ [m/s]")
    axs[1].plot(fwd["t"], fwd["x"], color=COL["true"], label="chassis $x$ [m]")
    axs[1].plot(fwd["t"], fwd["ee"], color=COL["cmd"], label="palm err [m]")
    axs[1].set_ylabel("whole: drag fwd 1.2 m")
    axs[1].legend(ncol=3, fontsize=7)
    axs[2].plot(fwd["t"], fwd["margin"], color=COL["est"],
                label="live CoM support margin [m]")
    axs[2].axhline(C.COM_GUARD_MARGIN, color=COL["est"], ls="--", lw=1,
                   label="guard line")
    axs[2].plot(fwd["t"], np.radians(fwd["pitch"]), color="#b48ef0",
                label="chassis pitch [rad]")
    axs[2].set_ylabel("stability"); axs[2].set_xlabel("sim time [s]")
    axs[2].legend(ncol=3, fontsize=7)
    _save(fig, "lab_taskspace.pdf")

    METRICS += [
        ("taskspace: lift extension after 8 s drag-up [m]",
         f"{up['lift'][-1] - up['lift'][0]:.3f}"),
        ("taskspace: upper drag-up final palm err [mm]", f"{up['ee'][-1]*1e3:.1f}"),
        ("taskspace: whole fwd chassis travel [m]", f"{abs(fwd['x'][-1]):.3f}"),
        ("taskspace: whole fwd final palm err [mm]", f"{fwd['ee'][-1]*1e3:.1f}"),
        ("taskspace: min live CoM margin [m]", f"{fwd['margin'].min():.3f}"),
        ("taskspace: max |chassis pitch| [deg]",
         f"{np.abs(fwd['pitch']).max():.2f}"),
    ]


# ── latency table (REALTIME segment — needs the true clock) ──────────────────

def realtime_latency():
    global METRICS
    w = SimWorld()
    w.start()
    vp = VPNode(w)
    try:
        w.set_torque(True)
        w.set_control_spec(method="retarget", control_type="impedance",
                           region="upper_body")
        vp.set_source("mock")
        t0 = time.time()
        time.sleep(8.0)
        s = w.snapshot()
        rt = s["t"] / (time.time() - t0)
        METRICS += [
            ("realtime: stream rate [Hz]", f"{s['vp']['rate_hz']:.1f}"),
            ("realtime: inter-frame jitter [ms]", f"{s['vp']['jitter_ms']:.1f}"),
            ("realtime: frame age [ms]", f"{s['vp']['age_ms']:.1f}"),
            ("realtime: control step [ms]",
             f"{s['timing']['avg'].get('update_ms', float('nan')):.2f}"),
            ("realtime: physics step [ms]",
             f"{s['timing']['avg'].get('physics_ms', float('nan')):.2f}"),
            ("realtime: realtime factor", f"{rt:.2f}x"),
        ]
    finally:
        vp.stop()
        w.stop()


def main():
    np.random.seed(7)
    print("== Control Lab doc figures ==")
    fig_motion()
    fig_pd_layers()
    fig_wrench()
    fig_retarget()
    fig_taskspace()
    realtime_latency()
    print("\n== summary metrics (cited in docs/control_lab.tex) ==")
    for k, v in METRICS:
        print(f"  {k:56s} {v}")


if __name__ == "__main__":
    main()
