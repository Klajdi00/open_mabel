#!/usr/bin/env python3
"""Measure the two retargeting modes on synthetic ground truth.

    python3 controller/experiments/exp_retarget_modes.py

Everything here is deterministic and headless; figures land in
`controller/docs/figures/retarget_*.pdf` and every number quoted in
`controller/docs/retargeting_modes.tex` comes from this file's summary table.

  E1  VEHICLE INVARIANCE (the headline). Same operator, same task, run twice —
      once with an ARKit-anchor drift superposed. Local mode must not notice.
  E2  TORSO-FIT CONDITIONING. Where in the workspace eq. (14) is informative,
      and where the estimate is carried by continuity instead.
  E3  HANDOFF. Does a mode switch move the robot, and how does the base leave?
  E4  FIDELITY. What local mode costs when the world estimate is FINE — plus
      what eq. (17)'s position blend does that the shipped magnitude blend does
      not, since that is the other change riding along.
  E5  NECK DRIFT. Local mode's gaze comes from the torso fit, so it cannot
      inherit a drifting heading; the world-anchored reference does, 1:1.

There is no hardware in the loop and none is claimed: a synthetic operator is
the only way to superpose a repeatable vehicle trajectory on a repeatable
bimanual task. What it validates is the FRAME ALGEBRA and the estimator, which
is exactly where the mode split lives.
"""
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from mabel.retargeting import (BodyRetargeter, LocalTorsoFit, OperatorModel,   # noqa: E402
                               RetargetMode, WorldMeasurements,
                               device_measurements)
from mabel.retargeting.frames import log_so3, smoothstep                       # noqa: E402
from mabel.retargeting.local_frame import _R_from_ypr, _ypr_from_R             # noqa: E402
from mabel.retargeting.synthetic import (SyntheticOperator, VehicleDrift,      # noqa: E402
                                         mabel_like_robot)

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
FIGS = os.path.join(REPO, "controller", "docs", "figures")
os.makedirs(FIGS, exist_ok=True)

DT = 1.0 / 60.0
plt.rcParams.update({"figure.dpi": 120, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 9, "legend.frameon": False})
ACCENT, WARN, INK, BLUE = "#c25b2a", "#e9a679", "#2b2724", "#5eb0ff"

RESULTS = {}


# ── the shared task ──────────────────────────────────────────────────────────
def task(op, t, rho=0.75):
    """A repeatable bimanual task: both hands trace a circle in front of the
    torso. Fixed relative to the TORSO, so any change in the robot targets is
    the retargeter's doing, never the operator's."""
    a = 2.0 * np.pi * 0.25 * t
    for side, s in (("right", 1.0), ("left", -1.0)):
        op.wrist[side] = (rho, np.array([s * 0.35 + 0.15 * np.cos(a),
                                         -0.2 + 0.15 * np.sin(a), 0.9]))


def run(mode, drift=None, seconds=8.0, dt=DT, switch=None, rho=0.75, op=None):
    """Drive the retargeter; return dict of per-frame arrays."""
    op = op or SyntheticOperator()
    rt = BodyRetargeter(mode=mode)
    robot = mabel_like_robot()
    rec = {"t": [], "L": [], "R": [], "twist": [], "switched": [],
           "conf": [], "R_bo": [], "lift": []}
    for i in range(int(seconds / dt)):
        t = i * dt
        task(op, t, rho)
        if switch is not None and abs(t - switch) < 0.5 * dt:
            rt.set_mode(RetargetMode.GLOBAL if rt.mode == RetargetMode.LOCAL
                        else RetargetMode.LOCAL)
        T_ad, hands, skel = op.frame(drift(t) if drift else None)
        out = rt.update(device_measurements(T_ad, hands, skel, timestamp=t),
                        WorldMeasurements(T_ad), robot, dt)
        if not out.valid or len(out.wrist_targets) < 2:
            continue
        rec["t"].append(t)
        rec["L"].append(out.wrist_targets["left"])
        rec["R"].append(out.wrist_targets["right"])
        rec["twist"].append(out.base_twist)
        rec["switched"].append(bool(out.diag.get("switched")))
        rec["conf"].append(out.confidence)
        rec["R_bo"].append(out.R_bo)
        rec["lift"].append(out.lift_rate if out.lift_rate is not None else np.nan)
    return {k: np.array(v) for k, v in rec.items()}


def _pair_err(a, b):
    """Per-frame wrist-target error between two runs [m]."""
    n = min(len(a["t"]), len(b["t"]))
    dl = np.linalg.norm(a["L"][:n] - b["L"][:n], axis=1)
    dr = np.linalg.norm(a["R"][:n] - b["R"][:n], axis=1)
    return a["t"][:n], np.maximum(dl, dr)


# ── E1 vehicle invariance ────────────────────────────────────────────────────
def e1_vehicle_invariance():
    print("\nE1  VEHICLE INVARIANCE")
    print("    the operator sits still relative to the cabin; the ANCHOR drifts")
    cases = [
        ("gentle  (5 deg/s, 2 deg tilt)",
         VehicleDrift(yaw_rate=np.deg2rad(5), tilt_amp=np.deg2rad(2),
                      drift_vel=(0.005, 0.0, 0.002))),
        ("cruise  (20 deg/s, 6 deg tilt)",
         VehicleDrift(yaw_rate=np.deg2rad(20), tilt_amp=np.deg2rad(6),
                      drift_vel=(0.02, 0.0, 0.01))),
        ("harsh   (45 deg/s, 12 deg tilt)",
         VehicleDrift(yaw_rate=np.deg2rad(45), tilt_amp=np.deg2rad(12),
                      drift_vel=(0.05, 0.0, 0.02))),
    ]
    base = {m: run(m) for m in ("local", "global")}
    fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.1))
    rows = []
    for k, (label, veh) in enumerate(cases):
        for mode, color, ls in (("local", ACCENT, "-"), ("global", BLUE, "--")):
            t, e = _pair_err(base[mode], run(mode, veh))
            rows.append((label, mode, float(e.mean()), float(e.max())))
            if k == 1:
                ax[0].plot(t, e * 1000, ls, color=color, lw=1.4, label=mode)
        ax[1].bar(k - 0.18, rows[-2][3] * 1000, 0.34, color=ACCENT,
                  label="local" if k == 0 else None)
        ax[1].bar(k + 0.18, rows[-1][3] * 1000, 0.34, color=BLUE,
                  label="global" if k == 0 else None)
    ax[0].set(xlabel="time [s]", ylabel="wrist-target error [mm]",
              title="cruise: 20 deg/s anchor yaw, 6 deg tilt")
    ax[0].set_yscale("symlog", linthresh=1e-9)
    ax[0].legend()
    ax[1].set(xticks=range(3), xticklabels=["gentle", "cruise", "harsh"],
              ylabel="max error [mm]", title="peak target error vs baseline")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "retarget_vehicle_invariance.pdf"))
    plt.close(fig)

    print(f"    {'vehicle':<28} {'mode':<8} {'mean [mm]':>12} {'max [mm]':>12}")
    for label, mode, mean, mx in rows:
        print(f"    {label:<28} {mode:<8} {mean*1000:>12.6f} {mx*1000:>12.6f}")
    loc = [r for r in rows if r[1] == "local"]
    glo = [r for r in rows if r[1] == "global"]
    RESULTS["E1_local_max_mm"] = max(r[3] for r in loc) * 1000
    RESULTS["E1_global_max_mm"] = max(r[3] for r in glo) * 1000
    print(f"    -> local  worst {RESULTS['E1_local_max_mm']:.3e} mm "
          f"(machine precision: the anchor cancels exactly)")
    print(f"    -> global worst {RESULTS['E1_global_max_mm']:.1f} mm")


# ── E2 torso-fit conditioning ────────────────────────────────────────────────
def e2_conditioning():
    print("\nE2  TORSO-FIT CONDITIONING (eq. 14)")
    print("    where the data orients the torso, and where continuity carries it")
    rhos = np.linspace(0.30, 1.0, 8)
    yaws = np.arange(-70, 71, 20.0)
    err = np.zeros((len(yaws), len(rhos)))
    conf = np.zeros_like(err)
    for i, y in enumerate(yaws):
        for j, r in enumerate(rhos):
            op = SyntheticOperator()
            op.set_reach(r)
            op.neck = (np.deg2rad(y), 0.0, 0.0)
            fit = LocalTorsoFit(OperatorModel())
            for _ in range(90):
                T_ad, hands, skel = op.frame()
                fit.update(device_measurements(T_ad, hands, skel), DT)
            err[i, j] = abs(np.rad2deg(_ypr_from_R(fit.R_do.T))[0] - y)
            conf[i, j] = fit.confidence

    # the wrist-only ablation (the paper-literal residual), at one yaw
    abl_err, abl_conf = [], []
    for r in rhos:
        op = SyntheticOperator()
        op.set_reach(r)
        op.neck = (np.deg2rad(30.0), 0.0, 0.0)
        fit = LocalTorsoFit(OperatorModel(), use_elbow=False)
        for _ in range(90):
            T_ad, hands, skel = op.frame(elbows=False)
            fit.update(device_measurements(T_ad, hands, skel), DT)
        abl_err.append(abs(np.rad2deg(_ypr_from_R(fit.R_do.T))[0] - 30.0))
        abl_conf.append(fit.confidence)

    fig, ax = plt.subplots(1, 3, figsize=(11.0, 3.0))
    im = ax[0].imshow(err, aspect="auto", origin="lower", cmap="magma_r",
                      extent=[rhos[0], rhos[-1], yaws[0], yaws[-1]], vmin=0, vmax=12)
    fig.colorbar(im, ax=ax[0], label="yaw error [deg]")
    ax[0].set(xlabel=r"arm extension $\rho$", ylabel="true torso yaw [deg]",
              title="accuracy")
    ax[0].grid(False)
    im = ax[1].imshow(conf, aspect="auto", origin="lower", cmap="viridis",
                      extent=[rhos[0], rhos[-1], yaws[0], yaws[-1]])
    fig.colorbar(im, ax=ax[1], label="confidence")
    ax[1].set(xlabel=r"arm extension $\rho$", title="observability")
    ax[1].grid(False)
    ax[2].plot(rhos, err.mean(axis=0), "o-", color=ACCENT, lw=1.4,
               label="wrist + elbow")
    ax[2].plot(rhos, abl_err, "s--", color=BLUE, lw=1.4, label="wrist only (14 literal)")
    ax[2].set(xlabel=r"arm extension $\rho$", ylabel="yaw error [deg]",
              title="the elbow residual's contribution")
    ax[2].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "retarget_conditioning.pdf"))
    plt.close(fig)

    RESULTS["E2_err_mean_deg"] = float(err.mean())
    RESULTS["E2_err_max_deg"] = float(err.max())
    RESULTS["E2_conf_min"] = float(conf.min())
    RESULTS["E2_ablation_mean_deg"] = float(np.mean(abl_err))
    print(f"    yaw error over the sweep: mean {err.mean():.2f} deg, "
          f"max {err.max():.2f} deg")
    print(f"    confidence: {conf.min():.3f} .. {conf.max():.3f} "
          f"(floor {LocalTorsoFit(OperatorModel()).conf_floor})")
    print(f"    wrist-only ablation at 30 deg: {np.mean(abl_err):.2f} deg mean "
          f"vs {err[len(yaws)//2 + 2].mean():.2f} with the elbow")


# ── E3 handoff ───────────────────────────────────────────────────────────────
def e3_handoff():
    print("\nE3  MODE HANDOFF (eq. 15's alignment latch)")
    fig, ax = plt.subplots(1, 3, figsize=(11.0, 3.0))
    rows = []
    for k, start in enumerate(("local", "global")):
        r = run(start, seconds=8.0, switch=4.0)
        i = int(np.argmax(r["switched"]))
        jump = max(np.linalg.norm(r["L"][i] - r["L"][i - 1]),
                   np.linalg.norm(r["R"][i] - r["R"][i - 1]))
        step = max(np.linalg.norm(r["L"][i - 1] - r["L"][i - 2]),
                   np.linalg.norm(r["R"][i - 1] - r["R"][i - 2]))
        rows.append((start, jump, step))
        ax[k].plot(r["t"], r["R"][:, 0], color=ACCENT, lw=1.2, label="x")
        ax[k].plot(r["t"], r["R"][:, 1], color=BLUE, lw=1.2, label="y")
        ax[k].plot(r["t"], r["R"][:, 2], color=INK, lw=1.2, label="z")
        ax[k].axvline(r["t"][i], color=WARN, lw=1.0, ls="--")
        ax[k].set(xlabel="time [s]", ylabel="right wrist target [m]",
                  title=f"{start} -> {'global' if start=='local' else 'local'}")
        ax[k].legend(ncol=3, fontsize=7)
        if start == "global":
            ax[2].plot(r["t"] - r["t"][i], np.linalg.norm(r["twist"], axis=1),
                       color=ACCENT, lw=1.5)
    ax[2].set(xlabel="time since switch [s]", ylabel=r"$|\nu_b^{cmd}|$",
              title="base ramps out, never cuts", xlim=(-1.0, 3.0))
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "retarget_handoff.pdf"))
    plt.close(fig)

    for start, jump, step in rows:
        print(f"    {start:>6} -> {'global' if start=='local' else 'local':<7}"
              f" target jump {jump*1000:8.4f} mm   "
              f"(one frame of task motion: {step*1000:.3f} mm)")
    RESULTS["E3_jump_max_mm"] = max(r[1] for r in rows) * 1000


# ── E4 fidelity ──────────────────────────────────────────────────────────────
def _shipped_magnitude_blend(p, s, mu_o, robot, model, R_bo, side, gate=0.85):
    """The blend the SHIPPED engine uses: GMR position, then the MAGNITUDE about
    the robot's own shoulder renormalised toward the extension radius near
    rho = 1. Eq. (17) blends the POSITIONS instead. Reproduced here (not
    imported) so the comparison needs no MuJoCo."""
    r = p - s
    d = float(np.linalg.norm(r))
    L_o = model.arm_length(side)
    rho = min(d / L_o, 1.0)
    lam = smoothstep(rho, gate, 1.0)
    s_r, L_b = robot.shoulders[side], robot.reach[side]
    t_raw = robot.root + (L_b / L_o) * (R_bo @ (p - mu_o))
    dv = t_raw - s_r
    dn = float(np.linalg.norm(dv))
    mag = (1.0 - lam) * dn + lam * (rho * L_b)
    return s_r + dv / dn * mag


def e4_fidelity():
    print("\nE4  FIDELITY  (what local mode costs when F_a is FINE)")
    rhos = np.linspace(0.30, 1.0, 15)
    out = {m: {"ext": [], "shape": []} for m in ("local", "global", "shipped")}
    from mabel.retargeting.types import BodyFrameEstimate
    from mabel.retargeting.wrist_targets import wrist_targets
    model, robot = OperatorModel(), mabel_like_robot()
    for rho in rhos:
        op = SyntheticOperator()
        for mode in ("local", "global"):
            r = run(mode, seconds=5.0, rho=rho, op=SyntheticOperator())
            n = len(r["t"]) // 2
            L_b = robot.reach["right"]
            e = np.linalg.norm(r["R"][n:] - robot.shoulders["right"], axis=1) / L_b
            sep = np.linalg.norm(r["R"][n:] - r["L"][n:], axis=1) / L_b
            out[mode]["ext"].append(float(np.mean(np.abs(e - rho))))
            # operator's own normalised separation, for the shape metric
            op2 = SyntheticOperator()
            task(op2, r["t"][n], rho)
            hsep = (np.linalg.norm(op2.hand("right")[:3, 3] - op2.hand("left")[:3, 3])
                    / model.arm_length("right"))
            out[mode]["shape"].append(float(abs(np.mean(sep) - hsep)))
        # the shipped magnitude blend, on the same geometry
        op2 = SyntheticOperator()
        task(op2, 0.0, rho)
        mu_o = op2.root
        est = BodyFrameEstimate(basis=op2.torso_basis, root=mu_o, valid=True,
                                shoulders={s: op2.shoulder(s) for s in ("left", "right")})
        R_bo = robot.basis @ op2.torso_basis.T
        pts = {s: op2.hand(s)[:3, 3] for s in ("left", "right")}
        t_pos, _ = wrist_targets(pts, est, robot, model, R_bo)
        t_mag = {s: _shipped_magnitude_blend(pts[s], est.shoulders[s], mu_o,
                                             robot, model, R_bo, s)
                 for s in ("left", "right")}
        L_b = robot.reach["right"]
        out["shipped"]["ext"].append(
            abs(np.linalg.norm(t_mag["right"] - robot.shoulders["right"]) / L_b - rho))
        hsep = np.linalg.norm(pts["right"] - pts["left"]) / model.arm_length("right")
        out["shipped"]["shape"].append(
            abs(np.linalg.norm(t_mag["right"] - t_mag["left"]) / L_b - hsep))

    fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.0))
    for mode, color, ls in (("local", ACCENT, "-"), ("global", BLUE, "--"),
                            ("shipped", INK, ":")):
        ax[0].plot(rhos, out[mode]["ext"], ls, color=color, lw=1.5, label=mode)
        ax[1].plot(rhos, out[mode]["shape"], ls, color=color, lw=1.5, label=mode)
    ax[0].set(xlabel=r"operator extension $\rho$",
              ylabel=r"$|\rho_{robot}-\rho_{human}|$", title="extension fidelity")
    ax[1].set(xlabel=r"operator extension $\rho$",
              ylabel="normalised separation error", title="bimanual shape fidelity")
    for a in ax:
        a.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "retarget_fidelity.pdf"))
    plt.close(fig)

    for mode in ("local", "global", "shipped"):
        print(f"    {mode:<8} extension err {np.mean(out[mode]['ext']):.4f}   "
              f"shape err {np.mean(out[mode]['shape']):.4f}")
    RESULTS["E4_local_ext"] = float(np.mean(out["local"]["ext"]))
    RESULTS["E4_global_ext"] = float(np.mean(out["global"]["ext"]))
    RESULTS["E4_local_shape"] = float(np.mean(out["local"]["shape"]))
    RESULTS["E4_global_shape"] = float(np.mean(out["global"]["shape"]))


# ── E5 neck drift ────────────────────────────────────────────────────────────
def e5_neck_drift():
    print("\nE5  NECK / GAZE UNDER AN ANCHOR HEADING DRIFT")
    veh = VehicleDrift(yaw_rate=np.deg2rad(20))
    op = SyntheticOperator()
    op.set_reach(0.75)
    fit = LocalTorsoFit(OperatorModel())
    ts, local_err, world_err = [], [], []
    head0 = None
    for i in range(int(10.0 / DT)):
        t = i * DT
        task(op, t)
        neck_true = np.deg2rad(25.0) * np.sin(2 * np.pi * 0.1 * t)
        op.neck = (neck_true, 0.0, 0.0)
        T_ad, hands, skel = op.frame(veh(t))
        fit.update(device_measurements(T_ad, hands, skel), DT)
        if head0 is None:
            head0 = np.asarray(T_ad, float)[:3, :3]
            continue
        # LOCAL: head relative to the fitted torso — no world in the chain
        local_yaw = _ypr_from_R(fit.neck_rotation())[0]
        # WORLD-ANCHORED (what the engine does today): head relative to a latched
        # world reference, so the anchor's drift enters the neck command 1:1
        world_yaw = _ypr_from_R(head0.T @ np.asarray(T_ad, float)[:3, :3])[0]
        ts.append(t)
        local_err.append(np.rad2deg(local_yaw - neck_true))
        world_err.append(np.rad2deg(world_yaw - neck_true))

    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    ax.plot(ts, world_err, "--", color=BLUE, lw=1.5, label="world-anchored reference")
    ax.plot(ts, local_err, "-", color=ACCENT, lw=1.5, label=r"local: $\hat{R}_{do}^{-1}$")
    ax.set(xlabel="time [s]", ylabel="neck yaw error [deg]",
           title="20 deg/s anchor heading drift")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "retarget_neck_drift.pdf"))
    plt.close(fig)

    RESULTS["E5_local_final_deg"] = float(abs(local_err[-1]))
    RESULTS["E5_world_final_deg"] = float(abs(world_err[-1]))
    print(f"    after 10 s: local {abs(local_err[-1]):6.2f} deg   "
          f"world-anchored {abs(world_err[-1]):6.2f} deg")


# ── timing ───────────────────────────────────────────────────────────────────
def e6_timing():
    print("\nE6  COST PER TICK")
    op = SyntheticOperator()
    rt = BodyRetargeter(mode="local")
    robot = mabel_like_robot()
    T_ad, hands, skel = op.frame()
    dev = device_measurements(T_ad, hands, skel)
    w = WorldMeasurements(T_ad)
    for _ in range(60):
        rt.update(dev, w, robot, DT)

    def bench(fn, n=600):
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        return (time.perf_counter() - t0) / n * 1e3

    a = bench(lambda: rt.local.update(dev, DT))
    b = bench(lambda: rt.global_.update(dev, w, DT))
    c = bench(lambda: rt.update(dev, w, robot, DT))
    RESULTS.update({"E6_local_ms": a, "E6_global_ms": b, "E6_full_ms": c})
    print(f"    local fit {a:.3f} ms | global est {b:.3f} ms | full tick {c:.3f} ms")
    print(f"    the global estimator already ran before this package, so the "
          f"ADDED cost is ~{c - b:.2f} ms/frame")


# ── E7 the limitation ────────────────────────────────────────────────────────
def e7_neck_wobble():
    """What local mode COSTS: the reach frame is only as steady as the torso
    fit, whose only evidence is the hands. Hold the hands still relative to the
    torso, move only the head, and measure how far the robot's targets travel
    on their own. Reported because it is the honest weak spot of the mode, and
    because the alternative (dead-reckoning the head) trades it for the very
    property local mode exists to provide."""
    print("\nE7  NECK-MOTION WOBBLE (the cost of local mode)")
    print("    hands FIXED on the torso; every millimetre here is uncommanded")

    def wobble(neck_fn, dead_reckon, seconds=10.0):
        op = SyntheticOperator()
        op.set_reach(0.75)
        rt = BodyRetargeter(mode="local")
        if dead_reckon:
            rt.local = LocalTorsoFit(rt.model, dead_reckon=True)
        robot = mabel_like_robot()
        tg, ts = [], []
        for i in range(int(seconds / DT)):
            t = i * DT
            op.neck = neck_fn(t)
            T_ad, hands, skel = op.frame()
            out = rt.update(device_measurements(T_ad, hands, skel, timestamp=t),
                            WorldMeasurements(T_ad), robot, DT)
            if t > 2.0 and out.valid:
                tg.append(np.concatenate([out.wrist_targets[s]
                                          for s in ("left", "right")]))
                ts.append(t)
        tg = np.array(tg).reshape(-1, 2, 3)
        d = np.linalg.norm(tg - tg.mean(axis=0), axis=2).max(axis=1)
        return np.array(ts), d * 1000

    motions = [
        ("still", lambda t: (0.0, 0.0, 0.0)),
        ("look-around", lambda t: (np.deg2rad(60) * np.sin(2 * np.pi * 0.2 * t),
                                   0.0, 0.0)),
        ("nod", lambda t: (0.0, np.deg2rad(30) * np.sin(2 * np.pi * 0.15 * t),
                           0.0)),
        ("both", lambda t: (np.deg2rad(45) * np.sin(2 * np.pi * 0.2 * t),
                            np.deg2rad(30) * np.sin(2 * np.pi * 0.31 * t), 0.0)),
    ]
    fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.0))
    rows = []
    for k, (label, fn) in enumerate(motions):
        t, d = wobble(fn, False)
        rows.append((label, float(d.mean()), float(d.max())))
        if label != "still":
            ax[0].plot(t, d, lw=1.3, label=label,
                       color=[ACCENT, BLUE, INK][k - 1])
    ax[0].set(xlabel="time [s]", ylabel="uncommanded target motion [mm]",
              title="local mode: the head moves, the hands do not")
    ax[0].legend()
    # the alternative, and what it costs
    veh = VehicleDrift(yaw_rate=np.deg2rad(20), tilt_amp=np.deg2rad(6))
    inv = []
    for dr in (False, True):
        base, drifted = [], []
        for drift, sink in ((None, base), (veh, drifted)):
            op = SyntheticOperator()
            rt2 = BodyRetargeter(mode="local")
            if dr:
                rt2.local = LocalTorsoFit(rt2.model, dead_reckon=True)
            robot = mabel_like_robot()
            for i in range(int(8.0 / DT)):
                t = i * DT
                task(op, t)
                T_ad, hands, skel = op.frame(drift(t) if drift else None)
                o = rt2.update(device_measurements(T_ad, hands, skel, timestamp=t),
                               WorldMeasurements(T_ad), robot, DT)
                if t > 2.0 and o.valid:
                    sink.append(np.concatenate([o.wrist_targets[s]
                                                for s in ("left", "right")]))
        n = min(len(base), len(drifted))
        e = np.linalg.norm((np.array(base)[:n] - np.array(drifted)[:n]
                            ).reshape(n, 2, 3), axis=2)
        inv.append(float(e.mean()) * 1000)
    ax[1].bar([0, 1], [rows[3][1], wobble(motions[3][1], True)[1].mean()],
              0.5, color=ACCENT, label="neck wobble (both)")
    ax[1].bar([0.0, 1.0], [-inv[0], -inv[1]], 0.5, color=BLUE,
              label="vehicle-drift error")
    ax[1].axhline(0, color=INK, lw=0.8)
    ax[1].set(xticks=[0, 1], xticklabels=["paper-literal", "dead-reckon"],
              ylabel="[mm]   (up: wobble / down: drift)",
              title="the trade you cannot avoid")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "retarget_neck_wobble.pdf"))
    plt.close(fig)

    for label, mean, mx in rows:
        print(f"    {label:<14} mean {mean:8.2f} mm   peak {mx:8.2f} mm")
    print(f"    dead-reckoning instead: wobble ~0 mm, but vehicle-drift error "
          f"{inv[0]:.3f} -> {inv[1]:.1f} mm")
    RESULTS["E7_wobble_look_mm"] = rows[1][1]
    RESULTS["E7_wobble_nod_mm"] = rows[2][1]
    RESULTS["E7_wobble_both_mm"] = rows[3][1]
    RESULTS["E7_deadreckon_drift_mm"] = inv[1]


def main():
    t0 = time.time()
    print("=" * 74)
    print("RETARGETING MODES — LOCAL (device frame) vs GLOBAL (anchor frame)")
    print("=" * 74)
    e1_vehicle_invariance()
    e2_conditioning()
    e3_handoff()
    e4_fidelity()
    e5_neck_drift()
    e7_neck_wobble()
    e6_timing()
    print("\n" + "=" * 74)
    print("SUMMARY")
    print("=" * 74)
    for k, v in RESULTS.items():
        print(f"  {k:<26} {v:>14.6g}")
    print(f"\nfigures -> controller/docs/figures/retarget_*.pdf   "
          f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
