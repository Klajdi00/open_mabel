#!/usr/bin/env python3
"""Validate the torso-frame estimator on SYNTHETIC ground truth.

    python3 controller/experiments/exp_body_frame.py

A forward operator model generates head + two wrist 6-DOF poses from KNOWN torso
yaw, neck yaw, root translation and arm angles, so we can check that the
estimator attributes each canonical motion to the right cause. Deterministic;
figures land in `controller/docs/figures/bodyframe_*.pdf`.

  S1  whole-body turn   -> torso yaw tracked, neck ~0, arms hold body-relative
  S2  look-around       -> torso ~0, all of it read as NECK (the head-only bug)
  S3  one-arm reach     -> torso ~0 (not mistaken for a turn), arm moves
  S4  walk + turn + reach (dynamic) -> torso yaw + root tracked, arm disentangled

Each is compared against the HEAD-ONLY baseline (`calibration.body_frame`), which
cannot tell torso from neck and therefore fails S2/S4 by construction.
"""
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from mabel.whole_body_retargeting.body_frame_estimator import (   # noqa: E402
    BodyFrameEstimator, head_yaw, UP, _flatten, _wrap, _REF_FWD, _rot_about_up)
from mabel.whole_body_retargeting.calibration import body_frame    # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
FIGS = os.path.join(REPO, "controller", "docs", "figures")
os.makedirs(FIGS, exist_ok=True)

DT = 1.0 / 60.0
plt.rcParams.update({"figure.dpi": 120, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 9, "legend.frameon": False})
ACCENT, WARN, INK, BLUE = "#c25b2a", "#e9a679", "#2b2724", "#5eb0ff"

# nominal operator anthropometry (metres), in the torso frame [right, up, fwd]
SHOULDER_HALF = 0.20
NECK_UP = 0.22           # head above shoulder centre
UPPER = 0.28
FORE = 0.26


def _mat(R, p):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def torso_R(psi):
    return _rot_about_up(psi)


def operator_frame(psi, neck, root, armL, armR):
    """Forward model: known (torso yaw, neck yaw, root xyz, per-arm angles) ->
    head + two wrist 4x4 world poses.

    Arm angle `arm` = (azimuth, elevation) of the straight arm in the torso
    frame; the wrist frame is built so its columns rotate with torso+arm."""
    Rt = torso_R(psi)
    fwd = Rt @ _REF_FWD
    right = np.cross(fwd, UP); right /= np.linalg.norm(right)
    B = np.column_stack([right, UP, fwd])                 # torso [right,up,fwd]
    chest = np.asarray(root, float)

    # head: torso yaw + neck yaw, sitting NECK_UP above the chest
    head_pos = chest + B @ np.array([0.0, NECK_UP, 0.0])
    Rhead = _rot_about_up(psi + neck)                     # camera looks -Z
    # build a head rotation whose flattened -Z gives heading psi+neck
    fwd_h = _rot_about_up(psi + neck) @ _REF_FWD
    right_h = np.cross(fwd_h, UP); right_h /= np.linalg.norm(right_h)
    Rhead = np.column_stack([right_h, UP, -fwd_h])        # col2 = -fwd => -Z fwd

    def wrist(side, arm):
        az, el = arm
        sh = chest + B @ np.array([side * SHOULDER_HALF, NECK_UP * 0.4, 0.0])
        # arm direction in torso frame, then to world
        d_t = np.array([np.sin(az) * np.cos(el), -np.sin(el), np.cos(az) * np.cos(el)])
        d_w = B @ d_t
        wp = sh + (UPPER + FORE) * d_w
        # wrist orientation: torso yaw composed with the arm azimuth about up
        Rw = _rot_about_up(psi) @ _rot_about_up(az) @ np.column_stack(
            [_REF_FWD, UP, np.cross(_REF_FWD, UP)])
        return _mat(Rw, wp)

    return _mat(Rhead, head_pos), wrist(-1.0, armL), wrist(+1.0, armR)


# ── scenario generators: yield (t, head, wl, wr, truth) ──────────────────────
def s1_whole_body_turn(T=3.0):
    """Torso yaws 0->60 deg; neck fixed; arms fixed body-relative."""
    n = int(T / DT)
    for k in range(n):
        u = k / (n - 1)
        psi = np.radians(60.0) * (0.5 - 0.5 * np.cos(np.pi * u))   # smooth
        h, wl, wr = operator_frame(psi, 0.0, [0, 0, 0], (0.5, 0.3), (-0.5, 0.3))
        yield k * DT, h, wl, wr, dict(psi=psi, neck=0.0, root=np.zeros(3),
                                      armL=(0.5, 0.3), armR=(-0.5, 0.3))


def s2_look_around(T=3.0):
    """Torso fixed; neck yaws +-60 deg; arms fixed. The head-only bug's home."""
    n = int(T / DT)
    for k in range(n):
        neck = np.radians(60.0) * np.sin(2 * np.pi * (k / (n - 1)))
        h, wl, wr = operator_frame(0.0, neck, [0, 0, 0], (0.5, 0.3), (-0.5, 0.3))
        yield k * DT, h, wl, wr, dict(psi=0.0, neck=neck, root=np.zeros(3),
                                      armL=(0.5, 0.3), armR=(-0.5, 0.3))


def s3_one_arm_reach(T=3.0):
    """Torso + neck fixed; right arm sweeps. Must NOT read as a body turn."""
    n = int(T / DT)
    for k in range(n):
        u = k / (n - 1)
        azR = -0.5 + 1.2 * (0.5 - 0.5 * np.cos(np.pi * u))
        h, wl, wr = operator_frame(0.0, 0.0, [0, 0, 0], (0.5, 0.3), (azR, 0.3))
        yield k * DT, h, wl, wr, dict(psi=0.0, neck=0.0, root=np.zeros(3),
                                      armL=(0.5, 0.3), armR=(azR, 0.3))


def s3b_both_arms_parallel(T=3.0):
    """THE DISCRIMINATOR (plan case R3b): BOTH arms swing the same way while the
    body is still. Bearing-consensus reads this as a turn (both bearings rotate
    together); the twist fit rejects it (arms orbit the shoulders, not the body
    centre -> high residual)."""
    n = int(T / DT)
    for k in range(n):
        u = k / (n - 1)
        sweep = 0.9 * (0.5 - 0.5 * np.cos(np.pi * u))     # both azimuths shift +
        h, wl, wr = operator_frame(0.0, 0.0, [0, 0, 0],
                                   (0.4 + sweep, 0.3), (-0.4 + sweep, 0.3))
        yield k * DT, h, wl, wr, dict(psi=0.0, neck=0.0, root=np.zeros(3),
                                      armL=(0.4 + sweep, 0.3), armR=(-0.4 + sweep, 0.3))


def s5_gaze_fixed_turn(T=3.0):
    """GAZE-FIXED body turn: torso yaws 0->50 deg while the neck COUNTER-rotates
    so the head stays fixed on a target (head yaw ~ 0). Arms held body-relative,
    so they orbit the spine RIGIDLY. The head cannot corroborate (it is still);
    only the rigid-coherence gate (wrists orbit the spine, low residual) can tell
    this real body turn apart from the both-arms reach (S3b)."""
    n = int(T / DT)
    for k in range(n):
        u = k / (n - 1)
        psi = np.radians(50.0) * (0.5 - 0.5 * np.cos(np.pi * u))
        neck = -psi                                       # gaze stays fixed
        h, wl, wr = operator_frame(psi, neck, [0, 0, 0], (0.5, 0.3), (-0.5, 0.3))
        yield k * DT, h, wl, wr, dict(psi=psi, neck=neck, root=np.zeros(3),
                                      armL=(0.5, 0.3), armR=(-0.5, 0.3))


def s6_point_and_look(T=3.0):
    """CASE 3 (operator's words): the arms move AND the head follows the SAME way,
    torso otherwise still -> we WANT the base to turn. Identical arm motion to S3b
    (both azimuths sweep +0.9) but here the head yaws with them. The controlled
    contrast S3b (head still -> arms) vs S6 (head follows -> base) is exactly the
    head-corroboration flip. Ground-truth base = the shared head+arm rotation."""
    n = int(T / DT)
    for k in range(n):
        u = k / (n - 1)
        sweep = 0.9 * (0.5 - 0.5 * np.cos(np.pi * u))     # arms AND head
        h, wl, wr = operator_frame(0.0, sweep, [0, 0, 0],
                                   (0.4 + sweep, 0.3), (-0.4 + sweep, 0.3))
        # we intend the base to absorb the shared rotation -> desired psi = sweep
        yield k * DT, h, wl, wr, dict(psi=sweep, neck=0.0, root=np.zeros(3),
                                      armL=(0.4 + sweep, 0.3), armR=(-0.4 + sweep, 0.3))


def s4_walk_turn_reach(T=5.0):
    """DYNAMIC: operator walks forward, turns 45 deg, and reaches -- all at once."""
    n = int(T / DT)
    for k in range(n):
        u = k / (n - 1)
        psi = np.radians(45.0) * u
        root = np.array([0.0, 0.0, -0.6 * u])            # walk along -Z (forward)
        # note: walk in WORLD, independent of turn
        neck = np.radians(15.0) * np.sin(2 * np.pi * u)  # glances around
        azR = -0.5 + 0.8 * np.sin(np.pi * u)             # reaches
        h, wl, wr = operator_frame(psi, neck, root, (0.5, 0.3), (azR, 0.3))
        yield k * DT, h, wl, wr, dict(psi=psi, neck=neck, root=root,
                                      armL=(0.5, 0.3), armR=(azR, 0.3))


# ── run one scenario through estimator + baseline ─────────────────────────────
def run(scenario, Est=None):
    frames = list(scenario)
    est = (Est or BodyFrameEstimator)()
    _, h0, wl0, wr0, _ = frames[0]
    est.calibrate_facing(h0, wl0, wr0)
    psi0_true = frames[0][4]["psi"]

    rec = {k: [] for k in ("t", "psi_true", "psi_est", "psi_head", "neck_true",
                           "neck_est", "wristL_rel_est", "wristL_rel_true",
                           "root_true_x", "root_true_z")}
    # torso-relative right-wrist position: should stay CONSTANT during S1
    for t, h, wl, wr, gt in frames:
        origin, B = est.update(h, wl, wr, DT)
        # baseline head-only frame heading
        _, Bh = body_frame(h)
        psi_head = _yaw_of_B(Bh)
        rec["t"].append(t)
        rec["psi_true"].append(_wrap(gt["psi"] - psi0_true))
        rec["psi_est"].append(est.torso_yaw())
        rec["psi_head"].append(_wrap(psi_head - _yaw_of_B(body_frame(h0)[1])))
        rec["neck_true"].append(gt["neck"])
        rec["neck_est"].append(est.neck_yaw(h[:3, :3]))
        # right wrist expressed in the ESTIMATED torso frame
        wrel = B.T @ (wr[:3, 3] - origin)
        rec["wristL_rel_est"].append(wrel.copy())
        # ground-truth torso-relative (rebuild from truth)
        Bt = _B_of_psi(gt["psi"])
        # ground-truth chest centre (estimator-agnostic)
        ctrue = np.asarray(gt["root"], float) + Bt @ np.array([0, NECK_UP, 0]) \
            + Bt @ np.array([0.0, -0.22, -0.02])
        rec["wristL_rel_true"].append(Bt.T @ (wr[:3, 3] - ctrue))
        rec["root_true_x"].append(gt["root"][0])
        rec["root_true_z"].append(gt["root"][2])
    for k in rec:
        rec[k] = np.array(rec[k])
    return rec


def _jitter(T, rng, pos_mm=5.0, ori_deg=1.0):
    """Add realistic tracker noise: gaussian position + small random rotation."""
    T = np.asarray(T, float).copy()
    T[:3, 3] += rng.normal(0, pos_mm / 1000.0, 3)
    ax = rng.normal(0, 1, 3); ax /= np.linalg.norm(ax) + 1e-9
    ang = rng.normal(0, np.radians(ori_deg))
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0.0]])
    dR = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)
    T[:3, :3] = dR @ T[:3, :3]
    return T


def _yaw_of_B(B):
    return _yaw_of(B[:, 2])


def _yaw_of(fwd):
    from mabel.whole_body_retargeting.body_frame_estimator import _yaw_between
    return _yaw_between(_REF_FWD, fwd)


def _B_of_psi(psi):
    fwd = _rot_about_up(psi) @ _REF_FWD
    right = np.cross(fwd, UP); right /= np.linalg.norm(right)
    return np.column_stack([right, UP, fwd])


# ── metrics + plots ───────────────────────────────────────────────────────────
def _yaw_rms(rec):
    return float(np.degrees(np.sqrt(np.mean(
        [_wrap(a - b) ** 2 for a, b in zip(rec["psi_est"], rec["psi_true"])]))))


def _head_rms(rec):
    return float(np.degrees(np.sqrt(np.mean(
        [_wrap(a - b) ** 2 for a, b in zip(rec["psi_head"], rec["psi_true"])]))))


def summarise(name, rec):
    de = np.degrees(np.array([_wrap(a - b) for a, b in
                              zip(rec["psi_est"], rec["psi_true"])]))
    dh = np.degrees(np.array([_wrap(a - b) for a, b in
                              zip(rec["psi_head"], rec["psi_true"])]))
    nde = np.degrees(np.array([_wrap(a - b) for a, b in
                               zip(rec["neck_est"], rec["neck_true"])]))
    wl = np.array(rec["wristL_rel_est"])
    wl_drift = np.degrees(0)  # placeholder
    wl_std = float(np.linalg.norm(wl - wl.mean(0), axis=1).std()) * 1000.0
    print(f"\n{name}")
    print(f"   torso yaw  RMS err (estimator) : {np.sqrt(np.mean(de**2)):6.2f} deg"
          f"   max {np.max(np.abs(de)):6.2f}")
    print(f"   torso yaw  RMS err (head-only) : {np.sqrt(np.mean(dh**2)):6.2f} deg"
          f"   max {np.max(np.abs(dh)):6.2f}   <- baseline")
    print(f"   neck yaw   RMS err (estimator) : {np.sqrt(np.mean(nde**2)):6.2f} deg")
    print(f"   torso-rel wrist wobble (est)   : {wl_std:6.2f} mm  (0 = arm held still)")
    return dict(est_rms=float(np.sqrt(np.mean(de**2))),
                head_rms=float(np.sqrt(np.mean(dh**2))),
                neck_rms=float(np.sqrt(np.mean(nde**2))), wrist_mm=wl_std)


def plot_scenarios(recs):
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5))
    titles = ["S1 whole-body turn", "S2 look-around", "S3 one-arm reach",
              "S4 walk + turn + reach"]
    for ax, (name, rec), title in zip(axes.ravel(), recs.items(), titles):
        t = rec["t"]
        ax.plot(t, np.degrees(rec["psi_true"]), color=INK, lw=2, label="torso yaw (truth)")
        ax.plot(t, np.degrees(rec["psi_est"]), color=ACCENT, lw=1.8, label="estimator")
        ax.plot(t, np.degrees(rec["psi_head"]), color=BLUE, ls="--", lw=1.4,
                label="head-only (baseline)")
        ax.set_title(title); ax.set_xlabel("t [s]"); ax.set_ylabel("torso yaw [deg]")
        ax.legend(fontsize=7)
    fig.suptitle("Torso yaw: estimator tracks the BODY; head-only tracks the HEAD")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "bodyframe_scenarios.pdf"))
    plt.close(fig)


def plot_decomposition(recs):
    """S2 + S4: show the head yaw split into torso (base) + neck."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, key, title in zip(axes, ("S2", "S4"),
                              ("S2 look-around: all head yaw -> NECK",
                               "S4 dynamic: head yaw split torso + neck")):
        rec = recs[key]
        t = rec["t"]
        head = np.degrees([_wrap(a + b) for a, b in
                           zip(rec["psi_true"], rec["neck_true"])])
        ax.plot(t, head, color=INK, lw=2, label="head yaw (world)")
        ax.plot(t, np.degrees(rec["psi_est"]), color=ACCENT, lw=1.6, label="-> torso (base)")
        ax.plot(t, np.degrees(rec["neck_est"]), color=WARN, lw=1.6, label="-> neck")
        ax.set_title(title, fontsize=9); ax.set_xlabel("t [s]"); ax.set_ylabel("yaw [deg]")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "bodyframe_decomposition.pdf"))
    plt.close(fig)


def plot_cases(rows):
    """All seven canonical cases: our unified estimator vs the two ablations."""
    labels = [r[0] for r in rows]
    ours = [r[1] for r in rows]
    twist = [r[2] for r in rows]
    head = [r[3] for r in rows]
    x = np.arange(len(rows)); w = 0.26
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.bar(x - w, ours, w, color=ACCENT, label="ours (unified)")
    ax.bar(x, twist, w, color=WARN, label="twist-fit only (ablation)")
    ax.bar(x + w, head, w, color=BLUE, label="head-only (baseline)")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=7.5)
    ax.set_ylabel("torso-yaw RMS error [deg]")
    ax.axhline(3.0, color=INK, lw=0.8, ls=":", alpha=0.6)
    ax.set_title("Base-frame decomposition error across the seven canonical motions")
    ax.legend(fontsize=8, ncol=3, loc="upper center")
    ax.set_ylim(0, max(max(twist), max(head)) * 1.15)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "bodyframe_cases.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    t0 = time.time()
    print("=" * 70)
    print("TORSO-FRAME ESTIMATOR  vs  head-only baseline  (synthetic truth)")
    print("=" * 70)
    from mabel.whole_body_retargeting.twist_estimator import TwistFitEstimator
    scen = {"S1 whole-body turn": s1_whole_body_turn,
            "S2 look-around": s2_look_around,
            "S3a one-arm reach": s3_one_arm_reach,
            "S3b both-arms, head still": s3b_both_arms_parallel,
            "S5 gaze-fixed turn": s5_gaze_fixed_turn,
            "S6 arms+head (case 3)": s6_point_and_look,
            "S4 walk+turn+reach": s4_walk_turn_reach}
    want = {"S1 whole-body turn": "base", "S2 look-around": "neck",
            "S3a one-arm reach": "arm", "S3b both-arms, head still": "arm",
            "S5 gaze-fixed turn": "base", "S6 arms+head (case 3)": "base",
            "S4 walk+turn+reach": "base"}
    print("\n" + "=" * 78)
    print(f"{'scenario':26s} {'want':>5s} {'ours':>10s} {'twist-only':>11s} {'head-only':>11s}")
    print(f"{'':26s} {'':>5s} {'(unified)':>10s} {'(plan abl.)':>11s} {'(baseline)':>11s}")
    print("-" * 78)
    recs = {}
    case_rows = []
    for name, gen in scen.items():
        ru = run(gen(), BodyFrameEstimator)     # our unified estimator
        rt = run(gen(), TwistFitEstimator)      # plan's pure twist-fit (ablation)
        eu = _yaw_rms(ru); et = _yaw_rms(rt); eh = _head_rms(ru)
        recs[name.split()[0]] = ru
        case_rows.append((name.split(" ", 1)[0], eu, et, eh))
        print(f"{name:26s} {want[name]:>5s} {eu:7.2f} deg {et:8.2f} deg {eh:8.2f} deg")
    plot_cases(case_rows)
    print("=" * 78)
    print("torso-yaw RMS error vs ground truth, deg (lower is better).")
    print("'want' = the motion the base SHOULD absorb: base=turn tracked, "
          "arm/neck=base must stay ~0.")

    # FALSE-DRIVE RATE (safety): fraction of frames the base is commanded to turn
    # (|torso rate| > 0.05 rad/s) during a motion that should NOT drive it -- a
    # look-around or an arm reach. This is the metric that matters operationally:
    # a false drive during manipulation moves the whole robot unexpectedly.
    print("\nFALSE-DRIVE RATE (base commanded during a NON-body motion):")
    for name, gen in (("S2 look-around", s2_look_around),
                      ("S3a one-arm reach", s3_one_arm_reach),
                      ("S3b both-arms reach", s3b_both_arms_parallel)):
        for label, Est in (("ours", BodyFrameEstimator),
                           ("head-only", None)):
            fr = list(gen())
            if Est is None:                       # head-only base heading rate
                _, B0 = body_frame(fr[0][1]); prev = _yaw_of_B(B0); fd = 0
                for _, h, _, _, _ in fr:
                    _, B = body_frame(h); y = _yaw_of_B(B)
                    if abs(_wrap(y - prev)) / DT > 0.05:
                        fd += 1
                    prev = y
                rate = 100.0 * fd / len(fr)
            else:
                est = Est(); _, h0, wl0, wr0, _ = fr[0]
                est.calibrate_facing(h0, wl0, wr0); fd = 0
                for _, h, wl, wr, _ in fr:
                    est.update(h, wl, wr, DT)
                    if abs(est.psi_rate) > 0.05:
                        fd += 1
                rate = 100.0 * fd / len(fr)
            print(f"   {name:22s} {label:10s}: {rate:5.1f}% of frames")

    recs = {k: run(v(), BodyFrameEstimator) for k, v in
            {"S1": s1_whole_body_turn, "S2": s2_look_around,
             "S3": s3_one_arm_reach, "S4": s4_walk_turn_reach}.items()}
    metrics = {k: summarise(k, recs[k]) for k in recs}
    # S4 root (walk) tracking: does the chest origin follow the operator?
    r4 = recs["S4"]
    est = BodyFrameEstimator(); frames = list(s4_walk_turn_reach())
    _, h0, wl0, wr0, _ = frames[0]; est.calibrate_facing(h0, wl0, wr0)
    root_err = []
    for _, h, wl, wr, gt in frames:
        origin, _ = est.update(h, wl, wr, DT)
        # chest origin minus its t0 value should track the operator root delta
        root_err.append(origin)
    root_err = np.array(root_err)
    walked = root_err - root_err[0]
    truth = np.array([f[4]["root"] for f in frames])
    hxz = np.linalg.norm((walked - truth)[:, [0, 2]], axis=1) * 1000.0
    print(f"\nS4 root (walk) tracking: horizontal RMS {np.sqrt(np.mean(hxz**2)):.1f} mm"
          f"   max {hxz.max():.1f} mm   (operator walked 0.6 m)")

    # noise robustness: 5 mm position + 1 deg orientation jitter on every frame
    print("\nNOISE ROBUSTNESS (5 mm + 1 deg jitter/frame):")
    rng = np.random.default_rng(0)
    for name, gen in (("S1", s1_whole_body_turn), ("S2", s2_look_around),
                      ("S4", s4_walk_turn_reach)):
        est = BodyFrameEstimator(); fr = list(gen())
        _, h0, wl0, wr0, _ = fr[0]; est.calibrate_facing(h0, wl0, wr0)
        de = []
        for _, h, wl, wr, gt in fr:
            hn, wln, wrn = (_jitter(x, rng) for x in (h, wl, wr))
            est.update(hn, wln, wrn, DT)
            de.append(np.degrees(_wrap(est.torso_yaw() - _wrap(gt["psi"] - fr[0][4]["psi"]))))
        print(f"   {name}: torso yaw RMS {np.sqrt(np.mean(np.square(de))):5.2f} deg")

    plot_scenarios(recs)
    plot_decomposition(recs)
    print("\n" + "=" * 70)
    print(f"figures -> controller/docs/figures/bodyframe_*.pdf   ({time.time()-t0:.0f}s)")
    print("=" * 70)
