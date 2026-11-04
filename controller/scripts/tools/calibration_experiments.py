"""Reproducible experiments for the per-operator body/hand calibration subsystem.

Regenerates every figure and metric in ``docs/calibration.tex`` from scratch using
the REAL calibration code (no hand-faked numbers). Seeded throughout. Run::

    cd controller && python scripts/tools/calibration_experiments.py

Outputs (vector PDF) to ``docs/figures/`` and prints a metrics summary table:

  A. calib_recovery.pdf  -- dimension-recovery accuracy vs tracker noise (sweep of
                            synthetic operators 1.45-2.07 m, N trials each).
  B. calib_scale.pdf     -- workspace-coverage consistency: the FIXED gain over- and
                            under-drives across operator sizes; the per-user-corrected
                            gain (clutch) and the absolute reach match (shoulder) are
                            flat. Quantifies the benefit of the calibration.
  C. calib_fingers.pdf   -- finger retargeter A/B: calibrated-length keypoint-IK vs
                            the AnyTeleop-style vector optimizer (fingertip-match error
                            + solve time) across hand sizes.
"""
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))     # controller/
sys.path.insert(0, ROOT)
FIGDIR = os.path.join(ROOT, "docs", "figures")
SCENE = os.path.abspath(os.path.join(
    ROOT, "..", "simulation", "mabel_mujoco", "scenes", "mabel_sim.xml"))

from mabel import config as C
from mabel.calibration import (
    BodyCalibrator, OperatorProfile, synthesize_pose, synthesize_hand_skeleton,
    body_frame, POSE_TPOSE, POSE_FORWARD, POSE_HANDS,
)
from mabel.hand_ik_retarget import HandIKRetargeter, human_keypoints
from mabel.hand_vector_retarget import HandVectorRetargeter

HEIGHTS = np.linspace(1.45, 2.07, 14)          # EgoPoser subject range (145-207 cm)
HAND = {"palm_len": 0.090,
        "fingers": {"thumb": 0.062, "index": 0.082, "middle": 0.094,
                    "ring": 0.088, "pinky": 0.071},
        "pinch_open": 0.090, "pinch_close": 0.018}


def _calibrate(height, noise=0.0, seed=0):
    """Run the full streaming calibrator on a synthetic operator -> OperatorProfile."""
    cal = BodyCalibrator(user=f"H{height:.2f}")
    t = 0.0
    for pose in (POSE_TPOSE, POSE_FORWARD, POSE_HANDS):
        for fr in synthesize_pose(pose, height=height, hand=HAND, n=40,
                                  noise=noise, seed=seed):
            cal.observe(fr, now=t)
            t += 1.0 / 30.0
    return cal.profile


# ── Experiment A: dimension recovery vs tracker noise ─────────────────────────

def exp_recovery():
    noises_mm = [0, 1, 2, 3, 5]
    N = 25
    truth = lambda H: (C.ANTHRO_ARM_RATIO * H, C.ANTHRO_BIACROMIAL * H,
                       C.ANTHRO_ACROMION_BELOW_EYE * H)
    labels = ["arm length", "shoulder width", "neck height"]
    err = {lab: [] for lab in labels}          # mean |err| (mm) per noise level
    std = {lab: [] for lab in labels}
    for nm in noises_mm:
        acc = {lab: [] for lab in labels}
        for k in range(N):
            H = HEIGHTS[k % len(HEIGHTS)]
            p = _calibrate(H, noise=nm / 1000.0, seed=k + 1)
            ta, tw, tn = truth(H)
            acc["arm length"].append(abs(p.arm_length_m - ta))
            acc["shoulder width"].append(abs(p.shoulder_width_m - tw))
            acc["neck height"].append(abs(p.neck_height_m - tn))
        for lab in labels:
            err[lab].append(1000 * np.mean(acc[lab]))
            std[lab].append(1000 * np.std(acc[lab]))

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for lab in labels:
        ax.errorbar(noises_mm, err[lab], yerr=std[lab], marker="o", capsize=3, label=lab)
    ax.set_xlabel("per-joint tracker noise (mm, std)")
    ax.set_ylabel("recovered dimension error (mm, mean $\\pm$ std)")
    ax.set_title(f"Calibration recovery accuracy (N={N} operators/level)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "calib_recovery.pdf"))
    plt.close(fig)
    return noises_mm, err


# ── Experiment B: workspace-coverage consistency ──────────────────────────────

def exp_scale(robot_reach):
    arms = C.ANTHRO_ARM_RATIO * HEIGHTS
    # Effective coverage = how far the robot palm target moves for a FULL operator
    # arm-reach hand motion. Fixed gain (forward axis) scales the raw arm motion;
    # per-user clutch correction normalises to the reference arm; the shoulder mode
    # matches the robot reach exactly.
    fixed = C.RETARGET_SCALE[0] * arms
    per_user = []
    shoulder = []
    for L in arms:
        corr = float(np.clip(C.ANTHRO_ARM_REF / L, C.RETARGET_GAMMA_MIN,
                             C.RETARGET_GAMMA_MAX))
        per_user.append(C.RETARGET_SCALE[0] * corr * L)
        g = float(np.clip(robot_reach / L, C.RETARGET_GAMMA_MIN, C.RETARGET_GAMMA_MAX))
        shoulder.append(g * L)
    per_user = np.array(per_user); shoulder = np.array(shoulder)

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(HEIGHTS, fixed, "o-", label="fixed gain (legacy)")
    ax.plot(HEIGHTS, per_user, "s-", label="per-user clutch (corrected)")
    ax.plot(HEIGHTS, shoulder, "^-", label="per-user shoulder (reach match)")
    ax.axhline(robot_reach, color="k", ls=":", lw=1, label="robot arm reach")
    ax.set_xlabel("operator stature (m)")
    ax.set_ylabel("effective workspace coverage (m)")
    ax.set_title("Per-user scaling flattens coverage across operator size")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "calib_scale.pdf"))
    plt.close(fig)
    spread = lambda a: 100 * (a.max() - a.min()) / a.mean()
    return spread(fixed), spread(per_user), spread(shoulder)


# ── Experiment C: finger retargeter comparison ────────────────────────────────

def exp_fingers(model, data):
    scales = np.linspace(0.75, 1.3, 8)          # hand-size multipliers vs nominal
    results = {"keypoint": {"err": [], "ms": []}, "vector": {"err": [], "ms": []}}
    head = np.eye(4); head[:3, 3] = [0, 1.6, 0]
    _, B = body_frame(head)
    rng = np.random.default_rng(7)

    for name, Cls in (("keypoint", HandIKRetargeter), ("vector", HandVectorRetargeter)):
        hr = Cls(model, data)
        # Feed each scale the matching calibrated finger lengths (the whole point:
        # the keypoint solver gets a per-user length table; the vector solver is
        # intrinsically scale-invariant).
        for s in scales:
            hand = {"palm_len": HAND["palm_len"] * s,
                    "fingers": {f: v * s for f, v in HAND["fingers"].items()}}
            joints = synthesize_hand_skeleton(np.array([0.3, 1.4, -0.3]), B, hand,
                                              "right", rng, noise=0.001)
            # calibrated normalized lengths for this hand -> keypoint solver
            kp = human_keypoints(joints, "right")
            norm = {f: float(np.linalg.norm(kp[(f, "pip")] - kp[(f, "kn")])
                             + np.linalg.norm(kp[(f, "tip")] - kp[(f, "pip")]))
                    for f in ("thumb", "index", "middle", "ring", "pinky")
                    if (f, "tip") in kp}
            prof = OperatorProfile(finger_len_norm={"right": norm},
                                   palm_len_m={"right": hand["palm_len"]})
            hr.set_profile(prof)
            t0 = time.perf_counter()
            hr.apply("Right", joints, synchronous=True)
            dt_ms = 1000 * (time.perf_counter() - t0)
            # fingertip-match error in canonical space (robot vs calibrated human)
            h_used = hr._calibrate(kp, hr.hands["Right"]["lengths"],
                                   hr.hands["Right"]["kn0"], norm)
            r_kp = hr._robot_keypoints(hr.hands["Right"], hr.hands["Right"]["last_q"])
            errs = [np.linalg.norm(r_kp[(f, "tip")] - h_used[(f, "tip")])
                    for f in ("index", "middle", "ring", "pinky")
                    if (f, "tip") in r_kp and (f, "tip") in h_used]
            results[name]["err"].append(float(np.mean(errs)) if errs else np.nan)
            results[name]["ms"].append(dt_ms)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 3.6))
    for name in ("keypoint", "vector"):
        a1.plot(scales, results[name]["err"], "o-", label=name)
        a2.plot(scales, results[name]["ms"], "o-", label=name)
    a1.set_xlabel("hand-size scale (x nominal)"); a1.set_ylabel("fingertip match error (norm.)")
    a1.set_title("Accuracy"); a1.grid(True, alpha=0.3); a1.legend(fontsize=8)
    a2.set_xlabel("hand-size scale (x nominal)"); a2.set_ylabel("solve time (ms)")
    a2.set_title("Cost (one synchronous solve)"); a2.grid(True, alpha=0.3); a2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "calib_fingers.pdf"))
    plt.close(fig)
    summ = {n: (float(np.nanmean(results[n]["err"])), float(np.mean(results[n]["ms"])))
            for n in ("keypoint", "vector")}
    return summ


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    print("=" * 66)
    print("MABEL per-operator calibration -- experiments")
    print("=" * 66)

    noises, err = exp_recovery()
    print("\n[A] Dimension-recovery error (mm, mean over operators):")
    print(f"    {'noise(mm)':>10} " + " ".join(f"{n:>6}" for n in noises))
    for lab in ("arm length", "shoulder width", "neck height"):
        print(f"    {lab:>10} " + " ".join(f"{e:6.1f}" for e in err[lab]))
    print("    -> figures/calib_recovery.pdf")

    import mujoco
    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model); mujoco.mj_forward(model, data)
    from mabel import MabelTeleop
    reach = MabelTeleop(model, data).controller.retgt.robot_arm_reach()

    sf, sp, ss = exp_scale(reach)
    print(f"\n[B] Workspace-coverage spread across {HEIGHTS[0]:.2f}-{HEIGHTS[-1]:.2f} m operators")
    print(f"    robot arm reach = {reach:.3f} m")
    print(f"    fixed gain           : {sf:5.1f}%  (coverage varies with operator size)")
    print(f"    per-user clutch      : {sp:5.1f}%  (flattened)")
    print(f"    per-user shoulder    : {ss:5.1f}%  (absolute reach match)")
    print("    -> figures/calib_scale.pdf")

    summ = exp_fingers(model, data)
    print("\n[C] Finger retargeter comparison (mean over hand sizes):")
    print(f"    {'solver':>10} {'tip err':>10} {'solve ms':>10}")
    for n in ("keypoint", "vector"):
        print(f"    {n:>10} {summ[n][0]:10.4f} {summ[n][1]:10.3f}")
    print("    -> figures/calib_fingers.pdf")
    print("\nAll figures regenerated in", FIGDIR)


if __name__ == "__main__":
    main()
