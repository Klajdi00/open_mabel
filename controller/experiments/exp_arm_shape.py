#!/usr/bin/env python3
"""SHAPE FIDELITY — does the robot's arm match the operator's arm SHAPE?

THE OPERATOR REPORT (2026-08-01)
  "fully utilize the initial calibration data — the calibration is when the user
   has their arms fully extended forward. The robot needs to match the SHAPE of
   the human too: when the human does a T shape the robot needs to do a T shape
   as well. Right now the arms are still slightly bent and the robot cannot do
   the T shape properly."

WHAT 'MATCHING THE SHAPE' MEANS, MEASURABLY. Tracking the wrist POSITION is not
the same as reproducing the arm's shape. Two robot configurations can put the
wrist in the same place with very different elbows, and the one that looks wrong
to the operator is the bent one. So the metrics here are about the arm, not the
target:

  elbow angle       180 deg = straight. At the operator's full extension the
                    robot's elbow should be straight too, because that is what
                    the human's is doing.
  extension rho_r   |robot wrist - robot shoulder| / robot arm reach. 1.0 = the
                    robot is as extended as it can be. If the operator is at
                    full extension and this is 0.9, the robot is holding back
                    9% of its reach for no reason.
  T-pose spread     the angle between the two arms, at the mid-shoulder. A real
                    T is 180 deg.

THE SUSPECTED CAUSE (confirmed by reading, measured below). The GMR arm map
scales by sigma = L_robot / L_human, and rho = |wrist-shoulder| / L_human decides
when the extension-exact blend engages (ARM_GMR_EXT_GATE = 0.85). L_human comes
from `ShoulderEstimator.arm_length`, whose own docstring states the limitation:
the anthropometric prior FLOORS the live estimate, so "a shorter-than-prior arm
needs the measured profile to read rho = 1".

Meanwhile the 1-pose calibration measures the operator at full forward extension
and stores the result as `self.reach` — which is read in exactly ONE place,
`_wmap_head`, the LEGACY map. The default map is 'gmr'. So the calibration is
captured, saved, logged, and then never consulted by the arm map that is
actually running. An operator with shorter-than-prior arms therefore never
reaches rho = 1, the extension gate never fully opens, and the elbow keeps a few
degrees of bend forever.

    python3 controller/experiments/exp_arm_shape.py [--quick]
Figure: controller/docs/figures/arm_shape.pdf
"""
import argparse
import json
import os
import sys

import numpy as np
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
import exp_vp_bimanual_latency as h                              # noqa: E402

FIGDIR = os.path.join(_REPO, "controller", "docs", "figures")
RESULTS = os.path.join(_HERE, "results")

# Operator anthropometry for the synthetic subject. SHORTER than the anthro
# prior on purpose: that is the case the estimator's own docstring flags as
# needing the measured profile, and the case the operator is living in.
ARM_L = 0.545          # m, shoulder->wrist
SHOULDER_W = 0.38
HEAD_Y = 1.55


def _shoulder(side):
    sgn = +1.0 if side == "right" else -1.0
    return np.array([sgn * SHOULDER_W * 0.5, HEAD_Y - 0.22, 0.0])


def pose_forward(side, t, frac=1.0):
    """Arms straight FORWARD (the calibration pose), at `frac` of full reach."""
    return _shoulder(side) + np.array([0.0, 0.0, -ARM_L * frac])


def pose_T(side, t, frac=1.0):
    """A T: arms straight out to the SIDE, level with the shoulders."""
    sgn = +1.0 if side == "right" else -1.0
    return _shoulder(side) + np.array([sgn * ARM_L * frac, 0.0, 0.0])


class ArmGeom:
    """Robot-side arm geometry from the live MuJoCo state."""

    def __init__(self, model, data):
        self.m, self.d = model, data
        self.bid = {}
        for side, p in (("right", "r"), ("left", "l")):
            self.bid[side] = {
                "shoulder": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{p}_arm_mount"),
                "elbow": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{p}_elbow"),
                "wrist": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{p}_wrist"),
            }

    def _p(self, side, key):
        return self.d.xpos[self.bid[side][key]].copy()

    def wrist_anchor(self, side):
        """The point `robot_arm_max_reach` is DEFINED on: the ORCA wrist joint
        anchor. Measuring the `*_wrist` BODY origin instead (as the first version
        of this probe did) compares a different point against that reach and
        under-reports extension by a third — the same trap exp_arm_fidelity
        documents for the palm body."""
        jid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT,
                                f"{'Right' if side == 'right' else 'Left'} Wrist")
        return self.d.xanchor[jid].copy()

    def elbow_deg(self, side):
        """Interior angle at the elbow; 180 = straight."""
        s, e = self._p(side, "shoulder"), self._p(side, "elbow")
        w = self.wrist_anchor(side)
        a, b = s - e, w - e
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-6 or nb < 1e-6:
            return np.nan
        return float(np.degrees(np.arccos(np.clip(a @ b / (na * nb), -1, 1))))

    def extension(self, side, reach):
        s, w = self._p(side, "shoulder"), self.wrist_anchor(side)
        return float(np.linalg.norm(w - s) / max(reach, 1e-6))

    def spread_deg(self):
        """Angle between the two arms about the mid-shoulder — 180 for a real T."""
        mid = 0.5 * (self._p("right", "shoulder") + self._p("left", "shoulder"))
        vr = self.wrist_anchor("right") - mid
        vl = self.wrist_anchor("left") - mid
        nr, nl = np.linalg.norm(vr), np.linalg.norm(vl)
        if nr < 1e-6 or nl < 1e-6:
            return np.nan
        return float(np.degrees(np.arccos(np.clip(vr @ vl / (nr * nl), -1, 1))))


def run_pose(pose_fn, secs=6.0, calibrate=False, frac=1.0):
    """Hold one operator pose; return the settled robot arm geometry."""
    op = h.SyntheticVPOperator(lambda s, t: pose_fn(s, t, frac),
                               head_y=HEAD_Y, shoulder_w=SHOULDER_W,
                               upper=ARM_L * 0.53, fore=ARM_L * 0.47)
    rig = h.Rig(frame_hz=30.0)
    rig.teleop.set_control_spec(method="retarget", control_type="impedance",
                                region="whole_body", scope="local")
    if calibrate:
        # Freeze the MEASURED anthropometry, exactly as a completed 1-pose
        # calibration should: this is the knob under test.
        rig.eng.shoulder.set_profile(arm_length_lr=(ARM_L, ARM_L),
                                     shoulder_width=SHOULDER_W)
    g = ArmGeom(rig.model, rig.data)
    t, nxt = 0.0, 0.0
    while t < secs:
        pose = None
        if t >= nxt:
            pose = op.frame(t)
            nxt += 1.0 / 30.0
        rig.teleop.update(pose, None, h.DT)
        rig._plant()
        t += h.DT
    return {
        "elbow_r": g.elbow_deg("right"), "elbow_l": g.elbow_deg("left"),
        "ext_r": g.extension("right", rig.eng.arm_reach["right"]),
        "ext_l": g.extension("left", rig.eng.arm_reach["left"]),
        "spread": g.spread_deg(),
        "L_used": float(rig.eng.shoulder.arm_length("right") or np.nan),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    secs = 4.0 if a.quick else 6.0

    print("\nARM SHAPE FIDELITY")
    print(f"synthetic operator: arm {ARM_L*100:.1f} cm (SHORTER than the anthro "
          f"prior on purpose)\n")
    print(f"{'pose':22s} {'calib':6s} {'elbow R':>9s} {'elbow L':>9s} "
          f"{'ext R':>7s} {'ext L':>7s} {'spread':>8s} {'L used':>7s}")
    rows = []
    for name, fn in (("arms FORWARD (calib)", pose_forward), ("T-POSE", pose_T)):
        for cal in (False, True):
            r = run_pose(fn, secs=secs, calibrate=cal)
            rows.append({"pose": name, "calibrated": cal, **r})
            print(f"{name:22s} {'yes' if cal else 'no':6s} "
                  f"{r['elbow_r']:8.1f}° {r['elbow_l']:8.1f}° "
                  f"{r['ext_r']:7.3f} {r['ext_l']:7.3f} "
                  f"{r['spread']:7.1f}° {r['L_used']:7.3f}")

    print("\n  (elbow 180° = straight · ext 1.0 = fully extended · T spread 180°)")
    # figure
    os.makedirs(FIGDIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    labels = [f"{r['pose']}\n{'calibrated' if r['calibrated'] else 'uncalibrated'}"
              for r in rows]
    axes[0].bar(range(len(rows)), [r["elbow_r"] for r in rows],
                color=["#C44536" if not r["calibrated"] else "#2A6F97" for r in rows])
    axes[0].axhline(180, ls=":", color="k", label="straight (180°)")
    axes[0].set_ylim(120, 190); axes[0].set_ylabel("right elbow [deg]", fontsize=9)
    axes[0].set_xticks(range(len(rows)))
    axes[0].set_xticklabels(labels, fontsize=6.5)
    axes[0].set_title("elbow straightness", fontsize=10)
    axes[0].grid(alpha=0.25, lw=0.6); axes[0].legend(fontsize=7)
    axes[1].bar(range(len(rows)), [r["ext_r"] for r in rows],
                color=["#C44536" if not r["calibrated"] else "#2A6F97" for r in rows])
    axes[1].axhline(1.0, ls=":", color="k", label="full reach")
    axes[1].set_ylabel("right extension fraction", fontsize=9)
    axes[1].set_xticks(range(len(rows)))
    axes[1].set_xticklabels(labels, fontsize=6.5)
    axes[1].set_title("how much reach is used", fontsize=10)
    axes[1].grid(alpha=0.25, lw=0.6); axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "arm_shape.pdf"), format="pdf", bbox_inches="tight")
    plt.close(fig)

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "arm_shape.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nfigure -> controller/docs/figures/arm_shape.pdf\n")


if __name__ == "__main__":
    main()
