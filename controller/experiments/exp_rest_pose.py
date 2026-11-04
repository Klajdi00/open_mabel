#!/usr/bin/env python3
"""REST POSE — where do MABEL's arms go when nothing is asking them to be anywhere?

THE OPERATOR REPORT (2026-09-15)
  "the default beginning and ending pose of the robot, the elbow should be
   extended down — right now the default is still with a 90 degree bent elbow —
   make sure everywhere change to fully extended, at the naturally gravity down
   position."

WHAT WAS ACTUALLY WRONG. Every client — the studio's timeline, both apps' 3-D
viewers, the offline demo — built its default pose by filling a dictionary with
ZEROS. That is a natural thing to write and it is wrong, because MABEL's qpos
zero is not the robot at rest. At qpos zero the upper arm hangs but the forearm
juts straight forward: the two segments meet at 91.3 deg. It is the pose a
person makes holding an invisible tray, and it costs 38.6 Nm of gravity holding
torque to stand in. The arm only hangs when the ELBOW is extended, which on this
model is arm_4 = +-1.52, nowhere near 0.

WHAT THIS MEASURES.
  E1  The holding-torque landscape over the two joints that decide the hanging
      arm: shoulder roll (arm_2, how far the arm is held off the body) and elbow
      (arm_4). Gravity torque from mj_inverse at qvel = qacc = 0, so it is the
      static torque each actuator must hold, not a simulation artefact.
  E2  The chosen rest pose against the two it replaces — qpos zero, and the
      per-arm solve the apps shipped before — on holding torque, elbow angle,
      hand drop, and whether every joint is inside the MJCF's own limits.
  E3  Arm geometry as a function of the elbow: the bend angle and the hand's
      height, showing that 15.1 deg is the linkage's floor rather than a choice,
      and that the torque minimum sits at the elbow's end-stop (which is why the
      shipped value backs 0.022 rad off it).
  E4  A model defect found on the way: the lift column self-penetrates the
      chassis below 0.155 m per stage, so only the top half of its declared
      [0, 0.3175] stroke is usable in simulation.

    python3 controller/experiments/exp_rest_pose.py [--quick]

Figure: controller/docs/figures/rest_pose.pdf  (feeds controller/docs/rest_pose.tex)
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import mujoco
except ImportError:                                     # pragma: no cover
    raise SystemExit("needs mujoco:  pip install mujoco")
from scipy.optimize import minimize_scalar

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
MJCF = os.path.join(REPO, "simulation", "mabel_mujoco", "models", "mabel_full.xml")
FIGS = os.path.join(REPO, "controller", "docs", "figures")
RESULTS = os.path.join(HERE, "results")

# The pose this experiment derives, as it is written into every client. Kept
# here so the script can VERIFY what shipped rather than only propose it.
SHIPPED = {"arm_1": 0.0, "arm_2": -0.1711, "arm_3": 0.0, "arm_4": 1.52,
           "arm_5": 0.0, "arm_6": 0.0, "arm_7": 0.0}
ELBOW_MARGIN = 0.022        # rad kept off the tighter (left) end-stop
ARMJ = [f"{s} Arm {k}" for s in ("Left", "Right") for k in range(1, 8)]


class Model:
    """The MJCF, with the handful of lookups this experiment needs."""

    def __init__(self, path=MJCF):
        self.m = mujoco.MjModel.from_xml_path(path)
        self.d = mujoco.MjData(self.m)
        self.jid = {mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_JOINT, i): i
                    for i in range(self.m.njnt)}
        self.key = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_KEY, "home")

    def qadr(self, name):
        return self.m.jnt_qposadr[self.jid[name]]

    def limits(self, name):
        return tuple(self.m.jnt_range[self.jid[name]])

    def body(self, name):
        return self.d.xpos[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, name)].copy()

    def place(self, arms=None, extra=None):
        """Seed the home keyframe, then set arms (symmetric) and any extra joints."""
        mujoco.mj_resetDataKeyframe(self.m, self.d, self.key)
        if arms:
            for k in range(1, 8):
                v = arms.get(f"arm_{k}", 0.0)
                self.d.qpos[self.qadr(f"Left Arm {k}")] = v
                self.d.qpos[self.qadr(f"Right Arm {k}")] = -v
        for k, v in (extra or {}).items():
            self.d.qpos[self.qadr(k)] = v
        mujoco.mj_forward(self.m, self.d)
        return self

    def gravity_torque(self):
        """Static holding torque per arm joint (Nm): mj_inverse at rest."""
        self.d.qvel[:] = 0
        self.d.qacc[:] = 0
        mujoco.mj_inverse(self.m, self.d)
        return np.array([self.d.qfrc_inverse[self.m.jnt_dofadr[self.jid[n]]] for n in ARMJ])

    def arm_geometry(self, side="l"):
        sh, el, hd = (self.body(f"{side}_shoulder_3"), self.body(f"{side}_elbow"),
                      self.body(f"{side}_hand_mount"))
        u, v = el - sh, hd - el
        bend = np.degrees(np.arccos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1)))
        return {"drop": float(sh[2] - hd[2]), "bend": float(bend),
                "reach": float(np.linalg.norm(hd - sh)), "hand_z": float(hd[2]),
                "hand_fwd": float(-(hd[0] - sh[0])), "hand_out": float(abs(hd[1] - sh[1]))}

    def out_of_limits(self, pose):
        bad = []
        for k in range(1, 8):
            v = pose.get(f"arm_{k}", 0.0)
            for side, sgn in (("Left", 1), ("Right", -1)):
                lo, hi = self.limits(f"{side} Arm {k}")
                if not (lo - 1e-9 <= sgn * v <= hi + 1e-9):
                    bad.append(f"{side} Arm {k}={sgn * v:+.4f} not in [{lo:+.3f},{hi:+.3f}]")
        return bad


# ── E1 · the holding-torque landscape ──────────────────────────────────────
def e1_landscape(mdl, n=41):
    a2lo, a2hi = mdl.limits("Left Arm 2")
    a4lo, a4hi = mdl.limits("Left Arm 4")
    a2 = np.linspace(max(a2lo, -0.30), min(a2hi, 0.30), n)
    a4 = np.linspace(max(a4lo, 0.6), a4hi, n)
    Z = np.empty((len(a4), len(a2)))
    for i, e in enumerate(a4):
        for j, r in enumerate(a2):
            t = mdl.place({"arm_2": r, "arm_4": e}).gravity_torque()
            Z[i, j] = np.abs(t).sum()
    return {"a2": a2, "a4": a4, "tau": Z}


# ── E2 · the pose that ships, against the two it replaces ──────────────────
def e2_candidates(mdl):
    cands = {
        "qpos zero": {},
        "shipped before (per-arm solve)": {
            "arm_1": -0.0044, "arm_2": -0.1745, "arm_3": 0.0811, "arm_4": 1.5795,
            "arm_5": 0.0125, "arm_6": -0.0215, "arm_7": 0.0077},
        "REST (this experiment)": SHIPPED,
    }
    rows = []
    for name, pose in cands.items():
        bad = mdl.out_of_limits(pose)
        clipped = dict(pose)
        for k in range(1, 8):                    # what MuJoCo will actually hold
            lo, hi = mdl.limits(f"Left Arm {k}")
            clipped[f"arm_{k}"] = float(np.clip(pose.get(f"arm_{k}", 0.0), lo, hi))
        tau = mdl.place(clipped).gravity_torque()
        g = mdl.arm_geometry("l")
        rows.append({"name": name, "sum_tau": float(np.abs(tau).sum()),
                     "max_tau": float(np.abs(tau).max()), **g,
                     "out_of_limits": bad})
    return rows


# ── E3 · geometry and torque as a function of the elbow ────────────────────
def e3_elbow_sweep(mdl, n=60):
    lo, hi = mdl.limits("Left Arm 4")
    a4 = np.linspace(max(lo, 0.0), hi, n)
    bend, drop, tau, a2star = [], [], [], []
    for e in a4:
        r = minimize_scalar(
            lambda x: float(np.sum(mdl.place({"arm_2": x, "arm_4": e}).gravity_torque() ** 2)),
            bounds=(mdl.limits("Left Arm 2")[0], 0.05), method="bounded")
        t = mdl.place({"arm_2": r.x, "arm_4": e}).gravity_torque()
        g = mdl.arm_geometry("l")
        bend.append(g["bend"]); drop.append(g["drop"])
        tau.append(float(np.abs(t).sum())); a2star.append(float(r.x))
    return {"a4": a4, "bend": np.array(bend), "drop": np.array(drop),
            "tau": np.array(tau), "a2star": np.array(a2star)}


# ── E4 · the lift's usable stroke ──────────────────────────────────────────
def e4_lift(mdl, n=33):
    lo, hi = mdl.limits("Lift Lower")
    h = np.linspace(lo, hi, n)
    ncon = []
    for v in h:
        mdl.place(SHIPPED, {"Lift Lower": v, "Lift Upper": v})
        ncon.append(int(mdl.d.ncon))
    ncon = np.array(ncon)
    # The grid only brackets the threshold; bisect for it, so the number the
    # pose library is allowed to use is the real one and not a sample point.
    clear = h[ncon == 0]
    if clear.size:
        a = h[h < clear.min()].max() if (h < clear.min()).any() else lo
        b = float(clear.min())
        for _ in range(24):
            mid = 0.5 * (a + b)
            mdl.place(SHIPPED, {"Lift Lower": mid, "Lift Upper": mid})
            if mdl.d.ncon:
                a = mid
            else:
                b = mid
        usable = float(np.ceil(b * 1000) / 1000)        # round UP: it is a floor
    else:
        usable = float("nan")
    return {"h": h, "ncon": ncon, "declared": (float(lo), float(hi)),
            "usable_min": usable}


def figure(e1, e2, e3, e4, out):
    fig, ax = plt.subplots(2, 2, figsize=(10.5, 7.6))

    a = ax[0, 0]
    im = a.pcolormesh(e1["a2"], e1["a4"], np.log10(e1["tau"]), shading="auto", cmap="viridis")
    a.plot([SHIPPED["arm_2"]], [SHIPPED["arm_4"]], "o", ms=8, mfc="none", mec="w", mew=2)
    a.annotate("REST", (SHIPPED["arm_2"], SHIPPED["arm_4"]), color="w",
               xytext=(8, -14), textcoords="offset points", fontsize=9, weight="bold")
    a.plot([0], [0], "x", ms=9, color="crimson", mew=2)
    a.annotate("qpos zero", (0, 0), color="crimson", xytext=(6, 6),
               textcoords="offset points", fontsize=9, weight="bold")
    fig.colorbar(im, ax=a, label=r"$\log_{10}\,\Sigma|\tau|$  [Nm]")
    a.set_xlabel("shoulder roll  arm_2  [rad]")
    a.set_ylabel("elbow  arm_4  [rad]")
    a.set_title("E1 · gravity holding torque of the two arms")

    a = ax[0, 1]
    names = [r["name"].replace(" (", "\n(") for r in e2]
    vals = [r["sum_tau"] for r in e2]
    cols = ["crimson", "darkorange", "seagreen"]
    b = a.bar(range(len(vals)), vals, color=cols)
    a.set_yscale("log")
    a.set_xticks(range(len(vals)))
    a.set_xticklabels(names, fontsize=8)
    for r, rect in zip(e2, b):
        a.text(rect.get_x() + rect.get_width() / 2, rect.get_height() * 1.12,
               f"{r['sum_tau']:.2f} Nm\n{r['bend']:.1f}$^\\circ$ bend",
               ha="center", fontsize=8)
    a.set_ylabel(r"$\Sigma|\tau|$  [Nm]   (log)")
    a.set_ylim(top=max(vals) * 4)
    a.set_title("E2 · what each default costs to stand in")

    a = ax[1, 0]
    a.plot(e3["a4"], e3["bend"], color="steelblue", label="elbow bend [deg]")
    a.set_xlabel("elbow  arm_4  [rad]")
    a.set_ylabel("upper arm $\\wedge$ forearm  [deg]", color="steelblue")
    a.axvline(SHIPPED["arm_4"], color="seagreen", ls="--", lw=1.2)
    a.axvline(e3["a4"].max(), color="0.4", ls=":", lw=1.2)
    a.text(e3["a4"].max(), 70, " MJCF stop", rotation=90, va="top", fontsize=8, color="0.3")
    a.text(SHIPPED["arm_4"], 70, " REST", rotation=90, va="top", fontsize=8, color="seagreen")
    a2 = a.twinx()
    a2.plot(e3["a4"], e3["tau"], color="darkorange", label=r"$\Sigma|\tau|$")
    a2.set_ylabel(r"$\Sigma|\tau|$  [Nm]", color="darkorange")
    a2.set_yscale("log")
    a.set_title("E3 · the bend floor, and where the torque minimum sits")

    a = ax[1, 1]
    a.plot(e4["h"], e4["ncon"], color="crimson", marker=".")
    a.axvline(e4["usable_min"], color="seagreen", ls="--", lw=1.2)
    a.text(e4["usable_min"], max(e4["ncon"]) * 0.75,
           f" collision-free\n above {e4['usable_min']:.3f} m", fontsize=8, color="seagreen")
    a.set_xlabel("lift stage height [m]   (declared stroke %.3f-%.3f)" % e4["declared"])
    a.set_ylabel("self-contacts at rest")
    a.set_title("E4 · MODEL DEFECT: the lift's lower half self-penetrates")

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="coarser sweeps")
    args = ap.parse_args()
    n = 17 if args.quick else 41

    os.makedirs(FIGS, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)
    mdl = Model()

    e1 = e1_landscape(mdl, n)
    e2 = e2_candidates(mdl)
    e3 = e3_elbow_sweep(mdl, 24 if args.quick else 60)
    e4 = e4_lift(mdl, 17 if args.quick else 33)

    # the optimum, to state how far the shipped margin costs
    lo2 = mdl.limits("Left Arm 2")[0]
    hi4 = mdl.limits("Left Arm 4")[1]
    r = minimize_scalar(
        lambda x: float(np.sum(mdl.place({"arm_2": x, "arm_4": hi4}).gravity_torque() ** 2)),
        bounds=(lo2, 0.05), method="bounded")
    tau_at_stop = float(np.abs(mdl.place({"arm_2": r.x, "arm_4": hi4}).gravity_torque()).sum())

    out = os.path.join(FIGS, "rest_pose.pdf")
    figure(e1, e2, e3, e4, out)

    print(f"\nE1  torque landscape over ({len(e1['a2'])} x {len(e1['a4'])}) "
          f"(arm_2, arm_4), minimum {e1['tau'].min():.3f} Nm\n")
    print("E2  candidate rest poses")
    print(f"    {'pose':<32}{'S|tau|':>9}{'max|tau|':>10}{'bend':>8}{'drop':>8}  limits")
    for row in e2:
        print(f"    {row['name']:<32}{row['sum_tau']:>9.3f}{row['max_tau']:>10.3f}"
              f"{row['bend']:>7.1f}d{row['drop']:>8.3f}  "
              f"{'OK' if not row['out_of_limits'] else row['out_of_limits'][0]}")
    print(f"\nE3  elbow stop {hi4:.3f} rad; bend floor {e3['bend'].min():.1f} deg; "
          f"torque at the stop {tau_at_stop:.3f} Nm vs {e2[-1]['sum_tau']:.3f} Nm at "
          f"REST ({ELBOW_MARGIN:.3f} rad of margin)")
    print(f"E4  lift declared {e4['declared'][0]:.3f}-{e4['declared'][1]:.3f} m/stage, "
          f"collision-free only above {e4['usable_min']:.3f} m "
          f"({e4['ncon'].max()} contacts when bottomed)\n")
    print(f"figure: {os.path.relpath(out, REPO)}")

    js = os.path.join(RESULTS, "rest_pose.json")
    with open(js, "w") as f:
        json.dump({"e2": e2,
                   "e3": {k: (v.tolist() if hasattr(v, "tolist") else v)
                          for k, v in e3.items()},
                   "e4": {"declared": e4["declared"], "usable_min": e4["usable_min"],
                          "max_contacts": int(e4["ncon"].max())},
                   "tau_at_stop": tau_at_stop, "shipped": SHIPPED}, f, indent=2)
    print(f"results: {os.path.relpath(js, REPO)}")


if __name__ == "__main__":
    main()
