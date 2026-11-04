#!/usr/bin/env python3
"""
wbc_stability.py -- rigorous tip-over experiments for the MABEL whole-body controller.

Question (user, re: HoMMI's CoM-over-base support cost): should we add a CoM-over-base
term, and does it contradict the existing tip-over MOTION MODEL (config.safe_base_step)
or the lazy-base comfort box?

We answer with measurements on the physical MuJoCo model (floating base -> it can really
fall). All numbers/figures are regenerated from scratch by this script; nothing is
hand-tuned. Scene = mabel_home.xml (no scene props), CoM = the robot SUBTREE CoM (so the
workstation prop is excluded), support polygon = the three wheel-contact XY positions.

Four experiments:
  E1 STATIC ENVELOPE  -- CoM support margin vs hand PAYLOAD and lift. Establishes that
                         the bare robot is stable everywhere (arms ~10% of mass); only a
                         heavy hand payload threatens static stability.
  E2 DYNAMIC ACCEL    -- the real tip risk: a forward base-accel pulse, raw vs the motion
                         model. The motion model (accel + ZMP, lift-aware) prevents it.
  E3 PAYLOAD REGIME   -- where CoM-over-base helps: a heavy payload, holding the robot up
                         with none / motion-model / CoM-support base-centring.
  E4 CoM ACCURACY     -- the motion model's FIXED CoM constant vs the TRUE live CoM
                         (XY bias + height-vs-lift fit error).

Run:  python experiments/wbc_stability.py        # all + figures (docs/figures/wbc_*.pdf) + tables
"""
import os
import sys
import argparse

import numpy as np
import mujoco

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
from mabel import config as C                           # noqa: E402
from mabel import com_support as cs                      # noqa: E402
from mabel.swerve import swerve_ik                       # noqa: E402

SCENE = os.path.join(_HERE, "..", "..", "simulation", "mabel_mujoco", "scenes", "mabel_home.xml")
FIGDIR = os.path.join(_HERE, "..", "docs", "figures")
FALL_DEG = 25.0
TIMESTEP = 0.004


# =====================================================================
class Robot:
    def __init__(self):
        self.m = mujoco.MjModel.from_xml_path(SCENE)
        self.m.opt.timestep = TIMESTEP
        self.d = mujoco.MjData(self.m)
        self.root = self.m.body(C.CHASSIS_BODY).id
        self._lift_act = self.m.actuator("Lift").id
        self._lp, self._rp = self.m.body("left_palm").id, self.m.body("right_palm").id
        self._m_lp0, self._m_rp0 = self.m.body_mass[self._lp], self.m.body_mass[self._rp]
        self._lift_cmd = 0.0

    def set_payload(self, kg):
        """Split a hand payload [kg] across both palms and refresh the inertial model."""
        self.m.body_mass[self._lp] = self._m_lp0 + kg / 2.0
        self.m.body_mass[self._rp] = self._m_rp0 + kg / 2.0
        mujoco.mj_setConst(self.m, self.d)

    def reset(self, lift=0.0, settle=1.6):
        """Place + COMMAND the lift to `lift` and settle to true equilibrium before any
        manoeuvre.

        The lift is a cascaded tendon actuator: the Total Lift tendon is the SUM of the
        two stages, so a height of `lift` means `lift / 2` on EACH joint -- writing
        `lift` into both over-extends it to 2x and tips the robot.

        Seeding those joints matters because the model's qpos0 assembles the lift
        half-raised (0.3175 m). Resetting to qpos0 and merely COMMANDING lift=0 makes the
        kp=10000 servo slam the stages down 31.75 cm during the settle, and the lurch
        rolls the whole robot across the floor -- which showed up as a bogus 33.8 cm CoM
        XY "bias" in E4. Starting at the commanded height leaves zero servo error, so the
        robot never moves and the measurement is of the pose we asked for.
        """
        d, m = self.d, self.m
        mujoco.mj_resetData(m, d)
        for jname in (C.LIFT_JOINT, "Lift Upper"):
            try:
                d.joint(jname).qpos[0] = float(lift) / 2.0
            except KeyError:
                pass
        mujoco.mj_forward(m, d)
        self._lift_cmd = float(lift)
        for a in range(m.nu):
            if a == self._lift_act:
                d.ctrl[a] = self._lift_cmd
            elif m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT:
                jid = m.actuator_trnid[a, 0]
                d.ctrl[a] = d.qpos[m.jnt_qposadr[jid]]
            else:
                d.ctrl[a] = 0.0
        for _ in range(int(settle / TIMESTEP)):
            mujoco.mj_step(m, d)

    def lift_height_cmd(self):
        return self._lift_cmd

    def tilt_deg(self):
        z = self.d.xmat[self.root].reshape(3, 3)[:, 2]
        return float(np.degrees(np.arccos(np.clip(z[2], -1.0, 1.0))))

    def margin(self):
        return cs.support_margin(cs.robot_com(self.m, self.d)[:2],
                                 cs.support_polygon(self.m, self.d))

    def chassis_xy(self):
        return self.d.body(C.CHASSIS_BODY).xpos[:2].copy()

    def drive(self, vx, vy, w):
        swerve_ik(vx, vy, w, self.d)

    def step(self):
        mujoco.mj_step(self.m, self.d)


# =====================================================================
# The "live CoM" motion model: config.safe_base_step against the TRUE CoM.
def safe_base_step_live(robot, vx, vy, w, tvx, tvy, tw, dt):
    com = cs.robot_com(robot.m, robot.d)
    e_fwd, e_left = cs.com_offset_base(robot.m, robot.d)
    com_xy = np.array([-e_fwd, -e_left])
    z_eff = (float(com[2]) - C.COM_Z0) / C.COM_DZDLIFT
    z_eff = min(max(z_eff, C.LIFT_RANGE[0]), C.LIFT_RANGE[1])
    saved_xy, saved_edges = C.COM_XY, C._SUPPORT_EDGES
    try:
        C.COM_XY = com_xy
        C._SUPPORT_EDGES = C._support_edges()
        return C.safe_base_step(vx, vy, w, tvx, tvy, tw, dt, z_eff)
    finally:
        C.COM_XY, C._SUPPORT_EDGES = saved_xy, saved_edges


def _save(name):
    os.makedirs(FIGDIR, exist_ok=True)
    plt.savefig(os.path.join(FIGDIR, name), format="pdf", bbox_inches="tight")
    plt.close()
    print(f"  figure -> docs/figures/{name}")


# =====================================================================
# E1 -- STATIC ENVELOPE: CoM support margin vs payload and lift (settle, no manoeuvre).
# =====================================================================
def e1_static_envelope(robot):
    payloads = np.linspace(0, 28, 15)
    lifts = [0.0, 0.3, 0.635]
    margin = {lz: [] for lz in lifts}
    fell = {lz: [] for lz in lifts}
    for lz in lifts:
        for kg in payloads:
            robot.set_payload(float(kg))
            robot.reset(lift=lz)
            # hold 1.5 s; record whether it stays up and the settled margin
            pk = 0.0
            for _ in range(int(1.5 / TIMESTEP)):
                robot.drive(0, 0, 0)
                robot.step()
                pk = max(pk, robot.tilt_deg())
            margin[lz].append(robot.margin() * 100.0)
            fell[lz].append(pk > FALL_DEG)
    robot.set_payload(0.0)
    # tip threshold = first payload that falls, per lift
    thr = {}
    for lz in lifts:
        idx = next((i for i, f in enumerate(fell[lz]) if f), None)
        thr[lz] = float(payloads[idx]) if idx is not None else float("inf")
    plt.figure(figsize=(6.4, 4.2))
    for lz in lifts:
        mvals = [m if not f else np.nan for m, f in zip(margin[lz], fell[lz])]
        plt.plot(payloads, mvals, marker="o", ms=4, label=f"lift {lz:.2f} m")
    plt.axhline(0.0, ls="--", c="k", lw=1, label="tip line (margin = 0)")
    plt.xlabel("hand payload [kg]")
    plt.ylabel("CoM support margin [cm]")
    plt.title("Static stability envelope: margin vs payload & lift")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    _save("wbc_static_envelope.pdf")
    return {"payloads": payloads.tolist(), "margin0": margin[0.0], "thr": thr,
            "margin_home": margin[0.0][0]}


# =====================================================================
# E2 -- DYNAMIC ACCEL: forward velocity pulse, raw vs motion model, peak tilt vs speed.
# =====================================================================
def _accel_pulse(robot, strat, vcmd, lift, dur=1.4, t_on=(0.2, 0.8)):
    robot.reset(lift=lift)
    v = (0.0, 0.0, 0.0)
    peak = 0.0
    for k in range(int(dur / TIMESTEP)):
        t = k * TIMESTEP
        tv = vcmd if t_on[0] <= t < t_on[1] else 0.0      # accelerate to vcmd, then stop
        if strat == "raw":
            v = (tv, 0.0, 0.0)
        elif strat == "motion_model":
            v = C.safe_base_step(v[0], v[1], v[2], tv, 0.0, 0.0, TIMESTEP, lift)
        elif strat == "motion_live":
            v = safe_base_step_live(robot, v[0], v[1], v[2], tv, 0.0, 0.0, TIMESTEP)
        robot.drive(*v)
        robot.step()
        peak = max(peak, robot.tilt_deg())
        if robot.tilt_deg() > 80:
            break
    return peak


def e2_dynamic_accel(robot):
    speeds = np.linspace(0.2, 1.8, 9)
    lift = 0.3
    strats = ["raw", "motion_model", "motion_live"]
    res = {s: [] for s in strats}
    for s in strats:
        for v in speeds:
            res[s].append(_accel_pulse(robot, s, float(v), lift))
    plt.figure(figsize=(6.4, 4.2))
    lab = {"raw": "raw (no protection)", "motion_model": "motion model (fixed CoM)",
           "motion_live": "motion model (live CoM)"}
    for s in strats:
        plt.plot(speeds, res[s], marker="o", ms=4, label=lab[s])
    plt.axhline(FALL_DEG, ls="--", c="k", lw=1, label=f"fall threshold ({FALL_DEG:.0f}°)")
    plt.xlabel("commanded base speed pulse [m/s]")
    plt.ylabel("peak chassis tilt [deg]")
    plt.title(f"Dynamic base-acceleration tip (lift {lift:.2f} m)")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    _save("wbc_dynamic_accel.pdf")
    return {"speeds": speeds.tolist(), "res": res, "lift": lift}


# =====================================================================
# E3 -- PAYLOAD REGIME: hold the robot up under a heavy payload; none vs motion vs CoM-support.
# =====================================================================
def e3_payload_regime(robot):
    # safe (no-payload) home CoM offset = the CoM-support set-point r*
    robot.set_payload(0.0)
    robot.reset(lift=0.3)
    ref_fwd, ref_left = cs.com_offset_base(robot.m, robot.d)
    payloads = np.linspace(8, 26, 10)
    strats = ["none", "motion_model", "com_support"]
    fell = {s: [] for s in strats}
    for s in strats:
        for kg in payloads:
            robot.set_payload(float(kg))
            robot.reset(lift=0.3)
            sup = cs.CoMBaseSupport(robot.m, robot.d)
            sup.ref_fwd, sup.ref_left = ref_fwd, ref_left      # hold the SAFE offset
            sup.bx = sup.by = 0.015
            sup.kp = 4.0
            v = (0.0, 0.0, 0.0)
            pk = 0.0
            for k in range(int(3.0 / TIMESTEP)):
                if s == "com_support":
                    cx, cy, _ = sup.command(robot.d)
                    v = C.safe_base_step(v[0], v[1], v[2], cx, cy, 0.0, TIMESTEP, 0.3)
                    robot.drive(*v)
                elif s == "motion_model":
                    robot.drive(0, 0, 0)        # accel-only protection cannot move under a static CoM
                else:
                    robot.drive(0, 0, 0)
                robot.step()
                pk = max(pk, robot.tilt_deg())
                if robot.tilt_deg() > 80:
                    break
            fell[s].append(pk > FALL_DEG)
    robot.set_payload(0.0)
    thr = {}
    for s in strats:
        idx = next((i for i, f in enumerate(fell[s]) if f), None)
        thr[s] = float(payloads[idx]) if idx is not None else float("inf")
    plt.figure(figsize=(6.4, 4.2))
    labels = {"none": "no base support", "motion_model": "motion model (accel only)",
              "com_support": "CoM-support base centring"}
    bars = [thr[s] if np.isfinite(thr[s]) else payloads[-1] + 2 for s in strats]
    cols = ["#c0392b", "#e67e22", "#27ae60"]
    plt.bar(range(len(strats)), bars, color=cols)
    plt.xticks(range(len(strats)), [labels[s] for s in strats], rotation=15, ha="right", fontsize=8)
    plt.ylabel("payload at first tip-over [kg]  (higher = safer)")
    plt.title("Heavy-payload static stability: where CoM-over-base helps (lift 0.30 m)")
    plt.tight_layout()
    _save("wbc_payload_regime.pdf")
    return {"payloads": payloads.tolist(), "thr": thr}


# =====================================================================
# E4 -- CoM ACCURACY: the motion model's fixed CoM constant vs the true live CoM.
# =====================================================================
def e4_com_accuracy(robot):
    robot.set_payload(0.0)
    lifts = np.linspace(0.0, 0.635, 9)
    true_h, fit_h = [], []
    for lz in lifts:
        robot.reset(lift=float(lz))
        true_h.append(float(cs.robot_com(robot.m, robot.d)[2]))
        fit_h.append(float(C.com_height(lz)))
    robot.reset(lift=0.0)
    com0 = cs.robot_com(robot.m, robot.d)
    xy_bias = float(np.hypot(com0[0] - C.COM_XY[0], com0[1] - C.COM_XY[1]))
    h_err = float(max(abs(t - f) for t, f in zip(true_h, fit_h)))
    plt.figure(figsize=(6.4, 4.2))
    plt.plot(lifts, true_h, marker="o", ms=4, label="true CoM height (measured)")
    plt.plot(lifts, fit_h, marker="s", ms=4, ls="--", label="motion-model fit (config.com_height)")
    plt.xlabel("lift command [m]")
    plt.ylabel("CoM height [m]")
    plt.title(f"CoM-height model vs. truth (max error {h_err*100:.1f} cm)")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    _save("wbc_com_accuracy.pdf")
    return {"xy_bias_cm": xy_bias * 100, "h_err_cm": h_err * 100,
            "true_h_full": true_h[-1], "fit_h_full": fit_h[-1]}


# =====================================================================
def _tbl(title, header, rows):
    print(f"\n=== {title} ===")
    print("  " + "  ".join(f"{h:>26s}" for h in header))
    for r in rows:
        print("  " + "  ".join(f"{c:>26s}" for c in r))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    robot = Robot()
    print(f"MABEL WBC stability  (mabel_home.xml, dt {TIMESTEP*1e3:.0f} ms, "
          f"robot {robot.m.body_subtreemass[robot.root]:.1f} kg, fall>{FALL_DEG:.0f} deg)")

    if args.only in (None, "1"):
        e1 = e1_static_envelope(robot)
        rows = [[f"lift {lz:.2f} m",
                 (f"{e1['thr'][lz]:.0f} kg" if np.isfinite(e1['thr'][lz]) else ">28 kg")]
                for lz in (0.0, 0.3, 0.635)]
        print(f"\n  home margin (0 kg): {e1['margin_home']:.1f} cm")
        _tbl("E1  STATIC ENVELOPE: static tip-over payload threshold",
             ["lift", "payload at first tip"], rows)

    if args.only in (None, "2"):
        e2 = e2_dynamic_accel(robot)
        sp = e2["speeds"]
        rows = []
        for s in ["raw", "motion_model", "motion_live"]:
            mx = max(e2["res"][s])
            tip = next((f"{sp[i]:.1f} m/s" for i, t in enumerate(e2["res"][s]) if t > FALL_DEG), "never")
            rows.append([s, f"{mx:.1f}", tip])
        _tbl(f"E2  DYNAMIC ACCEL (lift {e2['lift']:.2f} m): peak tilt over a speed pulse",
             ["strategy", "max peak tilt [deg]", "tips at pulse>="], rows)

    if args.only in (None, "3"):
        e3 = e3_payload_regime(robot)
        rows = [[s, (f"{e3['thr'][s]:.0f} kg" if np.isfinite(e3['thr'][s]) else ">26 kg")]
                for s in ["none", "motion_model", "com_support"]]
        _tbl("E3  PAYLOAD REGIME (lift 0.30 m): payload tip threshold per base strategy",
             ["base strategy", "payload at first tip"], rows)

    if args.only in (None, "4"):
        e4 = e4_com_accuracy(robot)
        print("\n=== E4  CoM ACCURACY (fixed motion-model constant vs. true live CoM) ===")
        print(f"  fixed-CoM XY bias at home       : {e4['xy_bias_cm']:.1f} cm")
        print(f"  CoM-height fit max error        : {e4['h_err_cm']:.1f} cm")
        print(f"  at full lift: true {e4['true_h_full']:.3f} m vs fit {e4['fit_h_full']:.3f} m")

    print("\nDone. Figures in docs/figures/wbc_*.pdf")


if __name__ == "__main__":
    main()
