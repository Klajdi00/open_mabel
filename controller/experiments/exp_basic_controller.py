#!/usr/bin/env python3
"""Characterise the pure-PD task-space arm controller (`mabel.basic`).

    python3 controller/experiments/exp_basic_controller.py

Regenerates every figure in `controller/docs/figures/basic_*.pdf` from scratch
and prints the summary tables. Deterministic: fixed seeds, fixed poses, no
wall-clock dependence.

THIS IS THE BASELINE, AND IT IS EXPECTED TO DROOP. Pure PD has no term that
cancels gravity, so a loaded joint settles where `kp * e` balances the bias
torque: the steady-state error is `tau_gravity / kp` by construction. The point
of E1/E2 is to MEASURE that, so the value of adding a gravity feed-forward is a
number rather than an opinion.

  E1  hold error vs kp          — the droop, and its 1/kp law
  E2  droop per joint           — which joints actually carry the load
  E3  static wrist tracking     — reach a target, and the IK/PD error split
  E4  moving wrist tracking     — RMS lag vs target speed
  E5  IK conditioning           — residual vs target distance (reach limit)
"""
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from mabel.basic import TaskSpaceArm, ARM_JOINTS  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
MJCF = os.path.join(REPO, "simulation", "mabel_mujoco", "scenes", "mabel_home.xml")
FIGS = os.path.join(REPO, "controller", "docs", "figures")
os.makedirs(FIGS, exist_ok=True)

SIDE = "right"
DT_SETTLE = 4.0            # seconds of physics per steady-state measurement

# ── house style (matches the other controller figures) ───────────────────────
plt.rcParams.update({
    "figure.figsize": (7.0, 4.0), "figure.dpi": 120,
    "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 9, "legend.frameon": False,
})
ACCENT, WARN, INK = "#c25b2a", "#e9a679", "#2b2724"


def fresh():
    model = mujoco.MjModel.from_xml_path(MJCF)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def settle(model, data, arm, seconds):
    for _ in range(int(seconds / model.opt.timestep)):
        arm.step()
        mujoco.mj_step(model, data)


def gravity_torque(model, data, joints):
    """The bias (gravity+Coriolis at rest) torque each joint must hold."""
    d = mujoco.MjData(model)
    d.qpos[:] = data.qpos
    d.qvel[:] = 0.0
    mujoco.mj_forward(model, d)
    return {j: float(d.qfrc_bias[int(model.joint(j).dofadr[0])]) for j in joints}


# ── E1: steady-state droop vs kp ─────────────────────────────────────────────
def e1_hold_vs_kp():
    print("\nE1  HOLD ERROR vs kp  (pure PD, no gravity term)")
    kps = [25, 50, 100, 200, 400, 800, 1600]
    worst, rms = [], []
    for kp in kps:
        model, data = fresh()
        arm = TaskSpaceArm(model, data, side=SIDE, kp=kp, kd=2.0 * np.sqrt(kp)).attach()
        settle(model, data, arm, DT_SETTLE)
        err = {j: abs(v) for j, v in arm.pd.error().items()}
        worst.append(np.degrees(max(err.values())))
        rms.append(np.degrees(np.sqrt(np.mean([v ** 2 for v in err.values()]))))
        print(f"   kp={kp:5d}  worst={worst[-1]:7.3f} deg   rms={rms[-1]:6.3f} deg")

    fig, ax = plt.subplots()
    ax.loglog(kps, worst, "o-", color=ACCENT, label="worst joint")
    ax.loglog(kps, rms, "s--", color=WARN, label="RMS over the chain")
    ref = np.array(worst[0]) * kps[0] / np.array(kps, dtype=float)
    ax.loglog(kps, ref, ":", color=INK, alpha=0.6, label=r"ideal $1/k_p$")
    ax.set_xlabel(r"proportional gain $k_p$  [N·m/rad]")
    ax.set_ylabel("steady-state hold error  [deg]")
    ax.set_title("E1 · pure PD droops as $1/k_p$ — nothing cancels gravity")
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "basic_e1_hold_vs_kp.pdf"))
    plt.close(fig)
    return kps, worst


# ── E2: which joints carry the load ──────────────────────────────────────────
def e2_droop_per_joint(kp=100.0):
    print(f"\nE2  DROOP PER JOINT  (kp={kp:g})")
    model, data = fresh()
    arm = TaskSpaceArm(model, data, side=SIDE, kp=kp, kd=2.0 * np.sqrt(kp)).attach()
    tau_g = gravity_torque(model, data, arm.joints)
    settle(model, data, arm, DT_SETTLE)
    err = arm.pd.error()
    names = [j.replace(f"{SIDE.capitalize()} Arm ", "J") for j in arm.joints]
    deg = [np.degrees(err[j]) for j in arm.joints]
    pred = [np.degrees(tau_g[j] / kp) for j in arm.joints]
    for j, d_, p_ in zip(names, deg, pred):
        print(f"   {j:4s}  measured={d_:+7.3f} deg   predicted(tau_g/kp)={p_:+7.3f} deg")

    x = np.arange(len(names))
    fig, ax = plt.subplots()
    ax.bar(x - 0.2, deg, 0.4, color=ACCENT, label="measured droop")
    ax.bar(x + 0.2, pred, 0.4, color=WARN, label=r"predicted $\tau_g/k_p$")
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_ylabel("steady-state error  [deg]")
    ax.set_title("E2 · the droop IS the gravity torque the PD cannot cancel")
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "basic_e2_droop_per_joint.pdf"))
    plt.close(fig)
    return deg, pred


# ── E3: static wrist tracking ────────────────────────────────────────────────
def e3_static_tracking(kp=400.0):
    print(f"\nE3  STATIC WRIST TRACKING  (kp={kp:g})")
    model, data = fresh()
    arm = TaskSpaceArm(model, data, side=SIDE, kp=kp, kd=2.0 * np.sqrt(kp)).attach()
    home = arm.palm().copy()
    offsets = [(0.10, 0, 0), (-0.10, 0, 0), (0, 0.10, 0), (0, -0.10, 0),
               (0, 0, 0.10), (0, 0, -0.10), (0.08, 0.08, 0.08)]
    rows = []
    for off in offsets:
        model, data = fresh()
        arm = TaskSpaceArm(model, data, side=SIDE, kp=kp, kd=2.0 * np.sqrt(kp)).attach()
        tgt = home + np.array(off)
        arm.set_wrist_target(tgt)
        settle(model, data, arm, DT_SETTLE)
        rows.append((off, arm.ik_err * 1000.0, arm.task_error() * 1000.0,
                     np.degrees(arm.joint_error())))
        print(f"   d={str(off):>22s}  ik_resid={rows[-1][1]:6.2f} mm   "
              f"palm_err={rows[-1][2]:7.2f} mm   joint_err={rows[-1][3]:6.2f} deg")

    labels = ["+x", "-x", "+y", "-y", "+z", "-z", "diag"]
    ikr = [r[1] for r in rows]; tsk = [r[2] for r in rows]
    x = np.arange(len(labels))
    fig, ax = plt.subplots()
    ax.bar(x - 0.2, ikr, 0.4, color=INK, alpha=0.75, label="IK residual (solver)")
    ax.bar(x + 0.2, tsk, 0.4, color=ACCENT, label="palm error (solver + PD)")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("error  [mm]")
    ax.set_title("E3 · where the error lives: the IK found it, the PD did not hold it")
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "basic_e3_static_tracking.pdf"))
    plt.close(fig)
    return rows


# ── E4: moving target ────────────────────────────────────────────────────────
def e4_moving_target(kp=400.0):
    print(f"\nE4  MOVING WRIST TRACKING  (circle, kp={kp:g})")
    speeds = [0.02, 0.05, 0.10, 0.20, 0.40]      # m/s along a 6 cm circle
    R, out = 0.06, []
    for v in speeds:
        model, data = fresh()
        arm = TaskSpaceArm(model, data, side=SIDE, kp=kp, kd=2.0 * np.sqrt(kp)).attach()
        c = arm.palm().copy()
        arm.set_wrist_target(c); settle(model, data, arm, 1.5)
        omega, errs = v / R, []
        n = int(6.0 / model.opt.timestep)
        for k in range(n):
            t = k * model.opt.timestep
            arm.set_wrist_target(c + np.array([0.0, R * np.cos(omega * t) - R,
                                               R * np.sin(omega * t)]))
            arm.step(); mujoco.mj_step(model, data)
            if t > 2.0:
                errs.append(arm.task_error())
        rms = float(np.sqrt(np.mean(np.square(errs)))) * 1000.0
        mx = float(np.max(errs)) * 1000.0
        out.append((v, rms, mx))
        print(f"   v={v:.2f} m/s   rms={rms:7.2f} mm   max={mx:7.2f} mm")

    fig, ax = plt.subplots()
    ax.plot([o[0] for o in out], [o[1] for o in out], "o-", color=ACCENT, label="RMS")
    ax.plot([o[0] for o in out], [o[2] for o in out], "s--", color=WARN, label="max")
    ax.set_xlabel("target speed  [m/s]"); ax.set_ylabel("palm tracking error  [mm]")
    ax.set_title("E4 · tracking lag grows with speed (no feed-forward)")
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "basic_e4_moving_target.pdf"))
    plt.close(fig)
    return out


# ── E5: IK conditioning vs reach ─────────────────────────────────────────────
def e5_reach_limit():
    print("\nE5  IK RESIDUAL vs TARGET DISTANCE  (damping keeps it bounded)")
    model, data = fresh()
    arm = TaskSpaceArm(model, data, side=SIDE).attach()
    home = arm.palm().copy()
    shoulder = np.array(data.body(f"{SIDE[0]}_wrist").xpos, dtype=float)
    direction = home - shoulder
    direction /= max(np.linalg.norm(direction), 1e-9)
    dists = np.linspace(0.0, 0.60, 25)
    resid = []
    for d_ in dists:
        q, err = arm.ik.solve(home + direction * d_, q0=arm.ik.q_now())
        resid.append(err * 1000.0)
    for d_, r in list(zip(dists, resid))[::6]:
        print(f"   reach +{d_:.2f} m   residual={r:8.2f} mm")

    fig, ax = plt.subplots()
    ax.plot(dists * 100, resid, "-", color=ACCENT)
    ax.set_xlabel("target distance beyond the home palm position  [cm]")
    ax.set_ylabel("IK residual  [mm]")
    ax.set_title("E5 · past the reach limit the solver saturates, it does not diverge")
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "basic_e5_reach_limit.pdf"))
    plt.close(fig)
    return dists, resid


# ── E6: pure PD vs gravity feed-forward, head to head ────────────────────────
def e6_gravity_vs_pure():
    print("\nE6  PURE PD  vs  + GRAVITY FEED-FORWARD")
    kps = [50, 100, 200, 400, 800]
    hold = {False: [], True: []}
    task = {False: [], True: []}
    for g in (False, True):
        for kp in kps:
            for meas in ("hold", "task"):
                model, data = fresh()
                arm = TaskSpaceArm(model, data, side=SIDE, kp=kp,
                                   kd=2.0 * np.sqrt(kp), gravity=g).attach()
                # feed the MEASURED torso lean, as the robot would (encoders/IMU)
                pid = int(model.body_parentid[
                    model.body(model.joint(ARM_JOINTS[SIDE][0]).bodyid[0]).id])
                arm.set_base_orientation(np.array(data.body(pid).xmat).reshape(3, 3))
                if meas == "task":
                    arm.set_wrist_target(arm.palm().copy() + np.array([0.08, 0.08, 0.08]))
                settle(model, data, arm, DT_SETTLE)
                if meas == "hold":
                    hold[g].append(np.degrees(max(abs(v) for v in arm.pd.error().values())))
                else:
                    task[g].append(arm.task_error() * 1000.0)
    print("     kp | hold PD  hold +g |  palm PD   palm +g |  improvement")
    for i, kp in enumerate(kps):
        imp = task[False][i] / max(task[True][i], 1e-9)
        print(f"   {kp:4d} | {hold[False][i]:7.3f}  {hold[True][i]:7.3f} |"
              f" {task[False][i]:8.2f}  {task[True][i]:8.2f} | {imp:6.1f}x")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 3.8))
    a1.loglog(kps, hold[False], "o-", color=ACCENT, label="pure PD")
    a1.loglog(kps, hold[True], "s-", color=INK, label="+ gravity FF")
    a1.set_xlabel(r"$k_p$"); a1.set_ylabel("hold error [deg]")
    a1.set_title("joint space"); a1.legend()
    a2.loglog(kps, task[False], "o-", color=ACCENT, label="pure PD")
    a2.loglog(kps, task[True], "s-", color=INK, label="+ gravity FF")
    a2.set_xlabel(r"$k_p$"); a2.set_ylabel("palm error [mm]")
    a2.set_title("task space"); a2.legend()
    fig.suptitle("E6 · the gravity term buys accuracy that raising $k_p$ cannot")
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "basic_e6_gravity_vs_pure.pdf"))
    plt.close(fig)
    return kps, hold, task


# ── E7: the gravity model matches MuJoCo without touching the simulator ──────
def e7_model_fidelity():
    print("\nE7  ANALYTIC GRAVITY  vs  MuJoCo qfrc_bias  (numpy only at runtime)")
    from mabel.basic.gravity import ArmGravity
    model, data = fresh()
    jn = ARM_JOINTS[SIDE]
    qadr = [int(model.joint(n).qposadr[0]) for n in jn]
    dof = [int(model.joint(n).dofadr[0]) for n in jn]
    lo = np.array([model.joint(n).range[0] for n in jn])
    hi = np.array([model.joint(n).range[1] for n in jn])
    pid = int(model.body_parentid[model.body(model.joint(jn[0]).bodyid[0]).id])
    G = ArmGravity(SIDE)
    rng = np.random.default_rng(7)
    err, mag = [], []
    for _ in range(300):
        q = rng.uniform(lo, hi)
        data.qpos[:] = model.qpos0
        for a, v in zip(qadr, q):
            data.qpos[a] = v
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        truth = np.array([data.qfrc_bias[i] for i in dof])
        base_R = np.array(data.body(pid).xmat).reshape(3, 3)
        err.append(np.abs(G.torque_array(q, base_R=base_R) - truth))
        mag.append(np.abs(truth))
    err, mag = np.array(err), np.array(mag)
    print(f"   max |error| = {err.max():.2e} N*m   (torques up to {mag.max():.1f} N*m)")
    fig, ax = plt.subplots()
    ax.scatter(mag.ravel(), err.ravel(), s=4, color=ACCENT, alpha=0.4)
    ax.set_xlabel("|MuJoCo gravity torque|  [N*m]")
    ax.set_ylabel("|analytic - MuJoCo|  [N*m]")
    ax.set_title("E7 · model + measured angles reproduce qfrc_bias to machine eps")
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "basic_e7_model_fidelity.pdf"))
    plt.close(fig)
    return err.max()


if __name__ == "__main__":
    t0 = time.time()
    print("=" * 72)
    print("BASELINE: pure PD + task-space wrist tracking  (mabel.basic)")
    print("NO gravity compensation — the droop below is the thing to beat.")
    print("=" * 72)
    kps, worst = e1_hold_vs_kp()
    e2_droop_per_joint()
    e3_static_tracking()
    e4_moving_target()
    e5_reach_limit()
    e6_gravity_vs_pure()
    e7_model_fidelity()
    print("\n" + "=" * 72)
    print(f"figures -> controller/docs/figures/basic_*.pdf   ({time.time()-t0:.0f}s)")
    print("=" * 72)
