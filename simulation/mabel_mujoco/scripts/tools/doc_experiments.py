"""
Doc-specific experiment figures for the MABEL technical papers:
  motion_model      -> tip_caps_vs_lift.pdf, swerve_track.pdf   (+ reuse tip_envelope.pdf)
  arm_control       -> gravcomp_hold.pdf                         (+ reuse backdrivability.pdf)
  whole_body_control-> ik_convergence.pdf, armsonly_freeze.pdf

    python scripts/tools/doc_experiments.py
"""
import os, sys, math
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from mabel import config as C
from mabel.dynamics import UpperBodyDynamics, SimPlant, ComplianceController, COMPLIANT_ARM_JOINTS
from mabel.wbc import WholeBodyIK
from mabel.swerve import swerve_ik

FIG = os.path.normpath(os.path.join(HERE, "..", "..", "docs", "figures"))
os.makedirs(FIG, exist_ok=True)
XML = os.path.join(HERE, "..", C.SCENE_XML)
plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "lines.linewidth": 1.3})
COL = (3.4, 2.5)


def model():
    m = mujoco.MjModel.from_xml_path(XML); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    return m, d


# ---- motion_model: tip-safe caps vs lift height (analytic) -------------------
def fig_tip_caps_vs_lift():
    z = np.linspace(0, 0.635, 40)
    vf, vb, vl, af, ab, al = [], [], [], [], [], []
    for zz in z:
        a = C.max_decel_dir(zz); s = C.max_speed_dir(zz)
        af.append(a[0]); ab.append(a[1]); al.append(a[2])
        vf.append(s[0]); vb.append(s[1]); vl.append(s[2])
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.5))
    ax[0].plot(z, vf, label="fwd"); ax[0].plot(z, vl, label="lat"); ax[0].plot(z, vb, label="back")
    ax[0].set_xlabel("lift height $z$ [m]"); ax[0].set_ylabel("speed cap [m/s]")
    ax[0].set_title("(a) tip-safe speed"); ax[0].legend(fontsize=6)
    ax[1].plot(z, af, label="fwd"); ax[1].plot(z, al, label="lat"); ax[1].plot(z, ab, label="back")
    ax[1].set_xlabel("lift height $z$ [m]"); ax[1].set_ylabel("accel cap [m/s$^2$]")
    ax[1].set_title("(b) tip-safe acceleration"); ax[1].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(f"{FIG}/tip_caps_vs_lift.pdf"); plt.close(fig)


# ---- motion_model: swerve realizes commanded twist ---------------------------
def fig_swerve_track():
    m, d = model(); dt = m.opt.timestep; cid = m.body("mabel_chassis").id
    ts, cmd_w, act_w, cmd_v, act_v = [], [], [], [], []
    vx = vy = w = 0.0
    for k in range(1200):
        tvx, tvy, tw = (-0.6, 0.0, 1.5) if k < 800 else (0, 0, 0)
        vx, vy, w = C.safe_base_step(vx, vy, w, tvx, tvy, tw, dt, 0.0)
        swerve_ik(vx, vy, w, d); mujoco.mj_step(m, d)
        Rb = d.body(cid).xmat.reshape(3, 3)
        wz = d.qvel[m.joint("base_root").dofadr[0] + 5]   # base yaw rate (world ~ body z)
        vlin = Rb.T @ d.qvel[m.joint("base_root").dofadr[0]:m.joint("base_root").dofadr[0]+3]
        ts.append(k*dt); cmd_w.append(w); act_w.append(wz); cmd_v.append(-vx); act_v.append(-vlin[0])
    fig, ax = plt.subplots(figsize=COL)
    ax.plot(ts, cmd_v, "C0--", label="cmd fwd v"); ax.plot(ts, act_v, "C0", label="actual fwd v")
    ax.plot(ts, cmd_w, "C1--", label="cmd yaw"); ax.plot(ts, act_w, "C1", label="actual yaw")
    ax.set_xlabel("time [s]"); ax.set_ylabel("m/s  ,  rad/s")
    ax.set_title("Swerve twist tracking (fwd + yaw)"); ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(f"{FIG}/swerve_track.pdf"); plt.close(fig)


# ---- arm_control: gravity comp holds vs sags ---------------------------------
def fig_gravcomp_hold():
    fig, ax = plt.subplots(figsize=COL)
    for comp_on, lab, st in [(False, "no comp", "C3"), (True, "gravity comp", "C0")]:
        m, d = model()
        plant = SimPlant(m, d); dyn = UpperBodyDynamics(m, d)
        ctrl = ComplianceController(dyn, plant) if comp_on else None
        if not comp_on:
            for a in COMPLIANT_ARM_JOINTS + ["Torso"]: plant.disable_servo(a)
        z0 = d.body(C.LEFT_PALM).xpos[2]; ts, dz = [], []
        for k in range(1500):
            if comp_on: ctrl.update()
            else: d.qfrc_applied[:] = 0
            mujoco.mj_step(m, d)
            if k % 20 == 0: ts.append(k*m.opt.timestep); dz.append(d.body(C.LEFT_PALM).xpos[2]-z0)
        ax.plot(ts, np.array(dz)*100, st, label=lab)
    ax.set_xlabel("time [s]"); ax.set_ylabel("left-palm height drop [cm]")
    ax.set_title("Gravity compensation: hold vs sag"); ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(f"{FIG}/gravcomp_hold.pdf"); plt.close(fig)


# ---- whole_body_control: IK convergence to a palm step -----------------------
def fig_ik_convergence():
    m, d = model(); ik = WholeBodyIK(m, d); dt = m.opt.timestep
    ik.target_l_pos = ik.target_l_pos + np.array([-0.10, -0.08, 0.06])
    ik.target_r_pos = ik.target_r_pos + np.array([-0.10, 0.08, 0.06])
    ts, el, er = [], [], []
    for k in range(1200):
        ik.apply(ik.solve(), dt); mujoco.mj_step(m, d)
        ts.append(k*dt); el.append(np.linalg.norm(ik.e_pos_l)); er.append(np.linalg.norm(ik.e_pos_r))
    fig, ax = plt.subplots(figsize=COL)
    ax.plot(ts, np.array(el)*100, label="left palm"); ax.plot(ts, np.array(er)*100, label="right palm")
    ax.set_xlabel("time [s]"); ax.set_ylabel("palm position error [cm]")
    ax.set_title("Bimanual IK convergence (12 cm step)"); ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(f"{FIG}/ik_convergence.pdf"); plt.close(fig)


# ---- whole_body_control: arms-only freezes lift+torso ------------------------
def fig_armsonly_freeze():
    fig, ax = plt.subplots(figsize=COL)
    for arms_only, lab, st in [(False, "whole body (M3)", "C0"), (True, "arms only (M2)", "C1")]:
        m, d = model(); ik = WholeBodyIK(m, d); dt = m.opt.timestep
        ts, lift, torso = [], [], []
        q_lift0 = d.joint(C.LIFT_JOINT).qpos[0]; q_tor0 = d.joint("Torso").qpos[0]
        for k in range(1500):
            ik.target_l_pos[2] += 0.0004   # raise left palm target steadily
            ik.apply(ik.solve(arms_only=arms_only), dt, arms_only=arms_only); mujoco.mj_step(m, d)
            if k % 20 == 0:
                ts.append(k*dt)
                lift.append(abs(d.joint(C.LIFT_JOINT).qpos[0]-q_lift0))
                torso.append(abs(d.joint("Torso").qpos[0]-q_tor0))
        ax.plot(ts, np.degrees(torso), st, label=f"torso, {lab}")
        ax.plot(ts, np.array(lift)*100, st+"--", label=f"lift[cm], {lab}")
    ax.set_xlabel("time [s]"); ax.set_ylabel("torso [deg] / lift [cm] motion")
    ax.set_title("Arms-only freezes lift + torso"); ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(f"{FIG}/armsonly_freeze.pdf"); plt.close(fig)


def main():
    fig_tip_caps_vs_lift(); print("tip_caps_vs_lift.pdf")
    fig_swerve_track();     print("swerve_track.pdf")
    fig_gravcomp_hold();    print("gravcomp_hold.pdf")
    fig_ik_convergence();   print("ik_convergence.pdf")
    fig_armsonly_freeze();  print("armsonly_freeze.pdf")
    print("done ->", FIG)


if __name__ == "__main__":
    main()
