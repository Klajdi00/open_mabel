#!/usr/bin/env python3
"""
Regenerate every figure for the per-subpackage technical papers
(docs/motion_model.tex, docs/dynamic_models.tex, docs/whole_body_controller.tex)
from scratch, and print the summary metrics the papers cite.

    python3 controller/experiments/doc_module_figures.py

All simulation experiments run headless MuJoCo on the bare-robot scene
(scenes/mabel_home.xml — robot + floor, no props), fixed seed, and write
vector PDFs into controller/docs/figures/. Analytic figures use the same
library code the robot runs (mabel.motion_model, mabel.whole_body_controller),
never a re-implementation.

Figures
-------
  mm_envelope.pdf        tip-safe speed / decel caps vs lift height (per direction)
  mm_zmp_usage.pdf       steady ZMP usage over (vx, omega) at low / full lift
  dm_gravity_hold.pdf    gravity-comp hold vs uncompensated sag (max joint drift)
  dm_observer_push.pdf   momentum-observer residual + force estimate under a push
  wbctrl_pd_droop.pdf    Day-1 PD droop, with vs without gravity feedforward
  wbctrl_admittance.pdf  base admittance step response (deadband, v_ss, coast)
  wbctrl_swerve_choke.pdf  anti-slip drive choke vs steering error
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

import mujoco  # noqa: E402

import mabel.config as C                                            # noqa: E402
from mabel.motion_model import tip_over as T                        # noqa: E402
from mabel.plant import SimPlant                                    # noqa: E402
from mabel.dynamic_models import (ArmsDynamicModel,                 # noqa: E402
                                  UpperBodyDynamicModel,
                                  WholeBodyDynamicModel, region_joints)
from mabel.whole_body_controller import (JointPositionController,   # noqa: E402
                                         GravityCompController)
from mabel.whole_body_controller.admittance import BaseAdmittance   # noqa: E402

SCENE = os.path.join(_HERE, "..", "..", "simulation", "mabel_mujoco",
                     "scenes", "mabel_home.xml")
FIGDIR = os.path.join(_HERE, "..", "docs", "figures")
SEED = 7

# Simple consistent styling (matches the existing controller experiment figures).
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 120, "savefig.bbox": "tight"})
COL = {"fwd": "#1f77b4", "back": "#17becf", "lat": "#ff7f0e",
       "a": "#1f77b4", "b": "#d62728", "c": "#2ca02c"}

METRICS = []                # (name, value) rows printed at the end


def _save(fig, name):
    os.makedirs(FIGDIR, exist_ok=True)
    path = os.path.join(FIGDIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  figure -> docs/figures/{name}")


def _load():
    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for _ in range(800):                     # settle to gravity equilibrium
        mujoco.mj_step(model, data)
    return model, data


# ── motion model ──────────────────────────────────────────────────────────────

def fig_mm_envelope():
    global METRICS
    zs = np.linspace(0, C.LIFT_RANGE[1], 60)
    speeds = np.array([T.max_speed_dir(z) for z in zs])
    decels = np.array([T.max_decel_dir(z) for z in zs])
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 2.8))
    for k, lab in enumerate(("fwd", "back", "lat")):
        axs[0].plot(zs, speeds[:, k], color=COL[lab], label=lab)
        axs[1].plot(zs, decels[:, k], color=COL[lab], label=lab)
    axs[0].set_xlabel("lift $z$ [m]"); axs[0].set_ylabel("speed cap [m/s]")
    axs[1].set_xlabel("lift $z$ [m]"); axs[1].set_ylabel("decel cap [m/s$^2$]")
    axs[0].legend(); axs[1].legend()
    axs[0].set_title("tip-safe speed caps"); axs[1].set_title("tip-safe decel caps")
    _save(fig, "mm_envelope.pdf")
    METRICS += [("speed cap fwd  z=0 / z=max [m/s]",
                 f"{speeds[0,0]:.2f} / {speeds[-1,0]:.2f}"),
                ("decel cap fwd  z=0 / z=max [m/s^2]",
                 f"{decels[0,0]:.2f} / {decels[-1,0]:.2f}")]


def fig_mm_zmp_usage():
    v = np.linspace(-1.0, 1.0, 121)
    w = np.linspace(-2.5, 2.5, 121)
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.0), sharey=True)
    for ax, z in zip(axs, (0.0, C.LIFT_RANGE[1])):
        U = np.array([[T.tip_usage(vx, 0.0, om, 0, 0, 0, z) for vx in v]
                      for om in w])
        cs = ax.contourf(v, w, np.minimum(U, 2.0), levels=20, cmap="viridis")
        ax.contour(v, w, U, levels=[1.0], colors="red", linewidths=1.4)
        ax.set_xlabel("$v_x$ [m/s]")
        ax.set_title(f"steady ZMP usage, lift $z$={z:.2f} m")
    axs[0].set_ylabel(r"$\omega$ [rad/s]")
    fig.colorbar(cs, ax=axs, shrink=0.9, label="usage $u$ (red: $u=1$)")
    _save(fig, "mm_zmp_usage.pdf")


# ── dynamic models ────────────────────────────────────────────────────────────

def fig_dm_gravity_hold():
    global METRICS
    seconds = 2.0
    traces = {}
    for mode in ("gravity comp", "no compensation"):
        model, data = _load()
        arms = ArmsDynamicModel(model, data)
        plant = SimPlant(model, data)
        if mode == "gravity comp":
            ctl = GravityCompController(arms, plant)
        else:
            for j in arms.joints:
                plant.disable_servo(j)
            ctl = None
        q0 = np.array([data.joint(j).qpos[0] for j in arms.joints])
        ts, drift = [], []
        dt = model.opt.timestep
        for i in range(int(seconds / dt)):
            if ctl is not None:
                ctl.update()
            else:
                plant.begin()
            mujoco.mj_step(model, data)
            if i % 10 == 0:
                q = np.array([data.joint(j).qpos[0] for j in arms.joints])
                ts.append(i * dt); drift.append(np.abs(q - q0).max())
        traces[mode] = (np.array(ts), np.array(drift))
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    ax.plot(*traces["gravity comp"], color=COL["c"], label="gravity comp ($\\tau=b(q,\\dot q)-D\\dot q$)")
    ax.plot(*traces["no compensation"], color=COL["b"], label="servos off, no comp (sags)")
    ax.set_xlabel("time [s]"); ax.set_ylabel("max joint drift [rad]")
    ax.legend(); ax.set_title("arms region: hold vs sag")
    _save(fig, "dm_gravity_hold.pdf")
    METRICS += [("gravity-comp max drift @2 s [rad]", f"{traces['gravity comp'][1][-1]:.3f}"),
                ("uncompensated sag @2 s [rad]", f"{traces['no compensation'][1][-1]:.3f}")]


def fig_dm_observer_push():
    global METRICS
    model, data = _load()
    arms = ArmsDynamicModel(model, data)
    whole = WholeBodyDynamicModel(model, data)
    ctl = GravityCompController(UpperBodyDynamicModel(model, data),
                                SimPlant(model, data))
    F = np.array([12.0, 0.0, 0.0])
    bid = model.body(C.LEFT_PALM).id
    dt = model.opt.timestep
    t_on, t_off, t_end = 0.5, 1.5, 2.0
    ts, res, fmag, cos = [], [], [], []
    joint = "Left Arm 2"
    for i in range(int(t_end / dt)):
        t = i * dt
        data.xfrc_applied[bid, :3] = F if (t_on <= t < t_off) else 0.0
        ctl.update()
        tau_cmd = data.qfrc_applied + data.qfrc_actuator
        est = arms.external_torques(tau_cmd, dt)
        whole.base_wrench(tau_cmd, dt)
        mujoco.mj_step(model, data)
        if i % 5 == 0:
            ts.append(t); res.append(est[joint])
            if whole.contacts:
                fe = np.asarray(whole.contacts[0]["force"])
                fmag.append(np.linalg.norm(fe))
                cos.append(fe @ F / (np.linalg.norm(fe) * np.linalg.norm(F) + 1e-9))
            else:
                fmag.append(0.0); cos.append(np.nan)
    ts, res, fmag = map(np.asarray, (ts, res, fmag))
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 2.8))
    axs[0].plot(ts, res, color=COL["a"])
    axs[0].axvspan(t_on, t_off, color="0.85")
    axs[0].set_xlabel("time [s]"); axs[0].set_ylabel(f"$r$ on {joint} [Nm]")
    axs[0].set_title("momentum-observer residual")
    axs[1].plot(ts, fmag, color=COL["c"], label=r"$\|\hat F\|$ (localized)")
    axs[1].axhline(np.linalg.norm(F), color=COL["b"], ls="--", label=r"applied $\|F\|$")
    axs[1].axvspan(t_on, t_off, color="0.85")
    axs[1].set_xlabel("time [s]"); axs[1].set_ylabel("force [N]")
    axs[1].legend(); axs[1].set_title("estimated contact force")
    _save(fig, "dm_observer_push.pdf")
    push = (ts >= t_on + 0.2) & (ts < t_off)
    cosv = np.asarray(cos)[push]
    METRICS += [("observer |F| mean during push [N]  (applied 12)",
                 f"{fmag[push].mean():.1f}"),
                ("force direction cosine (mean during push)",
                 f"{np.nanmean(cosv):.2f}")]


# ── whole-body controller ─────────────────────────────────────────────────────

def fig_wbctrl_pd_droop():
    global METRICS
    seconds = 1.5
    traces = {}
    for use_ff in (False, True):
        model, data = _load()
        plant = SimPlant(model, data)
        arms = ArmsDynamicModel(model, data)
        ctl = JointPositionController(
            plant, region_joints("arms"), kp=25.0, kd=2.0,
            feedforward=(arms.gravity_ff if use_ff else None))
        ctl.hold_current()
        q_des = dict(ctl.q_des)
        dt = model.opt.timestep
        ts, err = [], []
        for i in range(int(seconds / dt)):
            ctl.update()
            mujoco.mj_step(model, data)
            if i % 10 == 0:
                ts.append(i * dt)
                err.append(max(abs(plant.q(j) - q_des[j]) for j in ctl.joints))
        traces[use_ff] = (np.array(ts), np.array(err))
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    ax.plot(*traces[False], color=COL["b"], label="PD only (Day-1, $k_p{=}25$)")
    ax.plot(*traces[True], color=COL["c"], label="PD + gravity feedforward")
    ax.set_xlabel("time [s]"); ax.set_ylabel("max position error [rad]")
    ax.legend(); ax.set_title("gravity droop of the plain PD")
    _save(fig, "wbctrl_pd_droop.pdf")
    METRICS += [("PD droop @1.5 s, plain / +gravity ff [rad]",
                 f"{traces[False][1][-1]:.3f} / {traces[True][1][-1]:.3f}")]


def fig_wbctrl_admittance():
    global METRICS
    adm = BaseAdmittance()
    dt = 0.01
    Fs = np.zeros(int(9.0 / dt))
    Fs[int(1.0 / dt):int(5.0 / dt)] = 20.0                     # step on, then release
    Fs[int(0.2 / dt):int(0.8 / dt)] = 0.9 * C.ADMIT_F_STATIC   # sub-deadband poke
    ts, vs = [], []
    for i, f in enumerate(Fs):
        vx, _, _ = adm.step(f, 0.0, 0.0, dt)
        ts.append(i * dt); vs.append(vx)
    v_ss = 20.0 / C.ADMIT_LIN_DAMP
    fig, ax = plt.subplots(figsize=(5.6, 2.8))
    ax.plot(ts, vs, color=COL["a"], label="$v_x(t)$")
    ax.plot(ts, Fs / 40.0, color="0.6", lw=0.8, label="$F_x/40$ (drive)")
    ax.axhline(v_ss, color=COL["b"], ls="--", label=f"$v_{{ss}}=F/b_l={v_ss:.2f}$ m/s")
    ax.set_xlabel("time [s]"); ax.set_ylabel("base velocity [m/s]")
    ax.legend(loc="upper right"); ax.set_title("base admittance: deadband, rise, coast-to-stop")
    _save(fig, "wbctrl_admittance.pdf")
    vpk = max(vs)
    t_stop = next(t for t, v in zip(ts, vs) if t > 5.0 and v == 0.0)
    METRICS += [("admittance v_peak vs v_ss [m/s]", f"{vpk:.3f} vs {v_ss:.3f}"),
                ("coast-to-stop after release [s]", f"{t_stop - 5.0:.2f}")]


def fig_wbctrl_swerve_choke():
    err = np.linspace(0, np.pi / 2, 300)
    scale = np.where(err > C.STEER_DEADBAND, 0.0, np.cos(err))
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    ax.plot(np.degrees(err), scale, color=COL["a"])
    ax.axvline(np.degrees(C.STEER_DEADBAND), color=COL["b"], ls="--",
               label=f"deadband {np.degrees(C.STEER_DEADBAND):.0f}$^\\circ$")
    ax.set_xlabel("steering error [deg]"); ax.set_ylabel("drive power scale")
    ax.legend(); ax.set_title("anti-slip cosine choke")
    _save(fig, "wbctrl_swerve_choke.pdf")


def main():
    global METRICS
    np.random.seed(SEED)
    print("== per-module doc figures ==")
    fig_mm_envelope()
    fig_mm_zmp_usage()
    fig_dm_gravity_hold()
    fig_dm_observer_push()
    fig_wbctrl_pd_droop()
    fig_wbctrl_admittance()
    fig_wbctrl_swerve_choke()
    print("\n== summary metrics (cited in the docs) ==")
    for k, v in METRICS:
        print(f"  {k:52s} {v}")


if __name__ == "__main__":
    main()
