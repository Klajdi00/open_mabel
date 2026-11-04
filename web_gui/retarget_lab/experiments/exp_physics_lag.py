#!/usr/bin/env python3
"""
exp_physics_lag.py — E7: what NOVA's whole-body controller does to the retargeter's command.

The lab's Physics toggle sends the retargeter's output through NOVA's own stack (NOVA's
gateway -> engine_retarget -> NOVA's caps -> SimWorld). This experiment runs that SAME
chain in-process, deterministically, on a recorded Vision Pro take, at NOVA's control
rate (50 Hz, CTRL_HZ) and MuJoCo's own step, and logs three poses per tick:

    kin   the engine's command (= the lab's kinematic robot; parity 2e-15 rad)
    cmd   after NOVA's caps (arm_retarget: <= 80 % slew, PD error <= 0.3 rated/kp vs
          the MEASURED angle, limits, contact) — what NOVA actually commands
    meas  SimWorld's joints after its setpoint shaper + MuJoCo position servos + gravity

so the gap kin -> meas splits into what the CAPS hold back (kin - cmd) and what the
SERVO lags (cmd - meas). Plus the base: engine twist vs the simulated chassis, with the
closed loop on the measured base (engine_retarget._sync_base) on and off.

    ~/anaconda3/bin/python3 web_gui/retarget_lab/experiments/exp_physics_lag.py [seconds]
Outputs results/exp_physics_lag.json, docs/figures/rtl_e7_*.pdf.
"""
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
REPO = PKG.parents[1]
NOVA = Path(os.environ.get("MABEL_RTL_NOVA", str(REPO.parent / "NOVA")))
FIG = PKG / "docs" / "figures"
RES = HERE / "results"
TAKE = os.environ.get("E7_TAKE", "20261015_170936_test2-wholebodywalking")
ARM = [f"{s}_arm_{i}" for s in ("left", "right") for i in range(1, 8)]
CTRL_HZ = 50.0


def run(seconds, closed_loop=True, instance="rtl_e7"):
    sys.path.insert(0, str(NOVA)); sys.path.insert(0, str(NOVA / "controller"))
    sys.path.insert(0, str(REPO / "web_gui")); sys.path.insert(0, str(REPO / "controller"))
    os.environ.setdefault("NOVA_MABEL_ROOT", str(REPO))
    import mujoco
    from simulation_bridge.engine import SimWorld
    from server.plant import urdf_from_mjcf
    import engine_retarget as ER
    from retarget_lab.server import episodes
    _, frames = episodes.load(TAKE)
    frames = [f for f in frames if f["t"] <= seconds and f["kind"] == "wire"]
    w = SimWorld(instance=instance)
    m, d = w.model, w.data
    act = {urdf_from_mjcf(n): n for n in w._servo_acts}
    jq = {urdf_from_mjcf(m.joint(j).name): int(m.jnt_qposadr[j]) for j in range(m.njnt)}
    rt = ER.EngineRetargeter(log=lambda s: None)
    rt.offline_clock = True          # engine time = the frames' stamps (this loop is not wall-paced)
    rt.set_mode({"method": "retarget", "scope": "global"})
    if not closed_loop:
        rt._sync_base = lambda odom: None
    dt_ph = float(m.opt.timestep)
    n_sub = int(round(1.0 / CTRL_HZ / dt_ph))
    t, fi = 0.0, 0
    log = {k: [] for k in ("t", "kin", "cmd", "meas", "kbase", "pbase", "twist")}
    cmd_hold = {}
    # the gateway's ControlLoop, minus the network: newest frame, poll at 50 Hz, targets + twist to the plant
    while fi < len(frames):
        while fi + 1 < len(frames) and frames[fi + 1]["t"] <= t:
            fi += 1
        rt.feed(frames[fi]["p"])
        rt._frame_t = __import__("time").time()
        meas = {u: float(d.qpos[q]) for u, q in jq.items()}
        snap = w.snapshot()
        meas["_base_odom"] = ER.base_odom("sim", snap["base"])
        rt._prev_t = __import__("time").time() - 1.0 / CTRL_HZ
        out = rt.poll(meas, None)
        if out:
            cmd_hold.update(out)
            w.set_targets({act[u]: v for u, v in out.items() if u in act})
        tw = rt.base_twist
        if any(abs(v) > 1e-9 for v in tw):
            w.set_twist(*tw)
        else:
            w.set_twist(0.0, 0.0, 0.0)
        for _ in range(n_sub):
            w._apply_commands(); mujoco.mj_step(m, d)
        t += n_sub * dt_ph
        if rt._engaged and rt._eng_q:
            log["t"].append(t)
            log["kin"].append(list(rt._eng_q["left"]) + list(rt._eng_q["right"]))
            log["cmd"].append([cmd_hold.get(u, np.nan) for u in ARM])
            log["meas"].append([float(d.qpos[jq[u]]) for u in ARM])
            b = rt.kr.base_pose() or {}
            log["kbase"].append([b.get("x", np.nan), b.get("y", np.nan), b.get("yaw", np.nan)])
            log["pbase"].append([snap["base"]["x"], snap["base"]["y"], snap["base"]["yaw"]])
            log["twist"].append(list(tw))
        if fi + 1 >= len(frames) and t > frames[-1]["t"]:
            break
    w.stop()
    return {k: np.asarray(v, float) for k, v in log.items()}


def best_lag(a, b, dt, kmax):
    best, bk = np.inf, 0
    for k in range(kmax):
        e = np.nanmean((a[: len(a) - k] - b[k:]) ** 2)
        if e < best:
            best, bk = e, k
    return bk * dt, math.sqrt(best)


def main():
    secs = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
    L = run(secs, True)
    O = run(secs, False, instance="rtl_e7b")
    dt = 1.0 / CTRL_HZ
    kin, cmd, meas = L["kin"], L["cmd"], L["meas"]
    caps = np.abs(kin - cmd); servo = np.abs(cmd - meas); total = np.abs(kin - meas)
    lag_caps = best_lag(kin, cmd, dt, 150); lag_servo = best_lag(cmd, meas, dt, 150); lag_tot = best_lag(kin, meas, dt, 150)
    vel = np.abs(np.diff(meas, axis=0)) / dt
    pb = lambda B: np.hypot(B["kbase"][:, 0] - B["pbase"][:, 0], B["kbase"][:, 1] - B["pbase"][:, 1])
    res = {"take": TAKE, "seconds": secs, "ticks": int(len(L["t"])),
           "joint_err_rad": {"caps_p50": float(np.nanmedian(caps)), "caps_p90": float(np.nanpercentile(caps, 90)),
                             "servo_p50": float(np.nanmedian(servo)), "servo_p90": float(np.nanpercentile(servo, 90)),
                             "total_p50": float(np.nanmedian(total)), "total_p90": float(np.nanpercentile(total, 90))},
           "time_lag_s": {"caps": lag_caps[0], "servo": lag_servo[0], "total": lag_tot[0]},
           "residual_after_lag_rad": {"caps": lag_caps[1], "servo": lag_servo[1], "total": lag_tot[1]},
           "per_joint_total_p90": {u: float(np.nanpercentile(total[:, i], 90)) for i, u in enumerate(ARM)},
           "meas_joint_speed_peak": float(np.nanmax(vel)),
           "base_gap_m": {"closed_p50": float(np.nanmedian(pb(L))), "closed_p90": float(np.nanpercentile(pb(L), 90)),
                          "open_p50": float(np.nanmedian(pb(O))), "open_p90": float(np.nanpercentile(pb(O), 90)),
                          "closed_end": float(pb(L)[-1]), "open_end": float(pb(O)[-1])}}
    RES.mkdir(exist_ok=True)
    (RES / "exp_physics_lag.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    figures(L, O, res)


def figures(L, O, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 6.0), sharex=True)
    for ax, j in zip(axes[:2], ("left_arm_4", "right_arm_1")):
        i = ARM.index(j)
        ax.plot(L["t"], L["kin"][:, i], lw=1.0, color="#e8a33d", label="engine command (kinematic)")
        ax.plot(L["t"], L["cmd"][:, i], lw=.8, color="#c678dd", label="after NOVA's caps")
        ax.plot(L["t"], L["meas"][:, i], lw=.9, color="#2a9d8f", label="MuJoCo (measured)")
        ax.set_ylabel(f"{j} [rad]"); ax.legend(frameon=False, fontsize=6, ncol=3)
    ax = axes[2]
    pb = lambda B: np.hypot(B["kbase"][:, 0] - B["pbase"][:, 0], B["kbase"][:, 1] - B["pbase"][:, 1])
    ax.plot(L["t"], pb(L), lw=.9, color="#4c72b0", label="base gap, closed loop on the measured base")
    ax.plot(O["t"], pb(O), lw=.9, color="#e06c75", label="base gap, open loop (before)")
    ax.set_ylabel("engine vs sim base [m]"); ax.set_xlabel("take time [s]"); ax.legend(frameon=False, fontsize=6)
    fig.suptitle("E7 — the retargeter's command through NOVA's whole-body controller (MuJoCo)", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "rtl_e7_physics.pdf"); plt.close(fig)


if __name__ == "__main__":
    main()
