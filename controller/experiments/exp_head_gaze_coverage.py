#!/usr/bin/env python3
"""Where does the robot's HEAD CAMERA point when the Vision Pro operator looks around?

THE OPERATOR REQUIREMENT (2026-10-23)
  "when I move my head down the head cam will move down as well" and
  "I need to have 360 view as I move my head — the robot head cam needs to
   project anywhere".

Full immersion shows the head camera on a HEAD-LOCKED window, so the picture is
always in front of the operator's eyes. Whether that picture is the RIGHT one —
the view in the direction they are looking — is decided entirely server-side by
`neck_angles()` and the neck joint limits. This harness measures it end to end
against the canonical MJCF, with no control loop in the way:

    ARKit head pose (yaw psi, pitch theta)  ->  neck_angles(head, I)
      -> clip to the actuator ctrlrange (exactly as engine.py does)
      -> MuJoCo FK  ->  zed_left optical axis (camera -Z) in the chassis frame
      -> achieved (yaw, elevation)  vs  commanded (psi, theta).

Torso and chassis are held at the home keyframe (Upper-body scope, the default),
so this is the coverage the NECK ALONE provides.

    python3 controller/experiments/exp_head_gaze_coverage.py
writes controller/docs/figures/head_gaze_coverage.pdf and prints a summary.
"""
import os
import sys

import numpy as np
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_CTRL = os.path.dirname(_HERE)
_REPO = os.path.dirname(_CTRL)
sys.path.insert(0, _CTRL)
from mabel.teleop_engine.geometry import neck_angles  # noqa: E402

MJCF = os.path.join(_REPO, "simulation", "mabel_mujoco", "models", "mabel_full.xml")
FIG = os.path.join(_CTRL, "docs", "figures", "head_gaze_coverage.pdf")


def arkit_head(yaw_deg, pitch_deg):
    """ARKit head rotation (x right, y up, -z forward): yaw about +Y (+ = left),
    then pitch about the head's +X (+ = up)."""
    y, p = np.radians(yaw_deg), np.radians(pitch_deg)
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    M = np.eye(4)
    M[:3, :3] = Ry @ Rx
    return M


def main():
    m = mujoco.MjModel.from_xml_path(MJCF)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)          # EVERY MjData boot seeds keyframe 0
    cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "zed_left")
    base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "mabel_chassis")
    names = ("Neck Yaw", "Neck Pitch", "Neck Roll")
    qa = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in names]
    rng = [m.actuator_ctrlrange[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)] for n in names]

    def gaze(q):
        for a, v in zip(qa, q):
            d.qpos[a] = v
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        mujoco.mj_camlight(m, d)
        ax = -d.cam_xmat[cam].reshape(3, 3)[:, 2]           # optical axis, world
        ax = d.xmat[base].reshape(3, 3).T @ ax               # -> chassis frame
        return ax

    # Neutral optical axis defines (0, 0): the robot's "straight ahead".
    f0 = gaze((0, 0, 0))
    f0h = f0.copy(); f0h[2] = 0; f0h /= np.linalg.norm(f0h)
    up = np.array([0, 0, 1.0]); left = np.cross(up, f0h)

    def yaw_el(ax):
        return (np.degrees(np.arctan2(ax @ left, ax @ f0h)),
                np.degrees(np.arcsin(np.clip(ax @ up, -1, 1))))

    el0 = yaw_el(f0)[1]
    yaws = np.arange(-180, 181, 5)
    pitches = np.arange(-80, 61, 5)
    err = np.zeros((len(pitches), len(yaws)))
    ach_y = np.zeros_like(err); ach_p = np.zeros_like(err)
    sat = np.zeros_like(err, dtype=bool)
    I = np.eye(4)
    for i, p in enumerate(pitches):
        for j, y in enumerate(yaws):
            raw = neck_angles(arkit_head(y, p), I)
            q = [float(np.clip(a, r[0], r[1])) for a, r in zip(raw, rng)]
            sat[i, j] = any(abs(a - c) > 1e-6 for a, c in zip(raw, q))
            ay, ap = yaw_el(gaze(q))
            ap -= el0
            ach_y[i, j], ach_p[i, j] = ay, ap
            # Great-circle error between wanted and achieved gaze directions.
            w = np.array([np.cos(np.radians(p)) * np.cos(np.radians(y)),
                          np.cos(np.radians(p)) * np.sin(np.radians(y)), np.sin(np.radians(p))])
            g = np.array([np.cos(np.radians(ap)) * np.cos(np.radians(ay)),
                          np.cos(np.radians(ap)) * np.sin(np.radians(ay)), np.sin(np.radians(ap))])
            err[i, j] = np.degrees(np.arccos(np.clip(w @ g, -1, 1)))

    print(f"\nHEAD-CAM GAZE COVERAGE  (neck only, torso/chassis at home keyframe)")
    print(f"neck ctrlrange [deg]: yaw {np.degrees(rng[0])}, pitch {np.degrees(rng[1])}, roll {np.degrees(rng[2])}")
    print(f"neutral optical-axis elevation: {el0:+.2f} deg (0 = level)")
    print(f"\n{'look':>22} | {'achieved yaw':>12} {'achieved el':>11} | {'error':>6}")
    for (y, p) in [(0, -10), (0, -20), (0, -30), (0, -45), (0, -60), (0, -75),
                   (0, 20), (0, 45), (45, 0), (90, 0), (120, 0), (180, 0), (-90, -30)]:
        i, j = list(pitches).index(p), list(yaws).index(y)
        print(f"yaw {y:+4d}  pitch {p:+4d} deg | {ach_y[i, j]:+12.1f} {ach_p[i, j]:+11.1f} | {err[i, j]:6.1f}")
    ok = err <= 2.0
    print(f"\ndirections tracked within 2 deg: {100 * ok.mean():.1f}% of the "
          f"{len(yaws)}x{len(pitches)} grid (yaw ±180, pitch -80..+60, 5 deg)")
    print(f"sphere fraction (area-weighted) within 2 deg: "
          f"{100 * (ok * np.cos(np.radians(pitches))[:, None]).sum() / np.cos(np.radians(pitches)).sum() / len(yaws):.1f}%")

    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    im = ax.pcolormesh(yaws, pitches, err, shading="nearest", cmap="viridis", vmin=0, vmax=90)
    ax.contour(yaws, pitches, err, levels=[2.0], colors="w", linewidths=1)
    ax.set_xlabel("operator head yaw [deg] (+ = left)")
    ax.set_ylabel("operator head pitch [deg] (+ = up)")
    ax.set_title("Head-cam gaze error, neck only (white: 2° contour)", fontsize=9)
    fig.colorbar(im, ax=ax, label="gaze error [deg]")
    fig.tight_layout()
    fig.savefig(FIG)
    print(f"wrote {FIG}")
    closed_loop()


# ── PART B: the neck-yaw -> base-yaw handoff, closed loop in physics ────────────
SCENE = os.path.join(_REPO, "simulation", "mabel_mujoco", "scenes", "mabel_bare.xml")
FIG_B = os.path.join(_CTRL, "docs", "figures", "head_gaze_handoff.pdf")


def closed_loop(holds=(60, 100, 135, 180), T=10.0):
    """Operator snaps the head to `psi` and HOLDS it. Each control tick: neck_angles
    with the MEASURED chassis yaw (as engine.py), neck ctrl clipped to range, the
    handoff rate through the tip-safe motion model + swerve IK, mj_step. Reports the
    camera's world heading error vs the operator's heading over time."""
    from mabel.teleop_engine.neck_handoff import NeckYawHandoff
    from mabel.teleop_engine.motion_model import BaseMotionModel
    from mabel.teleop_engine.deps import swerve_ik
    from mabel.teleop_engine import hyperparams as hp

    m = mujoco.MjModel.from_xml_path(SCENE)
    cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "zed_left")
    base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "mabel_chassis")
    acts = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
            for n in ("Neck Yaw", "Neck Pitch", "Neck Roll")]
    lift = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "Lift")
    dt = m.opt.timestep
    ctrl_every = max(1, int(round(0.01 / dt)))          # 100 Hz control, as the bridge

    def yaw_of(R):
        return float(np.arctan2(R[1, 0], R[0, 0]))

    print(f"\nPART B — neck->base yaw handoff, closed loop on {os.path.basename(SCENE)} "
          f"(handoff {np.degrees(hp.NECK_YAW_HANDOFF):.0f} deg, gain {hp.NECK_HANDOFF_GAIN}/s, "
          f"cap {hp.NECK_HANDOFF_WMAX} rad/s)")
    print(f"{'hold':>6} | {'t90 [s]':>7} | {'final err':>9} | {'neck yaw':>8} | {'chassis':>7} | stalled")
    traces = {}
    for psi in holds:
        d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, 0)
        mujoco.mj_forward(m, d)
        y0 = yaw_of(d.xmat[base].reshape(3, 3))
        cam0 = -d.cam_xmat[cam].reshape(3, 3)[:, 2]
        cam_y0 = float(np.arctan2(cam0[1], cam0[0]))
        h = NeckYawHandoff(hp.NECK_YAW_HANDOFF, hp.NECK_HANDOFF_GAIN,
                           hp.NECK_HANDOFF_WMAX, hp.NECK_HANDOFF_STALL)
        mm = BaseMotionModel()
        vel = (0.0, 0.0, 0.0)
        head = arkit_head(psi, 0)
        ts, errs = [], []
        n = int(T / dt)
        for k in range(n):
            if k % ctrl_every == 0:
                ch = yaw_of(d.xmat[base].reshape(3, 3)) - y0
                na = neck_angles(head, np.eye(4), 0.0, ch)
                for a, v in zip(acts, na):
                    lo, hi = m.actuator_ctrlrange[a]
                    d.ctrl[a] = float(np.clip(v, lo, hi))
                h.observe(float(na[0]))
                wz = h.rate(ch, dt * ctrl_every)
                vel = mm.step(vel, (0.0, 0.0, wz), dt * ctrl_every, float(d.ctrl[lift]))
                swerve_ik(*vel, d)
                c = -d.cam_xmat[cam].reshape(3, 3)[:, 2]
                cy = float(np.arctan2(c[1], c[0])) - cam_y0
                e = np.degrees(np.arctan2(np.sin(np.radians(psi) - cy), np.cos(np.radians(psi) - cy)))
                ts.append(k * dt); errs.append(abs(e))
            mujoco.mj_step(m, d)
        errs = np.array(errs); ts = np.array(ts)
        e0 = errs[0]
        t90 = next((t for t, e in zip(ts, errs) if e <= 0.1 * max(e0, 1e-6) + 2.0), float("nan"))
        ch = np.degrees(yaw_of(d.xmat[base].reshape(3, 3)) - y0)
        nk = np.degrees(d.qpos[m.jnt_qposadr[m.actuator_trnid[acts[0], 0]]])
        print(f"{psi:+6d} | {t90:7.2f} | {errs[-1]:8.1f}° | {nk:+7.1f}° | {ch:+6.1f}° | {h.stalled}")
        traces[psi] = (ts, errs)

    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    for psi, (ts, errs) in traces.items():
        ax.plot(ts, errs, label=f"hold {psi}°")
    ax.axhline(2.0, color="k", lw=0.6, ls=":")
    ax.set_xlabel("time since head turn [s]")
    ax.set_ylabel("head-cam heading error [deg]")
    ax.set_title("Neck→base yaw handoff, MuJoCo closed loop", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_B)
    print(f"wrote {FIG_B}")


if __name__ == "__main__":
    main()
