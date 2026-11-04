#!/usr/bin/env python3
"""
wbc_wrist_limit.py — isolate the ORCA-wrist runaway in the WBC torque engine.

THE BUG (2026-07-19, operator report): "the wrist position when I switch from
drive to task space is not correct — the wrist rotated 180 in the orientation."

Measured on the live sim stack (server/tools/exp_sim_stack.py S8): with a wrist
target standing in the client's accumulated joint dict, a drive -> task/joint
switch drives the ORCA wrist HARD into its MJCF limit, punches THROUGH it, and
parks the joint at ~+3.85 rad (221 deg) — a visibly flipped hand that never
recovers.

WHAT THIS HARNESS SHOWS (a NEGATIVE result, and that is the point): driving the
raw WBCEngine in-process — same model, same law, same stiffness sweep, wrists
parked exactly ON their limit — the wrists HOLD at every K from 0.5 to 30. The
torque law is not unstable by itself.

So the runaway is a property of the SPLIT deployment, not of the control law:
under `sim:bridge` (what server/run.sh runs) physics lives in the
simulation_bridge process while the gateway computes the effort from a MIRRORED
qpos/qvel and ships it over RPC at SimBridgePlant.CMD_HZ = 120 Hz. Measured on
the live stack, the left wrist reaches -118.6 rad/s (~19 rev/s on a joint whose
whole range is 1.57 rad) and tunnels straight through its MJCF limit — the
signature of delayed feedback on a very low-inertia joint (ORCA wrist
M = 2.5e-3 kg m^2, ~100x lighter than an arm link), not of a static error.

Keep this harness as the control: if a future change makes the numbers below
diverge, the law itself has regressed. Reproduce the live failure with
server/tools/exp_sim_stack.py (section S8) against a running stack.

Run:  python3 controller/experiments/wbc_wrist_limit.py
"""
import os
import sys

import numpy as np
import mujoco

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "controller"))

from mabel.whole_body_controller.engine import WBCEngine, _wbc_key  # noqa: E402

MODEL = os.path.join(_ROOT, "simulation", "mabel_mujoco", "models", "mabel_full.xml")
WRISTS = ("Left Wrist", "Right Wrist")


def limits(m, name):
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
    return tuple(m.jnt_range[jid]), int(m.jnt_qposadr[jid]), int(m.jnt_dofadr[jid])


def run(stiffness, q_start, seconds=8.0, verbose=True):
    """Engage the WBC in impedance/arm with the wrists parked at `q_start`
    (fraction of the way to the limit) and report where they end up."""
    m = mujoco.MjModel.from_xml_path(MODEL)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    info = {w: limits(m, w) for w in WRISTS}
    # park each wrist at q_start of its range (1.0 == exactly on the lower limit)
    for w in WRISTS:
        (lo, hi), qadr, _ = info[w]
        d.qpos[qadr] = lo + (1.0 - q_start) * (hi - lo)
    mujoco.mj_forward(m, d)
    start = {w: float(d.qpos[info[w][1]]) for w in WRISTS}

    wbc = WBCEngine(m, d)
    wbc.set_stiffness(stiffness)
    # exactly what facade.update() does on the drive -> task switch
    q_des = {_wbc_key(w): start[w] for w in WRISTS}
    wbc.enter("impedance", "arm", q_des=q_des)

    managed = list(wbc.managed)
    if verbose:
        print(f"    wrists managed: {[w for w in WRISTS if w in managed]}")

    dt = m.opt.timestep
    n = int(seconds / dt)
    peak_tau = {w: 0.0 for w in WRISTS}
    breached_at = {w: None for w in WRISTS}
    for i in range(n):
        wbc.step(dt)
        for w in WRISTS:
            (lo, hi), qadr, dofadr = info[w]
            q = float(d.qpos[qadr])
            peak_tau[w] = max(peak_tau[w], abs(float(d.qfrc_applied[dofadr])))
            if breached_at[w] is None and not (lo - 1e-3 <= q <= hi + 1e-3):
                breached_at[w] = i * dt
        mujoco.mj_step(m, d)

    out = {}
    for w in WRISTS:
        (lo, hi), qadr, _ = info[w]
        q = float(d.qpos[qadr])
        out[w] = dict(start=start[w], end=q, lo=lo, hi=hi,
                      breach=breached_at[w], tau=peak_tau[w],
                      outside=not (lo - 1e-3 <= q <= hi + 1e-3))
    return out


def main():
    m = mujoco.MjModel.from_xml_path(MODEL)
    print("=" * 78)
    print("  WBC ORCA-wrist limit runaway — offline reproduction")
    print("=" * 78)
    for w in WRISTS:
        (lo, hi), qadr, dofadr = limits(m, w)
        M = 1.0 / max(float(m.dof_invweight0[dofadr]), 1e-12)
        print(f"  {w:12s} range=[{lo:+.4f},{hi:+.4f}] rad   qpos0={m.qpos0[qadr]:+.3f}   "
              f"eff. inertia M={M:.2e} kg m^2")
    print(f"  timestep dt = {m.opt.timestep * 1000:.2f} ms")

    print("\n  Wrist parked ON its limit, WBC engaged in impedance/arm:")
    print(f"  {'K':>6} {'joint':<13} {'start':>8} {'end':>10} {'limit':>18} "
          f"{'breach@':>9} {'peak tau':>9}  verdict")
    bad = 0
    for K in (0.5, 1.0, 3.0, 10.0, 30.0):
        r = run(K, q_start=1.0, verbose=False)
        for w in WRISTS:
            v = r[w]
            verdict = "ESCAPED LIMIT" if v["outside"] else "held"
            bad += int(v["outside"])
            b = f"{v['breach']:.2f}s" if v["breach"] is not None else "-"
            print(f"  {K:>6} {w:<13} {v['start']:>+8.3f} {v['end']:>+10.3f} "
                  f"[{v['lo']:+.3f},{v['hi']:+.3f}] {b:>9} {v['tau']:>9.2f}  {verdict}")
    print("\n" + "=" * 78)
    print(f"  {bad} wrist/stiffness combinations left the joint's MJCF limit")
    print("=" * 78)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
