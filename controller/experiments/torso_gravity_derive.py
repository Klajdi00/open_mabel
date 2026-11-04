#!/usr/bin/env python3
"""torso_gravity_derive.py — re-derive mabel.torso_control's gravity constants from the MJCF.

The torso gravity feed-forward (``mabel/torso_control/gravity.py``) bakes  τ_hold(θ) =
A·sin(PHI − θ)  so the setup studio can import it without MuJoCo. Those constants go stale
whenever the canonical MJCF changes anything above the torso joint — masses, the arms, the
hands, link origins. Run THIS after such a change (the tip_over.py E4 rule) and paste the
printed GRAV_A / GRAV_PHI back into the module if they moved.

The sweep sets every other joint to the nominal pose (qpos0 — arms hanging as assembled,
hands on if the MJCF carries them), so the whole subtree above the torso rotates rigidly
and the fit is exact (residual ~1e-5 N·m). τ_hold = qfrc_bias at the torso DoF with
qvel = 0: the torque the actuator must SUPPLY to hold static.

    ~/miniconda3/envs/mabel/bin/python3 controller/experiments/torso_gravity_derive.py
"""
import math
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[2]
MJCF = REPO / "simulation" / "mabel_mujoco" / "models" / "mabel_full.xml"


def main():
    m = mujoco.MjModel.from_xml_path(str(MJCF))
    d = mujoco.MjData(m)
    jid = m.joint("Torso").id
    dof, qadr = m.jnt_dofadr[jid], m.jnt_qposadr[jid]
    lo, hi = m.jnt_range[jid]

    def hold_tau(theta):
        d.qpos[:] = m.qpos0
        d.qpos[qadr] = theta
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        return float(d.qfrc_bias[dof])

    thetas = np.linspace(lo, hi, 41)
    taus = np.array([hold_tau(t) for t in thetas])
    # τ = A·sin(PHI − θ) = A·sin(PHI)cos(θ) − A·cos(PHI)sin(θ)  — linear least squares
    X = np.column_stack([np.cos(thetas), -np.sin(thetas)])
    p, q = np.linalg.lstsq(X, taus, rcond=None)[0]         # p = A·sin(PHI), q = A·cos(PHI)
    A, phi = math.hypot(p, q), math.atan2(p, q)
    resid = np.max(np.abs(A * np.sin(phi - thetas) - taus))

    print(f"MJCF torso range: [{lo:+.4f}, {hi:+.4f}] rad "
          f"([{math.degrees(lo):+.1f}, {math.degrees(hi):+.1f}] deg)")
    print(f"subtree mass above the torso joint: {m.body_subtreemass[m.body('torso').id]:.3f} kg")
    print(f"\n    GRAV_A = {A:.3f}            # N·m")
    print(f"    GRAV_PHI = {phi:.4f}          # rad")
    print(f"\nfit residual: {resid:.2e} N·m (should be ~1e-5 — the subtree is rigid)")
    for t in (lo, -0.61, 0.0, hi):
        print(f"  θ {math.degrees(t):+6.1f}° → hold τ {hold_tau(t):+7.2f} N·m")
    print("\nPaste GRAV_A / GRAV_PHI into controller/mabel/torso_control/gravity.py if changed.")


if __name__ == "__main__":
    main()
