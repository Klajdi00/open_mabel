#!/usr/bin/env python3
"""
Step 0 -- sanity check the standalone arm model.

Prints the kinematic / inertial summary and verifies that the analytic
Lagrangian gravity regressor reproduces MuJoCo's own bias force.  If this
passes, the dynamics model and the regressor agree to machine precision.
"""
import _bootstrap  # noqa: F401  (sets sys.path / MODEL_PATH)
import numpy as np

from mabel_sysid.arm_model import ArmModel
from mabel_sysid import regressor as R


def main():
    arm = ArmModel(_bootstrap.MODEL_PATH)
    print(f"model: {_bootstrap.MODEL_PATH}")
    print(f"DOF (nv) = {arm.nv}   moving links = {arm.n_links}\n")

    print(f"{'link':<14}{'mass(kg)':>10}{'COM (m) in body frame':>32}"
          f"{'joint range (rad)':>26}")
    for i, b in enumerate(arm.moving_body_ids):
        nm = arm.body_name(b)
        m = arm.model.body_mass[b]
        c = arm.model.body_ipos[b]
        lo, hi = arm.q_lower[i], arm.q_upper[i]
        print(f"{nm:<14}{m:>10.4f}   ({c[0]:+.4f}, {c[1]:+.4f}, {c[2]:+.4f})"
              f"      [{lo:+.3f}, {hi:+.3f}]")
    print(f"\ntotal arm mass = {arm.get_masses().sum():.3f} kg")

    # regressor vs MuJoCo
    theta = R.true_param_vector(arm)
    rng = np.random.default_rng(0)
    err = 0.0
    for q in arm.sample_configs(100, rng):
        Y = R.gravity_regressor_single(arm, q)
        err = max(err, np.abs(Y @ theta - arm.gravity_torque(q)).max())
    print(f"\nmax |Y_g(q)·theta - g_MuJoCo(q)| over 100 poses = {err:.2e}")
    print("OK -- regressor matches MuJoCo." if err < 1e-9 else "MISMATCH!")


if __name__ == "__main__":
    main()
