#!/usr/bin/env python3
"""
Step 2 -- use the identified parameters for gravity compensation.

This shows the *payoff* of the identification: given the identified (mass, COM)
loaded into a MuJoCo arm model, computing the feed-forward gravity-compensation
torque for any joint configuration is a one-liner:

    tau_ff = arm.gravity_torque(q)        # = g_hat(q)

On the real robot you would add this to your joint controller so the arm holds
its weight without drooping.  Here we verify, on a fresh set of random poses,
that the identified model's gravity torque matches the true robot's.

Run ``scripts/01_run_sysid.py`` first to produce results/identified_params.json.
"""
import json
import os

import _bootstrap  # noqa: F401
import numpy as np

from mabel_sysid.arm_model import ArmModel


def load_identified(arm: ArmModel, path: str):
    with open(path) as f:
        params = json.load(f)
    for b in arm.moving_body_ids:
        nm = arm.body_name(b)
        if nm in params["links"]:
            arm.model.body_mass[b] = params["links"][nm]["mass"]
            arm.model.body_ipos[b] = params["links"][nm]["com"]


def main():
    ppath = os.path.join(_bootstrap.EXP_DIR, "identified_params.json")
    if not os.path.exists(ppath):
        raise SystemExit("run scripts/01_run_sysid.py first (no identified_params.json)")

    truth = ArmModel(_bootstrap.MODEL_PATH)          # the "real robot"
    est = ArmModel(_bootstrap.MODEL_PATH)            # identified model
    load_identified(est, ppath)

    rng = np.random.default_rng(123)
    Q = truth.sample_configs(500, rng)

    err = np.array([est.gravity_torque(q) - truth.gravity_torque(q) for q in Q])
    per_joint = np.sqrt(np.mean(err ** 2, axis=0))

    print("Gravity-compensation feed-forward check on 500 unseen poses")
    print("-" * 60)
    print(f"per-joint gravity-torque RMS error (N*m):\n  "
          f"{np.array2string(per_joint, precision=4)}")
    print(f"overall RMS error: {np.sqrt(np.mean(err ** 2)):.4f} N*m")
    print(f"worst-case error : {np.abs(err).max():.4f} N*m")

    # show an example usage
    q_demo = Q[0]
    print("\nExample -- feed-forward gravity torque at a random pose:")
    print(f"  q      = {np.array2string(q_demo, precision=3)}")
    print(f"  tau_ff = {np.array2string(est.gravity_torque(q_demo), precision=3)} N*m")
    print("\nAdd tau_ff to your joint torque command for gravity compensation.")


if __name__ == "__main__":
    main()
