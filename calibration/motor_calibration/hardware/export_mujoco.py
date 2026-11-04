#!/usr/bin/env python3
r"""
export_mujoco.py
=======================

Turn identified actuator parameters into something you can use:

  1. Print the MuJoCo ``<joint .../>`` attributes (frictionloss / damping /
     armature) to paste into ``mabel_full.xml`` for the matching joint.
  2. Show the feed-forward **friction-compensation** torque the controller should
     add on top of the rigid-body gravity / inverse-dynamics torque, so the joint
     actually moves as commanded.

Usage:
  python export_mujoco.py --params ../results/DM-J4340P/params.json
  python export_mujoco.py --params ... --joint r_elbow   # name the joint

The feed-forward to add to your existing inverse-dynamics torque is simply
``tau_f(q_dot)`` from the identified model:

    tau_cmd = tau_rigid_body(q, q_dot, q_ddot)   +   tau_f(q_dot)
              \_________ from mabel_sysid _______/   \__ from this package __/
"""
import argparse
import json

import numpy as np

import _bootstrap                                   # noqa: F401
from mabel_actid.friction_model import FrictionParams, friction_torque


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True, help="act_*_params.json")
    ap.add_argument("--joint", default="JOINT_NAME",
                    help="MuJoCo joint name for the snippet")
    args = ap.parse_args()

    with open(args.params) as f:
        d = json.load(f)
    p = FrictionParams.from_dict(d["params"])
    motor = d.get("motor", "actuator")

    print(f"# identified actuator: {motor}")
    print(f"# friction RMS {d.get('friction_rms_Nm', float('nan')):.4f} N*m, "
          f"R^2 {d.get('friction_r2', float('nan')):.4f}")
    print()
    print("MuJoCo joint snippet (paste attributes onto the joint in "
          "mabel_full.xml):")
    a = p.mujoco_attrs()
    print(f'    <joint name="{args.joint}" '
          f'frictionloss="{a["frictionloss"]}" '
          f'damping="{a["damping"]}" '
          f'armature="{a["armature"]}" .../>')
    print()
    print("note: frictionloss is the direction-averaged Coulomb friction. The "
          "real joint is asymmetric")
    print(f"      (Fc+ {p.Fc_pos:.3f} vs Fc- {p.Fc_neg:.3f} N*m); MuJoCo's "
          "frictionloss is symmetric, so")
    print("      keep the asymmetric model for feed-forward compensation below.")
    print()

    # ---- feed-forward compensation table ----------------------------------
    print("feed-forward friction-compensation torque tau_f(w) to ADD to your "
          "inverse-dynamics torque:")
    print(f"  {'w [rad/s]':>10}{'tau_f [N*m]':>14}")
    for w in [-2.0, -1.0, -0.3, -0.05, 0.05, 0.3, 1.0, 2.0]:
        tf = float(friction_torque(p, np.array([w]))[0])
        print(f"  {w:>10.2f}{tf:>14.4f}")
    print()
    print("In the controller:")
    print("    tau_ff = arm.gravity_torque(q)            # from mabel_sysid")
    print("    tau_ff += friction_torque(params, qd)     # from this package")


if __name__ == "__main__":
    main()
