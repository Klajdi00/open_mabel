#!/usr/bin/env python3
"""
identify.py
==================

Fit the actuator model to logs recorded by ``collect.py`` (or to the
simulator's logs).  Pure post-processing -- no hardware needed.

  python identify.py \
      --staircase ../results/DM-J4310/logs/staircase.csv \
      --dynamic   ../results/DM-J4310/logs/chirp.csv \
      --motor DM-J4310 --model stribeck

Outputs (results/<motor>/):
  plots/friction_curve.png, plots/torque_fit.png,
  params.json   <- the identified params + MuJoCo attributes
"""
import argparse
import json
import os

import _bootstrap                                   # noqa: F401
from mabel_actid import identify, metrics
from mabel_actid.data_logging import CalibrationLog


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--staircase", required=True,
                    help="constant-velocity CSV (friction)")
    ap.add_argument("--dynamic", default=None,
                    help="sine/chirp CSV (reflected inertia); optional")
    ap.add_argument("--motor", default="actuator")
    ap.add_argument("--model", default="stribeck",
                    choices=["cv", "asym", "stribeck"])
    ap.add_argument("--gravity-split", action="store_true",
                    help="remove a position-dependent load via +/- pairing")
    args = ap.parse_args()

    stair = CalibrationLog.load(args.staircase)
    dyn = CalibrationLog.load(args.dynamic) if args.dynamic else None

    res = identify.identify_actuator(
        stair, dynamic=dyn, model=args.model,
        use_gravity_split=args.gravity_split)
    p = res.params

    print(metrics.param_table(p))
    print()
    print(f"friction fit: RMS {res.curve.rms:.4f} N*m "
          f"(inliers {res.curve.rms_inlier:.4f})   R^2 {res.curve.r2:.5f}"
          f"   ({res.report['n_levels']} levels, model={args.model})")
    if res.inertia is not None:
        print(f"inertia fit:  J {p.J:.5f} kg*m^2   RMS {res.inertia.rms:.4f} N*m")
    if args.gravity_split:
        print(f"removed load: tau_g ~ {res.report['gravity_split_tau_g_Nm']:.4f} N*m")
    print("\nMuJoCo joint attributes:")
    for k, v in p.mujoco_attrs().items():
        print(f"    {k}=\"{v}\"")

    plots = _bootstrap.result_dir(args.motor, "plots")
    params = _bootstrap.result_dir(args.motor)
    w_lvl, tau_lvl = stair.steady_points()
    if args.gravity_split:
        w_lvl, tau_lvl, _ = identify.gravity_split(w_lvl, tau_lvl)
    metrics.plot_friction_curve(
        w_lvl, tau_lvl, p,
        os.path.join(plots, "friction_curve.png"),
        title=f"{args.motor} friction curve")
    if dyn is not None:
        metrics.plot_torque_fit(
            dyn, p,
            os.path.join(plots, "torque_fit.png"),
            title=f"{args.motor} dynamic torque fit")

    out = os.path.join(params, "params.json")
    with open(out, "w") as f:
        json.dump({"motor": args.motor, **res.report}, f, indent=2)
    print(f"\nwrote {os.path.relpath(out, _bootstrap.CALIB_ROOT)} + plots")


if __name__ == "__main__":
    main()
