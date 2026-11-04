#!/usr/bin/env python3
"""
simulate_check.py
========================

Prove the actuator-identification pipeline *in simulation*, with no hardware --
exactly the philosophy of the inertial sysid demo: define a ground-truth
actuator, generate noisy measurements, run the identifier, and check it recovers
the truth to the noise floor.

What it does:
  1. Pick a ground-truth FrictionParams (Coulomb +/-, viscous, Stribeck, inertia).
  2. Synthesise a velocity staircase  -> friction curve data.
  3. Synthesise a frequency chirp      -> reflected-inertia data.
  4. Identify, then compare to truth and plot.

Outputs (results/simulation/):
  friction_curve.png, torque_fit.png, sim_summary.json
"""
import argparse
import json
import os

import _bootstrap                                   # noqa: F401
from mabel_actid import (FrictionParams, SyntheticActuator, velocity_staircase,
                         chirp_profile, identify, metrics)
from mabel_actid.simulate import SimSensors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motor", default="DM-J4340",
                    help="motor profile (for a realistic speed range)")
    ap.add_argument("--model", default="stribeck",
                    choices=["cv", "asym", "stribeck"])
    ap.add_argument("--tau-noise", type=float, default=0.02,
                    help="torque-estimate noise std [N*m]")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # ----- 1. ground truth (a plausible J4340-class joint) -----------------
    truth = FrictionParams(
        Fc_pos=0.45, Fc_neg=0.52, Fv=0.08,
        Fs_pos=0.70, Fs_neg=0.78, v_s=0.15, delta=2.0, J=0.012)
    print("ground-truth actuator parameters:")
    print(metrics.param_table(truth))
    print()

    sensors = SimSensors(tau_noise=args.tau_noise, w_quant=0.01, seed=args.seed)

    # ----- 2. velocity staircase (friction) --------------------------------
    from mabel_actid.motors import get as get_motor
    prof = get_motor(args.motor)
    v_safe = 0.30 * prof.v_max                       # stay well inside limits
    stair = velocity_staircase(v_max=v_safe, v_min=0.03, n_levels=16,
                               dwell=1.8, settle=1.0, rate=500.0,
                               log_spaced=True)
    # position-ramp velocity hold (kp>0): accurate even at the low speeds where
    # the Stribeck dip lives, unlike a pure velocity loop.
    act = SyntheticActuator(truth, kp=40.0, kd=2.0, sensors=sensors)
    stair_log = act.run_velocity(stair, rate=500.0)

    # ----- 3. chirp (inertia) ----------------------------------------------
    t, q, w, a = chirp_profile(amp=0.6, f0=0.2, f1=3.0, duration=20.0,
                               rate=500.0)
    act_pos = SyntheticActuator(truth, kp=60.0, kd=2.0, sensors=sensors)
    chirp_log = act_pos.run_position(t, q, w, rate=500.0)

    # ----- 4. identify ------------------------------------------------------
    res = identify.identify_actuator(stair_log, dynamic=chirp_log,
                                     model=args.model)
    p = res.params
    print("identified vs truth:")
    print(metrics.param_table(p, truth=truth))
    print()
    print(f"friction-curve fit:  RMS {res.curve.rms:.4f} N*m   "
          f"(inliers {res.curve.rms_inlier:.4f})   R^2 {res.curve.r2:.5f}   "
          f"({res.report['n_levels']} levels)")
    if res.inertia is not None:
        print(f"inertia fit:         J {p.J:.5f} kg*m^2   "
              f"(true {truth.J:.5f})   RMS {res.inertia.rms:.4f} N*m")
    print()
    print("MuJoCo joint attributes to paste into mabel_full.xml:")
    for k, v in p.mujoco_attrs().items():
        print(f"    {k} = \"{v}\"")

    # ----- plots + summary --------------------------------------------------
    simdir = _bootstrap.result_dir("simulation")
    w_lvl, tau_lvl = stair_log.steady_points()
    metrics.plot_friction_curve(
        w_lvl, tau_lvl, p, os.path.join(simdir, "friction_curve.png"),
        truth=truth, title=f"{args.motor} friction curve (sim test)")
    metrics.plot_torque_fit(
        chirp_log, p, os.path.join(simdir, "torque_fit.png"),
        title=f"{args.motor} chirp torque fit (sim test)")

    summary = {"motor": args.motor, "truth": truth.as_dict(), **res.report}
    out = os.path.join(simdir, "sim_summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote plots + {os.path.relpath(out, _bootstrap.CALIB_ROOT)}")


if __name__ == "__main__":
    main()
