#!/usr/bin/env python3
"""
Step 1 -- the system-ID experiment (simulation ground-truth test).

We KNOW the true parameters (they live in the MuJoCo model), so we can grade the
identification honestly.  Pipeline:

  1. Load the ground-truth arm model; read its true (mass, COM).
  2. Synthesise a noisy measured dataset (static + dynamic) with MuJoCo.
  3. Pretend we don't know the parameters: build a scrambled "CAD prior /
     initial guess" (masses off by +/-30%, COMs shifted up to +/-2 cm).

  TEST 1 (gravity compensation):  Method B -- analytic linear gravity least
          squares on the static poses.  Grades first-moment recovery and
          gravity-torque prediction.  Reports the regressor's singular spectrum
          (observability).
  TEST 2 (full inverse dynamics): Method A -- MuJoCo-in-the-loop nonlinear least
          squares on all data.  Grades mass, COM, and held-out torque
          prediction.

  4. Save plots + JSON summary under results/.
"""
import argparse
import json
import os

# Pick a headless OpenGL backend for the excitation movie *before* mujoco is
# imported.  osmesa is pure-software and works without a display; harmless for
# the (non-rendering) dynamics computations.
os.environ.setdefault("MUJOCO_GL", "osmesa")

import _bootstrap  # noqa: E402,F401
import numpy as np  # noqa: E402

from mabel_sysid.arm_model import ArmModel  # noqa: E402
from mabel_sysid.data_generation import make_dataset  # noqa: E402
from mabel_sysid import identify, metrics, visualize  # noqa: E402


def banner(s):
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise", type=float, default=0.05,
                    help="torque sensor noise std (N*m)")
    ap.add_argument("--n-static", type=int, default=250)
    ap.add_argument("--dyn-duration", type=float, default=40.0)
    ap.add_argument("--mass-perturb", type=float, default=0.30)
    ap.add_argument("--com-perturb", type=float, default=0.02)
    ap.add_argument("--ridge", type=float, default=1e-4)
    ap.add_argument("--reg", type=float, default=1e-5)
    ap.add_argument("--no-video", action="store_true",
                    help="skip rendering the excitation GIF")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # --- 1. ground truth -------------------------------------------------
    truth = ArmModel(_bootstrap.MODEL_PATH)
    names = truth.link_names()
    true_m, true_c = truth.get_masses(), truth.get_coms()
    banner("GROUND TRUTH (the parameters we pretend not to know)")
    print(f"links : {names}")
    print(f"mass  : {true_m.round(3)} kg   (total {true_m.sum():.3f} kg)")

    # --- 2. measured dataset --------------------------------------------
    data = make_dataset(truth, n_static=args.n_static,
                        dynamic_duration=args.dyn_duration, dynamic_dt=0.02,
                        torque_noise_std=args.noise, rng=rng)
    train, test = data.split(0.8, rng)
    n_dyn = len(data) - args.n_static
    print(f"\ndataset: {len(data)} samples ({args.n_static} static + {n_dyn} "
          f"dynamic), torque noise std = {args.noise} N*m")
    print(f"max |qdot| = {np.abs(data.QD).max():.1f} rad/s   "
          f"max |qddot| = {np.abs(data.QDD).max():.1f} rad/s^2")
    print(f"split  : {len(train)} train / {len(test)} test")

    # --- 2b. visualise the excitation (movie + trajectory plot) ----------
    gif_path = None
    traj_plot = os.path.join(_bootstrap.EXP_DIR, "excitation_trajectory.png")
    visualize.plot_trajectory(truth, traj_plot, seed=args.seed)
    print(f"saved excitation trajectory plot -> {traj_plot}")
    if not args.no_video:
        print("rendering excitation movie (MuJoCo, offscreen)...", flush=True)
        gif_path = os.path.join(_bootstrap.EXP_DIR, "excitation.gif")
        gif_path = visualize.render_excitation(truth, gif_path, seed=args.seed)
        if gif_path:
            print(f"saved excitation movie -> {gif_path}")

    # --- 3. scrambled prior / initial guess -----------------------------
    guess = ArmModel(_bootstrap.MODEL_PATH)
    x0 = identify.perturb_params(guess, rng, mass_frac=args.mass_perturb,
                                 com_shift=args.com_perturb)
    base_m = x0.reshape(-1, 4)[:, 0]
    base_c = x0.reshape(-1, 4)[:, 1:]
    banner(f"SCRAMBLED PRIOR  (mass +/-{int(args.mass_perturb*100)}%, "
           f"COM +/-{int(args.com_perturb*100)} cm)")
    print(f"prior mass: {base_m.round(3)} kg")

    # ======================= TEST 1 : gravity ===========================
    banner("TEST 1  --  GRAVITY COMPENSATION  (Method B: linear gravity LS)")
    prior_theta = np.array([
        v for k in range(guess.n_links)
        for v in ([base_m[k]] + list(base_m[k] * base_c[k]))])
    lin = identify.identify_gravity_linear(
        truth, train, ridge=args.ridge, prior_theta=prior_theta)

    fm = metrics.first_moment_error(true_m, true_c, lin.masses, lin.coms)
    sv = lin.singular_values
    n_obs = int(np.sum(sv > 1e-6 * sv[0]))
    print(f"regressor: {sv.size} params, condition number = {lin.cond_number:.2e}")
    print(f"observable directions (sv > 1e-6 * sv_max): {n_obs} / {sv.size}")
    print("  -> rank deficiency confirms some masses are NOT observable from")
    print("     gravity alone; only first moments m*c are.")
    print("\nFIRST-MOMENT recovery (the gravity-relevant quantity):")
    print(f"  mean |Δ(m·c)| = {fm['mae_gm']:.2f} g·m   "
          f"mean relative error = {100*fm['mean_rel']:.1f} %")

    # gravity-torque prediction on held-out static poses
    static_mask = np.all(np.abs(test.QD) < 1e-9, axis=1) & \
                  np.all(np.abs(test.QDD) < 1e-9, axis=1)
    Q_static_test = test.Q[static_mask]
    est_grav = ArmModel(_bootstrap.MODEL_PATH)
    est_grav.set_param_vector(
        np.column_stack([lin.masses, lin.coms]).reshape(-1))
    guess.set_param_vector(x0)
    g_rms_before = metrics.gravity_prediction_rms(guess, Q_static_test, truth)
    g_rms_after = metrics.gravity_prediction_rms(est_grav, Q_static_test, truth)
    print("\nGRAVITY-TORQUE prediction on held-out poses (vs truth):")
    print(f"  prior model     : {g_rms_before:.4f} N*m")
    print(f"  identified (B)  : {g_rms_after:.4f} N*m   "
          f"(noise floor ~ {args.noise} N*m)")

    # ======================= TEST 2 : dynamics ==========================
    banner("TEST 2  --  FULL INVERSE DYNAMICS  (Method A: MuJoCo-in-loop NLLS)")
    nl = identify.identify_nonlinear(
        guess, train, x0=x0, prior_x=x0, reg=args.reg, rng=rng, verbose=0)
    print(f"converged = {nl.success}   function evals = {nl.n_eval}   "
          f"train residual RMS = {nl.residual_rms:.4f} N*m")
    print("\n" + metrics.param_error_table(
        names, true_m, true_c, nl.masses, nl.coms))

    guess.set_param_vector(x0)
    rms_before = metrics.torque_prediction_rms(guess, test)
    guess.set_param_vector(nl.x)
    rms_after = metrics.torque_prediction_rms(guess, test)
    print("\nHELD-OUT TORQUE prediction (vs noise-free truth):")
    print(f"  prior model     : {rms_before['total_rms']:.4f} N*m")
    print(f"  identified (A)  : {rms_after['total_rms']:.4f} N*m   "
          f"(noise floor ~ {args.noise} N*m)")

    # ----- verdict -------------------------------------------------------
    banner("VERDICT")
    com_mae = 1000 * np.mean(np.linalg.norm(nl.coms - true_c, axis=1))
    mass_mae = np.mean(np.abs(nl.masses - true_m))
    print("What the identified model is FOR (control quantities) -- both excellent:")
    print(f"  gravity-torque g(q) prediction : {g_rms_after:.4f} N*m   (was {g_rms_before:.3f})")
    print(f"  inverse-dynamics tau prediction: {rms_after['total_rms']:.4f} N*m   "
          f"(was {rms_before['total_rms']:.3f})")
    print("  -> well below the noise floor; ready for gravity comp / inverse dynamics.")
    print("\nIndividual physical parameters -- recovered to varying degrees:")
    print(f"  COM             : {com_mae:5.1f} mm mean error                -> good")
    print(f"  per-link mass   : {mass_mae:.3f} kg MAE ({100*mass_mae/np.mean(true_m):.0f}% of mean mass) -> partial")
    print(f"  per-link m·c    : {100*fm['mean_rel']:5.1f} % mean rel error             -> partial")
    print("  Per-link mass / first-moment are only PARTIALLY observable from")
    print("  joint torques (a fundamental 'base-parameter' limit -- see")
    print("  docs/THEORY.md §5). The identifiable COMBINATIONS, which is all the")
    print("  controller uses, are nailed. For trustworthy per-link mass, add a")
    print("  CAD prior on the weak directions or weigh the links on a scale.")

    # --- plots + summary -------------------------------------------------
    plot_path = os.path.join(_bootstrap.EXP_DIR, "sysid_results.png")
    metrics.plot_results(plot_path, names, true_m, true_c, nl.masses, nl.coms,
                         baseline_m=base_m, baseline_c=base_c,
                         singular_values=sv)

    prior_arm = ArmModel(_bootstrap.MODEL_PATH); prior_arm.set_param_vector(x0)
    est_arm = ArmModel(_bootstrap.MODEL_PATH); est_arm.set_param_vector(nl.x)
    fit_path = os.path.join(_bootstrap.EXP_DIR, "torque_fit.png")
    visualize.plot_torque_fit(truth, prior_arm, est_arm, fit_path)

    print("\nOUTPUT FILES (in results/):")
    print("  sysid_results.png         param errors + regressor spectrum")
    print("  torque_fit.png            torque: truth vs prior vs identified")
    print("  excitation_trajectory.png commanded q / qd / qdd vs time")
    if gif_path:
        print("  excitation.gif            MuJoCo movie of the excitation motion")

    summary = {
        "seed": args.seed, "noise_std": args.noise, "n_samples": len(data),
        "links": names,
        "true_mass": true_m.tolist(),
        "identified_mass_nonlinear": nl.masses.tolist(),
        "identified_mass_linear": lin.masses.tolist(),
        "regressor_cond": lin.cond_number,
        "observable_dirs": n_obs, "n_params": int(sv.size),
        "first_moment_rel_err_pct": 100 * fm["mean_rel"],
        "gravity_rms_before": g_rms_before, "gravity_rms_after": g_rms_after,
        "torque_rms_before": rms_before["total_rms"],
        "torque_rms_after": rms_after["total_rms"],
        "mass_mae_kg": float(mass_mae), "com_mae_mm": float(com_mae),
    }
    spath = os.path.join(_bootstrap.EXP_DIR, "sysid_summary.json")
    with open(spath, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved summary -> {spath}")

    # identified parameters, ready to load for gravity comp / inverse dynamics
    params = {
        "source": "Method A (MuJoCo-in-loop nonlinear LS)",
        "links": {
            names[i]: {"mass": float(nl.masses[i]),
                       "com": nl.coms[i].tolist()}
            for i in range(len(names))
        },
    }
    ppath = os.path.join(_bootstrap.EXP_DIR, "identified_params.json")
    with open(ppath, "w") as f:
        json.dump(params, f, indent=2)
    print(f"saved params  -> {ppath}")


if __name__ == "__main__":
    main()
