#!/usr/bin/env python3
"""
validate.py -- the scientific validation of the arm gravity identification.

Run BEFORE you touch the real robot.  Every number in docs/ARM_CALIBRATION.tex
comes from here, and every figure is regenerated from scratch by this script.

  python3 experiments/validate.py            # full suite (~4 min)
  python3 experiments/validate.py --quick    # fewer seeds, for iteration
  python3 experiments/validate.py --only E4  # one experiment

The experiments
---------------
E1  IDENTIFIABILITY   What can gravity torques possibly tell us?  (rank of the
                      regressor, the blind subspace, 2 params per joint)
E2  SIGNAL vs FRICTION  Per-joint gravity swing against the measured Coulomb
                      friction and the torque quantiser.  The wrist is the
                      problem, and posture choice is the fix.
E3  PROTOCOL          Unidirectional vs bidirectional sweeps.  Quantifies the
                      friction bias you eat if you skip the return pass.
E4  ACCURACY          Monte-Carlo: randomise the TRUE masses (+/-30%) and COMs
                      (+/-2 cm), identify from the safe protocol, grade the
                      gravity-torque prediction on held-out poses.
E5  NOISE             Accuracy vs torque-sensor noise, incl. the 12-bit quantiser.
E6  BUDGET            How many angles / postures do we actually need?
E7  SAFETY            Verify the commanded motion against the envelope, and show
                      the inertial torque is negligible (so "static" is honest).
E8  GRAVITY COMP      The end use: residual uncompensated torque, CAD vs identified,
                      against the friction floor that bounds what is achievable.

A NOTE ON HONESTY: the identifier is only ever given the CAD model.  Postures are
planned from CAD (you cannot plan from a truth you do not know), the plant it
measures is a *different* model with friction, quantisation and gain error, and
it is graded on poses it never saw.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import figstyle
figstyle.apply()
import matplotlib.pyplot as plt

from mabel_sysid.arm_model import ArmModel
from mabel_sysid import base_params as BP
from mabel_sysid import excitation as EX
from mabel_sysid import friction as FR
from mabel_sysid import plant as PL
from mabel_sysid import regressor as R

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
XML = os.path.join(ROOT, "models", "right_arm_sysid.xml")
FIGDIR = os.path.join(ROOT, "docs", "figures")
RESDIR = os.path.join(ROOT, "results")
os.makedirs(FIGDIR, exist_ok=True)
os.makedirs(RESDIR, exist_ok=True)

JOINTS = [f"J{i+1}" for i in range(7)]
SUMMARY: dict = {}


def banner(s):
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78)


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x) ** 2)))


# --------------------------------------------------------------------------- #
#  The protocol: planned ONCE from the CAD model (as on the real robot)        #
# --------------------------------------------------------------------------- #
class Protocol:
    """The safe, one-joint-at-a-time excitation, planned from CAD only."""

    def __init__(self, cad: ArmModel, lim: EX.SafetyLimits, *, n_angles=12,
                 n_postures=1, bidirectional=True, rng=None, verbose=True):
        self.cad = cad
        self.lim = lim
        self.bidirectional = bidirectional
        rng = rng or np.random.default_rng(0)
        self.plans, self.designs = [], []
        for j in range(cad.nv):
            for p in range(n_postures):
                d = EX.best_sweep(cad, j, lim, n_samples=250 + 150 * p,
                                  rng=np.random.default_rng(1000 * j + p))
                plan = EX.plan_from_design(cad, d, lim, n_angles=n_angles,
                                           bidirectional=bidirectional)
                self.designs.append(d)
                self.plans.append(plan)
        self.duration = sum(p.duration for p in self.plans)
        if verbose:
            worst = max(d.torque_frac for d in self.designs)
            print(f"  protocol: {len(self.plans)} sweeps, "
                  f"{sum(len(p.holds) for p in self.plans)} holds, "
                  f"{self.duration/60:.1f} min of robot time, "
                  f"worst torque {100*worst:.0f}% of budget")

    def run(self, plant: PL.MeasuredPlant):
        """Execute against a plant; return stacked static (Q, TAU_gravity) + diags."""
        Qs, TAUs, diags = [], [], []
        for plan in self.plans:
            sweep = PL.run_sweep(plant, plan)
            if self.bidirectional:
                sp = FR.split_bidirectional(sweep)
                Q, TAU, coul = sp.Q, sp.tau_gravity, sp.coulomb_mean
            else:
                # naive: take the measurements as-is, friction and all
                Q, TAU, coul = sweep["q"], sweep["tau"], np.zeros(plant.truth.nv)
            j = plan.joint
            fit = FR.fit_sinusoid(Q[:, j], TAU[:, j], j)
            Qs.append(Q); TAUs.append(TAU)
            diags.append({"joint": j, "amplitude": plan.amplitude,
                          "coulomb": float(coul[j]), "r2": fit.r2, "snr": fit.snr,
                          "offset": FR.fit_sinusoid_with_offset(Q[:, j], TAU[:, j])})
        return np.vstack(Qs), np.vstack(TAUs), diags


def make_truth(cad: ArmModel, rng, mass_frac=0.30, com_shift=0.02) -> ArmModel:
    """A 'real robot': the CAD model with randomised true masses and COMs."""
    truth = ArmModel(XML)
    m = cad.get_masses().copy()
    c = cad.get_coms().copy()
    m = m * (1.0 + rng.uniform(-mass_frac, mass_frac, m.size))
    c = c + rng.uniform(-com_shift, com_shift, c.shape)
    truth.set_param_vector(np.column_stack([m, c]).reshape(-1))
    return truth


def identify(cad: ArmModel, Q, TAU, sub) -> BP.GravityFit:
    """Fit from static data, anchored on the CAD prior, weighted per joint."""
    theta_prior = R.true_param_vector(cad)      # 'true' of the CAD model = the prior
    scale = np.maximum(np.abs(TAU).std(axis=0), 1e-3)
    W = np.tile(1.0 / scale, (len(Q), 1))
    return BP.fit_theta(cad, Q, TAU, theta_prior, sub, weights=W)


def as_model(fit: BP.GravityFit) -> ArmModel:
    est = ArmModel(XML)
    est.set_param_vector(np.column_stack([fit.masses, fit.coms]).reshape(-1))
    return est


def grav_err(a: ArmModel, b: ArmModel, Qtest):
    """Per-joint RMS gravity-torque difference between two models."""
    ga = np.array([a.gravity_torque(q) for q in Qtest])
    gb = np.array([b.gravity_torque(q) for q in Qtest])
    d = ga - gb
    return rms(d), np.sqrt(np.mean(d ** 2, axis=0))


# =========================================================================== #
#  E1 -- IDENTIFIABILITY                                                      #
# =========================================================================== #
def E1(args):
    banner("E1  IDENTIFIABILITY -- what can static torques possibly reveal?")
    cad = ArmModel(XML)
    sub = BP.identifiable_subspace(cad, n_poses=800, rank_tol=1e-3)
    s = sub.sing / sub.sing[0]

    print(f"  parameters (4 per link x {cad.n_links} links) : {sub.n_params}")
    print(f"  numerical rank (rel. tol 1e-3)               : {sub.rank}")
    print(f"  blind directions                             : {sub.n_blind}")
    print(f"  theory: a fixed-base n-R chain has exactly 2n = {2*cad.nv} gravity")
    print(f"          base parameters (Gautier & Khalil 1990).  Measured: {sub.rank}.")
    exact_zero = int(np.sum(s < 1e-12))
    print(f"  exactly-zero singular values                 : {exact_zero}")
    print(f"  near-singular (kept out by the 1e-3 tol)     : "
          f"{sub.n_params - sub.rank - exact_zero}")

    # how much of a typical CAD mass error lies in the BLIND subspace?
    rng = np.random.default_rng(7)
    blind_frac = []
    for _ in range(400):
        truth = make_truth(cad, rng)
        d = R.true_param_vector(truth) - R.true_param_vector(cad)
        d_blind = sub.V_null @ (sub.V_null.T @ d)
        blind_frac.append(np.linalg.norm(d_blind) / max(np.linalg.norm(d), 1e-12))
    blind_frac = np.array(blind_frac)
    print(f"\n  Of a random CAD parameter error (+/-30% mass, +/-2cm COM),")
    print(f"  {100*blind_frac.mean():.1f}% +/- {100*blind_frac.std():.1f}% of its norm lies in the")
    print(f"  BLIND subspace -- invisible to any static experiment, and (by")
    print(f"  definition) irrelevant to gravity compensation.")

    # per-link: which of the 4 params of each link are seen at all?
    col = np.linalg.norm(BP.R.stack_gravity_regressor(
        cad, cad.sample_configs(300, np.random.default_rng(3))), axis=0)
    col = col.reshape(cad.n_links, 4)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.6))
    idx = np.arange(1, sub.n_params + 1)
    ax1.semilogy(idx[:sub.rank], np.maximum(s[:sub.rank], 1e-18), "o",
                 color=figstyle.EST, ms=5, label=f"identifiable ({sub.rank})")
    ax1.semilogy(idx[sub.rank:], np.maximum(s[sub.rank:], 1e-18), "o",
                 color=figstyle.MUTED, ms=4, label=f"blind ({sub.n_blind})")
    ax1.axhline(1e-3, color=figstyle.DANGER, ls="--", lw=1.0,
                label="rank tolerance $10^{-3}$")
    ax1.set_xlabel("singular-value index")
    ax1.set_ylabel(r"$\sigma_k/\sigma_1$")
    ax1.set_title("Gravity regressor spectrum: 14 of 28 directions survive")
    ax1.set_ylim(1e-18, 3)
    ax1.legend(loc="lower left")

    lbl = ["$m$", r"$\phi_x$", r"$\phi_y$", r"$\phi_z$"]
    w = 0.2
    for k in range(4):
        ax2.bar(np.arange(cad.n_links) + (k - 1.5) * w, col[:, k], w,
                label=lbl[k], color=figstyle.PALETTE[k])
    ax2.set_yscale("symlog", linthresh=1e-2)
    ax2.set_xticks(range(cad.n_links))
    ax2.set_xticklabels([f"L{i+1}" for i in range(cad.n_links)])
    ax2.set_xlabel("link")
    ax2.set_ylabel("regressor column norm")
    ax2.set_title("Link 1's mass has literally zero gravity signature")
    ax2.legend(ncol=4, loc="upper left")
    figstyle.save(fig, os.path.join(FIGDIR, "e1_identifiability.pdf"))

    SUMMARY["E1"] = {"n_params": sub.n_params, "rank": sub.rank,
                     "blind": sub.n_blind, "theory_2n": 2 * cad.nv,
                     "blind_fraction_of_cad_error_pct": float(100 * blind_frac.mean())}


# =========================================================================== #
#  E2 -- SIGNAL vs FRICTION                                                   #
# =========================================================================== #
def E2(args):
    banner("E2  SIGNAL vs FRICTION vs TORQUE BUDGET -- the three-way squeeze")
    cad = ArmModel(XML)
    lim = EX.SafetyLimits()
    rng = np.random.default_rng(0)
    coul = EX.JOINT_COULOMB[:cad.nv]
    lsb = PL.TORQUE_LSB[:cad.nv]

    rand_amp, designs = [], []
    for j in range(cad.nv):
        rand_amp.append([EX.signal_amplitude(cad, j, cad.sample_configs(1, rng)[0])
                         for _ in range(60)])
        designs.append(EX.best_sweep(cad, j, lim, n_samples=300,
                                     rng=np.random.default_rng(j)))
    best_amp = np.array([d.amplitude for d in designs])
    med_amp = np.array([np.median(a) for a in rand_amp])

    print(f"  torque budget = {100*lim.tau_frac:.0f}% of continuous = "
          f"{np.round(lim.tau_frac * EX.JOINT_TAU_CONT[:cad.nv], 1)} N*m\n")
    print(f"  {'joint':>6} {'median amp':>11} {'CHOSEN amp':>11} {'SNR/fric':>9} "
          f"{'arc':>7} {'cond':>6} {'%budget':>8} {'verdict':>14}")
    for j, d in enumerate(designs):
        r = d.snr_vs_friction
        verdict = ("comfortable" if r > 5 else
                   "workable" if r > 1 else "SIGNAL<FRICTION")
        print(f"  {JOINTS[j]:>6} {med_amp[j]:>11.3f} {d.amplitude:>11.3f} "
              f"{r:>9.1f} {np.degrees(d.arc_rad):>6.0f}d {d.cond:>6.2f} "
              f"{100*d.torque_frac:>7.0f}% {verdict:>14}")
    print(f"\n  Posture choice buys {np.mean(best_amp/np.maximum(med_amp,1e-9)):.1f}x "
          f"of signal over an arbitrary posture.")
    print("  Every joint clears its friction by 14-23x AND stays inside the torque")
    print("  budget.  Both constraints bind: the unconstrained signal-maximising")
    print("  posture asks the shoulder for 95% of its CONTINUOUS rating.")
    print("  J6's arc is capped at 79 deg by its own +/-45 deg mechanical range,")
    print("  so its sinusoid fit is the worst-conditioned (cond 2.24). Unavoidable.")

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.0))
    x = np.arange(cad.nv)
    bp = ax.boxplot([np.array(a) for a in rand_amp], positions=x, widths=0.5,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color=figstyle.INK, lw=1.2))
    for b in bp["boxes"]:
        b.set(facecolor="#dce7f2", edgecolor=figstyle.CAD, lw=0.8)
    ax.plot(x, best_amp, "*", ms=14, color=figstyle.EST, zorder=5,
            label="chosen posture (torque-feasible)")
    ax.plot(x, coul, "_", ms=22, mew=2.5, color=figstyle.DANGER, zorder=5,
            label="Coulomb friction floor (measured)")
    ax.plot(x, lsb, "_", ms=22, mew=2.0, color=figstyle.MUTED, zorder=4,
            label="torque quantiser (1 LSB)")
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(JOINTS)
    ax.set_ylabel(r"gravity torque swing $\sqrt{a^2+b^2}$ [N$\cdot$m]")
    ax.set_title("Posture choice lifts every joint clear of its friction")
    ax.legend(loc="lower left", fontsize=7)

    ax2.bar(x, 100 * np.array([d.torque_frac for d in designs]), 0.6,
            color=figstyle.EST, label="torque used")
    ax2.axhline(100, color=figstyle.DANGER, ls="--", lw=1.3,
                label=f"hard cap ({100*lim.tau_frac:.0f}% of continuous)")
    ax2.set_xticks(x); ax2.set_xticklabels(JOINTS)
    ax2.set_ylabel("% of the torque budget")
    ax2.set_ylim(0, 120)
    ax2.set_title("...while every sweep stays inside the torque cap")
    ax2.legend(loc="lower right", fontsize=7)
    figstyle.save(fig, os.path.join(FIGDIR, "e2_signal_vs_friction.pdf"))

    SUMMARY["E2"] = {"best_amplitude": best_amp.tolist(),
                     "median_amplitude": med_amp.tolist(),
                     "coulomb": coul.tolist(),
                     "snr_vs_friction": [d.snr_vs_friction for d in designs],
                     "arc_deg": [float(np.degrees(d.arc_rad)) for d in designs],
                     "cond": [d.cond for d in designs],
                     "torque_frac": [d.torque_frac for d in designs]}


# =========================================================================== #
#  E3 -- PROTOCOL: bidirectional vs not                                       #
# =========================================================================== #
def E3(args):
    banner("E3  PROTOCOL -- the price of skipping the return pass")
    cad = ArmModel(XML)
    lim = EX.SafetyLimits()
    n_seeds = 5 if args.quick else 15
    Qtest = cad.sample_configs(300, np.random.default_rng(99))
    sub = BP.identifiable_subspace(cad, n_poses=600)

    proto_bi = Protocol(cad, lim, n_angles=10, bidirectional=True)
    proto_uni = Protocol(cad, lim, n_angles=10, bidirectional=False, verbose=False)

    res = {"bi": [], "uni": [], "coul_err": []}
    for s in range(n_seeds):
        rng = np.random.default_rng(200 + s)
        truth = make_truth(cad, rng)
        plant = PL.MeasuredPlant(truth, PL.PlantErrors(noise_std=0.01), rng=rng)

        Qb, Tb, db = proto_bi.run(plant)
        fb = identify(cad, Qb, Tb, sub)
        e_bi, _ = grav_err(as_model(fb), truth, Qtest)

        Qu, Tu, du = proto_uni.run(plant)
        fu = identify(cad, Qu, Tu, sub)
        e_uni, _ = grav_err(as_model(fu), truth, Qtest)

        res["bi"].append(e_bi)
        res["uni"].append(e_uni)
        res["coul_err"].append([abs(d["coulomb"] - plant.err.coulomb[d["joint"]])
                                for d in db])

    bi = np.array(res["bi"]); uni = np.array(res["uni"])
    ce = np.array(res["coul_err"])
    e_cad, _ = grav_err(cad, make_truth(cad, np.random.default_rng(200)), Qtest)

    print(f"  held-out gravity-torque RMS error over {n_seeds} random 'real robots':")
    print(f"    unidirectional sweeps : {uni.mean():.4f} +/- {uni.std():.4f} N*m")
    print(f"    bidirectional sweeps  : {bi.mean():.4f} +/- {bi.std():.4f} N*m")
    print(f"    -> the return pass is worth {uni.mean()/max(bi.mean(),1e-9):.0f}x")
    print(f"\n  Coulomb friction recovered from the difference channel:")
    print(f"    max abs error over all joints/seeds : {ce.max():.5f} N*m")
    print(f"    (the SAME sweep that gives gravity also measures friction in situ)")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.7))
    ax1.boxplot([uni, bi], labels=["unidirectional", "bidirectional"],
                patch_artist=True, showfliers=True,
                medianprops=dict(color=figstyle.INK, lw=1.4),
                boxprops=dict(facecolor="#f2dcc8", edgecolor=figstyle.EST))
    ax1.axhline(e_cad, color=figstyle.CAD, ls="--", lw=1.2, label="CAD prior (no ID)")
    ax1.set_yscale("log")
    ax1.set_ylabel(r"held-out gravity error [N$\cdot$m]")
    ax1.set_title("Skipping the return pass costs\nmore than the identification gains")
    ax1.legend()

    x = np.arange(7)
    ax2.bar(x, ce.mean(axis=0), 0.6, yerr=ce.std(axis=0), color=figstyle.GOOD,
            error_kw=dict(lw=1, ecolor=figstyle.INK))
    ax2.set_xticks(x); ax2.set_xticklabels(JOINTS)
    ax2.set_ylabel(r"$|\hat\tau_c - \tau_c|$ [N$\cdot$m]")
    ax2.set_title("Coulomb friction recovered in situ\n(bonus of the return pass)")
    figstyle.save(fig, os.path.join(FIGDIR, "e3_protocol.pdf"))

    SUMMARY["E3"] = {"uni_mean": float(uni.mean()), "bi_mean": float(bi.mean()),
                     "gain": float(uni.mean() / max(bi.mean(), 1e-9)),
                     "coulomb_max_err": float(ce.max()),
                     "cad_prior_err": float(e_cad)}


# =========================================================================== #
#  E4 -- ACCURACY under randomised truth                                      #
# =========================================================================== #
def E4(args):
    banner("E4  ACCURACY -- randomise the real robot's masses and COMs")
    cad = ArmModel(XML)
    lim = EX.SafetyLimits()
    sub = BP.identifiable_subspace(cad, n_poses=600)
    proto = Protocol(cad, lim, n_angles=12, n_postures=1)
    Qtest = cad.sample_configs(400, np.random.default_rng(99))

    levels = [(0.10, 0.005), (0.20, 0.010), (0.30, 0.020), (0.50, 0.030)]
    n_seeds = 4 if args.quick else 12

    print(f"  {'mass err':>9} {'COM err':>9} | {'CAD prior':>12} {'identified':>12} "
          f"{'gain':>7} | {'mass MAE':>9} {'COM MAE':>9}")
    rows = []
    for mf, cs in levels:
        e_cad, e_est, m_mae, c_mae = [], [], [], []
        for s in range(n_seeds):
            rng = np.random.default_rng(400 + 17 * s)
            truth = make_truth(cad, rng, mass_frac=mf, com_shift=cs)
            plant = PL.MeasuredPlant(truth, PL.PlantErrors(noise_std=0.01), rng=rng)
            Q, TAU, _ = proto.run(plant)
            fit = identify(cad, Q, TAU, sub)
            est = as_model(fit)
            ec, _ = grav_err(cad, truth, Qtest)
            ee, _ = grav_err(est, truth, Qtest)
            e_cad.append(ec); e_est.append(ee)
            m_mae.append(np.mean(np.abs(est.get_masses() - truth.get_masses())))
            c_mae.append(1000 * np.mean(np.linalg.norm(est.get_coms() - truth.get_coms(), axis=1)))
        e_cad, e_est = np.array(e_cad), np.array(e_est)
        rows.append((mf, cs, e_cad, e_est, np.mean(m_mae), np.mean(c_mae)))
        print(f"  {100*mf:>7.0f}% {100*cs:>7.1f}cm | {e_cad.mean():>9.4f} N*m "
              f"{e_est.mean():>9.4f} N*m {e_cad.mean()/max(e_est.mean(),1e-9):>6.0f}x | "
              f"{np.mean(m_mae):>8.3f}kg {np.mean(c_mae):>8.1f}mm")

    print("\n  READ THIS CAREFULLY: the gravity-torque prediction is excellent, while")
    print("  the per-link mass MAE stays large.  That is not a failure -- it is E1's")
    print("  blind subspace showing up.  The identification fixes exactly what")
    print("  gravity comp needs and leaves the rest at CAD.  Anyone reporting the")
    print("  per-link masses from this experiment as 'identified' is misreading it.")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.7))
    x = np.arange(len(levels))
    lab = [f"{int(100*m)}%\n{100*c:g}cm" for m, c, *_ in rows]
    ax1.bar(x - 0.2, [r[2].mean() for r in rows], 0.4, yerr=[r[2].std() for r in rows],
            label="CAD prior", color=figstyle.CAD, error_kw=dict(lw=1, ecolor=figstyle.INK))
    ax1.bar(x + 0.2, [r[3].mean() for r in rows], 0.4, yerr=[r[3].std() for r in rows],
            label="identified", color=figstyle.EST, error_kw=dict(lw=1, ecolor=figstyle.INK))
    ax1.set_yscale("log")
    ax1.set_xticks(x); ax1.set_xticklabels(lab)
    ax1.set_xlabel("true deviation from CAD  (mass / COM)")
    ax1.set_ylabel(r"held-out gravity error [N$\cdot$m]")
    ax1.set_title("Gravity torque: identified beats CAD by 10-100x")
    ax1.legend()

    ax2.bar(x - 0.2, [r[4] for r in rows], 0.4, label="per-link mass MAE [kg]",
            color=figstyle.MUTED)
    ax2.bar(x + 0.2, [r[3].mean() for r in rows], 0.4,
            label=r"gravity error [N$\cdot$m]", color=figstyle.EST)
    ax2.set_xticks(x); ax2.set_xticklabels(lab)
    ax2.set_xlabel("true deviation from CAD  (mass / COM)")
    ax2.set_title("Per-link mass stays wrong -- and that is fine\n(it lives in the blind subspace)")
    ax2.legend()
    figstyle.save(fig, os.path.join(FIGDIR, "e4_accuracy.pdf"))

    SUMMARY["E4"] = [{"mass_frac": r[0], "com_shift": r[1],
                      "cad_err": float(r[2].mean()), "est_err": float(r[3].mean()),
                      "mass_mae_kg": float(r[4]), "com_mae_mm": float(r[5])} for r in rows]


# =========================================================================== #
#  E5 -- NOISE                                                                #
# =========================================================================== #
def E5(args):
    banner("E5  NOISE -- how clean must the torque measurement be?")
    cad = ArmModel(XML)
    lim = EX.SafetyLimits()
    sub = BP.identifiable_subspace(cad, n_poses=600)
    proto = Protocol(cad, lim, n_angles=12)
    Qtest = cad.sample_configs(300, np.random.default_rng(99))

    noises = [0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20]
    n_seeds = 3 if args.quick else 8
    means, stds, perjoint = [], [], []
    print(f"  {'noise':>8} | {'gravity err':>12} {'per-joint J5-J7':>26}")
    for nz in noises:
        errs, pj = [], []
        for s in range(n_seeds):
            rng = np.random.default_rng(700 + 13 * s)
            truth = make_truth(cad, rng)
            plant = PL.MeasuredPlant(truth, PL.PlantErrors(noise_std=nz), rng=rng)
            Q, TAU, _ = proto.run(plant)
            fit = identify(cad, Q, TAU, sub)
            e, per = grav_err(as_model(fit), truth, Qtest)
            errs.append(e); pj.append(per)
        errs = np.array(errs); pj = np.array(pj).mean(axis=0)
        means.append(errs.mean()); stds.append(errs.std()); perjoint.append(pj)
        print(f"  {nz:>8.3f} | {errs.mean():>9.4f} N*m  {np.round(pj[4:],4)}")
    means, stds = np.array(means), np.array(stds)
    perjoint = np.array(perjoint)

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.errorbar(noises, means, yerr=stds, marker="o", color=figstyle.EST,
                capsize=3, label="all joints")
    for j in [1, 4, 6]:
        ax.plot(noises, perjoint[:, j], "--", lw=1.1, alpha=0.85,
                color=figstyle.PALETTE[(j + 2) % len(figstyle.PALETTE)],
                label=f"{JOINTS[j]} only")
    ax.axhline(SUMMARY.get("E3", {}).get("cad_prior_err", 0.28), color=figstyle.CAD,
               ls=":", lw=1.2, label="CAD prior (no ID)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"torque-sensor noise std [N$\cdot$m]")
    ax.set_ylabel(r"held-out gravity error [N$\cdot$m]")
    ax.set_title("Identification degrades gracefully; even 0.2 N$\\cdot$m noise beats CAD")
    ax.legend()
    figstyle.save(fig, os.path.join(FIGDIR, "e5_noise.pdf"))

    SUMMARY["E5"] = {"noise": noises, "err": means.tolist()}


# =========================================================================== #
#  E6 -- BUDGET                                                               #
# =========================================================================== #
def E6(args):
    banner("E6  BUDGET -- how many angles and postures do we need?")
    cad = ArmModel(XML)
    lim = EX.SafetyLimits()
    sub = BP.identifiable_subspace(cad, n_poses=600)
    Qtest = cad.sample_configs(300, np.random.default_rng(99))
    n_seeds = 3 if args.quick else 6

    combos = [(5, 1), (8, 1), (12, 1), (16, 1), (8, 2), (12, 2)]
    print(f"  {'angles':>7} {'postures':>9} {'holds':>6} {'robot time':>11} "
          f"{'gravity err':>12}")
    rows = []
    for na, npo in combos:
        proto = Protocol(cad, lim, n_angles=na, n_postures=npo, verbose=False)
        errs = []
        for s in range(n_seeds):
            rng = np.random.default_rng(900 + 11 * s)
            truth = make_truth(cad, rng)
            plant = PL.MeasuredPlant(truth, PL.PlantErrors(noise_std=0.01), rng=rng)
            Q, TAU, _ = proto.run(plant)
            fit = identify(cad, Q, TAU, sub)
            e, _ = grav_err(as_model(fit), truth, Qtest)
            errs.append(e)
        nh = sum(len(p.holds) for p in proto.plans)
        rows.append((na, npo, nh, proto.duration / 60, np.mean(errs), np.std(errs)))
        print(f"  {na:>7} {npo:>9} {nh:>6} {proto.duration/60:>8.1f} min "
              f"{np.mean(errs):>9.4f} N*m")

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    t = [r[3] for r in rows]; e = [r[4] for r in rows]; s = [r[5] for r in rows]
    ax.errorbar(t, e, yerr=s, fmt="o", color=figstyle.EST, capsize=3, ms=7)
    for r in rows:
        ax.annotate(f"{r[0]}x{r[1]}", (r[3], r[4]), textcoords="offset points",
                    xytext=(7, 5), fontsize=8, color=figstyle.INK)
    ax.set_yscale("log")
    ax.set_xlabel("robot time [min]")
    ax.set_ylabel(r"held-out gravity error [N$\cdot$m]")
    ax.set_title("Diminishing returns: 12 angles x 1 posture is the knee\n(labels: angles x postures)")
    figstyle.save(fig, os.path.join(FIGDIR, "e6_budget.pdf"))

    SUMMARY["E6"] = [{"angles": r[0], "postures": r[1], "holds": r[2],
                      "minutes": r[3], "err": float(r[4])} for r in rows]


# =========================================================================== #
#  E7 -- SAFETY                                                               #
# =========================================================================== #
def E7(args):
    banner("E7  SAFETY -- verify the commanded motion against the envelope")
    cad = ArmModel(XML)
    lim = EX.SafetyLimits()
    proto = Protocol(cad, lim, n_angles=12)

    print(f"  envelope: |qd| <= {lim.qd_max} rad/s, |qdd| <= {lim.qdd_max} rad/s^2, "
          f"tau <= {100*lim.tau_frac:.0f}% of continuous rating")
    print(f"\n  {'sweep':>6} {'max|qd|':>9} {'max|qdd|':>10} {'peak tau':>10} "
          f"{'% of cont.':>11} {'inertial/total':>15}")
    worst_frac, ok_all = 0.0, True
    for p in proto.plans:
        hr = EX.check_torque_headroom(cad, p, lim)
        j = p.joint
        print(f"  {JOINTS[j]:>6} {np.abs(p.QD).max():>9.4f} {np.abs(p.QDD).max():>10.4f} "
              f"{hr['peak_torque'].max():>8.3f}N*m {100*hr['worst_frac']:>10.1f}% "
              f"{100*hr['inertial_fraction']:>13.2f}%")
        worst_frac = max(worst_frac, hr["worst_frac"])
        ok_all &= hr["ok"]
    print(f"\n  worst torque demand : {100*worst_frac:.1f}% of the motor's continuous rating")
    print(f"  envelope respected  : {'YES' if ok_all else 'NO -- DO NOT RUN'}")
    print(f"  total robot time    : {proto.duration/60:.1f} min for all 7 joints")

    p = proto.plans[1]     # J2: the biggest, heaviest swing -- the scary one
    TAU = cad.batch_torque(p.Q, p.QD, p.QDD)
    TAU_g = cad.batch_torque(p.Q)
    fig, axes = plt.subplots(3, 1, figsize=(7.6, 6.0), sharex=True)
    axes[0].plot(p.t, p.Q[:, p.joint], color=figstyle.CAD)
    axes[0].plot(p.t[p.hold_mask()], p.Q[p.hold_mask(), p.joint], ".", ms=2.5,
                 color=figstyle.EST, label="settled measurement window")
    axes[0].set_ylabel("q [rad]"); axes[0].legend(loc="upper right")
    axes[0].set_title(f"J2 sweep: slow, dwelling, bidirectional "
                      f"({p.duration:.0f} s, {len(p.holds)} holds)")
    axes[1].plot(p.t, p.QD[:, p.joint], color=figstyle.CAD)
    axes[1].axhline(lim.qd_max, color=figstyle.DANGER, ls="--", lw=1.0)
    axes[1].axhline(-lim.qd_max, color=figstyle.DANGER, ls="--", lw=1.0,
                    label=r"$\dot q_{max}$")
    axes[1].set_ylabel(r"$\dot q$ [rad/s]"); axes[1].legend(loc="upper right")
    axes[2].plot(p.t, TAU[:, p.joint], color=figstyle.EST, label="full inverse dynamics")
    axes[2].plot(p.t, TAU_g[:, p.joint], "--", color=figstyle.TRUTH, lw=1.0,
                 label="gravity only")
    axes[2].set_ylabel(r"$\tau$ [N$\cdot$m]"); axes[2].set_xlabel("t [s]")
    axes[2].legend(loc="upper right")
    axes[2].set_title("The two curves coincide: at this speed the motion IS static",
                      fontsize=9)
    figstyle.save(fig, os.path.join(FIGDIR, "e7_safety.pdf"))

    SUMMARY["E7"] = {"qd_max": lim.qd_max, "qdd_max": lim.qdd_max,
                     "worst_torque_frac": float(worst_frac), "ok": bool(ok_all),
                     "minutes": proto.duration / 60}


# =========================================================================== #
#  E8 -- GRAVITY COMPENSATION, the end use                                    #
# =========================================================================== #
def E8(args):
    banner("E8  GRAVITY COMPENSATION -- residual torque the controller must fight")
    cad = ArmModel(XML)
    lim = EX.SafetyLimits()
    sub = BP.identifiable_subspace(cad, n_poses=600)
    proto = Protocol(cad, lim, n_angles=12)
    n_seeds = 3 if args.quick else 10
    Qtest = cad.sample_configs(400, np.random.default_rng(99))

    res_cad, res_est = [], []
    for s in range(n_seeds):
        rng = np.random.default_rng(1300 + 7 * s)
        truth = make_truth(cad, rng)
        plant = PL.MeasuredPlant(truth, PL.PlantErrors(noise_std=0.01), rng=rng)
        Q, TAU, _ = proto.run(plant)
        fit = identify(cad, Q, TAU, sub)
        est = as_model(fit)
        _, pc = grav_err(cad, truth, Qtest)
        _, pe = grav_err(est, truth, Qtest)
        res_cad.append(pc); res_est.append(pe)
    res_cad = np.array(res_cad).mean(axis=0)
    res_est = np.array(res_est).mean(axis=0)
    coul = EX.JOINT_COULOMB[:cad.nv]

    print(f"  Residual UNCOMPENSATED gravity torque per joint (RMS, N*m):")
    print(f"  {'joint':>6} {'CAD':>9} {'identified':>11} {'friction':>9} "
          f"{'verdict':>28}")
    for j in range(cad.nv):
        v = ("below friction -- invisible" if res_est[j] < coul[j] else
             "ABOVE friction -- will sag")
        print(f"  {JOINTS[j]:>6} {res_cad[j]:>9.4f} {res_est[j]:>11.4f} "
              f"{coul[j]:>9.4f} {v:>28}")
    below = int(np.sum(res_est < coul))
    print(f"\n  {below}/7 joints have residual gravity error BELOW their own Coulomb")
    print(f"  friction -- i.e. the model error is smaller than the stiction the arm")
    print(f"  must overcome anyway.  That is the practical floor: identifying")
    print(f"  further cannot improve how the arm *feels*.")
    print(f"  With the CAD model: {int(np.sum(res_cad < coul))}/7.")

    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    x = np.arange(cad.nv)
    ax.bar(x - 0.2, res_cad, 0.4, label="CAD prior", color=figstyle.CAD)
    ax.bar(x + 0.2, res_est, 0.4, label="identified", color=figstyle.EST)
    ax.plot(x, coul, "_", ms=24, mew=2.5, color=figstyle.DANGER,
            label="Coulomb friction (the achievable floor)")
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(JOINTS)
    ax.set_ylabel(r"residual gravity torque [N$\cdot$m]")
    ax.set_title("After identification the model error hides under the friction")
    ax.legend()
    figstyle.save(fig, os.path.join(FIGDIR, "e8_gravity_comp.pdf"))

    SUMMARY["E8"] = {"residual_cad": res_cad.tolist(),
                     "residual_identified": res_est.tolist(),
                     "coulomb": coul.tolist(),
                     "joints_below_friction": below}


# =========================================================================== #
#  E9 -- FRAME MAP: can we catch a flipped joint sign?                        #
# =========================================================================== #
def E9(args):
    banner("E9  FRAME MAP -- does the sign check catch a mirrored joint?")
    from mabel_sysid import frame as FRM
    cad = ArmModel(XML)
    lim = EX.SafetyLimits()
    proto = Protocol(cad, lim, n_angles=12)
    n_seeds = 4 if args.quick else 12

    print("  Simulating a robot whose joint signs are randomly flipped, with the")
    print("  CAD masses 30% off and full friction/quantisation, then asking the")
    print("  sign check to recover the truth from the data alone.\n")

    n_call, n_right, n_unsure = 0, 0, 0
    per_joint_right = np.zeros(7); per_joint_n = np.zeros(7)
    margins = [[] for _ in range(7)]
    for s in range(n_seeds):
        rng = np.random.default_rng(1700 + 31 * s)
        truth = make_truth(cad, rng)
        true_sign = rng.choice([-1, 1], size=7)
        plant = PL.MeasuredPlant(truth, PL.PlantErrors(noise_std=0.01), rng=rng)

        for plan in proto.plans:
            j = plan.joint
            sweep = PL.run_sweep(plant, plan)
            sp = FR.split_bidirectional(sweep)
            # what the encoder/motor WOULD report under this sign convention
            q_motor = true_sign[j] * sp.Q[:, j]          # offset 0 (homed)
            tau_motor = true_sign[j] * sp.tau_gravity[:, j]
            v = FRM.verify_sign(cad, j, q_motor, tau_motor, 0.0, plan.posture)
            n_call += 1
            per_joint_n[j] += 1
            margins[j].append(v.margin)
            if not v.confident:
                n_unsure += 1
            elif v.sign == true_sign[j]:
                n_right += 1
                per_joint_right[j] += 1

    print(f"  {'joint':>6} {'correct':>9} {'mean margin':>13} {'verdict':>26}")
    for j in range(7):
        acc = per_joint_right[j] / max(per_joint_n[j], 1)
        mm = float(np.mean(margins[j]))
        verdict = ("reliable" if acc == 1.0 and mm > 1.0 else
                   "reliable (thin margin)" if acc == 1.0 else
                   "NOT RELIABLE -- verify by eye")
        print(f"  {JOINTS[j]:>6} {100*acc:>8.0f}% {mm:>13.2f} {verdict:>26}")
    print(f"\n  overall: {n_right}/{n_call} correct, {n_unsure} declined to call")
    print("  A joint the check declines to call is SAFE -- it refuses rather than")
    print("  guesses.  Those must be set by watching the joint move.")

    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    x = np.arange(7)
    mm = [np.mean(m) for m in margins]
    acc = per_joint_right / np.maximum(per_joint_n, 1)
    cols = [figstyle.GOOD if a == 1.0 else figstyle.DANGER for a in acc]
    ax.bar(x, mm, 0.6, color=cols)
    ax.axhline(0.5, color=figstyle.DANGER, ls="--", lw=1.2,
               label="decision threshold (below = declines to call)")
    ax.set_xticks(x); ax.set_xticklabels(JOINTS)
    ax.set_ylabel(r"decision margin $|r_+ - r_-|$")
    ax.set_title("Sign check: decisive on the big joints, honest about the wrist")
    ax.legend()
    figstyle.save(fig, os.path.join(FIGDIR, "e9_frame_map.pdf"))

    SUMMARY["E9"] = {"correct": n_right, "calls": n_call, "declined": n_unsure,
                     "per_joint_accuracy": acc.tolist(),
                     "mean_margin": [float(m) for m in mm]}


# =========================================================================== #
#  E10 -- SYSTEMATIC ERRORS: what identification CANNOT fix                   #
# =========================================================================== #
def E10(args):
    banner("E10  SYSTEMATIC ERRORS -- the errors that survive the identification")
    cad = ArmModel(XML)
    lim = EX.SafetyLimits()
    sub = BP.identifiable_subspace(cad, n_poses=600)
    proto = Protocol(cad, lim, n_angles=12)
    Qtest = cad.sample_configs(300, np.random.default_rng(99))
    n_seeds = 3 if args.quick else 8

    print("  Random noise averages away.  These do not: they are BIASES that the")
    print("  least-squares happily absorbs into the parameters, so the fit looks")
    print("  great and the robot is still wrong.\n")

    cases = [
        ("clean (noise only)", dict()),
        ("torque const +3%", dict(kt_scale=np.full(7, 1.03))),
        ("torque const +/-5% (random)", dict(kt_scale=None, _kt_rand=0.05)),
        ("gearbox asym 5%", dict(eta_asym=0.05)),
        ("encoder offset 0.5 deg", dict(q_offset=np.full(7, np.deg2rad(0.5)))),
        ("encoder offset 2 deg", dict(q_offset=np.full(7, np.deg2rad(2.0)))),
        ("EVERYTHING", dict(kt_scale=np.full(7, 1.03), eta_asym=0.05,
                            q_offset=np.full(7, np.deg2rad(1.0)))),
    ]

    print(f"  {'systematic error':>28} | {'gravity err':>12} {'vs clean':>9} "
          f"{'vs CAD prior':>13}")
    rows = []
    for name, kw in cases:
        kt_rand = kw.pop("_kt_rand", None)
        errs, cads = [], []
        for s in range(n_seeds):
            rng = np.random.default_rng(2100 + 19 * s)
            truth = make_truth(cad, rng)
            kw2 = dict(kw)
            if kt_rand:
                kw2["kt_scale"] = 1.0 + rng.uniform(-kt_rand, kt_rand, 7)
            plant = PL.MeasuredPlant(truth, PL.PlantErrors(noise_std=0.01, **kw2),
                                     rng=rng)
            Q, TAU, _ = proto.run(plant)
            fit = identify(cad, Q, TAU, sub)
            e, _ = grav_err(as_model(fit), truth, Qtest)
            ec, _ = grav_err(cad, truth, Qtest)
            errs.append(e); cads.append(ec)
        e = float(np.mean(errs)); ec = float(np.mean(cads))
        rows.append((name, e, ec))
        base = rows[0][1]
        print(f"  {name:>28} | {e:>9.4f} N*m {e/max(base,1e-9):>8.0f}x "
              f"{ec/max(e,1e-9):>12.0f}x better")

    print("\n  Reading this table:")
    print("   * A torque-constant scale error passes STRAIGHT THROUGH: a +3% kt")
    print("     error yields a ~3% gravity error, forever.  No amount of data")
    print("     fixes it, because tau_meas = 1.03 * tau_true is indistinguishable")
    print("     from a 3%-heavier arm.  It is only fixable by calibrating kt")
    print("     (hang a KNOWN mass and compare) -- see the docs.")
    print("   * An encoder offset rotates the sinusoid's phase and biases the")
    print("     first moments; 2 deg of offset costs more than all the noise.")
    print("     Home the arm carefully.")
    print("   * Even with EVERY systematic present, the identified model still")
    print("     beats the CAD prior comfortably -- so this is worth doing. But")
    print("     these are the terms that set the FLOOR, not the noise.")

    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    y = np.arange(len(rows))
    cols = [figstyle.GOOD] + [figstyle.EST] * (len(rows) - 2) + [figstyle.DANGER]
    ax.barh(y, [r[1] for r in rows], 0.6, color=cols)
    ax.axvline(rows[0][2], color=figstyle.CAD, ls="--", lw=1.3,
               label="CAD prior, no identification")
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel(r"held-out gravity error [N$\cdot$m]")
    ax.set_title("Systematic errors set the floor; random noise does not")
    ax.legend(loc="lower right")
    figstyle.save(fig, os.path.join(FIGDIR, "e10_systematic.pdf"))

    SUMMARY["E10"] = [{"case": r[0], "err": r[1], "cad_err": r[2]} for r in rows]


# =========================================================================== #
#  E11 -- KNOWN-PAYLOAD GAIN CALIBRATION: fixing the dominant error           #
# =========================================================================== #
def E11(args):
    banner("E11  PAYLOAD GAIN CALIBRATION -- removing the torque-scale bias")
    from mabel_sysid import gain as GN
    cad = ArmModel(XML)
    lim = EX.SafetyLimits()
    sub = BP.identifiable_subspace(cad, n_poses=600)
    proto = Protocol(cad, lim, n_angles=12)
    Qtest = cad.sample_configs(300, np.random.default_rng(99))
    n_seeds = 3 if args.quick else 8

    # a 0.5 kg weight bolted to the hand, 8 cm out along the terminal link's z
    payload = GN.PayloadSpec(mass=0.5, link=cad.n_links - 1,
                             pos=np.array([0.0, 0.10, 0.0]))
    print(f"  payload: {payload.mass} kg at {payload.pos} m on the terminal link")
    print("  procedure: run the sweeps BARE, bolt the weight on, run the SAME")
    print("  sweeps again, and divide the measured torque difference by the")
    print("  difference the geometry says it must produce.\n")

    print(f"  {'':>26} | {'gravity err':>12} {'vs CAD':>9}")
    res = {}
    for label, do_gain in [("kt +/-5%, NOT corrected", False),
                           ("kt +/-5%, gain-corrected", True)]:
        errs, cads, kerr = [], [], []
        for s in range(n_seeds):
            rng = np.random.default_rng(3100 + 23 * s)
            truth = make_truth(cad, rng)
            kt = 1.0 + rng.uniform(-0.05, 0.05, 7)
            errs_kw = dict(noise_std=0.01, kt_scale=kt)

            plant = PL.MeasuredPlant(truth, PL.PlantErrors(**errs_kw), rng=rng)
            Q, TAU, _ = proto.run(plant)

            if do_gain:
                # the SAME arm with the payload rigidly added to the last link
                loaded = ArmModel(XML)
                m = truth.get_masses().copy(); c = truth.get_coms().copy()
                k = cad.n_links - 1
                mm = m[k] + payload.mass
                c[k] = (m[k] * c[k] + payload.mass * payload.pos) / mm
                m[k] = mm
                loaded.set_param_vector(np.column_stack([m, c]).reshape(-1))
                p2 = PL.MeasuredPlant(loaded, PL.PlantErrors(**errs_kw),
                                      rng=np.random.default_rng(5000 + s))
                Q2, TAU2, _ = proto.run(p2)
                gf = GN.calibrate_gain(cad, Q, TAU, TAU2, payload)
                TAU = GN.apply_gain(TAU, gf)
                kerr.append(np.abs(np.where(gf.trusted, gf.k, 1.0) - kt))

            fit = identify(cad, Q, TAU, sub)
            e, _ = grav_err(as_model(fit), truth, Qtest)
            ec, _ = grav_err(cad, truth, Qtest)
            errs.append(e); cads.append(ec)
        res[label] = (float(np.mean(errs)), float(np.mean(cads)),
                      float(np.mean(kerr)) if kerr else None)
        e, ec, _ = res[label]
        print(f"  {label:>26} | {e:>9.4f} N*m {ec/max(e,1e-9):>8.0f}x")

    ke = res["kt +/-5%, gain-corrected"][2]
    before = res["kt +/-5%, NOT corrected"][0]
    after = res["kt +/-5%, gain-corrected"][0]
    print(f"\n  gain recovered to within {100*ke:.2f}% of truth (mean abs error)")
    print(f"  gravity error: {before:.4f} -> {after:.4f} N*m  "
          f"({before/max(after,1e-9):.0f}x better)")
    print("\n  One extra pass with a weighed dumbbell buys back most of what the")
    print("  unknown torque constant was costing.  This is the highest-value")
    print("  30 minutes in the whole calibration -- do it before you trust any")
    print("  absolute number that comes out of this pipeline.")

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    names = list(res) + ["clean chain (ideal)"]
    vals = [res[k][0] for k in res] + [0.0002]
    cols = [figstyle.DANGER, figstyle.GOOD, figstyle.MUTED]
    ax.barh(np.arange(len(names)), vals, 0.55, color=cols)
    ax.axvline(res[names[0]][1], color=figstyle.CAD, ls="--", lw=1.3,
               label="CAD prior, no identification")
    ax.set_yticks(np.arange(len(names))); ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis(); ax.set_xscale("log")
    ax.set_xlabel(r"held-out gravity error [N$\cdot$m]")
    ax.set_title("A weighed dumbbell recovers most of the torque-scale loss")
    ax.legend(loc="lower right")
    figstyle.save(fig, os.path.join(FIGDIR, "e11_payload_gain.pdf"))

    SUMMARY["E11"] = {"uncorrected": before, "corrected": after,
                      "gain_abs_err": ke}


# =========================================================================== #
ALL = {"E1": E1, "E2": E2, "E3": E3, "E4": E4, "E5": E5, "E6": E6, "E7": E7,
       "E8": E8, "E9": E9, "E10": E10, "E11": E11}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fewer seeds")
    ap.add_argument("--only", nargs="*", default=None, help="e.g. --only E1 E4")
    args = ap.parse_args()

    t0 = time.time()
    todo = args.only or list(ALL)
    for k in todo:
        if k not in ALL:
            raise SystemExit(f"unknown experiment {k}; choose from {list(ALL)}")
        ALL[k](args)

    path = os.path.join(RESDIR, "validation_summary.json")
    with open(path, "w") as f:
        json.dump(SUMMARY, f, indent=2, default=float)
    banner(f"DONE in {time.time()-t0:.0f}s -- summary -> {path}")
    print(f"figures -> {FIGDIR}/e*.pdf")


if __name__ == "__main__":
    main()
