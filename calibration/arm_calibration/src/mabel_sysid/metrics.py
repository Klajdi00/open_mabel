"""
metrics.py
==========

Reporting helpers: compare identified (mass, COM) against ground truth, evaluate
torque-prediction accuracy on held-out data, and render plots.
"""
from __future__ import annotations

import numpy as np

from .arm_model import ArmModel
from .data_generation import Dataset


def param_error_table(names, true_m, true_c, est_m, est_c) -> str:
    """Pretty per-link comparison table as a string."""
    lines = []
    hdr = (f"{'link':<14}{'m_true':>9}{'m_est':>9}{'Δm%':>8}"
           f"{'|c_true|':>10}{'|c_est|':>10}{'Δcom(mm)':>10}")
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for i, nm in enumerate(names):
        dm = 100.0 * (est_m[i] - true_m[i]) / true_m[i]
        cerr_mm = 1000.0 * np.linalg.norm(est_c[i] - true_c[i])
        lines.append(
            f"{nm:<14}{true_m[i]:>9.3f}{est_m[i]:>9.3f}{dm:>8.1f}"
            f"{np.linalg.norm(true_c[i]):>10.4f}{np.linalg.norm(est_c[i]):>10.4f}"
            f"{cerr_mm:>10.2f}")
    # summary
    mass_mae = np.mean(np.abs(est_m - true_m))
    com_mae_mm = 1000.0 * np.mean(np.linalg.norm(est_c - true_c, axis=1))
    lines.append("-" * len(hdr))
    lines.append(f"mean |Δmass| = {mass_mae:.4f} kg     "
                 f"mean |Δcom| = {com_mae_mm:.2f} mm")
    return "\n".join(lines)


def first_moment_error(true_m, true_c, est_m, est_c) -> dict:
    """Compare first moments phi = m*c (the gravity-relevant quantity).

    Returned in g*m (gram-metre) units, which are intuitive for small links.
    """
    true_phi = true_m[:, None] * true_c
    est_phi = est_m[:, None] * est_c
    err = np.linalg.norm(est_phi - true_phi, axis=1)
    rel = err / (np.linalg.norm(true_phi, axis=1) + 1e-12)
    return {
        "per_link_err_gm": 1000.0 * err,        # g*m
        "per_link_rel": rel,
        "mae_gm": float(1000.0 * np.mean(err)),
        "mean_rel": float(np.mean(rel)),
    }


def gravity_prediction_rms(est_arm: ArmModel, Q: np.ndarray,
                           true_arm: ArmModel) -> float:
    """RMS error of the predicted *gravity* torque g(q) over poses ``Q``."""
    pred = np.array([est_arm.gravity_torque(q) for q in Q])
    truth = np.array([true_arm.gravity_torque(q) for q in Q])
    return float(np.sqrt(np.mean((pred - truth) ** 2)))


def torque_prediction_rms(est_arm: ArmModel, data: Dataset) -> dict:
    """RMS torque error of the identified model on ``data`` (per joint + total).

    Compares against the *clean* (noise-free) torque so we measure model error,
    not the injected noise floor.
    """
    pred = est_arm.batch_torque(data.Q, data.QD, data.QDD)
    err = pred - data.TAU_CLEAN
    per_joint = np.sqrt(np.mean(err ** 2, axis=0))
    return {
        "per_joint_rms": per_joint,
        "total_rms": float(np.sqrt(np.mean(err ** 2))),
    }


def plot_results(out_path, names, true_m, true_c, est_m, est_c,
                 baseline_m=None, baseline_c=None,
                 singular_values=None):
    """Bar charts of mass / COM error (+ optional regressor spectrum)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(names)
    x = np.arange(n)
    ncols = 3 if singular_values is not None else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5.5 * ncols, 4.2))

    # --- mass ---
    ax = axes[0]
    w = 0.35
    if baseline_m is not None:
        ax.bar(x - w / 2, baseline_m, w, label="initial guess", color="#cccccc")
        ax.bar(x + w / 2, est_m, w, label="identified", color="#d1495b")
    else:
        ax.bar(x, est_m, w, label="identified", color="#d1495b")
    ax.plot(x, true_m, "k_", markersize=18, markeredgewidth=2.5, label="truth")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("mass (kg)"); ax.set_title("Link mass"); ax.legend(fontsize=8)

    # --- com error ---
    ax = axes[1]
    com_err_mm = 1000.0 * np.linalg.norm(est_c - true_c, axis=1)
    if baseline_c is not None:
        base_err = 1000.0 * np.linalg.norm(baseline_c - true_c, axis=1)
        ax.bar(x - w / 2, base_err, w, label="initial guess", color="#cccccc")
        ax.bar(x + w / 2, com_err_mm, w, label="identified", color="#0073b7")
    else:
        ax.bar(x, com_err_mm, w, color="#0073b7")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("|COM error| (mm)"); ax.set_title("COM error"); ax.legend(fontsize=8)

    # --- singular spectrum ---
    if singular_values is not None:
        ax = axes[2]
        ax.semilogy(singular_values, "o-", color="#2e7d32")
        ax.set_xlabel("index"); ax.set_ylabel("singular value")
        ax.set_title("Gravity regressor spectrum")
        ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path
