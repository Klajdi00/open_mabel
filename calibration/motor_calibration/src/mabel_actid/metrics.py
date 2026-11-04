"""
metrics.py
==========

Reporting for the actuator identification: a friction-curve plot, a torque
time-series fit plot, and text tables comparing identified vs (in the sim test)
true parameters.

Plotting uses a non-interactive Agg backend so it runs headless.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt           # noqa: E402

from .friction_model import FrictionParams, friction_torque, torque_speed_envelope
from .data_logging import CalibrationLog
from .identify import differentiate


def param_table(p: FrictionParams, truth: Optional[FrictionParams] = None
                ) -> str:
    """Pretty text table of identified (vs true) parameters."""
    rows = [
        ("Coulomb +  Fc_pos", "N*m", p.Fc_pos, truth.Fc_pos if truth else None),
        ("Coulomb -  Fc_neg", "N*m", p.Fc_neg, truth.Fc_neg if truth else None),
        ("viscous    Fv", "N*m*s/rad", p.Fv, truth.Fv if truth else None),
        ("static +   Fs_pos", "N*m", p.Fs_pos, truth.Fs_pos if truth else None),
        ("static -   Fs_neg", "N*m", p.Fs_neg, truth.Fs_neg if truth else None),
        ("Stribeck v v_s", "rad/s", p.v_s, truth.v_s if truth else None),
        ("inertia    J", "kg*m^2", p.J, truth.J if truth else None),
    ]
    if truth is None:
        lines = [f"{'parameter':<20}{'unit':<12}{'identified':>14}"]
        lines.append("-" * 46)
        for name, unit, val, _ in rows:
            lines.append(f"{name:<20}{unit:<12}{val:>14.5f}")
    else:
        lines = [f"{'parameter':<20}{'unit':<12}{'true':>12}"
                 f"{'identified':>14}{'error':>12}"]
        lines.append("-" * 70)
        for name, unit, val, tv in rows:
            err = abs(val - tv)
            lines.append(f"{name:<20}{unit:<12}{tv:>12.5f}"
                         f"{val:>14.5f}{err:>12.5f}")
    return "\n".join(lines)


def plot_friction_curve(w_lvl, tau_lvl, p: FrictionParams, path: str, *,
                        truth: Optional[FrictionParams] = None,
                        title: str = "Actuator friction curve") -> None:
    """Measured steady points + fitted (and optional true) friction model."""
    order = np.argsort(w_lvl)
    w_lvl, tau_lvl = np.asarray(w_lvl)[order], np.asarray(tau_lvl)[order]
    wgrid = np.linspace(min(w_lvl.min(), -1e-3), max(w_lvl.max(), 1e-3), 400)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axhline(0, color="0.8", lw=0.8)
    ax.axvline(0, color="0.8", lw=0.8)
    ax.plot(w_lvl, tau_lvl, "o", ms=6, color="tab:blue",
            label="measured (constant-v)")
    ax.plot(wgrid, friction_torque(p, wgrid), "-", color="tab:red", lw=2,
            label="identified model")
    if truth is not None:
        ax.plot(wgrid, friction_torque(truth, wgrid), "--", color="k", lw=1.2,
                label="ground truth")
    ax.set_xlabel("joint velocity  w  [rad/s]")
    ax.set_ylabel("friction torque  tau_f  [N*m]")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_torque_fit(log: CalibrationLog, p: FrictionParams, path: str, *,
                    diff_window: float = 0.02,
                    title: str = "Dynamic torque: measured vs model") -> None:
    """Overlay measured torque and the model prediction on a sweep."""
    a = log.a
    if not np.all(np.isfinite(a)) or np.allclose(a, 0.0):
        a = differentiate(log.t, log.w, window=diff_window)
    pred = p.J * a + friction_torque(p, log.w)
    resid = pred - log.tau
    rms = float(np.sqrt(np.mean(resid ** 2)))

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    axes[0].plot(log.t, log.tau, color="tab:blue", lw=1.0, label="measured")
    axes[0].plot(log.t, pred, color="tab:red", lw=1.0, alpha=0.8,
                 label="model J*a + tau_f(w)")
    axes[0].set_ylabel("torque [N*m]")
    axes[0].set_title(f"{title}   (residual RMS {rms:.4f} N*m)")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(log.t, resid, color="0.4", lw=0.8)
    axes[1].set_ylabel("residual")
    axes[1].set_xlabel("time [s]")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_torque_speed(log: CalibrationLog, dc, path: str, *,
                      title: str = "Torque-speed envelope") -> None:
    """Scatter the swept (w, tau) and overlay the fitted DC-motor envelope."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axhline(0, color="0.8", lw=0.8); ax.axvline(0, color="0.8", lw=0.8)
    ax.plot(log.w, log.tau, ".", ms=3, color="tab:blue", alpha=0.5,
            label="measured (max-effort sweep)")
    if dc is not None and dc.w_noload > 0:
        wgrid = np.linspace(-dc.w_noload, dc.w_noload, 400)
        ax.plot(wgrid, torque_speed_envelope(dc, wgrid), "-", color="tab:red",
                lw=2, label="fitted envelope")
        tag = "measured" if dc.observed else "extrapolated / datasheet"
        ax.plot([], [], " ",
                label=f"stall {dc.tau_stall:.1f} N·m   no-load {dc.w_noload:.1f} rad/s ({tag})")
        if dc.tau_cont > 0:
            ax.axhline(dc.tau_cont, color="tab:green", ls="--", lw=1,
                       label=f"continuous (datasheet) {dc.tau_cont:.1f} N·m")
            ax.axhline(-dc.tau_cont, color="tab:green", ls="--", lw=1)
    ax.set_xlabel("joint velocity  w  [rad/s]")
    ax.set_ylabel("torque  tau  [N·m]")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
