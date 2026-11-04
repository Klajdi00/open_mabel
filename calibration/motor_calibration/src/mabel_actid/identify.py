"""
identify.py
===========

Fit the actuator model to measured data.  Two stages, mirroring the two
excitation types:

**Stage 1 -- friction curve (constant velocity).**
From the staircase steady points ``(w_i, tau_i)`` fit ``tau_f(w)``.  At constant
velocity there is no inertia term, so this isolates friction cleanly.
* ``model="cv"``       Coulomb + viscous, symmetric   -> linear least squares
* ``model="asym"``     + direction asymmetry          -> linear least squares
* ``model="stribeck"`` + break-away / Stribeck dip    -> nonlinear least squares

**Stage 2 -- reflected inertia (acceleration).**
From the sine / chirp time series, with the friction curve now known, the only
unknown left in ``tau = J*a + tau_f(w) + tau_g`` is ``J`` (and an optional gravity
term), recovered by linear least squares on the residual ``tau - tau_f(w)``.

Helpers:
* :func:`differentiate` -- Savitzky-Golay velocity->acceleration for real logs.
* :func:`gravity_split` -- the +/- direction trick that removes a position-
  dependent gravity load from the friction estimate.
* :func:`identify_actuator` -- run both stages and return one :class:`FrictionParams`.
"""
from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np
from scipy import optimize
from scipy.signal import savgol_filter

from .friction_model import FrictionParams, DCMotorParams, friction_torque
from .data_logging import CalibrationLog


# --------------------------------------------------------------------------- #
#  preprocessing                                                              #
# --------------------------------------------------------------------------- #
def differentiate(t: np.ndarray, w: np.ndarray, *, window: float = 0.02,
                  poly: int = 2) -> np.ndarray:
    """Acceleration from velocity by a Savitzky-Golay filter.

    ``window`` is the smoothing window in seconds; it must span several samples.
    Filtering before differentiating is essential -- raw finite differences of a
    noisy/quantised velocity are useless for inertia fitting.
    """
    dt = float(np.median(np.diff(t)))
    n = max(poly + 2, int(round(window / dt)) | 1)     # odd, > poly
    if n >= w.size:
        n = (w.size - 1) | 1
    return savgol_filter(w, n, poly, deriv=1, delta=dt)


def gravity_split(w_levels: np.ndarray, tau_levels: np.ndarray):
    """Separate friction from a constant gravity/load using +/- velocity pairs.

    For a level run at +w and -w *at the same position band*:
        tau(+w) = tau_g + tau_f(+w)
        tau(-w) = tau_g + tau_f(-w)
    Since friction is odd in w, ``(tau(+w) - tau(-w))/2`` is the friction at +w
    and ``(tau(+w) + tau(-w))/2`` estimates the gravity/load ``tau_g``.

    Returns ``(w_pos, tau_friction_pos, tau_g_estimate)``.  Levels are matched by
    nearest |w|.
    """
    w_levels = np.asarray(w_levels)
    tau_levels = np.asarray(tau_levels)
    pos = w_levels > 0
    wp, tp = w_levels[pos], tau_levels[pos]
    wn, tn = -w_levels[~pos], tau_levels[~pos]      # store |w|
    w_out, tf_out, tg_out = [], [], []
    for wi, ti in zip(wp, tp):
        if wn.size == 0:
            w_out.append(wi); tf_out.append(ti); tg_out.append(0.0)
            continue
        j = int(np.argmin(np.abs(wn - wi)))
        w_out.append(wi)
        tf_out.append(0.5 * (ti - tn[j]))
        tg_out.append(0.5 * (ti + tn[j]))
    return np.asarray(w_out), np.asarray(tf_out), np.asarray(tg_out)


# --------------------------------------------------------------------------- #
#  stage 1 : friction curve                                                   #
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class CurveFit:
    params: FrictionParams
    rms: float            # N*m fit residual (all points)
    rms_inlier: float     # N*m residual over inliers (stick-slip rejected)
    r2: float             # coefficient of determination
    model: str


def fit_friction_curve(w: np.ndarray, tau: np.ndarray, *,
                       model: str = "stribeck") -> CurveFit:
    """Fit ``tau_f(w)`` to constant-velocity points ``(w, tau)``."""
    w = np.asarray(w, float)
    tau = np.asarray(tau, float)
    if w.size < 3:
        raise ValueError("need at least 3 velocity levels to fit friction")

    if model in ("cv", "asym"):
        params = _fit_linear(w, tau, asym=(model == "asym"))
    elif model == "stribeck":
        params = _fit_stribeck(w, tau)
    else:
        raise ValueError(f"unknown model '{model}'")

    pred = friction_torque(params, w)
    resid = pred - tau
    rms = float(np.sqrt(np.mean(resid ** 2)))
    # inlier RMS: drop the stick-slip outliers (residual > 3.5 MAD) so the
    # number reflects the quality of the fit on the points it actually trusts
    mad = float(np.median(np.abs(resid - np.median(resid)))) or 1e-9
    inliers = np.abs(resid) <= 3.5 * 1.4826 * mad
    rms_in = float(np.sqrt(np.mean(resid[inliers] ** 2))) if inliers.any() else rms
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((tau - tau.mean()) ** 2)) or 1.0
    return CurveFit(params=params, rms=rms, rms_inlier=rms_in,
                    r2=1.0 - ss_res / ss_tot, model=model)


def _fit_linear(w, tau, asym: bool) -> FrictionParams:
    """Coulomb (+/-) + viscous via plain least squares."""
    s = np.sign(w)
    if asym:
        cols = [np.where(w > 0, 1.0, 0.0), np.where(w < 0, -1.0, 0.0), w]
        A = np.column_stack(cols)
        coef, *_ = np.linalg.lstsq(A, tau, rcond=None)
        Fc_pos, Fc_neg, Fv = coef
    else:
        A = np.column_stack([s, w])
        coef, *_ = np.linalg.lstsq(A, tau, rcond=None)
        Fc_pos = Fc_neg = coef[0]
        Fv = coef[1]
    return FrictionParams(Fc_pos=abs(Fc_pos), Fc_neg=abs(Fc_neg), Fv=Fv,
                          Fs_pos=abs(Fc_pos), Fs_neg=abs(Fc_neg))


def _fit_stribeck(w, tau) -> FrictionParams:
    """Full asymmetric Coulomb + Stribeck + viscous (nonlinear).

    Staged initialisation: above the Stribeck dip the curve is pure Coulomb +
    viscous, so a linear fit to the *high-speed* points robustly pins
    ``Fc, Fv``; the break-away ``Fs`` and Stribeck velocity ``v_s`` are then
    seeded from the low-speed torque excess.  Starting the nonlinear solve here
    avoids the degenerate minimum (``v_s`` blown up so the exponential is flat)
    that a naive single-stage fit falls into.
    """
    wabs = np.abs(w)
    w_lo, w_hi = wabs[wabs > 0].min(), wabs.max()

    # stage 1: Coulomb + viscous from the upper half of the speed range
    thr = max(np.median(wabs), 3.0 * w_lo)
    hi_mask = wabs >= thr
    if hi_mask.sum() >= 3:
        lin = _fit_linear(w[hi_mask], tau[hi_mask], asym=True)
        # the high-speed residual is a clean estimate of the measurement noise,
        # used to scale the robust loss below
        hi_resid = friction_torque(lin, w[hi_mask]) - tau[hi_mask]
        sigma = float(np.std(hi_resid))
    else:
        lin = _fit_linear(w, tau, asym=True)
        sigma = float(np.std(friction_torque(lin, w) - tau))
    f_scale = max(5.0 * sigma, 1e-3)

    # stage 2: seed Fs from the low-speed torque excess over Coulomb
    lo_mask = wabs <= max(2.0 * w_lo, 0.3 * thr)
    def _excess(sign):
        m = lo_mask & (np.sign(w) == sign)
        if not m.any():
            return 0.0
        return max(np.mean(np.abs(tau[m])) - (lin.Fc_pos if sign > 0
                   else lin.Fc_neg), 0.0)
    dFs0_p, dFs0_n = _excess(1), _excess(-1)
    v_s0 = min(max(thr * 0.5, 2.0 * w_lo), 0.5 * w_hi)

    fc_scale = max(lin.Fc_pos, lin.Fc_neg, 1e-3)
    # x = [Fc_pos, Fc_neg, Fv, dFs_pos, dFs_neg, v_s] ; Fs = Fc + dFs (>=0)
    x0 = np.array([lin.Fc_pos, lin.Fc_neg, lin.Fv, dFs0_p, dFs0_n, v_s0])
    # v_s is bounded well inside the speed range: the dip lives at low speed, and
    # an unbounded v_s is exactly the runaway that flattens the exponential.
    lo = np.array([0, 0, -np.inf, 0, 0, max(w_lo, 1e-3)])
    hi = np.array([np.inf, np.inf, np.inf, 5 * fc_scale, 5 * fc_scale,
                   0.5 * w_hi])

    def unpack(x):
        return FrictionParams(
            Fc_pos=x[0], Fc_neg=x[1], Fv=x[2],
            Fs_pos=x[0] + x[3], Fs_neg=x[1] + x[4], v_s=x[5], delta=2.0)

    def resid(x):
        return friction_torque(unpack(x), w) - tau

    # robust loss: a handful of low-speed points are corrupted by stick-slip
    # (Stribeck's negative-slope region has no steady friction to measure), and
    # soft_l1 down-weights those outliers instead of letting them bias the fit.
    sol = optimize.least_squares(resid, x0, bounds=(lo, hi), method="trf",
                                 loss="soft_l1", f_scale=f_scale,
                                 xtol=1e-12, ftol=1e-12)
    return unpack(sol.x)


# --------------------------------------------------------------------------- #
#  stage 2 : reflected inertia                                                #
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class InertiaFit:
    J: float
    tau_g_amp: float      # fitted gravity amplitude (0 if not requested), N*m
    rms: float


def fit_inertia(w: np.ndarray, a: np.ndarray, tau: np.ndarray,
                friction: FrictionParams, *, q: Optional[np.ndarray] = None,
                fit_gravity: bool = False) -> InertiaFit:
    """Recover ``J`` from ``tau - tau_f(w) = J*a (+ gravity)`` by least squares.

    With ``fit_gravity`` and a position array ``q`` it also fits a single
    sinusoidal load ``A*sin(q)`` (a pendulum about a horizontal axis), useful
    when the bench has a known but unmeasured load.
    """
    resid_tau = tau - friction_torque(friction, w)
    if fit_gravity and q is not None:
        A = np.column_stack([a, np.sin(q)])
        coef, *_ = np.linalg.lstsq(A, resid_tau, rcond=None)
        J, g_amp = float(coef[0]), float(coef[1])
        pred = A @ coef
    else:
        J = float(np.dot(a, resid_tau) / np.dot(a, a))
        g_amp = 0.0
        pred = J * a
    rms = float(np.sqrt(np.mean((pred - resid_tau) ** 2)))
    return InertiaFit(J=J, tau_g_amp=g_amp, rms=rms)


# --------------------------------------------------------------------------- #
#  torque-speed envelope (Isaac DCMotor)                                      #
# --------------------------------------------------------------------------- #
def fit_torque_speed(w: np.ndarray, tau: np.ndarray, *,
                     tau_cont: float = 0.0, w_max_hint: float = 0.0
                     ) -> DCMotorParams:
    """Extract what the torque–speed sweep can reveal on a SINGLE FREE shaft.

    Physics caveat (important, and reflected in the provenance flags): a free
    low-inertia shaft driven at max torque snaps to its terminal speed in
    milliseconds, so the only point of the torque–speed line you reliably see is
    the **no-load (terminal) speed** — ``velocity_limit``.  The **stall torque**
    and the line *slope* need the motor loaded (a dynamometer / brake), so they
    are NOT recoverable here unless a genuine torque droop happens to appear.

    Returns:
      * ``w_noload``  measured terminal speed (the 99th-pct |w| reached) — this is
        ``velocity_limit``; slightly below the true zero-torque no-load because
        friction loads it. Always set when the shaft actually span up.
      * ``tau_peak``  max torque seen near zero speed (a lower bound on stall).
      * ``tau_stall`` + ``observed=True`` ONLY if a clear multi-bin declining
        envelope with good R² is present (e.g. a loaded run); else 0 / False, and
        the caller falls back to the datasheet for ``saturation_effort``.
    """
    w = np.asarray(w, float); tau = np.asarray(tau, float)
    aw, at = np.abs(w), np.abs(tau)
    drive = np.sign(w) == np.sign(tau)            # motor torque aids motion
    aw, at = aw[drive], at[drive]
    if aw.size < 20 or aw.max() < 1e-3:
        return DCMotorParams(tau_cont=tau_cont, observed=False)

    wmax = float(np.percentile(aw, 99))           # terminal / no-load speed
    low = aw <= max(0.15 * wmax, 1e-3)
    tau_peak = float(np.percentile(at[low], 90)) if low.any() else float(at.max())

    # try a declining-envelope fit (only succeeds if the motor was loaded)
    edges = np.linspace(0, wmax, 13)
    bw, bt = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (aw >= lo) & (aw < hi)
        if m.sum() >= 3:
            bw.append(0.5 * (lo + hi)); bt.append(float(np.percentile(at[m], 90)))
    bw, bt = np.asarray(bw), np.asarray(bt)
    observed, tau_stall, r2 = False, 0.0, 0.0
    if bw.size >= 5:
        decl = bt < 0.9 * tau_peak
        if decl.sum() >= 3:
            x, y = bw[decl], bt[decl]
            A = np.vstack([np.ones_like(x), x]).T
            (a0, a1), *_ = np.linalg.lstsq(A, y, rcond=None)
            pred = A @ np.array([a0, a1])
            ss = float(np.sum((pred - y) ** 2))
            st = float(np.sum((y - y.mean()) ** 2)) or 1.0
            r2 = 1.0 - ss / st
            if a1 < -1e-6 and r2 > 0.8:           # a real, clean droop
                observed = True
                tau_stall = float(a0)
                wmax = float(-a0 / a1)            # refine no-load from the line

    return DCMotorParams(tau_stall=tau_stall, w_noload=wmax, tau_cont=tau_cont,
                         tau_peak=tau_peak, r2=r2, observed=observed)


# --------------------------------------------------------------------------- #
#  full pipeline                                                              #
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class ActuatorResult:
    params: FrictionParams
    curve: CurveFit
    inertia: Optional[InertiaFit]
    report: dict


def identify_actuator(staircase: CalibrationLog, *,
                      dynamic: Optional[CalibrationLog] = None,
                      model: str = "stribeck",
                      use_gravity_split: bool = False,
                      diff_window: float = 0.02) -> ActuatorResult:
    """End-to-end fit: friction from the staircase, inertia from the sweep.

    ``staircase``  constant-velocity run (required) -> friction curve.
    ``dynamic``    optional sine/chirp run -> reflected inertia ``J``.
    ``model``      friction model: ``cv`` | ``asym`` | ``stribeck``.
    ``use_gravity_split``  apply the +/- pairing trick to the staircase first
                   (use when the bench has a position-dependent load).
    """
    w_lvl, tau_lvl = staircase.steady_points()
    tau_g_meas = 0.0
    if use_gravity_split:
        w_lvl, tau_lvl, tg = gravity_split(w_lvl, tau_lvl)
        tau_g_meas = float(np.mean(tg))

    curve = fit_friction_curve(w_lvl, tau_lvl, model=model)
    params = dataclasses.replace(curve.params)        # copy

    inertia = None
    if dynamic is not None:
        a = dynamic.a
        if not np.all(np.isfinite(a)) or np.allclose(a, 0.0):
            a = differentiate(dynamic.t, dynamic.w, window=diff_window)
        inertia = fit_inertia(dynamic.w, a, dynamic.tau, params)
        params.J = inertia.J

    report = {
        "model": model,
        "n_levels": int(w_lvl.size),
        "friction_rms_Nm": curve.rms,
        "friction_rms_inlier_Nm": curve.rms_inlier,
        "friction_r2": curve.r2,
        "gravity_split_tau_g_Nm": tau_g_meas,
        "params": params.as_dict(),
        "mujoco": params.mujoco_attrs(),
    }
    if inertia is not None:
        report["inertia_J_kgm2"] = inertia.J
        report["inertia_rms_Nm"] = inertia.rms
    return ActuatorResult(params=params, curve=curve, inertia=inertia,
                          report=report)
