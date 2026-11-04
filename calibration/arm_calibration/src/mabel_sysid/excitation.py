"""
excitation.py
=============

**Safe excitation: one joint at a time, slowly, and provably sufficient.**

The classical identification recipe (Swevers et al. 1997) is a fast multi-joint
Fourier trajectory that maximises the condition number of the full dynamic
regressor.  On a 6 kg arm swinging near a person, that is exactly the trajectory
you do *not* want to run.  This module implements the opposite extreme, and
shows it costs nothing for our purpose:

    Hold every joint but one.  Sweep joint j slowly through its range,
    pausing at a set of angles.  The static torque at joint j is then

        tau_j(q_j) = a_j cos(q_j) + b_j sin(q_j)                       (exact)

    -- a two-parameter sinusoid, verified to machine precision against MuJoCo
    (R^2 = 1.0000000, residual < 1e-14 N*m, zero constant offset).

Two parameters is *exactly* the number of gravity parameters joint j can ever
reveal (see :mod:`mabel_sysid.base_params`).  So the slow, safe, one-joint
protocol is not a degraded version of the aggressive one -- for gravity
identification it is **complete**.  Velocity and acceleration are kept low
enough that the inertial terms M(q)qddot and the Coriolis terms are negligible
compared to gravity, which is what lets us treat the measurement as static.

What the slow protocol genuinely cannot give you
------------------------------------------------
Link *inertia tensors* and *reflected rotor inertia* need q̈ ≠ 0 to show up in
the torque.  Those are NOT identified here -- they come from the separate
per-motor `motor_calibration/` package (armature) and, for the link tensors,
from CAD.  For gravity compensation and for the stiffness/damping terms of an
impedance controller, they are not needed.  Do not let anyone tell you this
protocol identified an inertia tensor: it did not.

Posture and arc are chosen together, under a hard torque cap
------------------------------------------------------------
The amplitude sqrt(a^2 + b^2) of joint j's gravity sinusoid depends on the
posture of the *other* joints, and it varies enormously: sweep a wrist joint in
a bad posture and its gravity signal drops below the DM-J4310's 0.083 N*m
Coulomb friction, i.e. below the very thing you must subtract.  So we do not
sweep at an arbitrary posture.

But the naive fix -- "maximise the signal" -- is a trap.  With the ORCA hand
attached this arm weighs 6.4 kg, and the posture that maximises the shoulder's
gravity swing is the arm held straight out: ~19 N*m, or **95% of the DM-J8009's
20 N*m continuous rating**, held for the minutes a sweep takes.  That cooks the
motor and leaves no headroom if something snags.  Worse, it is the *holding*
joints that blow the budget, not the joint being swept.

:func:`best_sweep` therefore co-optimises the posture AND the swept arc subject
to a **hard** cap of ``SafetyLimits.tau_frac`` of every joint's continuous
rating (a soft penalty is not a constraint -- gradient descent will happily
settle over budget whenever the signal gain pays for the penalty).  At the
default 50% cap all seven joints come out with a 14-23x signal-to-friction
margin and a well-conditioned arc, so nothing of value is given up.
"""
from __future__ import annotations

import dataclasses
from typing import List, Optional, Sequence

import numpy as np

from .arm_model import ArmModel


# --------------------------------------------------------------------------- #
#  Safety envelope                                                             #
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class SafetyLimits:
    """The hard envelope every generated motion is checked against.

    Defaults are deliberately timid: at 0.15 rad/s a joint takes ~20 s to cross
    a 3 rad range, and the peak inertial torque of even the shoulder link is
    ~1% of its gravity torque.  You should be able to stop the arm with your
    hand.  Raise these only after a first successful run.
    """

    qd_max: float = 0.15          # rad/s   peak joint speed  (~8.6 deg/s)
    qdd_max: float = 0.30         # rad/s^2 peak joint acceleration
    # Fraction of the motor's CONTINUOUS torque any joint may be asked to hold.
    # 0.50 is the lowest value at which all 7 joints still have a torque-feasible
    # excitation (the arm + ORCA hand is 6.4 kg and the shoulder needs ~19 N*m of
    # its 20 N*m continuous rating just to hold the arm out).  At 0.50 the
    # shoulder sits at 10 N*m = 25% of its 40 N*m PEAK, so there is ample
    # headroom for a snag.  Below ~0.45 the problem becomes infeasible.
    tau_frac: float = 0.50
    margin: float = 0.10          # rad, stay this far away from every joint limit
    dwell: float = 1.2            # s, hold at each sample angle (let it settle)
    settle: float = 0.4           # s, discard this much of each dwell before averaging
    backoff: float = 0.05         # rad, direction-establishing excursion (see below)

    def check(self, QD: np.ndarray, QDD: np.ndarray) -> None:
        """Raise if a generated trajectory violates the envelope."""
        vmax = float(np.abs(QD).max()) if QD.size else 0.0
        amax = float(np.abs(QDD).max()) if QDD.size else 0.0
        if vmax > self.qd_max * 1.001:
            raise ValueError(f"trajectory exceeds qd_max: {vmax:.4f} > {self.qd_max}")
        if amax > self.qdd_max * 1.001:
            raise ValueError(f"trajectory exceeds qdd_max: {amax:.4f} > {self.qdd_max}")


# Per-joint CONTINUOUS torque rating of the motor driving it (N*m), joints 1..7.
# DM-J8009P (rated 20), DM-J8009P, DM-J4340 (rated 9), DM-J4340,
# DM-J4310 (rated 3), DM-J4310, DM-J4310.  Source: calibration/.../motors.py.
JOINT_MOTORS = ["DM-J8009P", "DM-J8009P", "DM-J4340", "DM-J4340",
                "DM-J4310", "DM-J4310", "DM-J4310"]
JOINT_TAU_CONT = np.array([20.0, 20.0, 9.0, 9.0, 3.0, 3.0, 3.0])

# Measured Coulomb friction per motor type (N*m, output side) from the
# motor_calibration package.  DM-J8009P has NOT been characterised yet -- the
# value below is a conservative placeholder scaled from the DM-J4340 and MUST be
# replaced by a real bench measurement before you trust joint 1/2 numbers.
JOINT_COULOMB = np.array([0.60, 0.60,       # DM-J8009P  (PLACEHOLDER -- unmeasured)
                          0.4299, 0.4299,   # DM-J4340   (measured, R^2 0.9992)
                          0.0829, 0.0829, 0.0829])  # DM-J4310 (measured, R^2 0.9933)
COULOMB_MEASURED = np.array([False, False, True, True, True, True, True])


class InfeasiblePosture(RuntimeError):
    """No posture satisfies the torque budget -- the arm is too heavy for it."""


# --------------------------------------------------------------------------- #
#  Posture selection: maximise the gravity signal of the joint under test      #
# --------------------------------------------------------------------------- #
def sinusoid_coeffs(arm: ArmModel, j: int, posture: np.ndarray) -> tuple:
    """The exact (a, b) of ``tau_j(q_j) = a cos q_j + b sin q_j`` at ``posture``.

    Obtained by evaluating the true gravity torque at two angles 90 deg apart --
    no fitting needed, because the relation is exact.
    """
    q = np.asarray(posture, dtype=float).copy()
    q[j] = 0.0
    a = arm.gravity_torque(q)[j]              # tau(0)   = a
    q[j] = np.pi / 2
    b = arm.gravity_torque(q)[j]              # tau(pi/2) = b
    return float(a), float(b)


def signal_amplitude(arm: ArmModel, j: int, posture: np.ndarray) -> float:
    """Peak gravity torque swing of joint j at this posture: sqrt(a^2 + b^2)."""
    a, b = sinusoid_coeffs(arm, j, posture)
    return float(np.hypot(a, b))


def arc_bounds(arm: ArmModel, j: int, lim: SafetyLimits, center: Optional[float],
               span: float) -> tuple:
    """The (lo, hi) angles of joint j's swept arc, clipped into the safe range."""
    rlo = arm.q_lower[j] + lim.margin
    rhi = arm.q_upper[j] - lim.margin
    half = 0.5 * (rhi - rlo) * float(np.clip(span, 0.05, 1.0))
    c = 0.5 * (rlo + rhi) if center is None else float(center)
    c = float(np.clip(c, rlo + half, rhi - half))
    return c - half, c + half


def sweep_peak_torque(arm: ArmModel, j: int, posture: np.ndarray, lim: SafetyLimits,
                      *, center: Optional[float] = None, span: float = 1.0,
                      n_probe: int = 11) -> np.ndarray:
    """Peak |gravity torque| on EVERY joint over the swept arc of joint j.

    Not just joint j: the joints *holding* the posture carry torque too, and the
    shoulder holding a 6.4 kg arm out at arm's length is what will overheat or
    sag -- not the little wrist joint being swept.
    """
    a, b = arc_bounds(arm, j, lim, center, span)
    q = np.asarray(posture, float).copy()
    peak = np.zeros(arm.nv)
    for qj in np.linspace(a, b, n_probe):
        q[j] = qj
        peak = np.maximum(peak, np.abs(arm.gravity_torque(q)))
    return peak


def sinusoid_conditioning(angles: np.ndarray) -> float:
    """Condition number of the 2-parameter sinusoid design matrix [cos q, sin q].

    A short arc makes cos and sin nearly collinear, so (a, b) become hard to
    separate even with a clean, high-amplitude signal.  A 90 deg arc gives
    cond ~ 2.4; a 30 deg arc gives cond ~ 15.  Joint 6's mechanical range is only
    +/-45 deg, so it is stuck near the worst end of this and there is nothing the
    protocol can do about it -- report it, do not hide it.
    """
    A = np.column_stack([np.cos(angles), np.sin(angles)])
    s = np.linalg.svd(A, compute_uv=False)
    return float(s[0] / max(s[-1], 1e-12))


def best_posture(arm: ArmModel, j: int, lim: SafetyLimits, *,
                 n_restarts: int = 40,
                 rng: Optional[np.random.Generator] = None,
                 fixed: Optional[dict] = None,
                 respect_torque: bool = True) -> tuple:
    """Posture of the OTHER joints that maximises joint j's gravity signal,
    **subject to the torque budget**.

    The unconstrained optimum is a trap.  With the hand attached, the posture
    that maximises joint 2's gravity swing is the arm held straight out --
    17.1 N*m, or 85% of the DM-J8009's 20 N*m continuous rating, held for the
    ~2 minutes the sweep takes.  That is how you cook a motor, and it leaves no
    headroom if something snags.  We therefore maximise the signal only within
    ``lim.tau_frac`` of every joint's continuous rating.

    There is plenty to give up: at the corrected base orientation every joint has
    a 16-29x signal-to-friction margin available, so trading peak signal for
    thermal and collision headroom costs nothing that matters.

    Multi-start local search.  Joint j itself is left at 0 in the returned
    posture; the sweep will drive it.  ``fixed`` pins joints that may not move
    (e.g. during the frame check, joints whose sign is not yet known).

    Returns ``(posture, amplitude)``.
    """
    from scipy import optimize

    if rng is None:
        rng = np.random.default_rng(j)
    lo = arm.q_lower + lim.margin
    hi = arm.q_upper - lim.margin
    fixed = dict(fixed or {})
    fixed[j] = 0.0                             # joint j is the swept one

    free = [k for k in range(arm.nv) if k not in fixed]
    budget = lim.tau_frac * JOINT_TAU_CONT[: arm.nv]

    def build(x):
        q = np.zeros(arm.nv)
        for k, v in fixed.items():
            q[k] = np.clip(v, lo[k], hi[k])
        for idx, k in enumerate(free):
            q[k] = x[idx]
        return q

    def feasible(q) -> bool:
        if not respect_torque:
            return True
        return bool(np.all(sweep_peak_torque(arm, j, q, lim) <= budget))

    def neg_amp(x):
        q = build(x)
        amp = signal_amplitude(arm, j, q)
        if not respect_torque:
            return -amp
        peak = sweep_peak_torque(arm, j, q, lim)
        over = np.clip(peak / budget - 1.0, 0.0, None)
        return -amp + 50.0 * (1.0 + amp) * float(np.sum(over ** 2))

    if not free:
        # every other joint is pinned (e.g. during the first round of the frame
        # check, when no other joint's sign is known yet, so none may be moved).
        # There is nothing to optimise -- just report what we are stuck with.
        q = build(np.zeros(0))
        return q, signal_amplitude(arm, j, q)

    bounds = list(zip(lo[free], hi[free]))

    # ---- stage 1: find FEASIBLE postures by sampling ------------------------
    # A soft penalty is not good enough for a torque limit: gradient descent
    # happily settles 1-27% over budget because the signal gain outweighs the
    # penalty.  A torque budget is a constraint, not a preference.  So we search
    # among postures that are feasible BY CONSTRUCTION and only ever return one
    # we have re-verified.
    cands = []
    for _ in range(max(30 * n_restarts, 300)):
        q = build(rng.uniform(lo[free], hi[free]))
        if feasible(q):
            cands.append((signal_amplitude(arm, j, q), q))
    if not cands:
        raise InfeasiblePosture(
            f"joint {j+1}: no posture within {100*lim.tau_frac:.0f}% of the "
            f"continuous torque rating produces any gravity signal.\n"
            f"The arm (6.4 kg with the hand) cannot hold most postures inside "
            f"that budget. Either raise SafetyLimits.tau_frac (and accept the "
            f"thermal load), or accept that this joint stays at its CAD value.")
    cands.sort(key=lambda t: -t[0])

    # ---- stage 2: polish the best few, keeping only verified-feasible results
    best_q, best_amp = cands[0][1], cands[0][0]
    for _, q0 in cands[: max(3, n_restarts // 4)]:
        x0 = np.array([q0[k] for k in free])
        sol = optimize.minimize(neg_amp, x0, bounds=bounds, method="L-BFGS-B",
                                options={"maxiter": 120})
        q = build(sol.x)
        if feasible(q):                       # HARD re-check; polish may violate
            a = signal_amplitude(arm, j, q)
            if a > best_amp:
                best_q, best_amp = q, a
    return best_q, float(best_amp)


# --------------------------------------------------------------------------- #
#  Co-optimised (posture, arc) design -- the one the protocol actually uses    #
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class SweepDesign:
    """A torque-feasible excitation design for one joint."""
    joint: int
    posture: np.ndarray       # hold angles of the other joints
    center: float             # centre of joint j's swept arc (rad)
    span: float               # arc as a fraction of the joint's usable range
    arc_rad: float            # the actual arc width (rad)
    amplitude: float          # gravity swing of joint j at this posture (N*m)
    peak_torque: np.ndarray   # peak |tau| on EVERY joint over the arc
    torque_frac: float        # worst fraction of the torque BUDGET used (<= 1)
    cond: float               # conditioning of the 2-param sinusoid fit over the arc

    @property
    def snr_vs_friction(self) -> float:
        return self.amplitude / JOINT_COULOMB[self.joint]


def _best_arc(arm, j, lim, posture, budget, min_arc, respect_torque):
    """Widest torque-feasible arc for joint j at this posture (None if none)."""
    rlo = arm.q_lower[j] + lim.margin
    rhi = arm.q_upper[j] - lim.margin
    full = rhi - rlo
    if full <= 1e-6:
        return None
    min_span = min(1.0, max(min_arc / full, 0.05))

    for span in np.linspace(1.0, min_span, 8):
        best = None
        for center in np.linspace(rlo, rhi, 21):
            a, b = arc_bounds(arm, j, lim, center, span)
            peak = sweep_peak_torque(arm, j, posture, lim, center=center, span=span)
            frac = float(np.max(peak / budget)) if respect_torque else 0.0
            if respect_torque and frac > 1.0:
                continue
            if best is None or frac < best[1]:
                best = (center, frac, peak, (a, b), span)
        if best is not None:
            return best          # widest span that admits a feasible arc
    return None


def best_sweep(arm: ArmModel, j: int, lim: SafetyLimits, *,
               n_samples: int = 300,
               min_arc: float = np.deg2rad(90.0),
               rng: Optional[np.random.Generator] = None,
               fixed: Optional[dict] = None,
               respect_torque: bool = True) -> SweepDesign:
    """Co-optimise the holding posture AND the swept arc, under a HARD torque cap.

    Three coupled facts force this to be a *joint* search:

    * The gravity **amplitude** of joint j depends only on the *posture* of the
      other joints -- not on how far j swings.
    * The **torque load** depends on the posture AND on how far j swings, and it
      is the *holding* joints (the shoulder carrying 6.4 kg of arm + hand) that
      blow the budget, not the little wrist joint being swept.
    * The **conditioning** of the 2-parameter sinusoid fit depends only on the
      *arc width*: over a short arc cos and sin go collinear and (a, b) stop
      being separable no matter how clean the signal is.

    So the posture cannot be chosen before the arc.  We sample postures and, for
    each, take the widest arc that stays inside the torque budget subject to a
    floor of ``min_arc`` for conditioning.  A posture with no feasible arc is
    discarded.  Everything returned has been verified feasible -- a torque budget
    is a constraint, not a preference, and a soft penalty is not a constraint.
    """
    if rng is None:
        rng = np.random.default_rng(j)
    lo = arm.q_lower + lim.margin
    hi = arm.q_upper - lim.margin
    fixed = dict(fixed or {})
    fixed[j] = 0.0
    free = [k for k in range(arm.nv) if k not in fixed]
    budget = lim.tau_frac * JOINT_TAU_CONT[: arm.nv]

    def build(x):
        q = np.zeros(arm.nv)
        for k, v in fixed.items():
            q[k] = np.clip(v, lo[k], hi[k])
        for idx, k in enumerate(free):
            q[k] = x[idx]
        return q

    postures = ([build(np.zeros(0))] if not free else
                [build(rng.uniform(lo[free], hi[free])) for _ in range(n_samples)])

    best: Optional[SweepDesign] = None
    for q in postures:
        arc = _best_arc(arm, j, lim, q, budget, min_arc, respect_torque)
        if arc is None:
            continue
        center, frac, peak, (a, b), span = arc
        amp = signal_amplitude(arm, j, q)
        d = SweepDesign(joint=j, posture=q, center=float(center), span=float(span),
                        arc_rad=float(b - a), amplitude=float(amp),
                        peak_torque=peak, torque_frac=float(frac),
                        cond=sinusoid_conditioning(np.linspace(a, b, 12)))
        if best is None or d.amplitude > best.amplitude:
            best = d

    if best is None:
        raise InfeasiblePosture(
            f"joint {j+1}: NO posture + arc keeps every joint within "
            f"{100*lim.tau_frac:.0f}% of its continuous torque rating.\n"
            f"With the ORCA hand the arm weighs ~6.4 kg and the shoulder needs "
            f"~19 N*m -- 95% of the DM-J8009's continuous 20 N*m -- just to hold "
            f"it out. Raise SafetyLimits.tau_frac and accept the thermal load, or "
            f"calibrate the bare arm (tools/extract_arm_model.py --no-hand).")
    return best


# --------------------------------------------------------------------------- #
#  Rest-to-rest quintic segment (the only motion primitive we ever command)    #
# --------------------------------------------------------------------------- #
def _quintic_duration(delta: float, lim: SafetyLimits) -> float:
    """Shortest rest-to-rest quintic duration honouring BOTH vel and accel caps.

    For s(u) = 10u^3 - 15u^4 + 6u^5 (u = t/T):  max|s'| = 1.875, max|s''| = 5.7735.
    """
    d = abs(float(delta))
    if d < 1e-9:
        return 0.0
    t_vel = 1.875 * d / lim.qd_max
    t_acc = np.sqrt(5.7735 * d / lim.qdd_max)
    return float(max(t_vel, t_acc, 0.25))     # never shorter than 0.25 s


def _quintic(q0: float, q1: float, T: float, dt: float):
    """Sampled rest-to-rest quintic. Returns (t, q, qd, qdd) 1-D arrays."""
    if T <= 0:
        return np.zeros(1), np.array([q1]), np.zeros(1), np.zeros(1)
    t = np.arange(0.0, T + 1e-12, dt)
    u = np.clip(t / T, 0.0, 1.0)
    s = 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5
    sd = (30 * u ** 2 - 60 * u ** 3 + 30 * u ** 4) / T
    sdd = (60 * u - 180 * u ** 2 + 120 * u ** 3) / T ** 2
    d = q1 - q0
    return t, q0 + d * s, d * sd, d * sdd


# --------------------------------------------------------------------------- #
#  The sweep plan                                                              #
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class Hold:
    """One static measurement point: the arm is stationary at ``q`` for ``dwell``."""
    q: np.ndarray            # (nv,) full configuration
    joint: int               # which joint is under test
    angle: float             # q[joint]
    direction: int           # +1 if reached while increasing q_j, -1 if decreasing
    t_start: float
    t_end: float


@dataclasses.dataclass
class SweepPlan:
    """A complete, safety-checked, one-joint-at-a-time excitation."""
    joint: int
    posture: np.ndarray
    amplitude: float               # predicted gravity swing at this posture (N*m)
    t: np.ndarray                  # (T,)
    Q: np.ndarray                  # (T, nv)
    QD: np.ndarray                 # (T, nv)
    QDD: np.ndarray                # (T, nv)
    holds: List[Hold]
    limits: SafetyLimits

    @property
    def duration(self) -> float:
        return float(self.t[-1]) if self.t.size else 0.0

    def hold_mask(self, settle: Optional[float] = None) -> np.ndarray:
        """Boolean mask over ``t`` selecting the settled part of every dwell."""
        settle = self.limits.settle if settle is None else settle
        m = np.zeros(self.t.size, dtype=bool)
        for h in self.holds:
            m |= (self.t >= h.t_start + settle) & (self.t <= h.t_end)
        return m


def plan_from_design(arm: ArmModel, design: SweepDesign, lim: SafetyLimits, *,
                     n_angles: int = 12, dt: float = 0.01,
                     bidirectional: bool = True) -> SweepPlan:
    """Turn a torque-feasible :class:`SweepDesign` into an executable plan."""
    return plan_joint_sweep(arm, design.joint, design.posture, lim,
                            n_angles=n_angles, dt=dt,
                            bidirectional=bidirectional,
                            span=design.span, center=design.center)


def plan_joint_sweep(arm: ArmModel, j: int, posture: np.ndarray,
                     lim: SafetyLimits, *, n_angles: int = 12,
                     dt: float = 0.01, bidirectional: bool = True,
                     span: float = 1.0,
                     center: Optional[float] = None) -> SweepPlan:
    """Slow, dwelling, bidirectional sweep of joint ``j`` alone.

    The joint steps through ``n_angles`` angles spanning ``span`` x its usable
    range, pausing at each.  With ``bidirectional=True`` it then walks the same
    angles backwards, so every angle is visited once while increasing and once
    while decreasing.  Averaging the two torques cancels the Coulomb friction
    term (which flips sign with the direction of travel) and the *difference*
    measures it -- see :mod:`mabel_sysid.friction`.  Every other joint stays
    exactly at ``posture``.

    Direction-establishing back-off
    -------------------------------
    A hold's friction sign is set by *how the joint arrived*, so every hold must
    be preceded by real motion in the direction we are about to label it with.
    Two holds would otherwise be mislabelled:

    * the **first** hold of the up-pass (nothing moved before it -- friction sign
      undefined), and
    * the **turnaround** hold, which the naive plan visits twice in a row at the
      same angle *without the joint ever reversing*, so its "down" sample still
      carries up-friction.

    Left uncorrected these two bad samples bias the recovered Coulomb friction by
    a clean, deceptive factor: with K angles you measure 0.85 * tau_c at K=10 --
    a smooth 15% error that looks like a modelling result rather than a bug.  We
    therefore bracket each pass with a ``lim.backoff`` excursion (kept inside the
    joint-limit margin) that forces a genuine reversal before the pass begins.
    """
    nv = arm.nv
    lo = arm.q_lower[j] + lim.margin
    hi = arm.q_upper[j] - lim.margin
    a_lo, a_hi = arc_bounds(arm, j, lim, center, span)
    angles = np.linspace(a_lo, a_hi, n_angles)

    # back-off must stay inside the safety margin we already reserved: it may eat
    # into the unswept part of the range, or into the joint-limit margin, but it
    # must never approach the hard stop.
    room = min(a_lo - lo, hi - a_hi) + 0.5 * lim.margin
    back = float(np.clip(lim.backoff, 1e-3, max(room, 1e-3)))

    # (target, direction, measure?) -- the back-off points are transit only.
    steps: List[tuple] = []
    steps.append((angles[0] - back, +1, False))          # establish "increasing"
    for a in angles:
        steps.append((a, +1, True))
    if bidirectional:
        steps.append((angles[-1] + back, -1, False))     # establish "decreasing"
        for a in angles[::-1]:
            steps.append((a, -1, True))

    q_now = np.asarray(posture, dtype=float).copy()

    ts, Qs, QDs, QDDs, holds = [], [], [], [], []
    t_cursor = 0.0

    def emit(t_local, qj, qdj, qddj):
        nonlocal t_cursor
        for i in range(len(t_local)):
            q = q_now.copy(); q[j] = qj[i]
            qd = np.zeros(nv); qd[j] = qdj[i]
            qdd = np.zeros(nv); qdd[j] = qddj[i]
            ts.append(t_cursor + t_local[i]); Qs.append(q); QDs.append(qd); QDDs.append(qdd)
        t_cursor += float(t_local[-1]) + dt

    q_prev = float(steps[0][0])
    for i, (target, dirn, measure) in enumerate(steps):
        # ---- transit to this angle (rest-to-rest quintic) ----
        if i > 0:
            T = _quintic_duration(target - q_prev, lim)
            tl, qj, qdj, qddj = _quintic(q_prev, target, T, dt)
            emit(tl, qj, qdj, qddj)
        q_prev = float(target)
        if not measure:
            continue
        # ---- dwell (the actual measurement) ----
        n_hold = max(int(round(lim.dwell / dt)), 2)
        tl = np.arange(n_hold) * dt
        t0 = t_cursor
        emit(tl, np.full(n_hold, target), np.zeros(n_hold), np.zeros(n_hold))
        holds.append(Hold(q=np.where(np.arange(nv) == j, target, posture).astype(float),
                          joint=j, angle=float(target), direction=int(dirn),
                          t_start=t0, t_end=t0 + tl[-1]))

    t = np.asarray(ts)
    Q = np.asarray(Qs)
    QD = np.asarray(QDs)
    QDD = np.asarray(QDDs)
    lim.check(QD, QDD)

    return SweepPlan(joint=j, posture=np.asarray(posture, float),
                     amplitude=signal_amplitude(arm, j, posture),
                     t=t, Q=Q, QD=QD, QDD=QDD, holds=holds, limits=lim)


def plan_transit(arm: ArmModel, q_from: np.ndarray, q_to: np.ndarray,
                 lim: SafetyLimits, *, dt: float = 0.01,
                 order: Optional[Sequence[int]] = None):
    """Move between two configurations **one joint at a time**.

    Never moves two joints simultaneously, so at every instant exactly one link
    chain is in motion and you always know what is about to move.  Default order
    is distal -> proximal (wrist first, shoulder last), which keeps the large,
    high-inertia links stationary for as long as possible.
    """
    nv = arm.nv
    order = list(order) if order is not None else list(range(nv - 1, -1, -1))
    q = np.asarray(q_from, float).copy()
    q_to = np.asarray(q_to, float)

    ts, Qs, QDs, QDDs = [], [], [], []
    t_cursor = 0.0
    for j in order:
        d = q_to[j] - q[j]
        if abs(d) < 1e-6:
            continue
        T = _quintic_duration(d, lim)
        tl, qj, qdj, qddj = _quintic(q[j], q_to[j], T, dt)
        for i in range(len(tl)):
            qq = q.copy(); qq[j] = qj[i]
            qd = np.zeros(nv); qd[j] = qdj[i]
            qdd = np.zeros(nv); qdd[j] = qddj[i]
            ts.append(t_cursor + tl[i]); Qs.append(qq); QDs.append(qd); QDDs.append(qdd)
        t_cursor += float(tl[-1]) + dt
        q[j] = q_to[j]

    if not ts:
        return np.zeros(1), q_to[None, :], np.zeros((1, nv)), np.zeros((1, nv))
    t, Q, QD, QDD = map(np.asarray, (ts, Qs, QDs, QDDs))
    lim.check(QD, QDD)
    return t, Q, QD, QDD


# --------------------------------------------------------------------------- #
#  Torque-headroom check (run BEFORE touching hardware)                        #
# --------------------------------------------------------------------------- #
def check_torque_headroom(arm: ArmModel, plan: SweepPlan,
                          lim: SafetyLimits) -> dict:
    """Predict the torque the plan will demand and compare to the motor ratings.

    Uses full inverse dynamics (so the -- tiny -- inertial term is included, not
    just gravity).  Returns a dict of per-joint peak torque and the fraction of
    the motor's continuous rating it represents.  Raise-worthy if > ``tau_frac``.
    """
    TAU = arm.batch_torque(plan.Q, plan.QD, plan.QDD)
    peak = np.abs(TAU).max(axis=0)
    frac = peak / JOINT_TAU_CONT[: arm.nv]
    worst = int(np.argmax(frac))
    return {
        "peak_torque": peak,
        "frac_of_continuous": frac,
        "worst_joint": worst + 1,
        "worst_frac": float(frac[worst]),
        "ok": bool(frac.max() <= lim.tau_frac),
        "inertial_fraction": float(
            np.abs(TAU - arm.batch_torque(plan.Q)).max()
            / max(np.abs(TAU).max(), 1e-9)),
    }
