"""Mode arbitration — eq. (15). The ENTIRE mode-dependent surface lives here.

    R_bo    = R_ba Rhat_ao   (m = G)   |   R_b* Rhat_do   (m = L)
    nu_b^cmd = nu_b^ret      (m = G)   |   0              (m = L)

Everything downstream (eqs. 16-21: wrist targets, swivel, gaze, fingers) reads
only R_bo and is therefore mode-invariant by construction. Adding a third mode
later means adding an estimator and a row here, nothing else.

THE ALIGNMENT LATCH, DERIVED
----------------------------
The two estimators express the operator's body basis in DIFFERENT frames -- B_G
in F_a, B_L in F_d -- so switching naively would jump the robot's targets by
whatever the two disagree about. Write each mode's map from its source frame
into operator-body coordinates, both referred to F_a for comparison (a physical
vector satisfies v_d = R_ad^T v_a):

    P_G = align_G B_G^T                 P_L = align_L B_L^T R_ad^T

Continuity of the TARGETS is P_new == P_old at the switch instant, which gives a
closed form for the new mode's constant (B orthonormal, so (B^T)^-1 = B):

    G -> L:   align_L = P_G R_ad B_L
    L -> G:   align_G = P_L B_G

so the handoff is EXACT, not merely smooth -- the paper's "R_b*, a constant
alignment fixed at mode entry". Note the latch reads T_ad ONCE, at the switch,
purely to relate two bases; local mode never touches it again. If the operator
is switching to local precisely because F_a has gone bad, that corruption enters
only as a fixed offset in the mapping's zero -- which is exactly the semantics
an operator expects from "re-anchor here", and strictly better than a jump.

RAMPS
-----
The base is "ramped rather than cut": a cross-fade from the pre-switch value to
the live target over ~3 tau, after which the output is the target EXACTLY -- a
permanent low-pass would tax steady-state driving with lag for no reason. The
same cross-fade carries the lift, because (by product decision) global keeps the
head-height lift law while local uses the eq. (20) comfort region, so the two
laws must be blended across a switch rather than stepped.
"""
from __future__ import annotations

import numpy as np

RAMP_TAU = 0.4          # s; base + lift cross-fade time constant on a switch
RAMP_SPAN = 3.0         # cross-fade is considered finished after RAMP_SPAN*tau


class RetargetMode:
    """The two modes of eq. (12), as the strings the wire carries."""
    LOCAL = "local"
    GLOBAL = "global"
    ALL = (LOCAL, GLOBAL)

    @staticmethod
    def normalize(value, default=LOCAL):
        """Wire value -> a valid mode. Unknown/None falls back to `default`
        (local), so a stale or malformed client can never silently obtain base
        authority it did not ask for."""
        v = str(value or "").strip().lower()
        return v if v in RetargetMode.ALL else default


class _Crossfade:
    """Blend from a held value to a live target over ~RAMP_SPAN*tau, then pass
    the target straight through."""

    def __init__(self, tau=RAMP_TAU):
        self.tau = float(tau)
        self.held = None
        self.t = float("inf")      # time since the fade started

    def start(self, held):
        self.held = np.asarray(held, float).copy()
        self.t = 0.0

    @property
    def active(self):
        return self.held is not None and self.t < RAMP_SPAN * self.tau

    def __call__(self, target, dt):
        target = np.asarray(target, float)
        if not self.active:
            self.held = None
            return target
        self.t += max(float(dt), 0.0)
        g = 1.0 - float(np.exp(-self.t / max(self.tau, 1e-6)))
        return g * target + (1.0 - g) * self.held


class ModeArbiter:
    """Holds the active mode, the alignment latch and the switch ramps."""

    def __init__(self, mode=RetargetMode.LOCAL, tau=RAMP_TAU):
        self.mode = RetargetMode.normalize(mode)
        self.tau = float(tau)
        self._align = {RetargetMode.GLOBAL: np.eye(3),
                       RetargetMode.LOCAL: np.eye(3)}
        self._base_fade = _Crossfade(tau)
        self._twist = np.zeros(3)
        self._blend_t = float("inf")   # time since the last switch
        self.exact_latch = True        # was the last handoff computed exactly?
        self.pending = None            # requested mode, applied on the next update
        self.switches = 0

    # ── mode requests ─────────────────────────────────────────────────────────
    def request(self, mode):
        """Ask for a mode. The switch is applied inside `update`, where both
        estimates for THIS frame are available to compute the latch — a switch
        applied between frames would have to use stale bases."""
        m = RetargetMode.normalize(mode, default=self.mode)
        if m != self.mode:
            self.pending = m
        return self

    def _latch(self, new_mode, est_g, est_l, R_ad):
        """Set the new mode's alignment so the map is continuous (see docstring).
        Falls back to leaving the constant untouched when the inputs needed for
        an exact latch are missing — a small step, ramped, beats a wrong latch."""
        old = self.mode
        B_old = (est_g.basis if old == RetargetMode.GLOBAL else est_l.basis)
        B_new = (est_g.basis if new_mode == RetargetMode.GLOBAL else est_l.basis)
        if B_old is None or B_new is None or R_ad is None:
            return False
        R_ad = np.asarray(R_ad, float)
        if new_mode == RetargetMode.LOCAL:
            # P_G = align_G B_G^T  ->  align_L = P_G R_ad B_L
            P_old = self._align[RetargetMode.GLOBAL] @ B_old.T
            self._align[RetargetMode.LOCAL] = P_old @ R_ad @ B_new
        else:
            # P_L = align_L B_L^T R_ad^T  ->  align_G = P_L B_G
            P_old = self._align[RetargetMode.LOCAL] @ B_old.T @ R_ad.T
            self._align[RetargetMode.GLOBAL] = P_old @ B_new
        return True

    # ── per-frame ─────────────────────────────────────────────────────────────
    def update(self, est_g, est_l, R_ad, nu_ret, dt):
        """Apply any pending switch, then produce this frame's mode-dependent
        quantities.

        Returns (mode, align, base_twist, blend, switched).
          align       the 3x3 constant for the ACTIVE mode; compose as
                      R_bo = R_rob @ align @ B_active.T
          base_twist  nu_b^cmd — the RETARGET channel only. An explicit nav
                      joystick is a separate, deliberate input and is never
                      zeroed by local mode.
          blend       0 -> 1 authority of the ACTIVE mode's body-DoF law after a
                      switch. The caller cross-fades ITS OWN two lift/torso laws
                      with it (`blend * active + (1 - blend) * previous`), so
                      the arbiter never has to know either law — global keeps
                      its head-height lift, local uses the eq. (20) comfort
                      region, and the change of law is a fade, not a step.
        """
        switched = False
        if self.pending is not None and self.pending != self.mode:
            self.exact_latch = self._latch(self.pending, est_g, est_l, R_ad)
            self._base_fade.start(self._twist)
            self.mode = self.pending
            self.pending = None
            self.switches += 1
            self._blend_t = 0.0
            switched = True

        target_twist = (np.asarray(nu_ret, float)
                        if self.mode == RetargetMode.GLOBAL else np.zeros(3))
        self._twist = self._base_fade(target_twist, dt)

        self._blend_t += max(float(dt), 0.0)
        blend = (1.0 if self._blend_t >= RAMP_SPAN * self.tau
                 else 1.0 - float(np.exp(-self._blend_t / max(self.tau, 1e-6))))

        return (self.mode, self._align[self.mode].copy(),
                tuple(float(v) for v in self._twist), float(blend), switched)

    # ── queries ───────────────────────────────────────────────────────────────
    @property
    def ramping(self):
        return self._base_fade.active or self._blend_t < RAMP_SPAN * self.tau

    def R_bo(self, robot_basis, body_basis):
        """R_bo = R_rob @ align @ B^T — the ONE rotation everything downstream
        consumes. Composing with the LIVE chassis basis is what makes the map
        body-relative: the operator's heading and the robot's chassis heading
        both cancel, so the same body-relative reach lands the same
        body-relative target however either of them is turned."""
        if robot_basis is None or body_basis is None:
            return None
        return (np.asarray(robot_basis, float)
                @ self._align[self.mode] @ np.asarray(body_basis, float).T)
