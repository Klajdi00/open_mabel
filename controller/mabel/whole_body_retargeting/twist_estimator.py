"""Torso frame from head + two wrists by a WEIGHTED RIGID PLANAR TWIST FIT.

This is the estimator of record. It supersedes `body_frame_estimator.py`
(wrist-bearing consensus): that one attributed torso yaw to the two wrists'
agreeing bearing change, which is a special case this fit subsumes, and it gives
a FALSE positive when both arms swing the same way (both bearings rotate
together though the body is still). This estimator resolves that -- and all five
canonical cases -- with one linear solve and its residual.

THE ASYMMETRY THAT MAKES IT WORK
--------------------------------
There is a neck (one rotational DOF that lets head ORIENTATION decouple from the
torso) but no "translational neck": moving your arms never moves your head. So

  * for TRANSLATION the head is ground truth -- if the head moved, the body
    moved (arms cannot move it); if the head is still, wrist motion is reach;
  * for ROTATION the head is NOT trustworthy alone -- the neck lets it yaw
    freely -- so base yaw must be corroborated by the wrists orbiting.

THE FIT
-------
Per frame we have three world points: head p_h, wrists p_wL, p_wR (horizontal).
About the body centre c (spine, a fixed offset behind/below the head) fit the
planar base twist xi = (vx, vy, omega) that best explains their deltas:

    xi* = argmin_xi  sum_i  w_i || dp_i - ( v + omega * z_hat x (p_i - c) ) ||^2

Each point gives two linear rows (a_i = p_i - c):
    dp_{i,x} = vx - omega a_{i,y}
    dp_{i,y} = vy + omega a_{i,x}
so xi* is a 3x3 weighted least squares -- closed form, stream rate.

WEIGHT THE HEAD HEAVILY. Two things then happen automatically from the geometry:
the head sits near c (small radius |a_h|), so it barely constrains omega but,
being heavily weighted, dominates v -> v* tracks the BODY (an arm moving with the
head still cannot pull v* off zero); the wrists have large radius, so they drive
omega -> base yaw comes from the ORBIT, not from head yaw.

THE RESIDUAL IS THE ARTICULATION
--------------------------------
    r_i = dp_i - ( v* + omega* z_hat x a_i )
The head-orientation residual (head yaw - omega* dt) is the NECK. The wrist
residuals are the ARMS, in the moving body frame. And the wrist residual
MAGNITUDE is the discriminator: a coherent whole-body motion fits (residual ~0),
an arm-dominated motion does not (residual large) because its instantaneous
centre of rotation is not c. That single number places the ICR automatically --
no explicit case classification.

GATES (turn xi* into a base command)
------------------------------------
    coherence   alpha = exp(-(rho/sigma)^2),  rho = normalised wrist residual
    deadband + hysteresis on v*, omega*
    (a workspace/ROM gate belongs in the allocation layer, not here.)
The frame this class exposes is `alpha`-gated so a body turn moves the base while
an arm reach does not; `sigma` is the one real knob (the R3a/R3b boundary).

numpy only -- no simulator, no learned model -- runs on the real headset stream.
Validated against synthetic ground truth (incl. the both-arms-parallel case) and
head-on against `body_frame_estimator` in `exp_body_frame.py`.
"""
from __future__ import annotations

import numpy as np

from .body_frame_estimator import (UP, _flatten, _wrap, _REF_FWD, head_yaw,
                                    _rot_about_up, _yaw_about_up)


# ── One-Euro filter (Casiez, Roussel & Vogel, CHI 2012) ──────────────────────
class OneEuro:
    """Adaptive low-pass: heavy smoothing when still, light when moving fast, so
    positions are clean without lagging a real motion. AVP poses are accurate but
    their frame-to-frame DELTAS are noisy, and the twist fit differentiates them,
    so filter first or the residual flickers."""

    def __init__(self, min_cutoff=1.2, beta=0.03, d_cutoff=1.0, shape=3):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x = None
        self._dx = np.zeros(shape)

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, dt):
        x = np.asarray(x, float)
        if self._x is None:
            self._x = x.copy()
            return self._x
        a_d = self._alpha(self.d_cutoff, dt)
        dx = (x - self._x) / dt
        self._dx = a_d * dx + (1 - a_d) * self._dx
        cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(self._dx))
        a = self._alpha(cutoff, dt)
        self._x = a * x + (1 - a) * self._x
        return self._x.copy()


class TwistFitEstimator:
    """Weighted rigid planar twist fit over head + two wrists.

    Parameters
    ----------
    head_weight : w_h in the fit. Large -> the head owns the translation (arms
        cannot move it) while its small radius keeps it out of the yaw estimate.
    spine_back, spine_down : body centre c = head + this offset in the body frame
        (metres). c is the rotation centre; putting it at the spine (behind/below
        the head) is what makes an in-place turn give v* ~ 0 instead of a false
        base translation. Calibration should set these per operator.
    sigma : coherence-gate scale on the NORMALISED wrist residual (dimensionless,
        ~fraction of wrist motion that is non-rigid). THE tuning knob: the
        arm-vs-body decision boundary.
    lp_hz, yaw_deadband, vel_deadband : output smoothing + deadbands.
    """

    def __init__(self, head_weight=6.0, spine_back=0.10, spine_down=0.22,
                 sigma=0.45, lp_hz=2.5, yaw_deadband=0.02, vel_deadband=0.03,
                 filter_input=True):
        self.head_weight = float(head_weight)
        self.spine_back = float(spine_back)
        self.spine_down = float(spine_down)
        self.sigma = float(sigma)
        self.lp_hz = float(lp_hz)
        self.yaw_deadband = float(yaw_deadband)
        self.vel_deadband = float(vel_deadband)
        self.filter_input = bool(filter_input)
        self.reset()

    def reset(self):
        self.ready = False
        self.psi = 0.0                 # torso yaw (absolute, wrapped)
        self.psi_rate = 0.0
        self.vel = np.zeros(2)         # base horizontal velocity (world)
        self.origin = np.zeros(3)      # body centre (world)
        self.alpha = 0.0               # last coherence gate
        self._prev = None
        self._psi0 = 0.0
        self._filt = {"h": OneEuro(), "L": OneEuro(), "R": OneEuro()}
        self.diag = {}
        return self

    # ── geometry ─────────────────────────────────────────────────────────────
    def _center(self, head_pos, psi):
        fwd = _rot_about_up(psi) @ _REF_FWD
        right = np.cross(fwd, UP); right /= (np.linalg.norm(right) + 1e-12)
        B = np.column_stack([right, UP, fwd])
        return np.asarray(head_pos, float) + B @ np.array(
            [0.0, -self.spine_down, -self.spine_back])

    def _pts(self, head, wl, wr, dt):
        hp = np.asarray(head, float)[:3, 3]
        lp = np.asarray(wl, float)[:3, 3]
        rp = np.asarray(wr, float)[:3, 3]
        if self.filter_input:
            hp = self._filt["h"](hp, dt)
            lp = self._filt["L"](lp, dt)
            rp = self._filt["R"](rp, dt)
        return hp, lp, rp

    # ── the fit ──────────────────────────────────────────────────────────────
    def update(self, head, wl, wr, dt):
        """Ingest one frame. Returns the torso frame `(origin, B)` -- same
        contract as `calibration.body_frame`."""
        dt = max(float(dt), 1e-3)
        hp, lp, rp = self._pts(head, wl, wr, dt)
        hyaw = head_yaw(np.asarray(head, float)[:3, :3])
        RwL = np.asarray(wl, float)[:3, :3]
        RwR = np.asarray(wr, float)[:3, :3]

        if not self.ready:
            self._prev = dict(hp=hp, lp=lp, rp=rp, hyaw=hyaw, RwL=RwL, RwR=RwR)
            self.ready = True
            self.origin = self._center(hp, self.psi)
            self.diag = dict(alpha=0.0, omega=0.0, speed=0.0, resid=0.0)
            return self.frame()

        p = self._prev
        c = self._center(hp, self.psi)                 # rotation centre (spine)
        pts = [("h", hp, p["hp"], self.head_weight),
               ("L", lp, p["lp"], 1.0),
               ("R", rp, p["rp"], 1.0)]

        # weighted planar twist least squares: A xi = b, xi = [vx, vy, omega]
        AtA = np.zeros((3, 3)); Atb = np.zeros(3)
        for _, cur, prev, w in pts:
            a = _flatten(0.5 * (cur + prev) - c)       # lever, horizontal
            dp = _flatten(cur - prev)                  # world delta, horizontal
            e = _horiz_basis()
            ax, ay = float(a @ e[0]), float(a @ e[1])
            dx, dy = float(dp @ e[0]), float(dp @ e[1])
            r1 = np.array([1.0, 0.0, -ay])             # dp_x = vx - omega ay
            r2 = np.array([0.0, 1.0,  ax])             # dp_y = vy + omega ax
            AtA += w * (np.outer(r1, r1) + np.outer(r2, r2))
            Atb += w * (r1 * dx + r2 * dy)
        xi = np.linalg.solve(AtA + 1e-9 * np.eye(3), Atb)
        vx, vy, omega = float(xi[0]), float(xi[1]), float(xi[2])

        # residual on the WRISTS (the head defines v, so its residual ~0) and its
        # normalisation by wrist displacement -> a scale-free "how non-rigid"
        e = _horiz_basis()
        res_sq = disp_sq = 0.0
        for key, cur, prev in (("L", lp, p["lp"]), ("R", rp, p["rp"])):
            a = _flatten(0.5 * (cur + prev) - c)
            pred = np.array([vx, vy]) + omega * np.array([-(a @ e[1]), (a @ e[0])])
            dp = _flatten(cur - prev)
            dxy = np.array([dp @ e[0], dp @ e[1]])
            res_sq += float(np.sum((dxy - pred) ** 2))
            disp_sq += float(np.sum(dxy ** 2))
        rho = np.sqrt(res_sq) / (np.sqrt(disp_sq) + 1e-4)     # normalised residual
        alpha = float(np.exp(-(rho / self.sigma) ** 2))       # coherence gate
        self.alpha = alpha

        # gate + low-pass the base twist (v in the horizontal e0/e1 basis, m/s)
        rate = alpha * omega / dt
        vel_g = alpha * np.array([vx, vy]) / dt
        a_lp = dt * 2 * np.pi * self.lp_hz; a_lp /= (1 + a_lp)
        self.psi_rate = (1 - a_lp) * self.psi_rate + a_lp * rate
        self.vel = (1 - a_lp) * self.vel + a_lp * vel_g
        eff_w = 0.0 if abs(self.psi_rate) < self.yaw_deadband else self.psi_rate
        self.psi = _wrap(self.psi + eff_w * dt)
        # the body centre follows the head (the translation ground truth); the
        # gated `vel` above is the base-translation COMMAND (base_twist()).
        self.origin = self._center(hp, self.psi)

        self._prev = dict(hp=hp, lp=lp, rp=rp, hyaw=hyaw, RwL=RwL, RwR=RwR)
        self.diag = dict(alpha=alpha, omega=omega,
                         speed=float(np.linalg.norm(self.vel)),
                         resid=rho, rate=self.psi_rate)
        return self.frame()

    # ── outputs (same surface as BodyFrameEstimator) ─────────────────────────
    def frame(self):
        R = _rot_about_up(self.psi)
        fwd = R @ _REF_FWD
        right = np.cross(fwd, UP); right /= (np.linalg.norm(right) + 1e-12)
        B = np.column_stack([right, UP, fwd])
        return self.origin.copy(), B

    def torso_yaw(self):
        return _wrap(self.psi - self._psi0)

    def neck_yaw(self, head_R):
        return _wrap(head_yaw(head_R) - self.psi)

    def base_twist(self):
        """(vx, vy, omega) world-frame base command, alpha-gated + dead-banded."""
        w = 0.0 if abs(self.psi_rate) < self.yaw_deadband else self.psi_rate
        v = self.vel.copy()
        if np.linalg.norm(v) < self.vel_deadband:
            v[:] = 0.0
        return float(v[0]), float(v[1]), float(w)

    def calibrate_facing(self, head, wl, wr):
        hp = np.asarray(head, float)[:3, 3]
        self.psi = head_yaw(np.asarray(head, float)[:3, :3])
        self._psi0 = self.psi
        self.origin = self._center(hp, self.psi)
        self._prev = dict(hp=hp,
                          lp=np.asarray(wl, float)[:3, 3],
                          rp=np.asarray(wr, float)[:3, 3],
                          hyaw=self.psi,
                          RwL=np.asarray(wl, float)[:3, :3],
                          RwR=np.asarray(wr, float)[:3, :3])
        for f in self._filt.values():
            f._x = None
        self.ready = True
        return self


def _horiz_basis():
    """Two orthonormal horizontal axes (e0=_REF_FWD, e1=up x e0)."""
    e0 = _REF_FWD
    e1 = np.cross(UP, e0); e1 /= (np.linalg.norm(e1) + 1e-12)
    return np.array([e0, e1])
