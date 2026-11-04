"""hand_vector_retarget.py — AnyTeleop-style VECTOR retargeting for the ORCA hands.

An alternative to the keypoint-IK in :mod:`hand_ik_retarget` (DexPilot-style absolute
keypoint matching), provided for a head-to-head comparison in the calibration
experiments (``config.FINGER_RETARGET``). Where the keypoint solver matches absolute
canonical keypoint *positions*, this solver matches relative **keyvectors** -- the
recipe of the ``dex-retargeting`` library / AnyTeleop (Qin et al., RSS 2023,
arXiv:2307.04577, Eq. 1)::

    min_q  sum_i || alpha * v_i_human  -  f_i(q) ||^2  +  beta * || q - q_prev ||^2
           s.t.  q_lo <= q <= q_hi

where ``v_i`` are human keyvectors (per-finger knuckle->tip "flexion" vectors and
thumb->fingertip "pinch" vectors), ``f_i(q)`` the corresponding robot keyvectors via
FK, and ``alpha`` the per-operator hand-size scale. Matching *vectors* rather than
absolute points makes the mapping intrinsically scale-invariant across hand sizes --
the complementary philosophy to feeding the keypoint solver calibrated lengths.

Implementation: subclasses :class:`HandIKRetargeter` to reuse all of its model
indexing, analytic FK (``_HandFK``), canonical keypoints + Jacobians (``_kp_and_jac``),
background worker, and PD streaming -- only the Gauss-Newton residual is replaced, so
the two paths are identical everywhere except the objective (a clean A/B).
"""
import numpy as np

from .. import config as C
from .keypoint_ik import HandIKRetargeter, _FINGERS

# Keyvector weights (mirror the keypoint solver's shape/pinch/smooth/reg balance so the
# comparison isolates the objective form, not the tuning).
WV_FLEX = 30.0          # per-finger knuckle->tip extension vectors (shape)
WV_PINCH_LOW = 4.0      # thumb->fingertip vectors, hand open
WV_PINCH_HIGH = 50.0    # … boosted when the human pinches (DexPilot switch)
WV_PINCH_THRESH = 0.55  # normalized thumb-tip distance counting as a pinch
WV_SMOOTH = 1.0
WV_REG = 0.05
WV_ITERS = 4
WV_DAMPING = 1e-2


class HandVectorRetargeter(HandIKRetargeter):
    """Vector-optimization variant of the per-finger ORCA retargeter."""

    def set_profile(self, profile):
        """Store the per-operator hand-size scale alpha (palm-length ratio) used to
        rescale the human keyvectors. Falls back to alpha=1 (canonical space is
        already scale-normalised) when no profile / palm length is available."""
        super().set_profile(profile)
        self._alpha = {}
        if profile is None:
            return
        for side, key in (("Left", "left"), ("Right", "right")):
            # In the canonical (palm-length-normalised) frame both hands are already
            # unit-palm, so alpha defaults to 1; a profile may still nudge it if the
            # operator's measured proportions warrant (kept explicit for the sweep).
            if key in profile.palm_len_m:
                self._alpha[side] = 1.0

    def _alpha_for(self, hand):
        side = "Left" if hand.get("pre") == "left" else "Right"
        return getattr(self, "_alpha", {}).get(side, 1.0)

    def _solve_gn(self, hand, h_shape, h_pinch):
        """Damped Gauss-Newton on the stacked weighted keyVECTOR residuals
        (flexion + pinch) + smoothness + regularization. Reuses the parent's
        analytic canonical keypoints/Jacobians; only the residual rows differ.

        SIGN: residuals are ROBOT minus HUMAN, matching the parent's convention
        (`p - h`, `u_r - u_h`), because `rows_J` holds d(robot)/dq. Writing them
        human-minus-robot against that Jacobian makes `dq = -H^-1 J^T r` ASCEND
        the objective -- the hand then moves opposite the operator's, which
        measures as a NEGATIVE aperture correlation rather than as a crash.
        (Fixed 2026-08-26; this solver is a comparison baseline and is not in
        the live engine's path, so no shipped behaviour changed.)
        """
        q = hand["last_q"].copy()
        q_prev = hand["last_q"]
        n = len(q)
        alpha = self._alpha_for(hand)
        sw_smooth = np.sqrt(WV_SMOOTH)
        sw_reg = np.sqrt(WV_REG)

        def vec(d, a, b):
            """Keyvector b-a from a keypoint dict, or None if either is missing."""
            pa, pb = d.get(a), d.get(b)
            return None if pa is None or pb is None else (pb - pa)

        for _ in range(WV_ITERS):
            kp, jac = self._kp_and_jac(hand, q)
            rows_r, rows_J = [], []

            # Flexion vectors: knuckle -> tip per finger (+ thumb).
            for f in _FINGERS + ("thumb",):
                kn, tip = (f, "kn"), (f, "tip")
                v_h = vec(h_shape, kn, tip)
                v_r = vec(kp, kn, tip)
                if v_h is None or v_r is None:
                    continue
                w = np.sqrt(WV_FLEX)
                rows_r.append(w * (v_r - alpha * v_h))
                rows_J.append(w * (jac[tip] - jac[kn]))

            # Pinch vectors: thumb tip -> each fingertip, boosted when pinching.
            h_thumb = h_pinch.get(("thumb", "tip"))
            r_thumb_J = jac.get(("thumb", "tip"))
            if h_thumb is not None and r_thumb_J is not None:
                for f in _FINGERS:
                    v_h = vec(h_pinch, ("thumb", "tip"), (f, "tip"))
                    v_r = vec(kp, ("thumb", "tip"), (f, "tip"))
                    if v_h is None or v_r is None:
                        continue
                    w = np.sqrt(WV_PINCH_HIGH if np.linalg.norm(v_h) < WV_PINCH_THRESH
                                else WV_PINCH_LOW)
                    rows_r.append(w * (v_r - alpha * v_h))
                    rows_J.append(w * (jac[(f, "tip")] - jac[("thumb", "tip")]))

            rows_r.append(sw_smooth * (q - q_prev))
            rows_J.append(sw_smooth * np.eye(n))
            rows_r.append(sw_reg * q)
            rows_J.append(sw_reg * np.eye(n))

            r = np.concatenate([np.atleast_1d(x).ravel() for x in rows_r])
            J = np.vstack([x if x.ndim == 2 else x[None, :] for x in rows_J])
            H = J.T @ J + WV_DAMPING * np.eye(n)
            dq = np.linalg.solve(H, -(J.T @ r))
            q = np.clip(q + dq, hand["lo"], hand["hi"])
        return q


def make_hand_retargeter(model, data):
    """Factory: build the per-finger retargeter selected by config.FINGER_RETARGET
    ('keypoint' -> HandIKRetargeter, 'vector' -> HandVectorRetargeter)."""
    if C.FINGER_RETARGET == "vector":
        return HandVectorRetargeter(model, data)
    return HandIKRetargeter(model, data, solver=C.HAND_SOLVER)
