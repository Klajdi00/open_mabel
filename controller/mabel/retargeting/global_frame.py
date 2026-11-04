"""GLOBAL mode's body-frame estimator — eq. (13), heading integrated in F_a.

    dpsi_o = (1 - chi) dpsi_rig + chi psi_beta gamma(dpsi_d, psi_beta)

This is a thin ADAPTER, not a reimplementation. The shared-wrist-yaw fusion
(dpsi_rig), the arm-sweep ambiguity (chi) and the head-consistency score (gamma)
already exist, measured and validated, in
`mabel.whole_body_retargeting.body_frame_estimator.BodyFrameEstimator`:
bearing common-mode + wrist-orientation common-mode fused robustly, with the
anti-correlated parallel-sweep case gated on head corroboration so a two-arm
reach with a still head cannot command the base. Rewriting that would throw away
its validation for nothing (7/7 canonical motions within ~2 deg, no false drive);
this class only re-expresses its output in the package's BodyFrameEstimate /
RobotFrames vocabulary so the two modes meet at one interface.

The adapter is also the ONLY place that re-composes F_a poses from the device
frame (T_ah = T_ad T_dh). That keeps the S_G \\ S_L dependency in one readable
place instead of spread through the pipeline.
"""
from __future__ import annotations

import numpy as np

from ..whole_body_retargeting.body_frame_estimator import BodyFrameEstimator
from .frames import gravity_aligned_basis
from .types import BodyFrameEstimate


class GlobalBodyFrame:
    """Operator body frame in F_a, plus the retargeted base twist nu_b^ret."""

    def __init__(self, model, estimator=None):
        self.model = model
        self.est = estimator if estimator is not None else BodyFrameEstimator()
        self.reset()

    def reset(self):
        self.est.reset()
        self._basis = None
        self._p_ad = None
        self._fresh = False
        return self

    # ── measurement re-composition (the only S_G read) ────────────────────────
    @staticmethod
    def _world_wrists(T_ad, dev):
        """T_ah = T_ad T_dh for each tracked side."""
        out = {}
        for side in dev.sides():
            out[side] = np.asarray(T_ad, float) @ np.asarray(dev.wrists[side], float)
        return out

    def update(self, dev, world, dt):
        """Ingest one frame. `world` carries T_ad — the F_a-referenced part of
        S_G. Without it (or without both wrists) the estimator cannot run and we
        fall back to the head's own gravity-aligned heading, which is what the
        engine has always done before the first body-yaw sample."""
        self.model.observe(dev)
        T_ad = None if world is None else world.T_ad
        if T_ad is None:
            self._fresh = False
            return BodyFrameEstimate(valid=False, frame="anchor")

        T_ad = np.asarray(T_ad, float)
        self._p_ad = T_ad[:3, 3].copy()
        w = self._world_wrists(T_ad, dev)
        if "left" in w and "right" in w:
            if not self.est.ready:
                self.est.calibrate_facing(T_ad, w["left"], w["right"])
            elif not self._fresh:
                # Resuming after a dropout: refresh the geometry WITHOUT
                # integrating the whole gap as one increment.
                self.est.seed(T_ad, w["left"], w["right"])
            _, B = self.est.update(T_ad, w["left"], w["right"], dt)
            self._basis = np.asarray(B, float)
            self._fresh = True
        else:
            # One hand or none: hold the integrated heading (the estimator is
            # wrist-driven), and seed on resume so no gap is integrated.
            self._fresh = False
            if self._basis is None:
                self._basis = gravity_aligned_basis(T_ad[:3, :3])

        return self._estimate()

    def _estimate(self):
        if self._basis is None or self._p_ad is None:
            return BodyFrameEstimate(valid=False, frame="anchor")
        B = self._basis
        # SAME root construction as the local fit, so eqs. (16)-(18) receive
        # structurally identical inputs in both modes: the torso origin is the
        # mid-shoulder root, and the device sits on it at p_d^o.
        root = self._p_ad - B @ self.model.device_body()
        shoulders = {side: root + B @ self.model.shoulder_body(side)
                     for side in ("left", "right")}
        conf = float(self.est.diag.get("conf", 0.0)) if self.est.diag else 0.0
        return BodyFrameEstimate(basis=B.copy(), shoulders=shoulders, root=root,
                                 frame="anchor", device_origin=self._p_ad.copy(),
                                 confidence=conf, valid=True,
                                 diag=dict(self.est.diag or {}))

    # ── queries ───────────────────────────────────────────────────────────────
    def base_twist(self):
        """nu_b^ret = (v_forward, v_left, omega) in the operator body frame —
        the RETARGETED base channel of eq. (15). Zero until the estimator is
        ready, and already deadbanded by the estimator."""
        if not self.est.ready:
            return (0.0, 0.0, 0.0)
        return tuple(float(v) for v in self.est.base_twist())

    def seed_heading(self, dev, world):
        """Refresh the estimator's previous-frame geometry without disturbing
        the integrated heading — used when re-entering global mode so the first
        update does not integrate the whole local-mode interval as one step."""
        T_ad = None if world is None else world.T_ad
        if T_ad is None:
            return self
        w = self._world_wrists(np.asarray(T_ad, float), dev)
        if "left" in w and "right" in w and self.est.ready:
            self.est.seed(np.asarray(T_ad, float), w["left"], w["right"])
            self._fresh = True
        return self
