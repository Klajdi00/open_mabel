"""
Whole-body dynamic model — upper body + the mobile base.

The base contributes no torque-controlled joints: it is force-controlled
through the admittance law. This model therefore exposes the upper-body joint
feedforward PLUS `base_wrench(tau_cmd, dt)`, the net external planar wrench
(fx, fy, Mz in the chassis frame) estimated from joint-torque sensing and
reduced from the LOCALIZED contact points — the input the base admittance
integrates so the robot glides when pushed anywhere on its arms.
"""
from ..regions import RegionDynamicModel

__all__ = ["WholeBodyDynamicModel"]


class WholeBodyDynamicModel(RegionDynamicModel):
    """Upper body + mobile base. Adds the estimated external base wrench that
    drives the admittance (whole_body_controller/admittance)."""
    region = "whole_body"

    def __init__(self, model, data, observer_gain=35.0, tau_thresh=0.6, lam=3e-2):
        super().__init__(model, data, observer_gain=observer_gain)
        # Contact localization + base-wrench reduction (imported at call time:
        # the admittance package imports dynamic_models, not the other way
        # around at module import time).
        from ...whole_body_controller.admittance.wrench_estimation import ContactEstimator
        self._contact_est = ContactEstimator(model, data, gain=observer_gain,
                                             lam=lam, tau_thresh=tau_thresh)

    @property
    def contacts(self):
        """Localized contacts from the last base_wrench() call: a list of
        {body, joint, point[3], force[3]} dicts (world frame)."""
        return self._contact_est.contacts

    def base_wrench(self, tau_cmd, dt):
        """Estimated net external planar wrench (fx, fy, Mz) on the chassis,
        from joint-torque sensing via the momentum observer + per-arm contact
        localization. Feed it to BaseAdmittance.step(...) and route the result
        through motion_model.safe_base_step before swerve_ik."""
        return self._contact_est.update(tau_cmd, dt)
