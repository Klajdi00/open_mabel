"""
mabel.whole_body_controller — the output side: targets -> safe joint commands.

Everything that turns desired motion (retargeted poses, app joint targets, an
operator's physical push) into per-actuator commands, with safety layered in.
Organised by control law; each subpackage is independently usable and they
compose through the runtime engine:

    ik                    WholeBodyIK — bimanual SE(3) differential IK over the
                          18-DOF upper chain (weighted DLS + null-space posture).
    position_controller   JointPositionController — plain PD, Day-1 hardware
                          (works before system ID; add gravity_ff for Day-2).
    gravity_compensation  GravityCompController — model feedforward that holds
                          any region against gravity (back-drivable float).
    collision_avoidance   SelfCollisionGuard — capsule CBF/velocity-damper
                          guard; the ONLY arm-arm defense (the MJCF disables
                          arm-arm contacts). ON by default.
    admittance            external-force sensing & the base's response:
                          wrench_estimation (momentum observer -> wrench ->
                          WHERE/how-hard/which-direction contact localization)
                          + BaseAdmittance (virtual planar mass-damper glide).
    compliance            ComplianceController — gravity-comp'd, back-drivable
                          region the operator can move by hand ("dance with
                          the robot"); pick arms / upper_body / whole_body.
    base_controller       swerve_ik — body twist -> 3-module delta-swerve
                          steering + wheel commands. NEVER call it on a raw
                          twist: route through motion_model.safe_base_step.
    task_space            BaseFollow — wrist/task-space whole-body reach: the
                          base follows the palm TARGETS (LazyBase comfort box,
                          stick override, safe_base_step inside) — the ported
                          sim-teleop whole-body cascade for dragged targets.
    engine                WBCEngine — the runtime that the facade drives:
                          impedance ("Hold") / compliance ("Soft") x region,
                          bumpless entry/exit, servo neutralisation, base
                          admittance in whole-body compliance.
"""
from .ik import WholeBodyIK                                        # noqa: F401
from .engine import WBCEngine, CONTROL_TYPES, CONTROL_REGIONS      # noqa: F401
from .position_controller import JointPositionController          # noqa: F401
from .gravity_compensation import GravityCompController           # noqa: F401
from .collision_avoidance import SelfCollisionGuard, solve_qp     # noqa: F401
from .compliance import ComplianceController                      # noqa: F401
from .base_controller import swerve_ik                            # noqa: F401
from .task_space import BaseFollow                                # noqa: F401

__all__ = ["WholeBodyIK", "WBCEngine", "CONTROL_TYPES", "CONTROL_REGIONS",
           "JointPositionController", "GravityCompController",
           "SelfCollisionGuard", "solve_qp", "ComplianceController",
           "swerve_ik", "BaseFollow"]
