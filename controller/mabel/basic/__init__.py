"""A minimal, measurable controller: pure PD + task-space wrist tracking.

Built as a clean baseline after the layered stack became hard to reason about:
every stage here is separately observable, and the first version deliberately
has NO gravity compensation so its droop can be measured rather than assumed.

    from mabel.basic import TaskSpaceArm
    arm = TaskSpaceArm(model, data, side="right").attach()
    arm.set_wrist_target([0.45, -0.20, 0.95])
    while True:
        arm.step()
        mujoco.mj_step(model, data)

Validated by `controller/experiments/exp_basic_controller.py`.
"""
from .pd import JointPD
from .wrist_ik import WristIK
from .arm import TaskSpaceArm, ARM_JOINTS, PALM_BODY

__all__ = ["JointPD", "WristIK", "TaskSpaceArm", "ARM_JOINTS", "PALM_BODY"]
