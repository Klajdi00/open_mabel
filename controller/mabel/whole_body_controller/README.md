# `mabel.whole_body_controller`

Turns desired motion (retargeted poses, client joint targets, a physical push) into per-actuator commands. One law per module; `engine.py` (`WBCEngine`) composes them as control type (`pd`, `impedance`, `compliance`) × region (`arm`, `upper_body`, `whole_body`).

Rules: base twists reach `base_controller.swerve_ik` only through `mabel.motion_model.safe_base_step`, and the self-collision guard stays on (the MJCF disables arm-arm contacts).

| Module | Role |
|---|---|
| `ik.py` | `WholeBodyIK`: bimanual damped-least-squares IK with null-space posture |
| `position_controller/` | `JointPositionController`: per-joint PD, MJCF servo gains, optional `feedforward=` |
| `gravity_compensation/` | `GravityCompController`: back-drivable gravity hold |
| `collision_avoidance/` | `SelfCollisionGuard` (capsules; `MABEL_SC_METHOD=project\|stop\|cbf`), `JointTargetProjector` for client targets |
| `admittance/` | wrench/contact estimation, `BaseAdmittance` |
| `compliance/` | `ComplianceController`: move the robot by hand |
| `base_controller/` | `swerve_ik`: 3-module swerve IK |
| `task_space.py` | `BaseFollow`: base follows out-of-reach palm targets |

```python
from mabel.whole_body_controller import JointPositionController, WBCEngine
from mabel.dynamic_models import region_joints
pd = JointPositionController(plant, region_joints("arms"))
eng = WBCEngine(model, data); eng.enter("compliance", "whole_body"); eng.step(dt)
```

The browser studios preview the same collision law in `web_gui/*/web/js/collision.js`. `controller/tests/test_wrist_target_projection.py` checks the two agree.

Report: [`controller/docs/whole_body_controller.pdf`](../../docs/whole_body_controller.pdf).
