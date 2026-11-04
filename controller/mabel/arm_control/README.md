# `mabel.arm_control` — real-robot arm control law

Gravity feed-forward, joint control modes, soft (hand-guided) modes and the command gate that bounds what the host sends to the DaMiao arms. The HAL only does calibration, reading and writing; the control law lives here.

The motors close `T = kp·(pos − q) + kd·(vel − q̇) + t_ff`, so a mode is just a choice of `(pos, vel, kp, kd, t_ff)`:

| Mode | Behaviour |
|---|---|
| `PURE_PD` | baseline; droops under gravity |
| `GRAVITY_PD` | gravity fed forward; normal hold/move mode (`stiff`, `position`) |
| `COMPLIANT` | drag and hold (`soft`, `compliance`) |

Gravity FF is mandatory: the firmware caps `kd ≤ 5`, so stiff gains alone cannot hold the arm.

| Module | Contents |
|---|---|
| `gravity.py` | `ArmGravityCompensator`, `HandsConfig` (`NONE`/`ORCA`; env `MABEL_ARMS_HANDS`) |
| `impedance.py` | `ArmControlMode`, `ArmImpedance`, `JointCommand` |
| `drag_follow.py`, `admittance.py` | soft modes |
| `ff_schedule.py`, `friction.py` | FF scheduling, optional Coulomb FF |
| `rate_limit.py`, `wire_gate.py` | `JointGlideLimiter`; `deployed_glide_limiter()` builds the full gate |

```python
from mabel.arm_control import ArmGravityCompensator, ArmImpedance, ArmControlMode, HandsConfig
grav = ArmGravityCompensator(mjcf_path, dir_by_joint, tmax_by_joint, hands=HandsConfig.NONE)
imp = ArmImpedance(base_gains={joint: (kp, kd)}, mode=ArmControlMode.GRAVITY_PD)
cmd = imp.command(joint, grav.tff(model_angles, ramp).get(joint, 0.0))
```

## Safety

- FF is clamped to 0.6·t_max, ramps in over 1.5 s, and is zeroed if any joint leaves its MJCF range (stale-calibration guard).
- The glide limiter bounds commanded error and velocity from a torque budget. Overrides: `MABEL_ARM_MAX_SPEED`, `MABEL_ARM_ERR_MAX`, `MABEL_ARM_TORQUE_BUDGET` (see `wire_gate.py`).
- The HAL has no stall guard or slew limit. Keeping joints off their stops is this package's job.

Reports: [`hardware_bridge/docs/arm_controllers.pdf`](../../../hardware_bridge/docs/arm_controllers.pdf), [`controller/docs/arm_pd_tuning.pdf`](../../docs/arm_pd_tuning.pdf).
