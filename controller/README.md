# controller — the `mabel` control library

The single control layer between every operator device and the robot: an operator pose (Vision Pro, iOS, browser/webcam, keyboard, gamepad, Nav2) goes in, bounded per-actuator commands come out. The same code drives the MuJoCo twin and the real robot, and all robot quantities come from FK on `simulation/mabel_mujoco/models/mabel_full.xml`. It never opens a device: it runs inside the teleop gateway (`server/sim_teleop_bridge.py`), which forwards to `hardware_bridge/` or `simulation_bridge/`. See the root [Architecture](../README.md#architecture).

## Safety rules

- `controller/` is the only gate on motion. Every base twist goes through `mabel.motion_model.safe_base_step(...)` (lift-aware ZMP tip-over gate).
- Self-collision avoidance stays on (`config.SELF_COLLISION_ENABLE = True`); it is the only arm-arm defence.
- The arm glide limiter in `arm_control/` bounds commanded error and velocity, which are the only torque bounds the host has.
- On the real robot the arms start limp. The operator has to enable torque explicitly.

## Install and test

```bash
pip install -e controller/              # or: cd controller && ./install.sh [--qp] [--check]
python3 -m pytest controller/tests/ -q
```

## Use

```python
from mabel import MabelTeleop, OperatorPose, app_joystick
teleop = MabelTeleop(model, data)                  # MuJoCo model/data (sim or kinematic mirror)
cmd = teleop.update(OperatorPose.from_teleop_frame(msg), app_joystick(msg.get("navJoystick")), dt)

from mabel.config import safe_base_step            # NumPy-only, e.g. from a ROS node
vx, vy, w = safe_base_step(vx, vy, w, tvx, tvy, tw, dt, lift_z)
```

## Layout (`mabel/`)

| Subpackage | Role |
|---|---|
| `facade.py`, `types.py` | `MabelTeleop`; `OperatorPose` in, `RobotCommand` out |
| `teleop_engine/` | the deployed retargeter + kinematic whole-body stack (`hyperparams.py`) |
| [`sources/`](mabel/sources/) | device ingest → `OperatorPose` |
| [`hand_retargeting/`](mabel/hand_retargeting/), [`whole_body_retargeting/`](mabel/whole_body_retargeting/) | fingers → ORCA; head + wrists → body |
| [`motion_model/`](mabel/motion_model/) | tip-over envelope, `safe_base_step`, base/swerve models |
| [`dynamic_models/`](mabel/dynamic_models/) | gravity feed-forward, momentum observer |
| [`whole_body_controller/`](mabel/whole_body_controller/) | IK, PD, self-collision, admittance, compliance, swerve IK |
| [`arm_control/`](mabel/arm_control/) | real-robot arm control law and command gate |
| `torso_control/`, `safety/`, `basic/`, `runtime/` | torso FF; tracking-loss fail-safe; PD baseline; HUD/recorder |

Config: `mabel/config.py` (constants), `control_param.json` (gains, gravity FF, limits), `mabel/motion_model/tip_over.py` (CoM constants).

## After an MJCF change

```bash
python3 controller/experiments/wbc_stability.py                           # E4 prints CoM constants; update tip_over.py if >~1 cm off
python3 controller/experiments/exp_front_workspace.py --phase W3 --build  # rebuild posture tables
```

`experiments/exp_*.py` are seeded and write figures to `docs/figures/`.

## Reports ([`docs/`](docs/))

Start with [`mabel_controller.pdf`](docs/mabel_controller.pdf) (end-to-end stack) and [`control_stack.pdf`](docs/control_stack.pdf) (layer atlas). Per module: `sources`, `hand_retargeting`, `whole_body_retargeting`, `body_frame`, `retargeting_modes`, `retarget_scope`, `front_workspace_retargeting`, `egodex_retargeting`, `vp_bimanual_latency`, `calibration`, `motion_model`, `wbc_stability`, `dynamic_models`, `whole_body_controller`, `basic_controller`, `arm_pd_tuning`, `rest_pose`. Hardware arm control law: [`hardware_bridge/docs/arm_controllers.pdf`](../hardware_bridge/docs/arm_controllers.pdf).
