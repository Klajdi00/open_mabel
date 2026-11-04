# simulation — the digital twin

The MuJoCo model, scene library and interactive scripts. `mabel_mujoco/models/mabel_full.xml` is **the canonical robot model**: the URDF, web GLB, app rigs and controller CoM constants are all derived from it (regenerate them as in the root README's [Step 11](../README.md#step-11--change-the-robot-model)). Seed every `MjData` from keyframe 0 (`home`), because the lift's default is half-extended, not q = 0.

The physics runtime behind the teleop stack is [`../simulation_bridge/`](../simulation_bridge/). RL environments live in [`../robot_learning/reinforcement_learning/`](../robot_learning/reinforcement_learning/).

## Run

```bash
bash server/run.sh                                     # full stack (gateway + studio) on the twin, from the repo root

cd simulation/mabel_mujoco                             # standalone scripts (mjpython on macOS)
mjpython scripts/viewer.py
mjpython scripts/keyboard_teleop.py
mjpython scripts/compliance.py --region whole
mjpython scripts/visualize_sensors.py --scene scenes/mabel_kitchen_fk.xml
```

| Path | Contents |
|---|---|
| `mabel_mujoco/models/` | `mabel_full.xml` + per-subsystem includes |
| `mabel_mujoco/assets/meshes/` | per-link meshes |
| `mabel_mujoco/scenes/` | scene library, registered in `scenes.json`; `*.dynamics.json` script moving actors |
| [`mabel_mujoco/scripts/`](mabel_mujoco/scripts/) | viewer, teleop, compliance, sensor check, scene tools |
| `mabel_mujoco/third_party/` | object/arena assets, credited in `ATTRIBUTION.md` |
| `tools/` | model → app-rig exporters (`glb_to_rig.py`), motion clips, `platform_specs.py` |

Reports ([`mabel_mujoco/docs/`](mabel_mujoco/docs/)): `teleop_system.pdf`, `motion_model.pdf`, `whole_body_control.pdf`, `arm_control.pdf`, `compliance_study.pdf`, `self_collision.pdf`.
