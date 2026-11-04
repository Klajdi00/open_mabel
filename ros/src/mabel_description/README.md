# mabel_description

URDF and meshes for the full robot (base, lift, torso, head, two 7-DOF arms, two ORCA hands), generated from `simulation/mabel_mujoco/models/mabel_full.xml`. Never edit the URDF by hand.

## Use

```bash
ros2 launch mabel_description display.launch.py              # RViz + joint sliders
ros2 launch mabel_description display.launch.py gui:=false   # another node drives /joint_states
```

## Regenerate (from the repo root, after any MJCF change)

```bash
python3 ros/src/mabel_description/scripts/mjcf_to_urdf.py \
  --mjcf simulation/mabel_mujoco/models/mabel_full.xml \
  --out  ros/src/mabel_description/urdf/mabel_full.urdf \
  --mesh-prefix package://mabel_description/meshes
cp -r simulation/mabel_mujoco/assets/meshes/* ros/src/mabel_description/meshes/   # if meshes changed
python3 ros/src/mabel_description/scripts/validate_urdf.py    # compare against MuJoCo FK
```

Then continue the model pipeline (`urdf_to_glb.py`, …) in the [root README](../../../README.md#step-11--change-the-robot-model).

Note: the 14 ORCA finger/wrist joints have MuJoCo `ref ≠ 0`; URDF `q = 0` is the assembled pose, so clients rendering raw MuJoCo `qpos` must subtract `ref`.
