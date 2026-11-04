# mabel_hand_hardware

ROS tools for the two ORCA hands: a feedback relay for RViz, a per-hand terminal dashboard, and per-hand config (joint ↔ servo ID, angle ranges, tick limits). The hand serial buses belong to the HAL, so these launches are only clients of its hand topics. Start the HAL first, e.g. `MABEL_HW_HANDS=true hardware_bridge/run_bridge.sh`.

## Use

```bash
ros2 launch mabel_hand_hardware left_hand_bringup.launch.py      # dashboard (Tab toggles torque)
ros2 launch mabel_hand_hardware visualize.launch.py sides:=left  # relay + robot_state_publisher + RViz
ros2 launch mabel_hand_hardware both_hands.launch.py             # relay only -> /hands/joint_states
ros2 topic pub --once /left_hand/command sensor_msgs/msg/JointState '{name: [index_mcp], position: [1.2]}'
```

Per hand (`/left_hand`, `/right_hand`): `command`, `torque_enable` in; `joint_states`, `temperatures`, `voltages`, `torque_state` (latched) out.

| Executable | Role |
|---|---|
| `hand_status` | terminal dashboard (topics only) |
| `hand_joint_state_relay` | URDF-named joint states; maps firmware thumb `cmc/mp/ip` → URDF `mcp/pip/dip` |
| `hand_node` | direct serial driver for benches without the HAL. Never run it while the HAL owns the hand |

Config: `config/<side>_hand.yaml` (IDs, angle ranges, `invert`) and `config/<side>_calibration.json` (tick limits; every command is clamped to them).

Serial permissions (adapters enumerate as `/dev/ttyACM*`):

```bash
sudo cp ros/src/mabel_hand_hardware/udev/99-mabel-hand.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
```
