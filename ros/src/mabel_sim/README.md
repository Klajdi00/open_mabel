# mabel_sim

MuJoCo simulation on the hardware topic contract: physics plus rendered cameras, LiDAR and IMU, so teleop, SLAM, Nav2 and RViz cannot tell it from the robot. Used for SLAM/Nav2 work; the teleop gateway's sim (`server/run.sh`) uses `simulation_bridge/` instead and needs no ROS.

## Run

```bash
ros2 launch mabel_sim sim.launch.py                                   # physics + sensors + RViz
ros2 launch mabel_sim sim.launch.py front_only:=true rviz:=false      # lowest CPU load
ros2 launch mabel_sim sim.launch.py scene:=$PWD/simulation/mabel_mujoco/scenes/mabel_bare.xml
ros2 topic pub --once /left_hand/command sensor_msgs/JointState "{name: [index_mcp], position: [1.2]}"
```

Launch either this or `mabel_bringup hardware.launch.py`, never both. Other args: `sensors`, `sensor_rate` (15), `pointcloud_stride` (2), `headless`, `foxglove`, `imu`, noise parameters (see `launch/sim.launch.py`).

| Node | Role |
|---|---|
| `mujoco_node` | applies commands, steps MuJoCo, publishes joint states, IMU, `/odom_ground_truth` |
| `mujoco_sensors_node` | renders cameras, depth clouds, LiDAR |
| `bringup_status` | terminal status panel (shared with the real bring-up) |

The topic contract (`/mabel_cmd`, `/arms/command`, `/body/command`, `/<side>_hand/command`, `/joint_states`, `/odom`, `/scan`, `/camera/*`) is defined in [`mabel_sim/interface_config.py`](mabel_sim/interface_config.py). `/odom` is wheel odometry, as on the robot. Requires `rclpy` and `mujoco` in one interpreter (the `mabel` conda env).
