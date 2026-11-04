# ros — ROS 2 Jazzy workspace

Robot description, a MuJoCo sim that speaks the hardware topic contract, thin clients of the HAL, SLAM (2D LiDAR or GPU visual) and Nav2. Sim (`mabel_sim`) and real (`mabel_bringup`) serve the same topics, so SLAM/Nav2/teleop run unchanged; launch exactly one backend.

- ROS never opens a device: on the robot `hardware_bridge/` owns every bus and camera.
- ROS is not its own command source: Nav2/keyboard twists go through `cmd_vel_bridge` → the controller's tip-safe gate.

See the root [Architecture](../README.md#architecture). The URDF is generated from the MJCF (`mabel_description`).

## Setup (Ubuntu 24.04)

```bash
cd ros
./dev_up.sh                  # conda env + colcon build + 4-pane tmux session "mabel"
./dev_up.sh --visual         # visual SLAM instead of LiDAR; also --no-build, --no-tmux, --clean, --with-sdk
source scripts/mabel_env.sh  # in every new shell
colcon build --symlink-install && source install/setup.bash
```

Native SDKs (ZED, Isaac ROS) and manual install: [SETUP.md](SETUP.md).

## Run

```bash
# simulation
ros2 launch mabel_sim sim.launch.py front_only:=true       # physics + sensors + RViz
ros2 launch mabel_slam lidar_slam.launch.py use_sim_time:=true
ros2 run mabel_teleop keyboard_teleop
ros2 launch mabel_navigation navigation.launch.py          # Nav2 goals from RViz
ros2 launch mabel_navigation exploration.launch.py         # frontier exploration

# real robot (full stack incl. HAL: server/real_run.sh); ROS side only, HAL already running:
ros2 launch mabel_bringup hardware.launch.py arms:=false hands:=false   # disable absent subsystems
```

GPU visual SLAM: `source scripts/slam_env.sh`, then `ros2 launch mabel_slam visual_slam.launch.py`.

## Packages

| Package | Purpose |
|---|---|
| [`mabel_description`](src/mabel_description/) | URDF + meshes from the MJCF; RViz display |
| `mabel_interfaces` | `MabelSwerveCmd` (twist, lift, e-stop state) |
| [`mabel_sim`](src/mabel_sim/) | MuJoCo physics + simulated cameras/LiDAR/IMU; topic contract in `mabel_sim/interface_config.py` |
| `mabel_bringup` | real-robot entry (`hardware.launch.py`), sim/real teleop launch |
| `mabel_{base,arms,body,audio}_hardware` | thin HAL clients (odometry, e-stop mirror, liveness) |
| [`mabel_hand_hardware`](src/mabel_hand_hardware/) | ORCA hand feedback relay, dashboard, RViz |
| [`mabel_sensors`](src/mabel_sensors/) | RealSense/ZED launches, camera calibration |
| [`mabel_slam`](src/mabel_slam/) | `slam_toolbox` 2D SLAM, cuVSLAM + nvblox |
| [`mabel_navigation`](src/mabel_navigation/) | Nav2, `cmd_vel_bridge`, exploration, EKF |
| `mabel_teleop` | keyboard teleop |
| `mabel_control`, `mabel_gateway` | ROS plant + launcher for the teleop gateway (`server/sim_teleop_bridge.py`) |

"Package not found" almost always means `install/setup.bash` was not sourced. If sim sensors stutter, run one sim only, use `front_only:=true` and `pointcloud_stride:=4`.
