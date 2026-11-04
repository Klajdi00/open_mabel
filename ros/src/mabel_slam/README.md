# mabel_slam

SLAM over the sensor topics that `mabel_sim` or the robot already publish (it starts no robot or camera driver). `use_sim_time:=auto` detects the clock, so the same command works on both.

| Launch | Method | Compute |
|---|---|---|
| `lidar_slam.launch.py` | `slam_toolbox` 2D on `/scan` → `/map` | CPU |
| `visual_slam.launch.py` | cuVSLAM + nvblox (3-D mesh) | NVIDIA GPU, Isaac ROS |

Run one per session (both publish `map → odom`; wheel odometry owns `odom → base_link`).

## Run

```bash
ros2 launch mabel_slam lidar_slam.launch.py                  # rviz:=true, fresh_map:=false
ros2 run nav2_map_server map_saver_cli -f my_map

source ros/scripts/slam_env.sh                               # Isaac ROS shell, see ../../SETUP.md
ros2 launch mabel_slam visual_slam.launch.py                 # env:=navigation (front D435i, 5 cm)
ros2 launch mabel_slam visual_slam.launch.py env:=manipulation   # head ZED, 1 cm mesh
ros2 launch mabel_slam visual_slam.launch.py nav_2d_map:=true    # ESDF slice on /map for Nav2
```

Config: `config/lidar_slam.yaml`, `config/nvblox_{navigation,manipulation}.yaml`. Nodes: `map_relay`, `slice_to_occupancy`, `mesh_to_pointcloud`, `laser_odometry`.

Troubleshooting: no `/map` → `ros2 lifecycle get /slam_toolbox`; "resource not found in ament index" from visual SLAM → source `slam_env.sh`.
