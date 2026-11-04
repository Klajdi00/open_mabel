# mabel_navigation

Nav2 navigation, frontier exploration, the localization EKF and `cmd_vel_bridge`. Navigation uses only `/map`, `/scan` and TF, so it runs unchanged on sim or robot and either SLAM backend. It only emits `/cmd_vel`; `cmd_vel_bridge` turns that into `/mabel_cmd` through the controller's tip-safe gate (`safe_base_step`), which can't be turned off.

## Run

Bring up the backend and SLAM first (these launches start neither):

```bash
ros2 launch mabel_sim sim.launch.py                    # or: mabel_bringup hardware.launch.py
ros2 launch mabel_slam lidar_slam.launch.py
ros2 launch mabel_navigation navigation.launch.py      # goals from RViz "2D Goal Pose"
ros2 launch mabel_navigation exploration.launch.py     # autonomous mapping; saves ~/mabel_map.{pgm,yaml}
ros2 launch mabel_navigation ekf.launch.py             # wheel + cuVSLAM fusion
```

On the real robot add `use_sim_time:=false`. Both launches first tuck the arms (`prepose:=true`). `bringup_nav_slam.launch.py` brings up visual SLAM and Nav2 together.

| Node | Role |
|---|---|
| `cmd_vel_bridge` | `/cmd_vel` → `MabelSwerveCmd` through the motion model (lift-aware caps + ZMP clamp) |
| `goal_relay` | RViz `/goal_pose` → `navigate_to_pose` |
| `frontier_explorer` | sends frontiers until the map is complete |
| `nav_pose_setter` | tucks arms, latches `/mabel/body_ready` |
| `command_odometry` | `/mabel_cmd` through the base model → `/odom_cmd` |

Config: `config/nav2_params.yaml` (MPPI, costmaps), `config/ekf.yaml`. Scripts: `scripts/footprint_from_urdf.py` (regenerate the footprint after base changes), `scripts/odom_eval.py`. Localization benchmark: `experiments/collect_localization.py` → `experiments/analyze_localization.py`.

```bash
colcon test --packages-select mabel_navigation && colcon test-result --verbose
```
