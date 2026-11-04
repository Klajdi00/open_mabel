"""
macnav — the ROS-free navigation + 2-D SLAM fallback for the Mac sim path.

On Ubuntu the ROS stack owns mapping/navigation (slam_toolbox / cuVSLAM+nvblox
+ Nav2); a Mac cannot run those packages, so this package re-implements the
pipeline natively, consuming ONLY what the simulation_bridge publishes:

  wheel joint states + gyro  →  odometry.SwerveOdometry   (the SAME swerve FK
                                                           frame conventions as
                                                           mabel_base_hardware/
                                                           swerve_odometry)
  360-ray planar lidar       →  slam.SlamEngine           (gmapping-style:
                                                           odometry-proposed,
                                                           correlative-scan-
                                                           matcher-corrected,
                                                           log-odds grid)
  occupancy grid + goal      →  planner.astar             (Nav2 NavFn analog)
  plan + grid + pose         →  mppi.MPPI                 (Nav2 MPPI analog;
                                                           faces the velocity
                                                           direction, avoids
                                                           obstacles)
  everything                 →  navigator.Navigator       (goal lifecycle)

The motion limits come from controller/mabel/config.py — the same motion
model the teleop stack and the ROS side share.
"""
