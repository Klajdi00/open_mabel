"""Fused localization: robot_localization EKF (wheel velocity + cuVSLAM pose).

Owns the odom->base_link TF, fusing:
  /odom                              -- swerve wheel VELOCITY (down-weighted; slips)
  /visual_slam/tracking/odometry     -- cuVSLAM absolute pose (trusted)
-> /odometry/filtered.

Run swerve_odometry with publish_tf:=false and cuVSLAM with
publish_odom_to_base_tf:=false so this node is the single owner of odom->base_link.
Tuned from the ground-truth benchmark; see config/ekf.yaml.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetParameter


def generate_launch_description():
    pkg = get_package_share_directory("mabel_navigation")
    default_cfg = os.path.join(pkg, "config", "ekf.yaml")
    cfg = LaunchConfiguration("ekf_config")
    return LaunchDescription([
        SetParameter(name="use_sim_time", value=True),   # fuse on the sim clock
        DeclareLaunchArgument("ekf_config", default_value=default_cfg,
                              description="robot_localization EKF params file."),
        Node(
            package="robot_localization",
            executable="ekf_node",
            name="ekf_filter_node",
            output="screen",
            parameters=[cfg],
        ),
    ])
