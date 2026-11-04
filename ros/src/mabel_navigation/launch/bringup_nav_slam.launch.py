"""Full autonomy bring-up for MABEL: SLAM (Isaac ROS) + Nav2.

    ros2 launch mabel_navigation bringup_nav_slam.launch.py

Assumes the robot base/hardware (or mabel_sim) is already running and
publishing /odom + /joint_states, and that the robot description
(robot_state_publisher) is up so TF has base_link and the camera frame.

Order of operations on the real robot:
    1. ros2 launch mabel_bringup ...        # base hardware + robot_state_publisher
    2. ros2 launch mabel_navigation bringup_nav_slam.launch.py
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    slam_pkg = get_package_share_directory('mabel_slam')
    nav_pkg = get_package_share_directory('mabel_navigation')

    publish_odom_tf = LaunchConfiguration('publish_odom_to_base_tf')

    return LaunchDescription([
        DeclareLaunchArgument(
            'publish_odom_to_base_tf', default_value='true',
            description='cuVSLAM owns odom->base_link (disable wheel-odom TF).'),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(slam_pkg, 'launch', 'visual_slam.launch.py')),
            launch_arguments={
                'env': 'navigation',   # front RealSense + 2D nav slice for Nav2
                'publish_odom_to_base_tf': publish_odom_tf,
            }.items()),

        # navigation.launch.py is Nav2 + the bridge ONLY (it never composes SLAM),
        # and visual_slam is already composed above -- so there is a single
        # map->odom owner and no duplicate stack.
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(nav_pkg, 'launch', 'navigation.launch.py'))),
    ])
