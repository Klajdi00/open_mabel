"""Spawn MABEL into Gazebo Classic.

    ros2 launch mabel_description gazebo.launch.py

REQUIRES Gazebo Classic + gazebo_ros (not installed by default on this box):
    sudo apt install ros-humble-gazebo-ros-pkgs ros-humble-gazebo-ros2-control

This starts gzserver/gzclient, publishes robot_description via
robot_state_publisher, and spawns the robot as an entity. The robot's joints
are passive in Gazebo until a ros2_control controller is added (see README).
For an Ignition/`gz` setup, swap the gazebo.launch.py include + spawn node for
the ros_gz equivalents (`ros_gz_sim`).
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory("mabel_description")
    urdf_path = os.path.join(pkg, "urdf", "mabel_full.urdf")
    with open(urdf_path) as f:
        robot_description = f.read()

    gazebo_ros = get_package_share_directory("gazebo_ros")
    z = LaunchConfiguration("z")

    return LaunchDescription([
        DeclareLaunchArgument("z", default_value="0.20",
                              description="Spawn height [m] above the floor."),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(gazebo_ros, "launch", "gazebo.launch.py")),
        ),
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            output="screen",
            parameters=[{"robot_description": robot_description,
                         "use_sim_time": True}],
        ),
        Node(
            package="gazebo_ros",
            executable="spawn_entity.py",
            name="spawn_mabel",
            output="screen",
            arguments=["-topic", "robot_description",
                       "-entity", "mabel",
                       "-z", z],
        ),
    ])
