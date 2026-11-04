"""View the MABEL robot in RViz with interactive joint sliders.

    ros2 launch mabel_description display.launch.py

Brings up robot_state_publisher (from the generated URDF), the
joint_state_publisher_gui (slider panel for all 60 joints), and RViz2 with a
preconfigured RobotModel + TF view.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory("mabel_description")
    urdf_path = os.path.join(pkg, "urdf", "mabel_full.urdf")
    rviz_path = os.path.join(pkg, "rviz", "mabel.rviz")

    with open(urdf_path) as f:
        robot_description = f.read()

    gui = LaunchConfiguration("gui")

    return LaunchDescription([
        DeclareLaunchArgument(
            "gui", default_value="true",
            description="Launch joint_state_publisher_gui sliders."),

        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            output="screen",
            parameters=[{"robot_description": robot_description}],
        ),
        Node(
            package="joint_state_publisher_gui",
            executable="joint_state_publisher_gui",
            name="joint_state_publisher_gui",
            condition=IfCondition(gui),
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            output="screen",
            arguments=["-d", rviz_path],
        ),
    ])
