"""Visualise the real ORCA hands on the mabel_description model (bridge client).

The hardware bridge owns the Feetech buses; ROS never opens a device. This launch wires
the bridge's live hand feedback into RViz:

    hardware bridge  ->  /<side>_hand/joint_states   (ORCA joint names)
        -> hand_joint_state_relay  ->  /joint_states  (URDF joint names, thumb remapped)
        -> robot_state_publisher (mabel_description URDF)  ->  RViz

The non-hand joints stay at zero; the hand(s) move for real. Back-drive a hand (torque
off — toggle it from the control center or `hand_status`) and the model tracks it.

Prerequisite — the bridge must be up with the hand subsystem enabled:
    MABEL_HW_HANDS=true MABEL_HW_HANDS_LEFT_PORT=/dev/ttyACM0 hardware_bridge/run_bridge.sh

    ros2 launch mabel_hand_hardware visualize.launch.py
    ros2 launch mabel_hand_hardware visualize.launch.py sides:=left
    ros2 launch mabel_hand_hardware visualize.launch.py rviz:=false

Args:
    sides : which hand feedback to relay -- left | right | both (default both).
    rviz  : launch RViz2 (default true).
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    desc_pkg = get_package_share_directory("mabel_description")
    urdf_path = os.path.join(desc_pkg, "urdf", "mabel_full.urdf")
    rviz_path = os.path.join(desc_pkg, "rviz", "mabel.rviz")
    with open(urdf_path) as f:
        robot_description = f.read()

    sides = LaunchConfiguration("sides")
    rviz = LaunchConfiguration("rviz")

    # Relay only the requested side(s); a missing side stays at the URDF home pose.
    inputs = PythonExpression([
        "{'left': ['/left_hand/joint_states'], 'right': ['/right_hand/joint_states']}"
        ".get('", sides, "', ['/left_hand/joint_states', '/right_hand/joint_states'])"])

    return LaunchDescription([
        DeclareLaunchArgument("sides", default_value="both",
                              description="Which hand feedback to relay: left | right | both."),
        DeclareLaunchArgument("rviz", default_value="true",
                              description="Launch RViz2."),

        # --- bridge feedback -> URDF joint names ---------------------------
        Node(
            package="mabel_hand_hardware", executable="hand_joint_state_relay",
            name="hand_joint_state_relay", output="screen",
            parameters=[{"input_topics": inputs,
                         "output_topic": "/joint_states",
                         "remap_thumb": True}],
        ),

        # --- robot model + RViz --------------------------------------------
        Node(
            package="robot_state_publisher", executable="robot_state_publisher",
            name="robot_state_publisher", output="screen",
            parameters=[{"robot_description": robot_description}],
        ),
        Node(
            package="rviz2", executable="rviz2", name="rviz2", output="screen",
            arguments=["-d", rviz_path], condition=IfCondition(rviz),
        ),
    ])
