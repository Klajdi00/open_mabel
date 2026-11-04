"""Arms bring-up: the thin bridge CLIENT only.

The arm hardware (Teensy ethernet link / CAN, calibration, torque safety) is owned by
the standalone hardware_bridge (auto-started by systemd); this launch adds the ROS-side
translator/monitor — /mabel_cmd e-stop mirror + liveness — mirroring base.launch.py.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "follow_estop", default_value="false",
            description="Mirror /mabel_cmd e-stop onto /arms/torque_enable "
                        "(engage=limp, release=energise-with-hold). OFF by default "
                        "until the arms are calibrated — when false this node never "
                        "publishes torque_enable, so the arms stay limp unless "
                        "energised explicitly."),
        Node(
            package="mabel_arms_hardware",
            executable="arms_node",
            name="arms_hardware_node",
            output="screen",
            parameters=[{
                "follow_estop": LaunchConfiguration("follow_estop"),
            }],
        ),
    ])
