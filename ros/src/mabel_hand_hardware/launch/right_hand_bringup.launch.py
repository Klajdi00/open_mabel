"""RIGHT-hand live status dashboard (bridge client).

The hardware bridge owns the hand; ROS never opens the serial bus. This launch starts
only ``hand_status`` -- a compact in-place terminal panel showing all 17 motors (ID,
joint, position, velocity, torque, temperature) -- which subscribes to the bridge's
``/right_hand/{joint_states,temperatures,voltages,torque_state}`` and toggles torque via
``/right_hand/torque_enable``.

Prerequisite -- the bridge must be up with the hand subsystem enabled:
    MABEL_HW_HANDS=true MABEL_HW_HANDS_RIGHT_PORT=/dev/ttyACM1 hardware_bridge/run_bridge.sh

    ros2 launch mabel_hand_hardware right_hand_bringup.launch.py

Press Tab in the dashboard to toggle TORQUE ENGAGED <-> FREE-DRIVE live (the bridge
de-energises the motors so the joints move freely by hand while feedback keeps streaming).
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="mabel_hand_hardware", executable="hand_status",
            name="hand_status", output="screen", emulate_tty=True,
            parameters=[{"side": "right"}],
        ),
    ])
