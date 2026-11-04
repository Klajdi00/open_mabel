"""LEFT-hand live status dashboard (bridge client).

The hardware bridge owns the hand; ROS never opens the serial bus. This launch starts
only ``hand_status`` -- a compact in-place terminal panel showing all 17 motors (ID,
joint, position, velocity, torque, temperature) in the keyboard_teleop style -- which
subscribes to the bridge's ``/left_hand/{joint_states,temperatures,voltages,torque_state}``
and toggles torque via ``/left_hand/torque_enable``.

Prerequisite -- the bridge must be up with the hand subsystem enabled:
    MABEL_HW_HANDS=true MABEL_HW_HANDS_LEFT_PORT=/dev/ttyACM0 hardware_bridge/run_bridge.sh

    ros2 launch mabel_hand_hardware left_hand_bringup.launch.py

Press Tab in the dashboard to toggle TORQUE ENGAGED <-> FREE-DRIVE live (the bridge
de-energises the motors so the joints move freely by hand while feedback keeps streaming).
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        # Live status dashboard (writes to /dev/tty, renders under ros2 launch).
        Node(
            package="mabel_hand_hardware", executable="hand_status",
            name="hand_status", output="screen", emulate_tty=True,
            parameters=[{"side": "left"}],
        ),
    ])
