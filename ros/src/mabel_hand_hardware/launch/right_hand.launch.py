"""RIGHT-hand feedback relay (bridge client) — RViz mirrors the real right hand.

The hardware bridge owns the device; this just relays the bridge's /right_hand feedback
to URDF-named /hands/joint_states. Thin wrapper over both_hands.launch.py sides:=right.
The bridge must be up with the hand subsystem enabled (see both_hands.launch.py)."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    here = os.path.join(get_package_share_directory("mabel_hand_hardware"),
                        "launch", "both_hands.launch.py")
    return LaunchDescription([
        IncludeLaunchDescription(PythonLaunchDescriptionSource(here),
                                 launch_arguments={"sides": "right"}.items()),
    ])
