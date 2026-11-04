"""Audio bring-up: the thin bridge CLIENT only.

The sound card (ALSA ownership, capture/playback, mixer, VID:PID detection) is
owned by the standalone hardware_bridge (auto-started by systemd); this launch
adds the ROS-side monitor + /mabel_cmd e-stop mirror, mirroring arms.launch.py.

Prerequisite: the hardware bridge must be running with the audio subsystem
enabled, e.g.
    MABEL_HW_AUDIO=true hardware_bridge/run_bridge.sh
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "follow_estop", default_value="true",
            description="Mirror /mabel_cmd e-stop onto /audio/enable (engage=mic "
                        "muted, release=mic live). ON by default: unlike torque, "
                        "muting is not a motion risk, and an e-stopped robot "
                        "should stop streaming the room's audio."),
        Node(
            package="mabel_audio_hardware",
            executable="audio_node",
            name="audio_hardware_node",
            output="screen",
            parameters=[{
                "follow_estop": LaunchConfiguration("follow_estop"),
            }],
        ),
    ])
