"""Standalone Foxglove Studio bridge for MABEL.

Starts only the foxglove_bridge WebSocket server (ws://localhost:8765) so you can
attach Foxglove Studio to an ALREADY-running stack -- the sim, or the real
hardware bring-up -- without restarting it.

    ros2 launch mabel_sim foxglove.launch.py
    ros2 launch mabel_sim foxglove.launch.py port:=8766

Then in Foxglove Studio: Open connection -> Foxglove WebSocket -> ws://localhost:8765
Requires:  sudo apt install ros-humble-foxglove-bridge

If you want the URDF robot model in Foxglove's 3D panel, make sure a
robot_state_publisher is running somewhere (the sim launch starts one); this
launch does not, so it works for bare-topic inspection on any stack.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    port = LaunchConfiguration("port")
    return LaunchDescription([
        DeclareLaunchArgument(
            "port", default_value="8765",
            description="Port for the Foxglove WebSocket bridge."),
        Node(
            package="foxglove_bridge",
            executable="foxglove_bridge",
            name="foxglove_bridge",
            output="screen",
            parameters=[{
                "port": ParameterValue(port, value_type=int),
                "address": "0.0.0.0",
                "tls": False,
                "topic_whitelist": [".*"],
                "send_buffer_limit": 10000000,
                "asset_uri_allowlist": [
                    "^package://(?:\\w+/)*\\w+\\.(?:dae|fbx|glb|gltf|jpeg|jpg|"
                    "mtl|obj|png|stl|tif|tiff|urdf|webp|xacro)$"],
            }],
        ),
    ])
