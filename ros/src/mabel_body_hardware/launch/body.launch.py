from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='mabel_body_hardware',
            executable='body_node',
            name='body_hardware_node',
            output='screen'
        )
    ])
