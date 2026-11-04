"""Hand bringup as a BRIDGE CLIENT — RViz mirrors the real ORCA hands.

The hardware bridge (``hardware_bridge/``) is the ONLY owner of the Feetech serial
buses; ROS never opens a device. This launch therefore starts NO driver — it starts the
joint-state relay, which subscribes to the bridge's per-side feedback
(``/<side>_hand/joint_states``, published with ORCA joint names), remaps the thumb chain
to the URDF names, and republishes the union on ``/hands/joint_states`` so
``joint_state_publisher`` / ``robot_state_publisher`` animate the model live. Physically
back-driving a hand moves it in RViz.

Prerequisite: the hardware bridge must be running with the hand subsystem enabled, e.g.
    MABEL_HW_HANDS=true MABEL_HW_HANDS_LEFT_PORT=/dev/ttyACM0 hardware_bridge/run_bridge.sh

Commands still go to the bridge on the SAME contract (``/<side>_hand/command``,
``/torque_enable``) — published by teleop or the control center. End-stop calibration is
NOT a ROS action: it is done directly on the bus by the hardware bridge (the Control
Center's Hands → Calibration tab, or ``python3 -m hands.orcahand.calibrate``), with the
bridge's hand subsystem stopped so the adapter is free.

    ros2 launch mabel_hand_hardware both_hands.launch.py            # both hands' feedback
    ros2 launch mabel_hand_hardware both_hands.launch.py sides:=left
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    sides = LaunchConfiguration("sides")
    out_topic = LaunchConfiguration("output_topic")

    # Feed the relay only the side(s) requested; a side the bridge isn't publishing
    # simply never arrives and its joints stay at the model's home pose.
    inputs = PythonExpression([
        "{'left': ['/left_hand/joint_states'], 'right': ['/right_hand/joint_states']}"
        ".get('", sides, "', ['/left_hand/joint_states', '/right_hand/joint_states'])"])

    return LaunchDescription([
        DeclareLaunchArgument("sides", default_value="both",
                              description="Which hand feedback to relay: left | right | both."),
        DeclareLaunchArgument("output_topic", default_value="/hands/joint_states",
                              description="URDF-named joint_states the relay republishes "
                                          "(merged into /joint_states upstream)."),
        Node(
            package="mabel_hand_hardware", executable="hand_joint_state_relay",
            name="hand_joint_state_relay", output="screen",
            parameters=[{"input_topics": inputs,
                         "output_topic": out_topic,
                         "remap_thumb": True}],
        ),
    ])
