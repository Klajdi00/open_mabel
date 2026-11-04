"""Real MABEL swerve base -- now a CLIENT of the standalone hardware_bridge.

The base HARDWARE (Teensy link + validated ps5 IK + calibration) is owned by
``~/open_mabel/hardware_bridge`` (the single HAL, auto-started at boot). This launch brings
up only the ROS-side pieces:

  base_bridge_client : /mabel_cmd  ->  hardware_bridge DDS contract
                       (/mabel/hw/base/cmd_vel + estop + /lift/command), and the
                       bridge's /mabel/hw/base/joint_states  ->  /joint_states.
  swerve_odometry    : /joint_states -> /odom + odom->base_link (the SAME forward-
                       kinematics motion model the sim runs, fed by the real Teensy
                       encoders relayed through the bridge -- no sim privilege).

cuVSLAM (mabel_slam) corrects map->odom on top, so set publish_odom_tf:=false here if
you instead let cuVSLAM own odom->base_link.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    wheel_radius = LaunchConfiguration('wheel_radius')
    odometry = LaunchConfiguration('odometry')
    publish_odom_tf = LaunchConfiguration('publish_odom_tf')
    joint_states_topic = LaunchConfiguration('joint_states_topic')

    return LaunchDescription([
        # Accepted for backward-compat (hardware.launch.py still passes it), but IGNORED:
        # the Teensy link + IP are now owned by ~/open_mabel/hardware_bridge (set there via
        # MABEL_HW_BASE_TEENSY_IP). Kept so existing launch callers don't error.
        DeclareLaunchArgument('teensy_ip', default_value='192.168.124.10',
                              description='DEPRECATED/ignored here — the hardware_bridge '
                                          'owns the Teensy link. Set the bridge IP via '
                                          'MABEL_HW_BASE_TEENSY_IP.'),
        DeclareLaunchArgument('joint_states_topic', default_value='/joint_states',
                              description='Topic the base encoder JointState is '
                                          'republished on (and that swerve_odometry '
                                          'reads). Standalone: /joint_states. Under '
                                          'the full hardware bring-up move it to '
                                          '/base/joint_states so a joint_state_'
                                          'publisher can merge ALL subsystems.'),
        DeclareLaunchArgument('wheel_radius', default_value='0.0381',
                              description='Swerve rolling radius (m); REV 3in '
                                          'MAXSwerve = 0.0381. MUST match the value '
                                          'the hardware_bridge base uses.'),
        DeclareLaunchArgument('odometry', default_value='true',
                              description='Run swerve_odometry (wheel encoders -> '
                                          '/odom + odom->base_link), the real motion '
                                          'model. Off = command-only base.'),
        DeclareLaunchArgument('publish_odom_tf', default_value='true',
                              description='Whether swerve_odometry broadcasts the '
                                          'odom->base_link TF. Set false when cuVSLAM '
                                          'owns it or an EKF fuses wheel + visual.'),

        # Thin client: translates /mabel_cmd <-> the hardware_bridge DDS contract.
        # The base hardware itself is owned by ~/open_mabel/hardware_bridge (the HAL).
        Node(
            package='mabel_base_hardware',
            executable='swerve_node',
            name='base_bridge_client',
            output='screen',
            remappings=[('/joint_states', joint_states_topic)],
        ),

        # Motion model: swerve forward kinematics from the REAL /joint_states (relayed
        # by the bridge) -> /odom + odom->base_link. Identical node/params to the sim.
        Node(
            package='mabel_base_hardware',
            executable='swerve_odometry',
            name='swerve_odometry',
            output='screen',
            parameters=[{
                'wheel_radius': ParameterValue(wheel_radius, value_type=float),
                'publish_tf': ParameterValue(publish_odom_tf, value_type=bool),
            }],
            remappings=[('/joint_states', joint_states_topic)],
            condition=IfCondition(odometry),
        ),
    ])
