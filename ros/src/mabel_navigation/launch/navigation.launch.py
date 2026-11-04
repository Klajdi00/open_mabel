"""Autonomous point-to-point navigation for MABEL -- pick a goal in RViz, drive there.

    ros2 launch mabel_navigation navigation.launch.py

WHAT THIS IS.  The Nav2 stack + the /cmd_vel -> /mabel_cmd base bridge, and NOTHING
else. Set a goal pose with the "2D Goal Pose" tool and the robot plans a path and
follows it (MPPI controller, holonomic swerve motion). It is the SIBLING of
exploration.launch.py, which reuses this exact stack but drives the goals itself.

This launch is NAVIGATION ONLY. It starts no HAL, no physics, and -- deliberately --
no SLAM. The intended layering is three independent launches, in order:
    1. HAL + sensors + odom + TF   (mabel_sim sim.launch.py  /  mabel_bringup hardware.launch.py)
    2. a SLAM backend              (mabel_slam lidar_slam.launch.py  /  visual_slam.launch.py)
    3. THIS -- navigation          (mabel_navigation navigation.launch.py)

RVIZ.  This launch opens NO RViz of its own by default (rviz:=false). The sim's
RViz (mabel_sim sim_sensors.rviz) already carries the map + costmaps + plan +
frontier + goal displays AND the "2D Goal Pose" tool, so navigation builds on top
of that ONE existing window. Pass rviz:=true for a standalone nav viewer.

INDEPENDENCE.  Navigation is deliberately decoupled from BOTH the SLAM and the
hardware:
  * It consumes only a universal contract -- ``/map`` (occupancy grid),
    ``/scan`` (LaserScan), and the ``map->odom->base_link`` TF -- which EVERY
    backend publishes, so the SAME Nav2 config (config/nav2_params.yaml) runs
    unchanged on the LiDAR 2D SLAM (slam_toolbox) or the visual 3D SLAM
    (cuVSLAM + nvblox). It never composes or references either one -- bring SLAM
    up yourself (step 2 above) before launching this.
  * It emits only ``/cmd_vel``; the ``cmd_vel_bridge`` turns that into the base's
    ``MabelSwerveCmd`` through the tip-safe motion model, so navigation is identical
    on the MuJoCo sim and the real swerve base.

PREREQUISITES (this launch starts NONE of these -- one-physics / one-HAL / one-SLAM rule):
  1. The robot HAL + sensors + wheel odometry + TF:
        sim:   ros2 launch mabel_sim sim.launch.py
        real:  ros2 launch mabel_bringup hardware.launch.py
  2. A SLAM backend publishing /map + map->odom:
        lidar:  ros2 launch mabel_slam lidar_slam.launch.py
        visual: ros2 launch mabel_slam visual_slam.launch.py

The cmd chain Nav2 runs internally:
    controller -> /cmd_vel_nav -> velocity_smoother -> /cmd_vel_smoothed
               -> collision_monitor -> /cmd_vel -> cmd_vel_bridge -> /mabel_cmd
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetParameter
from launch_ros.parameter_descriptions import ParameterValue


# Core Nav2 servers MABEL uses (NO route_server / waypoint_follower / docking_server
# -- the stock nav2_bringup launch brings those up and docking_server aborts the
# whole lifecycle with "Charging dock plugins not given!"; MABEL has no dock). The
# activation ORDER matters: costmap owners (controller/planner) before bt_navigator.
LIFECYCLE_NODES = [
    'controller_server',
    'smoother_server',
    'planner_server',
    'behavior_server',
    'bt_navigator',
    'velocity_smoother',
    'collision_monitor',
]


def generate_launch_description():
    nav_pkg = get_package_share_directory('mabel_navigation')

    # NOTE: this arg is 'nav2_params_file', NOT 'params_file'. A distinctly-named arg
    # cannot collide with the 'params_file' that other launches (e.g. the SLAM
    # backends, or a parent that composes several stacks) also declare, so these nav
    # params can never leak in and override another node's config.
    params_file = LaunchConfiguration('nav2_params_file')
    use_sim_time = LaunchConfiguration('use_sim_time')
    use_stamped = LaunchConfiguration('use_stamped_cmd_vel')
    autostart = LaunchConfiguration('autostart')
    rviz = LaunchConfiguration('rviz')
    rviz_config = LaunchConfiguration('rviz_config')
    prepose = LaunchConfiguration('prepose')

    tf_remap = [('/tf', 'tf'), ('/tf_static', 'tf_static')]
    # Nodes that emit a Twist route it through the smoother + collision_monitor, so
    # they publish to /cmd_vel_nav, not /cmd_vel.
    cmd_remap = tf_remap + [('cmd_vel', 'cmd_vel_nav')]
    common = {'use_sim_time': use_sim_time}

    # Custom behavior tree with AGGRESSIVE unstick recovery (clear costmaps ->
    # back up -> bidirectional spin -> wait, 12 retries). bt_navigator loads this
    # instead of the stock nav2 BT; the path must be absolute (resolved here).
    bt_xml = os.path.join(nav_pkg, 'behavior_trees',
                          'navigate_w_aggressive_recovery.xml')

    def server(pkg, exe, name, remaps, extra=None):
        params = [params_file, common]
        if extra:
            params = params + [extra]
        return Node(
            package=pkg, executable=exe, name=name, output='screen',
            parameters=params, remappings=remaps,
        )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time', default_value='true',
            description='Run Nav2 on the sim /clock. Set false on the real robot.'),
        DeclareLaunchArgument(
            'nav2_params_file',
            default_value=os.path.join(nav_pkg, 'config', 'nav2_params.yaml'),
            description='Nav2 parameter file (backend-agnostic costmaps). Named '
                        'distinctly so it cannot leak into the composed SLAM launch.'),
        DeclareLaunchArgument('autostart', default_value='true',
                              description='Auto-activate the Nav2 lifecycle nodes.'),
        DeclareLaunchArgument(
            'use_stamped_cmd_vel', default_value='false',
            description='Set true if Nav2 is configured to publish TwistStamped.'),
        DeclareLaunchArgument(
            'rviz', default_value='false',
            description="Open a SEPARATE navigation RViz (map + costmaps + path + "
                        "frontiers + the '2D Goal Pose' tool). DEFAULT false: the sim's "
                        "RViz (mabel_sim sim_sensors.rviz) already carries the nav "
                        "displays + the goal tool, so navigation builds ON TOP of that "
                        "ONE window instead of popping a second one. Set true only to run "
                        "a standalone nav viewer (e.g. on another machine, or when no sim "
                        "RViz is up)."),
        DeclareLaunchArgument(
            'rviz_config',
            default_value=os.path.join(nav_pkg, 'rviz', 'navigation.rviz'),
            description='RViz layout (map + costmaps + path + goal tool).'),
        DeclareLaunchArgument(
            'prepose', default_value='true',
            description="Before the base may move, tuck the arms into the navigation "
                        "pose (forearms folded straight DOWN, not flung 90 deg out) and "
                        "HOLD every Nav2 goal until they are down. nav_pose_setter latches "
                        "/mabel/body_ready; goal_relay + frontier_explorer wait on it. "
                        "Needs an arms backend (sim sim.launch.py / hardware.launch.py) "
                        "up to receive /arms/command. Set false to skip the pre-pose and "
                        "drive immediately."),

        SetParameter(name='use_sim_time', value=use_sim_time),

        # --- Navigation frame: base_link_nav = base_link rotated 180 deg about z -----
        # so its +x points along the robot's TRUE FRONT (base_link -X). Nav2 plans and
        # controls in this frame (robot_base_frame=base_link_nav in nav2_params.yaml),
        # so the STOCK MPPI forward-preference squares the FRONT to the path. The
        # cmd_vel_bridge converts twists/odom between this frame and the physical base.
        Node(
            package='tf2_ros', executable='static_transform_publisher',
            name='base_link_nav_tf', output='screen',
            arguments=['--x', '0', '--y', '0', '--z', '0', '--yaw', '3.14159265',
                       '--frame-id', 'base_link', '--child-frame-id', 'base_link_nav'],
            parameters=[{'use_sim_time': use_sim_time}],
        ),

        # --- Pre-navigation posture: fold the arms DOWN, then release the stack ----
        # Publishes the nav pose on /arms/command and latches /mabel/body_ready when
        # the arms have settled (or after a fail-safe timeout, so a missing arms
        # backend can never deadlock navigation). goal_relay + frontier_explorer hold
        # their goals until that flag is True.
        Node(
            package='mabel_navigation', executable='nav_pose_setter',
            name='nav_pose_setter', output='screen',
            parameters=[{'use_sim_time': use_sim_time}],
            condition=IfCondition(prepose),
        ),

        # --- Nav2 core servers -------------------------------------------------
        GroupAction([
            server('nav2_controller', 'controller_server', 'controller_server', cmd_remap),
            server('nav2_smoother', 'smoother_server', 'smoother_server', tf_remap),
            server('nav2_planner', 'planner_server', 'planner_server', tf_remap),
            server('nav2_behaviors', 'behavior_server', 'behavior_server', cmd_remap),
            server('nav2_bt_navigator', 'bt_navigator', 'bt_navigator', tf_remap,
                   extra={'default_nav_to_pose_bt_xml': bt_xml}),
            server('nav2_velocity_smoother', 'velocity_smoother', 'velocity_smoother', cmd_remap),
            server('nav2_collision_monitor', 'collision_monitor', 'collision_monitor', tf_remap),
            Node(
                package='nav2_lifecycle_manager', executable='lifecycle_manager',
                name='lifecycle_manager_navigation', output='screen',
                parameters=[{'autostart': autostart, 'node_names': LIFECYCLE_NODES,
                             'use_sim_time': use_sim_time}],
            ),
        ]),

        # --- Base bridge: Nav2 /cmd_vel -> tip-safe MabelSwerveCmd --------------
        # Output goes to /nav/mabel_cmd, NOT the canonical /mabel_cmd: the gateway's
        # plant is /mabel_cmd's only publisher and re-injects this twist through the
        # controller facade (so Nav2 is tip-gated like every teleop device, and the
        # operator's stick preempts it). Two publishers on /mabel_cmd fight and stall
        # the base — that is what this explicit topic prevents. Keep it equal to the
        # plant's MABEL_NAV_CMD_TOPIC.
        Node(
            package='mabel_navigation', executable='cmd_vel_bridge',
            name='cmd_vel_bridge', output='screen',
            parameters=[{'use_stamped': use_stamped, 'use_sim_time': use_sim_time,
                         'cmd_out_topic': '/nav/mabel_cmd'}],
        ),

        # --- RViz "2D Goal Pose" (/goal_pose) -> NavigateToPose action ---------
        Node(
            package='mabel_navigation', executable='goal_relay',
            name='goal_relay', output='screen',
            parameters=[{'use_sim_time': use_sim_time,
                         # Hold goals until the arms are down ONLY when the pre-pose
                         # node is running; else /mabel/body_ready never comes and a
                         # goal would hang forever.
                         'wait_for_body_ready': ParameterValue(prepose, value_type=bool)}],
        ),

        # --- Operator view: set goals with the RViz "Nav2 Goal" tool -----------
        Node(
            package='rviz2', executable='rviz2', name='rviz2', output='screen',
            arguments=['-d', rviz_config],
            parameters=[{'use_sim_time': use_sim_time}],
            condition=IfCondition(rviz),
        ),
    ])
