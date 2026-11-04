"""Autonomous frontier EXPLORATION for MABEL -- map an unknown space hands-free.

    ros2 launch mabel_navigation exploration.launch.py

This is navigation.launch.py with one extra node on top: ``frontier_explorer``.
That node watches the live SLAM map, repeatedly sends the nearest information-rich
frontier (free/unknown boundary) to Nav2 as a goal, and stops when no reachable
frontier remains -- i.e. the environment is fully mapped. Save the result with:

    ros2 run nav2_map_server map_saver_cli -f my_map

Because it shares the SAME Nav2 + bridge stack as manual navigation (and the SAME
backend-agnostic /map + /scan + map->odom contract), exploration runs identically
on the LiDAR 2D SLAM or the visual 3D SLAM, in sim or on the real robot. It is
independent of both the SLAM internals and the hardware: it only uses the public
``navigate_to_pose`` action and the ``/map`` topic.

EXPLORATION ONLY. Like navigation.launch.py, this starts no HAL, no physics, and
no SLAM. Bring those up first, in order, then launch this:
    1. HAL + sensors + odom + TF   (mabel_sim sim.launch.py  /  mabel_bringup hardware.launch.py)
    2. a SLAM backend              (mabel_slam lidar_slam.launch.py  /  visual_slam.launch.py)
    3. THIS -- exploration         (mabel_navigation exploration.launch.py)

Watch RViz: orange spheres are candidate frontiers, the green one is the current
goal; the run announces "EXPLORATION COMPLETE" when done.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetParameter
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    nav_pkg = get_package_share_directory('mabel_navigation')

    use_sim_time = LaunchConfiguration('use_sim_time')
    rviz = LaunchConfiguration('rviz')
    min_frontier_size = LaunchConfiguration('min_frontier_size')
    gain_scale = LaunchConfiguration('gain_scale')
    planner_frequency = LaunchConfiguration('planner_frequency')
    return_to_start = LaunchConfiguration('return_to_start')
    prepose = LaunchConfiguration('prepose')
    save_map = LaunchConfiguration('save_map')
    map_save_path = LaunchConfiguration('map_save_path')
    save_mesh = LaunchConfiguration('save_mesh')

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time', default_value='true',
            description='Run on the sim /clock. Set false on the real robot.'),
        DeclareLaunchArgument(
            'rviz', default_value='false',
            description="Open a SEPARATE navigation RViz (map + path + frontier markers). "
                        "DEFAULT false: the sim's RViz already shows the map, frontiers, "
                        "plan and goal, so exploration builds ON TOP of that ONE window. "
                        "Set true only for a standalone viewer."),
        DeclareLaunchArgument(
            'min_frontier_size', default_value='12',
            description='Smallest frontier (grid cells) worth visiting; raise to '
                        'ignore noise/clutter, lower to chase thin gaps.'),
        DeclareLaunchArgument(
            'gain_scale', default_value='0.5',
            description='Information-gain weight: 0 = always nearest frontier, '
                        'higher = prefer large (high-payoff) frontiers even if farther.'),
        DeclareLaunchArgument(
            'planner_frequency', default_value='0.5',
            description='Exploration replanning rate (Hz).'),
        DeclareLaunchArgument(
            'return_to_start', default_value='false',
            description='Drive back to the start pose once mapping is complete.'),
        DeclareLaunchArgument(
            'prepose', default_value='true',
            description="Fold the arms into the navigation pose (forearms DOWN, not "
                        "flung 90 deg out) and hold ALL exploration goals until they "
                        "are down. Forwarded to navigation.launch.py (which runs "
                        "nav_pose_setter + latches /mabel/body_ready) and gates the "
                        "frontier explorer. Set false to explore immediately."),
        DeclareLaunchArgument(
            'save_map', default_value='true',
            description="Auto-save the finished map (PGM + YAML, nav2 map_server "
                        "format) when exploration completes. Set false to skip."),
        DeclareLaunchArgument(
            'map_save_path', default_value='~/mabel_map',
            description="Basename (no extension, '~' expanded) for the auto-saved map; "
                        "writes <path>.pgm + <path>.yaml. Point it at the repo maps dir "
                        "to keep it, e.g. map_save_path:=$PWD/src/mabel_navigation/maps/explored."),
        DeclareLaunchArgument(
            'save_mesh', default_value='true',
            description="Also save the nvblox 3D reconstruction on completion: "
                        "<map_save_path>.ply (mesh) + .nvblx (relocalization map), via "
                        "nvblox's save_ply/save_map services (shelled through slam_env.sh). "
                        "Needs nvblox running; set false for 2D-only (e.g. slam_toolbox)."),

        SetParameter(name='use_sim_time', value=use_sim_time),

        # The navigation stack (Nav2 + cmd_vel_bridge + optional RViz). No SLAM --
        # bring a SLAM backend up yourself before launching this (see module docstring).
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(nav_pkg, 'launch', 'navigation.launch.py')),
            launch_arguments={
                'use_sim_time': use_sim_time,
                'rviz': rviz,
                'prepose': prepose,
            }.items()),

        # The explorer: drives Nav2 to the frontiers until the map is complete.
        Node(
            package='mabel_navigation', executable='frontier_explorer',
            name='frontier_explorer', output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'min_frontier_size': ParameterValue(min_frontier_size, value_type=int),
                'gain_scale': ParameterValue(gain_scale, value_type=float),
                'planner_frequency': ParameterValue(planner_frequency, value_type=float),
                'return_to_start': ParameterValue(return_to_start, value_type=bool),
                # Hold frontier goals until the arms are down ONLY when the pre-pose
                # node is running (navigation.launch.py runs it under the same arg);
                # else /mabel/body_ready never comes and exploration would never start.
                'wait_for_body_ready': ParameterValue(prepose, value_type=bool),
                'save_map_on_complete': ParameterValue(save_map, value_type=bool),
                'map_save_path': map_save_path,
                'save_mesh_on_complete': ParameterValue(save_mesh, value_type=bool),
            }],
        ),
    ])
