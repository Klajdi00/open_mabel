"""2D LiDAR SLAM for MABEL via slam_toolbox -- the GPU-free, base-navigation SLAM.

This is the LiDAR alternative to the Isaac ROS visual stack (visual_slam.launch.py =
cuVSLAM + nvblox). It runs ONE node -- slam_toolbox's async_slam_toolbox_node, the
de-facto ROS 2 standard for 2D laser SLAM (pose-graph + loop closure, the default
SLAM in Nav2). We use slam_toolbox rather than gmapping (unmaintained ROS 1 port,
no graph optimisation) or ORB-SLAM (that's *visual* SLAM, not LiDAR), and rather
than hand-rolling a scan matcher.

Data flow (no cameras, no GPU):

  /scan (LaserScan, frame lidar_link)  ->  slam_toolbox  ->  /map (OccupancyGrid)
  TF odom->base_link (swerve_odometry) ->               ->  TF map->odom

The /scan topic + frames are IDENTICAL in sim (mabel_sim) and on the real robot, so
this file does not distinguish sim vs real -- like visual_slam.launch.py it consumes
the same topics in both. The ONLY difference is the clock (use_sim_time): sim time in
sim, wall clock on the robot. That is detected AUTOMATICALLY by probing the graph for
a /clock publisher, so you pass nothing:

    ros2 launch mabel_sim   sim.launch.py sensor_rate:=30   # /scan, wheel odom, /clock
    source ros/scripts/slam_env.sh                     # apt ROS + CycloneDDS/domain 0
    ros2 launch mabel_slam  lidar_slam.launch.py            # auto clock, sim or real

slam_toolbox is an apt /opt/ros package, so (like the Isaac visual stack) source
`scripts/slam_env.sh` first: it puts apt ROS + the mabel_slam overlay on PATH and forces
CycloneDDS on domain 0 to MATCH the conda sim. If the launching shell's RMW/domain do
NOT match the sim, slam_toolbox sees no /scan and no /clock -> it logs clock=wall, never
builds a /map, and the stale latched /map from a prior run lingers in RViz (the "old map
won't go away / won't launch again" symptom).

WARNING: do NOT run this together with visual_slam.launch.py (cuVSLAM). Both publish
map->odom and will fight over the TF tree -- choose ONE map->odom source.

Save the finished map with:

    ros2 run nav2_map_server map_saver_cli -f ~/mabel_map        # or:
    ros2 service call /slam_toolbox/save_map slam_toolbox/srv/SaveMap "{name: {data: 'mabel_map'}}"
"""
import os
import subprocess
import time

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, EmitEvent, LogInfo,
                            OpaqueFunction, RegisterEventHandler)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessStart
from launch.events import matches_action
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LifecycleNode, Node
from launch_ros.event_handlers import OnStateTransition
from launch_ros.events.lifecycle import ChangeState
from lifecycle_msgs.msg import Transition

from mabel_slam.sim_detect import resolve_is_sim


# LiDAR-SLAM-OWNED processes to reap before a new launch. A leftover slam_toolbox from
# a prior (crashed / Ctrl-C'd) run keeps publishing map->odom AND latches its old /map
# (Transient Local) into the shared RViz -- so a "fresh" launch would otherwise show the
# PREVIOUS map. Killing it guarantees we start from the current scan with a NEW map.
# Matched by install-lib path / RViz config name so we only ever hit a prior SLAM
# instance, never this launch (its node has not spawned when the reap runs).
_LIDAR_PROC_PATTERNS = [
    "async_slam_toolbox_node",
    "lib/mabel_slam/map_relay",       # stale relay would keep latching the OLD /map
    "rviz2.*lidar_slam.rviz",
]


def _reap_prior_slam(patterns, label):
    """Kill any lingering nodes matching `patterns` so the new SLAM owns map->odom and
    starts from a clean, empty map (no stale latched /map in the reused RViz)."""
    reaped = False
    for pat in patterns:
        # pkill returns 0 only if it matched + signalled at least one process.
        if subprocess.run(["pkill", "-9", "-f", pat]).returncode == 0:
            reaped = True
    if reaped:
        time.sleep(0.5)   # let the OS release DDS endpoints before we re-publish
        print(f"[{label}] reaped a previous SLAM instance -> starting a fresh map.")


def launch_setup(ctx, *args, **kwargs):
    params_file = LaunchConfiguration('params_file')
    scan_topic = LaunchConfiguration('scan_topic')
    autostart = LaunchConfiguration('autostart')

    # Start from a CLEAN slate: reap any prior slam_toolbox so its stale, latched /map
    # cannot leak into the reused RViz -- we map the current scan into a brand-new map.
    if LaunchConfiguration('fresh_map').perform(ctx).lower() in ('true', '1', 'yes'):
        _reap_prior_slam(_LIDAR_PROC_PATTERNS, 'lidar_slam')

    # The only sim/real difference is the clock; auto-detected (or forced via the arg).
    use_sim_time = resolve_is_sim(LaunchConfiguration('use_sim_time').perform(ctx))

    # In ROS 2 Jazzy slam_toolbox's async node is a LIFECYCLE node: it boots in the
    # 'unconfigured' state and publishes NOTHING (no /map, no map->odom) until it is
    # transitioned configure -> activate. A plain launch_ros Node leaves it idle
    # forever, so we drive the transitions ourselves (autostart), mirroring
    # slam_toolbox's own online_async_launch.py.
    # Secondary-map support (for running ALONGSIDE visual_slam without a map->odom
    # fight): map_frame overrides the TF parent slam_toolbox publishes (default 'map'
    # -> the primary; 'map_lidar' -> a secondary tree), and map_topic remaps the
    # OccupancyGrid output. With defaults this is the standalone primary lidar SLAM
    # (unchanged); set both to run as the non-conflicting secondary.
    map_frame = LaunchConfiguration('map_frame').perform(ctx) or 'map'
    map_topic = LaunchConfiguration('map_topic').perform(ctx) or '/map_lidar'
    remaps = [('/scan', scan_topic)]
    if map_topic != '/map':
        # Provenance naming: the LiDAR grid lives on its own topic (default
        # /map_lidar) so `ros2 topic list` always shows WHICH SLAM produced the
        # map (the visual stack's slice_to_occupancy publishes /map_visual).
        remaps.append(('/map', map_topic))
        remaps.append(('/map_metadata', map_topic + '_metadata'))
    slam_node = LifecycleNode(
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='slam_toolbox',
        namespace='',
        output='screen',
        parameters=[
            params_file,
            {'use_sim_time': use_sim_time,
             'map_frame': map_frame},   # later dict overrides the yaml
        ],
        remappings=remaps,
    )

    # Canonical /map relay — ONLY when this slam_toolbox is the PRIMARY tree
    # (map_frame 'map') with its grid on a provenance topic: every consumer
    # (Nav2/RViz/apps) keeps reading /map. A SECONDARY instance running alongside
    # visual_slam (map_frame:=map_lidar) must NOT feed /map — the visual stack
    # owns it there.
    relay_nodes = []
    if map_topic != '/map' and map_frame == 'map':
        relay_nodes.append(Node(
            package='mabel_slam', executable='map_relay', name='map_relay',
            output='screen',
            parameters=[{'source_topic': map_topic,
                         'use_sim_time': use_sim_time}]))

    # unconfigured -> inactive
    configure_event = EmitEvent(
        event=ChangeState(
            lifecycle_node_matcher=matches_action(slam_node),
            transition_id=Transition.TRANSITION_CONFIGURE,
        ),
        condition=IfCondition(autostart),
    )

    # inactive -> active (once configuring completes)
    activate_event = RegisterEventHandler(
        OnStateTransition(
            target_lifecycle_node=slam_node,
            start_state='configuring',
            goal_state='inactive',
            entities=[
                LogInfo(msg='[lidar_slam] slam_toolbox configured; activating.'),
                EmitEvent(event=ChangeState(
                    lifecycle_node_matcher=matches_action(slam_node),
                    transition_id=Transition.TRANSITION_ACTIVATE,
                )),
            ],
        ),
        condition=IfCondition(autostart),
    )

    # Emit CONFIGURE only AFTER the node process has started -- otherwise the event
    # races ahead of slam_toolbox creating its lifecycle service and is silently
    # LOST (node stays 'unconfigured' forever: no /map, no map->odom). OnProcessStart
    # makes autostart deterministic under load.
    configure_on_start = RegisterEventHandler(
        OnProcessStart(target_action=slam_node, on_start=[configure_event]),
        condition=IfCondition(autostart),
    )

    actions = [
        LogInfo(msg=f"[lidar_slam] clock={'sim (/clock)' if use_sim_time else 'wall'}"),
        slam_node,
        *relay_nodes,                  # provenance topic -> canonical /map (primary only)
        configure_on_start,
        activate_event,
    ]

    # By DEFAULT this launch does NOT open its own RViz -- it renders the 2D /map into
    # the ALREADY-RUNNING window (mabel_sim sim_sensors.rviz / mabel_bringup
    # hardware.rviz both carry a "SLAM Map (2D)" display on /map). Pass rviz:=true only
    # if you want the standalone lidar_slam.rviz instead.
    if LaunchConfiguration('rviz').perform(ctx).lower() in ('true', '1', 'yes'):
        from ament_index_python.packages import get_package_share_directory
        cfg = LaunchConfiguration('rviz_config').perform(ctx) or os.path.join(
            get_package_share_directory('mabel_slam'), 'rviz', 'lidar_slam.rviz')
        actions.append(Node(
            package='rviz2', executable='rviz2', name='rviz2',
            arguments=['-d', cfg], output='screen',
            parameters=[{'use_sim_time': use_sim_time}]))
    return actions


def generate_launch_description():
    pkg = get_package_share_directory('mabel_slam')
    default_params = os.path.join(pkg, 'config', 'lidar_slam.yaml')

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time', default_value='auto',
            description='auto (default) = probe /clock to use sim time in sim and the '
                        'wall clock on the robot, automatically. Force with true/false.'),
        DeclareLaunchArgument(
            'params_file', default_value=default_params,
            description='slam_toolbox parameter YAML (frames/topic/tuning).'),
        DeclareLaunchArgument(
            'scan_topic', default_value='/scan',
            description='Input LaserScan topic (matches mabel_sim sensor_config).'),
        DeclareLaunchArgument(
            'map_frame', default_value='map',
            description="TF parent frame slam_toolbox publishes ('map' = primary; set "
                        "'map_lidar' to run as a secondary alongside visual_slam without "
                        "fighting over map->odom)."),
        DeclareLaunchArgument(
            'map_topic', default_value='/map_lidar',
            description="OccupancyGrid output topic. Default '/map_lidar' = provenance "
                        "naming; as the PRIMARY (map_frame 'map') a map_relay feeds the "
                        "canonical /map from it, so consumers are unchanged. As a "
                        "SECONDARY alongside visual_slam (map_frame:=map_lidar) no relay "
                        "runs — visual owns /map."),
        DeclareLaunchArgument(
            'autostart', default_value='true',
            description='Auto-configure+activate the slam_toolbox lifecycle node. '
                        'Set false to drive transitions manually / via a Nav2 '
                        'lifecycle_manager.'),
        DeclareLaunchArgument('rviz', default_value='false',
                              description='false (default): render the 2D /map into the '
                                          'EXISTING RViz (sim_sensors.rviz / hardware.rviz '
                                          'have a SLAM Map display). true: open the '
                                          'standalone lidar_slam.rviz instead.'),
        DeclareLaunchArgument('rviz_config', default_value='',
                              description='Override RViz config (empty -> lidar_slam.rviz).'),
        DeclareLaunchArgument('fresh_map', default_value='true',
                              description='true (default): reap any prior slam_toolbox '
                                          'before starting so we map from the current scan '
                                          'into a NEW map (no stale latched /map). false: '
                                          'leave existing nodes alone.'),
        OpaqueFunction(function=launch_setup),
    ])
