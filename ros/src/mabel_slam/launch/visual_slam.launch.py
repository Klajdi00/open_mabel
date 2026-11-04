"""GPU visual SLAM for MABEL: cuVSLAM (localize) + nvblox (3D mesh) -- native, no Docker.

Pure SLAM. Starts NO camera driver, NO robot, NO RViz -- it only SUBSCRIBES to
camera topics that something else already publishes (mabel_sim in sim, mabel_sensors
on the real robot) and publishes the SLAM products. Sim and real use the SAME topic
contract, so this file does not distinguish them; the ONLY difference is the clock
(use_sim_time), auto-detected by probing the graph for a /clock publisher:

    ros2 launch mabel_slam visual_slam.launch.py                    # auto clock, sim or real
    ros2 launch mabel_slam visual_slam.launch.py env:=manipulation  # ZED head, fine mesh
    ros2 launch mabel_slam visual_slam.launch.py cuvslam:=false      # nvblox-only (wheel odom)

Two envs pick the camera + nvblox detail:
  navigation   (default) -- FRONT RealSense D435i IR stereo (/camera/front/*). Coarse
                            voxels + a 2D ESDF slice -> /map for Nav2. "Map the room."
  manipulation           -- ZED Mini head (/camera/zed/*). Fine voxels, detailed mesh,
                            no nav slice. "Scan the workspace."

Graph: stereo pair -> cuVSLAM (map->odom localization); depth+color -> nvblox
(TSDF -> /nvblox_node/mesh). Composed in ONE component_container_mt so Isaac ROS
passes GPU buffers by pointer (NITROS zero-copy).

TF ownership: publish_odom_to_base_tf:=false (DEFAULT) -- the base wheel odom owns
odom->base_link, cuVSLAM only corrects map->odom on top. :=true -- cuVSLAM is the
sole odometry (no wheel odom running).

NOTE: requires Isaac ROS installed natively (ros-jazzy-isaac-ros-visual-slam +
-nvblox). Those are apt /opt/ros packages and are NOT visible in the RoboStack
*conda* env that runs the sim -- launching this from the conda shell silently runs
WITHOUT nvblox (no 3D mesh). Source `scripts/slam_env.sh` FIRST: it sets up apt ROS +
the mabel_slam overlay + CycloneDDS/domain 0 so nvblox is found AND talks to the conda
sim over DDS. With the sim running, the nvblox mesh is re-emitted as a TRIANGLE_LIST
Marker on /nvblox_node/mesh_marker (shaded surfaces) + a PointCloud2 on
/nvblox_node/mesh_pointcloud, BOTH of which the sim's RViz (sim_sensors.rviz) and the
hardware bringup RViz (hardware.rviz) already display. Do NOT run together with
lidar_slam.launch.py -- both publish map->odom.

    source scripts/slam_env.sh
    ros2 launch mabel_slam visual_slam.launch.py use_sim_time:=true   # in sim
"""
import os
import subprocess
import time

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode

from mabel_slam.sim_detect import detect_has_publisher, resolve_is_sim


# Visual-SLAM-OWNED processes to reap before a new launch. A leftover cuVSLAM/nvblox
# from a prior (crashed / Ctrl-C'd) run keeps publishing map->odom AND latches its old
# nvblox mesh (Transient Local) into the shared RViz -- so a "fresh" launch would show
# the PREVIOUS reconstruction. Killing them guarantees nvblox rebuilds the mesh from the
# current frames into a NEW, empty TSDF. Matched by container/install-lib path / RViz
# config name so we only hit a prior SLAM instance, never this launch.
_VSLAM_PROC_PATTERNS = [
    "visual_slam_container",                       # composition=true (the one container)
    "lib/isaac_ros_visual_slam/isaac_ros_visual_slam",  # composition=false standalone
    "lib/nvblox_ros/nvblox_node",
    "lib/mabel_slam/mesh_to_pointcloud",
    "lib/mabel_slam/slice_to_occupancy",
    "lib/mabel_slam/map_relay",       # stale relay would keep latching the OLD /map
    "rviz2.*visual_slam.rviz",
]


def _reap_prior_slam(patterns, label):
    """Kill any lingering nodes matching `patterns` so the new SLAM owns map->odom and
    nvblox rebuilds from an empty map (no stale latched mesh in the reused RViz)."""
    reaped = False
    for pat in patterns:
        # pkill returns 0 only if it matched + signalled at least one process.
        if subprocess.run(["pkill", "-9", "-f", pat]).returncode == 0:
            reaped = True
    if reaped:
        time.sleep(0.5)   # let the OS release the GPU + DDS endpoints before re-launch
        print(f"[{label}] reaped a previous SLAM instance -> starting a fresh map.")


# Per-env sensor contract + nvblox tuning. These topic NAMES are identical in sim
# (mabel_sim) and on the real robot (mabel_sensors) -- the unified contract.
ENVS = {
    'navigation': {
        'left_image':  '/camera/front/infra1/image_rect_raw',
        'left_info':   '/camera/front/infra1/camera_info',
        'right_image': '/camera/front/infra2/image_rect_raw',
        'right_info':  '/camera/front/infra2/camera_info',
        'optical_frames': ['front_infra1_optical_frame', 'front_infra2_optical_frame'],
        'imu_topic': '/camera/front/imu',            # D435i motion module (if published)
        'imu_frame': 'front_imu_optical_frame',
        'depth_image': '/camera/front/depth/image_rect_raw',
        'depth_info':  '/camera/front/depth/camera_info',
        'color_image': '/camera/front/color/image_raw',
        'color_info':  '/camera/front/color/camera_info',
        'nvblox_config': 'nvblox_navigation.yaml',
        'occupancy': True,    # ESDF 2D slice -> /map (navigation)
    },
    'manipulation': {
        'left_image':  '/camera/zed/left/image_raw',
        'left_info':   '/camera/zed/left/camera_info',
        'right_image': '/camera/zed/right/image_raw',
        'right_info':  '/camera/zed/right/camera_info',
        'optical_frames': ['zed_left_camera_optical_frame', 'zed_right_camera_optical_frame'],
        'imu_topic': '',                             # bridge ZED driver publishes no IMU
        'imu_frame': '',
        'depth_image': '/camera/zed/depth/image_rect_raw',
        'depth_info':  '/camera/zed/depth/camera_info',
        'color_image': '/camera/zed/left/image_raw',
        'color_info':  '/camera/zed/left/camera_info',
        'nvblox_config': 'nvblox_manipulation.yaml',
        'occupancy': False,
    },
}


def _bool(ctx, name):
    return LaunchConfiguration(name).perform(ctx).lower() in ('true', '1', 'yes')


def _arg_or(ctx, name, default):
    val = LaunchConfiguration(name).perform(ctx)
    return val if val else default


def launch_setup(ctx, *args, **kwargs):
    env = LaunchConfiguration('env').perform(ctx)
    if env not in ENVS:
        raise RuntimeError(f"unknown env '{env}'; choose 'navigation' or 'manipulation'")
    p = ENVS[env]

    # Start from a CLEAN slate: reap any prior cuVSLAM/nvblox so a stale, latched mesh
    # cannot leak into the reused RViz -- nvblox rebuilds the current view into a NEW map.
    if LaunchConfiguration('fresh_map').perform(ctx).lower() in ('true', '1', 'yes'):
        _reap_prior_slam(_VSLAM_PROC_PATTERNS, 'visual_slam')

    use_sim_time = resolve_is_sim(LaunchConfiguration('use_sim_time').perform(ctx))
    base_frame = LaunchConfiguration('base_frame').perform(ctx)
    publish_odom_tf = _bool(ctx, 'publish_odom_to_base_tf')
    enable_mapping = _bool(ctx, 'enable_mapping')
    visualization = _bool(ctx, 'visualization')
    run_nvblox = _bool(ctx, 'nvblox')
    run_cuvslam = _bool(ctx, 'cuvslam')

    left_image = _arg_or(ctx, 'left_image', p['left_image'])
    left_info = _arg_or(ctx, 'left_info', p['left_info'])
    right_image = _arg_or(ctx, 'right_image', p['right_image'])
    right_info = _arg_or(ctx, 'right_info', p['right_info'])
    depth_image = _arg_or(ctx, 'depth_image', p['depth_image'])
    depth_info = _arg_or(ctx, 'depth_info', p['depth_info'])
    color_image = _arg_or(ctx, 'color_image', p['color_image'])
    color_info = _arg_or(ctx, 'color_info', p['color_info'])

    composition = _bool(ctx, 'composition')
    pkg = get_package_share_directory('mabel_slam')
    nvblox_cfg = os.path.join(pkg, 'config', p['nvblox_config'])

    # IMU fusion (cuVSLAM tracking_mode=1, VIO): stereo-only cuVSLAM drifts badly
    # under aggressive/handheld 6-DOF motion — the D435i gyro+accel anchor it.
    # 'auto' (default) probes the graph: fuse ONLY when the camera driver actually
    # publishes the env's IMU topic (a VIO tracker with a silent IMU topic never
    # tracks at all, so this must never turn on blind).
    imu_arg = LaunchConfiguration('imu').perform(ctx).strip().lower()
    imu_topic = p['imu_topic']
    if imu_arg in ('true', '1', 'yes'):
        use_imu = bool(imu_topic)
    elif imu_arg in ('false', '0', 'no'):
        use_imu = False
    else:                                   # auto
        use_imu = bool(imu_topic) and detect_has_publisher(imu_topic)

    # Frame-stamp jitter tolerance: the sim renders cameras on the SAME GPU as
    # cuVSLAM, so frames arrive very jittery under load -> generous 500 ms. The
    # real D435 produces hardware-steady frames; 100 ms keeps real stalls (e.g. a
    # blocked bridge camera thread) VISIBLE in the log instead of silently eaten.
    jitter_ms = LaunchConfiguration('image_jitter_threshold_ms').perform(ctx)
    jitter_ms = float(jitter_ms) if jitter_ms else (500.0 if use_sim_time else 100.0)

    info = [LogInfo(msg=f"[visual_slam] env={env} clock="
                        f"{'sim (/clock)' if use_sim_time else 'wall'} "
                        f"cuvslam={run_cuvslam} nvblox={run_nvblox} "
                        f"imu_fusion={use_imu} jitter_thresh={jitter_ms:.0f}ms "
                        f"{'composed (NITROS zero-copy)' if composition else 'standalone nodes'}")]

    # Node specs (package, plugin, standalone-executable, params, remaps) built once,
    # then run EITHER as composable nodes in one container (composition=true, default:
    # Isaac NITROS passes GPU buffers by pointer -- fastest) OR as separate standalone
    # processes (composition=false: plain native nodes, no shared container, but every
    # frame is copied over DDS instead of zero-copy -> heavier).
    specs = []
    if run_cuvslam:
        vslam_params = {
            'use_sim_time': use_sim_time,
            'num_cameras': 2,
            'min_num_images': 2,
            'enable_image_denoising': False,
            'rectified_images': True,
            'enable_localization_n_mapping': enable_mapping,
            'enable_slam_visualization': visualization,
            'enable_landmarks_view': visualization,
            'enable_observations_view': visualization,
            'map_frame': 'map',
            'odom_frame': 'odom',
            'base_frame': base_frame,
            'camera_optical_frames': p['optical_frames'],
            'publish_odom_to_base_tf': publish_odom_tf,
            'publish_map_to_odom_tf': True,
            'image_jitter_threshold_ms': jitter_ms,
        }
        vslam_remaps = [
            ('visual_slam/image_0', left_image),
            ('visual_slam/camera_info_0', left_info),
            ('visual_slam/image_1', right_image),
            ('visual_slam/camera_info_1', right_info),
        ]
        if use_imu:
            # VIO: fuse the camera IMU (NVIDIA reference D435i config; the noise
            # density / random-walk defaults already match the D435i's BMI055).
            vslam_params.update({
                'tracking_mode': 1,                   # 1 = VIO (IMU fusion)
                'imu_frame': p['imu_frame'],
                'gyro_noise_density': 0.000244,
                'gyro_random_walk': 0.000019393,
                'accel_noise_density': 0.001862,
                'accel_random_walk': 0.003,
                'calibration_frequency': 200.0,
            })
            vslam_remaps.append(('visual_slam/imu', imu_topic))
        specs.append(dict(
            name='visual_slam',
            package='isaac_ros_visual_slam',
            plugin='nvidia::isaac_ros::visual_slam::VisualSlamNode',
            executable='isaac_ros_visual_slam',
            parameters=[vslam_params],
            remappings=vslam_remaps,
        ))
    if run_nvblox:
        specs.append(dict(
            name='nvblox_node',
            package='nvblox_ros',
            plugin='nvblox::NvbloxNode',
            executable='nvblox_node',
            parameters=[
                nvblox_cfg,
                {
                    'use_sim_time': use_sim_time,
                    'num_cameras': 1,
                    'map_clearing_frame_id': base_frame,
                    'esdf_slice_bounds_visualization_attachment_frame_id': base_frame,
                    'workspace_height_bounds_visualization_attachment_frame_id': base_frame,
                },
            ],
            remappings=[
                ('camera_0/depth/image', depth_image),
                ('camera_0/depth/camera_info', depth_info),
                ('camera_0/color/image', color_image),
                ('camera_0/color/camera_info', color_info),
            ],
        ))

    if composition:
        nodes = info + [ComposableNodeContainer(
            name='visual_slam_container', namespace='',
            package='rclcpp_components', executable='component_container_mt',
            composable_node_descriptions=[
                ComposableNode(name=s['name'], package=s['package'],
                               plugin=s['plugin'], parameters=s['parameters'],
                               remappings=s['remappings']) for s in specs],
            output='screen',
        )]
    else:
        nodes = info + [Node(
            package=s['package'], executable=s['executable'], name=s['name'],
            parameters=s['parameters'], remappings=s['remappings'], output='screen')
            for s in specs]

    # nvblox post-processing.
    if run_nvblox:
        # nvblox's mesh (/nvblox_node/mesh) can only be drawn by the apt-only
        # nvblox_rviz_plugin, which the conda sim/hardware RViz can't load. This node
        # re-emits it in plugin-free forms that RViz's stock displays show in that same
        # window: a TRIANGLE_LIST Marker (/nvblox_node/mesh_marker -- shaded surfaces,
        # the default "nvblox Mesh (3D)" display) plus a PointCloud2 fallback
        # (/nvblox_node/mesh_pointcloud). This is the visual_slam 3D product.
        nodes.append(Node(
            package='mabel_slam', executable='mesh_to_pointcloud',
            name='mesh_to_pointcloud', output='screen',
            parameters=[{'use_sim_time': use_sim_time}]))
        # The 2D ESDF slice -> /map OccupancyGrid is OFF by default: visual_slam is the
        # 3D-mesh stack, so RViz shows only the mesh (no 2D grid under it). Turn it on
        # with nav_2d_map:=true ONLY when you want a 2D costmap for Nav2 from the camera
        # (the navigation env). lidar_slam remains the usual 2D /map source.
        if p['occupancy'] and _bool(ctx, 'nav_2d_map'):
            # Provenance naming: the VISUAL grid lives on /map_visual so `ros2 topic
            # list` always shows WHICH SLAM produced the map (slam_toolbox publishes
            # /map_lidar). map_relay feeds the canonical /map all consumers read.
            nodes.append(Node(
                package='mabel_slam', executable='slice_to_occupancy',
                name='slice_to_occupancy', output='screen',
                parameters=[{'use_sim_time': use_sim_time}],
                remappings=[('/map', '/map_visual'),
                            ('/map_costmap', '/map_visual_costmap')]))
            nodes.append(Node(
                package='mabel_slam', executable='map_relay', name='map_relay',
                output='screen',
                parameters=[{'source_topic': '/map_visual',
                             'use_sim_time': use_sim_time}]))

    # By DEFAULT this launch does NOT open its own RViz -- it renders the nvblox 3D mesh
    # into the ALREADY-RUNNING window (mabel_sim sim_sensors.rviz / mabel_bringup
    # hardware.rviz both carry a "nvblox Mesh (3D)" display on /nvblox_node/
    # mesh_pointcloud). Pass rviz:=true for the standalone visual_slam.rviz instead.
    if _bool(ctx, 'rviz'):
        rviz_cfg = _arg_or(ctx, 'rviz_config',
                           os.path.join(pkg, 'rviz', 'visual_slam.rviz'))
        nodes.append(Node(
            package='rviz2', executable='rviz2', name='rviz2',
            arguments=['-d', rviz_cfg], output='screen',
            parameters=[{'use_sim_time': use_sim_time}]))
    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'env', default_value='navigation',
            choices=['navigation', 'manipulation'],
            description="navigation = front RealSense IR stereo + 2D nav slice. "
                        "manipulation = ZED Mini, fine detailed 3D mesh."),
        DeclareLaunchArgument(
            'use_sim_time', default_value='auto',
            description="auto = probe /clock (sim time in sim, wall clock on robot). "
                        "Force with true/false."),
        DeclareLaunchArgument('base_frame', default_value='base_link'),
        DeclareLaunchArgument(
            'publish_odom_to_base_tf', default_value='false',
            description='false: wheel odom owns odom->base_link, cuVSLAM adds map->odom. '
                        'true: cuVSLAM is the sole odometry.'),
        DeclareLaunchArgument('enable_mapping', default_value='true',
                              description='cuVSLAM loop-closure mapping (false = VIO-only).'),
        DeclareLaunchArgument(
            'imu', default_value='auto',
            description="cuVSLAM IMU fusion (tracking_mode=1 VIO). auto = fuse iff the "
                        "env's IMU topic has a live publisher (D435i: /camera/front/imu); "
                        "true/false force. Stereo-only cuVSLAM drifts under handheld "
                        "6-DOF motion — fuse whenever the IMU is available."),
        DeclareLaunchArgument(
            'image_jitter_threshold_ms', default_value='',
            description='Frame-delta warn threshold. Empty (default) = 500 in sim '
                        '(GPU-contended rendering), 100 on the real camera.'),
        DeclareLaunchArgument('nvblox', default_value='true',
                              description='Run nvblox 3D reconstruction.'),
        DeclareLaunchArgument(
            'nav_2d_map', default_value='false',
            description='false (default): visual_slam publishes only the 3D mesh (no 2D '
                        '/map grid in RViz). true: also emit the nvblox ESDF slice as a '
                        '2D /map OccupancyGrid for Nav2 (navigation env).'),
        DeclareLaunchArgument('cuvslam', default_value='true',
                              description='Run cuVSLAM localization (false = nvblox-only, '
                                          'uses wheel odom; use in sim where cuVSLAM cannot '
                                          'track the untextured simulated IR).'),
        DeclareLaunchArgument('visualization', default_value='false',
                              description='Publish cuVSLAM landmark/observation clouds (debug).'),
        DeclareLaunchArgument('rviz', default_value='false',
                              description='false (default): render the nvblox 3D mesh into '
                                          'the EXISTING RViz (sim_sensors.rviz / '
                                          'hardware.rviz have an nvblox Mesh display). '
                                          'true: open the standalone visual_slam.rviz.'),
        DeclareLaunchArgument('rviz_config', default_value='',
                              description='Override RViz config (empty -> visual_slam.rviz).'),
        DeclareLaunchArgument('fresh_map', default_value='true',
                              description='true (default): reap any prior cuVSLAM/nvblox '
                                          'before starting so the mesh rebuilds from the '
                                          'current frames into a NEW map (no stale latched '
                                          'mesh). false: leave existing nodes alone.'),
        DeclareLaunchArgument(
            'composition', default_value='true',
            description='true (default): run cuVSLAM + nvblox as composable nodes in one '
                        'process (Isaac NITROS zero-copy, fastest). false: run them as '
                        'separate standalone native nodes (no component container; simpler '
                        'process layout, but frames are copied over DDS -> heavier).'),
        # Topic overrides (empty -> env default) for non-standard camera contracts.
        DeclareLaunchArgument('left_image', default_value=''),
        DeclareLaunchArgument('left_info', default_value=''),
        DeclareLaunchArgument('right_image', default_value=''),
        DeclareLaunchArgument('right_info', default_value=''),
        DeclareLaunchArgument('depth_image', default_value=''),
        DeclareLaunchArgument('depth_info', default_value=''),
        DeclareLaunchArgument('color_image', default_value=''),
        DeclareLaunchArgument('color_info', default_value=''),
        OpaqueFunction(function=launch_setup),
    ])
