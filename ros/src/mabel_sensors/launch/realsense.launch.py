"""Front Intel RealSense D435i for MABEL -- the REAL drop-in for the sim front camera.

Launched as node 'front' in namespace 'camera', so topics come out as
``/camera/front/<stream>`` and (with ``camera_name:=front`` in the config) frames
as ``front_<stream>_optical_frame`` -- EXACTLY what ``mabel_sim`` publishes. The
same cuVSLAM + nvblox (``mabel_slam``) therefore runs unchanged on real hardware;
``robot.launch.py use_sim:=false`` and ``use_sim:=true`` feed it identical topics.

TF: realsense2_camera publishes ``front_link -> front_*_optical_frame`` internally.
The ``base_link -> front_link`` MOUNT transform (where the camera sits on the
chassis) is published here as a static transform, so the camera data lands in the
robot frame for cuVSLAM/nvblox. Its default matches the sim camera exactly:
mabel_full.xml body_front_cam is at chassis (-0.25, 0, 0.13) looking down robot
-X (xyaxes 0 1 0  0 0 1), and base_link == mabel_chassis (identity base_joint), so
the mount is xyz=(-0.25, 0, 0.13), rpy=(0, 0, pi). Override mount_* to re-calibrate
without touching the auto-generated URDF.

    ros2 launch mabel_sensors realsense.launch.py                 # IMU off (default)
    ros2 launch mabel_sensors realsense.launch.py enable_imu:=true
"""
import os
import platform

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    # The realsense2_camera XYZ pointcloud filter is named DIFFERENTLY per CPU:
    # 'pointcloud__neon_' on the ARM/Jetson robot build (NEON SIMD), plain
    # 'pointcloud' on the x86 dev box. The shared realsense_front.yaml hard-codes
    # the ARM 'pointcloud__neon_.*' keys, so on x86 the cloud would never enable.
    # Force the arch-correct filter on here so /camera/front/depth/points streams
    # on BOTH the dev box and the robot.
    _arm = platform.machine().lower() in ('aarch64', 'arm64')
    pc = 'pointcloud__neon_' if _arm else 'pointcloud'
    pointcloud_params = {
        f'{pc}.enable': True,
        f'{pc}.ordered_pc': False,
        f'{pc}.pointcloud_qos': 'SENSOR_DATA',
    }

    cfg = os.path.join(get_package_share_directory('mabel_sensors'),
                       'config', 'realsense_front.yaml')

    enable_imu = LaunchConfiguration('enable_imu')
    base_frame = LaunchConfiguration('base_frame')

    return LaunchDescription([
        DeclareLaunchArgument(
            'enable_imu', default_value='false',
            description='Enable the D435i IMU (publishes /camera/front/imu, like '
                        'the sim). Default OFF: this unit reports "HID Motion '
                        'Sensor Failure" on the Jetson. Flip on once the Tegra '
                        'hid_sensor issue is patched, then set '
                        'enable_imu_fusion:=true in visual_slam.launch.py.'),
        DeclareLaunchArgument(
            'base_frame', default_value='base_link',
            description='Robot frame the camera mount transform is parented to.'),
        DeclareLaunchArgument('mount_x', default_value='-0.25'),
        DeclareLaunchArgument('mount_y', default_value='0.0'),
        DeclareLaunchArgument('mount_z', default_value='0.13'),
        DeclareLaunchArgument('mount_roll', default_value='0.0'),
        DeclareLaunchArgument('mount_pitch', default_value='0.0'),
        DeclareLaunchArgument('mount_yaw', default_value='3.14159265'),

        # base_link -> front_link camera mount (the URDF is auto-generated and has
        # no camera frame, so the extrinsic lives here). front_link is the
        # RealSense body frame; the driver attaches front_*_optical_frame under it.
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='front_camera_mount_tf',
            output='screen',
            arguments=[
                '--x', LaunchConfiguration('mount_x'),
                '--y', LaunchConfiguration('mount_y'),
                '--z', LaunchConfiguration('mount_z'),
                '--roll', LaunchConfiguration('mount_roll'),
                '--pitch', LaunchConfiguration('mount_pitch'),
                '--yaw', LaunchConfiguration('mount_yaw'),
                '--frame-id', base_frame,
                '--child-frame-id', 'front_link',
            ],
        ),

        Node(
            package='realsense2_camera',
            executable='realsense2_camera_node',
            # namespace + node name give the /camera/front/<stream> topic tree
            # (verified on thor: matches mabel_sim sensor_config.py exactly).
            namespace='camera',
            name='front',
            output='screen',
            parameters=[cfg, {
                # Launch flag overrides the yaml so the IMU can be toggled without
                # editing the config. unite_imu_method (in the yaml) only applies
                # when these are true.
                'enable_gyro': ParameterValue(enable_imu, value_type=bool),
                'enable_accel': ParameterValue(enable_imu, value_type=bool),
                # Arch-correct pointcloud filter (see top of file). The filter
                # publishes the XYZRGB cloud on /camera/front/depth/color/points;
                # the relay below republishes it onto the sim contract topic.
                **pointcloud_params,
            }],
        ),

        # The realsense pointcloud filter publishes on .../depth/color/points, but
        # the sim contract (and hardware.rviz "Front Cloud") expects
        # /camera/front/depth/points. Relay so real == sim. Cheap + local.
        Node(
            package='topic_tools',
            executable='relay',
            name='front_cloud_relay',
            arguments=['/camera/front/depth/color/points',
                       '/camera/front/depth/points'],
            output='log',
        ),
    ])
