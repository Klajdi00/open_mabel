"""Head ZED Mini for MABEL -- the REAL drop-in for the sim head ZED stereo.

Drives the Stereolabs ``zed-ros2-wrapper`` (built from source against ZED SDK
4.2.5; see ``scripts/build_sdk_cams.sh``) for the ZED-M on the robot head, and
republishes it onto the sim's ``/camera/zed/*`` contract so the SAME nvblox /
cuVSLAM manipulation stack and the SAME RViz layout run on real hardware.

    ros2 launch mabel_sensors zed.launch.py                 # head ZED + contract relays
    ros2 launch mabel_sensors zed.launch.py contract:=false # native /zed topics only

Native wrapper topics (camera_name:=zed, node:=zed_node):
    /zed/zed_node/rgb/image_rect_color           (+ camera_info)   RGB
    /zed/zed_node/left|right/image_rect_color     (+ camera_info)
    /zed/zed_node/depth/depth_registered          (+ camera_info)
    /zed/zed_node/point_cloud/cloud_registered    XYZRGB pointcloud
Contract relays (``contract:=true``, the default) mirror these onto what
``mabel_sim`` publishes (sensor_config.py "zed_*"):
    /camera/zed/left|right/image_raw   /camera/zed/depth/image_rect_raw
    /camera/zed/depth/points

TF: the wrapper's own state_publisher emits the camera-internal frames under
``zed_camera_link`` (-> zed_*_camera_optical_frame). We DON'T let it publish any
odom/map TF (pos_tracking + publish_tf off): the robot owns odometry. Instead we
static-publish ``head -> zed_camera_link`` so the cloud lands at the head in the
robot frame. Translation default is the ZED-M centre measured from the canonical
MJCF (mabel_full.xml head body: zed_left/zed_right mean pos ~ (0, 0.006, -0.109)).
The rotation maps the MJCF head-camera view axis into REP-103 (x-forward) and
WILL need a small calibration tweak -- override mount_* to refine without
touching the auto-generated URDF.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    zed_share = get_package_share_directory('zed_wrapper')
    override = os.path.join(get_package_share_directory('mabel_sensors'),
                           'config', 'zedm_head.yaml')

    contract = LaunchConfiguration('contract')
    mount_frame = LaunchConfiguration('mount_frame')

    return LaunchDescription([
        DeclareLaunchArgument('camera_name', default_value='zed',
                              description='ZED node namespace -> /<name>/zed_node/*.'),
        DeclareLaunchArgument('serial_number', default_value='0',
                              description='ZED serial (0 = first camera found).'),
        DeclareLaunchArgument('contract', default_value='true',
                              description='Also relay onto the /camera/zed/* sim '
                                          'contract (RViz + manipulation SLAM).'),
        DeclareLaunchArgument('mount_frame', default_value='head',
                              description='Robot link the ZED is bolted to.'),
        # head -> zed_camera_link mount (translation from the MJCF; see module docstring)
        DeclareLaunchArgument('mount_x', default_value='0.0'),
        DeclareLaunchArgument('mount_y', default_value='0.006'),
        DeclareLaunchArgument('mount_z', default_value='-0.109'),
        # head -> zed_camera_link. The `head` link frame is rotated by the neck euler
        # chain, so the old (-pi/2,0,-pi/2) left the cloud's optical z-forward pointing
        # at base -Z (under the ground). (-3pi/4, pi/2, -pi/4) makes the net
        # head -> zed_*_optical_frame match camera_hal's MJCF value (z-forward = robot
        # front, base -X). See hardware_bridge/config.py zed mount for the derivation.
        DeclareLaunchArgument('mount_roll', default_value='-2.3561945'),
        DeclareLaunchArgument('mount_pitch', default_value='1.5707963'),
        DeclareLaunchArgument('mount_yaw', default_value='-0.7853982'),

        # The Stereolabs wrapper. publish_tf/map_tf OFF: the robot owns odom/SLAM;
        # we provide the static head mount below. The ZED's OWN state_publisher
        # still emits zed_camera_link -> optical frames (publish_urdf default on).
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(zed_share, 'launch', 'zed_camera.launch.py')),
            launch_arguments={
                'camera_model': 'zedm',
                'camera_name': LaunchConfiguration('camera_name'),
                'serial_number': LaunchConfiguration('serial_number'),
                'ros_params_override_path': override,
                'publish_tf': 'false',
                'publish_map_tf': 'false',
                'publish_imu_tf': 'true',
            }.items()),

        # base_link/head -> zed_camera_link mount (the auto-generated URDF has no
        # camera frame, so the extrinsic lives here -- mirrors realsense.launch.py).
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='zed_head_mount_tf',
            output='screen',
            arguments=[
                '--x', LaunchConfiguration('mount_x'),
                '--y', LaunchConfiguration('mount_y'),
                '--z', LaunchConfiguration('mount_z'),
                '--roll', LaunchConfiguration('mount_roll'),
                '--pitch', LaunchConfiguration('mount_pitch'),
                '--yaw', LaunchConfiguration('mount_yaw'),
                '--frame-id', mount_frame,
                '--child-frame-id', 'zed_camera_link',
            ],
        ),

        # --- /camera/zed/* contract relays (sim parity) ----------------------
        # Lightweight topic_tools relays so the real ZED feeds the EXACT topics
        # mabel_sim publishes. RViz/manipulation SLAM subscribe to these. The
        # relayed messages keep their zed_*_optical_frame frame_id (resolved via
        # the static mount above), so they render correctly in the robot frame.
        Node(package='topic_tools', executable='relay', name='zed_relay_left',
             arguments=['/zed/zed_node/left/image_rect_color',
                        '/camera/zed/left/image_raw'],
             condition=IfCondition(contract), output='log'),
        Node(package='topic_tools', executable='relay', name='zed_relay_right',
             arguments=['/zed/zed_node/right/image_rect_color',
                        '/camera/zed/right/image_raw'],
             condition=IfCondition(contract), output='log'),
        Node(package='topic_tools', executable='relay', name='zed_relay_depth',
             arguments=['/zed/zed_node/depth/depth_registered',
                        '/camera/zed/depth/image_rect_raw'],
             condition=IfCondition(contract), output='log'),
        Node(package='topic_tools', executable='relay', name='zed_relay_cloud',
             arguments=['/zed/zed_node/point_cloud/cloud_registered',
                        '/camera/zed/depth/points'],
             condition=IfCondition(contract), output='log'),
    ])
