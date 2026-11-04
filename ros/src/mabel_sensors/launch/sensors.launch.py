import os
import sys
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                            LogInfo, ExecuteProcess, OpaqueFunction)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import (get_package_share_directory,
                                          PackageNotFoundError)


def _camera_hardware_node(context, *args, **kwargs):
    """Launch the SINGLE camera-hardware node (sensors/ros_camera_node.py).

    ROS does NOT open the cameras directly. This one node (camera_hal: D435i +
    ZED-M) owns the USB depth cameras and publishes the /camera/front/* +
    /camera/zed/* contract over DDS; the bring-up just connects ROS to it. The
    same node also (--serve) feeds the server MJPEG fan-out. It runs in THIS conda
    env (pyrealsense2 V4L2 + pyzed installed here) -- no udev rules, no cross-env
    bridge. Override the script path with $MABEL_CAMERA_NODE."""
    if LaunchConfiguration('depth_cams').perform(context).lower() != 'true':
        return []
    script = os.environ.get(
        'MABEL_CAMERA_NODE', os.path.expanduser('~/open_mabel/sensors/ros_camera_node.py'))
    if not os.path.exists(script):
        return [LogInfo(msg=f"[MISSING] camera hardware node not found at {script} "
                            "-- set MABEL_CAMERA_NODE or pass depth_cams:=false.")]
    cams = LaunchConfiguration('cameras').perform(context)
    serve = LaunchConfiguration('camera_serve').perform(context).lower() == 'true'
    cmd = [sys.executable, script, '--ros', '--cameras', cams]
    if serve:
        cmd.append('--serve')
    return [LogInfo(msg=f"[camera_hal] owning {cams} cameras -> /camera/* "
                        f"(serve={serve})"),
            ExecuteProcess(cmd=cmd, output='screen')]


def generate_launch_description():
    ld = LaunchDescription()
    pkg_share = get_package_share_directory('mabel_sensors')
    config_dir = os.path.join(pkg_share, 'config')

    # ==========================================
    # 0. DEPTH CAMERAS -- the single camera-hardware node (D435i front + ZED-M head)
    # ==========================================
    # ROS does NOT touch the camera hardware directly. One dedicated node
    # (sensors/ros_camera_node.py, built on the ROS-free camera_hal) OWNS both USB
    # depth cameras and publishes the sim contract:
    #     front D435i -> /camera/front/{color,depth/points,depth,imu}
    #     head  ZED-M -> /camera/zed/{left,depth/points,depth,imu}
    # so the same RViz / nvblox / cuVSLAM consume real pixels over DDS. The SAME
    # node also pushes to the server's MJPEG fan-out when camera_serve:=true.
    #   depth_cams:=false   -> don't start it
    #   cameras:=realsense|zed|all   -> which cameras to own (default all)
    #   camera_serve:=true  -> also stream MJPEG to the network (Vision Pro/web)
    ld.add_action(DeclareLaunchArgument(
        'depth_cams', default_value='true',
        description='Start the camera-hardware node (front D435i + head ZED-M).'))
    ld.add_action(DeclareLaunchArgument(
        'cameras', default_value='all',
        description='Which depth cameras the node owns: all | realsense | zed.'))
    ld.add_action(DeclareLaunchArgument(
        'camera_serve', default_value='true',
        description='Also stream camera frames as MJPEG to the network (Vision '
                    'Pro / web / iOS) on http://<host>:8080/camera/*/stream.mjpg. '
                    'Non-fatal if the port is busy (e.g. another server) -- ROS '
                    'publishing continues. Set false to skip the MJPEG server.'))
    ld.add_action(OpaqueFunction(function=_camera_hardware_node))

    # ==========================================
    # 1. WRIST CAMERAS (Arducam Global-Shutter, left + right)
    # ==========================================
    # The wrist cams are owned by the SAME single camera-hardware node as the depth
    # cams (ros_camera_node.py, camera_hal UvcCamera): it publishes
    #     /camera/wrist_left/image_raw   (+ .../camera_info)   rgb8
    #     /camera/wrist_right/image_raw  (+ .../camera_info)   rgb8
    # and also fans them out over the MJPEG/UDP video server. So ROS does NOT open
    # the wrist /dev/videoN here -- one owner, no double-publish (the old stub that
    # spun up a separate v4l2_camera_node per side is gone). Disable via the node's
    # --no-wrist flag (or CAM_WRIST=0). The udev rule 99-mabel-wrist-cams pins the
    # hub ports to /dev/wrist_{left,right}; the node falls back to the by-path node.
    ld.add_action(LogInfo(
        msg="[camera_hal] wrist cams -> /camera/wrist_{left,right}/image_raw "
            "(owned by ros_camera_node.py; pass --no-wrist there to skip)."))

    # ==========================================
    # 2. HEAD STEREO CAMERA (Air-Gap Bridge)
    # ==========================================
    stereo_path = '/dev/v4l/by-path/platform-a80aa10000.usb-usb-0:4.2:1.0-video-index0'
    
    if os.path.exists(stereo_path):
        ld.add_action(LogInfo(msg="[SUCCESS] Booting Head Stereo Pipeline..."))
        
        ld.add_action(Node(
            package='v4l2_camera', executable='v4l2_camera_node', name='head_stereo_master', namespace='sensors/head',
            parameters=[{'video_device': stereo_path, 'image_size': [1280, 480], 'pixel_format': 'YUYV', 'camera_frame_id': 'head_center_link'}],
            output='screen'
        ))

        # --- LEFT EYE ---
        ld.add_action(Node(
            package='image_proc', executable='crop_decimate_node', name='left_eye_crop', namespace='sensors/head',
            parameters=[{'offset_x': 0, 'offset_y': 0, 'width': 640, 'height': 480}],
            remappings=[('in/image_raw', 'image_raw'), ('in/camera_info', 'camera_info'), ('out/image_raw', '/sensors/head/left_temp/image_raw')],
            output='screen'
        ))
        ld.add_action(Node(
            executable='python3', arguments=[os.path.join(get_package_share_directory('mabel_sensors'), 'scripts/info_injector.py')],
            name='left_bridge',
            parameters=[{'yaml_path': os.path.join(config_dir, 'head_left_cam.yaml'), 'camera_frame_id': 'head_left_cam_link',
                         'input_image_topic': '/sensors/head/left_temp/image_raw', 'output_image_topic': '/sensors/head/left/image_raw', 'output_info_topic': '/sensors/head/left/camera_info'}],
            output='screen'
        ))
        ld.add_action(Node(
            package='image_proc', executable='rectify_node', name='left_rectify', namespace='sensors/head/left',
            remappings=[('image', 'image_raw')], output='screen'
        ))

        # --- RIGHT EYE ---
        ld.add_action(Node(
            package='image_proc', executable='crop_decimate_node', name='right_eye_crop', namespace='sensors/head',
            parameters=[{'offset_x': 640, 'offset_y': 0, 'width': 640, 'height': 480}],
            remappings=[('in/image_raw', 'image_raw'), ('in/camera_info', 'camera_info'), ('out/image_raw', '/sensors/head/right_temp/image_raw')],
            output='screen'
        ))
        ld.add_action(Node(
            executable='python3', arguments=[os.path.join(get_package_share_directory('mabel_sensors'), 'scripts/info_injector.py')],
            name='right_bridge',
            parameters=[{'yaml_path': os.path.join(config_dir, 'head_right_cam.yaml'), 'camera_frame_id': 'head_right_cam_link',
                         'input_image_topic': '/sensors/head/right_temp/image_raw', 'output_image_topic': '/sensors/head/right/image_raw', 'output_info_topic': '/sensors/head/right/camera_info'}],
            output='screen'
        ))
        ld.add_action(Node(
            package='image_proc', executable='rectify_node', name='right_rectify', namespace='sensors/head/right',
            remappings=[('image', 'image_raw')], output='screen'
        ))

    return ld