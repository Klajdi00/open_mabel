from launch import LaunchDescription, LaunchContext
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def launch_setup(context: LaunchContext, *args, **kwargs):
    camera_target = LaunchConfiguration('camera').perform(context)
    device_path = LaunchConfiguration('device').perform(context)
    nodes_to_start = []

    # ==========================================
    # STEREO ARCHITECTURE (Head)
    # ==========================================
    if camera_target == 'head':
        print("\n[UNIVERSAL CALIBRATOR] Booting STEREO Pipeline for HEAD\n")
        
        nodes_to_start.append(Node(
            package='v4l2_camera', executable='v4l2_camera_node', name='head_stereo_master', namespace='sensors/head',
            parameters=[{'video_device': device_path, 'image_size': [1280, 480], 'pixel_format': 'YUYV', 'camera_frame_id': 'head_center_link'}],
            output='screen'
        ))
        nodes_to_start.append(Node(
            package='image_proc', executable='crop_decimate_node', name='left_eye_crop', namespace='sensors/head',
            parameters=[{'offset_x': 0, 'offset_y': 0, 'width': 640, 'height': 480}],
            remappings=[
                ('in/image_raw', 'image_raw'), ('in/camera_info', 'camera_info'),
                ('out/image_raw', '/sensors/head/left/image_raw'), ('out/camera_info', '/sensors/head/left/camera_info')
            ], output='screen'
        ))
        nodes_to_start.append(Node(
            package='image_proc', executable='crop_decimate_node', name='right_eye_crop', namespace='sensors/head',
            parameters=[{'offset_x': 640, 'offset_y': 0, 'width': 640, 'height': 480}],
            remappings=[
                ('in/image_raw', 'image_raw'), ('in/camera_info', 'camera_info'),
                ('out/image_raw', '/sensors/head/right/image_raw'), ('out/camera_info', '/sensors/head/right/camera_info')
            ], output='screen'
        ))
        nodes_to_start.append(Node(
            package='camera_calibration', executable='cameracalibrator', name='stereo_calibrator',
            arguments=[
                '--size', '8x6', '--square', '0.025', '--approximate', '0.1', '--no-service-check',
                'left:=/sensors/head/left/image_raw', 'right:=/sensors/head/right/image_raw',
                'left_camera:=/sensors/head/left', 'right_camera:=/sensors/head/right'
            ], output='screen'
        ))

    # ==========================================
    # MONOCULAR ARCHITECTURE (Wrists)
    # ==========================================
    elif camera_target in ['left_wrist', 'right_wrist']:
        print(f"\n[UNIVERSAL CALIBRATOR] Booting MONOCULAR Pipeline for {camera_target.upper()}\n")
        
        ns = f'sensors/{camera_target}'
        nodes_to_start.append(Node(
            package='v4l2_camera', executable='v4l2_camera_node', name=f'{camera_target}_cam', namespace=ns,
            parameters=[{'video_device': device_path, 'image_size': [640, 480], 'pixel_format': 'YUYV', 'camera_frame_id': f'{camera_target}_cam_link'}],
            output='screen'
        ))
        nodes_to_start.append(Node(
            package='camera_calibration', executable='cameracalibrator', name='mono_calibrator',
            arguments=['--size', '8x6', '--square', '0.025', '--no-service-check'],
            remappings=[('image', f'/{ns}/image_raw'), ('camera', f'/{ns}')], 
            output='screen'
        ))
    else:
        print(f"\n[ERROR] Unknown target: '{camera_target}'. Use 'left_wrist', 'right_wrist', or 'head'.\n")

    return nodes_to_start

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('camera', description="Target: 'left_wrist', 'right_wrist', or 'head'"),
        DeclareLaunchArgument('device', description="Path to physical USB device"),
        OpaqueFunction(function=launch_setup)
    ])