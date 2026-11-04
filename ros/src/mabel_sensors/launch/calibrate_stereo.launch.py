from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    stereo_path = '/dev/v4l/by-path/platform-a80aa10000.usb-usb-0:4.2:1.0-video-index0'

    return LaunchDescription([
        # 1. Master Camera (Double-Wide)
        Node(
            package='v4l2_camera', executable='v4l2_camera_node', name='head_stereo_master', namespace='sensors/head',
            parameters=[{'video_device': stereo_path, 'image_size': [1280, 480], 'pixel_format': 'YUYV', 'camera_frame_id': 'head_center_link'}],
            output='screen'
        ),

        # 2. Left Scissors
        Node(
            package='image_proc', executable='crop_decimate_node', name='left_eye_crop', namespace='sensors/head',
            parameters=[{'offset_x': 0, 'offset_y': 0, 'width': 640, 'height': 480}],
            remappings=[
                ('in/image_raw', 'image_raw'), ('in/camera_info', 'camera_info'),
                ('out/image_raw', '/sensors/head/left/image_raw'), ('out/camera_info', '/sensors/head/left/camera_info')
            ],
            output='screen'
        ),

        # 3. Right Scissors
        Node(
            package='image_proc', executable='crop_decimate_node', name='right_eye_crop', namespace='sensors/head',
            parameters=[{'offset_x': 640, 'offset_y': 0, 'width': 640, 'height': 480}],
            remappings=[
                ('in/image_raw', 'image_raw'), ('in/camera_info', 'camera_info'),
                ('out/image_raw', '/sensors/head/right/image_raw'), ('out/camera_info', '/sensors/head/right/camera_info')
            ],
            output='screen'
        ),

        # 4. Stereo Calibration GUI (FIXED)
        Node(
            package='camera_calibration', executable='cameracalibrator', name='stereo_calibrator',
            arguments=[
                '--size', '8x6',
                '--square', '0.025',
                '--approximate', '0.1',
                '--no-service-check',
                # Moved from remappings into raw arguments to force Stereo Mode!
                'left:=/sensors/head/left/image_raw',
                'right:=/sensors/head/right/image_raw',
                'left_camera:=/sensors/head/left',
                'right_camera:=/sensors/head/right'
            ],
            output='screen'
        ),
    ])