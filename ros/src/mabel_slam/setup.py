import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'mabel_slam'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='MABEL Authors',
    maintainer_email='maintainers@example.com',
    description='SLAM for MABEL: visual_slam (cuVSLAM + nvblox, navigation/'
                'manipulation envs) and lidar_slam (slam_toolbox 2D). Hardware-decoupled.',
    license='TODO: License declaration',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            # Start SLAM with the launch files (native Isaac ROS, no Docker):
            #   ros2 launch mabel_slam visual_slam.launch.py   (cuVSLAM + nvblox)
            #   ros2 launch mabel_slam lidar_slam.launch.py    (slam_toolbox 2D)
            # nvblox post-processing nodes (used inside visual_slam.launch.py):
            'mesh_to_pointcloud = mabel_slam.mesh_to_pointcloud:main',
            'slice_to_occupancy = mabel_slam.slice_to_occupancy:main',
            # Provenance-named grid -> canonical /map (spawned by each SLAM launch):
            'map_relay = mabel_slam.map_relay:main',
            # LiDAR-derived odom->base_link (real_run.sh spawns it in lidar mode,
            # in the conda env — needs scipy):
            'laser_odometry = mabel_slam.laser_odometry:main',
        ],
    },
)
