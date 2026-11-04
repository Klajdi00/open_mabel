import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'mabel_navigation'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'behavior_trees'), glob('behavior_trees/*.xml')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
        (os.path.join('share', package_name, 'maps'), glob('maps/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='MABEL Authors',
    maintainer_email='maintainers@example.com',
    description='Nav2 navigation for MABEL holonomic base + cmd_vel bridge.',
    license='TODO: License declaration',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'cmd_vel_bridge = mabel_navigation.cmd_vel_bridge:main',
            # swerve_odometry MOVED to mabel_base_hardware (the base/HAL) -- nav is
            # strictly modular and owns no wheel-odometry node.
            'command_odometry = mabel_navigation.command_odometry:main',
            'frontier_explorer = mabel_navigation.frontier_explorer:main',
            'goal_relay = mabel_navigation.goal_relay:main',
            'nav_pose_setter = mabel_navigation.nav_pose_setter:main',
        ],
    },
)
