import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'mabel_base_hardware'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Install the launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mabel',
    maintainer_email='maintainers@example.com',
    description='Hardware bridge for Mabel swerve base and lift',
    license='TODO: License declaration',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            # Thin CLIENT of ~/open_mabel/hardware_bridge (the base hardware owner): adapts
            # /mabel_cmd <-> the bridge's DDS contract. NO LONGER touches the Teensy.
            # 'swerve_node' kept as the legacy executable name; 'base_bridge_client'
            # is the descriptive alias (both run the same client main).
            'swerve_node = mabel_base_hardware.swerve_node:main',
            'base_bridge_client = mabel_base_hardware.swerve_node:main',
            # The base's wheel-odometry motion model (/joint_states -> /odom +
            # odom->base_link). Lives here (the base/HAL), shared by the real base
            # AND the sim; kept OUT of mabel_navigation so nav stays strictly modular.
            'swerve_odometry = mabel_base_hardware.swerve_odometry:main',
        ],
    },
)