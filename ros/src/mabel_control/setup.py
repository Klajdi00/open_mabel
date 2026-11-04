import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'mabel_control'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='MABEL Authors',
    maintainer_email='maintainers@example.com',
    description='ROS 2 control layer for MABEL: hosts the whole-body teleop controller '
                'against a MuJoCo mirror and republishes it on the command contract.',
    license='MIT',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            # Teleop control runs inside mabel_gateway (it owns the operator pose);
            # this package is currently a library (command_extract, ros_plant). A
            # standalone control node is added here when an app-independent control
            # entry point is needed.
        ],
    },
)
