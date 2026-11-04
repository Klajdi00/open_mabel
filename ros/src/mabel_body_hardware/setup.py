import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'mabel_body_hardware'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Install the launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mabel',
    maintainer_email='maintainers@example.com',
    description='Thin ROS client of the hardware_bridge neck (HAL owns the Dynamixel bus)',
    license='TODO: License declaration',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'body_node = mabel_body_hardware.body_node:main'
        ],
    },
)