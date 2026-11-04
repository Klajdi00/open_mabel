import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'mabel_sim'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # CRITICAL: This line tells ROS to install your launch files!
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        # Foxglove Studio layout(s)
        (os.path.join('share', package_name, 'foxglove'), glob('foxglove/*.json')),
        # RViz layouts
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mabel',
    maintainer_email='maintainers@example.com',
    description='Monolithic MuJoCo simulation that mirrors the MABEL hardware ROS interface',
    license='TODO: License declaration',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'mujoco_node = mabel_sim.mujoco_node:main',
            'mujoco_sensors_node = mabel_sim.mujoco_sensors_node:main',
            'bringup_status = mabel_sim.bringup_status:main',
        ],
    },
)