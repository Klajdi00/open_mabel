import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'mabel_hand_hardware'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
        (os.path.join('share', package_name, 'udev'), glob('udev/*')),
    ],
    # 'orcahand' is the shared hardware layer at firmware/orca_hand/feetech;
    # install it into the ROS Python env once with:
    #   pip install -e firmware/orca_hand/feetech
    install_requires=['setuptools', 'orcahand'],
    zip_safe=True,
    maintainer='MABEL Authors',
    maintainer_email='maintainers@example.com',
    description='ROS 2 hardware bringup for the MABEL ORCA dexterous hands (Feetech bus)',
    license='TODO: License declaration',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'hand_node = mabel_hand_hardware.hand_node:main',
            'hand_joint_state_relay = mabel_hand_hardware.hand_joint_state_relay:main',
            'hand_status = mabel_hand_hardware.hand_status:main',
        ],
    },
)
