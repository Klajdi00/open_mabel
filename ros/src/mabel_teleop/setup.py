from setuptools import find_packages, setup

package_name = 'mabel_teleop'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', __import__('glob').glob('launch/*.launch.py')),
    ],
    # pynput gives a global hold-to-move keyboard listener that works under
    # `ros2 launch` (no controlling TTY). Without it the node falls back to a
    # stdin reader that only works when run via `ros2 run` in a focused terminal.
    install_requires=['setuptools', 'pynput'],
    zip_safe=True,
    maintainer='mabel',
    maintainer_email='maintainers@example.com',
    description='Universal teleop interface for Mabel',
    license='TODO: License declaration',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'keyboard_teleop = mabel_teleop.keyboard_teleop:main',
        ],
    },
)