"""Bring up a MABEL backend ready for keyboard teleop, then start the keyboard.

Pick the backend explicitly -- hardware and sim are strictly separate (one-physics
rule); this launch includes EXACTLY ONE of them:

  ros2 launch mabel_bringup teleop.launch.py backend:=sim         # MuJoCo sim
  ros2 launch mabel_bringup teleop.launch.py backend:=hardware    # real hardware

Then, in a SECOND, focused terminal, start the keyboard with `ros2 run` (NOT
`ros2 launch`) so it can read that terminal's keys:

  ros2 run mabel_teleop keyboard_teleop                          # base+arms, /mabel_cmd
  ros2 run mabel_teleop keyboard_teleop --ros-args -p output:=cmd_vel   # nav via cmd_vel_bridge

Why two terminals: `ros2 launch` does not connect a node to your terminal's stdin,
so a launched keyboard node can't see your keypresses (it would fall back to
pynput's global X listener, which only catches keys a focused X window receives).
Running the keyboard with `ros2 run` gives it a real TTY and it just works.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    backend = LaunchConfiguration('backend')
    is_sim = PythonExpression(["'", backend, "' == 'sim'"])

    hardware_launch = PathJoinSubstitution(
        [FindPackageShare('mabel_bringup'), 'launch', 'hardware.launch.py'])
    sim_launch = PathJoinSubstitution(
        [FindPackageShare('mabel_sim'), 'launch', 'sim.launch.py'])

    return LaunchDescription([
        DeclareLaunchArgument('backend', default_value='sim',
                              description="Which backend to bring up: 'sim' or 'hardware'. "
                                          "Strictly one -- never both (one-physics rule)."),
        # PATH A: SIMULATION (sibling package, owns MuJoCo physics)
        IncludeLaunchDescription(sim_launch, condition=IfCondition(is_sim)),
        # PATH B: REAL HARDWARE (the federated driver nodes)
        IncludeLaunchDescription(hardware_launch, condition=UnlessCondition(is_sim)),
        LogInfo(msg=['Backend up. Now start the keyboard in a FOCUSED terminal:\n'
                     '    ros2 run mabel_teleop keyboard_teleop\n'
                     '(use `ros2 run`, not `ros2 launch`, so it can read your keys.)']),
    ])
