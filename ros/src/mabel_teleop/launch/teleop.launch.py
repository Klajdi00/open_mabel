"""MABEL keyboard teleop -- a standalone, backend-agnostic COMMAND producer.

This launch starts ONLY the keyboard driver: it brings up no robot, no sensors,
no physics. It publishes the SAME command topics the real hardware and the sim
both consume (/mabel_cmd, /arms/command, /body/command, /<side>_hand/command), so
the identical node drives EITHER backend with no change -- it is sim/real agnostic.

Run a backend first (in another terminal), then start this in a FOCUSED terminal so
the node can read your keystrokes:

  ros2 launch mabel_sim     sim.launch.py            # sim backend, OR
  ros2 launch mabel_bringup hardware.launch.py       # real-robot backend

  ros2 run    mabel_teleop keyboard_teleop           # recommended (reads your TTY)
  ros2 launch mabel_teleop teleop.launch.py          # needs pynput (no TTY under launch)
  ros2 launch mabel_teleop teleop.launch.py output:=cmd_vel    # Twist via cmd_vel_bridge (Nav2 path)

Two modes (TAB toggles), mirroring the iOS / browser teleop tab:
  Mode 1  Navigation    Arrows=translate  A/D=rotate  W/S=lift          -> /mabel_cmd
  Mode 2  Whole-body    WSAD/RF=palm xyz  Y/H+U/J=hand close/open        -> /arms,/body,/<side>_hand/command

`output:=mabel_cmd` (default) drives base/arms directly; `output:=cmd_vel` sends Twist on
/cmd_vel through cmd_vel_bridge's tip-safe model (the same path Nav2 uses). Prefer
`ros2 run` -- `ros2 launch` cannot wire a TTY and falls back to pynput's global X listener.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    output = LaunchConfiguration("output")
    cmd_topic = LaunchConfiguration("cmd_topic")
    use_sim_time = LaunchConfiguration("use_sim_time")
    return LaunchDescription([
        DeclareLaunchArgument("output", default_value="mabel_cmd",
                              description="mabel_cmd = drive base/arms directly; "
                                          "cmd_vel = Twist via cmd_vel_bridge (Nav2 path)."),
        DeclareLaunchArgument("cmd_topic", default_value="/cmd_vel",
                              description="Twist topic when output:=cmd_vel."),
        # Default false so keyboard timing runs on the WALL clock -- correct for human
        # input on BOTH sim and the real robot, and the node needs no /clock to start.
        # Set true only to align the node's clock with a running sim's /clock.
        DeclareLaunchArgument("use_sim_time", default_value="false",
                              description="Use the sim /clock instead of the wall clock. "
                                          "Leave false for real hardware (and usually sim too)."),
        Node(
            package="mabel_teleop",
            executable="keyboard_teleop",
            name="keyboard_teleop",
            output="screen",
            emulate_tty=True,
            parameters=[{
                "output": output,
                "cmd_topic": cmd_topic,
                "use_sim_time": ParameterValue(use_sim_time, value_type=bool),
            }],
        ),
    ])
