"""Run the MABEL app teleop gateway on the real robot (ROS plant).

    ros2 launch mabel_gateway gateway.launch.py

Launches the standalone teleop server (server/sim_teleop_bridge.py) with
MABEL_DISPATCH_TARGET=real:ros, so the gateway hosts the controller against a MuJoCo
mirror and publishes the canonical command contract (/mabel_cmd, /arms/command, ...) that
the mabel_*_hardware drivers consume. The app protocol (WebSocket :9090) and video
(MJPEG :8080 / UDP :9091) are unchanged from the Mac sim -- only the plant differs.

This brings up the CONTROL/teleop gateway; bring up the robot drivers separately with
`mabel_bringup hardware.launch.py`. The server lives in server/ (a plain Python program,
not an ament package), so it is run as a process; point to it with the `server_root` arg
or the MABEL_ROOT env (defaults to ~/open_mabel then ~/open_mabel).

Sim/Mac note: for the in-process MuJoCo gateway (no ROS) just run the server directly
(server/start_teleop.sh) -- do NOT launch this together with mabel_sim (one-physics rule).
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration, PythonExpression


def _default_server_root():
    for root in (os.environ.get("MABEL_ROOT"),
                 os.path.expanduser("~/open_mabel"),
                 os.path.expanduser("~/open_mabel")):
        if root and os.path.isdir(os.path.join(root, "server")):
            return os.path.join(root, "server")
    return os.path.expanduser("~/open_mabel/server")


def generate_launch_description():
    server_root = LaunchConfiguration("server_root")
    server_py = PythonExpression(["'", server_root, "' + '/sim_teleop_bridge.py'"])

    return LaunchDescription([
        DeclareLaunchArgument("server_root", default_value=_default_server_root(),
                              description="Path to the MABEL server/ directory."),
        # The server picks its own default scene for the kinematic mirror; to override,
        # run the server directly with --scene, or extend this launch.
        ExecuteProcess(
            cmd=["python3", server_py],
            additional_env={"MABEL_DISPATCH_TARGET": "real:ros"},
            output="screen",
            name="mabel_gateway",
        ),
    ])
