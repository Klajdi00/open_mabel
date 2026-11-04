"""Bring up the REAL MABEL hardware -- the robot HAL, hardware only.

    ros2 launch mabel_bringup hardware.launch.py

This is the hardware-only entry point. It composes the real subsystem drivers
(swerve base, arms, neck/body, both ORCA hands, cameras) and robot_state_publisher,
all publishing/subscribing the canonical topic contract (interface_config):
/mabel_cmd, /arms/command, /body/command, /<side>_hand/command + the matching
state topics. Everything ABOVE this layer (control, slam, nav, teleop) is
backend-agnostic and identical to simulation.

Just run it -- no flags needed. Every subsystem is launched; any whose hardware
is absent stays alive and IDLE (it never crashes the bring-up), and the pinned
status dashboard shows it as `waiting`/`offline` while the connected ones show
`live`. So a bench with only the left hand wired still comes up clean:

    ros2 launch mabel_bringup hardware.launch.py

For SIMULATION, launch the sibling package instead -- NEVER both at once
(one-physics rule):

    ros2 launch mabel_sim sim.launch.py

Each subsystem can be toggled off for bring-up / bench testing, e.g.:

    ros2 launch mabel_bringup hardware.launch.py arms:=false hands:=false
    ros2 launch mabel_bringup hardware.launch.py teensy_ip:=192.168.124.10 \\
        left_port:=/dev/serial/by-id/...-left right_port:=/dev/serial/by-id/...-right

Bench-test just the LEFT hand on its own adapter (nothing else wired):

    ros2 launch mabel_bringup hardware.launch.py \\
        base:=false arms:=false body:=false sensors:=false hand_sides:=left
"""
import os
import subprocess
import time

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess,
                            IncludeLaunchDescription, LogInfo, OpaqueFunction)
from launch.conditions import IfCondition
from launch.substitutions import (LaunchConfiguration, PathJoinSubstitution,
                                   PythonExpression)
from launch_ros.actions import Node, SetParameter
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def _sub(pkg, name):
    return PathJoinSubstitution([FindPackageShare(pkg), "launch", name])


# Processes a PREVIOUS bring-up may have left publishing on the contract topics.
# Matched by install-lib path / RViz config / driver-node name so we only ever
# reap a prior MABEL bring-up -- and a running SIM, which publishes the SAME
# contract topics (the one-physics rule) -- never an unrelated process, and never
# THIS launch (whose nodes have not spawned yet when the reaper runs).
_BRINGUP_PROC_PATTERNS = [
    # hardware drivers + the motion model
    "lib/mabel_base_hardware/swerve_node",
    "lib/mabel_arms_hardware/arms_node",
    "lib/mabel_body_hardware/body_node",
    "lib/mabel_hand_hardware/hand_node",
    "lib/mabel_audio_hardware/audio_node",
    "lib/mabel_base_hardware/swerve_odometry",
    # status panel + TF + viz from a prior bring-up. robot_state_publisher MUST be
    # reaped too: a stale one keeps publishing /robot_description + /tf, so without
    # this they pile up across restarts (N RViz TF sources, flicker). Match the
    # hardware RViz layout (hardware.rviz) as well as sim's.
    "lib/mabel_sim/bringup_status",
    "robot_state_publisher",
    "joint_state_publisher",
    "rviz2.*sim_sensors.rviz",
    "rviz2.*hardware",
    # camera stack (mabel_sensors)
    # the single camera-hardware node (camera_hal: D435i + ZED-M) owns the cameras
    "sensors/ros_camera_node.py",
    "lib/v4l2_camera/v4l2_camera_node",
    "mabel_sensors/scripts/info_injector",
    # legacy direct-hardware drivers (no longer launched, reaped if left over)
    "realsense2_camera_node",
    "front_cloud_relay",
    "zed_container",
    "zed_state_publisher",
    "zed_relay_",
    # a running SIM would flood the same topics -> reap it too (one physics)
    "lib/mabel_sim/mujoco_node",
    "lib/mabel_sim/mujoco_sensors_node",
]


def _kill_prior_bringup(context, *args, **kwargs):
    """Cleanly reap any lingering bring-up / sim processes before this launch, so
    the new bring-up owns the topic graph and there is no double-publishing
    (encoder/TF/camera) that would overflow DDS and confuse downstream nodes.

    SIGTERM first so each node deregisters its DDS endpoints cleanly, then a
    SIGKILL sweep for anything that ignored it."""
    if context.perform_substitution(LaunchConfiguration("kill_existing")) != "true":
        return []
    reaped = False
    for pat in _BRINGUP_PROC_PATTERNS:        # graceful pass
        if subprocess.run(["pkill", "-TERM", "-f", pat]).returncode == 0:
            reaped = True
    if reaped:
        time.sleep(1.2)                       # let them shut down + drop endpoints
        for pat in _BRINGUP_PROC_PATTERNS:    # force-kill stragglers
            subprocess.run(["pkill", "-KILL", "-f", pat])
        time.sleep(0.5)
        print("[hardware.launch] reaped a previous bring-up / sim instance "
              "(clean restart; pass kill_existing:=false to keep them).")
    return []


def _launch_slam_and_rviz(context, *args, **kwargs):
    """Bring up visual SLAM (cuVSLAM + nvblox) from the front RealSense, then RViz.

    SLAM lives in the apt Isaac ROS stack (cuVSLAM/nvblox are NOT in the conda
    RoboStack env this bring-up runs in), so it can't be a plain
    IncludeLaunchDescription -- it must run in its OWN environment. We shell out
    through scripts/slam_env.sh (the same bridge the server uses: it wipes conda,
    sources /opt/ros/jazzy, and forces CycloneDDS/domain 0 so the apt SLAM
    discovers the conda camera topics). It SUBSCRIBES to the front IR stereo +
    depth + colour the camera node publishes and emits /nvblox_node/mesh_* (the 3D
    mesh), /map (the 2D ESDF slice, nav_2d_map:=true) and the map->odom correction.

    RViz then opens LAST, on hardware.rviz (Fixed Frame = base_link) whether or not
    SLAM is on -- that layout already has the nvblox mesh + 2D /map displays, and a
    base_link root keeps the robot + camera clouds placed even before odom exists."""
    actions = []
    slam_on = LaunchConfiguration("slam").perform(context).lower() == "true"
    if slam_on:
        ws = os.environ.get("MABEL_WS") or os.path.expanduser("~/open_mabel/ros")
        slam_env = os.path.join(ws, "scripts", "slam_env.sh")
        if not os.path.exists(slam_env):
            actions.append(LogInfo(
                msg=f"[hardware.launch] slam:=true but {slam_env} is missing -- "
                    "skipping SLAM (install the Isaac ROS apt stack / set MABEL_WS). "
                    "Hardware + RViz still come up."))
        else:
            nav2d = "true" if LaunchConfiguration("nav_2d_map").perform(
                context).lower() == "true" else "false"
            # slam_odom:=true -> cuVSLAM publishes odom->base_link itself (self-
            # contained visual odometry; the base TF is turned off above), so the
            # static front camera generates odom + 3D mesh + 2D map with NO wheel
            # odom. false -> cuVSLAM only corrects map->odom (wheel odom owns base).
            slam_odom = "true" if LaunchConfiguration("slam_odom").perform(
                context).lower() == "true" else "false"
            # Source the apt SLAM env, then exec ros2 launch so this process IS the
            # launch (a later SIGINT reaches it and tears the SLAM nodes down).
            slam_cmd = (f'source "{slam_env}" && exec ros2 launch mabel_slam '
                        f'visual_slam.launch.py env:=navigation use_sim_time:=false '
                        f'nav_2d_map:={nav2d} publish_odom_to_base_tf:={slam_odom}')
            actions.append(LogInfo(
                msg="[hardware.launch] starting visual SLAM (cuVSLAM + nvblox) on the "
                    "front RealSense via scripts/slam_env.sh (apt Isaac ROS)."))
            actions.append(ExecuteProcess(
                cmd=["bash", "-c", slam_cmd], output="screen", name="visual_slam"))

    if LaunchConfiguration("rviz").perform(context).lower() == "true":
        cfg = LaunchConfiguration("rviz_config").perform(context)
        if not cfg:
            # ALWAYS hardware.rviz (Fixed Frame = base_link), even with SLAM. It
            # already carries the nvblox mesh + 2D /map displays, AND base_link as
            # the fixed frame keeps the robot + camera clouds fully placed even when
            # there is no odom->base_link yet (no wheel odom on the bench / before
            # the base is up). An odom-rooted view leaves everything in base_link
            # (robot, depth clouds) unplaceable -> piled at the origin. The SLAM
            # mesh/map (odom frame) still render once odom->base_link exists.
            cfg = os.path.join(
                get_package_share_directory("mabel_bringup"), "rviz", "hardware.rviz")
        actions.append(Node(
            package="rviz2", executable="rviz2", name="rviz2", output="screen",
            arguments=["-d", cfg], parameters=[{"use_sim_time": False}]))
    return actions


def _launch_lidar(context, *args, **kwargs):
    """Bring up the RPLIDAR A3 driver -> /scan (frame lidar_link, the URDF frame).

    sllidar_ros2 is an apt-only package (/opt/ros/jazzy), so -- like the SLAM stack
    -- shell out through scripts/slam_env.sh (apt ROS + CycloneDDS/domain 0) instead
    of an IncludeLaunchDescription from this conda bring-up. Gracefully logs + skips
    if sllidar isn't installed (ros-jazzy-sllidar-ros2) or MABEL_WS is unset, so the
    bring-up never dies; the bringup_status 'LiDAR /scan' row reads WAIT until the
    lidar is connected and publishing."""
    if LaunchConfiguration("lidar").perform(context).lower() != "true":
        return []
    ws = os.environ.get("MABEL_WS") or os.path.expanduser("~/open_mabel/ros")
    slam_env = os.path.join(ws, "scripts", "slam_env.sh")
    if not os.path.exists(slam_env):
        return [LogInfo(msg=f"[hardware.launch] lidar:=true but {slam_env} is "
                            "missing -- skipping the RPLIDAR driver.")]
    port = LaunchConfiguration("lidar_serial_port").perform(context)
    # Check the package exists in the apt env BEFORE launching, so a box without
    # the driver (e.g. this dev box) skips cleanly instead of spewing a stacktrace.
    # sllidar A3 = 256000 baud; publish in lidar_link to match the URDF + sim.
    cmd = (f'source "{slam_env}" >/dev/null 2>&1; '
           f'if ros2 pkg prefix sllidar_ros2 >/dev/null 2>&1; then '
           f'exec ros2 launch sllidar_ros2 sllidar_a3_launch.py '
           f'serial_port:={port} frame_id:=lidar_link; '
           f'else echo "[hardware.launch] RPLIDAR driver (sllidar_ros2) not '
           f'installed -- run: sudo apt install ros-jazzy-sllidar-ros2 (then '
           f'connect the RPLIDAR on {port}). Skipping; /scan stays empty."; fi')
    return [LogInfo(msg=f"[hardware.launch] RPLIDAR A3 -> /scan on {port} "
                        "(apt sllidar_ros2 via slam_env.sh)."),
            ExecuteProcess(cmd=["bash", "-c", cmd], output="screen", name="rplidar")]


def generate_launch_description():
    base = LaunchConfiguration("base")
    arms = LaunchConfiguration("arms")
    body = LaunchConfiguration("body")
    hands = LaunchConfiguration("hands")
    sensors = LaunchConfiguration("sensors")
    status = LaunchConfiguration("status")
    lidar = LaunchConfiguration("lidar")
    audio = LaunchConfiguration("audio")
    # NOTE: rviz / rviz_config / slam / nav_2d_map / lidar_serial_port are read inside
    # _launch_slam_and_rviz (an OpaqueFunction), which needs their concrete values
    # to choose the RViz layout + shell out to the apt SLAM env.

    teensy_ip = LaunchConfiguration("teensy_ip")
    hand_sides = LaunchConfiguration("hand_sides")
    left_port = LaunchConfiguration("left_port")
    right_port = LaunchConfiguration("right_port")

    # Robot model + TF, from the SAME URDF the sim uses (derived from the MJCF).
    desc_pkg = get_package_share_directory("mabel_description")
    urdf_path = os.path.join(desc_pkg, "urdf", "mabel_full.urdf")
    with open(urdf_path) as f:
        robot_description = f.read()

    # RViz layout (opened in _launch_slam_and_rviz): hardware.rviz, Fixed Frame
    # base_link, with or without SLAM -- the robot + camera clouds always render
    # fully posed (no dependence on odom->base_link), and it already carries the
    # nvblox mesh + 2D /map displays for when SLAM is up.

    return LaunchDescription([
        # Reap any previous bring-up / sim FIRST (before our nodes spawn) so the
        # new bring-up owns the topic graph with no double-publishing.
        DeclareLaunchArgument("kill_existing", default_value="true",
                              description="Cleanly kill a previous MABEL bring-up "
                                          "or running sim before starting, to avoid "
                                          "double-publishing / DDS overflow."),
        OpaqueFunction(function=_kill_prior_bringup),

        # Real hardware runs on WALL time (no /clock). Be explicit so a stray
        # use_sim_time elsewhere never leaves these nodes waiting on a sim clock.
        SetParameter(name="use_sim_time", value=False),

        DeclareLaunchArgument("base", default_value="true",
                              description="Start the swerve base + lift driver."),
        DeclareLaunchArgument("arms", default_value="true",
                              description="Start the bimanual arm driver."),
        # Safe at default TRUE: unlike camera_node/lidar below, this launches only
        # the thin bridge CLIENT (liveness + e-stop mirror) and never opens the
        # sound card — the hardware_bridge is the sole owner, so there is no
        # double-open to guard against.
        DeclareLaunchArgument("audio", default_value="true",
                              description="Start the speaker/mic bridge client "
                                          "(monitor + /mabel_cmd e-stop -> mic "
                                          "mute). The hardware_bridge owns the "
                                          "sound card; this opens no device."),
        # Default TRUE. As of the 2026-07 neck migration this node is a THIN CLIENT
        # (mabel_body_hardware/body_node.py = BodyBridgeClient): it opens no device —
        # the hardware_bridge HAL now owns the neck's Dynamixel bus (MABEL_HW_NECK) and
        # serves /body/command<->/body/joint_states, exactly like the arms. This node
        # only monitors liveness and mirrors e-stop to /body/torque_enable. Leave it
        # true so the neck's watchdog row + e-stop mirror run alongside the HAL.
        DeclareLaunchArgument("body", default_value="true",
                              description="Start the neck/body bridge CLIENT "
                                          "(mabel_body_hardware — thin HAL client; the "
                                          "hardware_bridge owns the Dynamixel neck)."),
        DeclareLaunchArgument("hands", default_value="true",
                              description="Start the ORCA hand driver(s)."),
        DeclareLaunchArgument("hand_sides", default_value="both",
                              description="Which hand(s) to start: left | right | "
                                          "both. Use left/right when only one hand "
                                          "is wired (avoids both nodes grabbing the "
                                          "same adapter)."),
        DeclareLaunchArgument("sensors", default_value="true",
                              description="Start the camera drivers."),
        # When the hardware_bridge owns the cameras (the normal case: the bridge is
        # the single HAL for front D435i + ZED head + wrist D405/UVC and serves the
        # MJPEG/UDP video itself), the conda camera node is fully redundant --
        # real_run.sh passes camera_node:=false so it never runs and never fights
        # the bridge for the devices / :8080. The cameras/camera_serve args below
        # only matter for a standalone launch with camera_node:=true.
        # DEFAULT FALSE: the hardware_bridge is the SOLE camera owner (it opens the
        # D435i/ZED/D405 via the vendor SDKs and serves :8080/:9091/:9092 itself).
        # This used to default TRUE, so the architecture was safe only because
        # real_run.sh happened to pass camera_node:=false — a bare
        # `ros2 launch mabel_bringup hardware.launch.py` double-opened the USB
        # devices and double-bound :8080 against the bridge. Safe by default now;
        # opt IN if you deliberately want the legacy conda node instead of the bridge.
        DeclareLaunchArgument("camera_node", default_value="false",
                              description="Run the LEGACY conda camera node "
                                          "(sensors/ros_camera_node.py). Default "
                                          "false: the hardware_bridge owns the "
                                          "cameras. Only set true if the bridge's "
                                          "camera subsystem is off — never both."),
        DeclareLaunchArgument("cameras", default_value="all",
                              description="Depth cams the camera node owns: all | "
                                          "realsense | zed | none (wrists only)."),
        DeclareLaunchArgument("camera_serve", default_value="true",
                              description="Camera node also serves MJPEG on :8080 "
                                          "(set false when the bridge serves it)."),
        # DEFAULT FALSE: the bridge's LIDAR subsystem owns the RPLIDAR's CP2102 with
        # an EXCLUSIVE (TIOCEXCL) open and publishes the same /scan + lidar_link
        # contract. The apt sllidar node fighting it for the serial port is a known
        # way to wedge the lidar. Was default true and only disabled by real_run.sh
        # passing lidar:=false — the same safe-only-by-accident pattern as camera_node.
        DeclareLaunchArgument("lidar", default_value="false",
                              description="Start the LEGACY apt sllidar_ros2 driver "
                                          "(-> /scan, frame lidar_link). Default "
                                          "false: the hardware_bridge's LIDAR "
                                          "subsystem owns the serial port and "
                                          "publishes /scan. Never run both."),
        DeclareLaunchArgument("lidar_serial_port", default_value="/dev/ttyUSB0",
                              description="Serial port of the RPLIDAR A3."),
        DeclareLaunchArgument("status", default_value="true",
                              description="Show the pinned bring-up status "
                                          "dashboard (same panel as sim, "
                                          "real-robot signals)."),
        DeclareLaunchArgument("rviz", default_value="true",
                              description="Launch RViz with the full sensor "
                                          "layout (same view as sim)."),
        DeclareLaunchArgument("rviz_config", default_value="",
                              description="RViz layout file ('' = hardware.rviz, "
                                          "Fixed Frame base_link: robot + clouds + "
                                          "SLAM mesh/map, used with or without SLAM)."),
        DeclareLaunchArgument("slam", default_value="true",
                              description="Also bring up visual SLAM (cuVSLAM + "
                                          "nvblox) from the front RealSense, like "
                                          "the sim server: live 3D mesh + 2D /map + "
                                          "map->odom. Runs in the apt Isaac ROS env "
                                          "via scripts/slam_env.sh. slam:=false for "
                                          "a hardware-only bench bring-up."),
        DeclareLaunchArgument("nav_2d_map", default_value="true",
                              description="With slam:=true, also publish the nvblox "
                                          "ESDF 2D /map OccupancyGrid (for RViz + "
                                          "Nav2). false = 3D mesh only."),
        DeclareLaunchArgument("slam_odom", default_value="false",
                              description="false (default, real robot): wheel "
                                          "swerve_odometry owns odom->base_link and "
                                          "cuVSLAM only corrects map->odom. true "
                                          "(self-contained, slam_toolbox-style): "
                                          "cuVSLAM OWNS odom->base_link from the "
                                          "static front camera and swerve_odometry's "
                                          "TF is turned OFF — so visual SLAM generates "
                                          "odom + 3D mesh + 2D map with NO wheel odom "
                                          "(needed on the bench / before the base is "
                                          "wired). Exactly one source must own the TF."),
        DeclareLaunchArgument("teensy_ip", default_value="192.168.124.10",
                              description="IP of the MAXSwerve Teensy (base driver)."),
        DeclareLaunchArgument("left_port", default_value="",
                              description="Serial port for the LEFT ORCA hand "
                                          "('' = auto-detect the one plugged-in adapter)."),
        DeclareLaunchArgument("right_port", default_value="none",
                              description="Serial port for the RIGHT ORCA hand. "
                                          "Default 'none' = OFF, so a single-adapter "
                                          "bench comes up unambiguously as the LEFT "
                                          "hand. Pass a real port (or by-id path) to "
                                          "enable the right hand."),

        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            output="screen",
            parameters=[{"robot_description": robot_description,
                         "use_sim_time": False}],
        ),

        # ALWAYS-complete /joint_states for the TF tree. Unlike sim (where
        # mujoco_node publishes every joint on /joint_states), the real drivers
        # each publish only their OWN subsystem's joints on separate topics, and
        # publish nothing when their hardware is not connected -- so without this
        # most links would have no transform and collapse onto the origin in RViz.
        # joint_state_publisher publishes the FULL joint set at home defaults and
        # OVERLAYS live feedback from source_list as each driver comes online, so
        # the robot is always posed and its joints update the moment hardware is
        # wired. The base is moved to /base/joint_states (below) so it merges here
        # rather than fighting for /joint_states.
        Node(
            package="joint_state_publisher",
            executable="joint_state_publisher",
            name="joint_state_publisher",
            output="screen",
            parameters=[{
                "robot_description": robot_description,
                "use_sim_time": False,
                "rate": 30.0,
                # Hands merge in directly from the bridge's per-side feedback. The
                # bridge now publishes URDF-native names on /<side>_hand/joint_states
                # (thumb remapped at the source, hands_subsystem._THUMB_TO_URDF), so the
                # old hand_joint_state_relay (which only renamed + merged onto
                # /hands/joint_states) is gone — one canonical hand-state topic per side.
                # The lift is the bridge's single "lift" joint (metres of stroke) split
                # into the URDF's two prismatic stages (lift_lower/lift_upper) by the base
                # bridge client on /lift/joint_states_urdf — without it the column renders
                # frozen at 0 while the real lift moves (the bridge's own /lift/joint_states
                # carries a joint name the URDF doesn't have).
                "source_list": ["/base/joint_states", "/arms/joint_states",
                                "/body/joint_states",
                                "/left_hand/joint_states", "/right_hand/joint_states",
                                "/lift/joint_states_urdf"],
                # Default ARM pose while /arms/joint_states is silent (arms
                # unpowered / Teensy link down). The URDF home pose points the arms
                # straight FORWARD, so an offline robot rendered as a zombie-armed
                # robot while the real arms hung limp at its sides — read as "the
                # arms are wrongly calibrated" when in fact there was simply no data.
                # A powered-down arm HANGS, so hanging is the honest default. Values
                # are the natural-hang reference pose (shoulder/elbow/wrist/palm on
                # one vertical line); single source of truth + derivation:
                # the arms' natural-hang reference pose (was tools/arms_capture_zero.py REF_HANG).
                # Live feedback still overrides this the moment the arms stream.
                "zeros": {
                    "left_arm_1": 0.2685, "left_arm_2": -0.1745,
                    "left_arm_3": 0.4241, "left_arm_4": 1.0388,
                    "left_arm_5": 0.0852, "left_arm_6": -0.0241,
                    "left_arm_7": -0.2463,
                    "right_arm_1": -0.2670, "right_arm_2": 0.1745,
                    "right_arm_3": -0.4140, "right_arm_4": -1.0417,
                    "right_arm_5": -0.0839, "right_arm_6": 0.0189,
                    "right_arm_7": 0.2440,
                },
            }],
        ),

        IncludeLaunchDescription(
            _sub("mabel_base_hardware", "base.launch.py"),
            condition=IfCondition(base),
            # Base encoders feed the joint_state_publisher merge (source_list),
            # not /joint_states directly, so there is a single TF source. When
            # slam_odom:=true, cuVSLAM owns odom->base_link, so turn the wheel
            # swerve_odometry TF OFF (exactly one publisher of odom->base_link).
            launch_arguments={
                "teensy_ip": teensy_ip,
                "joint_states_topic": "/base/joint_states",
                "publish_odom_tf": PythonExpression(
                    ["'false' if '", LaunchConfiguration("slam_odom"),
                     "' == 'true' else 'true'"]),
            }.items()),
        IncludeLaunchDescription(
            _sub("mabel_arms_hardware", "arms.launch.py"),
            condition=IfCondition(arms)),
        IncludeLaunchDescription(
            _sub("mabel_body_hardware", "body.launch.py"),
            condition=IfCondition(body)),
        IncludeLaunchDescription(
            _sub("mabel_audio_hardware", "audio.launch.py"),
            condition=IfCondition(audio)),
        # Hands are owned by the hardware bridge (Feetech serial); ROS never opens the
        # device. The bridge publishes URDF-native /<side>_hand/joint_states which the
        # joint_state_publisher above merges straight into /joint_states (source_list)
        # so RViz mirrors the real hands — no relay node. The bridge (run_bridge.sh / its
        # systemd unit) must be up with the hand subsystem enabled and the ports set.
        IncludeLaunchDescription(
            _sub("mabel_sensors", "sensors.launch.py"),
            launch_arguments={"depth_cams": LaunchConfiguration("camera_node"),
                              "cameras": LaunchConfiguration("cameras"),
                              "camera_serve": LaunchConfiguration("camera_serve")}.items(),
            condition=IfCondition(sensors)),

        # Pinned, at-a-glance bring-up dashboard -- the SAME panel as the sim
        # (mabel_sim/bringup_status.py), in its "hardware" profile: real driver
        # topics from the interface_config contract, a live "online" count (the
        # robot has no /clock for an RTF), and one row per subsystem that mirrors
        # the toggles above. Disable with status:=false.
        Node(
            package="mabel_sim",
            executable="bringup_status",
            name="bringup_status",
            output="screen",
            emulate_tty=True,
            parameters=[{
                "use_sim_time": False,
                "mode": "hardware",
                "sensors": ParameterValue(sensors, value_type=bool),
                "wheel_odom": ParameterValue(base, value_type=bool),
                "arms": ParameterValue(arms, value_type=bool),
                "body": ParameterValue(body, value_type=bool),
                "hands": ParameterValue(hands, value_type=bool),
                # The dashboard's RPLIDAR /scan row is ALWAYS shown: `lidar` here
                # only gates the OLD apt sllidar driver (which real_run disables
                # with lidar:=false now that the hardware_bridge owns the RPLIDAR
                # and publishes /scan itself) — the row must not vanish with it.
                "lidar": True,
                # Same reasoning as the /scan row above: `audio` gates the ROS
                # CLIENT package, but the AUDIO rows report the hardware_bridge's
                # sound card (via /audio/device_health) — so the rows must stay
                # even when the client is off, or the panel goes quiet about the
                # speaker exactly when you most want to know it is there.
                "audio": True,
            }],
            condition=IfCondition(status),
        ),

        # RPLIDAR A3 -> /scan (apt sllidar_ros2 via slam_env.sh; graceful skip if
        # the driver isn't installed). lidar:=false to disable.
        OpaqueFunction(function=_launch_lidar),

        # Visual SLAM (cuVSLAM + nvblox, apt Isaac ROS) from the front RealSense +
        # RViz LAST, so RViz subscribes to a graph that already has the SLAM
        # displays' topics. Both are built in an OpaqueFunction because SLAM must
        # shell out to its own (apt) environment and RViz's default layout depends
        # on whether SLAM is on. slam:=false / rviz:=false to skip either.
        OpaqueFunction(function=_launch_slam_and_rviz),
    ])
