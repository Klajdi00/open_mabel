"""Boot the MABEL MuJoCo simulation -- PURE simulation: physics + sensors + RViz.

    ros2 launch mabel_sim sim.launch.py                      # robot + sensors + RViz
    ros2 launch mabel_sim sim.launch.py rviz:=false          # headless (no RViz)
    ros2 launch mabel_sim sim.launch.py foxglove:=true       # + Foxglove bridge
    ros2 launch mabel_sim sim.launch.py scene:=/path/to/scene.xml

NO SLAM, NO Nav2, NO autonomy. This launch starts the simulated robot + its raw
sensor data + RViz of that data: MuJoCo physics + the camera/LiDAR sensor bridge +
robot_state_publisher + the robot's own WHEEL ODOMETRY (swerve_odometry) + RViz.
The RViz layout (sim_sensors.rviz) shows ONLY simulated sensors -- robot model,
LiDAR scan, camera images, depth point clouds -- nothing map/costmap/path-related.

POSE in RViz is an ESTIMATE, never sim ground truth (the TurtleBot/Gazebo model):
  * /odom + odom->base_link come from swerve_odometry, which DEAD-RECKONS the pose
    from /joint_states alone (wheel + steer encoder feedback) -- the SAME node the
    real base runs. So the robot you see in RViz is where the robot THINKS it is
    (it drifts), not where it truly is. SLAM, added later, publishes map->odom to
    correct that drift -- and only then does the estimate get good.
  * The sim's TRUE pose is published ONLY on /odom_ground_truth, and is used ONLY
    internally to render the cameras/LiDAR from the robot's real location (a real
    sensor sees the real world). Nothing in RViz reads it. This is the single,
    unavoidable "privilege" -- the same one Gazebo uses to render its sensors.

It publishes the SAME topics the real hardware does (/joint_states, /odom,
/mabel_cmd, ...) so the SLAM/Nav layers compose ON TOP in their OWN launches/RViz:
  * mabel_slam {lidar_slam,visual_slam}.launch.py  -- mapping + map->odom correction
  * mabel_navigation {navigation,exploration}.launch.py  -- Nav2 (its own RViz)
  * mabel_teleop teleop.launch.py                  -- keyboard driving

robot_state_publisher always runs (the URDF TF tree is needed to render the robot
and is consumed by any layer added later), driven by the simulated /joint_states
and swerve_odometry's odom->base_link transform.

Foxglove: with foxglove:=true, a foxglove_bridge WebSocket server starts on
ws://localhost:8765. Open Foxglove Studio -> "Open connection" -> Foxglove
WebSocket -> ws://localhost:8765. Requires `ros-humble-foxglove-bridge`.
"""

import os
import subprocess
import sys
import time

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node, SetParameter
from launch_ros.parameter_descriptions import ParameterValue


# Sim-OWNED processes to reap before a new launch (the "one physics at a time"
# rule). Matched by their install-lib paths / RViz config names so we ONLY ever
# kill a *previous* sim instance -- never an unrelated rviz / robot_state_publisher,
# and never THIS launch (whose own nodes have not spawned yet when this runs).
_SIM_PROC_PATTERNS = [
    "lib/mabel_sim/mujoco_node",
    "lib/mabel_sim/mujoco_sensors_node",
    "lib/mabel_sim/bringup_status",
    "lib/mabel_base_hardware/swerve_odometry",
    "rviz2.*sim_sensors.rviz",
]


def _nvidia_egl_device(vendor_dir):
    """Return the EGL device index whose driver is NVIDIA (as a str), or None.

    Runs the EGL enumeration in a SUBPROCESS (with the same vendor dir the sensor
    node will use) so importing/binding EGL never pollutes this launch process, and
    so a probe failure (no PyOpenGL, odd driver) just falls back to the default
    instead of breaking the launch. NVIDIA devices advertise ``EGL_NV_device_cuda``
    in their per-device extension string (EGL_EXTENSIONS, 0x3055)."""
    probe = (
        "import os,ctypes;os.environ.pop('MUJOCO_EGL_DEVICE_ID',None);"
        "from OpenGL import EGL;"
        "q=ctypes.CFUNCTYPE(ctypes.c_uint,ctypes.c_int,ctypes.POINTER(ctypes.c_void_p),"
        "ctypes.POINTER(ctypes.c_int))(EGL.eglGetProcAddress(b'eglQueryDevicesEXT'));"
        "s=ctypes.CFUNCTYPE(ctypes.c_char_p,ctypes.c_void_p,ctypes.c_int)"
        "(EGL.eglGetProcAddress(b'eglQueryDeviceStringEXT'));"
        "d=(ctypes.c_void_p*16)();n=ctypes.c_int(0);q(16,d,ctypes.byref(n));"
        "[print(i) for i in range(n.value) if b'EGL_NV_device' in (s(d[i],0x3055) or b'')]"
    )
    env = dict(os.environ)
    if vendor_dir:
        env["__EGL_VENDOR_LIBRARY_DIRS"] = vendor_dir
    try:
        out = subprocess.run([sys.executable, "-c", probe], env=env,
                             capture_output=True, text=True, timeout=20)
        for tok in out.stdout.split():
            if tok.strip().isdigit():
                return tok.strip()
    except Exception:       # noqa: BLE001  -- a probe failure must never break launch
        pass
    return None


def _kill_prior_sim(context, *args, **kwargs):
    """Enforce ONE sim at a time: kill any lingering sim-owned processes left by a
    previous (crashed / Ctrl-C'd) launch before this one starts, so the new sim
    owns the physics, the GL/EGL device and the topic graph cleanly."""
    if context.perform_substitution(LaunchConfiguration("kill_existing")) != "true":
        return []
    reaped = False
    for pat in _SIM_PROC_PATTERNS:
        # pkill returns 0 only if it matched+signalled at least one process.
        if subprocess.run(["pkill", "-9", "-f", pat]).returncode == 0:
            reaped = True
    if reaped:
        time.sleep(0.7)   # let the OS release the GL device + DDS endpoints
        print("[sim.launch] reaped a previous sim instance (one-physics rule).")
    return []


def generate_launch_description():
    desc_pkg = get_package_share_directory("mabel_description")
    sim_pkg = get_package_share_directory("mabel_sim")
    urdf_path = os.path.join(desc_pkg, "urdf", "mabel_full.urdf")

    # Force the EGL sensor renderer onto the NVIDIA GPU. libglvnd binds its EGL
    # vendor ICD from __EGL_VENDOR_LIBRARY_DIRS at PROCESS START (before `import
    # mujoco`), so the node CANNOT fix this itself -- it must be set here, in the
    # child's env. A conda/venv activation often exports a dir holding ONLY the
    # Mesa software ICD, which silently drops EGL to llvmpipe (CPU): ~300 ms/frame
    # vs ~0.4 ms on the GPU here. Point it at the SYSTEM vendor dir (which lists the
    # NVIDIA ICD) -- but only when that NVIDIA ICD actually exists there, so a board
    # with a different layout (e.g. some Jetson setups) keeps its own working env.
    _sys_egl_dir = "/usr/share/glvnd/egl_vendor.d"
    sensors_env = {}
    if os.path.exists(os.path.join(_sys_egl_dir, "10_nvidia.json")):
        sensors_env["__EGL_VENDOR_LIBRARY_DIRS"] = _sys_egl_dir
        # Pointing at the right ICD only makes the NVIDIA device VISIBLE; on an
        # Optimus laptop EGL_DEFAULT_DISPLAY still resolves to the Intel iGPU (~14
        # ms/frame) or even the llvmpipe software device (~300 ms), not the dGPU
        # (~1 ms). So also pin MUJOCO_EGL_DEVICE_ID to the NVIDIA device, which makes
        # MuJoCo SELECT it by enumeration. Auto-detected; override with
        # MABEL_EGL_DEVICE=<idx>; falls back to device 0 (the NVIDIA card on the 4090
        # box, and the single GPU on a Jetson -- so this stays correct there too).
        sensors_env["MUJOCO_EGL_DEVICE_ID"] = (
            os.environ.get("MABEL_EGL_DEVICE")
            or _nvidia_egl_device(_sys_egl_dir) or "0")
    # Default RViz layout: the full SENSOR view -- robot + every camera RGB (front/
    # back/ZED head/wrists/body) + LiDAR + the front, back AND ZED-head point clouds.
    # This is the "see everything the sim publishes" layout for a standalone bring-up.
    default_rviz = os.path.join(sim_pkg, "rviz", "sim_sensors.rviz")
    with open(urdf_path) as f:
        robot_description = f.read()

    scene = LaunchConfiguration("scene")
    rviz = LaunchConfiguration("rviz")
    rviz_config = LaunchConfiguration("rviz_config")
    foxglove = LaunchConfiguration("foxglove")
    fox_port = LaunchConfiguration("foxglove_port")
    sensors = LaunchConfiguration("sensors")
    physics = LaunchConfiguration("physics")
    publish_odom = LaunchConfiguration("publish_odom")
    wheel_odom = LaunchConfiguration("wheel_odom")
    wheel_odom_tf = LaunchConfiguration("wheel_odom_tf")
    wheel_radius = LaunchConfiguration("wheel_radius")
    headless = LaunchConfiguration("headless")
    sensor_rate = LaunchConfiguration("sensor_rate")
    imu = LaunchConfiguration("imu")
    imu_rate = LaunchConfiguration("imu_rate")
    imu_accel_noise = LaunchConfiguration("imu_accel_noise")
    imu_gyro_noise = LaunchConfiguration("imu_gyro_noise")
    render_backend = LaunchConfiguration("render_backend")
    pointcloud_stride = LaunchConfiguration("pointcloud_stride")
    lidar_noise_scale = LaunchConfiguration("lidar_noise_scale")
    depth_noise_base = LaunchConfiguration("depth_noise_base")
    depth_noise_k = LaunchConfiguration("depth_noise_k")
    depth_noise_lat = LaunchConfiguration("depth_noise_lat")
    lidar_dropout_prob = LaunchConfiguration("lidar_dropout_prob")
    lidar_outlier_prob = LaunchConfiguration("lidar_outlier_prob")
    depth_hole_prob = LaunchConfiguration("depth_hole_prob")
    depth_outlier_prob = LaunchConfiguration("depth_outlier_prob")
    front_only = LaunchConfiguration("front_only")
    status = LaunchConfiguration("status")

    return LaunchDescription([
        # ONE sim at a time: declare the toggle, then reap any leftover sim from a
        # previous launch BEFORE starting anything (runs synchronously, first).
        DeclareLaunchArgument(
            "kill_existing", default_value="true",
            description="Before starting, kill any lingering sim processes from a "
                        "previous launch (one-physics rule). Set false to keep them."),
        OpaqueFunction(function=_kill_prior_sim),

        # mujoco_node is the SIMULATION TIME AUTHORITY: it publishes /clock from the
        # sim clock and stamps every message with sim time. EVERY node here runs on
        # sim time so odometry/SLAM integrate correctly even when the sim runs below
        # real time (RTF<1) -- the cause of the old ~3x odometry over-read.
        SetParameter(name="use_sim_time", value=True),
        DeclareLaunchArgument(
            "scene", default_value="",
            description="MuJoCo scene xml (empty = package default)."),
        DeclareLaunchArgument(
            "rviz", default_value="true",
            description="Launch RViz (+ robot_state_publisher)."),
        DeclareLaunchArgument(
            "rviz_config", default_value=default_rviz,
            description="RViz layout file."),
        DeclareLaunchArgument(
            "foxglove", default_value="false",
            description="Launch the Foxglove Studio WebSocket bridge."),
        DeclareLaunchArgument(
            "foxglove_port", default_value="8765",
            description="Port for the Foxglove WebSocket bridge."),
        DeclareLaunchArgument(
            "sensors", default_value="true",
            description="Start the camera + LiDAR sensor bridge (mujoco_sensors_node)."),
        DeclareLaunchArgument(
            "physics", default_value="true",
            description="Run the in-launch MuJoCo PHYSICS node (mujoco_node) -- the sim's "
                        "physics + /clock + /joint_states + /odom_ground_truth source. Set "
                        "false when an EXTERNAL physics authority owns those (the teleop "
                        "server's sim_teleop_bridge in sim:mujoco+ros mode -- see "
                        "sim_ros_server.sh): then this launch runs ONLY the sensor renderer "
                        "+ swerve_odometry + robot_state_publisher + RViz, all fed by the "
                        "bridge's identical-to-Mac physics. There must be exactly ONE /clock "
                        "publisher, so never run physics:=true alongside that bridge."),
        DeclareLaunchArgument(
            "wheel_odom", default_value="true",
            description="Start swerve_odometry -- the robot's WHEEL ODOMETRY motion "
                        "model: /joint_states (encoder feedback only, NO sim privilege) "
                        "-> dead-reckoned /odom + odom->base_link. This is the SAME node "
                        "the real base runs (mabel_base_hardware/base.launch.py), so /odom "
                        "is a realistic (drifting) ESTIMATE -- exactly how a TurtleBot's "
                        "diff-drive odom behaves -- not sim ground truth. RViz then shows "
                        "the robot at its ESTIMATED pose; SLAM corrects map->odom on top. "
                        "The sim's TRUE pose stays on /odom_ground_truth (used ONLY to "
                        "render the cameras/LiDAR from where the robot really is). Set "
                        "false to instead let MuJoCo publish ground-truth /odom (debug "
                        "only -- that IS reading privileged sim state)."),
        DeclareLaunchArgument(
            "wheel_odom_tf", default_value="true",
            description="Whether swerve_odometry publishes the odom->base_link TF. Set "
                        "false when an EKF (robot_localization) fuses wheel + visual and "
                        "owns the TF -- swerve then publishes only /odom as an EKF input."),
        DeclareLaunchArgument(
            "wheel_radius", default_value="0.0378",
            description="Swerve wheel rolling radius (m); sim=0.0378, real REV MAXSwerve=0.0381."),
        DeclareLaunchArgument(
            "publish_odom",
            default_value=PythonExpression(
                ["'false' if '", wheel_odom, "' == 'true' else 'true'"]),
            description="Whether the SIM owns /odom + odom->base_link (ground truth). Default "
                        "FOLLOWS wheel_odom: when wheel_odom:=true the sim yields odom to "
                        "swerve_odometry (truth still on /odom_ground_truth). Override only to "
                        "force one source; do not set true together with wheel_odom:=true."),
        DeclareLaunchArgument(
            "headless", default_value="false",
            description="Run physics without the on-screen MuJoCo viewer."),
        DeclareLaunchArgument(
            "imu", default_value="true",
            description="Publish the front D435i IMU (sensor_msgs/Imu on "
                        "/camera/front/imu) from the MuJoCo accelerometer/gyro, with "
                        "simulated BMI055 noise. Feeds cuVSLAM IMU fusion."),
        DeclareLaunchArgument(
            "imu_rate", default_value="200.0",
            description="IMU publish rate (Hz); real D435i BMI055 runs ~200-400."),
        DeclareLaunchArgument(
            "imu_accel_noise", default_value="0.0147",
            description="IMU accelerometer white-noise std (m/s^2). 0 = noiseless."),
        DeclareLaunchArgument(
            "imu_gyro_noise", default_value="0.0024",
            description="IMU gyro white-noise std (rad/s). 0 = noiseless."),
        DeclareLaunchArgument(
            "sensor_rate", default_value="15.0",
            description="Camera/LiDAR publish rate (Hz)."),
        DeclareLaunchArgument(
            "render_backend", default_value="egl",
            description="Offscreen GL backend: egl (headless GPU) | glfw | osmesa."),
        DeclareLaunchArgument(
            "pointcloud_stride", default_value="2",
            description="Depth pixel decimation for point clouds (2 = quarter density)."),
        DeclareLaunchArgument(
            "lidar_noise_scale", default_value="1.0",
            description="RPLIDAR A3 range-noise scale (1.0 = datasheet %, 0 = off)."),
        DeclareLaunchArgument(
            "depth_noise_base", default_value="0.001",
            description="D435 axial depth noise constant term (m)."),
        DeclareLaunchArgument(
            "depth_noise_k", default_value="0.005",
            description="D435 axial depth noise quadratic term: std = base + k*z^2."),
        DeclareLaunchArgument(
            "depth_noise_lat", default_value="0.0005",
            description="D435 lateral (x,y) point-cloud noise: std = lat*z."),
        DeclareLaunchArgument(
            "lidar_dropout_prob", default_value="0.02",
            description="Fraction of LiDAR beams that return no hit (+inf)."),
        DeclareLaunchArgument(
            "lidar_outlier_prob", default_value="0.01",
            description="Fraction of LiDAR beams replaced by a random range."),
        DeclareLaunchArgument(
            "depth_hole_prob", default_value="0.03",
            description="Fraction of depth pixels invalidated (stereo holes)."),
        DeclareLaunchArgument(
            "depth_outlier_prob", default_value="0.005",
            description="Fraction of depth pixels turned into flying-pixel outliers."),
        DeclareLaunchArgument(
            "front_only", default_value="false",
            description="Render only the front D435i (color+depth+IR) -- big GPU/mem "
                        "saving for SLAM runs; skips ZED/wrist/body/back cameras."),
        DeclareLaunchArgument(
            "status", default_value="true",
            description="Show the pinned colour bring-up STATUS dashboard at the top "
                        "of the terminal (physics/state/odom/LiDAR/camera/cloud "
                        "liveness + RTF). Set false for plain interleaved logs."),

        Node(
            package="mabel_sim",
            executable="mujoco_node",
            name="mabel_mujoco_sim",
            output="screen",
            # Force the GLFW backend for the on-screen viewer. If MUJOCO_GL=egl is
            # in the environment (set for the headless sensor renderer), the MuJoCo
            # passive viewer can't open a window with it -- so the 3D viewer window
            # silently never appears. mujoco_node does NO offscreen rendering, so
            # glfw is always correct here (and unused when headless:=true).
            additional_env={"MUJOCO_GL": "glfw"},
            parameters=[{
                # The time AUTHORITY stays on WALL time: its publisher timers fire at
                # wall rate and it stamps content with sim time directly + publishes
                # /clock. Putting use_sim_time:=true here would make its timers chase
                # its own /clock -> a feedback storm that starves physics (RTF -> ~2%).
                "use_sim_time": False,
                "scene": scene,
                "publish_odom": ParameterValue(publish_odom, value_type=bool),
                "headless": ParameterValue(headless, value_type=bool),
                "imu": ParameterValue(imu, value_type=bool),
                "imu_rate": ParameterValue(imu_rate, value_type=float),
                "imu_accel_noise": ParameterValue(imu_accel_noise, value_type=float),
                "imu_gyro_noise": ParameterValue(imu_gyro_noise, value_type=float),
            }],
            # Skipped when an external physics authority (the teleop bridge in
            # sim:mujoco+ros) owns physics + /clock; the rest of this launch then
            # runs on the bridge's published state. ONE /clock publisher only.
            condition=IfCondition(physics),
        ),

        # Camera + LiDAR rendering bridge (separate process: EGL offscreen,
        # reconstructs pose from /joint_states + /odom; see mujoco_sensors_node.py).
        Node(
            package="mabel_sim",
            executable="mujoco_sensors_node",
            name="mabel_mujoco_sensors",
            output="screen",
            # MUJOCO_GL MUST be in the child env BEFORE it imports mujoco: mujoco
            # binds its GL backend at import, so a value set later (as the node used
            # to) silently falls back to SOFTWARE EGL (llvmpipe, ~130 ms/frame) even
            # though it reports "egl". Setting it here (pre-exec) is what actually
            # lands rendering on the GPU. The EGL vendor/device pins ride along.
            additional_env={**sensors_env, "MUJOCO_GL": render_backend},
            parameters=[{
                "scene": scene,
                "sensor_rate": ParameterValue(sensor_rate, value_type=float),
                "render_backend": render_backend,
                "pointcloud_stride": ParameterValue(pointcloud_stride, value_type=int),
                "lidar_noise_scale": ParameterValue(lidar_noise_scale, value_type=float),
                "depth_noise_base": ParameterValue(depth_noise_base, value_type=float),
                "depth_noise_k": ParameterValue(depth_noise_k, value_type=float),
                "depth_noise_lat": ParameterValue(depth_noise_lat, value_type=float),
                "lidar_dropout_prob": ParameterValue(lidar_dropout_prob, value_type=float),
                "lidar_outlier_prob": ParameterValue(lidar_outlier_prob, value_type=float),
                "depth_hole_prob": ParameterValue(depth_hole_prob, value_type=float),
                "depth_outlier_prob": ParameterValue(depth_outlier_prob, value_type=float),
                "front_only": ParameterValue(front_only, value_type=bool),
            }],
            condition=IfCondition(sensors),
        ),

        # Pinned bring-up STATUS dashboard: a colour panel locked to the top of the
        # terminal (logs scroll below it) showing live liveness + Hz of physics,
        # /joint_states, odom, /tf, LiDAR, every camera and every point cloud --
        # the sim's answer to the real robot's per-driver "connected" log lines.
        # use_sim_time:=False so its redraw timer runs on the WALL clock (a slow RTF
        # never freezes the panel). It writes straight to /dev/tty, so it coexists
        # with -- and is independent of -- the other nodes' screen output.
        Node(
            package="mabel_sim",
            executable="bringup_status",
            name="bringup_status",
            output="screen",
            emulate_tty=True,
            parameters=[{
                "use_sim_time": False,
                "sensors": ParameterValue(sensors, value_type=bool),
                "wheel_odom": ParameterValue(wheel_odom, value_type=bool),
                "front_only": ParameterValue(front_only, value_type=bool),
                "foxglove": ParameterValue(foxglove, value_type=bool),
            }],
            condition=IfCondition(status),
        ),

        # Wheel odometry: swerve FK from /joint_states (real encoder feedback, NO sim
        # privilege) -> /odom + odom->base_link TF. This is the robot's own base odom,
        # exactly what the real base publishes, so sim.launch.py is a self-sufficient,
        # drivable HAL. cuVSLAM (visual_slam.launch.py) corrects map->odom on top.
        Node(
            package="mabel_base_hardware",
            executable="swerve_odometry",
            name="swerve_odometry",
            output="screen",
            parameters=[{
                "wheel_radius": ParameterValue(wheel_radius, value_type=float),
                "publish_tf": ParameterValue(wheel_odom_tf, value_type=bool),
            }],
            condition=IfCondition(wheel_odom),
        ),

        # Robot model + TF -- ALWAYS on. The URDF TF tree (base_link->lidar_link,
        # base_link->camera optical frames, ...) is needed by SLAM, Nav2 and the
        # controllers, not just the visualisers, so it must run even headless
        # (rviz:=false). This mirrors hardware.launch.py, which always starts it.
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            output="screen",
            parameters=[{"robot_description": robot_description}],
        ),

        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            output="screen",
            arguments=["-d", rviz_config],
            condition=IfCondition(rviz),
        ),

        # Foxglove Studio bridge (WebSocket server on ws://localhost:<port>).
        Node(
            package="foxglove_bridge",
            executable="foxglove_bridge",
            name="foxglove_bridge",
            output="screen",
            parameters=[{
                "port": ParameterValue(fox_port, value_type=int),
                "address": "0.0.0.0",
                "tls": False,
                "topic_whitelist": [".*"],
                "send_buffer_limit": 10000000,
                # serve package:// meshes so the URDF renders in the 3D panel
                "asset_uri_allowlist": [
                    "^package://(?:\\w+/)*\\w+\\.(?:dae|fbx|glb|gltf|jpeg|jpg|"
                    "mtl|obj|png|stl|tif|tiff|urdf|webp|xacro)$"],
                "use_sim_time": True,
            }],
            condition=IfCondition(foxglove),
        ),
    ])
