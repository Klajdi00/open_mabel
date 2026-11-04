"""Camera + LiDAR contract for the MABEL MuJoCo sensor bridge.

Single source of truth (mirrors interface_config.py) mapping each MuJoCo
``<camera>``/``<rangefinder>`` defined in
``simulation/mabel_mujoco/models/mabel_full.xml`` to the ROS topics, optical
frames and render sizes the ``mujoco_sensors_node`` publishes.

Naming follows the real-robot RealSense conventions but in *sim* namespaces
(``/camera/front/*`` ...), so the visual-SLAM stack is wired via the
``visual_slam.launch.py`` remappings rather than the hardware launch files.

Frame convention: every camera publishes a ROS *optical* frame
(+z forward, +x right, +y down) parented to ``BASE_FRAME``; the node computes
the transform from the live MuJoCo camera pose each tick, so head/wrist cameras
move correctly with the neck/arms.
"""

# Root frame every camera optical frame is parented to (matches the sim odom TF).
BASE_FRAME = "base_link"

# --------------------------------------------------------------------------- #
#  Mono RGB cameras  --  publish color/image_raw (rgb8) + camera_info
#  (mujoco_cam_name, ros_namespace, optical_frame, render_w, render_h)
# --------------------------------------------------------------------------- #
MONO_CAMERAS = [
    ("body_cam",    "body",        "body_optical_frame",        320, 240),
    ("l_wrist_cam", "left_wrist",  "left_wrist_optical_frame",  320, 240),
    ("r_wrist_cam", "right_wrist", "right_wrist_optical_frame", 320, 240),
    ("zed_left",    "zed/left",    "zed_left_optical_frame",    640, 480),
    ("zed_right",   "zed/right",   "zed_right_optical_frame",   640, 480),
]

# --------------------------------------------------------------------------- #
#  D435i RGB-D cameras  --  color + depth (32FC1, metres) + XYZRGB point cloud
#  dict: mj (color/depth optical centre), ns, color_frame, depth_frame, w, h
# --------------------------------------------------------------------------- #
DEPTH_CAMERAS = [
    {"mj": "body_front_cam", "ns": "front",
     "color_frame": "front_color_optical_frame",
     "depth_frame": "front_depth_optical_frame", "w": 640, "h": 360},
    {"mj": "body_back_cam",  "ns": "back",
     "color_frame": "back_color_optical_frame",
     "depth_frame": "back_depth_optical_frame",  "w": 640, "h": 360},
]

# --------------------------------------------------------------------------- #
#  IR stereo (mono8) for cuVSLAM  --  only the front D435i has the synthesised
#  infra1/infra2 pair (added to the XML at a 50 mm baseline).
#  (mujoco_cam_name, ros_namespace, optical_frame, render_w, render_h)
# --------------------------------------------------------------------------- #
# Render at the D435 NATIVE 848x480 infra profile (the MJCF cameras already
# declare 848x480). cuVSLAM is feature-based: ~1.77x more pixels per axis than
# the old 640x360 means more keypoints and sub-pixel-sharper localization, which
# directly cuts visual-odometry drift. The mono render is cheap relative to the
# colour/depth passes, so this is a low-cost accuracy win.
IR_CAMERAS = [
    ("front_ir_left",  "front/infra1", "front_infra1_optical_frame", 848, 480),
    ("front_ir_right", "front/infra2", "front_infra2_optical_frame", 848, 480),
]

# Camera_info / point-cloud depth clamp (m). D435 is unreliable past ~5 m indoors.
DEPTH_MAX_M = 8.0

# --------------------------------------------------------------------------- #
#  ZED head stereo DEPTH  --  the ZED Mini is a passive stereo camera; the real
#  ``zed-ros2-wrapper`` emits a registered, colourised point cloud from the left
#  eye. We reproduce that head cloud: render depth from the ``zed_left`` MuJoCo
#  camera and publish an XYZRGB cloud on ``/camera/zed/depth/points`` (+ a
#  depth/image_rect_raw and depth/camera_info), so the operator gets a 3-D view
#  from the robot's head just like the real ZED.
#
#  It rides the SAME generic depth path as the D435i cameras (``_do_depth``), but
#  with two per-camera flags so it does NOT duplicate the ZED's existing outputs:
#    publish_color=False -> no redundant ``/camera/zed/color/image_raw`` (the RGB
#                           already comes from the zed_left/zed_right mono cams).
#    depth_frame is its OWN ``zed_depth_optical_frame`` (co-located with zed_left)
#                           and emit_tf=True, so the cloud is self-contained and
#                           never collides with / duplicates the zed_left TF.
#  Same render size (640x480) as the ZED mono eyes; runs in the AUX worker.
# --------------------------------------------------------------------------- #
STEREO_DEPTH_CAMERAS = [
    {"mj": "zed_left", "ns": "zed",
     "color_frame": "zed_left_optical_frame",
     "depth_frame": "zed_depth_optical_frame", "w": 640, "h": 480,
     "publish_color": False, "emit_tf": True},
]

# --------------------------------------------------------------------------- #
#  LiDAR  --  sensor_msgs/LaserScan from the 360 replicated rangefinder rays.
# --------------------------------------------------------------------------- #
LIDAR_TOPIC = "/scan"
LIDAR_FRAME = "lidar_link"
LIDAR_BODY = "lidar"          # MuJoCo body the ray sites hang off (for the TF)

# --------------------------------------------------------------------------- #
#  Defaults (overridable via ROS params on the node)
# --------------------------------------------------------------------------- #
# Per-GROUP publish rates. The old node rendered ALL cameras + all CPU
# post-processing serially in ONE thread, so the SLAM-critical front point cloud
# was gated behind every other camera and only reached ~2 Hz in RViz. The node
# now runs three INDEPENDENT render workers (see mujoco_sensors_node.py), each
# with its own GL context + MjData + rate, so the front camera no longer waits on
# the others. Benchmarked on the 4090 (EGL): a single front-D435 full pipeline
# sustains ~59 Hz and a full 11-camera serial tick ~17 Hz, so these rates have
# ample headroom (and match the real sensors: D435 ~30, RPLIDAR A3 ~15).
# GPU budget note: the workers render CONCURRENTLY and share ONE GPU (plus RViz +
# the MuJoCo viewer). A full front tick ~25 ms and a full aux tick ~33 ms (4090,
# bench_sensors.py), so front@30 + aux@10 ~= 0.75+0.33 GPU-s/s leaves headroom.
# The front (SLAM) worker is deliberately prioritised; aux are viewing cameras.
# Watch the node's "render rates (achieved/target)" log and tune from there.
DEFAULT_FRONT_RATE = 30.0    # Hz; front D435i (color+depth+cloud) + IR stereo -- SLAM/nav
DEFAULT_AUX_RATE = 10.0      # Hz; back D435i + body/wrist/ZED mono cams (viewing, not SLAM)
DEFAULT_LIDAR_RATE = 15.0    # Hz; RPLIDAR A3 spins ~10-20 Hz
DEFAULT_RATE = 0.0           # legacy `sensor_rate`: 0 = use the per-group rates above;
                             # >0 overrides ALL groups (back-compat for old launches)
DEFAULT_BACKEND = "egl"      # offscreen GL backend: egl (headless GPU) | glfw | osmesa
