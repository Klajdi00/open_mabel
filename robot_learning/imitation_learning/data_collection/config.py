"""
Configuration for the MABEL teleop data-collection pipeline.

Everything physical (joint names, actuator order, camera names, lidar layout) is
derived from the *single* source of truth — the MuJoCo model and the shared
controller config (``controller/mabel/config.py``) — so the logger can never
drift from the robot definition. A handful of recording knobs (FPS, image size,
which cameras get depth) live here and are the only things you normally tune.

The same schema is used for SIM and REAL collection; only the StateProvider that
fills it in differs (see ``providers.py``).
"""
import os
import sys

# ── Repo bootstrap: make the shared ``mabel`` controller package importable ───
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# data_collection/ -> imitation_learning/ -> robot_learning/ -> MABEL/
_MABEL_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))
_CONTROLLER_DIR = os.path.join(_MABEL_ROOT, "controller")
_SIM_SCRIPTS = os.path.join(_MABEL_ROOT, "simulation", "mabel_mujoco", "scripts")
for _p in (_CONTROLLER_DIR, _SIM_SCRIPTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mabel import config as C  # noqa: E402  (the shared controller config)

# Absolute path to the scene the controller expects (config.SCENE_XML is relative
# to the sim ``scripts/`` directory).
SCENE_XML = os.path.abspath(os.path.join(_SIM_SCRIPTS, C.SCENE_XML))

# ── Dataset output ────────────────────────────────────────────────────────────
DATASET_ROOT = os.path.join(_THIS_DIR, "datasets")   # episodes land in <root>/<name>/
HDF5_COMPRESSION = "gzip"                              # for non-image arrays
JPEG_QUALITY = 90                                      # RGB encode quality

# ── Recording rate ────────────────────────────────────────────────────────────
# Physics runs at 1/model.opt.timestep (500 Hz). We log a frame every 1/LOG_FPS
# seconds (decoupled from physics so the heavy image render does not stall sim).
LOG_FPS = 15.0

# ── Cameras ───────────────────────────────────────────────────────────────────
# All 7 on-board cameras defined in models/mabel_full.xml. Each entry:
#   name : MuJoCo camera name
#   rgb  : log an RGB frame (JPEG)
#   depth: log a metric depth frame (uint16, millimetres, clipped to DEPTH_MAX)
#   w,h  : render resolution for THIS camera (downscaled from the model's native
#          1280x720 to keep episodes small; raise for higher fidelity)
# Depth defaults: ON for the manipulation-relevant views (wrists + a head view +
# the front body cam), OFF for the rear/secondary views to bound dataset size.
# Flip any ``depth`` flag to capture more — the logger honours it verbatim.
RGB_W, RGB_H = 480, 270
CAMERAS = [
    {"name": "l_wrist_cam",    "rgb": True, "depth": True,  "w": RGB_W, "h": RGB_H},
    {"name": "r_wrist_cam",    "rgb": True, "depth": True,  "w": RGB_W, "h": RGB_H},
    {"name": "zed_left",       "rgb": True, "depth": True,  "w": RGB_W, "h": RGB_H},
    {"name": "zed_right",      "rgb": True, "depth": False, "w": RGB_W, "h": RGB_H},
    {"name": "body_cam",       "rgb": True, "depth": True,  "w": RGB_W, "h": RGB_H},
    {"name": "body_front_cam", "rgb": True, "depth": True,  "w": RGB_W, "h": RGB_H},
    {"name": "body_back_cam",  "rgb": True, "depth": False, "w": RGB_W, "h": RGB_H},
]
DEPTH_MAX_M = 8.0          # metres; depth clipped to this then scaled into uint16
DEPTH_SCALE = 1000.0       # uint16 stores millimetres (range 0..8000)

# ── LiDAR ─────────────────────────────────────────────────────────────────────
# 360-ray planar scan, auto-replicated in the model as lidar_range000..359.
LIDAR_PREFIX = "lidar_range"
LIDAR_N_RAYS = 360
LIDAR_ANGLE_RES_DEG = 1.0   # see the <replicate euler> in mabel_full.xml

# ── Joint / actuator grouping (names match the model & controller) ────────────
# Ordered exactly as MuJoCo's actuator array (data.ctrl indices line up 1:1).
BASE_ACTUATORS = ["FL Steer", "FL Drive", "FR Steer", "FR Drive", "B Steer", "B Drive"]
LIFT_TORSO     = ["Lift", "Torso"]
NECK           = list(C.NECK_ACTUATORS)                       # 3
LEFT_ARM       = [f"Left Arm {i}" for i in range(1, 8)] + ["Left Wrist"]   # 8
RIGHT_ARM      = [f"Right Arm {i}" for i in range(1, 8)] + ["Right Wrist"] # 8
_FINGERS = ["Thumb MCP", "Thumb Abd", "Thumb PIP", "Thumb DIP",
            "Index Abd", "Index MCP", "Index PIP",
            "Middle Abd", "Middle MCP", "Middle PIP",
            "Ring Abd", "Ring MCP", "Ring PIP",
            "Pinky Abd", "Pinky MCP", "Pinky PIP"]
LEFT_HAND  = [f"Left {f}"  for f in _FINGERS]                  # 16
RIGHT_HAND = [f"Right {f}" for f in _FINGERS]                 # 16

# The Lift actuator drives a tendon; its underlying measured joint:
LIFT_JOINT = C.LIFT_JOINT   # "Lift Lower"

# Bodies whose world pose (xpos+xquat, 7-vec) we log every frame.
TRACKED_BODIES = {
    "chassis":    C.CHASSIS_BODY,
    "head":       C.HEAD_BODY,
    "left_palm":  C.LEFT_PALM,
    "right_palm": C.RIGHT_PALM,
}
