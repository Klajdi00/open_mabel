"""
Simulation Control Center server configuration.

The SIM twin of control_center/: the same studio (client + server in one page)
but pointed at the MuJoCo teleop stack from MABEL/server (sim_teleop_bridge on
ws :9090, sim camera server on :8080) instead of the real robot. Everything the
real Control Center taps from ROS is either read from the simulation itself
(robot_state mirror + a MuJoCo qpos-memmap mirror for pointcloud/LiDAR/odom) or
degrades explicitly (no Nav2/SLAM on a Mac).

On Ubuntu with the ROS sim stack (server/sim_ros_server.sh) the RosSource and
the bridge's own :9092 map plane light up instead — set MABEL_MAP_PORT=9092.
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)                    # simulation_control_center/
_REPO = os.path.dirname(_PKG)                    # web_control_gui/ (or MABEL/ pre-reorg)
# folder-by-role reorg: the package lives under web_control_gui/, so the repo root
# is one level further up. Walk up until the sim tree is in sight.
if not os.path.isdir(os.path.join(_REPO, "simulation", "mabel_mujoco")):
    _REPO = os.path.dirname(_REPO)               # MABEL/

WEB_DIR = os.path.join(_PKG, "web")
WEBSITE_ASSETS = os.path.join(_REPO, "web_gui", "shared_assets")
CAMERA_RIG_JSON = os.path.join(
    _REPO, "robot_learning", "data_curation", "server", "mabel_camera_rig.json")

HOST = os.environ.get("MABEL_SCC_HOST", "0.0.0.0")
# 8092/8093 curation, 8094 control center, 8095 trainer, 8096 train queue.
PORT = int(os.environ.get("MABEL_SCC_PORT", "8097"))

# The sim-server planes the web client talks to directly. Defaults assume the
# sim stack runs on the SAME host as this backend (the normal Mac setup), so
# the page's own hostname is reused; override for a split deployment.
ROBOT_HOST = os.environ.get("MABEL_SCC_ROBOT_HOST", "")   # "" -> page hostname
TELEOP_PORT = int(os.environ.get("MABEL_WS_PORT", "9090"))
# 0 = this backend serves the map plane itself at /ws/map (same-origin), fed by
# the MuJoCo mirror — the pure-sim Mac path. Set 9092 on Ubuntu to use the
# bridge's ROS-fed map plane instead.
MAP_PORT = int(os.environ.get("MABEL_MAP_PORT", "0"))
VIDEO_PORT = int(os.environ.get("MABEL_HTTP_PORT", "8080"))
APP_TOKEN = os.environ.get("MABEL_TOKEN", "")

# Where this backend reaches the sim stack itself (camera proxy + the bridge
# websocket the telemetry SimSource subscribes to).
UPSTREAM_HOST = os.environ.get("MABEL_SCC_UPSTREAM", "") or "127.0.0.1"

# Telemetry source: "auto" tries ROS, then the sim bridge (robot_state ws),
# then falls back to mock; explicit "ros" / "sim" / "mock" force one.
TELEMETRY_SOURCE = os.environ.get("MABEL_SCC_SOURCE", "auto")
TELEMETRY_HZ = float(os.environ.get("MABEL_SCC_TELEMETRY_HZ", "15"))

# ── the simulation itself ────────────────────────────────────────────────────
SERVER_DIR = os.path.join(_REPO, "server")               # the sim teleop stack
SIM_DIR = os.path.join(_REPO, "simulation", "mabel_mujoco")
SCENES_DIR = os.path.join(SIM_DIR, "scenes")
SCENES_JSON = os.path.join(SCENES_DIR, "scenes.json")
DEFAULT_SCENE = os.environ.get(
    "MABEL_SCENE", os.path.join(SCENES_DIR, "mabel_home.xml"))
# qpos memmap the bridge publishes (sim_teleop_bridge --state-file); the
# MuJoCo mirror reads it for pointcloud / LiDAR / odom.
STATE_FILE = os.environ.get("MABEL_STATE_FILE", "/tmp/mabel_teleop_state.f64")

# Whether THIS backend owns the sim stack lifecycle (spawn/restart on scene
# change). "auto" = spawn unless a bridge already answers on :9090.
MANAGE_SIM = os.environ.get("MABEL_SCC_MANAGE_SIM", "auto")   # auto|1|0
SIM_LAUNCHER = os.environ.get(
    "MABEL_SCC_SIM_LAUNCHER", os.path.join(SERVER_DIR, "start_teleop.sh"))

# MuJoCo mirror cadences (the map plane the 3D viewer consumes).
MIRROR_HZ = float(os.environ.get("MABEL_SCC_MIRROR_HZ", "10"))      # odom+scan
CLOUD_HZ = float(os.environ.get("MABEL_SCC_CLOUD_HZ", "2"))         # depth cloud
CLOUD_RES = tuple(int(v) for v in os.environ.get(
    "MABEL_SCC_CLOUD_RES", "212x120").split("x"))                   # w x h
# head + front base depth clouds, merged into the live 'cloud' layer
CLOUD_CAMERAS = [c.strip() for c in os.environ.get(
    "MABEL_SCC_CLOUD_CAM", "zed_left,body_front_cam").split(",") if c.strip()]
CLOUD_MAX_RANGE = float(os.environ.get("MABEL_SCC_CLOUD_MAX_M", "8.0"))

# Episodes: MUST match the bridge's --episodes-dir (the launcher exports
# MABEL_EPISODES_DIR to both processes; the bridge's own default is the same).
EPISODES_DIR = os.path.expanduser(
    os.environ.get("MABEL_EPISODES_DIR", "~/Desktop/mabel_episodes"))
RECORDINGS_DIR = os.environ.get(
    "MABEL_RECORDINGS_DIR", os.path.join(_REPO, "recordings"))

# Actuator temperature grading (deg C) — mirrored in the web UI. (Sim has no
# temperatures; the strip stays empty unless a ROS source provides them.)
TEMP_WARN = float(os.environ.get("MABEL_CC_TEMP_WARN", "55"))
TEMP_HOT = float(os.environ.get("MABEL_CC_TEMP_HOT", "70"))
