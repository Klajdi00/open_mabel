"""
Control Center server configuration.

The Control Center is BOTH a client and a server: it runs on the robot host (mabel-jetson)
next to the real_run.sh stack and serves the studio web app to any browser;
each browser is then a live teleop client of the existing bridge (ws :9090)
while this backend adds the telemetry/monitoring planes the bridge does not
expose (joint vel/torque/temps from ROS, host stats, recordings-on-disk).
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)                    # control_center/
_REPO = os.path.dirname(_PKG)                    # web_control_gui/ (or MABEL/ pre-reorg)
# folder-by-role reorg: the package lives under web_control_gui/, so the repo root
# is one level further up. Walk up until the sim tree is in sight.
if not os.path.isdir(os.path.join(_REPO, "simulation", "mabel_mujoco")):
    _REPO = os.path.dirname(_REPO)               # MABEL/

WEB_DIR = os.path.join(_PKG, "web")
WEBSITE_ASSETS = os.path.join(_REPO, "web_gui", "shared_assets")
CAMERA_RIG_JSON = os.path.join(
    _REPO, "robot_learning", "data_curation", "server", "mabel_camera_rig.json")

HOST = os.environ.get("MABEL_CC_HOST", "0.0.0.0")
PORT = int(os.environ.get("MABEL_CC_PORT", "8094"))   # curation studio uses 8092/8093

# The robot-server planes the web client talks to directly. Defaults assume the
# Control Center runs ON the robot host, so the page's own hostname is reused;
# override for a split deployment.
ROBOT_HOST = os.environ.get("MABEL_CC_ROBOT_HOST", "")   # "" -> page hostname
TELEOP_PORT = int(os.environ.get("MABEL_WS_PORT", "9090"))
MAP_PORT = int(os.environ.get("MABEL_MAP_PORT", "9092"))
VIDEO_PORT = int(os.environ.get("MABEL_HTTP_PORT", "8080"))
APP_TOKEN = os.environ.get("MABEL_TOKEN", "")

# Depth of the /ws/cam H.264 proxy queue, in access units (~frames). This is a
# LATENCY budget, not a buffer: anything sitting here is a frame the viewer has
# not seen yet. A few frames absorb network jitter; beyond that we drop the stale
# GOP and resync at the next IDR rather than let the picture fall behind. See
# app.py:ws_cam_h264 and tools/exp_video_latency.py.
CAM_WS_QUEUE = int(os.environ.get("MABEL_CC_CAM_QUEUE", "4"))

# ── App-level login (styled popup) ───────────────────────────────────────────
# The Real-Robot Control Center is gated: a browser must log in with this
# username + passkey before ANY /api or /ws plane is served. See auth.py.
# Credentials come ONLY from the environment — NEVER hard-code them here.
# Provide them out-of-band, e.g. control_center/run.sh sources a machine-local,
# untracked ~/.mabel_cc_env.
# When unset, auth.check_credentials denies every login (fail-closed), so an
# unconfigured public checkout is locked rather than wide open.
CC_USER = os.environ.get("MABEL_CC_USER", "")
CC_PASSKEY = os.environ.get("MABEL_CC_PASSKEY", "")
CC_REQUIRE_LOGIN = os.environ.get("MABEL_CC_REQUIRE_LOGIN", "1") not in ("0", "false", "")

# Telemetry source: "auto" tries ROS (rclpy) and falls back to mock; "mock"
# forces the synthetic plant (dev / CI); "ros" fails hard without rclpy.
TELEMETRY_SOURCE = os.environ.get("MABEL_CC_SOURCE", "auto")
TELEMETRY_HZ = float(os.environ.get("MABEL_CC_TELEMETRY_HZ", "15"))

# Episode demos (task folders) + rosbag MCAP sessions, for the Data pane.
EPISODES_DIR = os.environ.get(
    "MABEL_EPISODES_DIR", os.path.join(_REPO, "robot_learning", "imitation_learning",
                                       "data_collection", "datasets"))
RECORDINGS_DIR = os.environ.get(
    "MABEL_RECORDINGS_DIR", os.path.join(_REPO, "recordings"))

# Actuator temperature grading (deg C) — mirrored in the web UI.
TEMP_WARN = float(os.environ.get("MABEL_CC_TEMP_WARN", "55"))
TEMP_HOT = float(os.environ.get("MABEL_CC_TEMP_HOT", "70"))

# ── MuJoCo mirror + physics preview (browser viewer, timeline ▶ Preview) ─────
# The gateway (ROSPlant) mirrors the encoders into the SAME qpos memmap the sim
# bridge writes, so the browser MuJoCo viewer (/ws/simview) shows the mirrored
# REAL robot, and the physics preview (/api/preview/*) can seed a long-horizon
# plan from the robot's true pose. Both degrade gracefully (log + error JSON)
# when mujoco / the memmap / simulation_bridge are absent on the host.
SIM_DIR = os.path.join(_REPO, "simulation", "mabel_mujoco")
SCENES_DIR = os.path.join(SIM_DIR, "scenes")
DEFAULT_SCENE = os.environ.get(
    "MABEL_SCENE", os.path.join(SCENES_DIR, "mabel_home.xml"))
STATE_FILE = os.environ.get("MABEL_STATE_FILE", "/tmp/mabel_teleop_state.f64")


# ---- which ORCA hands are mounted: MABEL's own v1, or NOVA's v2 swapped in -------------------
# The HAL's own rule (mabel_hw/drivers/hands.py Hand._resolve_port): per side, MABEL's v1 adapter
# wins; else NOVA's v2 adapter (HANDS.adapter_serials "<side>_v2") makes it a v2 hand. Evaluated
# LIVE so the 3D view follows a swap; a side with neither adapter keeps the last generation seen
# (default v1). The v2 rigs are baked by ros/src/mabel_description/build_hand_variants.py.
RIG_VARIANTS = os.path.join(_PKG, "rig_variants")
_HAND_LAST = {"left": "v1", "right": "v1"}
_HAND_CACHE = {"t": 0.0, "v": None}


def _adapter_serials():
    import sys
    hb = os.path.join(_REPO, "hardware_bridge")
    if os.path.isdir(hb) and hb not in sys.path:
        sys.path.insert(0, hb)
    try:
        from mabel_hw.config import HANDS
        return dict(HANDS.get("adapter_serials") or {})
    except Exception:                                    # noqa: BLE001 — no HAL tree here
        return {"left": "0000000001", "right": "0000000002", "left_v2": "0000000003", "right_v2": "0000000004"}


def hand_generations():
    """{'left': {'generation': 'v1'|'v2', 'present': bool}, 'right': {...}}, cached 1 s."""
    import time
    now = time.monotonic()
    if _HAND_CACHE["v"] is not None and now - _HAND_CACHE["t"] < 1.0:
        return _HAND_CACHE["v"]
    try:
        from serial.tools import list_ports
        present = {p.serial_number for p in list_ports.comports() if p.serial_number}
    except Exception:                                    # noqa: BLE001
        present = set()
    reg = _adapter_serials()
    out = {}
    for side in ("left", "right"):
        v1, v2 = reg.get(side), reg.get(f"{side}_v2")
        if v1 and v1 in present:
            _HAND_LAST[side] = "v1"
        elif v2 and v2 in present:
            _HAND_LAST[side] = "v2"
        out[side] = {"generation": _HAND_LAST[side], "present": bool((v1 and v1 in present) or (v2 and v2 in present))}
    _HAND_CACHE.update(t=now, v=out)
    return out


def rig_files(hands=None):
    """(glb, joints.json) drawn for the mounted hands; v1-v1 is the website's own mabel_rig.glb."""
    hands = hands or hand_generations()
    tag = f"hands-{hands['left']['generation']}-{hands['right']['generation']}"
    base = (os.path.join(WEBSITE_ASSETS, "mabel_rig.glb"), os.path.join(WEBSITE_ASSETS, "mabel_joints.json"))
    if tag == "hands-v1-v1":
        return base
    g, j = os.path.join(RIG_VARIANTS, f"mabel_rig-{tag}.glb"), os.path.join(RIG_VARIANTS, f"mabel_joints-{tag}.json")
    return (g, j) if os.path.exists(g) and os.path.exists(j) else base
