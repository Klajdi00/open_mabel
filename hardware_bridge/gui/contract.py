"""contract.py - the bridge's DDS contract + per-subsystem joint metadata.

ONE place describing what the control center talks to. Every topic here is exactly the
generic, mabel_interfaces-free contract the hardware bridge exposes (see config.py /
README); the GUI is a thin client of it, never of the hardware. Joint limits for the
arms / body / lift come from the canonical URDF (mabel_description/urdf/mabel_full.urdf,
itself derived from the MJCF); the hands load their joint<->ID map + ranges live from the
VENDORED config (hands/orcahand/configs) so the GUI can never drift from the driver.
"""

from __future__ import annotations

import os
import re
import urllib.request

# ── Cameras (MJPEG over HTTP — the web-teleop streams, served on :8080) ────────
# Host override with MABEL_MJPEG_URL (e.g. http://192.168.1.166:8080). The sensors
# stack (sensors/ros_camera_node.py --serve) publishes a stream PER physical camera
# at /camera/<name>/stream.mjpg and an index listing them at "/". We DISCOVER that
# index at startup so the tiles always match what is actually being served (the
# RealSense at the base front is /camera/front; the ZED head is /camera/main +
# /camera/head_left). If the server is unreachable we fall back to the static list
# below; absent streams simply show "offline".
_MJPEG = os.getenv("MABEL_MJPEG_URL", "http://127.0.0.1:8080").rstrip("/")
# The bridge is the SINGLE camera server and MIRRORS THE SIM: ALL cams — base/head AND
# the wrist cams it owns — are served on ONE port (MABEL_HTTP_PORT, default :8080), same
# /camera/<name>/stream.mjpg contract as the sim. So the wrist tiles use the SAME base
# URL as everything else; _WRIST_MJPEG defaults to _MJPEG and only differs if you point
# MABEL_WRIST_MJPEG_URL at a separate server (legacy split).
_WRIST_MJPEG = os.getenv("MABEL_WRIST_MJPEG_URL", _MJPEG).rstrip("/")

# The fixed tile layout the Control Center shows, in display order:
#   • Base Front (D435) — the RealSense bolted to the base, facing forward
#   • Head (ZED)        — ONE ZED-M head view (the wrapper serves it under two
#                         stream names, main/head_left; we show a single tile and
#                         pick whichever name is actually live)
#   • Left / Right Wrist — always shown; a wrist cam that isn't plugged in just
#                         renders "not connected" (the CameraTile's offline state)
def _stream_url(name: str) -> str:
    base = _WRIST_MJPEG if name.startswith("wrist_") else _MJPEG
    return f"{base}/camera/{name}/stream.mjpg"


# ZED head is served as both /camera/main and /camera/head_left — same physical
# camera. We only ever show one tile; prefer 'main', else 'head_left'.
_ZED_HEAD_NAMES = ("main", "head_left")

# Static fallback (server index unreachable): single ZED tile via 'main'.
CAMERAS = [
    ("Base Front (D435)", _stream_url("front")),
    ("Head (ZED)",        _stream_url("main")),
    ("Left Wrist",        _stream_url("wrist_left")),
    ("Right Wrist",       _stream_url("wrist_right")),
]


def discover_cameras(timeout: float = 1.5):
    """Return the fixed [(label, mjpg_url)] tile layout: base-front RealSense, a
    SINGLE ZED head, and both wrist cams. The wrist tiles are always present so an
    unplugged wrist cam shows "not connected" rather than vanishing. We query the
    MJPEG server index only to pick which ZED stream name is live (main vs
    head_left); everything else is fixed. Falls back to CAMERAS on any error."""
    served = set()
    try:
        with urllib.request.urlopen(_MJPEG + "/", timeout=timeout) as r:
            html = r.read().decode("utf-8", "replace")
        served = set(re.findall(r"/camera/([A-Za-z0-9_]+)/stream\.mjpg", html))
    except Exception:
        return list(CAMERAS)
    zed = next((n for n in _ZED_HEAD_NAMES if n in served), _ZED_HEAD_NAMES[0])
    return [
        ("Base Front (D435)", _stream_url("front")),
        ("Head (ZED)",        _stream_url(zed)),
        ("Left Wrist",        _stream_url("wrist_left")),
        ("Right Wrist",       _stream_url("wrist_right")),
    ]

# ── LiDAR (RPLIDAR A3) ────────────────────────────────────────────────────────
LIDAR = dict(scan="/scan")

# ── Base (MAXSwerve) ─────────────────────────────────────────────────────────
BASE = dict(
    cmd_vel="/mabel/hw/base/cmd_vel",          # geometry_msgs/Twist  (+x fwd, +y left)
    estop="/mabel/hw/base/estop",              # std_msgs/Bool
    joint_states="/mabel/hw/base/joint_states",  # sensor_msgs/JointState (6 swerve)
    vmax=1.0,    # m/s at full slider
    wmax=2.0,    # rad/s at full slider
)

# ── Arms (DaMiao MIT, 14 DoF) ────────────────────────────────────────────────
# (name, lower, upper) in radians, from the URDF. The wrist is driven by the HAND
# subsystem (Feetech id 17), so it is NOT here.
ARMS = dict(
    command="/arms/command",
    joint_states="/arms/joint_states",
    torque_enable="/arms/torque_enable",
    torque_state="/arms/torque_state",
    joints=[
        ("left_arm_1", -3.49066, 1.39626), ("left_arm_2", -0.174533, 3.31612),
        ("left_arm_3", -1.5708, 1.5708),   ("left_arm_4", -0.773461, 1.67),
        ("left_arm_5", -1.5708, 1.5708),   ("left_arm_6", -0.785398, 0.785398),
        ("left_arm_7", -1.5708, 1.5708),
        ("right_arm_1", -1.39626, 3.49066), ("right_arm_2", -3.31612, 0.174533),
        ("right_arm_3", -1.5708, 1.5708),   ("right_arm_4", -1.67, 0.773461),
        ("right_arm_5", -1.5708, 1.5708),   ("right_arm_6", -0.785398, 0.785398),
        ("right_arm_7", -1.5708, 1.5708),
    ],
)

# ── Body (DaMiao torso + 3-DoF neck) ─────────────────────────────────────────
BODY = dict(
    command="/body/command",
    joint_states="/body/joint_states",
    torque_enable="/body/torque_enable",
    torque_state="/body/torque_state",
    joints=[
        ("torso", -1.0472, 0.2618),
        ("neck_yaw", -1.5708, 1.5708),
        ("neck_pitch", -0.785398, 0.785398),
        ("neck_roll", -0.785398, 0.785398),
    ],
)

# ── Lift (Pico DC column, metres) ────────────────────────────────────────────
LIFT = dict(
    command="/lift/command",                   # std_msgs/Float64 target height (m)
    joint_states="/lift/joint_states",
    calibrate="/lift/calibrate",               # std_msgs/Bool  (true = stall-seek)
    limits="/lift/limits",                     # std_msgs/String {min,max,calibrated} latched
    joint="lift",
    min_m=0.0,
    max_m=0.5,                                  # matches config.LIFT max_pos default
)

# ── Hands (ORCA / Feetech, per side) ─────────────────────────────────────────
HANDS = dict(
    sides=["left", "right"],
    command="/{side}_hand/command",            # sensor_msgs/JointState (rad)
    torque_enable="/{side}_hand/torque_enable",  # std_msgs/Bool
    joint_states="/{side}_hand/joint_states",
    torque_state="/{side}_hand/torque_state",  # std_msgs/Bool (latched)
    temperatures="/{side}_hand/temperatures",  # std_msgs/Float64MultiArray
    voltages="/{side}_hand/voltages",          # std_msgs/Float64MultiArray
)
# NOTE: end-stop calibration is NOT a ROS topic - it is a ROS-free bench step run with
# the bridge stopped (`python3 -m mabel_hw.drivers.orcahand.calibrate --model left`).


def _vendored_config(side: str):
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from mabel_hw.drivers.orcahand import HandConfig
    return HandConfig.load(model=side)


def pose_names() -> list:
    """The predefined hand poses (names only); [] if the vendored package is absent."""
    try:
        import os
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from mabel_hw.drivers.orcahand.poses import POSES
        return list(POSES)
    except Exception:
        return []


def pose_targets(side: str, name: str) -> dict:
    """{joint_name: radians} for a named pose.

    Poses are stored as 0..100 % of each joint's calibrated range (0 = open/extended =
    angle_min, 100 = closed/flexed = angle_max). Converted here to the radian targets the
    bridge's /<side>_hand/command expects, using that side's vendored joint ranges so the
    pose lands the same as the firmware's ``set_named_position``."""
    out = {}
    try:
        cfg = _vendored_config(side)          # also puts hardware_bridge/ on sys.path
        from mabel_hw.drivers.orcahand.poses import POSES
    except Exception:
        return out
    pose = POSES.get(name, {})
    for j in cfg.joints:
        if j.name in pose:
            frac = max(0.0, min(1.0, pose[j.name] / 100.0))
            out[j.name] = j.angle_min + frac * (j.angle_max - j.angle_min)
    return out


def hand_joints(side: str):
    """[(name, lower, upper)] for one hand, from the VENDORED orcahand config.

    Falls back to the URDF-derived ranges baked below if the vendored package can't
    be imported (so the GUI still draws sliders without the HAL on the path)."""
    try:
        import os
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from mabel_hw.drivers.orcahand import HandConfig
        cfg = HandConfig.load(model=side)
        return [(j.name, min(j.angle_min, j.angle_max), max(j.angle_min, j.angle_max))
                for j in cfg.joints]
    except Exception:
        return list(_HAND_FALLBACK)


# Fallback hand joint ranges (bare names; from the URDF). Used only if the vendored
# orcahand config can't be loaded. Order roughly thumb->pinky->wrist.
_HAND_FALLBACK = [
    ("thumb_cmc", -0.87266, 0.87266), ("thumb_abd", -0.34907, 0.73304),
    ("thumb_mp", -0.20944, 1.81496), ("thumb_ip", -0.34907, 1.95477),
    ("index_abd", -0.64577, 0.64577), ("index_mcp", -0.34907, 1.65806),
    ("index_pip", -0.34907, 1.88496),
    ("middle_abd", -0.64577, 0.64577), ("middle_mcp", -0.34907, 1.58825),
    ("middle_pip", -0.34907, 1.8675),
    ("ring_abd", -0.64577, 0.63577), ("ring_mcp", -0.34907, 1.58825),
    ("ring_pip", -0.34907, 1.8675),
    ("pinky_abd", -0.64577, 0.64577), ("pinky_mcp", -0.34907, 1.71042),
    ("pinky_pip", -0.34907, 1.88496),
    ("wrist", -1.0472, 0.5236),
]
