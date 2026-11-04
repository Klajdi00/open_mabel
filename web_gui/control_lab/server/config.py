"""
Control Lab configuration — ports, paths, env overrides.

Follows the studio conventions (simulation_control_center/server/config.py):
walk-up repo-root resolution with the folder-by-role guard, env-overridable
port in the studio block (8092/8093 curation, 8094 control center, 8095
trainer, 8096 train queue, 8097 sim control center, 8110 genesis,
**8115 controller studio**), and website-asset paths for the shared GLB rig.

Unlike the other studios, PHYSICS RUNS IN-PROCESS here: this is the test
bench for the `mabel` control library (controller/), so the backend owns
mj_step and runs the controllers directly with per-stage instrumentation.
The Vision Pro node (when enabled) binds the standard teleop port 9090 so the
unmodified MABEL VP app can connect straight to this Mac — stop the main sim
bridge first (the two cannot share the port).
"""
import os
from pathlib import Path

_HERE = Path(__file__).resolve().parent          # server/
_PKG = _HERE.parent                              # control_lab/
_REPO = _PKG.parent                              # web_control_gui/
if not (_REPO / "simulation" / "mabel_mujoco").exists():
    _REPO = _REPO.parent                         # → MABEL/ (folder-by-role layout)

HOST = os.environ.get("MABEL_CTS_HOST", "0.0.0.0")
PORT = int(os.environ.get("MABEL_CTS_PORT", "8115"))

# Vision Pro teleop node: the apps hard-code :9090, so we serve it there.
VP_PORT = int(os.environ.get("MABEL_CTS_VP_PORT", "9090"))

# State stream + plot sampling rates.
STATE_HZ = float(os.environ.get("MABEL_CTS_STATE_HZ", "30"))
PLOT_HZ = float(os.environ.get("MABEL_CTS_PLOT_HZ", "20"))
PLOT_WINDOW_S = float(os.environ.get("MABEL_CTS_PLOT_WINDOW", "12"))

# Empty-field scene: the bare robot + floor (no props) — exactly what the
# controller test bench wants. Env-overridable for other scenes.
SCENE = os.environ.get(
    "MABEL_CTS_SCENE",
    str(_REPO / "simulation" / "mabel_mujoco" / "scenes" / "mabel_bare.xml"))

WEB_DIR = _PKG / "web"
WEBSITE_ASSETS = _REPO / "web_gui" / "shared_assets"    # mabel_rig.glb + mabel_joints.json
CONTROLLER_DIR = _REPO / "controller"

# Per-operator calibration candidates (first hit wins): the full
# OperatorProfile JSON cache dirs, then the AVP reach profile.
PROFILE_DIRS = [
    _REPO / "server" / "profiles",
    _REPO / "controller" / "profiles",
    _HERE.parent / "profiles",
]
AVP_REACH_PROFILE = _REPO / "server" / "profiles" / "operator.json"

DEFAULT_USER = os.environ.get("MABEL_CTS_USER", "default")

# qpos memmap for the external mjpython viewer mirror (empty string disables).
STATE_FILE = os.environ.get("MABEL_CTS_STATE_FILE",
                            "/tmp/mabel_control_lab_state.f64")


def lan_ip():
    """Best-effort LAN address for the 'connect your headset to …' hint."""
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "this-mac"

# Physics timestep override [s]: the scene ships dt=2 ms but raw mj_step costs
# ~3.2 ms wall on the dev Mac; 4 ms is the value the wbc_stability studies
# validated on this scene and paces 1x realtime with headroom.
TIMESTEP = float(os.environ.get("MABEL_CTS_TIMESTEP", "0.004"))
