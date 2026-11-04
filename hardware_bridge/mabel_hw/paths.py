"""paths.py — resolve the in-repo firmware/driver packages the bridge owns.

The subsystems reuse the SAME proven host libraries the rest of MABEL uses
(``firmware/lift/host``, ``firmware/orca_hand/orcahand``), rather than re-deriving
device protocols. They are not pip-installed, so we add their checkouts to sys.path
(installed-or-in-repo, the same pattern as the old ROS hardware nodes). MABEL_ROOT
overrides autodetection.
"""

import os
import sys

# This file lives at <MABEL>/hardware_bridge/mabel_hw/paths.py: the bridge dir is
# one up, the MABEL root two up. MABEL_ROOT overrides autodetection.
_HERE = os.path.dirname(os.path.abspath(__file__))
_BRIDGE_DIR = os.path.dirname(_HERE)
_DEFAULT_ROOT = os.path.dirname(_BRIDGE_DIR)


def bridge_dir() -> str:
    """<MABEL>/hardware_bridge — where calibration_data/, tools/, gui/ live."""
    return _BRIDGE_DIR


def mabel_root() -> str:
    return os.environ.get("MABEL_ROOT", _DEFAULT_ROOT)


def add_to_path(*rel_dirs) -> bool:
    """Prepend <MABEL_ROOT>/<rel> to sys.path for each existing dir. True if all found."""
    ok = True
    root = mabel_root()
    for rel in rel_dirs:
        p = os.path.join(root, rel)
        if os.path.isdir(p):
            if p not in sys.path:
                sys.path.insert(0, p)
        else:
            ok = False
    return ok


# Convenience helpers per subsystem -----------------------------------------
def add_lift_host():
    """firmware/lift/host -> import lift / lift_controller."""
    return add_to_path(os.path.join("firmware", "lift", "host"))


def add_orcahand():
    """DEPRECATED — the ORCA hand HAL is now VENDORED into the bridge at
    ``hands/orcahand`` (ported from firmware/orca_hand/feetech), so the hands
    subsystem imports ``from hands import HandsController`` directly and does NOT
    reach out to the firmware checkout. Kept only for any external caller."""
    return add_to_path(os.path.join("firmware", "orca_hand", "orcahand"))


def hand_config_dir(override: str = "") -> str:
    """Directory holding <side>_hand.yaml + <side>_calibration.json, used ONLY when
    ``HANDS_CONFIG_DIR`` is set to override the vendored packaged configs.

    Prefer an explicit override, then the ROS package share (if the workspace is
    sourced), then the in-repo ROS source config.
    """
    if override:
        return override
    try:
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(get_package_share_directory("mabel_hand_hardware"), "config")
    except Exception:
        return os.path.join(mabel_root(), "ros", "src",
                            "mabel_hand_hardware", "config")
