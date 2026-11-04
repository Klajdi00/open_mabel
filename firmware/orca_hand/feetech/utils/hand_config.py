"""Central configuration for the MABEL ORCA hand demos.

This is now a thin **compatibility shim** over the lean :mod:`orcahand` API:
the joint↔motor map, named poses, port detection and calibration all come from
``orcahand`` (the single source of truth shared with the GUI, the ROS node and
the controllers), but the legacy names the ``control/*`` demos import
(``JOINT_ID``, ``POSES``, ``load_calibration``, ``pct_to_raw``, the ANSI colour
constants, the ``DATA_DIR`` paths, ...) are preserved so nothing breaks.

New code should prefer the API directly::

    from utils.hand_config import connect_hand
    with connect_hand("left") as hand:          # an orcahand.OrcaHand
        hand.set_named_position("fist")

``ORCA_CORE_ROOT`` is still put on ``sys.path`` for the few demos that have not
yet been ported off the old ``orca_core.hardware.feetech`` transport.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
FEETECH_ROOT = Path(__file__).resolve().parent.parent
ORCA_CORE_ROOT = FEETECH_ROOT.parent / "orca_core"
DATA_DIR = FEETECH_ROOT / "data"
CALIB_JSON = DATA_DIR / "calibration" / "joint_limits.json"
CALIB_YAML = DATA_DIR / "calibration" / "calibration.yaml"
GESTURES_DIR = DATA_DIR / "gestures"
ANIMATIONS_DIR = DATA_DIR / "animations"
LOGS_DIR = DATA_DIR / "logs"

for _p in (str(FEETECH_ROOT), str(ORCA_CORE_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from orcahand import HandConfig, OrcaHand          # noqa: E402
from orcahand.poses import POSES as _POSES          # noqa: E402
from orcahand.ports import detect_port as _detect_port  # noqa: E402

# ─── Bus ──────────────────────────────────────────────────────────────────────
BAUDRATE = 1_000_000
TOTAL = 17
WRIST_ID = 17

# ─── Joint ↔ motor map (sourced from the packaged configs) ────────────────────
_LEFT = HandConfig.load("left")
_RIGHT = HandConfig.load("right")
JOINT_ID_BY_SIDE = {
    "left": {j.name: j.motor_id for j in _LEFT.joints},
    "right": {j.name: j.motor_id for j in _RIGHT.joints},
}
# Legacy default = LEFT hand (flat dict), matching the old module.
JOINT_ID: dict[str, int] = dict(JOINT_ID_BY_SIDE["left"])
ID_JOINT: dict[int, str] = {v: k for k, v in JOINT_ID.items()}

# ─── Finger groups — flexors listed first ────────────────────────────────────
FINGERS: dict[str, list[str]] = {
    "index": ["index_pip", "index_mcp", "index_abd"],
    "middle": ["middle_pip", "middle_mcp", "middle_abd"],
    "ring": ["ring_pip", "ring_mcp", "ring_abd"],
    "pinky": ["pinky_pip", "pinky_mcp", "pinky_abd"],
    "thumb": ["thumb_ip", "thumb_mp", "thumb_abd", "thumb_cmc"],
}
ALL_FINGERS = list(FINGERS.keys())

# ─── Named poses (from orcahand.poses) ────────────────────────────────────────
POSES: dict[str, dict[str, int]] = dict(_POSES)

# ─── ANSI colours (shared) ────────────────────────────────────────────────────
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
PURPLE = "\033[95m"
CYAN = "\033[96m"
ORANGE = "\033[38;5;208m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


# ─── Port detection (delegates to orcahand) ──────────────────────────────────
def detect_port() -> str:
    """Auto-detect the Feetech USB adapter (CH340 / CP210x)."""
    try:
        return _detect_port()
    except Exception as e:
        print(f"{RED}{e}{RESET}")
        raise SystemExit(1)


# ─── Calibration I/O (from the orcahand config) ──────────────────────────────
def load_calibration(side: str = "left") -> dict[int, dict]:
    """Return {motor_id: {min, max, center}} from the packaged calibration.

    Sourced from ``orcahand``'s config (the same ``<side>_calibration.json`` the
    driver uses), so demos and the driver can never disagree. Falls back to the
    full 0-4095 range for any motor without calibration.
    """
    cfg = _LEFT if side == "left" else _RIGHT
    return {
        j.motor_id: {"min": int(j.raw_min), "max": int(j.raw_max), "center": int(j.center)}
        for j in cfg.joints
    }


# ─── Position helpers (legacy signatures) ────────────────────────────────────
def pct_to_raw(motor_id: int, pct: float, calib: dict) -> int:
    """0-100% → raw position within the motor's calibrated range."""
    pct = max(0.0, min(100.0, pct))
    lo, hi = calib[motor_id]["min"], calib[motor_id]["max"]
    return int(lo + (pct / 100.0) * (hi - lo))


def raw_to_pct(motor_id: int, raw: int, calib: dict) -> float:
    """Raw position → 0-100% within the motor's calibrated range."""
    lo, hi = calib[motor_id]["min"], calib[motor_id]["max"]
    return 0.0 if hi == lo else max(0.0, min(100.0, (raw - lo) / (hi - lo) * 100.0))


# ─── Preferred entry point for new / ported demos ────────────────────────────
def connect_hand(side: str = "left", port: str | None = None,
                 mock: bool = False, enable_torque: bool = True) -> OrcaHand:
    """Open and return an :class:`orcahand.OrcaHand` (real or mock).

    The modern replacement for the per-demo ``Bus`` classes. Use it as a context
    manager (``with connect_hand('left') as hand:``) so torque is released on
    exit.
    """
    hand = OrcaHand.mock(side) if mock else OrcaHand(side, port=port)
    hand.connect(enable_torque=enable_torque)
    return hand
