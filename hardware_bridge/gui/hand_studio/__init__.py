"""hand_studio — the firmware ORCA Hand Studio, ported into the Control Center.

A near-verbatim copy of ``firmware/orca_hand/feetech/gui`` (Control / Setup / Calibration
/ Poses / Teleop), with ONE change: instead of opening the Feetech serial port directly,
every hardware action goes through the ROS-free **hand bridge** (``mabel_hw.rpc.hand_server``) —
so the GUI is a pure client of the one hardware owner, and it works on macOS where ROS
(rclpy) is unavailable. The look and the panel behaviour are identical to the firmware GUI.

This package sets up ``sys.path`` so the ported panels can keep their original imports
(``from style import *``, ``from orcahand.constants import …``) unchanged, and so the
client ``HandBus`` can reach ``mabel_hw.rpc.hand_server``.
"""

import os
import sys

_HAND_STUDIO = os.path.dirname(os.path.abspath(__file__))
_GUI = os.path.dirname(_HAND_STUDIO)                       # hardware_bridge/gui
_BRIDGE = os.path.dirname(_GUI)                            # hardware_bridge
_HANDS = os.path.join(_BRIDGE, "hands")                    # exposes top-level `orcahand`

for _p in (_GUI, _BRIDGE, _HANDS, _HAND_STUDIO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from .hand_bus import HandBus  # noqa: E402,F401
