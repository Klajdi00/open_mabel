"""orcahand -- clean hardware-abstraction layer for the MABEL ORCA hand.

The single entry point is :class:`OrcaHand`. Controllers, learned policies, and
the ROS node all import it; nothing else should talk to the Feetech bus
directly.

    from orcahand import OrcaHand

    with OrcaHand("right") as hand:
        hand.set_joint_positions({"index_mcp": 0.5})   # radians
        state = hand.get_state()

Layering (each layer depends only on the one below):

    OrcaHand            joint-space API, radians, names, safety   (hand.py)
      HandConfig        joint<->motor map, ROM, calibration       (config.py)
        Joint           angle<->tick affine map                   (joints.py)
        Calibration     measured raw tick end-stops               (calibration.py)
      FeetechBus        raw STS/SCS serial transport              (bus.py)
"""

from .bus import FeetechBus
from .calibration import Calibration, MotorCalibration
from .config import HandConfig
from .hand import OrcaHand
from .joints import Joint
from .mock import MockFeetechBus
from .poses import POSES
from .ports import detect_port
from .state import HandState

__version__ = "0.1.0"

__all__ = [
    "OrcaHand",
    "HandConfig",
    "HandState",
    "FeetechBus",
    "MockFeetechBus",
    "Joint",
    "Calibration",
    "MotorCalibration",
    "POSES",
    "detect_port",
    "__version__",
]
