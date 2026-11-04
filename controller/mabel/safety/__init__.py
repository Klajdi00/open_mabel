"""safety — the teleop-session safety layer (distinct from the tip-over MOTION model).

`motion_model/` bounds where the BASE may move (lift-aware ZMP tip-over envelope);
this package guards the OPERATOR LINK:

    EngagementSupervisor  tracking-loss fail-safe — eases the upper body to a safe
                          HOME pose when the headset/hand stream drops, and eases
                          back to the live pose (no snap) on re-acquire.
    ClutchController       single-hand double-pinch toggles a whole-body freeze /
                          re-indexing so the operator can recenter.
    position_actuators(model)  the position-servo actuator ids the supervisor drives.

Old flat path (`mabel.safety`) is unchanged — this package IS `mabel.safety`, so
`from mabel.safety import EngagementSupervisor` keeps working. Tunables live in
`config` (config.SAFE_*, config.CLUTCH_*).
"""
from .engagement import (                 # noqa: F401
    EngagementSupervisor,
    ClutchController,
    position_actuators,
)

__all__ = ["EngagementSupervisor", "ClutchController", "position_actuators"]
