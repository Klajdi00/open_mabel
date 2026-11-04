"""nav_axes.py — twist <-> navJoystick, the exact inverse of what MABEL's controller applies.

The apps send a normalised stick (`navJoystick` {lx, ly, rx, ry}); the gateway hands it to the
controller, which maps it to a body twist in ONE place:

    controller/mabel/sources/wire.py::app_joystick     (lx, ly, rx, ry) -> (-lx, -ly, rx, ry)
    controller/mabel/teleop_controller.py::_manipulate  vx = ly' * MAX_LIN_SPEED
                                                        vy = lx' * MAX_LIN_SPEED
                                                        wz = rx  * MAX_ANG_SPEED   (after NAV_DEADZONE)

i.e. vx = -ly * vmax, vy = -lx * vmax, wz = rx * wmax. A client that already holds a TWIST
(the Bluetooth link: ble_protocol's TELEOP_FRAME carries vx/vy/wz) must send the stick that
produces it — so `nav_from_twist` is that mapping inverted, with UNIT caps: the client sends
how much of full speed it wants and the SERVER's caps stay the single authority on how fast
the base may actually move. The controller's deadzone is a hard cut (|v| < 0.08 -> 0), so
the inverse needs no reshaping — only the clamp.

Stdlib only; shared by server/ble_bridge.py and anything else that speaks twists.
"""
from __future__ import annotations

from typing import Mapping, Tuple

__all__ = ["twist_from_nav", "nav_from_twist"]


def _clamp(v: float) -> float:
    return max(-1.0, min(1.0, float(v)))


def twist_from_nav(nav: Mapping[str, float], vmax: float = 1.0, wmax: float = 1.0) -> Tuple[float, float, float]:
    """Stick axes -> (vx, vy, wz), the controller's convention (reference for tests)."""
    return (-float(nav.get("ly", 0.0)) * vmax, -float(nav.get("lx", 0.0)) * vmax,
            float(nav.get("rx", 0.0)) * wmax)


def nav_from_twist(vx: float, vy: float, wz: float, vmax: float = 1.0, wmax: float = 1.0) -> dict:
    """(vx, vy, wz) -> {lx, ly, rx, ry}: the stick that makes the controller produce that twist
    (at unit caps, a fraction of full speed). `ry` (lift) is left at 0 for the caller."""
    u = lambda v, s: _clamp(v / s) if s else 0.0
    return {"ly": -u(vx, vmax), "lx": -u(vy, vmax), "rx": u(wz, wmax), "ry": 0.0}
