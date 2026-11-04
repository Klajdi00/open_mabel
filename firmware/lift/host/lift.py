"""YOR-compatible lift adapter.

Exposes the same method names and **metre** units as the YOR robot's lift API
(`robot/base_motor.py` PicoLift + Base, and `robot/lift.py`), wrapping MABEL's
richer `LiftController`. Drop-in for YOR-style code:

    from lift import Lift
    lift = Lift("/dev/ttyACM0")
    lift.lift_home()                  # stall-seek the floor
    lift.lift_to_height(0.30)         # metres, blocking
    print(lift.get_lift_height())     # metres
    lift.set_velocity_control(0.02)   # +20 mm/s (up)
    lift.set_neutral()                # free-wheel (torque off)

Difference from YOR's serial lift: YOR's Pico only does up/down/stop and fakes
position control by polling height from the host. MABEL runs closed-loop
trapezoidal position + PI velocity control **on the Pico**, so `lift_to_height`
and `set_velocity_control` are real on-device closed loops, not host polling.

Sign convention: positive = up (MABEL firmware already uses up-positive, so —
unlike YOR's TalonFX lift — no negation is applied here).
"""

from __future__ import annotations

import warnings
from typing import Optional

from lift_controller import LiftController

MM_PER_M = 1000.0


class Lift:
    def __init__(self, device_path: str = "/dev/ttyACM0", baud: int = 115200,
                 max_pos: float = 0.5):
        self._c = LiftController(device_path, baud)
        # Fallbacks used ONLY until telemetry arrives (or on an uncalibrated board that
        # reports 0). The firmware is the source of truth for the datum: after calibration
        # it pins the floor to LIFT_MIN_HEIGHT_MM and the ceiling to +LIFT_TRAVEL_MM, and
        # streams both back as min_height/max_height — so min_pos/max_pos below READ them.
        self._min_pos_default = 0.0    # metres
        self._max_pos_default = float(max_pos)
        self._homed = False

    # ------------------------------------------------------------- state
    @property
    def homed(self) -> bool:
        if self._homed:
            return True
        t = self._c.telemetry()
        return bool(t and t.calibrated)

    @property
    def min_pos(self) -> float:
        """Bottom-of-travel datum in metres, from the firmware's live telemetry
        (`min_height`). Hardcoding 0.0 here was a bug: the real floor datum is
        LIFT_MIN_HEIGHT_MM (0.3175 m), so a 0.0 min made every consumer's travel window
        start ~0.32 m BELOW the physical floor — the studio slider's lower third mapped
        under the floor and the column grounded long before the slider reached its min."""
        t = self._c.telemetry()
        if t is not None and t.min_height > 0:
            return t.min_height / MM_PER_M
        return self._min_pos_default

    @property
    def max_pos(self) -> float:
        """Top-of-travel datum in metres, from the firmware's live telemetry
        (`max_height`) — reflects the calibrated ceiling without waiting for an in-process
        calibrate() (a board calibrated in a previous session reports it on connect)."""
        t = self._c.telemetry()
        if t is not None and t.max_height > 0:
            return t.max_height / MM_PER_M
        return self._max_pos_default

    def get_lift_height(self) -> Optional[float]:
        """Current height in metres, or None if telemetry is unavailable."""
        t = self._c.telemetry()
        return None if t is None else t.height / MM_PER_M

    get_position = get_lift_height        # Phoenix6-style alias

    def get_velocity(self) -> float:
        """Current velocity in m/s."""
        t = self._c.telemetry()
        return 0.0 if t is None else t.vel / MM_PER_M

    def update_state(self) -> None:
        """No-op: telemetry refreshes itself in a background thread.

        Present for source-compatibility with YOR's TalonFX Lift.update_state().
        """

    # ------------------------------------------------------------- jog
    def lift_up(self) -> None:
        self._c.drive(1.0)

    def lift_down(self) -> None:
        self._c.drive(-1.0)

    def lift_stop(self) -> None:
        self._c.stop()

    # ------------------------------------------------------- home / calib
    def lift_home(self, wait: bool = True, timeout: float = 60.0) -> None:
        self._c.home(wait=wait, timeout=timeout)
        self._homed = True

    home = lift_home

    def set_home(self) -> None:
        """Treat the current position as the home/bottom datum (no motion)."""
        self._c.set_home()
        self._homed = True

    def calibrate(self, wait: bool = True, timeout: float = 60.0) -> None:
        """Stall-seek floor + ceiling. The firmware measures + persists the true travel
        envelope; `min_pos`/`max_pos` read it back live from telemetry (no assignment here)."""
        self._c.calibrate(wait=wait, timeout=timeout)
        self._homed = True

    # ------------------------------------------------- position (metres)
    def lift_to_height(self, target_m: float, v_max_mps: Optional[float] = None,
                       wait: bool = True, timeout_s: float = 30.0) -> bool:
        """Move to an absolute height in metres (closed-loop on the Pico).

        Returns True if it settled within 3 mm of the target (or immediately
        when wait=False).
        """
        target_m = max(self.min_pos, min(self.max_pos, float(target_m)))
        v_max = None if v_max_mps is None else abs(v_max_mps) * MM_PER_M
        self._c.move_to(target_m * MM_PER_M, v_max, wait=wait, timeout=timeout_s)
        if not wait:
            return True
        h = self.get_lift_height()
        return h is not None and abs(h - target_m) <= 0.003

    def lift_delta_height(self, delta_m: float, **kw) -> bool:
        """Move up/down by `delta_m` metres relative to the current height."""
        h = self.get_lift_height()
        if h is None:
            print("[lift_delta_height] height unknown; cannot move by delta")
            return False
        return self.lift_to_height(h + float(delta_m), **kw)

    def set_lift_position(self, target) -> bool:
        """Accepts a scalar or array-like (metres), like YOR's set_lift_position."""
        val = float(target[0]) if hasattr(target, "__len__") else float(target)
        return self.lift_to_height(val, wait=False)

    # ------------------------------------------------- velocity (m/s)
    def set_velocity_control(self, velocity_mps: float) -> None:
        """Closed-loop velocity hold in m/s (positive = up)."""
        if not self.homed:
            warnings.warn("Lift is not homed/calibrated")
        self._c.set_velocity(float(velocity_mps) * MM_PER_M)

    # ------------------------------------------------- neutral / torque
    def set_neutral(self) -> None:
        """Free-wheel: disable the motor driver (YOR's NeutralOut equivalent)."""
        self._c.set_torque(False)

    def set_torque(self, on: bool = True) -> None:
        self._c.set_torque(on)

    # ------------------------------------------------------------- lifecycle
    def close(self) -> None:
        self._c.close()

    def __enter__(self) -> "Lift":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
