"""lift.py — the standing-desk column as a typed subsystem (metres, up-positive).
Thin lease-gating veneer over :class:`~mabel_hw.drivers.lift.LiftDriver` — the
Pico runs the real closed loop; stop() is lease-free."""
from __future__ import annotations

from ..drivers.lift import LiftDriver
from ..types import LiftState

__all__ = ["LiftSubsystem"]


class LiftSubsystem:
    def __init__(self, cfg: dict, *, driver: LiftDriver = None, logger=None):
        self._cfg = cfg
        self._driver = driver if driver is not None else LiftDriver(
            port=cfg["port"], baud=int(cfg["baud"]),
            max_pos=float(cfg["max_pos"]), logger=logger)

    # lifecycle
    def connect(self) -> None:
        self._driver.connect()

    def close(self) -> None:
        self._driver.close()

    def connected(self) -> bool:
        return self._driver.connected()

    # read
    def state(self) -> LiftState:
        return self._driver.read()

    # write (lease-gated at the facade)
    def set_height(self, meters: float) -> None:
        self._driver.set_height(meters)

    def set_velocity(self, mps: float) -> None:
        self._driver.set_velocity(mps)

    def home(self, wait: bool = False) -> None:
        self._driver.home(wait=wait)

    def calibrate(self, wait: bool = False) -> None:
        self._driver.calibrate(wait=wait)

    def set_torque(self, on: bool) -> None:
        self._driver.set_torque(on)

    # safety (lease-FREE)
    def stop(self) -> None:
        self._driver.stop()

    # pacing
    def service(self) -> None:
        self._driver.service()
