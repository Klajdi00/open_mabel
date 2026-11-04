"""In-memory Feetech bus for development and CI without hardware.

``MockFeetechBus`` is a drop-in replacement for :class:`~orcahand.bus.FeetechBus`
exposing the identical surface ``OrcaHand`` uses. It stores each motor's last
commanded position and reports it straight back as the present position, so
``set_joint_positions`` -> ``get_joint_positions`` round-trips exactly (minus
real-world dynamics). Use via :meth:`orcahand.hand.OrcaHand.mock`.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from . import constants as C


class MockFeetechBus:
    """Simulated bus: holds motor state in a dict, no serial I/O."""

    def __init__(self, motor_ids: Iterable[int]):
        self.port = "mock"
        self.baudrate = C.DEFAULT_BAUDRATE
        self._pos: Dict[int, int] = {mid: C.RAW_CENTER for mid in motor_ids}
        self._open = False
        self.torque_on = False

    # lifecycle ------------------------------------------------------------
    def open(self) -> "MockFeetechBus":
        self._open = True
        return self

    def close(self) -> None:
        self._open = False

    @property
    def is_open(self) -> bool:
        return self._open

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()

    # config ---------------------------------------------------------------
    def set_torque_enable(self, enable: bool, motor_id: int = C.BROADCAST_ID) -> None:
        self.torque_on = bool(enable)

    def set_torque_limit(self, value: int, motor_id: int = C.BROADCAST_ID) -> None:
        pass

    def set_goal_speed(self, value: int, motor_id: int = C.BROADCAST_ID) -> None:
        pass

    def set_goal_acc(self, value: int, motor_id: int = C.BROADCAST_ID) -> None:
        pass

    # discovery ------------------------------------------------------------
    def ping(self, motor_id: int, timeout: float = 0.02) -> bool:
        return motor_id in self._pos

    def scan(self, id_range: Iterable[int] = range(1, 21)) -> List[int]:
        return [i for i in id_range if i in self._pos]

    # writes ---------------------------------------------------------------
    def sync_write_positions(self, id_to_raw: Dict[int, int]) -> None:
        for mid, raw in id_to_raw.items():
            if mid in self._pos:
                self._pos[mid] = int(raw)

    def write_register(self, motor_id: int, address: int, data) -> None:
        pass

    def ofs_cal(self, motor_id: int, position: int = 2048) -> None:
        """Re-zero: pretend the present pose is now ``position`` (centre)."""
        if motor_id in self._pos:
            self._pos[motor_id] = int(position)

    def read_register(self, motor_id: int, address: int, length: int = 1,
                      timeout: float = 0.02) -> Optional[int]:
        if motor_id not in self._pos:
            return None
        if address == C.ADDR_PRESENT_POSITION:
            return self._pos[motor_id]
        if address == C.ADDR_TORQUE_ENABLE:
            # Truthful register read-back so disable_torque_verified() confirms
            # against the mock exactly like the real bus.
            return 1 if self.torque_on else 0
        return 0

    # reads ----------------------------------------------------------------
    def read_positions(self, motor_ids: List[int], timeout: float = 0.01) -> Dict[int, int]:
        return {mid: self._pos[mid] for mid in motor_ids if mid in self._pos}

    def read_feedback(self, motor_id: int, timeout: float = 0.01) -> Optional[dict]:
        if motor_id not in self._pos:
            return None
        return {
            "position": self._pos[motor_id],
            "speed": 0,
            "load": 0,
            "voltage": 12.0,
            "temperature": 25.0,
        }
