"""Calibration I/O: the measured raw tick end-stops, per motor.

A calibration file records, for every motor ID, the raw tick range swept during
calibration (``calibrate_limits.py`` in the firmware). It is the hard safety
clamp -- no command is ever mapped outside ``[min, max]``.

File format (JSON, keyed by motor-id string) -- the same file the firmware's
``data/calibration/joint_limits.json`` and the ROS package's
``<side>_calibration.json`` already produce::

    {
      "7": {"joint": "index_mcp", "min": 1885, "max": 2910,
            "center": 2397, "range": 1025},
      ...
    }
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Union

from .constants import RAW_CENTER, RAW_MAX, RAW_MIN


@dataclass(frozen=True)
class MotorCalibration:
    motor_id: int
    raw_min: int
    raw_max: int
    center: int

    @property
    def span(self) -> int:
        return abs(self.raw_max - self.raw_min)


class Calibration:
    """Per-motor tick limits, with a safe full-range fallback."""

    def __init__(self, by_id: Dict[int, MotorCalibration]):
        self._by_id = by_id

    def __contains__(self, motor_id: int) -> bool:
        return motor_id in self._by_id

    def get(self, motor_id: int) -> MotorCalibration:
        """Calibration for ``motor_id``, or a full 0-4095 fallback if absent."""
        if motor_id in self._by_id:
            return self._by_id[motor_id]
        return MotorCalibration(motor_id, RAW_MIN, RAW_MAX, RAW_CENTER)

    @property
    def motor_ids(self) -> list:
        return sorted(self._by_id)

    @property
    def is_empty(self) -> bool:
        return not self._by_id

    @classmethod
    def empty(cls) -> "Calibration":
        return cls({})

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "Calibration":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"calibration file not found: {path}")
        with open(path) as f:
            raw = json.load(f)
        by_id: Dict[int, MotorCalibration] = {}
        for k, v in raw.items():
            mid = int(k)
            lo = int(v["min"])
            hi = int(v["max"])
            by_id[mid] = MotorCalibration(
                motor_id=mid,
                raw_min=lo,
                raw_max=hi,
                center=int(v.get("center", (lo + hi) // 2)),
            )
        return cls(by_id)
