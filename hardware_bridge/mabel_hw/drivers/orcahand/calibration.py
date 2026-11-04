"""Calibration I/O: the measured raw tick end-stops, per motor.

A calibration file records, for every motor ID, the raw tick range swept during
calibration. It is the hard safety clamp -- no command is ever mapped outside
``[min, max]``.

File format is chosen by extension: ``<side>_calibration.yaml`` (preferred, what the
Control Center's calibration tab writes) or ``<side>_calibration.json`` (legacy, still
read). Both carry the SAME schema, keyed by motor-id string::

    "7":
      joint: index_mcp
      min: 1885
      max: 2910
      center: 2397
      range: 1025

Use :func:`read_mapping` / :func:`dump_mapping` for round-trips so the format always
follows the path's extension.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Union

import yaml

from .constants import RAW_CENTER, RAW_MAX, RAW_MIN


def _is_yaml(path: Union[str, Path]) -> bool:
    return str(path).lower().endswith((".yaml", ".yml"))


def read_mapping(path: Union[str, Path]) -> Dict[str, dict]:
    """Read a calibration mapping (motor-id string -> entry), YAML or JSON by extension."""
    text = Path(path).read_text()
    data = yaml.safe_load(text) if _is_yaml(path) else json.loads(text)
    return data or {}


def dump_mapping(path: Union[str, Path], data: Dict[str, dict], *, backup: bool = True) -> None:
    """Atomically write a calibration mapping, format chosen by the path extension.

    Keeps a ``.bak`` of any existing file. String motor-id keys are preserved so YAML
    and JSON files are interchangeable (both loaders coerce the key with ``int(k)``).
    """
    path = Path(path)
    if backup and path.exists():
        path.with_suffix(path.suffix + ".bak").write_text(path.read_text())
    if _is_yaml(path):
        body = yaml.safe_dump({str(k): v for k, v in data.items()},
                              sort_keys=False, default_flow_style=False)
    else:
        body = json.dumps({str(k): v for k, v in data.items()}, indent=4) + "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(body)
    tmp.replace(path)


@dataclass(frozen=True)
class MotorCalibration:
    motor_id: int
    raw_min: int
    raw_max: int
    center: int
    offset: int = 0      # software wrap offset; see orcahand.joints.Joint (0 = legacy)

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
        raw = read_mapping(path)
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
                offset=int(v.get("offset", 0)) % 4096,
            )
        return cls(by_id)
