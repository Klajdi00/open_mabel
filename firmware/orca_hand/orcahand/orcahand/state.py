"""Immutable snapshots of hand state, in SI units.

``HandState`` is what ``OrcaHand.get_state()`` returns: a single coherent
reading of every joint, keyed by joint name. Positions are radians, velocities
rad/s, efforts a normalised [-1, 1] load, temperatures deg C, voltages volts.
Frozen so it is safe to hand to another thread (e.g. a ROS publisher) without
copying.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class HandState:
    """One synchronized reading of the whole hand (name -> value)."""

    position: Dict[str, float] = field(default_factory=dict)   # rad
    velocity: Dict[str, float] = field(default_factory=dict)   # rad/s
    effort: Dict[str, float] = field(default_factory=dict)     # normalised [-1, 1]
    temperature: Dict[str, float] = field(default_factory=dict)  # deg C
    voltage: Dict[str, float] = field(default_factory=dict)    # V
    raw_position: Dict[str, int] = field(default_factory=dict)  # 0-4095 ticks
    stamp: Optional[float] = None  # wall-clock seconds, set by the caller

    def positions_array(self, joint_names: List[str]) -> List[float]:
        """Positions [rad] in the given joint order (NaN if unread)."""
        return [self.position.get(n, float("nan")) for n in joint_names]

    def hottest(self) -> Optional[tuple]:
        """``(joint_name, temp_C)`` of the hottest joint, or None if empty."""
        if not self.temperature:
            return None
        name = max(self.temperature, key=self.temperature.get)
        return name, self.temperature[name]
