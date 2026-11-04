"""hands.py — one ORCA hand side as a typed subsystem.

Wraps a controller :class:`~mabel_hw.drivers.hands.Hand` (which owns the
Feetech TTL serial + hot-plug watchdog via HandsController). Joint names here
are the driver's NATIVE ORCA names — the URDF thumb rename is a ROS-contract
concern and lives in mabel_hw.ros.hands_node, not in the API. Targets are
already-calibrated radians; the tick-clamp (calibration_data/hands) is applied
inside the OrcaHand driver, as always.
"""
from __future__ import annotations

from typing import Dict, Mapping

from ..types import HandState, monotonic

__all__ = ["HandSubsystem"]


class HandSubsystem:
    def __init__(self, side: str, hand, *, logger=None):
        self.side = side
        self._hand = hand                          # drivers.hands.Hand duck-type
        self._log = logger or (lambda *a, **k: None)

    # lifecycle is owned by the shared HandsController (facade-level)
    def connected(self) -> bool:
        # controller Hand keeps .hand = the live OrcaHand (None while absent)
        return getattr(self._hand, "hand", None) is not None

    # ── read ──────────────────────────────────────────────────────────────────
    def state(self) -> HandState:
        st = self._hand.get_state()
        if st is None:
            return HandState(side=self.side, position={}, velocity={},
                             effort={}, temperature={}, voltage={},
                             torque_enabled=False, connected=False,
                             stamp=monotonic())
        return HandState(
            side=self.side,
            position=dict(st.position), velocity=dict(st.velocity),
            effort=dict(st.effort), temperature=dict(st.temperature),
            voltage=dict(st.voltage),
            torque_enabled=bool(self._hand.is_torque_enabled()),
            connected=True, stamp=monotonic())

    def health(self) -> Dict[str, Dict[str, float]]:
        st = self.state()
        return {n: {"loadPct": round(st.effort.get(n, 0.0) * 100.0, 1),
                    "tempC": round(st.temperature.get(n, 0.0), 1),
                    "voltage": round(st.voltage.get(n, 0.0), 1)}
                for n in st.position}

    # ── write (lease-gated at the facade) ─────────────────────────────────────
    def set_joint_positions(self, targets: Mapping[str, float]) -> None:
        """ORCA joint names -> radians (no-op while torque is off, per the
        controller's free-drive-safe contract)."""
        self._hand.set_positions(dict(targets))

    def set_torque(self, on: bool) -> None:
        """Engage (hold-then-enable, no snap) or free-drive."""
        self._hand.set_torque(bool(on))

    @property
    def is_torque_enabled(self) -> bool:
        return bool(self._hand.is_torque_enabled())

    # ── safety (lease-FREE) ───────────────────────────────────────────────────
    def stop(self) -> None:
        """Hold-in-place: re-target current positions (keeps torque state —
        dropping torque under load would let go of whatever is held)."""
        st = self._hand.get_state()
        if st is not None and self.is_torque_enabled:
            try:
                self._hand.set_positions(dict(st.position))
            except Exception:
                pass
