"""calibration — the bridge's one view of MABEL's as-deployed calibration.

Authoritative folder: ``hardware_bridge/calibration_data/`` — a mirror of the
upstream studio tree (``mabel.json``) plus the hand tick-clamp files
(``hands/<side>_calibration.json``). The studios dual-write here on every save;
every driver reads through :class:`CalibrationStore`, which falls back to the
legacy locations when the mirror is absent (a stale upstream checkout is never
fatal). Anchor math (center vs neutral, per family) lives in schema.py.
"""
from .schema import JointCalib
from .store import CalibrationStore

__all__ = ["CalibrationStore", "JointCalib"]
