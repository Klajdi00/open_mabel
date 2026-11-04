#!/usr/bin/env python3
"""hardware_bridge.py — MABEL's single hardware-abstraction layer (the HAL).

ONE process, the only owner of ALL robot hardware:

    base    MAXSwerve Teensy (ethernet)   — validated ps5_teleop IK + calibration
    arms    DaMiao / OpenArm (Teensy-ethernet or CAN, MIT)
    body    DaMiao torso + neck (CAN, MIT; RSSR neck via Teensy when enabled)
    hands   ORCA hands (Feetech STS/SCS over TTL serial)
    lift    BTS7960 DC motor on a Pico (USB serial)
    lidar   RPLIDAR A2M12 (CP2102 serial)
    cameras D435 front + ZED-M head + wrist D405s (native SDK, per-process)

This file is the stable ENTRY POINT (systemd's run_bridge.sh execs it with
cwd=hardware_bridge) — the implementation lives in the installable mabel_hw
package (mabel_hw/ros/bridge.py supervises; mabel_hw.MabelHardware is the
importable API the nodes share). Enable/disable + tune each subsystem in
mabel_hw/config.py (or via MABEL_HW_* env vars).
"""

import os
import sys

# Runnable from a bare checkout: make the bridge dir importable even when
# mabel_hw was never pip-installed.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Re-exports — tools/singleton_test.sh, server/real_run.sh docs, and the tests
# treat this module as the canonical surface for the lock + registry behavior.
from mabel_hw.singleton import (_LOCK_PATH, _acquire_singleton,  # noqa: E402,F401
                                _owner_alive, _try_lock)
from mabel_hw.ros.bridge import _REGISTRY, _build_enabled, main  # noqa: E402,F401

if __name__ == "__main__":
    main()
