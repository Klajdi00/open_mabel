"""Test bootstrap for the hardware bridge.

The bridge runs with cwd=hardware_bridge/ (run_bridge.sh does `cd` before exec), so
all internal imports are top-level names (`config`, `base`, `hands`, `subsystems`,
`cameras`). Tests emulate that by putting the bridge folder first on sys.path,
regardless of where pytest is invoked from.

Marker `ros`: tests that import rclpy (subsystem nodes, the bridge entrypoint).
They auto-skip on hosts without ROS 2 (the Mac); on thor (mabel conda env) they run.
"""
import os
import sys

import pytest

BRIDGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BRIDGE_DIR not in sys.path:
    sys.path.insert(0, BRIDGE_DIR)


def _has_rclpy() -> bool:
    try:
        import rclpy  # noqa: F401
        return True
    except Exception:
        return False


HAS_RCLPY = _has_rclpy()


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "ros: needs rclpy (ROS 2); auto-skipped when unavailable")


def pytest_collection_modifyitems(config, items):
    if HAS_RCLPY:
        return
    skip = pytest.mark.skip(reason="rclpy not importable on this host")
    for item in items:
        if "ros" in item.keywords:
            item.add_marker(skip)
