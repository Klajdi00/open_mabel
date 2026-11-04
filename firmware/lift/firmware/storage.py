# storage.py - Persist lift calibration + last position to the Pico flash.
#
# A tiny JSON file (config.STATE_FILE) on the MicroPython filesystem survives
# power-off, so the lift can remember its measured scale, travel limits, and
# last absolute height. Writes are infrequent (only on calibration and when
# motion stops) to spare the flash.

import json

import config

_VERSION = 1


def load():
    """Return the saved state dict, or None if absent/corrupt."""
    try:
        with open(config.STATE_FILE) as f:
            data = json.load(f)
        if isinstance(data, dict) and data.get("v") == _VERSION:
            return data
    except (OSError, ValueError):
        pass
    return None


def save(mm_per_count, min_height, max_height, height, calibrated):
    """Persist the lift state. Returns True on success."""
    try:
        with open(config.STATE_FILE, "w") as f:
            json.dump({
                "v": _VERSION,
                "mm_per_count": mm_per_count,
                "min_height": min_height,
                "max_height": max_height,
                "height": height,
                "calibrated": calibrated,
            }, f)
        return True
    except OSError:
        return False
