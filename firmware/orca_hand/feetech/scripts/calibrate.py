#!/usr/bin/env python3
"""End-stop calibration (stall-detection sweep). See ``orcahand.calibrate``.

    python3 scripts/calibrate.py --model left           # full hand
    python3 scripts/calibrate.py --model left --sim      # hardware-free self-test
    python3 scripts/calibrate.py --model left --joints index_mcp index_pip
"""
import _common  # noqa: F401  (path bootstrap)
from orcahand.calibrate import calibrate_main

if __name__ == "__main__":
    raise SystemExit(calibrate_main())
