#!/usr/bin/env python3
"""Ping the Feetech bus and list which motor IDs answer.

    python3 scripts/scan.py --model left
"""
import _common  # noqa: F401  (path bootstrap)
from orcahand.cli import scan_main

if __name__ == "__main__":
    raise SystemExit(scan_main())
