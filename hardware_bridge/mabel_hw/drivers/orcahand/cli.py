"""Tiny CLI helpers. ``orcahand-scan`` pings the bus and prints what answers."""

from __future__ import annotations

import argparse

from .bus import FeetechBus
from .config import HandConfig


def scan_main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Ping the Feetech bus and list motors.")
    ap.add_argument("--port", default=None, help="serial device ('' = auto-detect)")
    ap.add_argument("--model", default=None, choices=HandConfig.available_models(),
                    help="annotate found IDs with this hand's joint names")
    args = ap.parse_args(argv)

    bus = FeetechBus(port=args.port).open()
    try:
        found = bus.scan(range(1, 21))
    finally:
        bus.close()

    names = {}
    if args.model:
        cfg = HandConfig.load(model=args.model)
        names = {j.motor_id: j.name for j in cfg.joints}

    print(f"port {bus.port}: {len(found)} motors responding")
    for mid in found:
        print(f"  id {mid:>2}  {names.get(mid, '')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(scan_main())
