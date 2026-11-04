#!/usr/bin/env python3
"""exp_neck.py — detect + report MABEL's Dynamixel neck. READ-ONLY (no motion).

The neck counterpart of exp_audio.py: it probes the physical servo bus through
the same `mabel_hw.drivers.dynamixel` code the HAL uses and prints what it finds,
so "are the neck motors on the bus?" is one command:

    python3 experiments/exp_neck.py                 # scan the configured adapter
    python3 experiments/exp_neck.py --device /dev/ttyUSB0
    python3 experiments/exp_neck.py --all-bauds     # sweep every common baud

It broadcast-pings (reads Model Number + firmware) and does a targeted read of
Present Position / Voltage / Temperature for each responder. It NEVER enables
torque or writes a goal, so it is safe to run any time — including while a human
is holding the head to level it for calibration.

⚠ The HAL is the sole owner of the bus. If `mabel-hardware-bridge` is running
with NECK enabled it already holds /dev/ttyUSB0, and this scan will find the port
busy — stop the HAL first (`sudo systemctl stop mabel-hardware-bridge`) or run
this before it starts.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mabel_hw import config                                          # noqa: E402
from mabel_hw.drivers import dynamixel as dx                         # noqa: E402

PASS, FAIL, INFO = "\033[32m✓\033[0m", "\033[31m✗\033[0m", "\033[2m▸\033[0m"


def _id_map(spec: str) -> dict:
    out = {}
    for tok in spec.split(","):
        tok = tok.strip()
        if tok:
            name, _, sid = tok.partition(":")
            out[name.strip()] = int(sid.strip())
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Detect the Dynamixel neck (read-only).")
    ap.add_argument("--device", default=config.NECK.get("device", "/dev/ttyUSB0"))
    ap.add_argument("--serial", default=config.NECK.get("device_serial", "FTB8HR1Y"))
    ap.add_argument("--baud", type=int, default=int(config.NECK.get("baud", 1000000)))
    ap.add_argument("--all-bauds", action="store_true",
                    help="sweep 57.6k/1M/2M/3M/115.2k instead of just the configured baud")
    args = ap.parse_args()

    path = dx.resolve_device(args.device, args.serial)
    print(f"{INFO} adapter: serial={args.serial!r} device={args.device!r} -> "
          f"{path or 'NOT FOUND'}")
    if not path:
        print(f"{FAIL} no USB serial adapter present. Check the USB-C link to thor.")
        return 2

    bauds = (57600, 1000000, 2000000, 3000000, 115200) if args.all_bauds else (args.baud,)
    result = dx.scan(device=args.device, serial_id=args.serial, bauds=bauds)
    if result.get("error"):
        print(f"{FAIL} {result['error']}")
        return 2

    expected = _id_map(config.NECK.get("dxl_ids", ""))
    found_any = False
    for baud, motors in result.get("bauds", {}).items():
        if not isinstance(motors, dict) or "error" in motors:
            reason = motors.get("error") if isinstance(motors, dict) else motors
            print(f"{INFO} {baud:>8} baud: {reason}")
            continue
        if not motors:
            print(f"{INFO} {baud:>8} baud: no response")
            continue
        found_any = True
        print(f"{PASS} {baud:>8} baud: {len(motors)} servo(s)")
        for i in sorted(motors):
            m = motors[i]
            deg = round((m.get('position', 0.0)) * 180.0 / 3.14159, 1)
            print(f"      ID {i:<3} {str(m.get('model')):<12} model#{m.get('model_no')}  "
                  f"pos={m.get('position')}rad ({deg}deg)  "
                  f"{m.get('voltage')}V  {m.get('temperature')}C  err={m.get('err')}")

    # cross-check against config
    if found_any and expected:
        live_ids = {i for motors in result["bauds"].values()
                    if isinstance(motors, dict) and "error" not in motors
                    for i in motors}
        print()
        for name, sid in expected.items():
            ok = sid in live_ids
            print(f"  {PASS if ok else FAIL} config joint {name!r} -> id {sid} "
                  f"{'present' if ok else 'MISSING on the bus'}")
        missing = [n for n, s in expected.items() if s not in live_ids]
        if missing:
            print(f"\n{FAIL} {len(missing)} configured joint(s) not answering: {missing}")
            return 1

    if not found_any:
        print(f"\n{FAIL} no Dynamixels answered. Check 12 V power + the TTL daisy-chain.")
        return 1
    print(f"\n{PASS} neck bus healthy. Calibrate with the setup GUI "
          "(level the head, then 'Set Level').")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
