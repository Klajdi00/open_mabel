#!/usr/bin/env python3
"""exp_lift.py — detect + report MABEL's Pico lift column. READ-ONLY (no motion).

The lift counterpart of exp_neck.py / exp_audio.py: it locates the Pico by USB
vendor id (exactly as the HAL's ``mabel_hw.drivers.lift`` does) and reads
telemetry through the same firmware host API, so "is the lift connected?" is one
command:

    python3 experiments/exp_lift.py                 # auto-find the Pico by VID
    python3 experiments/exp_lift.py --device /dev/ttyACM1
    python3 experiments/exp_lift.py --device /dev/serial/by-id/usb-MicroPython...

It opens the serial link, reads height / velocity / homed + the min/max travel
envelope, then closes. It NEVER homes, calibrates, drives, or enables torque, so
it is safe to run any time.

Why VID and not a fixed ttyACM: /dev/ttyACM numbering is arrival-order, so the
Pico is NOT reliably ACM0 (on the current hub ACM0 is the base Teensy). The
driver — and this tool — resolve the board by USB vendor 0x2E8A (RP2040/RP2350),
which the Teensy (0x16C0) and hand adapters (0x1A86) can never masquerade as.

⚠ The HAL is the sole owner of the port. If ``mabel-hardware-bridge`` is running
with LIFT enabled it already holds the Pico serial and this open will fail busy —
stop the HAL first (``sudo systemctl stop mabel-hardware-bridge``) or run this
before it starts.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mabel_hw import config                                          # noqa: E402
from mabel_hw.drivers import lift as liftmod                         # noqa: E402

PASS, FAIL, INFO = "\033[32m✓\033[0m", "\033[31m✗\033[0m", "\033[2m▸\033[0m"


def _list_rp_boards():
    """Every USB-CDC device whose vendor is the Pico's (0x2E8A)."""
    try:
        import serial.tools.list_ports as lp
    except Exception as e:                                           # noqa: BLE001
        print(f"{FAIL} pyserial not available: {e}")
        return []
    return [c for c in lp.comports() if c.vid == liftmod._PICO_VID]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default=config.LIFT["port"],
                    help="serial path, or 'auto' (default) to resolve by USB VID")
    ap.add_argument("--baud", type=int, default=int(config.LIFT["baud"]))
    ap.add_argument("--settle", type=float, default=1.5,
                    help="seconds to let the telemetry thread fill (default 1.5)")
    args = ap.parse_args()

    print(f"{INFO} scanning for Pico boards (USB VID "
          f"0x{liftmod._PICO_VID:04x})...")
    boards = _list_rp_boards()
    if boards:
        for c in boards:
            print(f"  {PASS} {c.device}  pid=0x{(c.pid or 0):04x}  "
                  f"{c.serial_number or '?'}  ({c.description})")
    else:
        print(f"  {FAIL} no RP2040/Pico USB serial found")

    driver = liftmod.LiftDriver(port=args.device, baud=args.baud)
    resolved = driver._resolve_port()
    print(f"{INFO} configured port : {args.device!r}")
    print(f"{INFO} resolved port   : {resolved!r}")
    if resolved is None:
        print(f"{FAIL} nothing to open — lift is ABSENT")
        return 1

    try:
        driver.connect()
    except Exception as e:                                           # noqa: BLE001
        print(f"{FAIL} connect failed on {resolved}: {e}")
        return 1

    time.sleep(args.settle)
    st = driver.read()
    driver.close()

    if not st.connected or st.height is None:
        print(f"{FAIL} opened {resolved} but no telemetry — wrong device or hung Pico")
        return 1

    print(f"{PASS} lift CONNECTED on {resolved}")
    print(f"     height   = {st.height:.4f} m")
    print(f"     velocity = {st.velocity:.4f} m/s")
    print(f"     homed    = {st.homed}")
    print(f"     travel   = [{st.min_pos:.3f}, {st.max_pos:.3f}] m")
    if not st.homed:
        print(f"{INFO} not homed yet — home/calibrate from a control studio before use")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
