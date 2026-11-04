#!/usr/bin/env python3
"""Interactive host console for the Pico lift.

A thin REPL over LiftController with a live telemetry readout. Useful for
bring-up, tuning, and manual jogging from a PC.

    python3 console.py --port /dev/ttyACM0
"""

from __future__ import annotations

import argparse
import sys
import time

from lift_controller import LiftController

HELP = """\
Commands (forwarded to the lift, plus a few host helpers):
  up [e] / down [e]   open-loop jog
  vel <mm/s>          velocity hold
  pos <mm> [vmax]     move to height
  stop                halt
  calibrate / home    bring-up routines
  park / setpark <mm> resting height
  zero / set <mm>     redefine height
  watch               stream telemetry until Enter
  status              print one telemetry frame
  quit                exit
"""


def fmt(t) -> str:
    if t is None:
        return "(no telemetry yet)"
    return (f"{t.status_name:<6} {t.mode_name:<10} "
            f"h={t.height:7.2f}mm  v={t.vel:6.2f}mm/s  "
            f"vset={t.vset:6.2f}  eff={t.effort:+.2f}  "
            f"{'CAL' if t.calibrated else 'UNCAL'} (max {t.max_height:.1f})")


def watch(lift: LiftController) -> None:
    print("Streaming telemetry. Press Ctrl-C to stop.")
    try:
        while True:
            sys.stdout.write("\r" + fmt(lift.telemetry()) + "   ")
            sys.stdout.flush()
            time.sleep(0.1)
    except KeyboardInterrupt:
        print()


def main() -> int:
    ap = argparse.ArgumentParser(description="Pico lift host console")
    ap.add_argument("--port", "-p", default="/dev/ttyACM0",
                    help="serial port (default /dev/ttyACM0)")
    ap.add_argument("--baud", "-b", type=int, default=115200)
    args = ap.parse_args()

    try:
        lift = LiftController(args.port, args.baud)
    except Exception as exc:  # noqa: BLE001 - surface any open failure
        print(f"Failed to open {args.port}: {exc}", file=sys.stderr)
        return 1

    print(f"Connected to {args.port}. Type 'help' or 'quit'.")
    if not getattr(lift, "telemetry_ok", False):
        print("\n  !! No telemetry received from the Pico.\n"
              "     The firmware main loop probably isn't running. Fix with:\n"
              "       mpremote connect auto reset\n"
              "     then reconnect. (Check it boots: mpremote connect auto repl\n"
              "     -> press Ctrl-D -> you should see 'Lift firmware ready'.)\n")
    try:
        while True:
            try:
                line = input("lift> ").strip()
            except EOFError:
                break
            if not line:
                continue
            low = line.lower()
            if low in ("quit", "exit", "q"):
                break
            if low in ("help", "h", "?"):
                print(HELP)
                continue
            if low == "watch":
                watch(lift)
                continue
            if low == "status":
                print(fmt(lift.telemetry()))
                continue
            # Everything else is a raw firmware command.
            lift.send(line)
            time.sleep(0.1)
            print(fmt(lift.telemetry()))
    finally:
        lift.close()
        print("Disconnected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
