#!/usr/bin/env python3
"""
serial_demo.py
==================

Control a Damiao DM-J4310 over the DM-USB2FDCAN serial adapter, demonstrating
every MIT control mode through :class:`mabel_actid.damiao_serial.DamiaoSerialBus`.

SAFETY
------
* Bench-mount the motor with the OUTPUT SHAFT FREE; keep a hand on the e-stop.
* The script ALWAYS disables the motor on exit (even on Ctrl-C / error).
* Motion demos use small, conservative setpoints. Read the values before raising.

Usage
-----
    # read-only: query position/vel/torque WITHOUT energizing (safe, do this first)
    python serial_demo.py --id 1 --read

    # energizing demos (shaft must be free):
    python serial_demo.py --id 1 --hold        # hold current position
    python serial_demo.py --id 1 --velocity 1.0   # spin at 1 rad/s for 2s
    python serial_demo.py --id 1 --move 0.3     # move +0.3 rad and hold
    python serial_demo.py --id 1 --torque 0.1   # apply 0.1 N*m for 1.5s
"""
from __future__ import annotations

import argparse
import sys
import time

import _bootstrap  # noqa: F401  (import puts src/ on sys.path)

from mabel_actid.damiao_serial import DamiaoSerialBus  # noqa: E402


def _print_state(bus, mid, label=""):
    fb = bus.get(mid)
    age = "n/a" if fb.t == 0 else f"{(time.perf_counter()-fb.t)*1e3:.0f}ms ago"
    print(f"  {label:10s} pos={fb.pos:+.4f} rad  vel={fb.vel:+.4f} rad/s  "
          f"tau={fb.tau:+.4f} N*m  err={fb.err}  Tmos={fb.t_mos:.0f}C "
          f"Trotor={fb.t_rotor:.0f}C  ({age})")


def _confirm_link(bus, mid) -> bool:
    """Non-energizing 0xCC status request; returns True if the motor replied."""
    print("[link] sending 0xCC status request (non-energizing)...")
    for _ in range(5):
        bus.refresh_status(mid)
        time.sleep(0.05)
    fb = bus.get(mid)
    if fb.t != 0:
        print("[link] ✅ motor replied.")
        _print_state(bus, mid, "status")
        return True
    print("[link] ❌ no reply — check power(red LED?), CAN_H/L wiring, bitrate, motor ID.")
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=None, help="serial port (auto-detect if omitted)")
    ap.add_argument("--id", type=int, default=1, help="motor CAN id")
    ap.add_argument("--motor", default="DM-J4310", help="motor profile name")
    ap.add_argument("--read", action="store_true", help="read state only (no energize)")
    ap.add_argument("--hold", action="store_true", help="enable and hold current position")
    ap.add_argument("--move", type=float, metavar="DRAD", help="move this many rad and hold")
    ap.add_argument("--velocity", type=float, metavar="W", help="hold this velocity (rad/s) for 2s")
    ap.add_argument("--torque", type=float, metavar="TAU", help="apply this torque (N*m) for 1.5s")
    ap.add_argument("--kp", type=float, default=3.0, help="position gain for --move/--hold")
    ap.add_argument("--kd", type=float, default=0.6, help="damping gain")
    ap.add_argument("--duration", type=float, default=2.0, help="motion hold time (s)")
    args = ap.parse_args()

    try:
        bus = DamiaoSerialBus(port=args.port)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1
    print(f"[i] opened {bus.port}; motor id={args.id} as {args.motor}")
    bus.add_motor(args.id, args.motor)
    mid = args.id

    try:
        if not _confirm_link(bus, mid):
            return 2
        if args.read:
            return 0

        # any motion path from here on -> energize, then ALWAYS disable in finally
        print("[motor] enabling (shaft should be free)...")
        bus.enable(mid)
        time.sleep(0.2)
        bus.refresh_status(mid); time.sleep(0.05)
        _print_state(bus, mid, "enabled")
        q0 = bus.get(mid).pos

        if args.move is not None:
            target = q0 + args.move
            print(f"[move] {q0:+.3f} -> {target:+.3f} rad  (kp={args.kp}, kd={args.kd})")
            t_end = time.perf_counter() + args.duration
            while time.perf_counter() < t_end:
                bus.track_position(mid, target, 0.0, args.kp, args.kd)
                time.sleep(0.01)
            _print_state(bus, mid, "reached")

        elif args.hold:
            print(f"[hold] holding {q0:+.3f} rad for {args.duration}s (kp={args.kp}, kd={args.kd})")
            t_end = time.perf_counter() + args.duration
            while time.perf_counter() < t_end:
                bus.track_position(mid, q0, 0.0, args.kp, args.kd)
                time.sleep(0.01)
            _print_state(bus, mid, "held")

        elif args.velocity is not None:
            print(f"[vel] {args.velocity:+.2f} rad/s for {args.duration}s (kd={args.kd})")
            t_end = time.perf_counter() + args.duration
            while time.perf_counter() < t_end:
                bus.hold_velocity(mid, args.velocity, args.kd)
                time.sleep(0.01)
            bus.hold_velocity(mid, 0.0, args.kd)  # ramp to stop
            time.sleep(0.2)
            _print_state(bus, mid, "spun")

        elif args.torque is not None:
            print(f"[torque] {args.torque:+.3f} N*m for 1.5s")
            t_end = time.perf_counter() + 1.5
            while time.perf_counter() < t_end:
                bus.torque(mid, args.torque)
                time.sleep(0.01)
            _print_state(bus, mid, "torqued")
        else:
            print("[i] no motion flag given; motor enabled and idle. (use --hold/--move/--velocity/--torque)")
            time.sleep(0.5)

        return 0
    except KeyboardInterrupt:
        print("\n[!] interrupted")
        return 130
    finally:
        print("[motor] disabling.")
        bus.shutdown(disable_all=True)


if __name__ == "__main__":
    raise SystemExit(main())
