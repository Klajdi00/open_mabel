#!/usr/bin/env python3
"""Connect to a hand, report what's on the bus, and print live telemetry.

A quick health check: which motors answer, and a per-joint table of angle,
temperature, voltage and load. Use ``--watch`` for a refreshing dashboard.

    python3 scripts/bringup.py --model left
    python3 scripts/bringup.py --model left --watch
    python3 scripts/bringup.py --model left --pose flat
"""
from __future__ import annotations

import argparse
import math
import time

from _common import Term, add_common_args, make_hand


def _print_table(t: Term, hand, clear: bool) -> None:
    state = hand.get_state()
    if clear and t.color:
        print("\033[H\033[2J", end="")  # home + clear
    print(t.bold(f"  {hand.side.upper()} hand   ") +
          t.grey(time.strftime("%H:%M:%S")))
    print(t.grey(f"  {'joint':<12}{'rad':>8}{'deg':>8}{'temp':>7}{'volt':>7}{'load':>7}"))
    hot = None
    for name in hand.joint_names:
        if name not in state.position:
            print(f"  {name:<12}{t.grey('       —  (no reply)')}")
            continue
        rad = state.position[name]
        temp = state.temperature.get(name, 0.0)
        volt = state.voltage.get(name, 0.0)
        load = state.effort.get(name, 0.0)
        tcol = t.red if temp >= 60 else (t.yellow if temp >= 50 else t.green)
        print(f"  {name:<12}{rad:>8.2f}{math.degrees(rad):>8.1f}"
              f"{tcol(f'{temp:>7.1f}')}{volt:>7.1f}{load:>7.2f}")
        if hot is None or temp > hot[1]:
            hot = (name, temp)
    if hot:
        print(t.grey(f"  hottest: {hot[0]} @ {hot[1]:.0f}°C"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Hand bring-up / telemetry dashboard.")
    add_common_args(ap)
    ap.add_argument("--watch", action="store_true", help="refresh continuously")
    ap.add_argument("--hz", type=float, default=4.0, help="refresh rate for --watch")
    ap.add_argument("--pose", default=None, help="drive to a named pose on connect")
    ap.add_argument("--no-torque", action="store_true", help="connect with torque off")
    args = ap.parse_args(argv)

    t = Term(color=False if args.no_color else None)
    t.banner("ORCA HAND  ·  bring-up", f"{args.model.upper()} hand")

    hand = make_hand(args)
    hand.connect(enable_torque=not args.no_torque, scan=True)
    try:
        found = sorted(hand._active_ids)
        t.kv("port", hand.bus.port)
        t.kv("motors responding", f"{len(found)} / {len(hand.motor_ids)}  {found}")
        if hand.missing_joints:
            t.info(t.yellow(f"missing: {hand.missing_joints}"))
        if args.pose:
            hand.set_named_position(args.pose)
            t.info(f"holding pose {t.cyan(args.pose)}")
            time.sleep(0.5)
        print()

        if args.watch:
            period = 1.0 / max(0.5, args.hz)
            while True:
                _print_table(t, hand, clear=True)
                print(t.grey("  (Ctrl-C to stop)"))
                time.sleep(period)
        else:
            _print_table(t, hand, clear=False)
    except KeyboardInterrupt:
        pass
    finally:
        hand.disconnect()
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
