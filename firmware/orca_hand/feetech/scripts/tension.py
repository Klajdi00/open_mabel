#!/usr/bin/env python3
"""Seat the tendons with a gentle oscillation before calibration.

Sweeps every joint back and forth around the centre of its calibrated range at
low torque, so the cables settle into their spools and the subsequent end-stop
calibration is repeatable. Run after assembly or re-tensioning.

    python3 scripts/tension.py --model left
    python3 scripts/tension.py --model left --duration 60 --amplitude 25
"""
from __future__ import annotations

import argparse
import math
import time

from _common import Term, add_common_args, make_hand


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Gentle tendon-seating oscillation.")
    add_common_args(ap)
    ap.add_argument("--duration", type=float, default=30.0, help="seconds (default 30)")
    ap.add_argument("--amplitude", type=float, default=15.0,
                    help="percent of range each side of centre (default 15)")
    ap.add_argument("--hz", type=float, default=0.5, help="oscillation frequency (default 0.5)")
    ap.add_argument("--torque", type=int, default=300, help="torque ceiling 0-1000 (default 300)")
    args = ap.parse_args(argv)

    t = Term(color=False if args.no_color else None)
    t.banner("ORCA HAND  ·  tendon tensioning",
             f"{args.model.upper()} hand  ·  gentle seat pass")

    hand = make_hand(args, torque_limit=args.torque)
    with hand:
        joints = hand.joint_names
        t.kv("port", hand.bus.port)
        t.kv("joints", f"{len(joints)}")
        t.kv("duration", f"{args.duration:.0f} s")
        t.kv("amplitude", f"±{args.amplitude:.0f} %  @ {args.hz:.2f} Hz")
        t.kv("torque ceiling", f"{args.torque} / 1000")
        print()

        t0 = time.time()
        try:
            while time.time() - t0 < args.duration:
                el = time.time() - t0
                pct = 50.0 + args.amplitude * math.sin(2 * math.pi * args.hz * el)
                hand.set_joint_percent({j: pct for j in joints})
                t.live(f"      {t.cyan('tensioning')} {el:4.1f}/{args.duration:.0f}s  "
                       f"target {pct:5.1f}%  {t.bar(el / args.duration)}")
                time.sleep(0.02)
            t.endlive()
            hand.set_joint_percent({j: 50.0 for j in joints})
            time.sleep(0.3)
            t.info(t.ok + " tensioning complete — joints centred")
        except KeyboardInterrupt:
            t.endlive()
            t.info(t.yellow("interrupted — centring and releasing"))
            hand.set_joint_percent({j: 50.0 for j in joints})
            time.sleep(0.3)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
