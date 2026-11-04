#!/usr/bin/env python3
"""Cycle through named hand poses (a quick smoke-test / demo).

Poses are defined in :data:`orcahand.poses.POSES` and applied as percent of each
joint's calibrated range.

    python3 scripts/poses.py --model left                 # cycle all poses once
    python3 scripts/poses.py --model left --pose fist pinch point
    python3 scripts/poses.py --model left --loops 0       # loop forever
    python3 scripts/poses.py --list
"""
from __future__ import annotations

import argparse
import time

from _common import Term, add_common_args, make_hand
from orcahand.poses import POSES


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Cycle named hand poses.")
    add_common_args(ap)
    ap.add_argument("--pose", nargs="+", default=None, help="subset of poses to play")
    ap.add_argument("--loops", type=int, default=1, help="repeats; 0 = forever (default 1)")
    ap.add_argument("--hold", type=float, default=1.5, help="seconds per pose (default 1.5)")
    ap.add_argument("--list", action="store_true", help="list available poses and exit")
    args = ap.parse_args(argv)

    t = Term(color=False if args.no_color else None)
    if args.list:
        print(t.bold("available poses:"), ", ".join(sorted(POSES)))
        return 0

    seq = args.pose or list(POSES)
    unknown = [p for p in seq if p not in POSES]
    if unknown:
        t.info(t.bad + f" unknown pose(s): {unknown}; options: {sorted(POSES)}")
        return 2

    t.banner("ORCA HAND  ·  pose demo", f"{args.model.upper()} hand")
    hand = make_hand(args)
    with hand:
        t.kv("port", hand.bus.port)
        t.kv("sequence", " → ".join(seq))
        t.kv("hold / loops", f"{args.hold:.1f}s  /  {'∞' if args.loops == 0 else args.loops}")
        print()
        loop = 0
        try:
            while args.loops == 0 or loop < args.loops:
                loop += 1
                for name in seq:
                    hand.set_named_position(name)
                    t.info(f"{t.cyan(name)}  {t.grey(f'(loop {loop})')}")
                    time.sleep(args.hold)
            t.info(t.ok + " done")
        except KeyboardInterrupt:
            t.info(t.yellow("interrupted"))
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
