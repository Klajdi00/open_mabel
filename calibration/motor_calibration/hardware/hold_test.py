#!/usr/bin/env python3
"""Minimal, gentle energizing self-test for one Damiao motor on SocketCAN.

Enables, reads state, holds the CURRENT position with low gains for a short
time (should barely move), then ALWAYS disables. This is the smallest possible
real-hardware test before running collect.py.
"""
import argparse
import time

import _bootstrap  # noqa: F401
from mabel_actid.damiao import DamiaoBus


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motor", default="DM-J4310")
    ap.add_argument("--id", type=int, default=1)
    ap.add_argument("--channel", default="can0")
    ap.add_argument("--kp", type=float, default=3.0)
    ap.add_argument("--kd", type=float, default=0.6)
    ap.add_argument("--duration", type=float, default=1.5)
    ap.add_argument("--rate", type=float, default=200.0)
    args = ap.parse_args()

    with DamiaoBus(channel=args.channel) as bus:
        bus.add_motor(args.id, args.motor)
        bus.clear_faults(args.id)
        time.sleep(0.05)
        bus.enable(args.id)
        time.sleep(0.2)
        fb = bus.get(args.id)
        # DaMiao byte0 high nibble is a STATE code: 0=disabled, 1=enabled,
        # >=8 are genuine faults (8/9 over/under-volt, A overcurrent, B/C
        # overtemp, D comm-loss, E overload).
        state = {0: "disabled", 1: "enabled"}.get(fb.err, f"FAULT 0x{fb.err:X}")
        print(f"[enabled] pos={fb.pos:+.4f} vel={fb.vel:+.4f} tau={fb.tau:+.4f} "
              f"state={state} Tmos={fb.t_mos:.0f}C Trotor={fb.t_rotor:.0f}C")
        if fb.t == 0:
            print("  WARNING: no feedback received after enable; aborting")
            return
        if fb.err >= 8:
            print(f"  WARNING: motor fault 0x{fb.err:X}; aborting motion")
            return

        q0 = fb.pos
        print(f"[hold] holding {q0:+.4f} rad for {args.duration:.1f}s "
              f"(kp={args.kp}, kd={args.kd}) ... should barely move")
        dt = 1.0 / args.rate
        t_end = time.perf_counter() + args.duration
        i = 0
        while time.perf_counter() < t_end:
            bus.track_position(args.id, q0, 0.0, args.kp, args.kd)
            if i % int(args.rate // 4) == 0:
                fb = bus.get(args.id)
                print(f"  t={time.perf_counter():.2f} pos={fb.pos:+.4f} "
                      f"vel={fb.vel:+.4f} tau={fb.tau:+.4f}")
            i += 1
            time.sleep(dt)

        fb = bus.get(args.id)
        print(f"[done] pos={fb.pos:+.4f} (drift {fb.pos - q0:+.4f} rad) "
              f"vel={fb.vel:+.4f} tau={fb.tau:+.4f}")
    print("[motor] disabled.")


if __name__ == "__main__":
    main()
