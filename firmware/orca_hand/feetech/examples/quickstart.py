#!/usr/bin/env python3
"""Minimal OrcaHand usage -- the 30-second tour.

Run against hardware:        python examples/quickstart.py --model right
Run with no hardware (mock):  python examples/quickstart.py --model right --mock
"""

import argparse
import math
import time

from orcahand import OrcaHand


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="right", choices=["left", "right"])
    ap.add_argument("--port", default=None, help="serial device ('' = auto-detect)")
    ap.add_argument("--mock", action="store_true", help="use the in-memory bus")
    args = ap.parse_args()

    hand = (OrcaHand.mock(args.model) if args.mock
            else OrcaHand(args.model, port=args.port))

    with hand:                                   # connect() / disconnect()
        print(f"{hand.side} hand: {len(hand.joint_names)} joints on {hand.bus.port}")
        if hand.missing_joints:
            print("  not responding:", hand.missing_joints)

        # 1. command a few joints, in radians (partial dict is fine)
        hand.set_joint_positions({"index_mcp": 0.6, "middle_mcp": 0.6})
        time.sleep(0.5)

        # 2. named pose
        hand.set_named_position("pinch")
        time.sleep(0.5)

        # 3. a smooth sinusoidal wave on the index finger
        for k in range(40):
            a = 0.5 + 0.4 * math.sin(k / 6.0)
            hand.set_joint_positions({"index_pip": a, "index_mcp": a})
            time.sleep(0.02)

        # 4. read state back
        st = hand.get_state()
        for name in hand.joint_names[:3]:
            print(f"  {name:12s}  {st.position.get(name, float('nan')):+.3f} rad")
        hot = st.hottest()
        if hot:
            print(f"  hottest joint: {hot[0]} @ {hot[1]:.0f} C")


if __name__ == "__main__":
    main()
