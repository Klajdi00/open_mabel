#!/usr/bin/env python3
"""Turn off torque on all motors — the hand goes limp immediately.

    python control/torque_off.py
    python control/torque_off.py --port /dev/ttyUSB0
    python control/torque_off.py --hand right
"""
import argparse

from utils.hand_config import BOLD, RESET, YELLOW, connect_hand


def main() -> int:
    ap = argparse.ArgumentParser(description="Disable torque on all motors")
    ap.add_argument("--port", type=str, default=None)
    ap.add_argument("--hand", choices=["left", "right"], default="left")
    ap.add_argument("--mock", action="store_true", help="no hardware")
    args = ap.parse_args()

    hand = connect_hand(args.hand, port=args.port, mock=args.mock, enable_torque=False)
    try:
        hand.disable_torque()
    finally:
        hand.disconnect()
    print(f"{YELLOW}{BOLD}Torque OFF — hand is limp.{RESET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
