#!/usr/bin/env python3
"""
change_motor_id.py — Change a DaMiao motor's CAN ID over SocketCAN.

Protocol (two-step write → save):
  1. Write new ID:  CAN 0x7FF, 8B  [old_L, old_H, 0x55, 0x08, new_L, new_H, 0, 0]
  2. Save to flash: CAN 0x7FF, 4B  [old_L, old_H, 0xAA, 0x01]

Usage
-----
  python scripts/change_motor_id.py --old 1 --new 2
  python scripts/change_motor_id.py --old 1 --new 2 --channel can1
"""

import argparse, time, sys

try:
    import can
except ImportError:
    sys.exit("python-can not found: pip install python-can")

GLOBAL_ID = 0x7FF


def change_id(channel: str, old_id: int, new_id: int):
    if not (1 <= old_id <= 15 and 1 <= new_id <= 15):
        sys.exit("Motor IDs must be 1–15.")
    if old_id == new_id:
        sys.exit("Old and new IDs are the same — nothing to do.")

    print(f"Opening {channel}…")
    try:
        bus = can.interface.Bus(channel=channel, bustype="socketcan")
    except Exception as e:
        sys.exit(f"Cannot open {channel}: {e}\n"
                 "Ensure the interface is up:  sudo ip link set can0 up type can bitrate 1000000")

    try:
        # Step 1: write new ID to register 0x08
        write_frame = can.Message(
            arbitration_id=GLOBAL_ID,
            data=[old_id & 0xFF, (old_id >> 8) & 0xFF,
                  0x55, 0x08,
                  new_id & 0xFF, (new_id >> 8) & 0xFF,
                  0x00, 0x00],
            is_extended_id=False)
        bus.send(write_frame)
        print(f"  → Sent write-ID command  (old={old_id} → new={new_id})")
        time.sleep(0.05)

        # Step 2: save to flash
        save_frame = can.Message(
            arbitration_id=GLOBAL_ID,
            data=[old_id & 0xFF, (old_id >> 8) & 0xFF, 0xAA, 0x01],
            is_extended_id=False)
        bus.send(save_frame)
        print("  → Sent save-to-flash command")
        time.sleep(0.1)

        print(f"\nDone. Motor ID changed {old_id} → {new_id}.")
        print("Power-cycle the motor for the new ID to take effect.")
    finally:
        bus.shutdown()


def main():
    ap = argparse.ArgumentParser(description="Change a DaMiao motor's CAN ID")
    ap.add_argument("--old",     type=int, required=True, help="Current motor ID (1–15)")
    ap.add_argument("--new",     type=int, required=True, help="New motor ID (1–15)")
    ap.add_argument("--channel", default="can0",          help="SocketCAN interface (default: can0)")
    args = ap.parse_args()

    change_id(channel=args.channel, old_id=args.old, new_id=args.new)


if __name__ == "__main__":
    main()
