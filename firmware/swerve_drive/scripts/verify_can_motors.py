#!/usr/bin/env python3
"""
Verify CAN connectivity for the 3-swerve robot motors.

Motor IDs:
  ID 1 - Right Front drive
  ID 2 - Right Front steer
  ID 3 - Left Front drive
  ID 4 - Left Front steer
  ID 5 - Back drive
  ID 6 - Back steer

Run from the repo root:
  python extra/verify_can_motors.py

Make sure can0 is up first:
  sudo ip link set can0 type can bitrate 1000000
  sudo ip link set can0 up
  sudo ip link set can0 txqueuelen 1000
"""

import socket
import struct
import sys
import time

CAN_IF = "can0"

MOTOR_NAMES = {
    1: "RF drive",
    2: "RF steer",
    3: "LF drive",
    4: "LF steer",
    5: "B  drive",
    6: "B  steer",
}

DRIVE_IDS = {1, 3, 5}
STEER_IDS = {2, 4, 6}

# REV SPARK CAN arbitration ID format:
#   bits [31:24] = DEVICE_TYPE (0x02)
#   bits [23:16] = MANUFACTURER (0x05)
#   bits [15:10] = API class
#   bits [ 9: 6] = API index
#   bits [ 5: 0] = device ID
_REV_PREFIX = 0x0205
_CAN_EFF_FLAG = 0x80000000
_CAN_EFF_MASK = 0x1FFFFFFF
_CAN_FRAME_FMT = "=IB3x8s"
_CAN_FRAME_SIZE = struct.calcsize(_CAN_FRAME_FMT)


def sniff_device_ids(ifname: str, duration: float) -> set:
    """Passively listen on the CAN bus and return device IDs that transmitted."""
    sock = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
    sock.bind((ifname,))
    sock.settimeout(0.05)
    seen = set()
    t_end = time.monotonic() + duration
    while time.monotonic() < t_end:
        try:
            data = sock.recv(_CAN_FRAME_SIZE)
            if len(data) < _CAN_FRAME_SIZE:
                continue
            can_id, _dlc, _ = struct.unpack(_CAN_FRAME_FMT, data)
            if not (can_id & _CAN_EFF_FLAG):
                continue  # skip standard (11-bit) frames
            arb_id = can_id & _CAN_EFF_MASK
            if (arb_id >> 16) == _REV_PREFIX:
                device_id = arb_id & 0x3F
                if 1 <= device_id <= 62:
                    seen.add(device_id)
        except TimeoutError:
            pass
    sock.close()
    return seen


def main():
    try:
        from sparkcan_py import SparkFlex
    except ImportError:
        print("ERROR: sparkcan_py not importable. Activate the venv first.")
        sys.exit(1)

    print(f"Sniffing {CAN_IF} for 2 s to detect physically present devices...")
    seen_ids = sniff_device_ids(CAN_IF, duration=2.0)

    if seen_ids:
        print(f"  Raw CAN: frames received from device IDs {sorted(seen_ids)}")
    else:
        print("  Raw CAN: no REV SPARK frames seen — are motors powered?")
    print()

    print(f"Connecting to motors on {CAN_IF}...\n")
    motors = {}
    for can_id in sorted(MOTOR_NAMES):
        try:
            m = SparkFlex(CAN_IF, can_id)
            motors[can_id] = m
            tag = "[PRESENT]" if can_id in seen_ids else "[NOT ON BUS]"
            print(f"  ID {can_id} ({MOTOR_NAMES[can_id]}): created OK  {tag}")
        except Exception as e:
            print(f"  ID {can_id} ({MOTOR_NAMES[can_id]}): FAILED — {e}")
        time.sleep(0.003)

    print()
    print("Sending heartbeats and reading telemetry for 3 seconds...")
    print(f"{'ID':<4} {'Name':<12} {'Telemetry'}")
    print("-" * 50)

    results = {cid: [] for cid in motors}
    t_end = time.monotonic() + 3.0
    while time.monotonic() < t_end:
        for can_id, m in motors.items():
            try:
                m.Heartbeat()
                if can_id in DRIVE_IDS:
                    val = m.GetVelocity()
                    results[can_id].append(("vel", val))
                else:
                    val = m.GetAbsoluteEncoderPosition()
                    results[can_id].append(("pos_deg", val))
            except Exception as e:
                results[can_id].append(("error", str(e)))
        time.sleep(0.02)

    print()
    print("Results summary:")
    print(f"{'ID':<4} {'Name':<12} {'Bus':<13} {'Status':<12} {'Last value'}")
    print("-" * 68)

    all_ok = True
    for can_id in sorted(MOTOR_NAMES):
        name = MOTOR_NAMES[can_id]
        on_bus = "PRESENT" if can_id in seen_ids else "MISSING"

        if can_id not in motors:
            print(f"{can_id:<4} {name:<12} {on_bus:<13} {'NO OBJECT':<12}")
            all_ok = False
            continue

        samples = results[can_id]
        errors = [s for s in samples if s[0] == "error"]
        good = [s for s in samples if s[0] != "error"]

        if on_bus == "MISSING":
            print(f"{can_id:<4} {name:<12} {on_bus:<13} {'NOT PRESENT':<12} (no frames on bus)")
            all_ok = False
        elif not good:
            last = errors[-1][1] if errors else "n/a"
            print(f"{can_id:<4} {name:<12} {on_bus:<13} {'ALL ERRORS':<12} last={last}")
            all_ok = False
        else:
            label, last_val = good[-1]
            unit = "RPM" if label == "vel" else "deg"
            err_note = f" ({len(errors)} read errors)" if errors else ""
            print(f"{can_id:<4} {name:<12} {on_bus:<13} {'OK':<12} {label}={last_val:.3f} {unit}{err_note}")

    print()
    expected = set(MOTOR_NAMES.keys())
    present = seen_ids & expected
    missing = expected - seen_ids
    extra = seen_ids - expected

    print(f"Physically detected: {len(present)}/{len(expected)} expected motors on bus")
    if missing:
        print(f"  Missing IDs : {sorted(missing)} — check wiring/IDs/power")
    if extra:
        print(f"  Unexpected IDs on bus: {sorted(extra)}")
    if all_ok:
        print("All 6 motors connected and responding.")


if __name__ == "__main__":
    main()
