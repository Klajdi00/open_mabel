#!/usr/bin/env python3
"""MABEL ORCA Hand — thermal stress test.

Cycles the hand between configurable poses while monitoring each motor's
temperature.  Stops automatically when any motor exceeds MAX_TEMP.
Logs per-cycle temperature snapshots to a CSV file.

Mirrors orca_core/scripts/stress_test.py, adapted for MABEL.

Usage:
    python stress_test.py                        # flat ↔ fist, indefinitely
    python stress_test.py --cycles 100           # stop after 100 cycles
    python stress_test.py --pose_a flat --pose_b pinch
    python stress_test.py --hold 0.5 --trans 0.8
"""

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

_FEETECH   = Path(__file__).resolve().parent.parent
_ORCA_ROOT = _FEETECH.parent / "orca_core"
for _p in (str(_FEETECH), str(_ORCA_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from orca_core.hardware.feetech_client import FeetechClient, DEFAULT_POS_SCALE, POSITION_DIRECTION
from utils.hand_config import (
    BAUDRATE, TOTAL, ID_JOINT, JOINT_ID, POSES,
    detect_port, load_calibration, pct_to_raw,
    RED, GREEN, YELLOW, CYAN, BOLD, RESET,
)

MAX_TEMP    = 70    # °C — absolute cutoff
WARN_TEMP   = 55    # °C — orange warning
COOL_TEMP   = 40    # °C — back to green
TORQUE      = 600
SPEED       = 800


def _raw_to_rad(raw: float) -> float:
    return raw * DEFAULT_POS_SCALE * POSITION_DIRECTION


def _pose_rads(pose_name: str, calib: dict) -> tuple[list[int], np.ndarray]:
    pose = POSES[pose_name]
    mids, raws = [], []
    for jname, pct in pose.items():
        mid = JOINT_ID.get(jname)
        if mid:
            mids.append(mid)
            raws.append(_raw_to_rad(float(pct_to_raw(mid, pct, calib))))
    return mids, np.array(raws)


def _temp_color(t: float) -> str:
    if t >= MAX_TEMP:  return RED + BOLD
    if t >= WARN_TEMP: return YELLOW
    if t >= COOL_TEMP: return CYAN
    return GREEN


def _print_temps(temps: dict[int, float], cycle: int, elapsed: float) -> None:
    """Render temperature table in-place."""
    W = 60
    print(f"\033[2J\033[H", end="")   # clear screen
    print(f"{BOLD}{'='*W}{RESET}")
    print(f"{BOLD}  MABEL STRESS TEST   cycle {cycle}   elapsed {elapsed:.0f}s{RESET}")
    print(f"{'─'*W}")
    print(f" {'ID':>3}  {'Joint':<13}  {'Temp':>6}  Bar")
    print(f"{'─'*W}")
    for mid in range(1, TOTAL + 1):
        joint = ID_JOINT.get(mid, f"ID_{mid}")
        t     = temps.get(mid, 0.0)
        color = _temp_color(t)
        bar   = "█" * int(t / 5)   # 1 block per 5°C
        print(f" {mid:>3}  {joint:<13}  {color}{t:5.1f}°C{RESET}  {color}{bar}{RESET}")
    print(f"{'─'*W}")
    max_t = max(temps.values(), default=0)
    max_id = max(temps, key=temps.get, default=0)
    color = _temp_color(max_t)
    print(f" {color}Hottest: ID {max_id} ({ID_JOINT.get(max_id,'?')}) = {max_t:.1f}°C{RESET}")
    print(f"{'='*W}")


def main() -> int:
    parser = argparse.ArgumentParser(description="MABEL thermal stress test")
    parser.add_argument("--port",   type=str)
    parser.add_argument("--cycles", type=int, default=0,   help="0 = infinite")
    parser.add_argument("--pose_a", type=str, default="flat")
    parser.add_argument("--pose_b", type=str, default="fist")
    parser.add_argument("--hold",   type=float, default=0.8, help="Hold time per pose (s)")
    parser.add_argument("--trans",  type=float, default=1.0, help="Transition time (s)")
    args = parser.parse_args()

    for p in (args.pose_a, args.pose_b):
        if p not in POSES:
            print(f"{RED}Unknown pose '{p}'. Available: {list(POSES.keys())}{RESET}")
            return 1

    port  = args.port or detect_port()
    calib = load_calibration()
    ids   = list(range(1, TOTAL + 1))

    # Prepare commands
    mids_a, rads_a = _pose_rads(args.pose_a, calib)
    mids_b, rads_b = _pose_rads(args.pose_b, calib)

    client = FeetechClient(ids, port=port, baudrate=BAUDRATE)
    client.connect()

    # CSV log
    (_FEETECH / "data" / "logs").mkdir(parents=True, exist_ok=True)
    csv_path = _FEETECH / "data" / "logs" / f"stress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    csv_file = open(csv_path, "w", newline="")
    writer   = csv.writer(csv_file)
    writer.writerow(["cycle", "elapsed_s", "pose"] + [ID_JOINT.get(i, str(i)) for i in range(1, TOTAL + 1)])

    t_start  = time.monotonic()
    cycle    = 0
    max_seen = 0.0

    try:
        while True:
            cycle += 1
            if args.cycles and cycle > args.cycles:
                break

            for pose_name, mids, rads in ((args.pose_a, mids_a, rads_a), (args.pose_b, mids_b, rads_b)):
                client.write_positions_sync(mids, rads, speed=SPEED, torque=TORQUE)
                time.sleep(args.trans + args.hold)

                temps = {}
                try:
                    t_arr = client.read_temperature()
                    for i, mid in enumerate(ids):
                        temps[mid] = float(t_arr[i])
                except Exception:
                    pass

                elapsed = time.monotonic() - t_start
                _print_temps(temps, cycle, elapsed)

                row = [cycle, f"{elapsed:.1f}", pose_name] + [
                    f"{temps.get(i, 0):.1f}" for i in range(1, TOTAL + 1)
                ]
                writer.writerow(row)
                csv_file.flush()

                max_t = max(temps.values(), default=0)
                max_seen = max(max_seen, max_t)
                if max_t >= MAX_TEMP:
                    print(f"\n{RED}{BOLD}THERMAL CUTOFF — motor overheated ({max_t:.1f}°C ≥ {MAX_TEMP}°C){RESET}")
                    raise StopIteration

    except (KeyboardInterrupt, StopIteration):
        pass
    finally:
        csv_file.close()
        client.disconnect()

    elapsed = time.monotonic() - t_start
    print(f"\n{GREEN}Stress test complete.{RESET}")
    print(f"  Cycles: {cycle}   Elapsed: {elapsed:.0f}s   Peak temp: {max_seen:.1f}°C")
    print(f"  Log: {csv_path}\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print(f"\n{RED}Aborted.{RESET}")
        sys.exit(1)
