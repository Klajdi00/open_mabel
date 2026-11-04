#!/usr/bin/env python3
"""MABEL ORCA Hand — teach and repeat.

Teach: disable torque, physically pose the hand, press Enter to capture.
       Positions are normalised to 0.0–1.0 within each joint's calibrated range.
Repeat: smooth ease-in-out interpolation to the saved pose using one
        GroupSyncWrite packet per interpolation step.

All 17 motors are read/written simultaneously via GroupSyncRead / SyncWritePosEx.

Data:  data/gestures/saved_normalized_poses.json
Calib: data/calibration/joint_limits.json  (falls back to data/gestures/quick_grasp_limits.json)
"""

import json
import os
import sys
import time
from pathlib import Path

_FEETECH   = Path(__file__).resolve().parent.parent
_ORCA_ROOT = _FEETECH.parent / "orca_core"
for _p in (str(_FEETECH), str(_ORCA_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from orca_core.hardware.feetech import (
    PortHandler, sms_sts, GroupSyncRead, COMM_SUCCESS,
)
from orca_core.hardware.feetech.sms_sts import SMS_STS_PRESENT_POSITION_L
from utils.hand_config import detect_port, DATA_DIR

if os.name == "nt":
    os.system("")

class C:
    CYAN = "\033[96m"; GREEN = "\033[92m"; WARN = "\033[93m"
    FAIL = "\033[91m"; END   = "\033[0m";  BOLD = "\033[1m"

# ── Paths ─────────────────────────────────────────────────────────────────────
POSES_FILE = DATA_DIR / "gestures" / "saved_normalized_poses.json"
CALIB_FILE = DATA_DIR / "calibration" / "joint_limits.json"
CALIB_FILE_FALLBACK = DATA_DIR / "gestures" / "quick_grasp_limits.json"

BAUDRATE = 1_000_000
TOTAL    = 17

# Playback parameters
SMOOTH_DURATION = 1.2    # seconds for each pose transition
MOVE_SPEED      = 2000   # GOAL_SPEED cap
# torque = GOAL_TIME in ms; MOVE_SPEED capped high so time control dominates


# ── Bus ───────────────────────────────────────────────────────────────────────

class Bus:
    ACC = 30

    def __init__(self, port: str):
        self.ph = PortHandler(port)
        self.ph.baudrate = BAUDRATE
        if not self.ph.openPort():
            raise OSError(f"Cannot open {port}")
        self.pkt = sms_sts(self.ph)

    def close(self):
        try:
            self.pkt.write1ByteTxRx(0xFE, 40, 0)
        except Exception:
            pass
        try:
            self.ph.closePort()
        except Exception:
            pass

    def torque_broadcast(self, on: bool):
        self.pkt.write1ByteTxRx(0xFE, 40, int(on))

    def sync_read(self, ids: list) -> dict:
        """Read raw positions for all ids in one packet."""
        sr = GroupSyncRead(self.pkt, SMS_STS_PRESENT_POSITION_L, 2)
        for mid in ids:
            sr.addParam(mid)
        if sr.txRxPacket() != COMM_SUCCESS:
            return {mid: self._read_one(mid) for mid in ids}
        out = {}
        for mid in ids:
            ok, _ = sr.isAvailable(mid, SMS_STS_PRESENT_POSITION_L, 2)
            if ok:
                raw = sr.getData(mid, SMS_STS_PRESENT_POSITION_L, 2)
                out[mid] = self.pkt.scs_tohost(raw, 15)
            else:
                out[mid] = self._read_one(mid)
        return out

    def _read_one(self, mid: int) -> int | None:
        raw, rc, _ = self.pkt.read2ByteTxRx(mid, SMS_STS_PRESENT_POSITION_L)
        return self.pkt.scs_tohost(raw, 15) if rc == COMM_SUCCESS else None

    def sync_write(self, targets: dict, speed: int = 2000, torque: int = 500):
        """Send all motors in one SyncWritePosEx packet."""
        self.pkt.groupSyncWrite.clearParam()
        for mid, raw in targets.items():
            raw = max(0, min(4095, int(raw)))
            self.pkt.SyncWritePosEx(mid, raw, speed, self.ACC, torque)
        self.pkt.groupSyncWrite.txPacket()
        self.pkt.groupSyncWrite.clearParam()

    def smooth_move(self, targets: dict, duration: float = SMOOTH_DURATION):
        """Send targets in one SyncWritePosEx packet; motor firmware S-curve moves smoothly.

        torque = GOAL_TIME (ms) — motor reaches target in exactly `duration` seconds.
        speed  = GOAL_SPEED cap — high so time-based control dominates.
        """
        goal_time = int(duration * 1000)
        self.sync_write(targets, speed=MOVE_SPEED, torque=goal_time)
        time.sleep(duration)   # wait for motion to complete

    def ping(self, mid: int) -> bool:
        _, rc, _ = self.pkt.ping(mid)
        return rc == COMM_SUCCESS


# ── Calibration ───────────────────────────────────────────────────────────────

def load_calibration() -> dict:
    for path in (CALIB_FILE, CALIB_FILE_FALLBACK):
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            print(f"{C.CYAN}[+] Calibration: {path}{C.END}")
            return data
    sys.exit(f"{C.FAIL}No calibration file found. Run calibrate_limits.py first.{C.END}")


def calib_min_max(entry: dict) -> tuple[int, int]:
    """Support both joint_limits.json (min/max) and quick_grasp_limits (open/fist)."""
    lo = entry.get("min", entry.get("open", 0))
    hi = entry.get("max", entry.get("fist", 4095))
    return int(lo), int(hi)


# ── Pose I/O ──────────────────────────────────────────────────────────────────

def load_poses() -> dict:
    if POSES_FILE.exists():
        try:
            with open(POSES_FILE) as f:
                return json.load(f)
        except Exception as e:
            print(f"{C.FAIL}[!] Load error: {e}{C.END}")
    return {}


def save_poses(poses: dict):
    POSES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(POSES_FILE, "w") as f:
        json.dump(poses, f, indent=4)


# ── Modes ─────────────────────────────────────────────────────────────────────

def teach_mode(bus: Bus, active_ids: list, poses: dict, calib: dict):
    os.system("cls" if os.name == "nt" else "clear")
    print(f"{C.BOLD}──── TEACH MODE ────{C.END}")

    name = input("\nPose name (blank to cancel): ").strip()
    if not name:
        return
    if name in poses:
        if input(f"{C.WARN}'{name}' exists. Overwrite? (y/n): {C.END}").lower() != "y":
            return

    bus.torque_broadcast(False)
    print(f"\n{C.GREEN}Torque OFF — hand is limp.{C.END}")
    input("Pose the hand, then press Enter to capture …")

    # GroupSyncRead — all motors in one packet
    positions = bus.sync_read(active_ids)

    captured, ok = {}, 0
    for mid in active_ids:
        key = str(mid)
        if key not in calib:
            print(f"  {C.WARN}ID {mid} missing from calibration, skipping.{C.END}")
            continue
        raw = positions.get(mid)
        if raw is None:
            print(f"  {C.FAIL}ID {mid} read failed, skipping.{C.END}")
            continue
        lo, hi = calib_min_max(calib[key])
        rng = hi - lo
        norm = max(0.0, min(1.0, (raw - lo) / rng)) if rng else 0.5
        captured[key] = round(norm, 4)
        ok += 1

    if ok == 0:
        print(f"{C.FAIL}No data captured.{C.END}")
        return

    poses[name] = captured
    save_poses(poses)
    print(f"\n{C.GREEN}{C.BOLD}Saved '{name}' ({ok} joints).{C.END}")
    time.sleep(1.5)


def playback_mode(bus: Bus, active_ids: list, poses: dict, calib: dict):
    os.system("cls" if os.name == "nt" else "clear")
    print(f"{C.BOLD}──── PLAYBACK MODE ────{C.END}")

    if not poses:
        print(f"\n{C.WARN}No poses saved yet. Teach the hand first.{C.END}")
        input("\nEnter to return …")
        return

    names = list(poses.keys())
    for i, n in enumerate(names):
        print(f"  [{i+1}] {n}")
    print("  [0] Cancel")

    try:
        choice = int(input("\nSelect pose: "))
        if choice == 0:
            return
        selected = names[choice - 1]
    except (ValueError, IndexError):
        return

    norm_data = poses[selected]

    # Build raw targets from normalised values
    targets = {}
    for mid in active_ids:
        key = str(mid)
        if key not in norm_data or key not in calib:
            continue
        lo, hi = calib_min_max(calib[key])
        raw = int(lo + norm_data[key] * (hi - lo))
        raw = max(min(lo, hi), min(max(lo, hi), raw))
        targets[mid] = raw

    if not targets:
        print(f"{C.FAIL}No valid targets.{C.END}")
        time.sleep(1)
        return

    print(f"\n{C.CYAN}Moving to '{selected}' smoothly …{C.END}")
    bus.torque_broadcast(True)
    time.sleep(0.2)

    # Smooth interpolation — all motors in sync
    bus.smooth_move(targets)

    print(f"\n{C.GREEN}{C.BOLD}Holding '{selected}'.{C.END}")
    input("\nPress Enter to relax …")
    bus.torque_broadcast(False)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.system("cls" if os.name == "nt" else "clear")
    print(f"{C.CYAN}Connecting …{C.END}")
    port = detect_port()
    bus  = Bus(port)

    active_ids = [mid for mid in range(1, TOTAL + 1) if bus.ping(mid)]
    if not active_ids:
        sys.exit(f"{C.FAIL}No motors found. Check power.{C.END}")
    print(f"{C.GREEN}Found {len(active_ids)} motors.{C.END}")

    calib = load_calibration()
    poses = load_poses()

    try:
        while True:
            os.system("cls" if os.name == "nt" else "clear")
            print(f"{C.BOLD}╔══ ORCA POSE STUDIO ══════════════════╗{C.END}")
            print(f"  Active motors : {len(active_ids)}")
            print(f"  Saved poses   : {C.GREEN}{len(poses)}{C.END}")
            print(f"  Transition    : {SMOOTH_DURATION:.1f}s")
            print(f"{C.BOLD}╠══════════════════════════════════════╣{C.END}")
            print("  [1] Teach (record pose)")
            print("  [2] Playback")
            print("  [3] Delete pose")
            print("  [Q] Quit")

            ch = input("\n→ ").strip().lower()
            if ch == "1":
                teach_mode(bus, active_ids, poses, calib)
            elif ch == "2":
                playback_mode(bus, active_ids, poses, calib)
            elif ch == "3":
                names = list(poses.keys())
                for i, n in enumerate(names):
                    print(f"  [{i+1}] {n}")
                print("  [0] Cancel")
                try:
                    d = int(input("Delete: "))
                    if 1 <= d <= len(names):
                        del poses[names[d - 1]]
                        save_poses(poses)
                        print(f"{C.GREEN}Deleted.{C.END}")
                        time.sleep(1)
                except ValueError:
                    pass
            elif ch == "q":
                break
    finally:
        bus.close()
        print(f"\n{C.GREEN}Done.{C.END}")


if __name__ == "__main__":
    main()
