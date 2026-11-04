#!/usr/bin/env python3
"""MABEL ORCA Hand — timestamped animation record & playback.

Record: torque off, move the hand by hand.  Captures frames whenever any
        motor moves more than MOVEMENT_THRESHOLD raw units.
        All 17 motors are read in ONE GroupSyncRead packet per frame.

Playback: replays timestamps with SyncWritePosEx — all motors commanded in
          one packet per frame.  Crawl to start uses smooth interpolation.

Data:  data/animations/saved_animations.json
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

POSES_FILE = DATA_DIR / "animations" / "saved_animations.json"
BAUDRATE   = 1_000_000
TOTAL      = 17

# Recording
MOVEMENT_THRESHOLD = 5
RECORDING_FPS      = 30

# Playback
CRAWL_DURATION  = 2.5    # seconds for crawl-to-start
ANIM_TORQUE     = 100    # GOAL_TIME per frame (ms) — fast following
ANIM_SPEED      = 2000   # fast enough to follow recorded motion
ANIM_ACC        = 20


class C:
    GREEN = "\033[92m"; WARN = "\033[93m"; FAIL = "\033[91m"
    CYAN  = "\033[96m"; BOLD = "\033[1m";  END  = "\033[0m"


# ── Bus ───────────────────────────────────────────────────────────────────────

class Bus:
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
        """Read all motor positions in ONE GroupSyncRead packet."""
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

    def sync_write(self, targets: dict, speed: int = ANIM_SPEED,
                   torque: int = ANIM_TORQUE, acc: int = ANIM_ACC):
        # torque = GOAL_TIME (ms); 100ms per frame ensures tight following
        """Send all motors in ONE SyncWritePosEx packet."""
        self.pkt.groupSyncWrite.clearParam()
        for mid, raw in targets.items():
            raw = max(0, min(4095, int(raw)))
            self.pkt.SyncWritePosEx(mid, raw, speed, acc, torque)
        self.pkt.groupSyncWrite.txPacket()
        self.pkt.groupSyncWrite.clearParam()

    def smooth_move(self, targets: dict, duration: float = CRAWL_DURATION):
        """Send targets in one SyncWritePosEx packet.  Motor S-curve handles smooth motion.

        torque = GOAL_TIME (ms) — motor reaches target in `duration` seconds.
        """
        goal_time = int(duration * 1000)
        self.sync_write(targets, speed=2000, torque=goal_time, acc=30)
        time.sleep(duration)

    def ping(self, mid: int) -> bool:
        _, rc, _ = self.pkt.ping(mid)
        return rc == COMM_SUCCESS


# ── Data I/O ──────────────────────────────────────────────────────────────────

def load_animations() -> dict:
    if POSES_FILE.exists():
        try:
            with open(POSES_FILE) as f:
                return json.load(f)
        except Exception as e:
            print(f"{C.FAIL}Load error: {e}{C.END}")
    return {}


def save_animations(data: dict):
    POSES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(POSES_FILE, "w") as f:
        json.dump(data, f, indent=4)


# ── Record mode ───────────────────────────────────────────────────────────────

def record_mode(bus: Bus, active_ids: list, animations: dict):
    os.system("cls" if os.name == "nt" else "clear")
    print(f"{C.BOLD}──── RECORD MODE ────{C.END}")

    name = input("\nAnimation name (blank to cancel): ").strip()
    if not name:
        return
    if name in animations:
        if input(f"{C.WARN}'{name}' exists. Overwrite? (y/n): {C.END}").lower() != "y":
            return

    bus.torque_broadcast(False)
    print(f"\n{C.GREEN}Torque OFF — move the hand freely.{C.END}")
    input("Press Enter to START recording (Ctrl+C to stop) …\n")

    frames: list = []
    last_pos: dict = {}
    dt = 1.0 / RECORDING_FPS
    t0 = time.perf_counter()

    print(f"  Recording … (threshold >{MOVEMENT_THRESHOLD} raw units)  Ctrl+C to stop.")
    try:
        while True:
            loop_t = time.perf_counter()
            elapsed = loop_t - t0

            # ONE sync read for all motors
            cur = bus.sync_read(active_ids)
            moved = any(
                mid not in last_pos or abs((cur.get(mid) or 0) - last_pos[mid]) > MOVEMENT_THRESHOLD
                for mid in active_ids if cur.get(mid) is not None
            )

            if moved and any(cur.get(mid) is not None for mid in active_ids):
                frame = {str(mid): cur[mid] for mid in active_ids if cur.get(mid) is not None}
                frames.append({"time": round(elapsed, 3), "positions": frame})
                last_pos = {mid: cur[mid] for mid in active_ids if cur.get(mid) is not None}
                sys.stdout.write(f"\r  Frame {len(frames):04d}  t={elapsed:.2f}s   ")
                sys.stdout.flush()

            wait = dt - (time.perf_counter() - loop_t)
            if wait > 0:
                time.sleep(wait)

    except KeyboardInterrupt:
        print(f"\n\n{C.GREEN}Recording stopped.{C.END}")

    if frames:
        animations[name] = frames
        save_animations(animations)
        print(f"{C.BOLD}Saved '{name}' — {len(frames)} frames, "
              f"{frames[-1]['time']:.2f}s total.{C.END}")
    else:
        print(f"{C.WARN}No movement detected. Discarding.{C.END}")

    time.sleep(1.5)


# ── Playback mode ─────────────────────────────────────────────────────────────

def playback_mode(bus: Bus, active_ids: list, animations: dict):
    os.system("cls" if os.name == "nt" else "clear")
    print(f"{C.BOLD}──── PLAYBACK MODE ────{C.END}")

    if not animations:
        print(f"\n{C.WARN}No animations saved yet.{C.END}")
        input("\nEnter to return …")
        return

    names = list(animations.keys())
    for i, n in enumerate(names):
        dur = animations[n][-1]["time"] if animations[n] else 0
        print(f"  [{i+1}] {n}  ({len(animations[n])} frames, {dur:.1f}s)")
    print("  [0] Cancel")

    try:
        choice = int(input("\nSelect: "))
        if choice == 0:
            return
        selected = names[choice - 1]
        frames   = animations[selected]
    except (ValueError, IndexError):
        return

    if not frames:
        return

    # Smooth crawl to first frame
    first = {int(k): v for k, v in frames[0]["positions"].items()
             if int(k) in active_ids}
    print(f"\n{C.CYAN}Crawling to start …{C.END}")
    bus.torque_broadcast(True)
    time.sleep(0.2)
    bus.smooth_move(first, duration=CRAWL_DURATION)

    print(f"{C.CYAN}Playing '{selected}' ({len(frames)} frames) …  Ctrl+C to abort{C.END}")
    try:
        t0 = time.perf_counter()
        for idx, frame in enumerate(frames):
            target_t  = frame["time"]
            positions = frame["positions"]

            # Spin-wait for accurate timestamp replay
            while (time.perf_counter() - t0) < target_t:
                pass

            targets = {int(k): v for k, v in positions.items()
                       if int(k) in active_ids}

            # ONE SyncWrite packet for all motors
            bus.sync_write(targets, speed=ANIM_SPEED, torque=ANIM_TORQUE, acc=ANIM_ACC)

            sys.stdout.write(
                f"\r  Frame {idx+1:04d}/{len(frames)}  t={target_t:.2f}s   "
            )
            sys.stdout.flush()

        print(f"\n\n{C.GREEN}Animation complete.{C.END}")

    except KeyboardInterrupt:
        print(f"\n\n{C.WARN}Playback aborted.{C.END}")

    input("\nEnter to relax …")
    bus.torque_broadcast(False)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.system("cls" if os.name == "nt" else "clear")
    port       = detect_port()
    bus        = Bus(port)
    active_ids = [mid for mid in range(1, TOTAL + 1) if bus.ping(mid)]

    if not active_ids:
        sys.exit(f"{C.FAIL}No motors found.{C.END}")
    print(f"{C.GREEN}Found {len(active_ids)} motors.{C.END}")
    animations = load_animations()

    try:
        while True:
            os.system("cls" if os.name == "nt" else "clear")
            print(f"{C.BOLD}╔══ ORCA ANIMATION STUDIO ═════════════╗{C.END}")
            print(f"  Active motors  : {len(active_ids)}")
            print(f"  Saved animations: {C.GREEN}{len(animations)}{C.END}")
            print(f"{C.BOLD}╠══════════════════════════════════════╣{C.END}")
            print("  [1] Record new animation")
            print("  [2] Play animation")
            print("  [3] Delete animation")
            print("  [Q] Quit")

            ch = input("\n→ ").strip().lower()
            if ch == "1":
                record_mode(bus, active_ids, animations)
            elif ch == "2":
                playback_mode(bus, active_ids, animations)
            elif ch == "3":
                names = list(animations.keys())
                for i, n in enumerate(names):
                    print(f"  [{i+1}] {n}")
                print("  [0] Cancel")
                try:
                    d = int(input("Delete: "))
                    if 1 <= d <= len(names):
                        del animations[names[d - 1]]
                        save_animations(animations)
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
