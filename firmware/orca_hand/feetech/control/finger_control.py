#!/usr/bin/env python3
"""MABEL ORCA Hand — interactive finger control.

Loads calibrated joint limits from data/calibration/joint_limits.json and
lets you command individual joints, whole fingers, or named poses.

Supports left and right hand (--hand flag or prompted at startup).

Position convention (from calibration):
  min (0%)  = extended / open   (extend sweep end-stop)
  max (100%) = flexed  / closed  (flex sweep end-stop)
  → 'open'  sends PIP/MCP joints to ~5%
  → 'close' sends PIP/MCP joints to ~95%
  ABD joints always go to 50% (neutral) for open/close commands.

Usage:
    python finger_control.py
    python finger_control.py --hand left
    python finger_control.py --hand right --port /dev/ttyUSB0

Commands:
    open  <finger|all>         move flexion joints to open position
    close <finger|all>         move flexion joints to closed position
    move  <joint_name> <0-100> move one joint to % of calibrated range
    id    <motor_id>   <0-100> same, by motor ID
    wrist <-100..100>          wrist (-100=min, 0=centre, +100=max)
    pose  <name>               named pose
    torque <on|off>            enable/disable all torque
    status                     print all motor positions
    watch                      stream live positions (Ctrl+C to stop)
    help                       show this help
    quit / q                   release torque and exit
"""

import argparse
import json
import math
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
from orca_core.hardware.feetech.sms_sts import (
    SMS_STS_PRESENT_POSITION_L,
)
from utils.hand_config import DATA_DIR, detect_port

import serial.tools.list_ports

# ── Constants ─────────────────────────────────────────────────────────────────
BAUDRATE = 1_000_000
TOTAL    = 17
CALIB_PATH = DATA_DIR / "calibration" / "joint_limits.json"

# ── Per-hand motor ID ↔ joint name maps ───────────────────────────────────────
JOINT_ID_LEFT = {
    "middle_pip": 1,  "pinky_mcp":  2,  "pinky_pip":  3,
    "pinky_abd":  4,  "middle_abd": 5,  "index_abd":  6,
    "index_mcp":  7,  "index_pip":  8,  "thumb_mp":   9,
    "thumb_ip":  10,  "thumb_abd": 11,  "thumb_cmc": 12,
    "ring_abd":  13,  "ring_pip":  14,  "ring_mcp":  15,
    "middle_mcp":16,  "wrist":     17,
}

JOINT_ID_RIGHT = {
    "index_pip":  1,  "index_mcp":  2,  "index_abd":  3,
    "middle_abd": 4,  "pinky_abd":  5,  "pinky_pip":  6,
    "pinky_mcp":  7,  "middle_pip": 8,  "middle_mcp": 9,
    "ring_mcp":  10,  "ring_pip":  11,  "ring_abd":  12,
    "thumb_cmc": 13,  "thumb_abd": 14,  "thumb_mp":  15,
    "thumb_ip":  16,  "wrist":     17,
}

# Finger groups: flexion joints first, then abduction
FINGERS = {
    "index":  ["index_pip",  "index_mcp",  "index_abd"],
    "middle": ["middle_pip", "middle_mcp", "middle_abd"],
    "ring":   ["ring_pip",   "ring_mcp",   "ring_abd"],
    "pinky":  ["pinky_pip",  "pinky_mcp",  "pinky_abd"],
    "thumb":  ["thumb_ip",   "thumb_mp",   "thumb_abd", "thumb_cmc"],
}

# Open/close targets in % of calibrated range:
#   min(0%) = extended/open end   max(100%) = flexed/closed end
OPEN_PCT  =  5.0   # near the extend end-stop
CLOSE_PCT = 95.0   # near the flex end-stop
ABD_NEUTRAL = 50.0 # abduction joints stay centred

# Named poses — {joint_name: pct}.  0% = open/extended, 100% = closed/flexed.
POSES: dict[str, dict[str, float]] = {
    "flat": {
        "index_pip": 5,  "index_mcp": 5,  "index_abd": 50,
        "middle_pip": 5, "middle_mcp": 5, "middle_abd": 50,
        "ring_pip": 5,   "ring_mcp": 5,   "ring_abd": 50,
        "pinky_pip": 5,  "pinky_mcp": 5,  "pinky_abd": 50,
        "thumb_ip": 5,   "thumb_mp": 5,   "thumb_abd": 50, "thumb_cmc": 5,
        "wrist": 50,
    },
    "fist": {
        "index_pip": 95, "index_mcp": 90, "index_abd": 50,
        "middle_pip": 95,"middle_mcp": 90,"middle_abd": 50,
        "ring_pip": 95,  "ring_mcp": 90,  "ring_abd": 50,
        "pinky_pip": 95, "pinky_mcp": 90, "pinky_abd": 50,
        "thumb_ip": 80,  "thumb_mp": 80,  "thumb_abd": 20, "thumb_cmc": 80,
        "wrist": 50,
    },
    "pinch": {
        "index_pip": 60, "index_mcp": 60, "index_abd": 50,
        "middle_pip": 5, "middle_mcp": 5, "middle_abd": 50,
        "ring_pip": 5,   "ring_mcp": 5,   "ring_abd": 50,
        "pinky_pip": 5,  "pinky_mcp": 5,  "pinky_abd": 50,
        "thumb_ip": 55,  "thumb_mp": 60,  "thumb_abd": 65, "thumb_cmc": 55,
        "wrist": 50,
    },
    "point": {
        "index_pip": 5,  "index_mcp": 5,  "index_abd": 50,
        "middle_pip": 95,"middle_mcp": 90,"middle_abd": 50,
        "ring_pip": 95,  "ring_mcp": 90,  "ring_abd": 50,
        "pinky_pip": 95, "pinky_mcp": 90, "pinky_abd": 50,
        "thumb_ip": 65,  "thumb_mp": 65,  "thumb_abd": 30, "thumb_cmc": 65,
        "wrist": 50,
    },
    "peace": {
        "index_pip": 5,  "index_mcp": 5,  "index_abd": 50,
        "middle_pip": 5, "middle_mcp": 5, "middle_abd": 50,
        "ring_pip": 95,  "ring_mcp": 90,  "ring_abd": 50,
        "pinky_pip": 95, "pinky_mcp": 90, "pinky_abd": 50,
        "thumb_ip": 65,  "thumb_mp": 65,  "thumb_abd": 30, "thumb_cmc": 65,
        "wrist": 50,
    },
    "ok": {
        "index_pip": 65, "index_mcp": 65, "index_abd": 50,
        "middle_pip": 5, "middle_mcp": 5, "middle_abd": 50,
        "ring_pip": 5,   "ring_mcp": 5,   "ring_abd": 50,
        "pinky_pip": 5,  "pinky_mcp": 5,  "pinky_abd": 50,
        "thumb_ip": 60,  "thumb_mp": 65,  "thumb_abd": 70, "thumb_cmc": 50,
        "wrist": 50,
    },
}

# ANSI
RED, GREEN, YELLOW, BLUE, CYAN, BOLD, RESET = (
    "\033[91m", "\033[92m", "\033[93m", "\033[94m",
    "\033[96m", "\033[1m",  "\033[0m",
)

# ── Motion parameters ────────────────────────────────────────────────────────
# Feetech WritePosEx register mapping (via SyncWritePosEx):
#   speed  → GOAL_SPEED  (reg 46-47, units of 0.732 RPM)
#   torque → GOAL_TIME   (reg 44-45, milliseconds)
#              GOAL_TIME > 0: motor reaches target in exactly that many ms
#              GOAL_TIME = 0: motor uses GOAL_SPEED for velocity control
#
# Use torque=goal_ms and speed=2000 for all commanded motion.
# The motor firmware S-curve handles smooth acceleration/deceleration.
SMOOTH_DURATION = 1.2    # seconds for pose transitions (torque = 1200 ms)
MOVE_SPEED      = 2000   # GOAL_SPEED cap — high so time-based control dominates
SNAP_MS         = 400    # GOAL_TIME for single-joint moves (ms)


# ── Calibration ───────────────────────────────────────────────────────────────

def load_calibration() -> dict:
    """Load joint_limits.json → {motor_id: {min, max, center}}.

    Falls back to 0–4095 for every motor if the file is missing.
    """
    if CALIB_PATH.exists():
        with open(CALIB_PATH) as f:
            raw = json.load(f)
        result = {int(k): {"min": int(v["min"]), "max": int(v["max"]),
                            "center": int(v["center"])}
                  for k, v in raw.items()}
        print(f"{GREEN}  Calibration loaded: {CALIB_PATH}{RESET}")
        return result
    print(f"{YELLOW}  No calibration file found — using full range 0-4095.")
    print(f"  Run calibration/calibrate_limits.py for accurate control.{RESET}")
    return {mid: {"min": 0, "max": 4095, "center": 2047} for mid in range(1, TOTAL + 1)}


def pct_to_raw(mid: int, pct: float, calib: dict) -> int:
    """0% → min (open/extended),  100% → max (closed/flexed)."""
    c  = calib.get(mid, {"min": 0, "max": 4095})
    lo, hi = c["min"], c["max"]
    return int(lo + max(0.0, min(100.0, pct)) / 100.0 * (hi - lo))


def raw_to_pct(mid: int, raw: int, calib: dict) -> float:
    c  = calib.get(mid, {"min": 0, "max": 4095})
    lo, hi = c["min"], c["max"]
    if hi == lo:
        return 50.0
    return max(0.0, min(100.0, (raw - lo) / (hi - lo) * 100.0))


# ── Bus — raw sync write/read, no radian conversion ───────────────────────────

class Bus:
    """Thin serial bus that operates in raw encoder units.

    Using raw positions directly avoids the FeetechClient POSITION_DIRECTION=-1
    inversion which made the old finger_control.py send motors to the wrong end.
    """

    HOLD_ACC = 30

    def __init__(self, port: str):
        self.port = port
        self.ph   = PortHandler(port)
        self.ph.baudrate = BAUDRATE
        if not self.ph.openPort():
            raise OSError(f"Cannot open {port}")
        self.pkt = sms_sts(self.ph)

    def close(self) -> None:
        try:
            self.pkt.write1ByteTxRx(254, 40, 0)   # broadcast torque off
        except Exception:
            pass
        try:
            self.ph.closePort()
        except Exception:
            pass

    # — Write ─────────────────────────────────────────────────────────────────

    def sync_write(self, targets: dict, speed: int = 2000,
                   torque: int = 500) -> None:
        """Send {motor_id: raw_pos} in one SyncWritePosEx packet.

        speed  = GOAL_SPEED (0.732 RPM/unit — max velocity cap)
        torque = GOAL_TIME  (ms; motor reaches target in this many ms)
                 0 = pure speed control; >0 = time-based motion
        """
        self.pkt.groupSyncWrite.clearParam()
        for mid, raw in targets.items():
            raw = max(0, min(4095, int(raw)))
            self.pkt.SyncWritePosEx(mid, raw, speed, self.HOLD_ACC, torque)
        self.pkt.groupSyncWrite.txPacket()
        self.pkt.groupSyncWrite.clearParam()

    def torque_all(self, ids: list, enabled: bool) -> None:
        for mid in ids:
            self.pkt.write1ByteTxRx(mid, 40, int(enabled))

    # — Read ──────────────────────────────────────────────────────────────────

    def sync_read_pos(self, ids: list) -> dict:
        """Read raw positions for all ids in one GroupSyncRead packet."""
        sr = GroupSyncRead(self.pkt, SMS_STS_PRESENT_POSITION_L, 2)
        for mid in ids:
            sr.addParam(mid)
        if sr.txRxPacket() != COMM_SUCCESS:
            return {mid: self._read_single(mid) for mid in ids}
        result = {}
        for mid in ids:
            avail, _ = sr.isAvailable(mid, SMS_STS_PRESENT_POSITION_L, 2)
            if avail:
                raw = sr.getData(mid, SMS_STS_PRESENT_POSITION_L, 2)
                result[mid] = self.pkt.scs_tohost(raw, 15)
            else:
                result[mid] = self._read_single(mid)
        return result

    def _read_single(self, mid: int) -> int | None:
        raw, rc, _ = self.pkt.read2ByteTxRx(mid, SMS_STS_PRESENT_POSITION_L)
        return self.pkt.scs_tohost(raw, 15) if rc == COMM_SUCCESS else None


# ── Hand controller ───────────────────────────────────────────────────────────

class HandController:
    def __init__(self, bus: Bus, calib: dict, joint_id: dict):
        self.bus      = bus
        self.calib    = calib
        self.joint_id = joint_id
        self.id_joint = {v: k for k, v in joint_id.items()}
        self.ids      = list(range(1, TOTAL + 1))
        self.bus.torque_all(self.ids, True)
        print(f"{GREEN}  Connected — torque ON.{RESET}")

    # — Core ──────────────────────────────────────────────────────────────────

    def send_by_id(self, mid: int, pct: float) -> None:
        raw = pct_to_raw(mid, pct, self.calib)
        self.bus.sync_write({mid: raw}, speed=MOVE_SPEED, torque=SNAP_MS)

    def send_smooth(self, commands: dict, duration: float | None = None) -> None:
        """Move joints to targets.  Motor firmware S-curve handles smooth motion.

        torque = GOAL_TIME in ms — motor reaches target in exactly that many ms.
        speed  = GOAL_SPEED cap — set high so time control dominates.
        One SyncWrite packet; no software interpolation needed.
        """
        dur       = duration if duration is not None else SMOOTH_DURATION
        goal_time = int(dur * 1000)   # seconds → milliseconds

        targets: dict[int, int] = {}
        for jname, pct in commands.items():
            mid = self.joint_id.get(jname)
            if mid is None:
                print(f"{YELLOW}  Unknown joint '{jname}'{RESET}")
                continue
            targets[mid] = pct_to_raw(mid, pct, self.calib)

        if targets:
            self.bus.sync_write(targets, speed=MOVE_SPEED, torque=goal_time)

    # — Open / Close ──────────────────────────────────────────────────────────

    def open_finger(self, finger: str) -> None:
        """Extend finger smoothly: flexion joints → open end, ABD → neutral."""
        joints = FINGERS.get(finger)
        if not joints:
            print(f"{YELLOW}  Unknown finger '{finger}'{RESET}")
            return
        cmds = {j: (ABD_NEUTRAL if "abd" in j else OPEN_PCT) for j in joints}
        self.send_smooth(cmds)

    def close_finger(self, finger: str) -> None:
        """Flex finger smoothly: flexion joints → closed end, ABD → neutral."""
        joints = FINGERS.get(finger)
        if not joints:
            print(f"{YELLOW}  Unknown finger '{finger}'{RESET}")
            return
        cmds = {j: (ABD_NEUTRAL if "abd" in j else CLOSE_PCT) for j in joints}
        self.send_smooth(cmds)

    def open_all(self) -> None:
        cmds = {}
        for joints in FINGERS.values():
            for j in joints:
                cmds[j] = ABD_NEUTRAL if "abd" in j else OPEN_PCT
        cmds["wrist"] = 50
        self.send_smooth(cmds)

    def close_all(self) -> None:
        cmds = {}
        for joints in FINGERS.values():
            for j in joints:
                cmds[j] = ABD_NEUTRAL if "abd" in j else CLOSE_PCT
        self.send_smooth(cmds)

    # — Poses ─────────────────────────────────────────────────────────────────

    def pose(self, name: str) -> None:
        p = POSES.get(name)
        if p is None:
            print(f"{YELLOW}  Unknown pose '{name}'.  Available: {list(POSES)}{RESET}")
            return
        self.send_smooth(p)
        print(f"  → {BOLD}{name}{RESET}")

    def move_joint(self, jname: str, pct: float) -> None:
        mid = self.joint_id.get(jname)
        if mid is None:
            print(f"{YELLOW}  Unknown joint '{jname}'{RESET}")
            return
        raw = pct_to_raw(mid, pct, self.calib)
        self.bus.sync_write({mid: raw}, speed=MOVE_SPEED, torque=SNAP_MS)
        print(f"  → {jname} (ID {mid}): {pct:.1f}%  raw={raw}")

    def move_wrist(self, signed_pct: float) -> None:
        """signed_pct: -100 = min, 0 = centre, +100 = max."""
        pct = (signed_pct + 100.0) / 2.0
        mid = self.joint_id.get("wrist")
        if mid:
            raw = pct_to_raw(mid, pct, self.calib)
            self.bus.sync_write({mid: raw}, speed=MOVE_SPEED, torque=SNAP_MS)
        print(f"  → wrist {signed_pct:+.0f}  →  {pct:.1f}%")

    # — Status ────────────────────────────────────────────────────────────────

    def status(self) -> None:
        positions = self.bus.sync_read_pos(self.ids)
        print(f"\n{'─'*60}")
        print(f" {'ID':>3}  {'Joint':<13}  {'Raw':>5}  {'Pct':>6}  "
              f"{'Min':>5}  {'Max':>5}  {'Open%':>6}  {'Close%':>7}")
        print(f"{'─'*60}")
        for mid in range(1, TOTAL + 1):
            jname  = self.id_joint.get(mid, "?")
            raw    = positions.get(mid)
            c      = self.calib.get(mid, {"min": 0, "max": 4095, "center": 2047})
            pct_s  = f"{raw_to_pct(mid, raw, self.calib):5.1f}%" if raw is not None else " ERR%"
            raw_s  = str(raw) if raw is not None else "ERR"
            open_r = pct_to_raw(mid, OPEN_PCT,  self.calib)
            cls_r  = pct_to_raw(mid, CLOSE_PCT, self.calib)
            print(f" {mid:>3}  {jname:<13}  {raw_s:>5}  {pct_s:>6}  "
                  f"{c['min']:>5}  {c['max']:>5}  {open_r:>6}  {cls_r:>7}")
        print(f"{'─'*60}\n")

    def watch(self) -> None:
        print(f"  Streaming (Ctrl+C to stop) …\n")
        try:
            while True:
                pos = self.bus.sync_read_pos(self.ids)
                parts = [
                    f"{self.id_joint.get(i, i)[:6]}:{raw_to_pct(i, pos.get(i, 0), self.calib):4.0f}%"
                    for i in range(1, TOTAL + 1)
                ]
                sys.stdout.write("\r  " + "  ".join(parts) + "   ")
                sys.stdout.flush()
                time.sleep(0.1)
        except KeyboardInterrupt:
            print(f"\n  Stopped.\n")

    def torque(self, enabled: bool) -> None:
        self.bus.torque_all(self.ids, enabled)
        state = f"{GREEN}ON{RESET}" if enabled else f"{YELLOW}OFF (limp){RESET}"
        print(f"  → Torque: {state}")


# ── Help text ─────────────────────────────────────────────────────────────────

def help_text(hand: str, joint_id: dict) -> str:
    id_joint = {v: k for k, v in joint_id.items()}
    id_lines = "  " + "  ".join(f"{mid}={id_joint[mid]}" for mid in sorted(id_joint))
    return f"""
{BOLD}MABEL {hand.upper()} Hand — Finger Control{RESET}

  {CYAN}open  <finger|all>{RESET}         flex joints → open/extended position ({OPEN_PCT:.0f}% of range)
  {CYAN}close <finger|all>{RESET}         flex joints → closed/flexed position ({CLOSE_PCT:.0f}% of range)
  {CYAN}move  <joint> <0-100>{RESET}      move joint: 0%=open end, 100%=closed end
  {CYAN}id    <id> <0-100>{RESET}         same, by motor ID
  {CYAN}wrist <-100..100>{RESET}          wrist (-100=min, 0=centre, +100=max)
  {CYAN}pose  <name>{RESET}               {list(POSES.keys())}
  {CYAN}torque <on|off>{RESET}
  {CYAN}status{RESET}                     table of all joint positions
  {CYAN}watch{RESET}                      live stream (Ctrl+C to stop)
  {CYAN}quit / q{RESET}

{BOLD}Finger names:{RESET}  index  middle  ring  pinky  thumb  all
{BOLD}Joint names:{RESET}
  index_pip  index_mcp  index_abd  |  middle_pip  middle_mcp  middle_abd
  ring_pip   ring_mcp   ring_abd   |  pinky_pip   pinky_mcp   pinky_abd
  thumb_ip   thumb_mp   thumb_abd  thumb_cmc  |  wrist

{BOLD}Motor IDs ({hand} hand):{RESET}
{id_lines}
"""


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="MABEL finger control")
    parser.add_argument("--port", type=str)
    parser.add_argument("--hand", choices=["left", "right"],
                        help="Which hand (prompted if omitted)")
    args = parser.parse_args()

    # Hand selection
    hand = args.hand
    if not hand:
        print(f"\n{BOLD}Which hand?{RESET}")
        print("  [L] Left    [R] Right")
        while True:
            c = input("  Enter L or R: ").strip().lower()
            if c in ("l", "left"):  hand = "left";  break
            if c in ("r", "right"): hand = "right"; break

    joint_id = JOINT_ID_LEFT if hand == "left" else JOINT_ID_RIGHT

    port = args.port or detect_port()
    if not os.path.exists(port):
        print(f"{RED}Port '{port}' not found.{RESET}")
        return 1

    print(f"\n{BOLD}{'='*55}{RESET}")
    print(f"{BOLD}  MABEL {hand.upper()} HAND — FINGER CONTROL{RESET}")
    print(f"{BOLD}{'='*55}{RESET}")
    print(f"{CYAN}  Port: {port}{RESET}\n")

    calib = load_calibration()
    bus   = Bus(port)
    hand_ctrl = HandController(bus, calib, joint_id)

    print(f"\n  Type {BOLD}help{RESET} for commands, {BOLD}quit{RESET} to exit.\n")

    try:
        while True:
            try:
                line = input(f"{BOLD}hand>{RESET} ").strip()
            except EOFError:
                break
            if not line:
                continue

            tokens = line.lower().split()
            cmd    = tokens[0]

            if cmd in ("open", "o"):
                target = tokens[1] if len(tokens) > 1 else "all"
                if target == "all":
                    hand_ctrl.open_all()
                    print("  → Open all")
                elif target in FINGERS:
                    hand_ctrl.open_finger(target)
                    print(f"  → Open {target}")
                else:
                    print(f"{YELLOW}  Unknown: '{target}'.  Use: {list(FINGERS)+['all']}{RESET}")

            elif cmd in ("close", "c"):
                target = tokens[1] if len(tokens) > 1 else "all"
                if target == "all":
                    hand_ctrl.close_all()
                    print("  → Close all")
                elif target in FINGERS:
                    hand_ctrl.close_finger(target)
                    print(f"  → Close {target}")
                else:
                    print(f"{YELLOW}  Unknown: '{target}'.{RESET}")

            elif cmd in ("move", "m"):
                if len(tokens) < 3:
                    print(f"{YELLOW}  Usage: move <joint> <0-100>{RESET}")
                    continue
                try:
                    pct = float(tokens[2])
                except ValueError:
                    print(f"{YELLOW}  % must be a number.{RESET}")
                    continue
                hand_ctrl.move_joint(tokens[1], pct)

            elif cmd == "id":
                if len(tokens) < 3:
                    print(f"{YELLOW}  Usage: id <1-17> <0-100>{RESET}")
                    continue
                try:
                    mid = int(tokens[1])
                    pct = float(tokens[2])
                except ValueError:
                    print(f"{YELLOW}  Usage: id <motor_id> <pct>{RESET}")
                    continue
                if mid not in range(1, TOTAL + 1):
                    print(f"{YELLOW}  ID out of range (1-17){RESET}")
                    continue
                hand_ctrl.send_by_id(mid, pct)
                print(f"  → ID {mid} ({hand_ctrl.id_joint.get(mid,'?')}): "
                      f"{pct:.1f}%  raw={pct_to_raw(mid, pct, calib)}")

            elif cmd in ("wrist", "w"):
                if len(tokens) < 2:
                    print(f"{YELLOW}  Usage: wrist <-100 to 100>{RESET}")
                    continue
                try:
                    hand_ctrl.move_wrist(float(tokens[1]))
                except ValueError:
                    print(f"{YELLOW}  Value must be a number.{RESET}")

            elif cmd in ("pose", "p"):
                name = tokens[1] if len(tokens) > 1 else ""
                if not name:
                    print(f"{YELLOW}  Usage: pose <name>  {list(POSES)}{RESET}")
                    continue
                hand_ctrl.pose(name)

            elif cmd == "speed":
                # Adjust transition duration: 'speed 0.5' = faster, 'speed 2' = slower
                if len(tokens) < 2:
                    print(f"  Transition duration: {SMOOTH_DURATION:.1f}s  "
                          f"(speed <seconds>  e.g. 'speed 0.8')")
                    continue
                try:
                    new_dur = float(tokens[1])
                    SMOOTH_DURATION = new_dur      # module-level rebind
                    print(f"  → Transition duration: {SMOOTH_DURATION:.1f}s")
                except ValueError:
                    print(f"{YELLOW}  Usage: speed <seconds>{RESET}")

            elif cmd in ("torque", "t"):
                if len(tokens) < 2 or tokens[1] not in ("on", "off"):
                    print(f"{YELLOW}  Usage: torque <on|off>{RESET}")
                    continue
                hand_ctrl.torque(tokens[1] == "on")

            elif cmd in ("status", "s"):
                hand_ctrl.status()

            elif cmd == "watch":
                hand_ctrl.watch()

            elif cmd in ("help", "h", "?"):
                print(help_text(hand, joint_id))

            elif cmd in ("quit", "q", "exit"):
                break

            else:
                print(f"{YELLOW}  Unknown command '{cmd}'. Type 'help'.{RESET}")

    except KeyboardInterrupt:
        print()

    print(f"\n  Releasing torque …")
    bus.close()
    print(f"  {GREEN}Done.{RESET}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
