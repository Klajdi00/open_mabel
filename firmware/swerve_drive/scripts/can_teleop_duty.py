#!/usr/bin/env python3
"""
Keyboard CAN motor teleop — velocity drive, position steer.

Mirrors the init sequence from robot/base_motor.py exactly:
  - Drive motors: SetIdleMode(kCoast) + SetCtrlType(kVelocity) -> SetVelocity
  - Steer motors: SetIdleMode(kCoast) only                     -> SetPosition
  - No ResetFaults()

Motor IDs:
  1 RF drive  |  2 RF steer
  3 LF drive  |  4 LF steer
  5 B  drive  |  6 B  steer

Controls:
  up/dn    Drive RPM +/- 10  |  Steer position +/- 10 deg
  SPACE    Safe stop
  ENTER    Select motor IDs  (blank = all)
  Q        Quit

Run:
  python extra/can_teleop_duty.py
"""

import os
import select
import sys
import termios
import time
import tty
from typing import Optional

# ── safety / tuning ───────────────────────────────────────────────────────────
MAX_RPM    = 200.0
STEP_RPM   = 10.0
STEP_STEER = 10.0 / 360.0  # ~10 deg per key press
LOOP_HZ    = 50

CAN_IF = "can0"

MOTOR_NAMES = {
    1: "RF drive", 2: "RF steer",
    3: "LF drive", 4: "LF steer",
    5: "B  drive", 6: "B  steer",
}
DRIVE_IDS = {1, 3, 5}
STEER_IDS = {2, 4, 6}
ALL_IDS   = set(MOTOR_NAMES)

# ── ANSI ──────────────────────────────────────────────────────────────────────
_CLR  = "\033[2J\033[H"
_HIDE = "\033[?25l"
_SHOW = "\033[?25h"
_RST  = "\033[0m"
_BLD  = "\033[1m"
_DIM  = "\033[2m"
_GRN  = "\033[32m"
_RED  = "\033[31m"
_YLW  = "\033[33m"
_CYN  = "\033[36m"

def b(s):  return f"{_BLD}{s}{_RST}"
def g(s):  return f"{_GRN}{s}{_RST}"
def r(s):  return f"{_RED}{s}{_RST}"
def y(s):  return f"{_YLW}{s}{_RST}"
def c(s):  return f"{_CYN}{s}{_RST}"
def d(s):  return f"{_DIM}{s}{_RST}"

# ── raw terminal ──────────────────────────────────────────────────────────────
_saved_tc = None

def _go_raw():
    global _saved_tc
    fd = sys.stdin.fileno()
    _saved_tc = termios.tcgetattr(fd)
    tty.setraw(fd)

def _go_cooked():
    if _saved_tc is not None:
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, _saved_tc)

def _read_key(timeout: float) -> Optional[str]:
    r_, _, _ = select.select([sys.stdin], [], [], timeout)
    if not r_:
        return None
    ch = os.read(sys.stdin.fileno(), 1)
    if ch == b"\x1b":
        r2, _, _ = select.select([sys.stdin], [], [], 0.02)
        if not r2:
            return "ESC"
        ch2 = os.read(sys.stdin.fileno(), 1)
        if ch2 == b"[":
            r3, _, _ = select.select([sys.stdin], [], [], 0.02)
            if r3:
                ch3 = os.read(sys.stdin.fileno(), 1)
                return {b"A": "UP", b"B": "DOWN", b"C": "RIGHT", b"D": "LEFT"}.get(ch3)
        return "ESC"
    if ch in (b"\r", b"\n"):
        return "ENTER"
    if ch == b" ":
        return "SPACE"
    if ch in (b"q", b"Q", b"\x03"):
        return "QUIT"
    try:
        return ch.decode("ascii")
    except Exception:
        return None


def _sniff(ifname: str, duration: float) -> set:
    import socket, struct
    FMT = "=IB3x8s"
    SZ  = struct.calcsize(FMT)
    sock = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
    sock.bind((ifname,))
    sock.settimeout(0.05)
    seen: set = set()
    t_end = time.monotonic() + duration
    while time.monotonic() < t_end:
        try:
            data = sock.recv(SZ)
            if len(data) < SZ:
                continue
            can_id, _, _ = struct.unpack(FMT, data)
            if not (can_id & 0x80000000):
                continue
            arb = can_id & 0x1FFFFFFF
            if (arb >> 16) == 0x0205:
                dev = arb & 0x3F
                if 1 <= dev <= 62:
                    seen.add(dev)
        except TimeoutError:
            pass
    sock.close()
    return seen


def _prompt_ids() -> set:
    _go_cooked()
    sys.stdout.write(_SHOW)
    sys.stdout.write(
        f"\n  {c('Motor IDs to target')} "
        f"{d('(e.g. 1 5  or  1,3,5 -- blank = all):  ')}"
    )
    sys.stdout.flush()
    try:
        line = input().strip()
    except (EOFError, KeyboardInterrupt):
        line = ""
    sys.stdout.write(_HIDE)
    _go_raw()
    if not line:
        return set(ALL_IDS)
    ids = set()
    for tok in line.replace(",", " ").split():
        try:
            i = int(tok)
            if i in ALL_IDS:
                ids.add(i)
        except ValueError:
            pass
    return ids if ids else set(ALL_IDS)


def _bar(val: float, lo: float, hi: float, width: int = 12) -> str:
    frac = (val - lo) / (hi - lo) if hi > lo else 0.5
    frac = max(0.0, min(1.0, frac))
    pos  = int(round(frac * (width - 1)))
    zero = int(round(max(0.0, min(1.0, (0 - lo) / (hi - lo))) * (width - 1)))
    chars = [" "] * width
    chars[zero] = "|"
    if pos != zero:
        for i in range(min(pos, zero) + 1, max(pos, zero)):
            chars[i] = "-"
        chars[pos] = "<" if val < 0 else ">"
    return "[" + "".join(chars) + "]"


def _draw(seen_ids, selected, drive_vel, steer_frac, telemetry, notice):
    W   = 74
    SEP = "-" * W
    tgt = "ALL" if selected == ALL_IDS else f"IDs {sorted(selected)}"

    out = [_CLR]
    out.append(
        f"  {b('CAN Motor Teleop')}  |  Targeting: {b(tgt)}  "
        + d(f"cap +/-{MAX_RPM:.0f} RPM  {LOOP_HZ}Hz")
    )
    out.append("  " + SEP)
    out.append(
        f"  {c('up/dn')} +/-{STEP_RPM:.0f} RPM / +/-{STEP_STEER*360:.0f}deg   "
        f"{c('SPACE')} zero   {c('ENTER')} select IDs   {c('Q')} quit"
    )
    out.append("  " + SEP)
    out.append(
        f"  {'':1} {'ID':<4} {'Name':<12} {'Bus':<9} {'Command':<22} {'Feedback':<22} Bar"
    )
    out.append("  " + SEP)

    for cid in sorted(MOTOR_NAMES):
        name   = MOTOR_NAMES[cid]
        on     = cid in seen_ids
        in_sel = cid in selected

        if not on:
            mark = r("!")
        elif in_sel:
            mark = g(">")
        else:
            mark = d(".")

        bus_txt = "PRESENT  " if on else "MISSING  "
        bus = g(bus_txt) if on else r(bus_txt)

        if not on:
            cmd = d("--".ljust(22))
        elif not in_sel:
            cmd = d("(not targeted)".ljust(22))
        elif cid in DRIVE_IDS:
            sign = "+" if drive_vel >= 0 else ""
            cmd  = f"vel  {sign}{drive_vel:6.1f} RPM".ljust(22)
        else:
            deg  = (steer_frac % 1.0) * 360.0
            cmd  = f"pos  {steer_frac % 1.0:.3f} ({deg:5.1f} deg)".ljust(22)

        tel = telemetry.get(cid)
        if not on or tel is None:
            fb  = d("--".ljust(22))
            bar = ""
        elif tel[0] == "error":
            fb  = r(f"ERR {str(tel[1])[:17]}".ljust(22))
            bar = ""
        elif tel[0] == "vel":
            val = tel[1]
            fb  = f"vel  {val:+8.1f} RPM".ljust(22)
            bar = _bar(val, -MAX_RPM, MAX_RPM)
        else:
            val = tel[1]
            fb  = f"pos  {val:7.2f} deg".ljust(22)
            bar = _bar(val, 0, 360)

        out.append(f"  {mark} {cid:<4} {name:<12} {bus} {cmd} {fb} {bar}")

    out.append("  " + SEP)
    drv_bar = _bar(drive_vel, -MAX_RPM, MAX_RPM, width=16)
    deg = (steer_frac % 1.0) * 360.0
    out.append(
        f"  Drive: {b(f'{drive_vel:+.1f} RPM')} {drv_bar}   "
        f"Steer: {b(f'{deg:.1f} deg')} ({steer_frac % 1.0:.3f})"
    )
    if notice:
        out.append(f"\n  >> {y(notice)}")

    sys.stdout.write("\n".join(out))
    sys.stdout.flush()


def main():
    # ── imports ───────────────────────────────────────────────────────────────
    try:
        from sparkcan_py import SparkFlex
    except ImportError:
        print("ERROR: sparkcan_py not importable. Activate your venv first.")
        sys.exit(1)

    try:
        from sparkcan_py import CtrlType
        CTRL_VEL = CtrlType.kVelocity
        CTRL_POS = CtrlType.kPosition
    except ImportError:
        CTRL_VEL = CTRL_POS = None

    try:
        from sparkcan_py import IdleMode
        IDLE_COAST = IdleMode.kCoast
    except ImportError:
        IDLE_COAST = None

    # ── sniff ─────────────────────────────────────────────────────────────────
    print(f"Sniffing {CAN_IF} for 1 s ...", end="", flush=True)
    seen_ids = _sniff(CAN_IF, duration=1.0)
    found    = sorted(seen_ids & ALL_IDS)
    missing  = sorted(ALL_IDS - seen_ids)
    print(f" present: {found or 'none'}  missing: {missing or 'none'}")

    # ── connect (only motors present on bus) ──────────────────────────────────
    print("Connecting ...")
    motors: dict = {}
    for cid in sorted(MOTOR_NAMES):
        if cid not in seen_ids:
            print(f"  ID {cid} ({MOTOR_NAMES[cid]}): skipped (not on bus)")
            continue
        try:
            motors[cid] = SparkFlex(CAN_IF, cid)
            print(f"  ID {cid} ({MOTOR_NAMES[cid]}): OK")
        except Exception as e:
            print(f"  ID {cid} ({MOTOR_NAMES[cid]}): FAILED -- {e}")
        time.sleep(0.003)

    if not motors:
        print("No motors connected. Exiting.")
        sys.exit(1)

    # ── init each motor sequentially (mirrors base_motor.py DriveMotor /
    #    RotationMotor __init__):
    #      drive: SetIdleMode(kCoast) + SetCtrlType(kVelocity)
    #      steer: SetIdleMode(kCoast) only
    #    NO ResetFaults() — it disrupts kCtrlType writes
    # ──────────────────────────────────────────────────────────────────────────
    print("Initialising motors ...")
    for cid, m in motors.items():
        try:
            if IDLE_COAST is not None:
                m.SetIdleMode(IDLE_COAST)
        except Exception as e:
            print(f"  ID {cid}: SetIdleMode failed -- {e}")

        if cid in DRIVE_IDS:
            try:
                if CTRL_VEL is not None:
                    m.SetCtrlType(CTRL_VEL)
            except Exception as e:
                print(f"  ID {cid}: SetCtrlType(kVelocity) failed -- {e}")

        # Send a few heartbeats per motor before issuing the first command
        for _ in range(5):
            try:
                m.Heartbeat()
            except Exception:
                pass
            time.sleep(0.02)

    print("Ready. Starting teleop ...\n")
    time.sleep(0.2)

    # ── state ─────────────────────────────────────────────────────────────────
    selected   = set(ALL_IDS)
    drive_vel  = 0.0   # RPM
    steer_frac = 0.0   # [0, 1) wrapping
    telemetry: dict = {}
    notice     = ""

    def _zero():
        nonlocal drive_vel, steer_frac
        drive_vel  = 0.0
        steer_frac = 0.0

    import atexit
    atexit.register(_go_cooked)
    atexit.register(lambda: sys.stdout.write(_SHOW))

    _go_raw()
    sys.stdout.write(_HIDE)
    dt = 1.0 / LOOP_HZ

    try:
        while True:
            t0  = time.monotonic()
            key = _read_key(timeout=dt)

            if key == "QUIT":
                break
            elif key == "UP":
                drive_vel  = max(-MAX_RPM, min(MAX_RPM, drive_vel + STEP_RPM))
                steer_frac = steer_frac + STEP_STEER
                notice = ""
            elif key == "DOWN":
                drive_vel  = max(-MAX_RPM, min(MAX_RPM, drive_vel - STEP_RPM))
                steer_frac = steer_frac - STEP_STEER
                notice = ""
            elif key == "SPACE":
                _zero()
                notice = "Zeroed."
            elif key == "ENTER":
                _zero()
                selected = _prompt_ids()
                notice = (
                    "Targeting ALL motors."
                    if selected == ALL_IDS
                    else f"Targeting IDs {sorted(selected)}."
                )

            # ── heartbeat + command ────────────────────────────────────────
            for cid, m in motors.items():
                try:
                    m.Heartbeat()
                    if cid in selected:
                        if cid in DRIVE_IDS:
                            m.SetVelocity(float(drive_vel))
                        else:
                            m.SetPosition(float(steer_frac % 1.0))
                except Exception:
                    pass

            # ── telemetry ─────────────────────────────────────────────────
            for cid, m in motors.items():
                try:
                    if cid in DRIVE_IDS:
                        val = float(m.GetVelocity())
                        telemetry[cid] = ("vel", val)
                    else:
                        val = float(m.GetAbsoluteEncoderPosition())
                        telemetry[cid] = ("pos", val)
                except Exception as e:
                    telemetry[cid] = ("error", str(e)[:20])

            _draw(seen_ids, selected, drive_vel, steer_frac, telemetry, notice)

            leftover = dt - (time.monotonic() - t0)
            if leftover > 0.001:
                time.sleep(leftover)

    finally:
        for cid, m in motors.items():
            try:
                if cid in DRIVE_IDS:
                    m.SetVelocity(0.0)
                else:
                    m.SetPosition(float(steer_frac % 1.0))
                m.Heartbeat()
            except Exception:
                pass
        _go_cooked()
        sys.stdout.write(_SHOW + "\n")
        print("Motors zeroed. Goodbye.")


if __name__ == "__main__":
    main()
