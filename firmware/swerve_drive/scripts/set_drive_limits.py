#!/usr/bin/env python3
"""
One-time hardware speed cap for the swerve DRIVE motors (REV SPARK Flex).

What it does
------------
Writes the closed-loop output-range registers of PID slot 0 on each drive
controller and burns them to flash so the cap survives power cycles and is
honoured by *every* stack that talks to these motors (this teleop, ROS, Teensy):

    kOutputMin_0 (param 19, FLOAT) = -CAP
    kOutputMax_0 (param 20, FLOAT) = +CAP        CAP = 0.10  (10% of full output)
    kCtrlType    (param  5, UINT ) =  1 (Velocity)   # ensure slot-0 limits apply
    BurnFlash    (0xA3 0x3A)

With OutputMax_0 = 0.10 the velocity PID can never command more than 10% applied
output, so the drive motors physically top out near 10% of their free speed no
matter what RPM the teleop asks for.

This script is self-contained raw SocketCAN — it does NOT import sparkcan_py and
needs no rebuild. Steer motors (IDs 2/4/6) are never touched.

Safety
------
- Sends ONLY parameter writes + BurnFlash. No setpoints, so motors do not move.
- Prompts for confirmation before burning flash (flash has limited write cycles).
  Pass --yes to skip the prompt.
- Attempts a best-effort read-back. NOTE: most SPARK firmware does not answer
  parameter reads on a live (busy) bus, so a blank read-back does NOT mean the
  write failed — confirm by power-cycling and driving the teleop (wheels should
  top out ~10% speed).

Usage
-----
    sudo ip link set can0 up type can bitrate 1000000   # if not already up
    python scripts/set_drive_limits.py                  # interactive
    python scripts/set_drive_limits.py --cap 0.10 --yes
    python scripts/set_drive_limits.py --verify-only     # just read current values
"""

import argparse
import socket
import struct
import sys
import time

# ── config ────────────────────────────────────────────────────────────────────
CAN_IF      = "can0"
DRIVE_IDS   = [1, 3, 5]          # RF / LF / B drive
DEFAULT_CAP = 0.10               # 10% of full output

# ── REV CAN protocol constants (see REV_CAN_PROTOCOL.md) ───────────────────────
DEVICE_TYPE  = 0x02
MANUFACTURER = 0x05
CAN_EFF_FLAG = 0x80000000
CAN_EFF_MASK = 0x1FFFFFFF
FRAME_FMT    = "=IB3x8s"
FRAME_SIZE   = struct.calcsize(FRAME_FMT)

PARAM_API_CLASS = 48
PARAM_TYPE_UINT  = 0x01
PARAM_TYPE_FLOAT = 0x02
PARAM_TYPE_BOOL  = 0x03

# Parameter IDs (SparkBase.hpp)
P_CTRL_TYPE  = 5
P_OUTPUT_MIN = 19   # kOutputMin_0
P_OUTPUT_MAX = 20   # kOutputMax_0

CTRL_VELOCITY = 1

# BurnFlash command: apiClass 63, apiIndex 15  ->  (63<<4)|15
CMD_BURNFLASH = (63 << 4) | 15

# ── ANSI ───────────────────────────────────────────────────────────────────────
_RST = "\033[0m"; _BLD = "\033[1m"; _GRN = "\033[32m"; _RED = "\033[31m"
_YLW = "\033[33m"; _CYN = "\033[36m"; _DIM = "\033[2m"


def b(s):  return f"{_BLD}{s}{_RST}"
def g(s):  return f"{_GRN}{s}{_RST}"
def r(s):  return f"{_RED}{s}{_RST}"
def y(s):  return f"{_YLW}{s}{_RST}"
def c(s):  return f"{_CYN}{s}{_RST}"
def d(s):  return f"{_DIM}{s}{_RST}"


# ── arb-id helpers ──────────────────────────────────────────────────────────────
def _cmd_arb(cmd_enum: int, dev: int) -> int:
    api_class = (cmd_enum >> 4) & 0x3F
    api_index = cmd_enum & 0x0F
    return ((DEVICE_TYPE << 24) | (MANUFACTURER << 16)
            | (api_class << 10) | (api_index << 6) | dev)


def _param_arb(param_id: int, dev: int) -> int:
    return ((DEVICE_TYPE << 24) | (MANUFACTURER << 16)
            | (PARAM_API_CLASS << 10) | (param_id << 6) | dev)


def _send(sock, arb: int, data: bytes):
    payload = (data + b"\x00" * 8)[:8]
    frame = struct.pack(FRAME_FMT, arb | CAN_EFF_FLAG, min(len(data), 8), payload)
    for _ in range(1000):
        try:
            sock.send(frame)
            return
        except (BlockingIOError, InterruptedError):
            time.sleep(0.001)
    raise OSError("CAN TX buffer stayed full for 1 s")


def write_param(sock, dev: int, param_id: int, value, ptype: int):
    if ptype == PARAM_TYPE_FLOAT:
        raw = struct.pack("<f", float(value))
    elif ptype == PARAM_TYPE_BOOL:
        raw = bytes([1 if value else 0]) + b"\x00" * 3
    else:  # UINT
        raw = struct.pack("<I", int(value))
    _send(sock, _param_arb(param_id, dev), raw + bytes([ptype]))


def read_param(sock, dev: int, param_id: int, timeout: float = 0.25):
    """Request a parameter (DLC=0) and return (value, type) or None on timeout."""
    arb = _param_arb(param_id, dev)
    # drain any stale frames for this exact arb, then request
    _send(sock, arb, b"")
    t_end = time.monotonic() + timeout
    while time.monotonic() < t_end:
        try:
            data = sock.recv(FRAME_SIZE)
        except socket.timeout:
            continue
        except BlockingIOError:
            time.sleep(0.001)
            continue
        if len(data) < FRAME_SIZE:
            continue
        can_id, dlc, payload = struct.unpack(FRAME_FMT, data)
        if (can_id & CAN_EFF_MASK) != arb or dlc < 5:
            continue
        ptype = payload[4]
        if ptype == PARAM_TYPE_FLOAT:
            return struct.unpack("<f", payload[:4])[0], ptype
        if ptype == PARAM_TYPE_BOOL:
            return (payload[0] != 0), ptype
        return struct.unpack("<I", payload[:4])[0], ptype
    return None


def burn_flash(sock, dev: int):
    _send(sock, _cmd_arb(CMD_BURNFLASH, dev), bytes([0xA3, 0x3A]))


# ── main ────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Burn a 10% output cap into the drive SPARKs.")
    ap.add_argument("--cap", type=float, default=DEFAULT_CAP,
                    help=f"max applied output, 0..1 (default {DEFAULT_CAP} = 10%%)")
    ap.add_argument("--iface", default=CAN_IF, help="CAN interface (default can0)")
    ap.add_argument("--ids", type=int, nargs="+", default=DRIVE_IDS,
                    help=f"drive motor IDs (default {DRIVE_IDS})")
    ap.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    ap.add_argument("--no-ctrl-type", action="store_true",
                    help="do not assert kCtrlType=Velocity (only write output limits)")
    ap.add_argument("--verify-only", action="store_true",
                    help="read and print current registers; write nothing")
    args = ap.parse_args()

    cap = abs(args.cap)
    if not 0.0 < cap <= 1.0:
        print(r(f"--cap must be in (0, 1]; got {args.cap}"))
        return 1

    try:
        sock = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
        sock.bind((args.iface,))
        sock.settimeout(0.05)
    except OSError as e:
        print(r(f"Cannot open {args.iface}: {e}"))
        print(d("  Bring it up first:  sudo ip link set can0 up type can bitrate 1000000"))
        return 1

    print(b("Drive-motor output cap"))
    print(d(f"  interface {args.iface}   drive IDs {args.ids}   cap = "
            f"{cap:.3f}  (±{cap*100:.0f}% applied output)"))
    print()

    def try_read(tag):
        """Best-effort register read-back. NOTE: most SPARK firmware does not
        answer parameter-read requests on a live (busy) bus, so '----' here means
        'could not read back', NOT that a write failed. Returns (dict, got_any)."""
        print(f"  {tag}")
        res, got_any = {}, False
        for dev in args.ids:
            omn = read_param(sock, dev, P_OUTPUT_MIN)
            omx = read_param(sock, dev, P_OUTPUT_MAX)
            ct  = read_param(sock, dev, P_CTRL_TYPE)
            res[dev] = (omn, omx, ct)
            got_any = got_any or any(v is not None for v in (omn, omx, ct))
            f = lambda p: "----" if p is None else (f"{p[0]:+.3f}" if isinstance(p[0], float) else str(p[0]))
            print(f"    ID {dev}:  OutputMin={f(omn):>7}  OutputMax={f(omx):>7}  CtrlType={f(ct)}")
        if not got_any:
            print(d("    (no read-back — this firmware does not answer parameter reads on a live bus)"))
        print()
        return res, got_any

    try_read("Current registers (best-effort read):")

    if args.verify_only:
        sock.close()
        return 0

    if not args.yes:
        print(y(f"  This BURNS FLASH on drive IDs {args.ids} (limited write cycles)."))
        try:
            ans = input("  Type 'yes' to proceed: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            ans = ""
        if ans not in ("y", "yes"):
            print(d("  Aborted — nothing written."))
            sock.close()
            return 0

    print(b("\n  Writing …"))
    for dev in args.ids:
        if not args.no_ctrl_type:
            write_param(sock, dev, P_CTRL_TYPE, CTRL_VELOCITY, PARAM_TYPE_UINT)
            time.sleep(0.02)
        write_param(sock, dev, P_OUTPUT_MIN, -cap, PARAM_TYPE_FLOAT)
        time.sleep(0.02)
        write_param(sock, dev, P_OUTPUT_MAX, +cap, PARAM_TYPE_FLOAT)
        time.sleep(0.02)
        burn_flash(sock, dev)
        print(d(f"    ID {dev}: wrote ±{cap:.3f} + burned flash"))
        time.sleep(0.20)   # give the controller time to commit flash

    time.sleep(0.3)
    print()
    after, got_any = try_read("Read back (best-effort):")

    # ── verdict ────────────────────────────────────────────────────────────────
    print()
    if got_any:
        ok = True
        for dev in args.ids:
            omn, omx, _ = after[dev]
            good = (omx is not None and abs(omx[0] - cap) < 1e-3
                    and omn is not None and abs(omn[0] + cap) < 1e-3)
            ok = ok and good
            print(f"  ID {dev}: {g('PASS') if good else y('unverified')}")
        print()
        if ok:
            print(g(b(f"All drive motors read back ±{cap*100:.0f}% — burned to flash.")))
        else:
            print(y(b("Writes + BurnFlash sent, but read-back was incomplete.")))
            print(d("  Power-cycle the controllers and confirm by driving in swerve_teleop."))
    else:
        print(g(b(f"Wrote ±{cap*100:.0f}% output cap + BurnFlash to drive IDs {args.ids}.")))
        print(d("  This firmware doesn't answer parameter reads on a live bus, so the"))
        print(d("  values can't be read back here. Confirm the cap like this:"))
        print(d("    1. power-cycle the SPARK controllers"))
        print(d("    2. run  python scripts/swerve_teleop.py  and drive forward"))
        print(d("    3. the wheels should top out around 10% of full speed"))
    sock.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
