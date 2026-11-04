#!/usr/bin/env python3
"""
detect_serial.py
====================

Detect DaMiao actuators on a DM-USB2FDCAN *serial* adapter (/dev/ttyACM*).

This adapter does NOT speak slcan/SocketCAN — it uses DaMiao's own binary
serial framing (header 0x55 0xAA on send, 0xAA..0x55 on receive). This script
implements just enough of that protocol to PROVE the link and find which motor
IDs answer, without energizing anything: it sends the MIT **disable** command
(``FF FF FF FF FF FF FF FD``), which makes the motor reply with one feedback
frame while leaving it de-energized.

Usage:
    python detect_serial.py                  # scan IDs 1..8 on ttyACM0/1
    python detect_serial.py --port /dev/ttyACM1 --ids 1
    python detect_serial.py --raw             # also dump raw rx bytes
"""
from __future__ import annotations

import argparse
import sys
import time

import _bootstrap  # noqa: F401  (puts src/ on sys.path)

try:
    import serial  # pyserial
except ImportError:
    sys.exit("pyserial missing: pip install pyserial")

from mabel_actid.serial_ports import auto_port, list_ports

SEND_HEADER = (0x55, 0xAA)
RECV_HEADER = 0xAA
RECV_TAIL = 0x55
RECV_LEN = 16

# 30-byte send template from DaMiao DM_CAN.py (cmjang/DM_Control_Python).
# idx 0-1 header, 13-16 CAN id (LE), 18 DLC, 21-28 the 8 CAN data bytes.
SEND_TEMPLATE = [
    0x55, 0xAA, 0x1e, 0x03, 0x01, 0x00, 0x00, 0x00, 0x0a, 0x00, 0x00, 0x00,
    0x00, 0, 0, 0, 0, 0x00, 0x08, 0x00, 0x00, 0, 0, 0, 0, 0, 0, 0, 0, 0x00,
]

DISABLE = [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFD]
READ_CAN_ID = 0x7FF          # status-request frames are addressed to 0x7FF
CMD_STATUS = 0xCC            # "report your status" — non-energizing


def status_request(can_id: int) -> bytes:
    """0xCC status request (non-energizing) — the canonical detect, per DM_CAN.py."""
    return build_frame(READ_CAN_ID,
                       [can_id & 0xFF, (can_id >> 8) & 0xFF, CMD_STATUS, 0, 0, 0, 0, 0])


def build_frame(can_id: int, data8: list[int]) -> bytes:
    f = list(SEND_TEMPLATE)
    f[13] = can_id & 0xFF
    f[14] = (can_id >> 8) & 0xFF
    f[15] = (can_id >> 16) & 0xFF
    f[16] = (can_id >> 24) & 0xFF
    f[21:29] = data8
    return bytes(f)


def parse_frames(buf: bytes):
    """Yield (can_id, data8) for every well-formed 16-byte rx frame in buf."""
    i = 0
    n = len(buf)
    while i <= n - RECV_LEN:
        if buf[i] == RECV_HEADER and buf[i + RECV_LEN - 1] == RECV_TAIL:
            p = buf[i:i + RECV_LEN]
            can_id = p[3] | (p[4] << 8) | (p[5] << 16) | (p[6] << 24)
            data8 = p[7:15]
            yield can_id, data8
            i += RECV_LEN
        else:
            i += 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=None, help="serial port (auto-detect if omitted)")
    ap.add_argument("--baud", type=int, default=921600, help="CDC-ACM baud (usually ignored)")
    ap.add_argument("--ids", default="1-8", help="motor IDs to probe, e.g. '1' or '1-8' or '1,2,5'")
    ap.add_argument("--raw", action="store_true", help="hexdump raw bytes received")
    args = ap.parse_args()

    # resolve port (cross-platform: macOS cu.usbmodem*, Linux ttyACM*/ttyUSB*)
    port = args.port
    if port is None:
        avail = list_ports()
        if not avail:
            return _fail("no serial adapter found — is the DM-USB2FDCAN plugged in? "
                         "(macOS: ls /dev/cu.usbmodem*  Linux: ls /dev/ttyACM*)")
        port = auto_port()
        print(f"[i] auto-selected port {port} (override with --port)")
        if len(avail) > 1:
            print(f"    other ports seen: {', '.join(p.device for p in avail[1:])}")

    # parse ids
    ids: list[int] = []
    for tok in args.ids.split(","):
        if "-" in tok:
            a, b = tok.split("-"); ids += list(range(int(a), int(b) + 1))
        else:
            ids.append(int(tok))

    try:
        ser = serial.Serial(port, args.baud, timeout=0.25)
    except Exception as e:
        return _fail(f"could not open {port}: {e}")

    print(f"[i] opened {port} @ {args.baud}")
    time.sleep(0.3)
    ser.reset_input_buffer()

    found = {}
    raw_total = bytearray()
    for mid in ids:
        ser.reset_input_buffer()
        ser.write(status_request(mid))   # 0xCC non-energizing status query
        ser.flush()
        time.sleep(0.08)
        rx = ser.read(256)
        raw_total += rx
        for can_id, d in parse_frames(rx):
            rep_id = d[0] & 0x0F
            err = (d[0] >> 4) & 0x0F
            found[rep_id] = (err, bytes(d))
        print(f"  probed ID {mid}: {len(rx)} bytes back"
              + (f"  -> reply from motor {list(found)[-1]}" if found else ""))

    ser.close()

    if args.raw:
        print("\n[raw rx hexdump]")
        print(" ", raw_total.hex(" ") if raw_total else "(none)")

    print("\n" + "=" * 48)
    if found:
        for mid, (err, d) in sorted(found.items()):
            status = "OK (no fault)" if err == 0 else f"FAULT code 0x{err:X}"
            print(f"  ✅ Motor ID {mid}: responding — {status}")
            print(f"     feedback bytes: {d.hex(' ')}")
        print("=" * 48)
        print(f"SUCCESS: {len(found)} actuator(s) on the serial line.")
        return 0
    else:
        print("  ❌ No actuator replied.")
        print("=" * 48)
        print("Checks: 24 V motor power ON? CAN_H/CAN_L not swapped? correct IDs?")
        print("If raw rx had bytes but no parse, the framing/baud may differ — rerun with --raw.")
        return 2


def _fail(msg: str) -> int:
    print(f"[ERROR] {msg}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
