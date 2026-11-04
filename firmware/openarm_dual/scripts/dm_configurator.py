#!/usr/bin/env python3
# =============================================================================
# dm_configurator.py
#
# A terminal configurator for DAMIAO DM-J-series actuators (e.g. DM-J8009P-2EC)
# reached through a DAMIAO USB-to-CAN dongle driven over its NATIVE SERIAL
# framing protocol (NOT socketcan / slcan / python-can).
#
# Purpose: a safe, label-driven tool to inspect and change motor parameters --
# above all the CAN_ID and MST_ID -- and to recover misconfigured drives.
#
# -----------------------------------------------------------------------------
# TWO WAYS TO USE THIS FILE
# -----------------------------------------------------------------------------
# 1. Humans -> interactive menu:
#        python dm_configurator.py
#    (run with no subcommand; it scans, lists motors, and gives a menu.)
#
# 2. Automation / scripts -> non-interactive subcommands (stable output,
#    exit code 0 = success, non-zero = failure):
#        python dm_configurator.py scan
#        python dm_configurator.py info 1
#        python dm_configurator.py get 1 mst_id
#        python dm_configurator.py set 1 ctrl_mode MIT
#        python dm_configurator.py setid 1 2          # CAN_ID 1 -> 2, MST_ID -> 0x12
#        python dm_configurator.py setid 1 2 0x30     # CAN_ID -> 2, MST_ID -> 0x30
#        python dm_configurator.py save 1
#        python dm_configurator.py status 1
#    Global options: --port /dev/ttyACM2  --baud 921600  (defaults shown)
#
# -----------------------------------------------------------------------------
# CAN_ID vs MST_ID (read this before changing IDs)
# -----------------------------------------------------------------------------
# CAN_ID (RID 8, "ESC_ID"): the address the motor LISTENS on. Commands (MIT,
#     enable/disable, position) are sent to arbitration_id = CAN_ID.
# MST_ID (RID 7, "Master ID"): the address the motor TRANSMITS feedback on.
#     Every feedback frame uses arbitration_id = MST_ID.
# Rules enforced/encouraged by this tool:
#   * Both must be UNIQUE per motor on a shared bus. Duplicate MST_IDs cause CAN
#     arbitration collisions (two motors transmit the same ID) -- the worst case.
#   * A motor's MST_ID must not equal another motor's CAN_ID (keep command-space
#     and feedback-space disjoint).
#   * Never use 0x7FF (the parameter/broadcast bus). 11-bit IDs only (<= 0x7FF).
#   * Convention used by OpenArms/LeRobot and recommended here:
#         MST_ID = CAN_ID + 0x10   (CAN 0x01..0x08  ->  MST 0x11..0x18)
#
# -----------------------------------------------------------------------------
# WIRE PROTOCOL (verified against cmjang/DM_Control_Python and live hardware)
# -----------------------------------------------------------------------------
# Host -> dongle: 30-byte serial frame. Only 3 fields vary:
#     [0]=0x55 [1]=0xAA [2]=0x1E(len=30) ... [13]=arbID_lo [14]=arbID_hi
#     ... [18]=0x08(DLC) ... [21:29]=8 CAN payload bytes ... [29]=0x00
# dongle -> host: 16-byte serial frame:
#     [0]=0xAA [1]=CMD(0x11=data) [2]=DLC [3:7]=CANID(LE) [7:15]=8 data [15]=0x55
#
# CAN-layer payloads (the 8 data bytes):
#     Enable / Disable / Zero : arbID = CAN_ID, data = [FF FF FF FF FF FF FF, XX]
#                               XX = FC enable, FD disable, FE set-zero
#     Param READ  : arbID = 0x7FF, data = [id_lo, id_hi, 0x33, RID, 0,0,0,0]
#     Param WRITE : arbID = 0x7FF, data = [id_lo, id_hi, 0x55, RID, v0,v1,v2,v3]
#     Param SAVE  : arbID = 0x7FF, data = [id_lo, id_hi, 0xAA, 0,0,0,0,0]
#     A param-read REPLY carries: data[0:2]=slaveID(LE), data[2]=0x33,
#                                 data[3]=RID, data[4:8]=value(LE).
#     Value is uint32 for "int" RIDs (7..10, 13..16, 35..36), else float32.
# =============================================================================

import argparse
import struct
import sys
import time

try:
    import serial  # pyserial
except ImportError:
    sys.exit("ERROR: pyserial is required.  pip install pyserial  "
             "(on this machine: use the lerobot_g1 conda env's python)")

# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------
DEFAULT_PORT = "/dev/ttyACM2"   # DAMIAO HDSC CDC dongle on this workstation
DEFAULT_BAUD = 921600           # USB-CDC line rate used by DAMIAO examples
PARAM_BUS_ID = 0x7FF            # reserved CAN ID for parameter operations

# The fixed 30-byte host->dongle template; __send patches [13],[14],[21:29].
_SEND_TEMPLATE = bytes([
    0x55, 0xAA, 0x1E, 0x03, 0x01, 0x00, 0x00, 0x00, 0x0A, 0x00, 0x00, 0x00,
    0x00, 0, 0, 0, 0, 0x00, 0x08, 0x00, 0x00, 0, 0, 0, 0, 0, 0, 0, 0, 0x00,
])

# Special CAN-frame trailing bytes.
CMD_ENABLE, CMD_DISABLE, CMD_SET_ZERO = 0xFC, 0xFD, 0xFE

# Control-mode enum (RID 10).
CTRL_MODES = {1: "MIT", 2: "POS_VEL", 3: "VEL", 4: "TORQUE_POS"}
# CAN bitrate enum (RID 35) -> bits per second.
CAN_BAUDS = {0: 125000, 1: 200000, 2: 250000, 3: 500000, 4: 1000000,
             5: 2000000, 6: 2500000, 7: 3200000, 8: 4000000, 9: 5000000}


def _rid_is_int(rid: int) -> bool:
    """Per the DAMIAO SDK, these RID ranges are uint32; everything else float32."""
    return (7 <= rid <= 10) or (13 <= rid <= 16) or (35 <= rid <= 36)


# -----------------------------------------------------------------------------
# Register catalogue: friendly name -> metadata.
#   rid       : DAMIAO register id
#   writable  : may this tool write it
#   kind      : 'int' | 'float' | 'mode' | 'baud'  (for display/encode)
#   unit/desc : for the human-facing table
# Only the commonly-useful registers are listed; add more from the manual as
# needed (the get/set commands also accept a raw numeric RID).
# -----------------------------------------------------------------------------
REGISTERS = [
    # name          rid  writable  kind     unit     description
    ("can_id",        8,  True,  "int",   "",     "CAN_ID / ESC_ID -- address the motor LISTENS on"),
    ("mst_id",        7,  True,  "int",   "",     "Master ID -- address the motor SENDS feedback on"),
    ("ctrl_mode",    10,  True,  "mode",  "",     "Control mode (MIT/POS_VEL/VEL/TORQUE_POS)"),
    ("can_baud",     35,  True,  "baud",  "bps",  "CAN bus bitrate (changing this needs dongle baud match!)"),
    ("timeout",       9,  True,  "int",   "ms",   "Comm-loss timeout; 0 disables the auto-disable watchdog"),
    ("uv_value",      0,  True,  "float", "V",    "Under-voltage protection threshold"),
    ("ov_value",     29,  True,  "float", "V",    "Over-voltage protection threshold"),
    ("ot_value",      2,  True,  "float", "degC", "Over-temperature protection threshold"),
    ("oc_value",      3,  True,  "float", "A",    "Over-current protection threshold"),
    ("pmax",         21,  True,  "float", "rad",  "Position fixed-point limit (affects MIT scaling!)"),
    ("vmax",         22,  True,  "float", "rad/s","Velocity fixed-point limit (affects MIT scaling!)"),
    ("tmax",         23,  True,  "float", "Nm",   "Torque fixed-point limit (affects MIT scaling!)"),
    ("acc",           4,  True,  "float", "",     "Acceleration limit"),
    ("dec",           5,  True,  "float", "",     "Deceleration limit"),
    ("max_spd",       6,  True,  "float", "",     "Max speed (pos/vel modes)"),
    ("gear_ratio",   20,  False, "float", "",     "Gear reduction ratio (read-only)"),
    ("hw_ver",       13,  False, "int",   "",     "Hardware version (read-only)"),
    ("sw_ver",       14,  False, "int",   "",     "Firmware version (read-only)"),
    ("sub_ver",      36,  False, "int",   "",     "Sub version (read-only)"),
    ("serial_no",    15,  False, "int",   "",     "Serial number (read-only)"),
]
_BY_NAME = {r[0]: r for r in REGISTERS}
_BY_RID = {r[1]: r for r in REGISTERS}


# =============================================================================
# Dongle driver -- native serial framing only.
# =============================================================================
class DmDongle:
    def __init__(self, port=DEFAULT_PORT, baud=DEFAULT_BAUD, verbose=True):
        self.verbose = verbose
        if verbose:
            print(f"[open] {port} @ {baud} baud")
        self.ser = serial.Serial(port, baud, timeout=0.1)
        self._tail = b""          # leftover bytes between reads (partial frame)
        time.sleep(0.2)
        self.ser.reset_input_buffer()

    def close(self):
        try:
            self.ser.close()
        except Exception:
            pass

    # --- low level -----------------------------------------------------------
    def _send(self, arb_id: int, data8):
        """Wrap an 8-byte CAN payload addressed to arb_id and write it."""
        frame = bytearray(_SEND_TEMPLATE)
        frame[13] = arb_id & 0xFF
        frame[14] = (arb_id >> 8) & 0xFF
        frame[21:29] = bytes(data8)
        self.ser.write(bytes(frame))

    def _read_frames(self, settle=0.12):
        """Read for `settle` seconds, return list of decoded (cmd, canid, data8)."""
        time.sleep(settle)
        buf = self._tail + self.ser.read_all()
        frames, i, consumed = [], 0, 0
        while i <= len(buf) - 16:
            if buf[i] == 0xAA and buf[i + 15] == 0x55:
                pkt = buf[i:i + 16]
                cmd = pkt[1]
                canid = pkt[3] | (pkt[4] << 8) | (pkt[5] << 16) | (pkt[6] << 24)
                frames.append((cmd, canid, pkt[7:15]))
                i += 16
                consumed = i
            else:
                i += 1
        self._tail = buf[consumed:]
        return frames

    # --- parameter bus -------------------------------------------------------
    def get_param(self, motor_id: int, rid: int, retries=20):
        """Read register `rid` from `motor_id`. Returns int/float or None."""
        for _ in range(retries):
            self._send(PARAM_BUS_ID,
                       [motor_id & 0xFF, (motor_id >> 8) & 0xFF, 0x33, rid, 0, 0, 0, 0])
            for cmd, _canid, d in self._read_frames(0.05):
                if cmd == 0x11 and d[2] == 0x33 and d[3] == rid:
                    slave = d[0] | (d[1] << 8)
                    if slave == motor_id:
                        raw = bytes(d[4:8])
                        return (struct.unpack("<I", raw)[0] if _rid_is_int(rid)
                                else round(struct.unpack("<f", raw)[0], 4))
        return None

    def set_param(self, motor_id: int, rid: int, value, verify=True):
        """Write register `rid` (volatile until save). Returns True if verified."""
        if _rid_is_int(rid):
            payload = struct.pack("<I", int(value) & 0xFFFFFFFF)
        else:
            payload = struct.pack("<f", float(value))
        self._send(PARAM_BUS_ID,
                   [motor_id & 0xFF, (motor_id >> 8) & 0xFF, 0x55, rid] + list(payload))
        if not verify:
            return True
        time.sleep(0.05)
        readback = self.get_param(motor_id, rid)
        if readback is None:
            return False
        if _rid_is_int(rid):
            return int(readback) == (int(value) & 0xFFFFFFFF)
        return abs(float(readback) - float(value)) < 0.1

    def save(self, motor_id: int):
        """Persist all params to flash. Motor is disabled first (DAMIAO requirement)."""
        self.disable(motor_id)
        time.sleep(0.02)
        self._send(PARAM_BUS_ID,
                   [motor_id & 0xFF, (motor_id >> 8) & 0xFF, 0xAA, 0, 0, 0, 0, 0])
        time.sleep(0.1)   # flash programming is blocking on the motor

    # --- special frames ------------------------------------------------------
    def _special(self, motor_id, last):
        self._send(motor_id, [0xFF] * 7 + [last])

    def enable(self, motor_id):   self._special(motor_id, CMD_ENABLE)
    def disable(self, motor_id):  self._special(motor_id, CMD_DISABLE)
    def set_zero(self, motor_id): self._special(motor_id, CMD_SET_ZERO)

    # --- discovery & telemetry ----------------------------------------------
    def scan(self, id_range=range(1, 17)):
        """Ping each CAN_ID with a passive param-read; return {can_id: mst_id}.

        Uses the 0x33 read whose reply carries the slave id in its payload, so it
        works regardless of how MST_ID is configured (even 0x00).
        """
        found = {}
        for mid in id_range:
            self._send(PARAM_BUS_ID, [mid & 0xFF, (mid >> 8) & 0xFF, 0x33, 0x00, 0, 0, 0, 0])
            for cmd, canid, d in self._read_frames(0.03):
                if cmd == 0x11 and d[2] == 0x33:
                    slave = d[0] | (d[1] << 8)
                    # `canid` field of the reply == the motor's MST_ID.
                    found.setdefault(slave, canid)
        return found

    def status(self, motor_id):
        """Enable (no motion) to elicit one feedback frame; decode it. Leaves enabled."""
        pmax = self.get_param(motor_id, 21) or 12.5
        vmax = self.get_param(motor_id, 22) or 45.0
        tmax = self.get_param(motor_id, 23) or 40.0
        self.enable(motor_id)
        for cmd, canid, d in self._read_frames(0.1):
            if cmd != 0x11 or d[2] in (0x33, 0x55):   # skip param replies
                continue
            mid = d[0] & 0x0F   # low nibble of data[0] is always the motor's CAN node id
            err = d[0] >> 4
            q = (d[1] << 8) | d[2]
            dq = (d[3] << 4) | (d[4] >> 4)
            tau = ((d[4] & 0xF) << 8) | d[5]
            return {
                "motor": mid, "err": err,
                "pos": round(q / 65535 * 2 * pmax - pmax, 3),
                "vel": round(dq / 4095 * 2 * vmax - vmax, 3),
                "torque": round(tau / 4095 * 2 * tmax - tmax, 3),
                "t_mos": d[6], "t_rotor": d[7],
            }
        return None


# =============================================================================
# Value formatting / parsing helpers (shared by CLI and menu)
# =============================================================================
def fmt_value(name, value):
    """Render a raw register value with its human label where applicable."""
    if value is None:
        return "<no response>"
    reg = _BY_NAME.get(name)
    kind = reg[3] if reg else "int"
    if name in ("hw_ver", "sw_ver", "serial_no"):   # these RIDs hold ASCII, not a number
        raw = struct.pack("<I", int(value) & 0xFFFFFFFF)
        text = "".join(chr(b) if 32 <= b < 127 else "." for b in raw)
        return f"0x{int(value):X} ('{text}')"
    if kind == "mode":
        return f"{value} ({CTRL_MODES.get(int(value), '?')})"
    if kind == "baud":
        return f"{value} ({CAN_BAUDS.get(int(value), '?')} bps)"
    if kind == "int":
        return f"{int(value)} (0x{int(value):X})"
    return str(value)


def parse_value(name, text):
    """Parse a user/CLI string into the right numeric type for register `name`."""
    reg = _BY_NAME.get(name)
    kind = reg[3] if reg else "int"
    t = text.strip()
    if kind == "mode":
        if t.upper() in [v for v in CTRL_MODES.values()]:
            return next(k for k, v in CTRL_MODES.items() if v == t.upper())
        return int(t, 0)
    if kind == "baud":
        bps = int(t, 0)
        if bps in CAN_BAUDS.values():        # accept "1000000"
            return next(k for k, v in CAN_BAUDS.items() if v == bps)
        return bps                            # or a raw index
    if kind == "float":
        return float(t)
    return int(t, 0)                          # int: accepts 0x.. or decimal


def resolve_rid(token):
    """Map a 'name' or numeric RID token to (name, rid, writable, kind)."""
    if token in _BY_NAME:
        r = _BY_NAME[token]
        return r[0], r[1], r[2], r[3]
    rid = int(token, 0)
    if rid in _BY_RID:
        r = _BY_RID[rid]
        return r[0], r[1], r[2], r[3]
    return f"rid{rid}", rid, True, ("int" if _rid_is_int(rid) else "float")


# =============================================================================
# High-level operations (used by both CLI subcommands and the menu)
# =============================================================================
def do_scan(dev):
    print("[scan] sweeping CAN_IDs 1..16 ...")
    found = dev.scan()
    if not found:
        print("  no motors responded. Check 24V power, wiring, and that the "
              "dongle's CAN bitrate matches the motor's.")
        return found
    print(f"  found {len(found)} motor(s):")
    for can_id in sorted(found):
        mst = found[can_id]
        ok = "OK" if mst == can_id + 0x10 else "non-standard (expected 0x%X)" % (can_id + 0x10)
        print(f"   CAN_ID 0x{can_id:02X} (={can_id})  ->  MST_ID 0x{mst:02X}   [{ok}]")
    return found


def do_info(dev, motor_id):
    print(f"[info] motor {motor_id} (0x{motor_id:02X}) -- reading registers ...")
    for name, rid, writable, kind, unit, desc in REGISTERS:
        val = dev.get_param(motor_id, rid)
        flag = " " if writable else "r"
        unit_s = f" {unit}" if unit and val is not None and kind not in ("baud", "mode") else ""
        print(f"  [{flag}] {name:<11} = {fmt_value(name, val)}{unit_s:<6}  - {desc}")
    print("  ([r] = read-only;  changing limits/protections is advanced)")


def do_set(dev, motor_id, token, value_text):
    name, rid, writable, kind = resolve_rid(token)
    if not writable:
        print(f"  refusing: '{name}' is read-only.")
        return False
    value = parse_value(name, value_text)
    print(f"[set] motor {motor_id}: {name} (RID {rid}) <- {fmt_value(name, value)}")
    ok = dev.set_param(motor_id, rid, value)
    print("  verified OK (volatile -- run 'save' to persist)" if ok
          else "  WRITE NOT VERIFIED (no readback match)")
    return ok


def do_setid(dev, old_id, new_can, new_mst=None):
    """Guided CAN_ID/MST_ID change. Mirrors the proven write-to-old / save-to-new
    sequence. Volatile change takes effect immediately; a power cycle locks it in."""
    if new_mst is None:
        new_mst = new_can + 0x10            # apply the convention by default
    # --- validate ---
    for label, v in (("CAN_ID", new_can), ("MST_ID", new_mst)):
        if not (1 <= v <= 0x7FE):
            print(f"  invalid {label} 0x{v:X}: must be 1..0x7FE (and never 0x7FF).")
            return False
    if new_mst == new_can:
        print("  refusing: MST_ID must differ from CAN_ID (command/feedback collision).")
        return False
    # --- collision check against the live bus ---
    others = {cid: m for cid, m in dev.scan().items() if cid != old_id}
    for cid, mst in others.items():
        if new_can in (cid, mst) or new_mst in (cid, mst):
            print(f"  WARNING: 0x{new_can:X}/0x{new_mst:X} collides with motor "
                  f"0x{cid:X} (MST 0x{mst:X}). Aborting to avoid a bus clash.")
            return False
    print(f"[setid] motor 0x{old_id:X}: CAN_ID -> 0x{new_can:X}, MST_ID -> 0x{new_mst:X}")
    dev.disable(old_id)
    time.sleep(0.05)
    # Write MST_ID first (still addressed to the OLD id), then CAN_ID (which
    # switches the listening id), then SAVE addressed to the NEW id.
    dev.set_param(old_id, 7, new_mst, verify=False)
    time.sleep(0.05)
    dev.set_param(old_id, 8, new_can, verify=False)
    time.sleep(0.05)
    dev._send(PARAM_BUS_ID, [new_can & 0xFF, (new_can >> 8) & 0xFF, 0xAA, 0, 0, 0, 0, 0])
    time.sleep(0.1)
    print("  written + saved.  >>> POWER-CYCLE THE MOTOR NOW <<<  then run "
          f"'scan' to confirm it appears at CAN_ID 0x{new_can:X}.")
    return True


def do_status(dev, motor_id):
    st = dev.status(motor_id)
    if st is None:
        print(f"[status] motor {motor_id}: no feedback.")
        return
    err_ok = "enabled/OK" if st["err"] == 1 else f"FAULT code 0x{st['err']:X}"
    print(f"[status] motor {st['motor']}: err=0x{st['err']:X} ({err_ok})  "
          f"pos={st['pos']} rad  vel={st['vel']} rad/s  torque={st['torque']} Nm  "
          f"T_mos={st['t_mos']}C  T_rotor={st['t_rotor']}C")


# =============================================================================
# Interactive menu (human-facing)
# =============================================================================
def menu(dev):
    print("\n=== DAMIAO serial configurator ===")
    found = do_scan(dev)
    while True:
        try:
            cmd = input("\n(scan/info <id>/get <id> <param>/set <id> <param> <val>/"
                        "setid <old> <can> [mst]/save <id>/status <id>/"
                        "en <id>/dis <id>/zero <id>/quit) >> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not cmd:
            continue
        p = cmd.split()
        op = p[0].lower()
        try:
            if op in ("quit", "exit", "q"):
                break
            elif op == "scan":
                do_scan(dev)
            elif op == "info" and len(p) == 2:
                do_info(dev, int(p[1], 0))
            elif op == "get" and len(p) == 3:
                name, rid, _, _ = resolve_rid(p[2])
                print(f"  {name} = {fmt_value(name, dev.get_param(int(p[1], 0), rid))}")
            elif op == "set" and len(p) == 4:
                do_set(dev, int(p[1], 0), p[2], p[3])
            elif op == "setid" and len(p) in (3, 4):
                do_setid(dev, int(p[1], 0), int(p[2], 0),
                         int(p[3], 0) if len(p) == 4 else None)
            elif op == "save" and len(p) == 2:
                dev.save(int(p[1], 0)); print("  saved to flash.")
            elif op == "status" and len(p) == 2:
                do_status(dev, int(p[1], 0))
            elif op in ("en", "enable") and len(p) == 2:
                dev.enable(int(p[1], 0)); print("  enabled.")
            elif op in ("dis", "disable") and len(p) == 2:
                dev.disable(int(p[1], 0)); print("  disabled.")
            elif op == "zero" and len(p) == 2:
                dev.set_zero(int(p[1], 0)); print("  zero position set.")
            else:
                print("  ?  check the syntax in the prompt.")
        except Exception as e:
            print(f"  error: {e}")


# =============================================================================
# CLI entry point
# =============================================================================
def main():
    ap = argparse.ArgumentParser(
        description="DAMIAO serial configurator (native USB-CAN framing).")
    ap.add_argument("--port", default=DEFAULT_PORT, help=f"serial port (default {DEFAULT_PORT})")
    ap.add_argument("--baud", type=int, default=DEFAULT_BAUD, help=f"COM baud (default {DEFAULT_BAUD})")
    sub = ap.add_subparsers(dest="cmd")

    sub.add_parser("scan", help="discover motors on the bus")
    s = sub.add_parser("info", help="dump all known registers of a motor");      s.add_argument("id")
    s = sub.add_parser("get", help="read one register");                          s.add_argument("id"); s.add_argument("param")
    s = sub.add_parser("set", help="write one register (volatile)");              s.add_argument("id"); s.add_argument("param"); s.add_argument("value")
    s = sub.add_parser("setid", help="change CAN_ID (+MST_ID) and save");         s.add_argument("id"); s.add_argument("new_can"); s.add_argument("new_mst", nargs="?")
    s = sub.add_parser("save", help="persist params to flash");                   s.add_argument("id")
    s = sub.add_parser("status", help="enable + read live feedback");             s.add_argument("id")
    s = sub.add_parser("enable", help="enable a motor");                          s.add_argument("id")
    s = sub.add_parser("disable", help="disable a motor");                        s.add_argument("id")
    s = sub.add_parser("zero", help="set current position as zero");              s.add_argument("id")
    args = ap.parse_args()

    dev = DmDongle(args.port, args.baud)
    rc = 0
    try:
        if args.cmd is None:
            menu(dev)
        elif args.cmd == "scan":
            rc = 0 if do_scan(dev) else 1
        elif args.cmd == "info":
            do_info(dev, int(args.id, 0))
        elif args.cmd == "get":
            name, rid, _, _ = resolve_rid(args.param)
            val = dev.get_param(int(args.id, 0), rid)
            print(f"{name} = {fmt_value(name, val)}")
            rc = 0 if val is not None else 1
        elif args.cmd == "set":
            rc = 0 if do_set(dev, int(args.id, 0), args.param, args.value) else 1
        elif args.cmd == "setid":
            rc = 0 if do_setid(dev, int(args.id, 0), int(args.new_can, 0),
                               int(args.new_mst, 0) if args.new_mst else None) else 1
        elif args.cmd == "save":
            dev.save(int(args.id, 0)); print("saved.")
        elif args.cmd == "status":
            do_status(dev, int(args.id, 0))
        elif args.cmd in ("enable", "disable", "zero"):
            getattr(dev, {"enable": "enable", "disable": "disable", "zero": "set_zero"}[args.cmd])(int(args.id, 0))
            print(f"{args.cmd} sent.")
    finally:
        dev.close()
    sys.exit(rc)


if __name__ == "__main__":
    main()
