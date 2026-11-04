#!/usr/bin/env python3
"""
ble_protocol.py — the compact binary teleop frame NOVA exchanges over BLE.

JSON is what every other NOVA transport speaks, but it does not fit here: iOS negotiates
ATT_MTU ~185 bytes, and `{"left_arm_1": -0.123, ...}` for 21 joints is ~400 bytes — two or
three fragmented writes per control tick. So BLE gets a packed layout instead, and this
module is the ONE definition of it, shared by the Linux peripheral and the iOS / visionOS
central. Keep them in step by changing this file, not by editing either side by hand.

    byte 0        message type
    type 0x01     JOINT_COMMAND   u8 count, then count x (u8 joint_index, f32 radians; a grip is 0..1)
    type 0x05     HAND_POSE       u8 side (0 left, 1 right), u8 flags (bit0 tracked), then 21 x (i16 x, i16 y, i16 z)
                                  = the ARKit hand's joint POSITIONS relative to the wrist in 0.02 mm, HAND_JOINTS order
                                  (129 bytes: one write). The gateway's retargeter reads positions only, so this is the
                                  whole HandPose a teleop_frame carries, 30x smaller than its JSON.
    type 0x06     HEAD_POSE       u8 flags (bit0 tracked), then 9 x i16 = the head's 3x3 rotation (row-major, ARKit world
                                  frame, x1e-4). The gateway turns it into neck yaw / pitch / roll like a Wi-Fi frame's head.
    type 0x02     TELEOP_FRAME    f32 vx, f32 vy, f32 wz [, f32 lift]  (base twist + lift stick)
    type 0x03     ESTOP           u8 active (1 = engage, 0 = release)
    type 0x04     CONTROL_MODE    u8 flags  (bit0 soft, bit1 gravityComp)
    type 0x81     STATE (notify)  u8 flags, u8 count, count x f32 radians
    type 0x82     HEALTH (notify) u8 count, count x (u8 module, u8 n, u8 flags,
                                  ceil(n/8) bytes: bit k = actuator k is answering)

All floats are little-endian f32. Joint INDEX, not name: an index costs 1 byte where
"left_arm_1" costs 10, which is the whole reason this format exists. JOINTS below is the
canonical order — append only, never reorder or renumber, or an old app silently drives the
wrong joints.
"""
from __future__ import annotations

import struct

# CANONICAL JOINT ORDER — index IS the wire identity. APPEND ONLY.
JOINTS: tuple[str, ...] = (
    "left_arm_1", "left_arm_2", "left_arm_3", "left_arm_4",
    "left_arm_5", "left_arm_6", "left_arm_7",
    "right_arm_1", "right_arm_2", "right_arm_3", "right_arm_4",
    "right_arm_5", "right_arm_6", "right_arm_7",
    "torso",
    "neck_yaw", "neck_pitch", "neck_roll",
    "lift_lower", "lift_upper",
    # 20+: the HANDS (2026-09-22). A grip is what the apps' grip triggers send (the gateway expands it,
    # plant._expand_grips); the wrists and the 16 finger joints per hand are what a full pose carries.
    # COMMAND-ONLY over Bluetooth: STATE keeps to STATE_JOINTS so an older app build never meets an index
    # it does not know, and a 17-joint hand echo would not fit an ATT_MTU anyway (the apps echo their own targets).
    "left_grip", "right_grip", "left_wrist", "right_wrist",
    *[f"left_{j}" for j in ("thumb_mcp", "thumb_abd", "thumb_pip", "thumb_dip",
                            "index_abd", "index_mcp", "index_pip", "middle_abd", "middle_mcp", "middle_pip",
                            "ring_abd", "ring_mcp", "ring_pip", "pinky_abd", "pinky_mcp", "pinky_pip")],
    *[f"right_{j}" for j in ("thumb_mcp", "thumb_abd", "thumb_pip", "thumb_dip",
                             "index_abd", "index_mcp", "index_pip", "middle_abd", "middle_mcp", "middle_pip",
                             "ring_abd", "ring_mcp", "ring_pip", "pinky_abd", "pinky_mcp", "pinky_pip")],
)
INDEX = {name: i for i, name in enumerate(JOINTS)}
STATE_JOINTS = frozenset(JOINTS[:20])       # what a STATE notification may carry (see above)

JOINT_COMMAND, TELEOP_FRAME, ESTOP, CONTROL_MODE, HAND_POSE, HEAD_POSE, STATE, HEALTH = 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x81, 0x82
HEAD_UNIT = 1e-4            # the head's 3x3 rotation as i16 in 1e-4 (|entry| ≤ 1): 19 bytes
# HAND_POSE point order — the 21 ARKit joints the retargeter uses (wrist, then thumb / index / middle / ring /
# little from knuckle to tip; the same order as MediaPipe's landmarks). The apps carry the same list.
HAND_JOINTS: tuple[str, ...] = ("wrist",) + tuple(
    f"{f}_{lvl}" for f in ("thumb", "index", "middle", "ring", "little") for lvl in ("knuckle", "intermediate_base", "intermediate_tip", "tip"))
HAND_UNIT = 2e-5            # metres per count: i16 in 0.02 mm covers ±0.65 m — a wrist-relative hand is under 0.3 m
_IDENT = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]

# CANONICAL MODULE ORDER — index IS the wire identity, exactly like JOINTS. APPEND ONLY.
# Mirrors module_roster.MODULES, and the app mirrors this table (BLEProtocol.swift).
MODULES: tuple[str, ...] = ("base", "torso", "lift", "neck", "leftArm", "rightArm",
                            "leftHand", "rightHand", "camHead")
MODULE_INDEX = {name: i for i, name in enumerate(MODULES)}

HEALTH_HAS_DRIVER, HEALTH_TORQUE, HEALTH_OK = 0x01, 0x02, 0x04
MAX_ACTUATORS = 32          # a hand is 17; 32 keeps the bitmap at 4 bytes

MAX_PAYLOAD = 180          # keep one write inside a single ATT_MTU on iOS (~185)


class BleProtocolError(ValueError):
    """A malformed frame. Never raised past the adapter — a bad packet is dropped."""


def encode_joint_command(targets: dict[str, float]) -> bytes:
    """{joint: radians} -> packed. Unknown joints are skipped, not guessed."""
    items = [(INDEX[n], float(v)) for n, v in targets.items() if n in INDEX]
    if not items:
        return b""
    if 2 + 5 * len(items) > MAX_PAYLOAD:                 # one tick must be ONE write
        items = items[:(MAX_PAYLOAD - 2) // 5]
    out = bytearray([JOINT_COMMAND, len(items)])
    for idx, val in items:
        out += struct.pack("<Bf", idx, val)
    return bytes(out)


def encode_hand_pose(side: str, points: dict[str, tuple], tracked: bool = True) -> bytes:
    """{ARKit joint: (x, y, z) m, any frame} -> packed, wrist-relative. A joint that is missing is sent
    at the wrist (the retargeter ignores what it cannot use)."""
    w = points.get("wrist") or (0.0, 0.0, 0.0)
    out = bytearray([HAND_POSE, 0 if side == "left" else 1, 0x01 if tracked else 0x00])
    for n in HAND_JOINTS:
        q = points.get(n) or w
        out += struct.pack("<hhh", *(max(-32768, min(32767, int(round((float(q[k]) - float(w[k])) / HAND_UNIT)))) for k in range(3)))
    return bytes(out)


def encode_head_pose(R, tracked: bool = True) -> bytes:
    """A 3x3 rotation (rows) -> packed HEAD_POSE."""
    out = bytearray([HEAD_POSE, 0x01 if tracked else 0x00])
    for r in range(3):
        for c in range(3):
            out += struct.pack("<h", max(-32768, min(32767, int(round(float(R[r][c]) / HEAD_UNIT)))))
    return bytes(out)


def head_pose_payload(R, tracked: bool = True) -> dict:
    """A decoded HEAD_POSE -> the teleop_frame `head` dict (row-major 4x4, no translation)."""
    m = [float(R[r][c]) if c < 3 and r < 3 else (1.0 if r == c else 0.0) for r in range(4) for c in range(4)]
    return {"transform": {"matrix": m}, "trackingState": "normal" if tracked else "notAvailable"}


def hand_pose_payload(side: str, points: dict[str, tuple], tracked: bool = True) -> dict:
    """A decoded HAND_POSE -> the teleop_frame HandPose dict the gateway retargets (chirality, identity
    anchor, one row-major 4x4 per joint with the position in its last column, isTracked)."""
    joints = []
    for n, (x, y, z) in points.items():
        m = list(_IDENT); m[3], m[7], m[11] = float(x), float(y), float(z)
        joints.append({"joint": n, "localTransform": {"matrix": m}, "isTracked": True})
    return {"chirality": side, "anchorTransform": {"matrix": list(_IDENT)}, "joints": joints, "isTracked": bool(tracked)}


def decode(frame: bytes) -> tuple[int, object]:
    """packed -> (type, payload). Raises BleProtocolError on anything malformed."""
    if not frame:
        raise BleProtocolError("empty frame")
    kind = frame[0]
    body = frame[1:]
    if kind == JOINT_COMMAND:
        if not body:
            raise BleProtocolError("joint_command: missing count")
        n, rest = body[0], body[1:]
        if len(rest) != 5 * n:
            raise BleProtocolError(f"joint_command: count {n} needs {5*n} bytes, got {len(rest)}")
        out = {}
        for i in range(n):
            idx, val = struct.unpack_from("<Bf", rest, 5 * i)
            if idx >= len(JOINTS):
                raise BleProtocolError(f"joint index {idx} out of range")
            if val != val or val in (float("inf"), float("-inf")):   # NaN / inf
                raise BleProtocolError(f"joint {JOINTS[idx]}: non-finite target")
            out[JOINTS[idx]] = val
        return kind, out
    if kind == TELEOP_FRAME:
        # 16 bytes since the lift joined the frame; 12 is an older app and means "no lift
        # stick", NOT "lift stick at zero" — a build that cannot ask must not be read as
        # asking for a stop it never sent.
        if len(body) == 12:
            vx, vy, wz = struct.unpack("<fff", body)
            return kind, {"vx": vx, "vy": vy, "wz": wz}
        if len(body) == 16:
            vx, vy, wz, lift = struct.unpack("<ffff", body)
            if lift != lift:                                 # NaN
                raise BleProtocolError("teleop_frame: non-finite lift")
            return kind, {"vx": vx, "vy": vy, "wz": wz, "lift": max(-1.0, min(1.0, lift))}
        raise BleProtocolError(f"teleop_frame: expected 12 or 16 bytes, got {len(body)}")
    if kind == HAND_POSE:
        if len(body) != 2 + 6 * len(HAND_JOINTS):
            raise BleProtocolError(f"hand_pose: expected {2 + 6 * len(HAND_JOINTS)} bytes, got {len(body)}")
        side = "left" if body[0] == 0 else "right"
        pts = {n: tuple(v * HAND_UNIT for v in struct.unpack_from("<hhh", body, 2 + 6 * i)) for i, n in enumerate(HAND_JOINTS)}
        return kind, {"side": side, "tracked": bool(body[1] & 0x01), "points": pts}
    if kind == HEAD_POSE:
        if len(body) != 1 + 18:
            raise BleProtocolError(f"head_pose: expected 19 bytes, got {len(body)}")
        vals = struct.unpack_from("<9h", body, 1)
        return kind, {"tracked": bool(body[0] & 0x01), "R": [[vals[3 * r + c] * HEAD_UNIT for c in range(3)] for r in range(3)]}
    if kind == ESTOP:
        if len(body) != 1:
            raise BleProtocolError(f"estop: expected 1 byte, got {len(body)}")
        return kind, {"active": bool(body[0])}
    if kind == CONTROL_MODE:
        if len(body) != 1:
            raise BleProtocolError(f"control_mode: expected 1 byte, got {len(body)}")
        f = body[0]
        return kind, {"controlType": "soft" if f & 0x01 else "impedance",
                      "stiffness": 0.2 if f & 0x01 else 1.0,
                      "gravityComp": bool(f & 0x02)}
    raise BleProtocolError(f"unknown message type 0x{kind:02x}")


def encode_state(positions: dict[str, float], *, estop: bool, torque: bool) -> bytes:
    """STATE notification. Only joints present in `positions` are sent, so a robot with no
    neck or lift costs nothing for them."""
    items = [(INDEX[n], float(v)) for n, v in positions.items() if n in STATE_JOINTS]
    items.sort()
    if 3 + 5 * len(items) > MAX_PAYLOAD:
        items = items[:(MAX_PAYLOAD - 3) // 5]
    flags = (0x01 if estop else 0) | (0x02 if torque else 0)
    out = bytearray([STATE, flags, len(items)])
    for idx, val in items:
        out += struct.pack("<Bf", idx, val)
    return bytes(out)


def decode_state(frame: bytes) -> tuple[dict[str, float], bool, bool]:
    if not frame or frame[0] != STATE or len(frame) < 3:
        raise BleProtocolError("not a STATE frame")
    flags, n, rest = frame[1], frame[2], frame[3:]
    if len(rest) != 5 * n:
        raise BleProtocolError("state: truncated")
    pos = {}
    for i in range(n):
        idx, val = struct.unpack_from("<Bf", rest, 5 * i)
        if idx < len(JOINTS):
            pos[JOINTS[idx]] = val
    return pos, bool(flags & 0x01), bool(flags & 0x02)


# ── module / actuator health ─────────────────────────────────────────────────────────
# WHY THIS EXISTS: over Wi-Fi the apps get the full roster as JSON — every module, every
# motor, its own label and temperature. BLE has ~180 bytes a write, so that roster does
# not fit and the status panel showed nothing at all: connected over Bluetooth, every
# module sat on "WAIT / no data yet" including a lift that was plugged in and streaming.
# A dead motor is exactly what you need to know on the link you fall back to, so it gets
# a packed frame: counts and one BIT per motor. The app labels them from its mirror of
# MODULES (J1..Jn and so on) — names are the one thing worth dropping to make it fit.

def encode_health(rows: list) -> bytes:
    """module_roster.roster() output -> packed HEALTH. Rows this build has no index for
    are skipped rather than guessed, and the frame is truncated to ONE write."""
    out = bytearray([HEALTH, 0])
    count = 0
    for r in rows:
        i = MODULE_INDEX.get(r.get("id"))
        if i is None:
            continue
        acts = (r.get("actuators") or [])[:MAX_ACTUATORS]
        n = len(acts)
        nb = (n + 7) // 8
        if len(out) + 3 + nb > MAX_PAYLOAD:
            break
        flags = ((HEALTH_HAS_DRIVER if r.get("n") else 0)
                 | (HEALTH_TORQUE if r.get("torque") else 0)
                 | (HEALTH_OK if r.get("ok") else 0))
        bm = bytearray(nb)
        for k, a in enumerate(acts):
            if a.get("ok"):
                bm[k // 8] |= 1 << (k % 8)
        out += bytes([i, n, flags]) + bytes(bm)
        count += 1
    out[1] = count
    return bytes(out)


def decode_health(frame: bytes) -> list[dict]:
    """packed HEALTH -> [{id, n, up, live: [bool], hasDriver, torque, ok}]. For the
    selftest and any Python client; the apps decode this in Swift."""
    if not frame or frame[0] != HEALTH or len(frame) < 2:
        raise BleProtocolError("not a HEALTH frame")
    count, o, out = frame[1], 2, []
    for _ in range(count):
        if o + 3 > len(frame):
            raise BleProtocolError("health: truncated header")
        idx, n, flags = frame[o], frame[o + 1], frame[o + 2]
        nb = (n + 7) // 8
        o += 3
        if o + nb > len(frame):
            raise BleProtocolError("health: truncated bitmap")
        bm, o = frame[o:o + nb], o + nb
        live = [bool(bm[k // 8] >> (k % 8) & 1) for k in range(n)]
        if idx >= len(MODULES):
            continue                                   # a module this build does not know
        out.append({"id": MODULES[idx], "n": n, "up": sum(live), "live": live,
                    "hasDriver": bool(flags & HEALTH_HAS_DRIVER),
                    "torque": bool(flags & HEALTH_TORQUE), "ok": bool(flags & HEALTH_OK)})
    return out


def encode_teleop_frame(vx: float, vy: float, wz: float, lift: float = 0.0) -> bytes:
    """Base twist plus the LIFT STICK, in one frame.

    The lift rides here rather than in a message of its own because the gateway reads the
    whole joystick from one frame: a lift-only frame would arrive with lx/ly/rx absent and
    zero the base twist that the previous frame had just set."""
    return bytes([TELEOP_FRAME]) + struct.pack("<ffff", vx, vy, wz, lift)
