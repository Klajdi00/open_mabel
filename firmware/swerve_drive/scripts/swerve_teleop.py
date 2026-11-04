#!/usr/bin/env python3
"""
Swerve-drive keyboard teleop — velocity-vector control with a 3-module motion model.

Drive a holonomic command (vx, vy, omega); the script runs swerve inverse
kinematics for the three modules (two at the front, one at the back centre),
ramps the command smoothly, and sends per-wheel velocity + steer-angle setpoints
over CAN via sparkcan_py.

Module / motor map  (robot frame: +X forward, +Y left — REP-103)
    fl  Left-Front   drive ID 3   steer ID 4
    fr  Right-Front  drive ID 1   steer ID 2
    bc  Back-Centre  drive ID 5   steer ID 6

Controls (mirrors simulation/.../mobile_base_teleop.py)
    ↑ / ↓        Drive forward / backward      (vx)
    ← / →        Strafe left / right           (vy)
    W / S        Rotate CCW / CW               (omega)
    I / K        Max linear speed  +/- 0.1 m/s
    O / L        Max angular speed +/- 0.2 rad/s
    SPACE        Immediate stop (zero command)
    Q / Ctrl-C   Quit — drives zeroed on exit

Hold a key to move; release and the command ramps back to zero after a short
timeout (terminals give no key-up event, so movement auto-stops when idle).

SAFETY: the real top speed is enforced in hardware — run scripts/set_drive_limits.py
once to burn a ±10% output cap into the drive controllers. This script only ramps
and bounds the *command*; it does not raise that hardware ceiling.

Run:
    python scripts/swerve_teleop.py
"""

import argparse
import atexit
import math
import os
import select
import sys
import termios
import time
import tty
from typing import Optional

# ── interface / loop ────────────────────────────────────────────────────────────
CAN_IF  = "can0"
LOOP_HZ = 50
DRY     = False   # --dry-run: compute IK + draw, but send nothing on the bus

# ── module geometry (metres, robot frame +X fwd / +Y left) ──────────────────────
# Matches mabel_base_hardware/swerve_node.py: R = 0.30, front wheels at ±30°.
R = 0.30
MODULES = {
    "fl": dict(drive=3, steer=4, x=R * math.cos(math.radians(30)),  y=R * math.sin(math.radians(30))),
    "fr": dict(drive=1, steer=2, x=R * math.cos(math.radians(-30)), y=R * math.sin(math.radians(-30))),
    "bc": dict(drive=5, steer=6, x=-R,                              y=0.0),
}

# Per-module steer zero: the encoder value (rotations) at which the wheel points
# straight forward. Added to the commanded heading -> steer_set = (heading+offset)%1.
# Measured on the robot (encoder-0 = straight only for fr):
#   fr  straight at encoder 0          -> 0.0
#   fl  encoder-0 is 90° off straight  -> 0.25   (90°)
#   bc  module angled 45°, 135° off    -> 0.375  (135°)
# If a wheel points BACKWARD/sideways in --dry-run when you command forward, this
# module's offset sign is flipped — use 1-offset instead (fl 0.75, bc 0.625). If a
# module *steers mirrored* (turn left → it turns right), it needs a STEER_SIGN; ask.
STEER_OFFSET = {"fl": 0.25, "fr": 0.0, "bc": 0.375}   # rotations
DRIVE_SIGN   = {"fl": +1.0, "fr": +1.0, "bc": +1.0}

# ── drive conversion: wheel linear speed (m/s) → drive-motor RPM ────────────────
WHEEL_RADIUS     = 0.05    # m   (mabel_base_hardware/mujoco_node.py)
DRIVE_GEAR_RATIO = 1.0     # motor rev : wheel rev — SET to your module's ratio
WHEEL_CIRC       = 2.0 * math.pi * WHEEL_RADIUS

def mps_to_motor_rpm(v: float) -> float:
    return (v / WHEEL_CIRC) * 60.0 * DRIVE_GEAR_RATIO

# ── command limits & smoothing ──────────────────────────────────────────────────
MAX_LIN_SPEED = 0.5     # m/s   at full stick (live-tune with I/K)
MAX_ANG_SPEED = 1.5     # rad/s at full stick (live-tune with O/L)
MAX_LIN_ACCEL = 1.5     # m/s^2   ramp rate
MAX_ANG_ACCEL = 4.0     # rad/s^2 ramp rate
MAX_DRIVE_RPM = 300.0   # software clamp; hardware ±10% cap is the real ceiling
HOLD_TIMEOUT  = 0.30    # s   command decays to 0 if no key seen within this window

# Anti-slip: don't drive a wheel until it is within this heading error (rotations).
ALIGN_ERR_ROT = math.radians(30) / (2 * math.pi)   # ~30°

DRIVE_IDS = {m["drive"] for m in MODULES.values()}
STEER_IDS = {m["steer"] for m in MODULES.values()}
ALL_IDS   = DRIVE_IDS | STEER_IDS
NAME = {1: "RF drive", 2: "RF steer", 3: "LF drive",
        4: "LF steer", 5: "B  drive", 6: "B  steer"}

# ── ANSI ───────────────────────────────────────────────────────────────────────
_CLR = "\033[2J\033[H"; _HIDE = "\033[?25l"; _SHOW = "\033[?25h"
_RST = "\033[0m"; _BLD = "\033[1m"; _DIM = "\033[2m"
_GRN = "\033[32m"; _RED = "\033[31m"; _YLW = "\033[33m"; _CYN = "\033[36m"

def b(s):  return f"{_BLD}{s}{_RST}"
def g(s):  return f"{_GRN}{s}{_RST}"
def r(s):  return f"{_RED}{s}{_RST}"
def y(s):  return f"{_YLW}{s}{_RST}"
def c(s):  return f"{_CYN}{s}{_RST}"
def d(s):  return f"{_DIM}{s}{_RST}"

# ── raw terminal ────────────────────────────────────────────────────────────────
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
    if ch == b" ":
        return "SPACE"
    if ch in (b"q", b"Q", b"\x03"):
        return "QUIT"
    try:
        return ch.decode("ascii").lower()
    except Exception:
        return None


def _sniff(ifname: str, duration: float) -> set:
    import socket, struct
    FMT = "=IB3x8s"; SZ = struct.calcsize(FMT)
    sock = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
    sock.bind((ifname,))
    sock.settimeout(0.05)
    seen: set = set()
    t_end = time.monotonic() + duration
    while time.monotonic() < t_end:
        try:
            data = sock.recv(SZ)
        except (socket.timeout, BlockingIOError):
            continue
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
    sock.close()
    return seen


# ── swerve kinematics ───────────────────────────────────────────────────────────
def module_command(mod: dict, vx: float, vy: float, omega: float, last_rot: float):
    """
    Inverse kinematics for one module.

    Returns (steer_rot, motor_rpm) where steer_rot is the *continuous* commanded
    wheel heading in rotations (carry it back in as last_rot next cycle so the
    optimiser unwraps correctly and never spins the steer column more than 90°).
    """
    v_ix = vx - omega * mod["y"]
    v_iy = vy + omega * mod["x"]
    speed = math.hypot(v_ix, v_iy)            # m/s at the contact patch

    if speed < 1e-3:                          # idle: hold heading, no drive
        return last_rot, 0.0

    heading_rot = math.atan2(v_iy, v_ix) / (2.0 * math.pi)   # [-0.5, 0.5]

    # shortest angular move from last commanded heading, wrapped to (-0.5, 0.5]
    delta = (heading_rot - last_rot + 0.5) % 1.0 - 0.5
    flip = 1.0
    if delta > 0.25:                          # >90°: flip 180° and reverse wheel
        delta -= 0.5
        flip = -1.0
    elif delta < -0.25:
        delta += 0.5
        flip = -1.0

    new_rot = last_rot + delta

    # anti-slip: only spin the wheel once it is roughly aimed
    err = abs(delta)
    drive_scale = 0.0 if err > ALIGN_ERR_ROT else math.cos(delta * 2.0 * math.pi)

    rpm = mps_to_motor_rpm(speed) * flip * drive_scale * DRIVE_SIGN_FOR[mod["drive"]]
    rpm = max(-MAX_DRIVE_RPM, min(MAX_DRIVE_RPM, rpm))
    return new_rot, rpm


# drive_sign indexed by drive motor id, for quick lookup in module_command
DRIVE_SIGN_FOR = {MODULES[k]["drive"]: DRIVE_SIGN[k] for k in MODULES}


def _ramp(cur: float, tgt: float, max_step: float) -> float:
    if cur < tgt:
        return min(tgt, cur + max_step)
    if cur > tgt:
        return max(tgt, cur - max_step)
    return cur


def _bar(val, lo, hi, width=14):
    frac = (val - lo) / (hi - lo) if hi > lo else 0.5
    frac = max(0.0, min(1.0, frac))
    pos  = int(round(frac * (width - 1)))
    zero = int(round(max(0.0, min(1.0, (0 - lo) / (hi - lo))) * (width - 1)))
    ch = [" "] * width
    ch[zero] = "|"
    if pos != zero:
        for i in range(min(pos, zero) + 1, max(pos, zero)):
            ch[i] = "-"
        ch[pos] = "<" if val < 0 else ">"
    return "[" + "".join(ch) + "]"


def _draw(seen, tgt, cur, mod_state, telem, notice):
    vxform = lambda v: f"{v:+.2f}"
    out = [_CLR]
    out.append(f"  {b('Swerve Teleop')}{r(' [DRY RUN]') if DRY else ''}   "
               + d(f"{CAN_IF}  {LOOP_HZ}Hz   "
               f"vmax {MAX_LIN_SPEED:.1f} m/s   wmax {MAX_ANG_SPEED:.1f} rad/s   "
               "HW cap ±10%"))
    out.append("  " + "-" * 78)
    out.append(f"  {c('arrows')} drive/strafe   {c('W/S')} rotate   "
               f"{c('I/K')} v±   {c('O/L')} w±   {c('SPACE')} stop   {c('Q')} quit")
    out.append("  " + "-" * 78)
    out.append(
        f"  vx {b(vxform(cur[0]))}{d('/'+vxform(tgt[0]))}  "
        f"vy {b(vxform(cur[1]))}{d('/'+vxform(tgt[1]))} m/s   "
        f"ω {b(vxform(cur[2]))}{d('/'+vxform(tgt[2]))} rad/s   "
        f"{d('(cur/target)')}"
    )
    out.append("  " + "-" * 78)
    out.append(f"  {'mod':<4}{'drive':<6}{'steer':<6}{'cmd RPM':>9}  "
               f"{'fb RPM':>8}   {'steer→':>7} {'fb rot':>7}")
    out.append("  " + "-" * 78)
    for k in ("fl", "fr", "bc"):
        m = MODULES[k]
        di, si = m["drive"], m["steer"]
        scmd, rpm = mod_state[k]
        steer_set = (scmd + STEER_OFFSET[k]) % 1.0
        on = di in seen and si in seen
        dot = g("●") if on else r("○")
        fb_rpm = telem.get(di, ("--",))[-1]
        fb_rot = telem.get(si, ("--",))[-1]
        fb_rpm_s = f"{fb_rpm:+8.1f}" if isinstance(fb_rpm, float) else f"{'--':>8}"
        fb_rot_s = f"{fb_rot:7.3f}"  if isinstance(fb_rot, float) else f"{'--':>7}"
        out.append(f"  {dot}{k:<3}{di:<6}{si:<6}{rpm:>9.1f}  {fb_rpm_s}   "
                   f"{steer_set:>7.3f} {fb_rot_s}")
    out.append("  " + "-" * 78)
    spd = math.hypot(cur[0], cur[1])
    out.append(f"  speed {b(f'{spd:.2f} m/s')} {_bar(spd, 0, MAX_LIN_SPEED)}   "
               f"ω {_bar(cur[2], -MAX_ANG_SPEED, MAX_ANG_SPEED)}")
    if notice:
        out.append("\n  >> " + y(notice))
    sys.stdout.write("\n".join(out))
    sys.stdout.flush()


def main():
    global MAX_LIN_SPEED, MAX_ANG_SPEED, CAN_IF, DRY
    ap = argparse.ArgumentParser(description="3-module swerve keyboard teleop.")
    ap.add_argument("--iface", default=CAN_IF, help="CAN interface (default can0)")
    ap.add_argument("--dry-run", action="store_true",
                    help="compute IK + draw the dashboard but send NOTHING on the bus "
                         "(no setpoints, no heartbeats) — motors stay disabled")
    args = ap.parse_args()
    CAN_IF = args.iface
    DRY = args.dry_run

    try:
        from sparkcan_py import SparkFlex
    except ImportError:
        print("ERROR: sparkcan_py not importable. Activate the env that has it "
              "(e.g. conda activate yor).")
        return 1

    print(f"Sniffing {CAN_IF} for 1 s …", end="", flush=True)
    seen = _sniff(CAN_IF, 1.0)
    print(f" present: {sorted(seen & ALL_IDS) or 'none'}  "
          f"missing: {sorted(ALL_IDS - seen) or 'none'}")

    print("Connecting …")
    motors = {}
    for cid in sorted(ALL_IDS):
        try:
            motors[cid] = SparkFlex(CAN_IF, cid)
            print(f"  ID {cid} ({NAME[cid]}): OK"
                  + ("" if cid in seen else d("  (not on bus)")))
        except Exception as e:
            print(f"  ID {cid} ({NAME[cid]}): FAILED — {e}")
        time.sleep(0.003)
    if not any(cid in motors for cid in DRIVE_IDS):
        print("No drive motors connected. Exiting.")
        return 1

    # warm up heartbeats (no setpoints → no motion), then seed steer state from
    # the live encoder so wheels don't snap to zero on the first command.
    if not DRY:
        for _ in range(10):
            for m in motors.values():
                try: m.Heartbeat()
                except Exception: pass
            time.sleep(0.02)

    mod_last = {}
    for k, m in MODULES.items():
        p = 0.0
        if m["steer"] in motors:
            try:
                p = float(motors[m["steer"]].GetAbsoluteEncoderPosition())
            except Exception:
                p = 0.0
        mod_last[k] = p - STEER_OFFSET[k]      # continuous rotations

    # ── runtime state ────────────────────────────────────────────────────────
    cur = [0.0, 0.0, 0.0]                       # current vx, vy, omega (ramped)
    last_press = {k: -1e9 for k in ("UP", "DOWN", "LEFT", "RIGHT", "w", "s")}
    notice = ""

    atexit.register(_go_cooked)
    atexit.register(lambda: sys.stdout.write(_SHOW))
    _go_raw()
    sys.stdout.write(_HIDE)
    dt = 1.0 / LOOP_HZ

    try:
        while True:
            t0 = time.monotonic()
            key = _read_key(timeout=dt)

            if key == "QUIT":
                break
            elif key == "SPACE":
                cur = [0.0, 0.0, 0.0]
                for kk in last_press:
                    last_press[kk] = -1e9
                notice = "STOP"
            elif key in last_press:
                last_press[key] = t0
                notice = ""
            elif key == "i":
                MAX_LIN_SPEED = min(3.0, MAX_LIN_SPEED + 0.1); notice = f"vmax {MAX_LIN_SPEED:.1f}"
            elif key == "k":
                MAX_LIN_SPEED = max(0.1, MAX_LIN_SPEED - 0.1); notice = f"vmax {MAX_LIN_SPEED:.1f}"
            elif key == "o":
                MAX_ANG_SPEED = min(5.0, MAX_ANG_SPEED + 0.2); notice = f"wmax {MAX_ANG_SPEED:.1f}"
            elif key == "l":
                MAX_ANG_SPEED = max(0.2, MAX_ANG_SPEED - 0.2); notice = f"wmax {MAX_ANG_SPEED:.1f}"

            # targets from currently-held (recently-seen) keys; auto-zero on idle
            alive = lambda kk: (t0 - last_press[kk]) < HOLD_TIMEOUT
            tgt = [
                (MAX_LIN_SPEED if alive("UP")   else 0.0) - (MAX_LIN_SPEED if alive("DOWN")  else 0.0),
                (MAX_LIN_SPEED if alive("LEFT") else 0.0) - (MAX_LIN_SPEED if alive("RIGHT") else 0.0),
                (MAX_ANG_SPEED if alive("w")    else 0.0) - (MAX_ANG_SPEED if alive("s")     else 0.0),
            ]

            # smooth ramp toward target
            cur[0] = _ramp(cur[0], tgt[0], MAX_LIN_ACCEL * dt)
            cur[1] = _ramp(cur[1], tgt[1], MAX_LIN_ACCEL * dt)
            cur[2] = _ramp(cur[2], tgt[2], MAX_ANG_ACCEL * dt)

            # inverse kinematics → per-module setpoints
            mod_state = {}
            for k, m in MODULES.items():
                new_rot, rpm = module_command(m, cur[0], cur[1], cur[2], mod_last[k])
                mod_last[k] = new_rot
                mod_state[k] = (new_rot, rpm)

            # send commands + heartbeat (present motors only) — skipped in --dry-run
            if not DRY:
                for k, m in MODULES.items():
                    new_rot, rpm = mod_state[k]
                    steer_set = (new_rot + STEER_OFFSET[k]) % 1.0
                    if m["drive"] in motors and m["drive"] in seen:
                        try:
                            motors[m["drive"]].SetVelocity(float(rpm))
                        except Exception: pass
                    if m["steer"] in motors and m["steer"] in seen:
                        try:
                            motors[m["steer"]].SetPosition(float(steer_set))
                        except Exception: pass
                for cid, m in motors.items():
                    try: m.Heartbeat()
                    except Exception: pass

            # telemetry (best-effort)
            telem = {}
            for cid, m in motors.items():
                if cid not in seen:
                    continue
                try:
                    telem[cid] = ("v", float(m.GetVelocity())) if cid in DRIVE_IDS \
                                 else ("p", float(m.GetAbsoluteEncoderPosition()))
                except Exception:
                    telem[cid] = ("e", "--")

            _draw(seen, tgt, cur, mod_state, telem, notice)

            leftover = dt - (time.monotonic() - t0)
            if leftover > 0.001:
                time.sleep(leftover)

    finally:
        # zero drives, hold steer, a few heartbeats so the stop is honoured
        if not DRY:
            for _ in range(5):
                for k, m in MODULES.items():
                    try:
                        if m["drive"] in motors:
                            motors[m["drive"]].SetVelocity(0.0)
                        if m["steer"] in motors:
                            steer_set = (mod_last[k] + STEER_OFFSET[k]) % 1.0
                            motors[m["steer"]].SetPosition(float(steer_set))
                    except Exception: pass
                for m in motors.values():
                    try: m.Heartbeat()
                    except Exception: pass
                time.sleep(0.02)
        _go_cooked()
        sys.stdout.write(_SHOW + "\n")
        print("Drives zeroed. Goodbye.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
