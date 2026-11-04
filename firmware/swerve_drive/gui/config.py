"""Static configuration for the swerve drive: module layout, CAN, control limits.

Edit ``MODULES`` to match your wiring.  Everything else in the GUI is driven
off this table, so adding a fifth module or re-mapping a CAN ID never touches
UI code.
"""

import math
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
GUI_DIR   = Path(__file__).resolve().parent
REPO_ROOT = GUI_DIR.parent
CAN_BRINGUP = REPO_ROOT / "scripts" / "setup" / "can_bringup.sh"

# ── CAN bus ───────────────────────────────────────────────────────────────────
CAN_IFACE   = "can0"
CAN_BITRATE = 1_000_000

# ── Control loop rates ────────────────────────────────────────────────────────
HEARTBEAT_HZ = 50      # SPARK controllers disable without a steady heartbeat
SEND_HZ      = 40      # setpoint push rate
POLL_HZ      = 12      # telemetry read rate

# ── Periodic status-frame broadcast periods (milliseconds) ────────────────────
# The SPARK controllers don't transmit these frames until we ask them to, which
# is why velocity (frame 2) and the steer absolute encoder (frame 5) read blank.
# Enabled on connect. Lower = fresher telemetry, more bus traffic.
#   frame 0: applied output, temperature, voltage, current
#   frame 2: primary-encoder position + velocity   (drive wheel velocity)
#   frame 5: absolute-encoder position + velocity   (steer angle)
STATUS0_PERIOD_MS = 50
STATUS2_PERIOD_MS = 20
STATUS5_PERIOD_MS = 20

# ── Motor roles ───────────────────────────────────────────────────────────────
DRIVE = "drive"        # the road wheel
STEER = "steer"        # closed-loop position (rotations), absolute encoder

# Drive control mode.
#   Open-loop duty-cycle (throttle) is the default: it has NO feedback loop, so
#   the controller cannot oscillate — this is what stops drive motors 1 & 3 from
#   shaking violently the way they do under closed-loop SetVelocity with an
#   untuned velocity PID. Flip DRIVE_USE_DUTY to False only after you've tuned
#   the velocity PID on the controllers (SetP/I/D + BurnFlash).
DRIVE_USE_DUTY = True
DRIVE_DUTY_MAX = 0.40          # slider range ±, as a fraction of full output (0..1)
DRIVE_SLEW     = 1.2           # max duty change per second (ramp; smooths starts)
DRIVE_VEL_MAX  = 4.0           # used only when DRIVE_USE_DUTY is False (SetVelocity units)

# Position command range for steer motors (rotations of the module, 0..1).
STEER_POS_MIN, STEER_POS_MAX = 0.0, 1.0

# ── Steering angle display ────────────────────────────────────────────────────
# Like the YOR robot, the GUI shows BOTH the commanded angle (always live, since
# the SPARK closes the steering loop internally on the absolute encoder) AND the
# encoder readback (best-effort, only when status frame 5 is broadcasting).
STEER_DEG_PER_ROT = 360.0      # 1 module rotation = 360°, for human-readable display
# What unit GetAbsoluteEncoderPosition reports. YOR's controllers report degrees;
# ours command in rotations (0..1). If the encoder readback looks 360x off, flip.
ENCODER_IS_DEGREES = False     # False → value is rotations (×360 for degrees)

# A frame-2/5 telemetry value is considered STALE (frame not being broadcast) if
# it hasn't changed in this many seconds. Temperature (frame 0) is always live,
# so a stale velocity/encoder means "frame off", not "bus down".
TELEMETRY_STALE_SEC = 2.0

# ── Closed-loop control type (applied on connect, parity with YOR) ────────────
# Drive uses open-loop duty by default (DRIVE_USE_DUTY) so its ctrl type is set
# to match; steer is always position so the controller servos the encoder.
APPLY_CTRL_TYPE = True
SET_IDLE_COAST  = True

# ── Robot geometry & drive conversion (from scripts/swerve_teleop.py) ─────────
# Modules sit on a triangle of radius R; front wheels at ±30°, back at 180°.
R_BASE       = 0.30        # m, module distance from robot center
WHEEL_RADIUS = 0.05        # m
DRIVE_GEAR_RATIO = 1.0     # motor rev : wheel rev — set to your module's ratio
WHEEL_CIRC   = 2.0 * math.pi * WHEEL_RADIUS


def mps_to_motor_rpm(v_mps: float) -> float:
    """Wheel linear speed (m/s) → drive-motor RPM (the SetVelocity unit)."""
    return (v_mps / WHEEL_CIRC) * 60.0 * DRIVE_GEAR_RATIO


def motor_rpm_to_mps(rpm: float) -> float:
    """Drive-motor RPM (GetVelocity feedback) → wheel linear speed (m/s)."""
    return (rpm / 60.0) * WHEEL_CIRC / DRIVE_GEAR_RATIO


# ── Swerve module map ─────────────────────────────────────────────────────────
# Three modules: two at the front (left/right) and one at the back.
# Odd IDs (1, 3, 5) are the DRIVE wheels; even IDs (2, 4, 6) are the STEERING
# axes.  ``steer_offset`` is the absolute-encoder zero offset added to every
# steering setpoint.  (x, y) is the module position from robot center, metres
# (x forward, y left), used by the motion model (gui/motion.py).
MODULES = [
    {"name": "Front Left",  "abbr": "FL", "drive_id": 1, "steer_id": 2, "steer_offset": 0.00,
     "x": R_BASE * math.cos(math.radians(30)),  "y": R_BASE * math.sin(math.radians(30))},
    {"name": "Front Right", "abbr": "FR", "drive_id": 3, "steer_id": 4, "steer_offset": 0.00,
     "x": R_BASE * math.cos(math.radians(-30)), "y": R_BASE * math.sin(math.radians(-30))},
    {"name": "Back",        "abbr": "B",  "drive_id": 5, "steer_id": 6, "steer_offset": 0.00,
     "x": -R_BASE, "y": 0.0},
]


def all_motor_ids():
    """Flat, ordered list of every CAN ID the GUI manages."""
    ids = []
    for m in MODULES:
        ids.extend([m["drive_id"], m["steer_id"]])
    return ids


def motor_roles():
    """Mapping ``{can_id: role}`` for drive/steer classification."""
    roles = {}
    for m in MODULES:
        roles[m["drive_id"]] = DRIVE
        roles[m["steer_id"]] = STEER
    return roles


def steer_offsets():
    """Mapping ``{steer_id: offset}``."""
    return {m["steer_id"]: m["steer_offset"] for m in MODULES}


def cmd_to_deg(setpoint_frac):
    """Commanded steer fraction (0..1 rotations) → module angle in degrees."""
    return (float(setpoint_frac) % 1.0) * STEER_DEG_PER_ROT


def encoder_to_deg(steer_id, raw):
    """Raw absolute-encoder reading → module angle in degrees, aligned to the
    commanded frame (subtracts the per-module zero offset so cmd≈fb on track)."""
    rot = (raw / STEER_DEG_PER_ROT) if ENCODER_IS_DEGREES else raw
    off = steer_offsets().get(steer_id, 0.0)
    return ((float(rot) - off) % 1.0) * STEER_DEG_PER_ROT
