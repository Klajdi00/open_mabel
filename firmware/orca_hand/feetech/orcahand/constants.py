"""Low-level constants for the Feetech STS/SCS bus and the ORCA hand.

Single place for every protocol address, instruction byte, and tuning default.
Nothing in here knows about joints or radians -- it is pure hardware vocabulary.
"""

import math

# --- Feetech STS/SCS control-table addresses --------------------------------
ADDR_TORQUE_ENABLE = 0x28      # 1 byte : 0 = free, 1 = hold
ADDR_GOAL_ACC = 0x29           # 1 byte : goal acceleration
ADDR_GOAL_POSITION = 0x2A      # 2 bytes: goal position (ticks)
ADDR_GOAL_SPEED = 0x2E         # 2 bytes: goal speed cap (ticks/s)
ADDR_TORQUE_LIMIT = 0x30       # 2 bytes: torque ceiling (0-1000)
ADDR_PRESENT_POSITION = 0x38   # start of the 8-byte feedback block

# Feedback block layout (8 bytes starting at ADDR_PRESENT_POSITION):
#   [0:2] position   [2:4] speed   [4:6] load   [6] voltage(0.1V)   [7] temp(C)
FEEDBACK_LEN = 8

# --- Protocol instruction bytes ---------------------------------------------
INST_PING = 0x01
INST_READ = 0x02
INST_WRITE = 0x03
INST_OFSCAL = 0x0B             # offset-calibration: set CURRENT pose as a given tick
INST_SYNC_READ = 0x82
INST_SYNC_WRITE = 0x83
BROADCAST_ID = 0xFE

# Register addresses used by EEPROM-config / re-zero flows.
ADDR_LOCK = 55                 # 1 byte: 0 = unlock EEPROM, 1 = lock

# --- Encoder / unit conversion ----------------------------------------------
TICKS_PER_REV = 4096.0
TICK_TO_RAD = (2.0 * math.pi) / TICKS_PER_REV
RAW_MIN = 0
RAW_MAX = 4095
RAW_CENTER = 2047
WRAP = 4096                        # single-turn encoder modulus (ticks per turn)


# --- Single-turn wrap handling ----------------------------------------------
# The STS/SCS encoder is a *single-turn absolute* device: it reports 0-4095 and
# rolls straight over (4095 -> 0). A joint whose travel happens to straddle that
# 0/4095 seam reads discontinuously and its min<max math breaks. Because every
# joint's travel is well under one turn (< WRAP ticks), we can always relocate
# the seam in *software* with a per-joint ``offset`` so the travel sits in a
# contiguous window -- no hardware "set home" needed. The calibration then stores
# min/max/center in this *corrected* frame; only a joint that actually straddles
# the seam gets a non-zero offset (everything else keeps offset == 0 and behaves
# byte-identically to the legacy raw mapping).
def wrapped_delta(d: int) -> int:
    """Shortest signed tick delta across the seam, in ``[-2048, 2048)``.

    Used to integrate a continuous (unwrapped) position from successive raw
    reads: as long as the joint moves < half a turn between samples (always true
    at the poll rate), ``+= wrapped_delta(raw - prev)`` tracks true motion across
    the 0/4095 boundary instead of jumping a whole turn.
    """
    return ((int(d) + WRAP // 2) % WRAP) - WRAP // 2


def apply_offset(raw: int, offset: int = 0) -> int:
    """Raw encoder tick -> corrected tick in the joint's offset frame (0-4095)."""
    return (int(raw) - int(offset)) % WRAP


def remove_offset(corrected: int, offset: int = 0) -> int:
    """Corrected tick -> raw encoder tick to command the servo (0-4095)."""
    return (int(corrected) + int(offset)) % WRAP

# --- Bus / motion defaults --------------------------------------------------
DEFAULT_BAUDRATE = 1_000_000
DEFAULT_TORQUE_LIMIT = 500     # 0-1000, ~50 % of motor ceiling
DEFAULT_GOAL_SPEED = 2000      # ticks/s; 0 = max
DEFAULT_GOAL_ACC = 40          # acceleration units
DEFAULT_PUBLISH_RATE = 50.0    # Hz, feedback polling

# USB-serial adapter vendor IDs we ship with (CH340, CP210x).
USB_VENDOR_IDS = (0x1A86, 0x10C4)
