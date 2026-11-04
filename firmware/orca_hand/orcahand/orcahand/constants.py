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
INST_SYNC_READ = 0x82
INST_SYNC_WRITE = 0x83
BROADCAST_ID = 0xFE

# --- Encoder / unit conversion ----------------------------------------------
TICKS_PER_REV = 4096.0
TICK_TO_RAD = (2.0 * math.pi) / TICKS_PER_REV
RAW_MIN = 0
RAW_MAX = 4095
RAW_CENTER = 2047

# --- Bus / motion defaults --------------------------------------------------
DEFAULT_BAUDRATE = 1_000_000
DEFAULT_TORQUE_LIMIT = 500     # 0-1000, ~50 % of motor ceiling
DEFAULT_GOAL_SPEED = 2000      # ticks/s; 0 = max
DEFAULT_GOAL_ACC = 40          # acceleration units
DEFAULT_PUBLISH_RATE = 50.0    # Hz, feedback polling

# USB-serial adapter vendor IDs we ship with (CH340, CP210x).
USB_VENDOR_IDS = (0x1A86, 0x10C4)
