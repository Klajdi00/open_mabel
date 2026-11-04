# main.py - Lift firmware entry point.
#
# Boots the hardware, then runs a fixed-rate cooperative loop:
#   1. drain any serial command
#   2. tick the closed-loop controller at CONTROL_HZ
#   3. emit telemetry at the configured rate
#
# All real logic lives in the modules; this file is just the wiring + loop.

import sys
import utime
try:
    import uselect as select
except ImportError:
    import select

import config
from lift import Lift
from protocol import CommandConsole


# ----- non-blocking line reader -------------------------------------------
_poll = select.poll()
_poll.register(sys.stdin, select.POLLIN)
_buf = []

# Characters drained per pass. The loop used to read exactly ONE character per
# iteration, and an iteration costs at least the 1 ms sleep below — about 1000
# characters a second, total. A host streaming "pos 612.34 40\n" at 50 Hz sends
# ~750, leaving almost no margin: any jitter and the input backs up, so commands
# are acted on later and later and the column lags behind the setpoint. Draining
# the buffer decouples the command rate from the loop rate. The cap keeps a
# flood from starving the control tick.
_MAX_CHARS_PER_PASS = 256


def read_lines():
    """Return every COMPLETE line available right now (oldest first)."""
    lines = []
    for _ in range(_MAX_CHARS_PER_PASS):
        if not _poll.poll(0):
            break
        ch = sys.stdin.read(1)
        if not ch or ch == "\r":
            continue
        if ch == "\n":
            lines.append("".join(_buf))
            _buf.clear()
            continue
        if ch in ("\x08", "\x7f"):     # backspace / delete
            if _buf:
                _buf.pop()
            continue
        _buf.append(ch)
    return lines


def main():
    lift = Lift()
    console = CommandConsole(lift)

    control_dt_ms = int(1000 / config.CONTROL_HZ)
    last_control = utime.ticks_ms()
    last_telem = last_control

    print("Lift firmware ready. Cascaded position/velocity control.")
    print("Type 'help' for commands. Run 'calibrate' before first use.")

    try:
        while True:
            now = utime.ticks_ms()

            for line in read_lines():
                console.handle(line)

            # Fixed-rate control tick.
            if utime.ticks_diff(now, last_control) >= control_dt_ms:
                dt = utime.ticks_diff(now, last_control) / 1000.0
                last_control = now
                lift.update(now, dt)

            # Telemetry.
            telem_dt_ms = int(1000 / console.telemetry_hz)
            if utime.ticks_diff(now, last_telem) >= telem_dt_ms:
                last_telem = now
                if console.json_stream:
                    print(console.telemetry_json())

            utime.sleep_ms(1)
    finally:
        # Always cut motor power on exit (Ctrl-C, soft reset, or any error)
        # so the lift never runs away when the control loop stops.
        lift.motor.disable()
        print("Lift firmware stopped; motor disabled.")


main()
