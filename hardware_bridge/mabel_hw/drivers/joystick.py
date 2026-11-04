"""joystick.py — minimal DualSense (PS5) reader, extracted from ps5_teleop.py.

Zero-dependency reader for the legacy Linux joystick API (/dev/input/js*). Lets the
hardware bridge offer the SAME direct PS5 teleop as the validated ps5_teleop.py,
with no ROS in the loop — drive the base off a pad before the ROS stack is up.

Axis/button indices and stick->twist signs are the ps5_teleop values; run
``python -m base.joystick --probe`` to confirm them on a given pad.
"""

import argparse
import struct
import threading
import time

# DualSense (legacy /dev/input/js* API) axis/button indices.  These vary by driver;
# run with --probe to confirm on your machine, then adjust if needed.
AX_LX, AX_LY = 0, 1   # left stick  -> translation
AX_RX = 5             # right stick X -> yaw
BTN_DEADMAN = 4       # hold to enable motion (L1 on most mappings)

DEADZONE = 0.08       # joystick deadzone (fraction of full scale)


class DualSense:
    """Minimal non-blocking reader for /dev/input/js* (no external libraries).

    A background thread parses 8-byte js_event records and keeps the latest axis
    values (normalized to [-1, 1]) and button states. Thread-safe getters apply a
    deadzone to axes.
    """

    _FMT = "IhBB"                       # u32 time, s16 value, u8 type, u8 number
    _SIZE = struct.calcsize(_FMT)
    _AXIS, _BUTTON, _INIT = 0x02, 0x01, 0x80

    def __init__(self, dev="/dev/input/js0", deadzone=DEADZONE):
        self.dev = dev
        self.deadzone = deadzone
        self._axes = {}
        self._buttons = {}
        self._lock = threading.Lock()
        self._running = False
        self._fd = None
        self._thread = None

    def start(self):
        self._fd = open(self.dev, "rb")
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _loop(self):
        while self._running:
            try:
                data = self._fd.read(self._SIZE)
            except (OSError, ValueError):
                break
            if not data or len(data) < self._SIZE:
                continue
            _t, value, etype, number = struct.unpack(self._FMT, data)
            etype &= ~self._INIT       # ignore the synthetic init flag
            with self._lock:
                if etype == self._AXIS:
                    self._axes[number] = max(-1.0, min(1.0, value / 32767.0))
                elif etype == self._BUTTON:
                    self._buttons[number] = value

    def axis(self, number):
        with self._lock:
            v = self._axes.get(number, 0.0)
        return 0.0 if abs(v) < self.deadzone else v

    def button(self, number):
        with self._lock:
            return bool(self._buttons.get(number, 0))

    def snapshot(self):
        with self._lock:
            return dict(self._axes), dict(self._buttons)

    def stop(self):
        self._running = False
        if self._fd:
            self._fd.close()


def probe(dev):
    """Print live axis/button activity so you can confirm index mappings."""
    js = DualSense(dev, deadzone=0.0).start()
    print(f"Reading {dev}. Move sticks / press buttons. Ctrl-C to quit.")
    try:
        while True:
            axes, buttons = js.snapshot()
            ax = " ".join(f"a{n}:{v:+.2f}" for n, v in sorted(axes.items()))
            bt = " ".join(f"b{n}:{v}" for n, v in sorted(buttons.items()) if v)
            print(f"\r{ax}   {bt}        ", end="", flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print()
    finally:
        js.stop()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="DualSense probe")
    p.add_argument("--dev", default="/dev/input/js0", help="joystick device")
    p.add_argument("--probe", action="store_true", help="print axis/button indices")
    args = p.parse_args()
    probe(args.dev)
