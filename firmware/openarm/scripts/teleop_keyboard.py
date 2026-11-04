#!/usr/bin/env python3
"""
teleop_keyboard.py — DaMiao multi-motor keyboard teleop (MABEL openarm)

Usage
-----
  python scripts/teleop_keyboard.py
  python scripts/teleop_keyboard.py --channel can1

Key bindings
------------
  Enter       Type a motor ID to select or add it
  Tab / N     Cycle to next motor
  S           Scan CAN bus for motors
  E           Enable active motor
  D           Disable active motor
  C           Clear faults on active motor
  Z           Zero active motor (must be disabled first)
  ← →         Move target position by ±step
  ↑ ↓         Double / halve step size
  [ ]         Kp  −1 / +1
  { }         Kd  −0.1 / +0.1
  Q / Ctrl-C  Disable all motors and quit
"""
import argparse, can, curses, sys, threading, time

# ── Motor profiles (p_max rad, v_max rad/s, t_max Nm) ────────────────────────
PROFILES = {
    "DM-J4310":   dict(p_max=12.5,   v_max=30.0,  t_max=10.0),
    "DM-J4340":   dict(p_max=12.566, v_max=10.4,  t_max=27.0),
    "DM-J8009P":  dict(p_max=12.566, v_max=33.5,  t_max=40.0),
    "DM-J10422P": dict(p_max=12.566, v_max=12.5,  t_max=400.0),
    "DM-J3507":   dict(p_max=12.566, v_max=48.0,  t_max=3.0),
}
DEFAULT_LIM  = dict(p_max=12.566, v_max=30.0, t_max=40.0)
GLOBAL_ID    = 0x7FF
CTRL_HZ      = 50       # MIT command rate (Hz)
LEFT_W       = 24       # left panel width including border chars
MIN_COLS     = 82
MIN_ROWS     = 24

ERROR_NAMES  = {
    0x0: "OK",
    0x2: "ROTOR ENC", 0x3: "OUTPUT ENC",
    0x8: "OVER VOLT", 0x9: "UNDER VOLT",
    0xA: "OVER CURR", 0xB: "MOS TEMP",
    0xC: "COIL TEMP", 0xD: "COMM LOSS",
    0xE: "OVERLOAD",
}

# curses color pair numbers
_H  = 1   # header  — white bold on blue
_OK = 2   # green
_ER = 3   # red
_WN = 4   # yellow
_DM = 5   # dim
_SL = 6   # selection — black on cyan
_HN = 7   # hint — cyan


# ── Motor state container ─────────────────────────────────────────────────────
class Motor:
    __slots__ = ("mid", "lim", "pos", "vel", "tau", "t_mos", "t_rot",
                 "err", "enabled", "t_fb", "target", "step", "kp", "kd")

    def __init__(self, mid: int, lim: dict = None):
        self.mid     = mid
        self.lim     = lim or DEFAULT_LIM
        self.pos = self.vel = self.tau = 0.0
        self.t_mos = self.t_rot = 0.0
        self.err     = 0
        self.enabled = False
        self.t_fb    = 0.0
        self.target  = None   # latched on first feedback
        self.step    = 0.1
        self.kp      = 10.0
        self.kd      = 1.0

    def parse(self, d: bytes):
        self.err   = (d[0] >> 4) & 0xF
        p_int = (d[1] << 8) | d[2]
        v_int = (d[3] << 4) | (d[4] >> 4)
        t_int = ((d[4] & 0xF) << 8) | d[5]
        lim = self.lim
        self.pos = _u2f(p_int, -lim["p_max"], lim["p_max"], 16)
        self.vel = _u2f(v_int, -lim["v_max"], lim["v_max"], 12)
        self.tau = _u2f(t_int, -lim["t_max"], lim["t_max"], 12)
        self.t_mos = float(d[6]) if len(d) > 6 else 0.0
        self.t_rot = float(d[7]) if len(d) > 7 else 0.0
        self.t_fb  = time.perf_counter()
        if self.target is None:
            self.target = self.pos   # latch first seen pos as safe target

    @property
    def stale(self) -> bool:
        return self.t_fb > 0 and time.perf_counter() - self.t_fb > 0.5


def _u2f(x, lo, hi, bits):
    return x * (hi - lo) / ((1 << bits) - 1) + lo

def _f2u(x, lo, hi, bits):
    return int(max(0, min((1 << bits) - 1, (x - lo) * ((1 << bits) - 1) / (hi - lo))))


# ── CAN bus layer ─────────────────────────────────────────────────────────────
class Teleop:
    def __init__(self, channel: str):
        try:
            self.bus = can.interface.Bus(channel=channel, bustype="socketcan")
        except Exception as e:
            sys.exit(f"Cannot open {channel}: {e}\n"
                     "Run:  sudo ip link set can0 up type can bitrate 1000000")
        self.motors: dict[int, Motor] = {}
        self.lock = threading.Lock()
        self._stop = threading.Event()
        threading.Thread(target=self._rx_loop, daemon=True).start()

    def _rx_loop(self):
        while not self._stop.is_set():
            msg = self.bus.recv(timeout=0.5)
            if msg is None or msg.arbitration_id != 0 or len(msg.data) < 6:
                continue
            d = msg.data
            if d[2] == 0x33:                    # scan echo — register read
                mid = d[0] | (d[1] << 8)
                with self.lock:
                    if mid not in self.motors:
                        self.motors[mid] = Motor(mid)
            else:                               # MIT feedback
                mid = d[0] & 0x0F
                with self.lock:
                    if mid in self.motors:
                        self.motors[mid].parse(d)

    def scan(self):
        for mid in range(1, 16):
            self.bus.send(can.Message(
                arbitration_id=GLOBAL_ID,
                data=[mid & 0xFF, (mid >> 8) & 0xFF, 0x33, 0x00],
                is_extended_id=False))
            time.sleep(0.015)

    def _cmd(self, mid: int, byte: int):
        self.bus.send(can.Message(
            arbitration_id=mid, data=[0xFF]*7 + [byte], is_extended_id=False))

    def enable(self, mid: int):
        self._cmd(mid, 0xFC)
        with self.lock:
            if mid in self.motors:
                m = self.motors[mid]
                m.enabled = True
                if m.target is None:
                    m.target = m.pos

    def disable(self, mid: int):
        self._cmd(mid, 0xFD)
        with self.lock:
            if mid in self.motors:
                self.motors[mid].enabled = False

    def clear_faults(self, mid: int):
        self._cmd(mid, 0xFB)

    def set_zero(self, mid: int):
        self._cmd(mid, 0xFE)

    def send_mit(self, m: Motor):
        if m.target is None:
            return
        lim = m.lim
        pi = _f2u(m.target, -lim["p_max"], lim["p_max"], 16)
        vi = _f2u(0.0,      -lim["v_max"], lim["v_max"], 12)
        ki = _f2u(m.kp,     0.0,           500.0,        12)
        di = _f2u(m.kd,     0.0,           5.0,          12)
        ti = _f2u(0.0,      -lim["t_max"], lim["t_max"], 12)
        self.bus.send(can.Message(
            arbitration_id=m.mid,
            data=[(pi>>8)&0xFF, pi&0xFF,
                  (vi>>4)&0xFF, ((vi&0xF)<<4)|((ki>>8)&0xF), ki&0xFF,
                  (di>>4)&0xFF, ((di&0xF)<<4)|((ti>>8)&0xF), ti&0xFF],
            is_extended_id=False))

    def shutdown(self):
        self._stop.set()
        with self.lock:
            for mid in list(self.motors):
                try:
                    self.disable(mid)
                except Exception:
                    pass
        time.sleep(0.15)
        try:
            self.bus.shutdown()
        except Exception:
            pass


# ── curses UI ─────────────────────────────────────────────────────────────────
def _safe(win, r, c, text, attr=0):
    rows, cols = win.getmaxyx()
    if 0 <= r < rows and 0 <= c < cols - 1:
        try:
            win.addstr(r, c, str(text)[: cols - c - 1], attr)
        except curses.error:
            pass


def _prompt_id(stdscr, teleop: Teleop) -> int | None:
    rows, cols = stdscr.getmaxyx()
    prompt = "  Enter motor ID (1–15): "
    _safe(stdscr, rows - 1, 0, " " * (cols - 1))
    _safe(stdscr, rows - 1, 0, prompt, curses.color_pair(_WN) | curses.A_BOLD)
    curses.echo()
    curses.curs_set(1)
    stdscr.refresh()
    try:
        raw = stdscr.getstr(rows - 1, len(prompt), 4).decode().strip()
        mid = int(raw)
        if 1 <= mid <= 15:
            with teleop.lock:
                if mid not in teleop.motors:
                    teleop.motors[mid] = Motor(mid)
            return mid
    except (ValueError, UnicodeDecodeError):
        pass
    finally:
        curses.noecho()
        curses.curs_set(0)
    return None


def _draw(stdscr, teleop: Teleop, active_mid, channel: str, status_msg: str):
    stdscr.erase()
    rows, cols = stdscr.getmaxyx()
    rx = LEFT_W      # right panel start column
    rw = cols - rx   # right panel width

    # ── Header ─────────────────────────────────────────────────────────────
    hdr = f"  MABEL DaMiao Teleop"
    rhs = f"  {channel} @ 1 Mbps  "
    attr_h = curses.color_pair(_H) | curses.A_BOLD
    _safe(stdscr, 0, 0, " " * (cols - 1), attr_h)
    _safe(stdscr, 0, 0, hdr, attr_h)
    _safe(stdscr, 0, cols - len(rhs) - 1, rhs, attr_h)

    # ── Frame ──────────────────────────────────────────────────────────────
    b = curses.color_pair(_DM)
    sep_top = "├" + "─" * (LEFT_W - 2) + "┬" + "─" * (rw - 1) + "┤"
    sep_bot = "├" + "─" * (LEFT_W - 2) + "┴" + "─" * (rw - 1) + "┤"
    _safe(stdscr, 1,        0, sep_top, b)
    _safe(stdscr, rows - 2, 0, sep_bot, b)
    for r in range(2, rows - 2):
        _safe(stdscr, r, 0,        "│", b)
        _safe(stdscr, r, LEFT_W - 1, "│", b)
        _safe(stdscr, r, cols - 1,  "│", b)

    # ── Left panel ─────────────────────────────────────────────────────────
    _safe(stdscr, 2, 2, "MOTORS", curses.A_BOLD)

    with teleop.lock:
        motor_ids = sorted(teleop.motors.keys())
        snaps = {mid: (m.enabled, m.err) for mid, m in teleop.motors.items()}

    row = 4
    scroll_start = 0
    if active_mid in motor_ids:
        idx = motor_ids.index(active_mid)
        max_visible = rows - 8 - len(_LEFT_HINTS)
        scroll_start = max(0, idx - max_visible + 1)

    for mid in motor_ids[scroll_start:]:
        if row >= rows - 3 - len(_LEFT_HINTS):
            break
        enabled, err = snaps.get(mid, (False, 0))
        is_sel = (mid == active_mid)
        marker = "▸" if is_sel else " "

        if err:
            dot, dot_a = "✕", curses.color_pair(_ER) | curses.A_BOLD
            tag = f"ERR {err:X}"
        elif enabled:
            dot, dot_a = "●", curses.color_pair(_OK) | curses.A_BOLD
            tag = "ENABLED"
        else:
            dot, dot_a = "○", curses.color_pair(_DM)
            tag = "─"

        base_a = curses.color_pair(_SL) | curses.A_BOLD if is_sel else 0
        _safe(stdscr, row, 2, f"{marker} {mid:2d}  ", base_a)
        _safe(stdscr, row, 8, f"{dot} {tag}", dot_a if not is_sel else base_a)
        row += 1

    # Key hints (left panel, bottom-anchored)
    hint_row = rows - 2 - len(_LEFT_HINTS)
    for k, desc in _LEFT_HINTS:
        _safe(stdscr, hint_row, 2, f"[{k}]", curses.color_pair(_HN))
        _safe(stdscr, hint_row, 2 + len(k) + 3, desc, curses.color_pair(_DM))
        hint_row += 1

    # ── Right panel ────────────────────────────────────────────────────────
    if active_mid is None or active_mid not in {m for m in teleop.motors}:
        _safe(stdscr, 3, rx + 2,
              "No motor selected.  Press [S] to scan or [Enter] to add.",
              curses.color_pair(_DM))
    else:
        with teleop.lock:
            m = teleop.motors.get(active_mid)
        if m is None:
            return

        # Title + status
        if m.err:
            s_str  = ERROR_NAMES.get(m.err, f"ERR 0x{m.err:X}")
            s_attr = curses.color_pair(_ER) | curses.A_BOLD
            dot    = "✕"
        elif m.enabled:
            s_str  = "ENABLED"
            s_attr = curses.color_pair(_OK) | curses.A_BOLD
            dot    = "●"
        else:
            s_str  = "DISABLED"
            s_attr = curses.color_pair(_WN)
            dot    = "○"

        _safe(stdscr, 2, rx + 2, f"  Motor {m.mid}", curses.A_BOLD)
        badge = f"{dot} {s_str}  "
        _safe(stdscr, 2, cols - len(badge) - 1, badge, s_attr)

        # Live feedback
        stale_sfx = "  [NO SIGNAL]" if m.stale else ""
        _row = lambda r, lbl, val, unit, va=0: (
            _safe(stdscr, r, rx + 2,  f"{lbl:<11}", curses.color_pair(_DM)),
            _safe(stdscr, r, rx + 13, val,          curses.A_BOLD | (va or 0)),
            _safe(stdscr, r, rx + 13 + len(val) + 1, unit, curses.color_pair(_DM))
        )
        _row(4, "Position", f"{m.pos:+.4f}", f"rad  ({m.pos*57.296:+.1f}°){stale_sfx}")
        _row(5, "Velocity", f"{m.vel:+.4f}", "rad/s")
        _row(6, "Torque",   f"{m.tau:+.4f}", "Nm")
        t_attr = curses.color_pair(_ER) if m.t_mos > 80 else 0
        _row(7, "Temp MOS", f"{m.t_mos:.1f}", "°C", t_attr)
        err_str  = ERROR_NAMES.get(m.err, f"0x{m.err:X}")
        err_attr = curses.color_pair(_ER) if m.err else curses.color_pair(_OK)
        _safe(stdscr, 8, rx + 2,  "Error      ", curses.color_pair(_DM))
        _safe(stdscr, 8, rx + 13, f"{m.err}  {err_str}", err_attr | curses.A_BOLD)

        # Target position section
        div = "─" * max(0, rw - 4)
        _safe(stdscr, 10, rx + 2, f"── Target {div}", curses.color_pair(_DM))
        t_str = f"{m.target:+.4f}" if m.target is not None else "─"
        _row(11, "Desired",  t_str,        "rad    ← → to adjust")
        _row(12, "Step",     f"{m.step:.4f}", "rad    ↑ ↓ to resize")

        # Gains section
        _safe(stdscr, 14, rx + 2, f"── Gains {div}", curses.color_pair(_DM))
        _safe(stdscr, 15, rx + 2,  "Kp", curses.color_pair(_DM))
        _safe(stdscr, 15, rx + 6,  f"{m.kp:6.1f}", curses.A_BOLD)
        _safe(stdscr, 15, rx + 14, "[ decrease     ] increase", curses.color_pair(_HN))
        _safe(stdscr, 16, rx + 2,  "Kd", curses.color_pair(_DM))
        _safe(stdscr, 16, rx + 6,  f"{m.kd:6.1f}", curses.A_BOLD)
        _safe(stdscr, 16, rx + 14, "{ decrease     } increase", curses.color_pair(_HN))

    # ── Bottom hint / status bar ───────────────────────────────────────────
    if status_msg:
        bar = f"  ▶  {status_msg}"
        bar_a = curses.color_pair(_WN) | curses.A_BOLD
    else:
        bar = "  ← → pos±step    ↑ ↓ step×2/÷2    [ ] Kp    { } Kd    Q quit"
        bar_a = curses.color_pair(_HN)
    _safe(stdscr, rows - 1, 0, " " * (cols - 1))
    _safe(stdscr, rows - 1, 0, bar, bar_a)

    stdscr.refresh()


_LEFT_HINTS = [
    ("Enter", "Set ID  "),
    ("Tab",   "Next    "),
    ("S",     "Scan    "),
    ("E",     "Enable  "),
    ("D",     "Disable "),
    ("C",     "Clear   "),
    ("Z",     "Zero Pos"),
]


def _main_loop(stdscr, teleop: Teleop, channel: str):
    curses.curs_set(0)
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(_H,  curses.COLOR_WHITE,  curses.COLOR_BLUE)
    curses.init_pair(_OK, curses.COLOR_GREEN,  -1)
    curses.init_pair(_ER, curses.COLOR_RED,    -1)
    curses.init_pair(_WN, curses.COLOR_YELLOW, -1)
    curses.init_pair(_DM, curses.COLOR_WHITE,  -1)
    curses.init_pair(_SL, curses.COLOR_BLACK,  curses.COLOR_CYAN)
    curses.init_pair(_HN, curses.COLOR_CYAN,   -1)

    stdscr.timeout(1000 // CTRL_HZ)   # non-blocking getch at control rate

    active_mid  = None
    status_msg  = "Scanning bus…"
    status_time = time.perf_counter()

    # Initial scan
    threading.Thread(target=lambda: (teleop.scan(), time.sleep(0.6)), daemon=True).start()

    while True:
        rows, cols = stdscr.getmaxyx()

        if rows < MIN_ROWS or cols < MIN_COLS:
            stdscr.erase()
            _safe(stdscr, 0, 0,
                  f"Terminal too small — need {MIN_COLS}×{MIN_ROWS}, got {cols}×{rows}  (Q to quit)")
            stdscr.refresh()
            key = stdscr.getch()
            if key in (ord('q'), ord('Q')):
                return
            continue

        # Grab latest motor list
        with teleop.lock:
            ids = sorted(teleop.motors.keys())

        # Auto-select first motor when scan finishes
        if active_mid is None and ids:
            active_mid = ids[0]
            status_msg  = f"Found {len(ids)} motor(s) — selected Motor {active_mid}"
            status_time = time.perf_counter()

        # Send MIT to all enabled motors
        with teleop.lock:
            for m in teleop.motors.values():
                if m.enabled:
                    try:
                        teleop.send_mit(m)
                    except Exception:
                        pass

        # Clear status after 2.5 s
        if time.perf_counter() - status_time > 2.5:
            status_msg = ""

        _draw(stdscr, teleop, active_mid, channel, status_msg)

        key = stdscr.getch()
        if key == -1:
            continue

        with teleop.lock:
            ids  = sorted(teleop.motors.keys())
            m    = teleop.motors.get(active_mid)

        # ── Navigation & bus commands ─────────────────────────────────────
        if key in (ord('q'), ord('Q')):
            return

        elif key == ord('\t') or key in (ord('n'), ord('N')):
            if ids:
                idx = ids.index(active_mid) if active_mid in ids else -1
                active_mid = ids[(idx + 1) % len(ids)]

        elif key in (10, 13, curses.KEY_ENTER):
            result = _prompt_id(stdscr, teleop)
            if result:
                active_mid  = result
                status_msg  = f"Selected Motor {active_mid}"
            else:
                status_msg  = "Invalid ID — enter 1–15"
            status_time = time.perf_counter()

        elif key in (ord('s'), ord('S')):
            threading.Thread(target=teleop.scan, daemon=True).start()
            status_msg  = "Scanning bus…"
            status_time = time.perf_counter()

        elif key in (ord('e'), ord('E')) and active_mid:
            teleop.enable(active_mid)
            status_msg  = f"Motor {active_mid} enabled"
            status_time = time.perf_counter()

        elif key in (ord('d'), ord('D')) and active_mid:
            teleop.disable(active_mid)
            status_msg  = f"Motor {active_mid} disabled"
            status_time = time.perf_counter()

        elif key in (ord('c'), ord('C')) and active_mid:
            teleop.clear_faults(active_mid)
            status_msg  = f"Cleared faults on Motor {active_mid}"
            status_time = time.perf_counter()

        elif key in (ord('z'), ord('Z')) and active_mid:
            with teleop.lock:
                m2 = teleop.motors.get(active_mid)
                if m2 and not m2.enabled:
                    teleop.set_zero(active_mid)
                    m2.target = 0.0
                    status_msg  = f"Motor {active_mid} zeroed — power cycle to save"
                else:
                    status_msg  = "Disable the motor first before zeroing"
            status_time = time.perf_counter()

        # ── Position control ──────────────────────────────────────────────
        elif key == curses.KEY_LEFT and m:
            with teleop.lock:
                if m.target is not None:
                    m.target = max(-m.lim["p_max"], m.target - m.step)

        elif key == curses.KEY_RIGHT and m:
            with teleop.lock:
                if m.target is not None:
                    m.target = min(m.lim["p_max"], m.target + m.step)

        elif key == curses.KEY_UP and m:
            with teleop.lock:
                m.step = min(1.0, round(m.step * 2, 4))

        elif key == curses.KEY_DOWN and m:
            with teleop.lock:
                m.step = max(0.001, round(m.step / 2, 4))

        # ── Gain tuning ───────────────────────────────────────────────────
        elif key == ord(']') and m:
            with teleop.lock:
                m.kp = min(500.0, m.kp + 1.0)

        elif key == ord('[') and m:
            with teleop.lock:
                m.kp = max(0.0, m.kp - 1.0)

        elif key == ord('}') and m:
            with teleop.lock:
                m.kd = min(5.0, round(m.kd + 0.1, 2))

        elif key == ord('{') and m:
            with teleop.lock:
                m.kd = max(0.0, round(m.kd - 0.1, 2))


def main():
    ap = argparse.ArgumentParser(description="DaMiao keyboard teleop")
    ap.add_argument("--channel", default="can0", help="SocketCAN interface (default: can0)")
    args = ap.parse_args()

    teleop = Teleop(args.channel)
    try:
        curses.wrapper(_main_loop, teleop, args.channel)
    except KeyboardInterrupt:
        pass
    finally:
        print("Shutting down — disabling all motors…")
        teleop.shutdown()
        print("Done.")


if __name__ == "__main__":
    main()
