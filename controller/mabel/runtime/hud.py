"""Live performance HUD (per-stage rate + latency, aesthetic ANSI panel)."""
import sys
import time
import contextlib
import numpy as np

import logging
logger = logging.getLogger("mabel.hud")


# ANSI palette
_A = {"rst": "\033[0m", "dim": "\033[2m", "b": "\033[1m",
      "grn": "\033[38;5;78m", "ylw": "\033[38;5;221m", "red": "\033[38;5;203m",
      "cyn": "\033[38;5;81m", "gry": "\033[38;5;245m", "vio": "\033[38;5;141m"}


class PerfMonitor:
    """Live performance HUD. Tracks each pipeline stage's event RATE (Hz, over a
    sliding window) and per-call LATENCY (ms, EMA), and renders an aesthetic in-place
    ANSI panel — so the control / hand-estimation / hand-retargeting / arm-IK
    frequencies are all visible at a glance. Stages are registered up front; blocks
    report via record(key, ms) or the `with pm.timed(key):` context. Falls back to a
    one-line log when stdout is not a TTY (so piped logs stay clean)."""

    def __init__(self, stages, win=0.5):
        # stages: list of (key, label, target_hz) — target sets the bar full-scale
        self.stages = stages
        self.win = win
        self.count = {k: 0 for k, _, _ in stages}
        self.ema = {k: None for k, _, _ in stages}
        self.hz = {k: 0.0 for k, _, _ in stages}
        self._t0 = time.perf_counter()
        self.status = {}
        self.tty = bool(getattr(sys.stdout, "isatty", lambda: False)())
        self._nlines = 0

    def record(self, key, ms=None):
        if key not in self.count:
            return
        self.count[key] += 1
        if ms is not None:
            self.ema[key] = ms if self.ema[key] is None else 0.85 * self.ema[key] + 0.15 * ms

    @contextlib.contextmanager
    def timed(self, key):
        t = time.perf_counter()
        try:
            yield
        finally:
            self.record(key, (time.perf_counter() - t) * 1e3)

    def set_status(self, **kw):
        self.status.update(kw)

    @staticmethod
    def _bar(frac, n=12):
        frac = 0.0 if frac < 0 else (1.0 if frac > 1 else frac)
        f = int(round(frac * n))
        return "█" * f + "░" * (n - f)

    @staticmethod
    def _col(frac):
        return _A["grn"] if frac >= 0.7 else (_A["ylw"] if frac >= 0.35 else _A["red"])

    _ANSI_RE = None

    @classmethod
    def _vlen(cls, s):
        import re
        if cls._ANSI_RE is None:
            cls._ANSI_RE = re.compile(r"\033\[[0-9;?]*[A-Za-z]")
        return len(cls._ANSI_RE.sub("", s))

    def _row(self, inner, W):
        pad = max(0, W - 4 - self._vlen(inner))
        return f"{_A['cyn']}│{_A['rst']} {inner}{' ' * pad} {_A['cyn']}│{_A['rst']}"

    def lines(self):
        W = 56
        c = _A
        L = [f"{c['cyn']}╭─ {c['b']}MABEL teleop · live{c['rst']}{c['cyn']} "
             + "─" * (W - 24) + f"╮{c['rst']}"]
        for k, label, tgt in self.stages:
            hz = self.hz[k]; frac = hz / tgt if tgt else 0.0
            ms = self.ema[k]
            mss = f"{ms:5.2f}ms" if ms is not None else "   ── "
            col = self._col(frac)
            inner = (f"{c['gry']}{label:<15}{c['rst']} {col}{hz:6.1f}Hz{c['rst']} "
                     f"{c['dim']}{mss}{c['rst']} {col}{self._bar(frac)}{c['rst']}")
            L.append(self._row(inner, W))
        L.append(f"{c['cyn']}├" + "─" * (W - 2) + f"┤{c['rst']}")
        s = self.status
        v = s.get("base_v", (0.0, 0.0, 0.0))
        L.append(self._row(
            f"{c['vio']}base{c['rst']} v=({v[0]:+.2f},{v[1]:+.2f}) ω={v[2]:+.2f}  "
            f"{c['vio']}lift{c['rst']} {s.get('lift', 0.0):.2f}m  "
            f"{c['vio']}torso{c['rst']} {np.degrees(s.get('torso', 0.0)):+.0f}°", W))
        def dot(on): return f"{c['grn']}●{c['rst']}" if on else f"{c['red']}○{c['rst']}"
        sc = s.get("selfcol")
        if sc is not None and np.isfinite(sc):
            scc = c['grn'] if sc >= 0.04 else (c['ylw'] if sc >= 0.0 else c['red'])
            sctxt = f"   {c['vio']}selfcol{c['rst']} {scc}{sc * 100:+.0f}cm{c['rst']}"
        else:
            sctxt = ""
        L.append(self._row(
            f"R {dot(s.get('R'))}   L {dot(s.get('L'))}   neck {dot(s.get('neck'))}   "
            f"base {dot(s.get('base'))}{sctxt}", W))
        L.append(self._row(f"{c['dim']}{s.get('mode', '')}{c['rst']}", W))
        L.append(f"{c['cyn']}╰" + "─" * (W - 2) + f"╯{c['rst']}")
        return L

    def maybe_render(self):
        now = time.perf_counter(); el = now - self._t0
        if el < self.win:
            return
        for k, _, _ in self.stages:
            self.hz[k] = self.count[k] / el; self.count[k] = 0
        self._t0 = now
        L = self.lines()
        if self.tty:
            out = "\033[?25l"                                  # hide cursor
            if self._nlines:
                out += f"\033[{self._nlines}A"                 # move up over old panel
            out += "".join(line + "\033[K\n" for line in L)    # clear-to-EOL each row
            out += "\033[?25h"                                 # show cursor
            sys.stdout.write(out); sys.stdout.flush()
            self._nlines = len(L)
        else:
            logger.info("ctrl %.0fHz | vr %.0fHz | hand %.0fHz/%.1fms | armIK %.0fHz/%.1fms",
                        self.hz.get("ctrl", 0), self.hz.get("vr", 0),
                        self.hz.get("hand", 0), self.ema.get("hand") or 0,
                        self.hz.get("arm", 0), self.ema.get("arm") or 0)
