"""console.py — shared pretty console output for the MABEL server stack.

One place for every process (teleop bridge, camera server, MJPEG fan-out) to
get the same look:

  • `setup(name)`        — installs a color log formatter and returns a logger.
        14:32:05.1 ┊ ●  bridge   client connected 192.168.1.7:50312
  • `banner(...)`        — rounded box with the server identity + endpoints.
  • `rule(...)` / `kv()` — section separators and aligned key→value lines.
  • `lan_ip()`           — best-effort LAN address so printed URLs are real,
                           not `<mac>` placeholders.

Colors auto-disable when stderr is not a TTY (e.g. the camera server's log
file), so redirected logs stay clean plain text. NO_COLOR is honored too.
"""
from __future__ import annotations

import logging
import os
import socket
import sys

# ── Color plumbing ────────────────────────────────────────────────────────────
_TTY = sys.stderr.isatty() and os.getenv("NO_COLOR") is None


def _c(code: str, s: str) -> str:
    """Wrap `s` in an ANSI color when the output is a live terminal."""
    return f"\033[{code}m{s}\033[0m" if _TTY else s


# Palette (kept small + consistent across every server process)
DIM, BOLD = "2", "1"
CYAN, GREEN, YELLOW, RED, MAGENTA, BLUE = "36", "32", "33", "31", "35", "34"


# ── Log formatter ─────────────────────────────────────────────────────────────
class PrettyFormatter(logging.Formatter):
    """`14:32:05.1 ┊ ●  bridge   message` with the glyph colored by level."""

    _LEVEL = {
        logging.DEBUG:    (DIM,        "·"),
        logging.INFO:     (GREEN,      "●"),
        logging.WARNING:  (YELLOW,     "▲"),
        logging.ERROR:    (RED,        "✖"),
        logging.CRITICAL: (BOLD + ";" + RED, "✖"),
    }

    def format(self, record: logging.LogRecord) -> str:
        color, glyph = self._LEVEL.get(record.levelno, ("0", " "))
        ts = self.formatTime(record, "%H:%M:%S") + f".{int(record.msecs/100):01d}"
        head = (f"{_c(DIM, ts)} {_c(DIM, '┊')} {_c(color, glyph)}  "
                f"{_c(CYAN, f'{record.name:<8}')} ")
        msg = record.getMessage()
        if record.levelno >= logging.WARNING:        # make problems pop
            msg = _c(color, msg)
        if record.exc_info:
            msg += "\n" + self.formatException(record.exc_info)
        return head + msg


def setup(name: str, level: int = logging.INFO) -> logging.Logger:
    """Install the pretty formatter on the root logger (idempotent-enough for
    one process) and return the named logger every module shares."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(PrettyFormatter())
    logging.basicConfig(level=level, handlers=[handler], force=True)
    # Third-party connection chatter duplicates our own connect/disconnect
    # lines (and overflows the name column) — keep only their warnings.
    for noisy in ("websockets", "websockets.server", "aiohttp.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logging.getLogger(name)


# ── Network helper ────────────────────────────────────────────────────────────
def lan_ip() -> str:
    """This machine's LAN address (no traffic is actually sent)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


# ── Banner / structural output ────────────────────────────────────────────────
def banner(title: str, subtitle: str | None,
           rows: list[tuple[str, str]],
           footer: list[str] | None = None,
           min_width: int = 60) -> None:
    """Rounded banner box, auto-sized to its longest line. Widths are computed
    on the UNCOLORED text so the borders always line up; color is applied per
    cell afterwards."""
    klen = max((len(k) for k, _ in rows), default=0)
    row_plain = [f"{k:<{klen}}   {v}" for k, v in rows]
    foot = footer or []
    w = max([min_width, len(title), len(subtitle or "")]
            + [len(s) for s in row_plain] + [len(f) for f in foot])

    def line(plain: str, colored: str) -> str:
        return f"│ {colored}{' ' * (w - len(plain))} │"

    out = [f"╭{'─' * (w + 2)}╮"]
    out.append(line(title.center(w), _c(BOLD + ";" + CYAN, title.center(w))))
    if subtitle:
        out.append(line(subtitle.center(w), _c(DIM, subtitle.center(w))))
    if rows:
        out.append(f"├{'─' * (w + 2)}┤")
        for (k, _v), plain in zip(rows, row_plain):
            colored = f"{_c(DIM, f'{k:<{klen}}')}{plain[klen:]}"
            out.append(line(plain, colored))
    if foot:
        out.append(f"├{'─' * (w + 2)}┤")
        for f in foot:
            out.append(line(f, _c(DIM, f)))
    out.append(f"╰{'─' * (w + 2)}╯")
    print("\n" + "\n".join(out) + "\n", flush=True)


def rule(label: str = "", width: int = 64) -> None:
    """A dim horizontal rule, optionally labelled:  ── cameras ─────────"""
    if label:
        bar = f"── {label} " + "─" * max(0, width - len(label) - 4)
    else:
        bar = "─" * width
    print(_c(DIM, bar), flush=True)


def kv(key: str, value: str, indent: int = 2) -> str:
    """An aligned `key value` fragment for log lines:  `  head left   → /camera/...`"""
    return f"{' ' * indent}{_c(DIM, f'{key:<22}')} {value}"
