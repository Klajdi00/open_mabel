"""gui_style.py — iOS Dark Mode design tokens + vector widgets for the MABEL
command monitor, matching the firmware GUIs (swerve_drive/gui, lift/gui,
orca_hand/feetech/gui).

Same palette and type scale as the firmware Tk tools so every MABEL desktop
tool looks like one product. Fonts resolve to the best installed Apple-like
family (SF Pro on macOS), falling back gracefully on Linux. Widgets are drawn on
Canvas so corners are truly rounded and dots/pills are crisp — Tk's native
widgets can't do that.
"""
from __future__ import annotations

import platform
import tkinter as tk

# ── Colour palette — strict iOS Dark Mode (identical to firmware theme.py) ────
BG          = "#000000"   # window background
PANEL_BG    = "#121214"   # grouped / sidebar background
SURFACE     = "#1C1C1E"   # cards, toolbars
ELEVATED    = "#2C2C2E"   # raised controls, troughs
ELEVATED_HOVER = "#3A3A3E" # raised control under the cursor
SEP         = "#38383A"   # hairline separators
LABEL       = "#FFFFFF"   # primary text
LABEL2      = "#8E8E93"   # secondary text
LABEL3      = "#636366"   # tertiary / disabled text
ORANGE      = "#FF9F0A"   # MABEL accent
DARK_ORANGE = "#CC7A00"
GREEN       = "#30D158"   # nominal / success
RED         = "#FF453A"   # fault / danger
BLUE        = "#0A84FF"   # info / neutral action
YELLOW      = "#FFD60A"   # warning
PURPLE      = "#BF5AF2"   # tracking accent
TEAL        = "#64D2FF"

# Per-message-type accent used across the stream + pills (kept in one place so
# the log colour and the status pill for a concept never disagree).
TYPE_COLOR = {
    "hello":          GREEN,
    "bye":            LABEL3,
    "teleop_frame":   BLUE,
    "control_mode":   ORANGE,
    "joint_command":  PURPLE,
    "reset":          RED,
    "recalibrate":    TEAL,
    "record":         YELLOW,
    "external_force": ORANGE,
    "error":          RED,
}
LEVEL_COLOR = {
    "info": LABEL2, "strong": LABEL, "warn": YELLOW,
    "reset": RED, "rec": YELLOW,
}

# Device identity → (accent colour, short chip text, glyph). Keyed by the exact
# label the bridge resolves (see _identify_device / monitor_hub.resolved_device).
DEVICE_STYLE = {
    "Vision Pro": (PURPLE, "VISION PRO", "🥽"),
    "iPhone":     (BLUE,   "IPHONE",     "📱"),
    "iPad":       (BLUE,   "IPAD",       "📱"),
    "Website":    (TEAL,   "WEB",        "🌐"),
    "Keyboard":   (ORANGE, "KEYBOARD",   "⌨"),
    "Gamepad":    (GREEN,  "GAMEPAD",    "🎮"),
    "Linux":      (YELLOW, "LINUX",      "🐧"),
}
DEVICE_FALLBACK = (LABEL3, "UNKNOWN", "❔")


# Transport / network mode → (accent, short chip). Mirrors _classify_network.
TRANSPORT_STYLE = {
    "LAN":       (GREEN,  "LAN"),
    "Tailscale": (PURPLE, "TAILSCALE"),
    "Localhost": (LABEL2, "LOCAL"),
    "Internet":  (YELLOW, "INTERNET"),
}
TRANSPORT_FALLBACK = (LABEL3, "NET?")


def transport_style(label: str):
    """(accent, chip) for a transport label."""
    return TRANSPORT_STYLE.get(label, TRANSPORT_FALLBACK)


def device_style(label: str):
    """(accent, chip, glyph) for a resolved device label; inferred guesses
    (suffixed '?') reuse the base style but stay dim."""
    base = label.rstrip("?")
    color, chip, glyph = DEVICE_STYLE.get(base, DEVICE_FALLBACK)
    if label.endswith("?"):                 # heuristic guess, not self-reported
        return LABEL2, chip + "?", glyph
    return color, chip, glyph

# ── Typography ────────────────────────────────────────────────────────────────
_BASE = ".AppleSystemUIFont" if platform.system() == "Darwin" else "Helvetica Neue"
_MONO = "Menlo" if platform.system() == "Darwin" else "DejaVu Sans Mono"

FONT_TITLE  = (_BASE, 18, "bold")
FONT_H1     = (_BASE, 15, "bold")
FONT_H2     = (_BASE, 13, "bold")
FONT_BODY   = (_BASE, 12)
FONT_BOLD   = (_BASE, 12, "bold")
FONT_SMALL  = (_BASE, 11)
FONT_TINY   = (_BASE, 10, "bold")
FONT_MONO   = (_MONO, 12)
FONT_MONO_S = (_MONO, 11)

PAD_LG, PAD, PAD_SM, PAD_XS = 20, 14, 9, 5

# ── Liquid-glass palette (translucent popovers; pair with a Toplevel -alpha) ──
GLASS       = "#26262B"   # frosted panel fill
GLASS_TOP   = "#34343B"   # lighter top band for the sheen gradient
GLASS_HOVER = "#3C3C44"   # row hover
GLASS_BORDER = "#5C5C64"  # hairline edge
GLASS_SHEEN = "#9A9AA6"   # bright top highlight line (the "glass" edge)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def mix(a: str, b: str, t: float) -> str:
    """Blend hex colour a→b by t∈[0,1]."""
    ar, ag, ab = _hex_to_rgb(a)
    br, bg, bb = _hex_to_rgb(b)
    return f"#{round(ar+(br-ar)*t):02x}{round(ag+(bg-ag)*t):02x}{round(ab+(bb-ab)*t):02x}"


def round_rect(cv: tk.Canvas, x0, y0, x1, y1, r, **kw):
    """Draw a rounded rectangle as a single smoothed polygon."""
    r = min(r, (x1 - x0) / 2, (y1 - y0) / 2)
    pts = [
        x0 + r, y0,  x1 - r, y0,  x1, y0,  x1, y0 + r,
        x1, y1 - r,  x1, y1,  x1 - r, y1,  x0 + r, y1,
        x0, y1,  x0, y1 - r,  x0, y0 + r,  x0, y0,
    ]
    return cv.create_polygon(pts, smooth=True, **kw)


def glass_panel(cv: tk.Canvas, x0, y0, x1, y1, r=18):
    """A frosted rounded panel with a bright top sheen + hairline border — the
    'liquid glass' popover look. Pair with a Toplevel `-alpha` for translucency
    and a `-transparent` background so the corners read as truly rounded."""
    round_rect(cv, x0, y0, x1, y1, r, fill=GLASS, outline=GLASS_BORDER)
    cv.create_line(x0 + r - 4, y0 + 2, x1 - r + 4, y0 + 2, fill=GLASS_SHEEN)
    cv.create_line(x0 + r, y0 + 5, x1 - r, y0 + 5, fill=GLASS_TOP)


class Card(tk.Frame):
    """A rounded SURFACE panel with an optional header label."""

    def __init__(self, parent, title: str | None = None, bg: str = SURFACE, **kw):
        super().__init__(parent, bg=bg, highlightthickness=0, **kw)
        self._bg = bg
        if title:
            head = tk.Frame(self, bg=bg)
            head.pack(fill="x", padx=PAD, pady=(PAD_SM, 0))
            tk.Label(head, text=title.upper(), bg=bg, fg=LABEL2,
                     font=FONT_TINY).pack(side="left")
            self.header = head


class Dot(tk.Canvas):
    """A small status dot whose colour can change live."""

    def __init__(self, parent, color=LABEL3, r=5, bg=SURFACE):
        super().__init__(parent, width=2 * r + 4, height=2 * r + 4,
                         bg=bg, highlightthickness=0)
        self._r = r
        self._id = self.create_oval(2, 2, 2 + 2 * r, 2 + 2 * r,
                                    fill=color, outline="")

    def set(self, color: str):
        self.itemconfig(self._id, fill=color)


class Pill(tk.Canvas):
    """A rounded status pill: a title (small, dim) over a value, with a left
    accent dot. Used across the status strip for mode / control / tracking."""

    def __init__(self, parent, title: str, width=132, height=58, bg=PANEL_BG):
        super().__init__(parent, width=width, height=height, bg=bg,
                         highlightthickness=0)
        self._cw, self._ch = width, height
        self._title = title
        self._accent = ELEVATED
        self._value = "—"
        self._sub = ""
        self._draw()

    def _draw(self):
        self.delete("all")
        round_rect(self, 1, 1, self._cw - 1, self._ch - 1, 14,
                   fill=SURFACE, outline=mix(SURFACE, self._accent, 0.35))
        self.create_oval(13, 13, 21, 21, fill=self._accent, outline="")
        self.create_text(27, 17, text=self._title.upper(), anchor="w",
                         fill=LABEL2, font=FONT_TINY)
        self.create_text(14, 38, text=self._value, anchor="w",
                         fill=LABEL, font=FONT_H2)
        if self._sub:
            self.create_text(self._cw - 13, 38, text=self._sub, anchor="e",
                             fill=self._accent, font=FONT_SMALL)

    def set(self, value: str, accent: str | None = None, sub: str = ""):
        self._value = value
        self._sub = sub
        if accent is not None:
            self._accent = accent
        self._draw()


class Bar(tk.Canvas):
    """A thin value bar — bipolar (fills left/right from a centre tick, for signed
    quantities like a joint angle or twist) or unipolar (fills from the left, for
    0..1 quantities like a gripper). `set(frac, color)` with frac in [-1,1] or
    [0,1]; cheap to update in place, so it can refresh live without a rebuild."""

    def __init__(self, parent, width=150, height=12, bipolar=True, bg=SURFACE):
        super().__init__(parent, width=width, height=height, bg=bg,
                         highlightthickness=0)
        self._cw, self._ch, self._bipolar = width, height, bipolar
        self.set(0.0, LABEL3)

    def set(self, frac: float, color: str = BLUE):
        self.delete("all")
        w, h = self._cw, self._ch
        r = h / 2
        round_rect(self, 0, 0, w, h, r, fill=ELEVATED, outline="")
        try:
            frac = max(-1.0, min(1.0, float(frac)))
        except (TypeError, ValueError):
            frac = 0.0
        if self._bipolar:
            cx = w / 2
            x = cx + frac * (w / 2 - 2)
            x0, x1 = (cx, x) if x >= cx else (x, cx)
            if abs(x1 - x0) > 1:
                round_rect(self, x0, 1, x1, h - 1, min(r, (x1 - x0) / 2),
                           fill=color, outline="")
            self.create_line(cx, 1, cx, h - 1, fill=LABEL3)
        else:
            x = max(0.0, min(1.0, frac)) * w
            if x > 2:
                round_rect(self, 0, 1, x, h - 1, min(r, x / 2),
                           fill=color, outline="")


class Toggle(tk.Canvas):
    """A read-only iOS pill indicator with a label and ON/OFF/state colour —
    used for booleans like hand-tracking L/R and recording."""

    def __init__(self, parent, label: str, width=72, height=58, bg=PANEL_BG):
        super().__init__(parent, width=width, height=height, bg=bg,
                         highlightthickness=0)
        self._cw, self._ch, self._label = width, height, label
        self._on, self._color, self._text = False, LABEL3, "OFF"
        self._draw()

    def _draw(self):
        self.delete("all")
        round_rect(self, 1, 1, self._cw - 1, self._ch - 1, 14,
                   fill=SURFACE, outline=mix(SURFACE, self._color, 0.4))
        self.create_text(self._cw / 2, 16, text=self._label.upper(),
                         fill=LABEL2, font=FONT_TINY)
        track = self._color if self._on else ELEVATED
        round_rect(self, self._cw / 2 - 17, 30, self._cw / 2 + 17, 46, 8,
                   fill=track, outline="")
        kx = self._cw / 2 + 9 if self._on else self._cw / 2 - 9
        self.create_oval(kx - 7, 31, kx + 7, 45, fill=LABEL, outline="")

    def set(self, on: bool, color: str = GREEN, text: str = ""):
        self._on, self._color = on, color if on else LABEL3
        self._text = text or ("ON" if on else "OFF")
        self._draw()
