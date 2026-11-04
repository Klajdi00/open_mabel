"""Design tokens for the MABEL Swerve Studio GUI — strict iOS Dark Mode.

Palette and type scale follow Apple's iOS dark system. The critical detail on
Linux: Tk renders "Helvetica Neue" (which isn't installed) as an ugly X11 bitmap
font — that's the "dated" look. So instead of hard-coding a font name, we resolve
the best *actually-installed* iOS-like family at import time (SF Pro → Inter →
Helvetica Neue → Cantarell → Roboto → Ubuntu → DejaVu Sans) and use that
everywhere. Run the app under the bundled venv (system Tk w/ fontconfig) so a
real anti-aliased font is available — see swerve_gui.sh.
"""

import os
import platform

# ── Colour palette — strict iOS Dark Mode ─────────────────────────────────────
BG          = "#000000"   # window background
PANEL_BG    = "#121214"   # grouped/sidebar background
SURFACE     = "#1C1C1E"   # cards, toolbars
ELEVATED    = "#2C2C2E"   # raised controls, slider troughs
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
PURPLE      = "#BF5AF2"   # steering accent


# ── Font resolution ───────────────────────────────────────────────────────────
# Priority order: native Apple faces first, then the closest free equivalents,
# ending in DejaVu Sans which is present on essentially every Linux box.
_SANS_CANDIDATES = [
    ".AppleSystemUIFont", "SF Pro Text", "SF Pro Display", "SF Pro",
    "Inter", "Inter Display", "Helvetica Neue", "Cantarell",
    "Roboto", "Noto Sans", "Ubuntu", "DejaVu Sans",
]
_MONO_CANDIDATES = [
    "SF Mono", "Menlo", "JetBrains Mono", "Roboto Mono",
    "DejaVu Sans Mono", "Ubuntu Mono", "Noto Sans Mono",
]


def _resolve_families():
    """Pick the best installed (sans, mono) family pair.

    Needs a Tk interpreter to query font availability. We briefly borrow/create
    a hidden root so the choice is made once, at import, before any widget binds
    a default font. Falls back gracefully when no display is available.
    """
    if platform.system() == "Darwin":
        return ".AppleSystemUIFont", "Menlo"
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        return "DejaVu Sans", "DejaVu Sans Mono"
    try:
        import tkinter
        import tkinter.font as tkfont
        own = tkinter._default_root is None
        root = tkinter._default_root or tkinter.Tk()
        if own:
            root.withdraw()
        fams = set(tkfont.families())

        def best(cands, fallback):
            # 1) a candidate that is literally installed (we know exactly what renders)
            for c in cands:
                if c in fams:
                    return c
            # 2) a candidate fontconfig maps to a real (non-bitmap) face
            for c in cands:
                try:
                    if tkfont.Font(family=c, size=12).actual("family").lower() != "fixed":
                        return c
                except Exception:
                    continue
            return fallback

        sans = best(_SANS_CANDIDATES, "DejaVu Sans")
        mono = best(_MONO_CANDIDATES, "DejaVu Sans Mono")
        if own:
            root.destroy()
        return sans, mono
    except Exception:
        return "DejaVu Sans", "DejaVu Sans Mono"


_SANS, _MONO = _resolve_families()

# ── Typography — iOS type scale on the resolved family ────────────────────────
FONT_TITLE  = (_SANS, 18, "bold")
FONT_SUB    = (_SANS, 14, "bold")
FONT_BODY   = (_SANS, 13, "bold")
FONT_SMALL  = (_SANS, 11, "bold")
FONT_TINY   = (_SANS, 10, "bold")
FONT_REG    = (_SANS, 13)          # regular weight, for longer copy
FONT_MONO   = (_MONO, 12)
FONT_MONO_S = (_MONO, 11)


def lighten(hex_color: str, factor: float = 1.15, bias: int = 15) -> str:
    """Return a slightly brighter shade of ``hex_color`` (hover states)."""
    if not hex_color.startswith("#"):
        return hex_color
    try:
        r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
        return "#{:02x}{:02x}{:02x}".format(
            min(255, int(r * factor + bias)),
            min(255, int(g * factor + bias)),
            min(255, int(b * factor + bias)),
        )
    except ValueError:
        return hex_color
