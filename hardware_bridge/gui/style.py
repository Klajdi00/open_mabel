"""iOS Dark Mode design system for Tkinter.

Exact colour palette, SF Pro fonts, and vector-drawn widgets.
Import everything:  from style import *
"""

import tkinter as tk
import platform

# ── Colour palette - strict iOS Dark Mode ─────────────────────────────────────
BG          = "#000000"   
PANEL_BG    = "#121214"   
SURFACE     = "#1C1C1E"   
ELEVATED    = "#2C2C2E"   
SEP         = "#38383A"   
LABEL       = "#FFFFFF"   
LABEL2      = "#8E8E93"   
LABEL3      = "#636366"   
ORANGE      = "#FF9F0A"   
DARK_ORANGE = "#CC7A00"   
GREEN       = "#30D158"   
RED         = "#FF453A"   
BLUE        = "#0A84FF"   
YELLOW      = "#FFD60A"   

# ── Typography ────────────────────────────────────────────────────────────────
# The "dated" look came from naming fonts Tk cannot use: a bare X11/anaconda Tk exposes
# only ~47 legacy core families, so "Helvetica Neue"/"Inter"/"Noto Sans" silently fall
# back to the bitmap "fixed". We resolve against Tk's REAL families() and pick the
# cleanest scalable face actually available - Inter / SF Pro on a full Xft Tk, or the
# smooth Nimbus Sans L (a Helvetica clone) on a core-font Tk. Never the bitmap fallback.
_SANS_PREF = ["SF Pro Text", "SF Pro Display", ".AppleSystemUIFont", "Helvetica Neue",
              "Inter", "Inter Display", "Roboto", "Segoe UI", "Noto Sans", "Cantarell",
              "DejaVu Sans", "Nimbus Sans", "Nimbus Sans L", "Helvetica"]
_MONO_PREF = ["SF Mono", "Menlo", "JetBrains Mono", "Roboto Mono", "Cascadia Code",
              "Noto Sans Mono", "DejaVu Sans Mono", "Nimbus Mono L", "Liberation Mono",
              "Courier 10 Pitch", "Courier"]


def _resolve_fonts():
    if platform.system() == "Darwin":
        return ".AppleSystemUIFont", "Menlo"
    try:
        import tkinter.font as _tkfont
        created = tk._default_root is None
        root = tk._default_root or tk.Tk()
        if created:
            root.withdraw()
        have = {f.lower(): f for f in _tkfont.families()}
        def pick(prefs, fallback):
            for p in prefs:
                if p.lower() in have:
                    return have[p.lower()]
            return fallback
        sans = pick(_SANS_PREF, "Nimbus Sans L")
        mono = pick(_MONO_PREF, "Nimbus Mono L")
        if created:
            root.destroy()
        return sans, mono
    except Exception:
        return "Nimbus Sans L", "Nimbus Mono L"


_FONT_BASE, _FONT_MONO = _resolve_fonts()

FONT_TITLE  = (_FONT_BASE, 22, "bold")
FONT_SUB    = (_FONT_BASE, 15, "bold")
FONT_BODY   = (_FONT_BASE, 13)
FONT_SMALL  = (_FONT_BASE, 11)
FONT_MONO   = (_FONT_MONO, 12)
FONT_MONO_S = (_FONT_MONO, 10)


def apply_global_fonts(root):
    """Point every default-font widget at the resolved family so even unstyled bits look
    right. Call once on the Tk root at startup."""
    try:
        import tkinter.font as _tkfont
        for nm, size, weight in (("TkDefaultFont", 13, "normal"),
                                 ("TkTextFont", 13, "normal"),
                                 ("TkMenuFont", 13, "normal"),
                                 ("TkHeadingFont", 13, "bold"),
                                 ("TkFixedFont", 11, "normal")):
            f = _tkfont.nametofont(nm)
            f.configure(family=(_FONT_MONO if nm == "TkFixedFont" else _FONT_BASE),
                        size=size, weight=weight)
        root.option_add("*Font", (_FONT_BASE, 13))
    except Exception:
        pass

F_LARGE_TITLE = FONT_TITLE
F_TITLE1      = FONT_TITLE
F_TITLE2      = FONT_SUB
F_HEADLINE    = FONT_BODY
F_BODY        = FONT_BODY
F_CALLOUT     = (_FONT_BASE, 12, "bold")
F_FOOTNOTE    = FONT_SMALL
F_CAPTION     = (_FONT_BASE, 10)
F_MONO        = FONT_MONO
F_MONO_SM     = FONT_MONO_S

PAD_LG = 24
PAD    = 16
PAD_SM = 10
PAD_XS = 6


# ── AppleSlider - vector-drawn iOS-style slider ───────────────────────────────

class AppleSlider(tk.Canvas):
    def __init__(self, parent, from_, to, command=None, width=160, height=28, **kwargs):
        super().__init__(parent, width=width, height=height, bg=parent.cget("bg"), highlightthickness=0, **kwargs)
        self.min_val  = min(from_, to)
        self.max_val  = max(from_, to)
        self.val      = self.min_val
        self.command  = command
        self.padding  = 16
        self.track_y  = height // 2

        self.id_trough = self.create_line(0, 0, 0, 0, fill=ELEVATED, width=8, capstyle=tk.ROUND)
        self.id_fill   = self.create_line(0, 0, 0, 0, fill=ORANGE, width=8, capstyle=tk.ROUND)
        self.id_shadow = self.create_oval(0, 0, 0, 0, fill="#000000", outline="")
        self.id_thumb  = self.create_oval(0, 0, 0, 0, fill=LABEL, outline="")

        self.bind("<Configure>",   lambda e: self._redraw())
        self.bind("<Button-1>",    self._on_drag)
        self.bind("<B1-Motion>",   self._on_drag)

    def set(self, val):
        self.val = max(self.min_val, min(self.max_val, float(val)))
        self._redraw()

    def get(self):
        return self.val

    def _redraw(self):
        w = self.winfo_width()
        if w <= 1: w = int(self.cget("width"))
        pct = (0.0 if self.max_val == self.min_val else (self.val - self.min_val) / (self.max_val - self.min_val))
        x = self.padding + pct * (w - 2 * self.padding)

        self.coords(self.id_trough, self.padding, self.track_y, w - self.padding, self.track_y)
        self.coords(self.id_fill, self.padding, self.track_y, x, self.track_y)
        r = 13
        self.coords(self.id_shadow, x - r, self.track_y - r + 2, x + r, self.track_y + r + 2)
        self.coords(self.id_thumb, x - r, self.track_y - r, x + r, self.track_y + r)

    def _on_drag(self, event):
        w = self.winfo_width()
        if w <= 1: w = int(self.cget("width"))
        x = max(self.padding, min(w - self.padding, event.x))
        pct = (x - self.padding) / (w - 2 * self.padding)
        self.val = int(self.min_val + pct * (self.max_val - self.min_val))
        self._redraw()
        if self.command: self.command(self.val)


# ── AppleButton - vector-drawn pill / rounded-rect button ─────────────────────

class AppleButton(tk.Canvas):
    def __init__(self, parent, text, cmd, bg_color=ELEVATED, fg_color=LABEL, font=FONT_BODY, parent_bg=SURFACE, radius=12, width=100, height=38, **kwargs):
        super().__init__(parent, width=width, height=height, bg=parent_bg, highlightthickness=0, **kwargs)
        self.cmd         = cmd
        self.text_val    = text
        self.default_bg  = bg_color
        self.fg_color    = fg_color
        self.font        = font
        self.radius      = radius
        self.hover_bg    = _lighten(bg_color)
        self.current_bg  = bg_color
        self.current_fg  = fg_color
        self._shapes: list = []
        self._text_id      = None

        self.bind("<Configure>",     self._draw)
        self.bind("<Enter>",         lambda e: self._recolor(self.hover_bg, self.fg_color))
        self.bind("<Leave>",         lambda e: self._recolor(self.default_bg, self.fg_color))
        self.bind("<ButtonPress-1>", lambda e: self._recolor(self.fg_color, self.default_bg))
        self.bind("<ButtonRelease-1>", self._on_release)
        # Draw once immediately (at requested size) and again after layout settles,
        # so the button never appears as a 1px sliver before its first <Configure>.
        self._draw()
        self.after_idle(self._draw)

    def _draw(self, event=None):
        self.delete("all")
        # winfo_width() returns 1 before the widget is laid out, and "1 or X"
        # keeps the 1 (truthy) - which drew the button as a 1px sliver and made
        # adjacent buttons look stacked/overlapping until a later <Configure>.
        # Fall back to the requested width whenever the real size isn't ready.
        w = self.winfo_width()
        h = self.winfo_height()
        if w <= 1: w = int(self.cget("width"))
        if h <= 1: h = int(self.cget("height"))
        r = min(self.radius, h // 2, w // 2)
        c = self.current_bg
        kw = {"fill": c, "outline": ""}
        self._shapes = [
            self.create_oval(0, 0, 2*r, 2*r, **kw),
            self.create_oval(w-2*r, 0, w, 2*r, **kw),
            self.create_oval(0, h-2*r, 2*r, h, **kw),
            self.create_oval(w-2*r, h-2*r, w, h, **kw),
            self.create_rectangle(r, 0, w-r, h, **kw),
            self.create_rectangle(0, r, w, h-r, **kw),
        ]
        self._text_id = self.create_text(w // 2, h // 2, text=self.text_val, fill=self.current_fg, font=self.font)

    def _recolor(self, bg, fg):
        self.current_bg = bg
        self.current_fg = fg
        for s in self._shapes: self.itemconfig(s, fill=bg)
        if self._text_id: self.itemconfig(self._text_id, fill=fg)

    def _on_release(self, e):
        self._recolor(self.hover_bg, self.fg_color)
        w = self.winfo_width()
        h = self.winfo_height()
        if -15 <= e.x <= w + 15 and -15 <= e.y <= h + 15:
            if self.cmd: self.cmd()

    def update_colors(self, new_bg, new_fg=LABEL):
        self.default_bg  = new_bg
        self.fg_color    = new_fg
        self.hover_bg    = _lighten(new_bg)
        self._recolor(new_bg, new_fg)

    def update_text(self, text):
        self.text_val = text
        if self._text_id: self.itemconfig(self._text_id, text=text)


# ── AppleSegmentedControl - vector-drawn iOS segmented pill ───────────────────

class AppleSegmentedControl(tk.Canvas):
    def __init__(self, parent, items: list, default_item: str, command, width=220, height=38, radius=8, parent_bg=SURFACE, seg_font=FONT_BODY, **kwargs):
        # Size every segment to fit its label so long items (e.g. "Auto
        # Calibration") never overlap their neighbours. Measure the widest label
        # with the real font and give each segment text width + breathing room.
        import tkinter.font as tkfont
        try:
            fm = tkfont.Font(font=seg_font)
            seg_w = max(fm.measure(it) for it in items) + 40
            width = max(width, seg_w * len(items) + 8)
        except Exception:
            width = max(width, 120 * len(items))
        super().__init__(parent, width=width, height=height, bg=parent_bg, highlightthickness=0, **kwargs)
        self.seg_font    = seg_font
        self.items       = items
        self.active_item = default_item
        self.command     = command
        self.radius      = radius
        self.bg_color    = ELEVATED
        self.active_color= ORANGE

        self.bind("<Configure>", lambda e: self._draw())
        self.bind("<Button-1>",  self._on_click)

    def _draw(self):
        self.delete("all")
        w = self.winfo_width()
        h = self.winfo_height()
        if w <= 1: w = int(self.cget("width"))
        if h <= 1: h = int(self.cget("height"))
        self._rrect(0, 0, w, h, self.radius, self.bg_color)

        idx = self.items.index(self.active_item)
        sw  = w / len(self.items)
        pad = 3
        ax0, ay0 = idx * sw + pad, pad
        ax1, ay1 = (idx + 1) * sw - pad, h - pad
        self._rrect(ax0, ay0, ax1, ay1, max(0, self.radius - 2), self.active_color)

        for i, item in enumerate(self.items):
            cx = i * sw + sw / 2
            cy = h / 2
            is_active = (item == self.active_item)
            self.create_text(cx, cy, text=item, fill=LABEL if is_active else LABEL2,
                             font=self.seg_font)

    def _rrect(self, x0, y0, x1, y1, r, color):
        kw = {"fill": color, "outline": ""}
        self.create_oval(x0, y0, x0+2*r, y0+2*r, **kw)
        self.create_oval(x1-2*r, y0, x1, y0+2*r, **kw)
        self.create_oval(x0, y1-2*r, x0+2*r, y1, **kw)
        self.create_oval(x1-2*r, y1-2*r, x1, y1, **kw)
        self.create_rectangle(x0+r, y0, x1-r, y1, **kw)
        self.create_rectangle(x0, y0+r, x1, y1-r, **kw)

    def _on_click(self, event):
        w  = self.winfo_width() or int(self.cget("width"))
        sw = w / len(self.items)
        idx = int(event.x // sw)
        if 0 <= idx < len(self.items):
            sel = self.items[idx]
            if sel != self.active_item:
                self.active_item = sel
                self._draw()
                self.command(sel)

    def get(self) -> str: return self.active_item
    def set(self, item: str):
        if item in self.items:
            self.active_item = item
            self._draw()


# ── round_rect + Bar - value bars (matches the server monitor GUI) ────────────

class StatusDot(tk.Canvas):
    """A small filled status circle, DRAWN (not a glyph) so it renders on any font -
    the bitmap-font fonts here have no  glyph. ``set(color)`` recolours in place."""

    def __init__(self, parent, d=10, bg=None, color=LABEL3):
        bg = bg or parent.cget("bg")
        super().__init__(parent, width=d, height=d, bg=bg, highlightthickness=0)
        self._id = self.create_oval(1, 1, d - 1, d - 1, fill=color, outline="")

    def set(self, color):
        self.itemconfig(self._id, fill=color)


def round_rect(cv: tk.Canvas, x0, y0, x1, y1, r, **kw):
    """Draw a rounded rectangle as a single smoothed polygon on a Canvas."""
    r = min(r, (x1 - x0) / 2, (y1 - y0) / 2)
    pts = [
        x0 + r, y0,  x1 - r, y0,  x1, y0,  x1, y0 + r,
        x1, y1 - r,  x1, y1,  x1 - r, y1,  x0 + r, y1,
        x0, y1,  x0, y1 - r,  x0, y0 + r,  x0, y0,
    ]
    return cv.create_polygon(pts, smooth=True, **kw)


class Bar(tk.Canvas):
    """A thin rounded value bar that fills from the left for a 0..1 quantity
    (e.g. a joint's position as a fraction of its calibrated range). Mirrors the
    server monitor's ``Bar``; ``set(frac, color)`` redraws in place, cheap enough
    to refresh live at the telemetry rate."""

    def __init__(self, parent, width=120, height=10, bg=PANEL_BG, **kw):
        super().__init__(parent, width=width, height=height, bg=bg,
                         highlightthickness=0, **kw)
        self._cw, self._ch = width, height
        self.set(0.0, LABEL3)

    def set(self, frac: float, color: str = GREEN):
        self.delete("all")
        w, h = self._cw, self._ch
        r = h / 2
        round_rect(self, 0, 0, w, h, r, fill=ELEVATED, outline="")
        try:
            frac = max(0.0, min(1.0, float(frac)))
        except (TypeError, ValueError):
            frac = 0.0
        x = frac * w
        if x > 2:
            round_rect(self, 0, 0, x, h, min(r, x / 2), fill=color, outline="")


class RangeBar(tk.Canvas):
    """Manual-calibration range view. On a shared domain it draws: a dim
    reference band (the theoretical / URDF tick range), a coloured fill for the
    swept ``min..max`` found so far, and a bright cursor at the live joint
    position. Over-travel past the reference stays visible because the domain
    expands to include it. ``set(...)`` redraws in place."""

    def __init__(self, parent, width=300, height=14, bg=PANEL_BG, **kw):
        super().__init__(parent, width=width, height=height, bg=bg,
                         highlightthickness=0, **kw)
        self._cw, self._ch = width, height

    def set(self, dom_lo, dom_hi, theo_lo, theo_hi, obs_lo, obs_hi, cur,
            color=GREEN):
        self.delete("all")
        w, h = self._cw, self._ch
        r = h / 2
        span = (dom_hi - dom_lo) or 1.0

        def X(v):
            return max(0.0, min(1.0, (v - dom_lo) / span)) * w

        round_rect(self, 0, 0, w, h, r, fill=ELEVATED, outline="")        # track
        tl, tr = X(theo_lo), X(theo_hi)                                    # URDF band
        if tr - tl > 1:
            round_rect(self, tl, 2, tr, h - 2, min(r, (tr - tl) / 2),
                       fill=_lighten(ELEVATED, 22), outline="")
        ol, orr = X(obs_lo), X(obs_hi)                                     # swept range
        if orr - ol > 1:
            round_rect(self, ol, 1, orr, h - 1, min(r, (orr - ol) / 2),
                       fill=color, outline="")
        cx = X(cur)                                                        # live cursor
        self.create_line(cx, 0, cx, h, fill=LABEL, width=2)


# ── ProgressBar - composition-based ───────────────────────────────────────────

class ProgressBar:
    def __init__(self, parent, width=120, height=4, fill=ORANGE, track=ELEVATED):
        self._w = width
        self._h = height
        self.widget = tk.Frame(parent, bg=track, width=width, height=height)
        self.widget.pack_propagate(False)
        self._bar = tk.Frame(self.widget, bg=fill)
        self._bar.place(x=0, y=0, width=0, height=height)

    def pack(self, **kw):   self.widget.pack(**kw)
    def grid(self, **kw):   self.widget.grid(**kw)
    def place(self, **kw):  self.widget.place(**kw)
    def pack_forget(self):  self.widget.pack_forget()

    def set(self, pct: float):
        filled = int(self._w * max(0.0, min(100.0, float(pct))) / 100.0)
        self._bar.place(x=0, y=0, width=filled, height=self._h)


# ── AppleDropDown - fully custom iOS-style dropdown ───────────────────────────

class AppleDropDown:
    def __init__(self, parent, options: list, default: str | None = None, command=None, width=260, height=42, parent_bg=BG, font=FONT_BODY):
        self._options  = list(options)
        self._value    = default if default in options else (options[0] if options else "")
        self._command  = command
        self._popup: tk.Toplevel | None = None
        self._font     = font

        self._btn = AppleButton(
            parent, self._label(), self._toggle,
            bg_color=ELEVATED, fg_color=LABEL,
            font=font, parent_bg=parent_bg,
            width=width, height=height, radius=12
        )

    def get(self) -> str: return self._value

    def set(self, value: str):
        if value in self._options:
            self._value = value
            self._btn.update_text(self._label())

    def update_options(self, options: list, default: str | None = None):
        self._options = list(options)
        if default and default in options: self._value = default
        elif options and self._value not in options: self._value = options[0]
        self._btn.update_text(self._label())

    def pack(self, **kw):  self._btn.pack(**kw)
    def grid(self, **kw):  self._btn.grid(**kw)
    def place(self, **kw): self._btn.place(**kw)

    def _label(self) -> str:
        val = self._value
        if len(val) > 28: val = "..." + val[-26:]
        return f"  {val}  ▾"

    def _toggle(self):
        if self._popup and self._popup.winfo_exists(): self._close_popup()
        else: self._open_popup()

    def _open_popup(self):
        bx = self._btn.winfo_rootx()
        by = self._btn.winfo_rooty() + self._btn.winfo_height() + 4
        bw = max(self._btn.winfo_width(), 260)
        row_h = 46
        total_h = min(len(self._options) * row_h, 320)

        popup = tk.Toplevel()
        popup.overrideredirect(True)
        popup.configure(bg=ELEVATED)
        popup.geometry(f"{bw}x{total_h}+{bx}+{by}")
        popup.lift()
        popup.focus_set()
        self._popup = popup

        canvas = tk.Canvas(popup, bg=ELEVATED, highlightthickness=0)
        sb = tk.Scrollbar(popup, orient="vertical", command=canvas.yview, bg=ELEVATED, troughcolor=ELEVATED)
        if total_h < len(self._options) * row_h: sb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        
        inner = tk.Frame(canvas, bg=ELEVATED)
        canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.configure(yscrollcommand=sb.set)

        for opt in self._options:
            selected = (opt == self._value)
            default_bg = _lighten(ELEVATED, 10) if selected else ELEVATED
            check = "ok  " if selected else "    "

            row = tk.Frame(inner, bg=default_bg, cursor="hand2", height=row_h)
            row.pack(fill="x")
            row.pack_propagate(False)

            lbl_w = tk.Label(row, text=f"{check}{opt}", bg=default_bg, fg=(ORANGE if selected else LABEL), font=self._font, anchor="w", padx=14)
            lbl_w.pack(side="left", fill="both", expand=True)
            tk.Frame(row, bg=SEP, height=1).pack(side="bottom", fill="x")

            def _select(o=opt):
                self._value = o
                self._btn.update_text(self._label())
                self._close_popup()
                if self._command: self._command(o)

            def _enter(e, r=row, l=lbl_w, o=opt):
                bg = ORANGE if o == self._value else _lighten(ELEVATED, 25)
                r.config(bg=bg); l.config(bg=bg)

            def _leave(e, r=row, l=lbl_w, o=opt):
                bg = _lighten(ELEVATED, 10) if o == self._value else ELEVATED
                r.config(bg=bg); l.config(bg=bg)

            row.bind("<Button-1>",  lambda e, fn=_select: fn())
            lbl_w.bind("<Button-1>", lambda e, fn=_select: fn())
            row.bind("<Enter>",  _enter)
            lbl_w.bind("<Enter>", _enter)
            row.bind("<Leave>",  _leave)
            lbl_w.bind("<Leave>", _leave)

        popup.bind("<FocusOut>", lambda e: self._schedule_close(popup))

    def _schedule_close(self, popup):
        try: popup.after(120, lambda: self._close_popup() if (self._popup and self._popup.winfo_exists()) else None)
        except Exception: pass

    def _close_popup(self):
        if self._popup:
            try: self._popup.destroy()
            except Exception: pass
            self._popup = None


# ── Convenience factories ─────────────────────────────────────────────────────

def _btn_w(text: str, px: int = 16, font=FONT_BODY) -> int:
    """Compute a button width that always fits its rendered text.

    Measures the real pixel width with tkfont so bold SF Pro and wide unicode
    symbols (<->) never overflow the canvas and overlap adjacent buttons.
    """
    import tkinter.font as tkfont
    try:
        text_w = tkfont.Font(font=font).measure(text)
    except Exception:
        # Fallback estimate (no root yet): generous per-char width
        text_w = len(text) * 11
    return max(80, text_w + px * 2 + 18)

def btn(parent, text, cmd, bg=ELEVATED, fg=LABEL, font=FONT_BODY, px=16, py=8, parent_bg=None, width=None, height=None, radius=12):
    pbg = parent_bg or _try_bg(parent)
    w   = width  or _btn_w(text, px, font)
    h   = height or (py * 2 + 22)
    return AppleButton(parent, text, cmd, bg_color=bg, fg_color=fg, font=font, parent_bg=pbg, width=w, height=h, radius=radius)

def accent_btn(parent, text, cmd, color=ORANGE, font=FONT_BODY, px=16, py=8, parent_bg=None, width=None, height=None, radius=12):
    return btn(parent, text, cmd, bg=color, fg=LABEL, font=font, px=px, py=py, parent_bg=parent_bg, width=width, height=height, radius=radius)

def ghost_btn(parent, text, cmd, fg=LABEL2, font=F_CALLOUT, px=12, py=6, parent_bg=None):
    return btn(parent, text, cmd, bg=ELEVATED, fg=fg, font=font, px=px, py=py, parent_bg=parent_bg)

def lbl(parent, text="", fg=LABEL, font=FONT_BODY, bg=None, **kw):
    bg = bg or _try_bg(parent)
    return tk.Label(parent, text=text, fg=fg, font=font, bg=bg, **kw)

def section_header(parent, text):
    f = tk.Frame(parent, bg=BG)
    tk.Label(f, text=text.upper(), bg=BG, fg=ORANGE, font=F_CAPTION, anchor="w").pack(side="left", padx=PAD_SM)
    return f

def divider(parent, vertical=False):
    return tk.Frame(parent, bg=SEP, width=1) if vertical else tk.Frame(parent, bg=SEP, height=1)

def card(parent, **kw):
    return tk.Frame(parent, bg=SURFACE, padx=PAD_SM, pady=PAD_SM, **kw)

def scrolled_text(parent, **kw):
    from tkinter import scrolledtext
    return scrolledtext.ScrolledText(parent, bg=SURFACE, fg=LABEL, font=FONT_MONO_S, insertbackground=LABEL, relief="flat", bd=0, selectbackground=BLUE, selectforeground=LABEL, **kw)

# ── Internal helpers ──────────────────────────────────────────────────────────

def _try_bg(widget) -> str:
    try: return widget.cget("bg")
    except Exception: return SURFACE

def _lighten(hex_color: str, amount: int = 20) -> str:
    try:
        r = min(255, int(hex_color[1:3], 16) + amount)
        g = min(255, int(hex_color[3:5], 16) + amount)
        b = min(255, int(hex_color[5:7], 16) + amount)
        return f"#{r:02x}{g:02x}{b:02x}"
    except Exception: return hex_color

def _darken(hex_color: str, amount: int = 15) -> str:
    try:
        r = max(0, int(hex_color[1:3], 16) - amount)
        g = max(0, int(hex_color[3:5], 16) - amount)
        b = max(0, int(hex_color[5:7], 16) - amount)
        return f"#{r:02x}{g:02x}{b:02x}"
    except Exception: return hex_color