"""iOS Dark Mode design system for Tkinter.

Exact colour palette, SF Pro fonts, and vector-drawn widgets.
Import everything:  from style import *
"""

import tkinter as tk
import platform

# ── Colour palette — strict iOS Dark Mode ─────────────────────────────────────
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

# ── Typography — Native Apple SF Pro ──────────────────────────────────────────
_FONT_BASE  = ".AppleSystemUIFont" if platform.system() == "Darwin" else "Helvetica Neue"

FONT_TITLE  = (_FONT_BASE, 18, "bold")
FONT_SUB    = (_FONT_BASE, 14, "bold")
FONT_BODY   = (_FONT_BASE, 13, "bold")
FONT_SMALL  = (_FONT_BASE, 11, "bold")
FONT_MONO   = ("Menlo", 12)
FONT_MONO_S = ("Menlo", 10)

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


# ── AppleSlider — vector-drawn iOS-style slider ───────────────────────────────

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


# ── AppleButton — vector-drawn pill / rounded-rect button ─────────────────────

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
        # keeps the 1 (truthy) — which drew the button as a 1px sliver and made
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


# ── AppleSegmentedControl — vector-drawn iOS segmented pill ───────────────────

class AppleSegmentedControl(tk.Canvas):
    def __init__(self, parent, items: list, default_item: str, command, width=220, height=38, radius=8, parent_bg=SURFACE, **kwargs):
        super().__init__(parent, width=width, height=height, bg=parent_bg, highlightthickness=0, **kwargs)
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
            self.create_text(cx, cy, text=item, fill=LABEL, font=FONT_BODY)

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


# ── ProgressBar — composition-based ───────────────────────────────────────────

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


# ── AppleDropDown — fully custom iOS-style dropdown ───────────────────────────

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
        if len(val) > 28: val = "…" + val[-26:]
        return f"  {val}  ▾"

    def _toggle(self):
        """Open a NATIVE Tk menu at the button.

        The previous custom ``overrideredirect`` Toplevel popup was unreliable on
        macOS (borderless windows don't take focus / don't get clicks under Aqua,
        so the menu either never appeared interactive or closed before a row could
        be clicked). A native ``tk.Menu`` posted with ``tk_popup`` is handled by
        the OS: clicks, scrolling, and outside-click dismissal all just work.
        """
        m = tk.Menu(self._btn, tearoff=0)
        # Theme it where the platform honours menu colours (Linux/X11); macOS
        # renders a native menu and ignores these — which is fine, it just works.
        try:
            m.configure(bg=ELEVATED, fg=LABEL, activebackground=ORANGE,
                        activeforeground=LABEL, bd=0, relief="flat",
                        font=self._font)
        except Exception:
            pass
        for opt in self._options:
            mark = "✓  " if opt == self._value else "     "
            m.add_command(label=f"{mark}{opt}",
                          command=lambda o=opt: self._select(o))
        self._menu = m                      # keep a ref so it isn't GC'd while open
        x = self._btn.winfo_rootx()
        y = self._btn.winfo_rooty() + self._btn.winfo_height()
        try:
            m.tk_popup(x, y)
        finally:
            m.grab_release()

    def _select(self, o):
        self._value = o
        self._btn.update_text(self._label())
        if self._command:
            self._command(o)


# ── Convenience factories ─────────────────────────────────────────────────────

def _btn_w(text: str, px: int = 16, font=FONT_BODY) -> int:
    """Compute a button width that always fits its rendered text.

    Measures the real pixel width with tkfont so bold SF Pro and wide unicode
    symbols (▶ ■ ↔) never overflow the canvas and overlap adjacent buttons.
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