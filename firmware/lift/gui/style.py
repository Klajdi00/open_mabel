"""iOS Dark Mode design system for Tkinter.

Exact colour palette, hardware-probed OS fonts, and vector-drawn widgets.
Import everything:  from style import *
"""

import os
import platform
import tkinter as tk

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

# ── Font resolution ───────────────────────────────────────────────────────────
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
            for c in cands:
                if c in fams:
                    return c
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
_FONT_BASE = _SANS  # Exported for GUI components

# ── Typography ────────────────────────────────────────────────────────────────
FONT_TITLE  = (_SANS, 18, "bold")
FONT_SUB    = (_SANS, 14, "bold")
FONT_BODY   = (_SANS, 13, "bold")
FONT_SMALL  = (_SANS, 11, "bold")
FONT_MONO   = (_MONO, 12)
FONT_MONO_S = (_MONO, 10)

F_LARGE_TITLE = FONT_TITLE
F_TITLE1      = FONT_TITLE
F_TITLE2      = FONT_SUB
F_HEADLINE    = FONT_BODY
F_BODY        = FONT_BODY
F_CALLOUT     = (_SANS, 12, "bold")
F_FOOTNOTE    = FONT_SMALL
F_CAPTION     = (_SANS, 10, "bold")
F_MONO        = FONT_MONO
F_MONO_SM     = FONT_MONO_S

PAD_LG = 24
PAD    = 16
PAD_SM = 10
PAD_XS = 6


# ── AppleCard — Rounded Floating Platter ──────────────────────────────────────
class AppleCard(tk.Canvas):
    def __init__(self, parent, radius=16, bg_color=SURFACE, parent_bg=BG, **kwargs):
        super().__init__(parent, bg=parent_bg, highlightthickness=0, **kwargs)
        self.radius = radius
        self.bg_color = bg_color
        self.bind("<Configure>", self._draw)
        
    def _draw(self, event=None):
        self.delete("bg")
        w = self.winfo_width()
        h = self.winfo_height()
        if w <= 1 or h <= 1: return
        r = self.radius
        kw = {"fill": self.bg_color, "outline": "", "tags": "bg"}
        
        self.create_oval(0, 0, 2*r, 2*r, **kw)
        self.create_oval(w-2*r, 0, w, 2*r, **kw)
        self.create_oval(0, h-2*r, 2*r, h, **kw)
        self.create_oval(w-2*r, h-2*r, w, h, **kw)
        self.create_rectangle(r, 0, w-r, h, **kw)
        self.create_rectangle(0, r, w, h-r, **kw)
        self.lower("bg")


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
        self._draw()
        self.after_idle(self._draw)

    def _draw(self, event=None):
        self.delete("all")
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
        # macOS: an overrideredirect popup won't reliably receive clicks unless it
        # is forced topmost and actually grabs focus — otherwise the dropdown
        # appears but selections do nothing.
        try: popup.attributes("-topmost", True)
        except Exception: pass
        popup.update_idletasks()
        popup.lift()
        popup.focus_force()
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
            check = "✓  " if selected else "    "

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

# ── AppleToggle — iOS on/off switch (sliding knob) ────────────────────────────

class AppleToggle(tk.Canvas):
    """iOS switch — a sliding knob inside a pill that turns green when ON."""

    def __init__(self, parent, command=None, initial=False, width=52, height=31,
                 parent_bg=SURFACE, on_color=GREEN, **kwargs):
        super().__init__(parent, width=width, height=height, bg=parent_bg,
                         highlightthickness=0, **kwargs)
        self.command = command
        self.state = initial
        self.on_color = on_color
        self._def_w, self._def_h = width, height
        self.bind("<Button-1>", self._on_click)
        self.bind("<Configure>", lambda e: self._draw())
        self._draw()

    def _draw(self):
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        if w <= 1:
            w, h = self._def_w, self._def_h
        r = h / 2
        track = self.on_color if self.state else ELEVATED
        self.create_oval(0, 0, h, h, fill=track, outline="")
        self.create_oval(w - h, 0, w, h, fill=track, outline="")
        self.create_rectangle(r, 0, w - r, h, fill=track, outline="")
        pad = 2
        kr = r - pad
        cx = (w - r) if self.state else r
        self.create_oval(cx - kr, pad, cx + kr, h - pad, fill=LABEL, outline="")

    def set(self, state, fire=False):
        self.state = bool(state)
        self._draw()
        if fire and self.command:
            self.command(self.state)

    def get(self):
        return self.state

    def _on_click(self, event):
        self.state = not self.state
        self._draw()
        if self.command:
            self.command(self.state)


# ── Convenience factories ─────────────────────────────────────────────────────
def _btn_w(text: str, px: int = 16, font=FONT_BODY) -> int:
    import tkinter.font as tkfont
    try: text_w = tkfont.Font(font=font).measure(text)
    except Exception: text_w = len(text) * 11
    return max(80, text_w + px * 2 + 18)

def btn(parent, text, cmd, bg=ELEVATED, fg=LABEL, font=FONT_BODY, px=16, py=8, parent_bg=None, width=None, height=None, radius=12):
    pbg = parent_bg or _try_bg(parent)
    w   = width  or _btn_w(text, px, font)
    h   = height or (py * 2 + 22)
    return AppleButton(parent, text, cmd, bg_color=bg, fg_color=fg, font=font, parent_bg=pbg, width=w, height=h, radius=radius)

def accent_btn(parent, text, cmd, color=ORANGE, font=FONT_BODY, px=16, py=8, parent_bg=None, width=None, height=None, radius=12):
    return btn(parent, text, cmd, bg=color, fg=LABEL, font=font, px=px, py=py, parent_bg=parent_bg, width=width, height=height, radius=radius)

def lbl(parent, text="", fg=LABEL, font=FONT_BODY, bg=None, **kw):
    bg = bg or _try_bg(parent)
    return tk.Label(parent, text=text, fg=fg, font=font, bg=bg, **kw)

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