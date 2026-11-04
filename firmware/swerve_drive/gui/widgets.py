"""Vector-drawn iOS UI primitives shared by every panel.

Ports the AppleSlider / AppleButton / AppleSegmentedControl from the ORCA Hand
Studio verbatim (so styling stays identical) and adds two new primitives the
swerve GUI needs: AppleToggle (the iOS switch) and Card (a rounded surface
container).  All widgets are pure tk.Canvas — no images, no ttk themes — so
they render identically on any platform.
"""

import tkinter as tk

from . import theme as T


class AppleSlider(tk.Canvas):
    """Horizontal slider with a rounded track and a floating white thumb."""

    def __init__(self, parent, from_, to, command=None, width=160, height=28,
                 accent=T.ORANGE, **kwargs):
        super().__init__(parent, width=width, height=height, bg=T.SURFACE,
                         highlightthickness=0, **kwargs)
        self.min_val, self.max_val = min(from_, to), max(from_, to)
        self.val = self.min_val
        self.command = command
        self.accent = accent
        self.padding = 14
        self.track_y = height // 2

        self.id_trough = self.create_line(self.padding, self.track_y, width - self.padding,
                                          self.track_y, fill=T.ELEVATED, width=8, capstyle=tk.ROUND)
        self.id_fill = self.create_line(self.padding, self.track_y, self.padding, self.track_y,
                                        fill=accent, width=8, capstyle=tk.ROUND)
        self.id_shadow = self.create_oval(0, 0, 0, 0, fill="#000000", outline="")
        self.id_thumb = self.create_oval(0, 0, 0, 0, fill=T.LABEL, outline="")

        self.bind("<Button-1>", self._on_drag)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<Configure>", lambda e: self._redraw())

    def set(self, val):
        self.val = max(self.min_val, min(self.max_val, val))
        self._redraw()

    def get(self):
        return self.val

    def set_accent(self, color):
        self.accent = color
        self.itemconfig(self.id_fill, fill=color)

    def _redraw(self):
        w = self.winfo_width()
        if w <= 1:
            w = int(self.cget("width"))
        span = self.max_val - self.min_val
        pct = 0 if span == 0 else (self.val - self.min_val) / span
        x = self.padding + pct * (w - 2 * self.padding)
        self.coords(self.id_trough, self.padding, self.track_y, w - self.padding, self.track_y)
        self.coords(self.id_fill, self.padding, self.track_y, x, self.track_y)
        r = 13
        self.coords(self.id_shadow, x - r, self.track_y - r + 2, x + r, self.track_y + r + 2)
        self.coords(self.id_thumb, x - r, self.track_y - r, x + r, self.track_y + r)

    def _on_drag(self, event):
        w = self.winfo_width()
        if w <= 1:
            w = int(self.cget("width"))
        x_clamped = max(self.padding, min(w - self.padding, event.x))
        pct = (x_clamped - self.padding) / (w - 2 * self.padding)
        self.val = self.min_val + pct * (self.max_val - self.min_val)
        self._redraw()
        if self.command:
            self.command(self.val)


class AppleButton(tk.Canvas):
    """Rounded pill button with hover + press states."""

    def __init__(self, parent, text, cmd, bg_color, fg_color=T.LABEL, font=T.FONT_BODY,
                 parent_bg=T.SURFACE, radius=12, width=100, height=38, **kwargs):
        super().__init__(parent, width=width, height=height, bg=parent_bg,
                         highlightthickness=0, **kwargs)
        self.cmd = cmd
        self.text_val = text
        self.default_bg = bg_color
        self.fg_color = fg_color
        self.font = font
        self.radius = radius
        self.hover_bg = T.lighten(bg_color)
        self.current_bg = bg_color
        self.current_fg = fg_color
        self.shapes = []
        self.text_id = None

        self.bind("<Configure>", self._draw)
        self.bind("<Enter>", lambda e: self._recolor(self.hover_bg, self.fg_color))
        self.bind("<Leave>", lambda e: self._recolor(self.default_bg, self.fg_color))
        self.bind("<ButtonPress-1>", lambda e: self._recolor(self.fg_color, self.default_bg))
        self.bind("<ButtonRelease-1>", self._on_release)

    def _draw(self, event=None):
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        if w <= 1:
            w = int(self.cget("width"))
        if h <= 1:
            h = int(self.cget("height"))
        r = min(self.radius, h // 2, w // 2)
        kw = {"fill": self.current_bg, "outline": ""}
        self.shapes = [
            self.create_oval(0, 0, 2 * r, 2 * r, **kw),
            self.create_oval(w - 2 * r, 0, w, 2 * r, **kw),
            self.create_oval(0, h - 2 * r, 2 * r, h, **kw),
            self.create_oval(w - 2 * r, h - 2 * r, w, h, **kw),
            self.create_rectangle(r, 0, w - r, h, **kw),
            self.create_rectangle(0, r, w, h - r, **kw),
        ]
        self.text_id = self.create_text(w // 2, h // 2, text=self.text_val,
                                        fill=self.current_fg, font=self.font)

    def _recolor(self, bg_col, fg_col):
        self.current_bg = bg_col
        self.current_fg = fg_col
        for s in self.shapes:
            self.itemconfig(s, fill=self.current_bg)
        if self.text_id:
            self.itemconfig(self.text_id, fill=self.current_fg)

    def _on_release(self, e):
        self._recolor(self.hover_bg, self.fg_color)
        if -15 <= e.x <= self.winfo_width() + 15 and -15 <= e.y <= self.winfo_height() + 15:
            if self.cmd:
                self.cmd()

    def update_colors(self, new_bg, new_fg=T.LABEL):
        self.default_bg = new_bg
        self.fg_color = new_fg
        self.hover_bg = T.lighten(new_bg)
        self._recolor(self.default_bg, self.fg_color)

    def update_text(self, new_text):
        self.text_val = new_text
        if self.text_id:
            self.itemconfig(self.text_id, text=self.text_val)


class AppleSegmentedControl(tk.Canvas):
    """iOS segmented control — a rounded pill with a sliding active highlight."""

    def __init__(self, parent, items, default_item, command, width=220, height=38,
                 radius=8, parent_bg=T.SURFACE, accent=T.ORANGE, **kwargs):
        super().__init__(parent, width=width, height=height, bg=parent_bg,
                         highlightthickness=0, **kwargs)
        self.items = items
        self.active_item = default_item
        self.command = command
        self.radius = radius
        self.bg_color = T.ELEVATED
        self.active_color = accent
        self.bind("<Configure>", self._draw)
        self.bind("<Button-1>", self._on_click)

    def _draw(self, event=None):
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        if w <= 1:
            w = int(self.cget("width"))
        if h <= 1:
            h = int(self.cget("height"))
        self._rrect(0, 0, w, h, self.radius, self.bg_color)
        idx = self.items.index(self.active_item)
        seg_w = w / len(self.items)
        pad = 3
        self._rrect(idx * seg_w + pad, pad, (idx + 1) * seg_w - pad, h - pad,
                    self.radius - 2, self.active_color)
        for i, item in enumerate(self.items):
            self.create_text(i * seg_w + seg_w / 2, h / 2, text=item,
                             fill=T.LABEL, font=T.FONT_BODY)

    def _rrect(self, x0, y0, x1, y1, r, color):
        self.create_oval(x0, y0, x0 + 2 * r, y0 + 2 * r, fill=color, outline="")
        self.create_oval(x1 - 2 * r, y0, x1, y0 + 2 * r, fill=color, outline="")
        self.create_oval(x0, y1 - 2 * r, x0 + 2 * r, y1, fill=color, outline="")
        self.create_oval(x1 - 2 * r, y1 - 2 * r, x1, y1, fill=color, outline="")
        self.create_rectangle(x0 + r, y0, x1 - r, y1, fill=color, outline="")
        self.create_rectangle(x0, y0 + r, x1, y1 - r, fill=color, outline="")

    def set_active(self, item):
        if item in self.items and item != self.active_item:
            self.active_item = item
            self._draw()

    def _on_click(self, event):
        w = self.winfo_width()
        if w <= 1:
            w = int(self.cget("width"))
        seg_w = w / len(self.items)
        idx = int(event.x // seg_w)
        if 0 <= idx < len(self.items):
            selected = self.items[idx]
            if selected != self.active_item:
                self.active_item = selected
                self._draw()
                self.command(selected)


class AppleToggle(tk.Canvas):
    """iOS switch — a sliding knob inside a pill that turns green when ON."""

    def __init__(self, parent, command=None, initial=False, width=52, height=31,
                 parent_bg=T.SURFACE, on_color=T.GREEN, **kwargs):
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
        track = self.on_color if self.state else T.ELEVATED
        # rounded track
        self.create_oval(0, 0, h, h, fill=track, outline="")
        self.create_oval(w - h, 0, w, h, fill=track, outline="")
        self.create_rectangle(r, 0, w - r, h, fill=track, outline="")
        # knob
        pad = 2
        kr = r - pad
        cx = (w - r) if self.state else r
        self.create_oval(cx - kr, pad, cx + kr, h - pad, fill=T.LABEL, outline="")

    def set(self, state: bool, fire=False):
        self.state = bool(state)
        self._draw()
        if fire and self.command:
            self.command(self.state)

    def get(self) -> bool:
        return self.state

    def _on_click(self, event):
        self.state = not self.state
        self._draw()
        if self.command:
            self.command(self.state)


class Card(tk.Frame):
    """A flat rounded-feel surface container with consistent inner padding."""

    def __init__(self, parent, padx=14, pady=14, bg=T.SURFACE, **kwargs):
        super().__init__(parent, bg=bg, padx=padx, pady=pady, **kwargs)
        self.bg = bg


class StatusDot(tk.Label):
    """A small colored bullet used for live status indication."""

    def __init__(self, parent, color=T.LABEL2, bg=T.SURFACE, size=16):
        super().__init__(parent, text="●", fg=color, bg=bg,
                         font=("Helvetica", size))

    def set_color(self, color):
        self.config(fg=color)
