"""widgets.py - float-valued controls for the control center, in the iOS style.

The vendored ``style.AppleSlider`` quantises to integers (it was built for raw motor
ticks), so this adds a float slider for radian / metre joint targets, plus a compact
``JointRow`` that pairs a labelled float slider with a live read-back ``Bar`` - the row
used by every joint-control panel.
"""

import tkinter as tk

from style import (Bar, ELEVATED, GREEN, LABEL, LABEL2, LABEL3, ORANGE, PANEL_BG,
                   RED, SURFACE, FONT_MONO_S, FONT_SMALL, round_rect)


class LiftGauge(tk.Canvas):
    """Vertical rounded fill gauge (the Lift Studio hero): an orange column that fills
    from the bottom to the lift's height fraction of [lo, hi], with a cursor line at the
    current level. ``set(value, lo, hi)`` redraws in place."""

    def __init__(self, parent, width=48, height=250, bg=PANEL_BG, **kw):
        super().__init__(parent, width=width, height=height, bg=bg,
                         highlightthickness=0, **kw)
        self._cw, self._ch = width, height        # NOT _w/_h — tkinter owns self._w
        self.set(None, 0.0, 1.0)

    def set(self, value, lo, hi):
        self.delete("all")
        w, h, pad, tw = self._cw, self._ch, 6, 22
        x0, x1 = (w - tw) / 2, (w + tw) / 2
        r = tw / 2
        round_rect(self, x0, pad, x1, h - pad, r, fill=ELEVATED, outline="")
        if value is None or hi <= lo:
            return
        frac = max(0.0, min(1.0, (value - lo) / (hi - lo)))
        fill_h = frac * (h - 2 * pad)
        y_top = (h - pad) - fill_h
        if fill_h > 2:
            round_rect(self, x0, y_top, x1, h - pad, min(r, fill_h / 2),
                       fill=ORANGE, outline="")
        self.create_line(x0 - 5, y_top, x1 + 5, y_top, fill=LABEL, width=2)


class FloatSlider(tk.Canvas):
    """A vector-drawn iOS slider carrying a float value in [from_, to]."""

    def __init__(self, parent, from_, to, command=None, width=240, height=26,
                 bg=None, **kw):
        bg = bg or parent.cget("bg")
        super().__init__(parent, width=width, height=height, bg=bg,
                         highlightthickness=0, **kw)
        self.min_val = float(min(from_, to))
        self.max_val = float(max(from_, to))
        self.val = 0.0 if self.min_val <= 0.0 <= self.max_val else self.min_val
        self.command = command
        self.padding = 14
        self.track_y = height // 2
        self.id_trough = self.create_line(0, 0, 0, 0, fill=ELEVATED, width=6, capstyle=tk.ROUND)
        self.id_fill = self.create_line(0, 0, 0, 0, fill=ORANGE, width=6, capstyle=tk.ROUND)
        self.id_thumb = self.create_oval(0, 0, 0, 0, fill=LABEL, outline="")
        self.bind("<Configure>", lambda e: self._redraw())
        self.bind("<Button-1>", self._on_drag)
        self.bind("<B1-Motion>", self._on_drag)

    def set(self, val):
        self.val = max(self.min_val, min(self.max_val, float(val)))
        self._redraw()

    def get(self):
        return self.val

    def _redraw(self):
        w = self.winfo_width() or int(self.cget("width"))
        span = self.max_val - self.min_val
        pct = 0.0 if span == 0 else (self.val - self.min_val) / span
        x = self.padding + pct * (w - 2 * self.padding)
        self.coords(self.id_trough, self.padding, self.track_y, w - self.padding, self.track_y)
        self.coords(self.id_fill, self.padding, self.track_y, x, self.track_y)
        r = 11
        self.coords(self.id_thumb, x - r, self.track_y - r, x + r, self.track_y + r)

    def _on_drag(self, event):
        w = self.winfo_width() or int(self.cget("width"))
        x = max(self.padding, min(w - self.padding, event.x))
        pct = (x - self.padding) / (w - 2 * self.padding)
        self.val = self.min_val + pct * (self.max_val - self.min_val)
        self._redraw()
        if self.command:
            self.command(self.val)


class JointRow(tk.Frame):
    """One joint: name + target slider + live value read-out + position bar.

    ``on_change(name, value)`` fires while the user drags the target slider. Call
    ``update_live(pos, vel, eff)`` from the telemetry poll to refresh the read-back
    (the bar fills to the joint's fraction of its [lo, hi] range)."""

    def __init__(self, parent, name, lo, hi, on_change, unit="rad", width=300):
        super().__init__(parent, bg=PANEL_BG)
        self.name = name
        self.lo, self.hi = float(lo), float(hi)
        self.unit = unit
        self._on_change = on_change

        top = tk.Frame(self, bg=PANEL_BG)
        top.pack(fill="x")
        tk.Label(top, text=name, bg=PANEL_BG, fg=LABEL, font=FONT_SMALL,
                 anchor="w").pack(side="left")
        self._live = tk.Label(top, text="-", bg=PANEL_BG, fg=LABEL3,
                              font=FONT_MONO_S, anchor="e")
        self._live.pack(side="right")
        self._tgt = tk.Label(top, text="", bg=PANEL_BG, fg=ORANGE,
                             font=FONT_MONO_S, anchor="e")
        self._tgt.pack(side="right", padx=(0, 10))

        mid = tk.Frame(self, bg=PANEL_BG)
        mid.pack(fill="x", pady=(2, 0))
        self.slider = FloatSlider(mid, lo, hi, command=self._changed,
                                  width=width, height=24, bg=PANEL_BG)
        self.slider.pack(side="left", fill="x", expand=True)

        self.bar = Bar(self, width=width, height=6, bg=PANEL_BG)
        self.bar.pack(fill="x", pady=(2, 4))
        self._update_tgt(self.slider.get())

    def _changed(self, v):
        self._update_tgt(v)
        if self._on_change:
            self._on_change(self.name, v)

    def _update_tgt(self, v):
        self._tgt.config(text=f"{v:+.2f}")

    def set_target(self, v):
        self.slider.set(v)
        self._update_tgt(self.slider.get())

    def target(self):
        return self.slider.get()

    def update_live(self, pos, vel=0.0, eff=0.0):
        if pos is None:
            self._live.config(text="-", fg=LABEL3)
            self.bar.set(0.0, LABEL3)
            return
        span = self.hi - self.lo
        frac = 0.0 if span == 0 else (pos - self.lo) / span
        color = RED if abs(eff) > 0.8 else (ORANGE if abs(eff) > 0.5 else GREEN)
        self.bar.set(frac, color)
        self._live.config(text=f"{pos:+.2f} {self.unit}", fg=LABEL)


def _temp_color(t):
    return RED if t >= 55 else (ORANGE if t >= 45 else LABEL2)


def _eff_color(eff):
    a = abs(eff)
    return RED if a > 0.8 else (ORANGE if a > 0.5 else GREEN)


class TelemetryRow(tk.Frame):
    """One compact read-only joint readout: name · position bar · numeric position ·
    velocity (v) · torque/effort (t) · temperature. Fixed-width columns so a stack of
    rows aligns into a clean table. ``show_vel`` / ``show_temp`` drop those columns for
    subsystems that don't report them (arms/body/base have no temperature)."""

    def __init__(self, parent, name, lo, hi, unit="rad", bar_w=120, name_w=13,
                 show_vel=True, show_temp=True):
        super().__init__(parent, bg=PANEL_BG)
        self.lo, self.hi, self.unit = float(lo), float(hi), unit
        self._span = self.hi - self.lo

        tk.Label(self, text=name, bg=PANEL_BG, fg=LABEL2, font=FONT_MONO_S,
                 width=name_w, anchor="w").pack(side="left")
        self.bar = Bar(self, width=bar_w, height=6, bg=PANEL_BG)
        self.bar.pack(side="left", padx=(2, 8))
        self._pos = tk.Label(self, text="-", bg=PANEL_BG, fg=LABEL3, font=FONT_MONO_S,
                             width=9, anchor="e")
        self._pos.pack(side="left")
        self._vel = self._eff = self._temp = None
        if show_vel:
            self._vel = tk.Label(self, text="v -", bg=PANEL_BG, fg=LABEL3,
                                 font=FONT_MONO_S, width=8, anchor="e")
            self._vel.pack(side="left", padx=(6, 0))
        self._eff = tk.Label(self, text="t -", bg=PANEL_BG, fg=LABEL3, font=FONT_MONO_S,
                             width=8, anchor="e")
        self._eff.pack(side="left", padx=(6, 0))
        if show_temp:
            self._temp = tk.Label(self, text="-", bg=PANEL_BG, fg=LABEL3,
                                  font=FONT_MONO_S, width=5, anchor="e")
            self._temp.pack(side="left", padx=(6, 0))

    def update(self, pos, vel=0.0, eff=0.0, temp=None):
        if pos is None:
            self.bar.set(0.0, LABEL3)
            self._pos.config(text="-", fg=LABEL3)
            if self._vel is not None:
                self._vel.config(text="v -", fg=LABEL3)
            self._eff.config(text="t -", fg=LABEL3)
            if self._temp is not None:
                self._temp.config(text="-", fg=LABEL3)
            return
        frac = 0.0 if self._span == 0 else (pos - self.lo) / self._span
        self.bar.set(frac, _eff_color(eff))
        self._pos.config(text=f"{pos:+.2f}", fg=LABEL)
        if self._vel is not None:
            self._vel.config(text=f"v{vel:+.2f}", fg=GREEN if abs(vel) > 0.02 else LABEL3)
        self._eff.config(text=f"t{eff:+.2f}", fg=_eff_color(eff) if abs(eff) > 0.5 else LABEL2)
        if self._temp is not None:
            self._temp.config(text=f"{temp:.0f}°C" if temp else "-",
                              fg=_temp_color(temp) if temp else LABEL3)


class CommandRow(tk.Frame):
    """Compact one-line joint command: name · slider · target value. No live read-back
    here (the TelemetryColumn shows that), so a whole hand fits on one page."""

    def __init__(self, parent, name, lo, hi, on_change, name_w=11):
        super().__init__(parent, bg=PANEL_BG)
        self.name = name
        self.lo, self.hi = float(lo), float(hi)
        self._on = on_change
        tk.Label(self, text=name, bg=PANEL_BG, fg=LABEL, font=FONT_MONO_S,
                 width=name_w, anchor="w").pack(side="left")
        self._val = tk.Label(self, text="", bg=PANEL_BG, fg=ORANGE, font=FONT_MONO_S,
                             width=7, anchor="e")
        self._val.pack(side="right")
        self.slider = FloatSlider(self, lo, hi, command=self._changed, width=170,
                                  height=20, bg=PANEL_BG)
        self.slider.pack(side="left", fill="x", expand=True, padx=(6, 6))
        self._upd(self.slider.get())

    def _changed(self, v):
        self._upd(v)
        if self._on:
            self._on(self.name, v)

    def _upd(self, v):
        self._val.config(text=f"{v:+.2f}")

    def set_target(self, v):
        self.slider.set(v)
        self._upd(self.slider.get())

    def target(self):
        return self.slider.get()


class TelemetryColumn(tk.Frame):
    """Reusable LEFT sidebar: live read-only telemetry for ONE module — a header
    (●online · title · summary) over a scrolling column of ``TelemetryRow``s. The same
    look as the Overview, scoped to whatever module is active. ``show()`` (re)builds the
    rows for a joint set (called again when the hand side switches); ``refresh(snap)``
    updates them from the bridge snapshot."""

    def __init__(self, parent, link, width=330):
        super().__init__(parent, bg=PANEL_BG, width=width)
        self.link = link
        self.pack_propagate(False)
        head = tk.Frame(self, bg=PANEL_BG)
        head.pack(fill="x", padx=12, pady=(10, 2))
        self._dot = tk.Label(head, text="●", bg=PANEL_BG, fg=LABEL3, font=("Helvetica", 11))
        self._dot.pack(side="left", padx=(0, 6))
        self._title = tk.Label(head, text="", bg=PANEL_BG, fg=LABEL, font=FONT_SMALL)
        self._title.pack(side="left")
        self._sum = tk.Label(head, text="", bg=PANEL_BG, fg=LABEL3, font=FONT_MONO_S, anchor="e")
        self._sum.pack(side="right")
        tk.Label(self, text="LIVE TELEMETRY", bg=PANEL_BG, fg=LABEL2,
                 font=FONT_MONO_S).pack(anchor="w", padx=12, pady=(0, 2))
        self._holder = tk.Frame(self, bg=PANEL_BG)
        self._holder.pack(fill="both", expand=True)
        self._rows = {}
        self._key = self._side = None

    def show(self, title, joints, key, side=None, unit="rad", with_temp=False):
        self._title.config(text=title)
        self._key, self._side = key, side
        for w in self._holder.winfo_children():
            w.destroy()
        self._rows = {}
        outer, inner = scroll_column(self._holder, bg=PANEL_BG)
        outer.pack(fill="both", expand=True)
        for (n, lo, hi) in joints:
            r = TelemetryRow(inner, n, lo, hi, unit=unit, bar_w=90, name_w=11,
                             show_temp=with_temp)
            r.pack(fill="x", padx=10, pady=1)
            self._rows[n] = r

    def refresh(self, snap):
        if self._key is None:
            return
        temps, extra = {}, ""
        if self._side is not None:                              # a hand
            d = snap["hands"][self._side]
            joints = d.get("joints", {}); temps = d.get("temps", {})
            online = self.link.live("hands", self._side)
            tq = d.get("torque")
            extra = " · torque ON" if tq else (" · free-drive" if tq is not None else "")
        elif self._key == "lift":
            pos = snap["lift"].get("pos")
            joints = {"lift": (pos, 0.0, 0.0)} if pos is not None else {}
            online = self.link.live("lift")
        else:
            joints = snap.get(self._key, {}).get("joints", {})
            online = self.link.live(self._key)
        self._dot.config(fg=GREEN if online else LABEL3)
        n = 0
        for name, row in self._rows.items():
            v = joints.get(name)
            if v is None:
                row.update(None)
            else:
                pos, vel, eff = (list(v) + [0.0, 0.0])[:3]
                row.update(pos, vel, eff, temps.get(name))
                n += 1
        self._sum.config(text=f"{n}/{len(self._rows)}{extra}")


def scroll_column(parent, bg=PANEL_BG):
    """A vertically scrollable column. Returns (outer_frame, inner_frame); pack rows
    into inner. Mouse-wheel scrolls when the pointer is over the column."""
    outer = tk.Frame(parent, bg=bg)
    canvas = tk.Canvas(outer, bg=bg, highlightthickness=0)
    sb = tk.Scrollbar(outer, orient="vertical", command=canvas.yview,
                      bg=bg, troughcolor=bg)
    sb.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)
    inner = tk.Frame(canvas, bg=bg)
    win = canvas.create_window((0, 0), window=inner, anchor="nw")
    inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.bind("<Configure>", lambda e: canvas.itemconfig(win, width=e.width))
    canvas.configure(yscrollcommand=sb.set)

    def _wheel(e):
        canvas.yview_scroll(int(-1 * (e.delta / 120)) if e.delta else
                            (-1 if e.num == 4 else 1), "units")
    canvas.bind("<Enter>", lambda e: [canvas.bind_all(s, _wheel)
                                      for s in ("<MouseWheel>", "<Button-4>", "<Button-5>")])
    canvas.bind("<Leave>", lambda e: [canvas.unbind_all(s)
                                      for s in ("<MouseWheel>", "<Button-4>", "<Button-5>")])
    return outer, inner
