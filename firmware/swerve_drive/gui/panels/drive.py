"""Drive panel — per-module motor control.

One card per swerve module.  Each card carries:
    * a steering sub-control: enable toggle + position slider (absolute rot.)
    * a drive sub-control:    enable toggle + bidirectional velocity slider
    * live position / velocity / current readouts under each slider

A footer bar gives global Enable-All / Disable-All and a prominent E-STOP.
"""

import tkinter as tk

from .. import config as C
from .. import theme as T
from ..widgets import AppleButton, AppleSlider, AppleToggle, StatusDot
from .base import Panel


class DrivePanel(Panel):
    title = "Drive"
    accent = T.ORANGE

    def __init__(self, host, app):
        super().__init__(host, app)
        self._steer_sliders = {}   # steer_id -> AppleSlider
        self._drive_sliders = {}   # drive_id -> AppleSlider
        self._toggles = {}         # can_id  -> AppleToggle
        self._readouts = {}        # can_id  -> dict of StringVars
        self._dots = {}            # can_id  -> StatusDot
        self._build()

    # ── Construction ───────────────────────────────────────────────────────────
    def _build(self):
        grid = tk.Frame(self, bg=T.BG)
        grid.pack(fill="both", expand=True, padx=22, pady=22)

        n = len(C.MODULES)
        cols = min(3, n)                 # up to 3 modules across (FL · FR · Back)
        rows = (n + cols - 1) // cols
        for c in range(cols):
            grid.grid_columnconfigure(c, weight=1, uniform="mod")
        for r in range(rows):
            grid.grid_rowconfigure(r, weight=1, uniform="mod")

        for i, mod in enumerate(C.MODULES):
            card = self._module_card(grid, mod)
            card.grid(row=i // cols, column=i % cols, sticky="nsew", padx=10, pady=10)

        self._footer()

    def _module_card(self, parent, mod):
        card = tk.Frame(parent, bg=T.SURFACE)
        inner = tk.Frame(card, bg=T.SURFACE)
        inner.pack(fill="both", expand=True, padx=18, pady=16)

        # Header: abbreviation badge + full name
        header = tk.Frame(inner, bg=T.SURFACE)
        header.pack(fill="x", pady=(0, 14))
        tk.Label(header, text=mod["abbr"], bg=T.ORANGE, fg=T.BG, font=T.FONT_SUB,
                 padx=10, pady=2).pack(side="left")
        tk.Label(header, text=mod["name"], bg=T.SURFACE, fg=T.LABEL,
                 font=T.FONT_SUB).pack(side="left", padx=12)
        tk.Label(header, text=f"D{mod['drive_id']} · S{mod['steer_id']}",
                 bg=T.SURFACE, fg=T.LABEL2, font=T.FONT_MONO_S).pack(side="right")

        steer_mid = (C.STEER_POS_MIN + C.STEER_POS_MAX) / 2.0
        self._sub_control(inner, "STEERING", mod["steer_id"], C.STEER,
                          C.STEER_POS_MIN, C.STEER_POS_MAX, steer_mid,
                          T.PURPLE, lambda v: f"cmd {C.cmd_to_deg(v):3.0f}°")
        tk.Frame(inner, bg=T.SEP, height=1).pack(fill="x", pady=12)

        if C.DRIVE_USE_DUTY:
            lo, hi, fmt = -C.DRIVE_DUTY_MAX, C.DRIVE_DUTY_MAX, lambda v: f"cmd {v * 100:+.0f}%"
        else:
            lo, hi, fmt = -C.DRIVE_VEL_MAX, C.DRIVE_VEL_MAX, lambda v: f"cmd {v:+.2f} rev/s"
        self._sub_control(inner, "DRIVE", mod["drive_id"], C.DRIVE,
                          lo, hi, 0.0, T.ORANGE, fmt)
        return card

    def _sub_control(self, parent, label, can_id, role, lo, hi, default, accent, fmt):
        wrap = tk.Frame(parent, bg=T.SURFACE)
        wrap.pack(fill="x")

        top = tk.Frame(wrap, bg=T.SURFACE)
        top.pack(fill="x")
        dot = StatusDot(top, color=T.LABEL2, bg=T.SURFACE, size=12)
        dot.pack(side="left", padx=(0, 6))
        self._dots[can_id] = dot
        tk.Label(top, text=label, bg=T.SURFACE, fg=T.LABEL, font=T.FONT_BODY).pack(side="left")
        tk.Label(top, text=f"ID {can_id}", bg=T.SURFACE, fg=T.LABEL2,
                 font=T.FONT_SMALL).pack(side="left", padx=8)

        toggle = AppleToggle(top, parent_bg=T.SURFACE,
                             command=lambda on, cid=can_id: self._on_toggle(cid, on))
        toggle.pack(side="right")
        self._toggles[can_id] = toggle

        slider = AppleSlider(wrap, from_=lo, to=hi, height=26, accent=accent,
                             command=lambda v, cid=can_id: self._on_slider(cid, v))
        slider.set(default)
        slider.pack(fill="x", pady=(10, 6))
        if role == C.STEER:
            self._steer_sliders[can_id] = slider
        else:
            self._drive_sliders[can_id] = slider

        # Readout row: commanded target (left) · live telemetry (right)
        rowf = tk.Frame(wrap, bg=T.SURFACE)
        rowf.pack(fill="x")
        v_cmd = tk.StringVar(value=fmt(default))
        tk.Label(rowf, textvariable=v_cmd, bg=T.SURFACE, fg=accent,
                 font=T.FONT_MONO_S, anchor="w").pack(side="left")
        v_live = tk.StringVar(value="fb --  cur --")
        l_live = tk.Label(rowf, textvariable=v_live, bg=T.SURFACE, fg=T.LABEL2,
                          font=T.FONT_MONO_S, anchor="e")
        l_live.pack(side="right")
        self._readouts[can_id] = {"cmd": v_cmd, "live": v_live, "live_lbl": l_live,
                                  "fmt": fmt, "role": role, "default": default}

    def _footer(self):
        tk.Frame(self, bg=T.SEP, height=1).pack(fill="x")
        bar = tk.Frame(self, bg=T.SURFACE, height=84)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)

        left = tk.Frame(bar, bg=T.SURFACE)
        left.pack(side="left", padx=24, pady=22)
        AppleButton(left, "Enable All", lambda: self._set_all(True),
                    bg_color=T.GREEN, fg_color=T.BG, parent_bg=T.SURFACE,
                    width=120, height=40).pack(side="left", padx=(0, 10))
        AppleButton(left, "Disable All", lambda: self._set_all(False),
                    bg_color=T.ELEVATED, parent_bg=T.SURFACE,
                    width=120, height=40).pack(side="left")

        AppleButton(bar, "E-STOP", self._estop, bg_color=T.RED, parent_bg=T.SURFACE,
                    width=150, height=46, font=T.FONT_SUB).pack(side="right", padx=24, pady=19)

    # ── Handlers ───────────────────────────────────────────────────────────────
    def _push(self, can_id):
        """Send the slider's current value to the bus and refresh the cmd label."""
        sl = self._steer_sliders.get(can_id) or self._drive_sliders.get(can_id)
        if sl is not None:
            val = sl.get()
            self.bus.set_setpoint(can_id, val)
            ro = self._readouts.get(can_id)
            if ro is not None:
                ro["cmd"].set(ro["fmt"](val))

    def _on_toggle(self, can_id, on):
        self.bus.set_enabled(can_id, on)
        if on:
            self._push(can_id)        # sync setpoint + label so it doesn't lurch

    def _on_slider(self, can_id, value):
        ro = self._readouts[can_id]
        ro["cmd"].set(ro["fmt"](value))
        self.bus.set_setpoint(can_id, value)

    def _set_all(self, on):
        for cid, tog in self._toggles.items():
            tog.set(on)
            self.bus.set_enabled(cid, on)
            if on:
                self._push(cid)

    def _estop(self):
        self.bus.stop_all()
        for cid, tog in self._toggles.items():
            tog.set(False)
        for cid, sl in self._drive_sliders.items():
            sl.set(0.0)
            ro = self._readouts[cid]
            ro["cmd"].set(ro["fmt"](0.0))

    # ── Live telemetry ─────────────────────────────────────────────────────────
    def on_telemetry(self, snap):
        for can_id, st in snap.items():
            ro = self._readouts.get(can_id)
            if ro is None:
                continue
            cur = "--" if st.current != st.current else f"{st.current:.1f}A"
            if ro["role"] == C.STEER:
                # Encoder feedback in degrees, aligned to the commanded frame.
                # "—" when status frame 5 isn't actually broadcasting (stale),
                # so the controller-internal loop's commanded angle (left) is
                # always the trustworthy number.
                if st.pos_fresh:
                    fb = f"fb {C.encoder_to_deg(can_id, st.pos):3.0f}°"
                    ro["live_lbl"].config(fg=T.LABEL2)
                else:
                    fb = "fb —"
                    ro["live_lbl"].config(fg=T.LABEL3)
                ro["live"].set(f"{fb}  {cur}")
            else:
                # Drive velocity feedback, in wheel m/s (frame 2). "—" when the
                # frame isn't broadcasting, so it never shows a misleading 0.
                if st.vel_fresh:
                    vel = f"vel {C.motor_rpm_to_mps(st.vel):+.2f}m/s"
                    ro["live_lbl"].config(fg=T.LABEL2)
                else:
                    vel = "vel —"
                    ro["live_lbl"].config(fg=T.LABEL3)
                ro["live"].set(f"{vel}  {cur}")
            dot = self._dots.get(can_id)
            if dot is not None:
                dot.set_color(T.GREEN if st.enabled else (T.LABEL2 if st.online else T.RED))
