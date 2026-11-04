"""base.py — Mobile-base teleop (MAXSwerve, /mabel/hw/base/cmd_vel).

Two columns: LEFT a live telemetry sidebar (the 6 swerve joints — steer angle + drive
velocity), RIGHT the twist controls — vx/vy/wz sliders, WASD+Q/E hold-to-drive, E-STOP.
Frame is the real robot frame (+x fwd, +y left, wz CCW). Key-up zeroes (a dead commander
never latches motion); the bridge re-sends the setpoint to feed the Teensy watchdog.
"""

import tkinter as tk

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import contract
from style import (BG, ELEVATED, GREEN, LABEL, LABEL2, ORANGE, PANEL_BG, RED, SEP,
                   FONT_MONO, FONT_SMALL, FONT_SUB, accent_btn, btn)
from widgets import FloatSlider, TelemetryColumn

_BASE_JOINTS = [("b_steer", -3.1416, 3.1416), ("fr_steer", -3.1416, 3.1416),
                ("fl_steer", -3.1416, 3.1416), ("b_drive", -8.0, 8.0),
                ("fr_drive", -8.0, 8.0), ("fl_drive", -8.0, 8.0)]


class BasePanel(tk.Frame):
    def __init__(self, parent, link):
        super().__init__(parent, bg=BG)
        self.link = link
        self.vmax = contract.BASE["vmax"]
        self.wmax = contract.BASE["wmax"]
        self._estop = False
        self._keys = set()

        self.tele = TelemetryColumn(self, link, width=330)
        self.tele.pack(side="left", fill="y")
        self.tele.show("Mobile Base", _BASE_JOINTS, "base")
        tk.Frame(self, bg=SEP, width=1).pack(side="left", fill="y")

        right = tk.Frame(self, bg=BG)
        right.pack(side="left", fill="both", expand=True)
        tk.Label(right, text="Mobile Base", bg=BG, fg=LABEL, font=FONT_SUB).pack(
            anchor="w", padx=16, pady=(10, 2))
        tk.Label(right, text="Twist in the real frame (+x fwd, +y left). Click the pad, "
                 "then hold W/A/S/D + Q/E to drive; release to stop.",
                 bg=BG, fg=LABEL2, font=FONT_SMALL, anchor="w",
                 wraplength=620, justify="left").pack(fill="x", padx=16)

        card = tk.Frame(right, bg=PANEL_BG)
        card.pack(fill="x", padx=16, pady=12)
        self._sliders = {}
        for key, label, mx, unit in (("vx", "Forward  vx", self.vmax, "m/s"),
                                     ("vy", "Left  vy", self.vmax, "m/s"),
                                     ("wz", "Yaw  wz", self.wmax, "rad/s")):
            row = tk.Frame(card, bg=PANEL_BG)
            row.pack(fill="x", padx=14, pady=7)
            top = tk.Frame(row, bg=PANEL_BG)
            top.pack(fill="x")
            tk.Label(top, text=label, bg=PANEL_BG, fg=LABEL, font=FONT_SMALL).pack(side="left")
            val = tk.Label(top, text=f"0.00 {unit}", bg=PANEL_BG, fg=ORANGE, font=FONT_MONO)
            val.pack(side="right")
            sl = FloatSlider(row, -mx, mx, width=460, height=24, bg=PANEL_BG,
                             command=lambda v, k=key: self._on_slide(k, v))
            sl.pack(fill="x", pady=(2, 0))
            self._sliders[key] = (sl, val, unit)

        bar = tk.Frame(right, bg=BG)
        bar.pack(fill="x", padx=16)
        self._estop_btn = accent_btn(bar, "E-STOP", self._toggle_estop, color=RED, parent_bg=BG)
        self._estop_btn.pack(side="left")
        btn(bar, "Zero", self._zero, parent_bg=BG).pack(side="left", padx=8)

        pad = tk.Frame(right, bg=ELEVATED, height=110)
        pad.pack(fill="x", padx=16, pady=12)
        pad.pack_propagate(False)
        self._pad_lbl = tk.Label(pad, text="click here, then hold W/A/S/D · Q/E to drive",
                                 bg=ELEVATED, fg=LABEL2, font=FONT_SMALL)
        self._pad_lbl.pack(expand=True)
        pad.bind("<Button-1>", lambda e: pad.focus_set())
        pad.bind("<KeyPress>", self._key_down)
        pad.bind("<KeyRelease>", self._key_up)
        pad.bind("<FocusIn>", lambda e: self._pad_lbl.config(fg=GREEN, text="DRIVING — hold keys"))
        pad.bind("<FocusOut>", lambda e: (self._keys.clear(), self._apply_keys(),
                 self._pad_lbl.config(fg=LABEL2,
                                      text="click here, then hold W/A/S/D · Q/E to drive")))

    def _send(self):
        if self._estop:
            return
        self.link.base_twist(self._sliders["vx"][0].get(), self._sliders["vy"][0].get(),
                             self._sliders["wz"][0].get())

    def _on_slide(self, key, v):
        sl, val, unit = self._sliders[key]
        val.config(text=f"{v:+.2f} {unit}")
        self._send()

    def _set(self, key, v):
        sl, val, unit = self._sliders[key]
        sl.set(v)
        val.config(text=f"{sl.get():+.2f} {unit}")

    def _zero(self):
        for key in self._sliders:
            self._set(key, 0.0)
        self.link.base_twist(0.0, 0.0, 0.0)

    def _toggle_estop(self):
        self._estop = not self._estop
        self.link.base_estop(self._estop)
        if self._estop:
            self._zero()
            self._estop_btn.update_colors(GREEN, LABEL)
            self._estop_btn.update_text("RELEASE")
        else:
            self._estop_btn.update_colors(RED, LABEL)
            self._estop_btn.update_text("E-STOP")

    def _key_down(self, e):
        self._keys.add(e.keysym.lower())
        self._apply_keys()

    def _key_up(self, e):
        self._keys.discard(e.keysym.lower())
        self._apply_keys()

    def _apply_keys(self):
        k = self._keys
        self._set("vx", (("w" in k) - ("s" in k)) * self.vmax)
        self._set("vy", (("a" in k) - ("d" in k)) * self.vmax)
        self._set("wz", (("q" in k) - ("e" in k)) * self.wmax)
        self._send()

    def refresh(self, snapshot):
        self.tele.refresh(snapshot)
