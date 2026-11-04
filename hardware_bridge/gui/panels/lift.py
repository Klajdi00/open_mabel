"""lift.py — Lift control, in the look of the firmware Lift Studio.

LEFT: the signature Lift sidebar — a big height readout, the vertical LiftGauge, and
status rows (height / velocity / target / range). RIGHT: absolute positioning —
Park/Bottom/Top presets, a Target Height slider, Go to Target, and a Calibration card
that runs the on-Pico stall-seek and shows the measured travel envelope. Everything
routes through the bridge (/lift/command + /lift/calibrate out, /lift/joint_states +
/lift/limits in); velocity/jog is still omitted rather than faked.
"""

import tkinter as tk
from tkinter import messagebox

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import contract
from style import (BG, GREEN, LABEL, LABEL2, LABEL3, ORANGE, PANEL_BG, SEP, SURFACE,
                   FONT_MONO, FONT_MONO_S, FONT_SMALL, FONT_SUB, FONT_TITLE,
                   accent_btn, btn)
from widgets import FloatSlider, LiftGauge


class LiftPanel(tk.Frame):
    def __init__(self, parent, link):
        super().__init__(parent, bg=BG)
        self.link = link
        self._lo, self._hi = contract.LIFT["min_m"], contract.LIFT["max_m"]

        # ── left: gauge sidebar ──────────────────────────────────────────────
        side = tk.Frame(self, bg=PANEL_BG, width=300)
        side.pack(side="left", fill="y")
        side.pack_propagate(False)
        tk.Label(side, text="LIVE TELEMETRY", bg=PANEL_BG, fg=LABEL2,
                 font=FONT_SMALL).pack(anchor="w", padx=18, pady=(18, 6))

        self._big = tk.Label(side, text="—", bg=PANEL_BG, fg=LABEL,
                             font=(FONT_TITLE[0], 40, "bold"))
        self._big.pack(anchor="w", padx=18)
        tk.Label(side, text="HEIGHT (m)", bg=PANEL_BG, fg=LABEL3,
                 font=FONT_MONO_S).pack(anchor="w", padx=18, pady=(0, 8))

        mid = tk.Frame(side, bg=PANEL_BG)
        mid.pack(fill="x", padx=18, pady=(2, 8))
        self._gauge = LiftGauge(mid, width=48, height=250, bg=PANEL_BG)
        self._gauge.pack(side="left")
        self._rows = {}
        rowbox = tk.Frame(mid, bg=PANEL_BG)
        rowbox.pack(side="left", fill="x", expand=True, padx=(16, 0), anchor="n")
        for key, lbl in (("height", "Height"), ("velocity", "Velocity"),
                         ("target", "Target"), ("range", "Range"),
                         ("status", "Status")):
            r = tk.Frame(rowbox, bg=PANEL_BG)
            r.pack(fill="x", pady=3)
            tk.Label(r, text=lbl, bg=PANEL_BG, fg=LABEL2, font=FONT_SMALL,
                     width=8, anchor="w").pack(side="left")
            v = tk.Label(r, text="—", bg=PANEL_BG, fg=LABEL, font=FONT_MONO_S, anchor="w")
            v.pack(side="left")
            self._rows[key] = v
        self._rows["range"].config(text=f"{self._lo:.2f}–{self._hi:.2f} m")

        tk.Frame(self, bg=SEP, width=1).pack(side="left", fill="y")

        # ── right: absolute positioning ──────────────────────────────────────
        right = tk.Frame(self, bg=BG)
        right.pack(side="left", fill="both", expand=True)
        tk.Label(right, text="Lift", bg=BG, fg=LABEL, font=FONT_SUB).pack(
            anchor="w", padx=18, pady=(12, 2))
        tk.Label(right, text="ABSOLUTE POSITIONING", bg=BG, fg=ORANGE,
                 font=FONT_MONO_S).pack(anchor="w", padx=18, pady=(0, 8))

        pre = tk.Frame(right, bg=BG)
        pre.pack(fill="x", padx=18, pady=(0, 10))
        btn(pre, "Return to Park", lambda: self._preset((self._lo + self._hi) / 2),
            parent_bg=BG).pack(side="left")
        btn(pre, "Bottom", lambda: self._preset(self._lo), parent_bg=BG).pack(side="left", padx=8)
        btn(pre, "Top", lambda: self._preset(self._hi), parent_bg=BG).pack(side="left")

        card = tk.Frame(right, bg=SURFACE)
        card.pack(fill="x", padx=18, pady=(0, 12))
        head = tk.Frame(card, bg=SURFACE)
        head.pack(fill="x", padx=16, pady=(14, 2))
        tk.Label(head, text="Target Height", bg=SURFACE, fg=LABEL,
                 font=FONT_SMALL).pack(side="left")
        self._tgt_lbl = tk.Label(head, text="", bg=SURFACE, fg=ORANGE, font=FONT_SUB)
        self._tgt_lbl.pack(side="right")
        self._slider = FloatSlider(card, self._lo, self._hi, command=self._on_slide,
                                   width=460, height=30, bg=SURFACE)
        self._slider.pack(fill="x", padx=16, pady=(2, 16))
        self._on_slide(self._slider.get())

        bar = tk.Frame(right, bg=BG)
        bar.pack(fill="x", padx=18)
        accent_btn(bar, "Go to Target", self._go, parent_bg=BG).pack(side="left")
        btn(bar, "Sync to Live", self._sync, parent_bg=BG).pack(side="left", padx=8)

        # ── calibration card ─────────────────────────────────────────────────
        tk.Label(right, text="CALIBRATION", bg=BG, fg=ORANGE,
                 font=FONT_MONO_S).pack(anchor="w", padx=18, pady=(18, 6))
        cal = tk.Frame(right, bg=SURFACE)
        cal.pack(fill="x", padx=18, pady=(0, 12))
        tk.Label(cal, text="Stall-seek the floor then the ceiling to measure the travel "
                 "envelope. The Pico persists it across power cycles.",
                 bg=SURFACE, fg=LABEL2, font=FONT_SMALL, justify="left",
                 wraplength=440).pack(anchor="w", padx=16, pady=(14, 8))
        calbar = tk.Frame(cal, bg=SURFACE)
        calbar.pack(fill="x", padx=16, pady=(0, 14))
        accent_btn(calbar, "Calibrate", self._calibrate, parent_bg=SURFACE).pack(side="left")
        self._cal_state = tk.Label(calbar, text="calibration unknown", bg=SURFACE,
                                   fg=LABEL3, font=FONT_MONO_S)
        self._cal_state.pack(side="left", padx=14)
        self._cal_env = tk.Label(calbar, text="", bg=SURFACE, fg=LABEL2, font=FONT_MONO_S)
        self._cal_env.pack(side="right")

    def _on_slide(self, v):
        self._tgt_lbl.config(text=f"{v:.3f} m")
        self._rows["target"].config(text=f"{v:.3f} m")

    def _go(self):
        self.link.lift_command(self._slider.get())

    def _preset(self, v):
        self._slider.set(v)
        self._on_slide(v)
        self.link.lift_command(v)

    def _sync(self):
        pos = self.link.snapshot()["lift"].get("pos")
        if pos is not None:
            self._slider.set(pos)
            self._on_slide(pos)

    def _calibrate(self):
        # A stall-seek DRIVES the column into both hard stops — confirm, since it is a
        # real motion (mirrors the firmware Lift Studio's Calibrate).
        if messagebox.askyesno(
                "Calibrate lift",
                "This drives the lift to the FLOOR then the CEILING to measure its "
                "travel.\n\nMake sure the column is clear. Continue?"):
            self.link.lift_calibrate()

    def refresh(self, snapshot):
        lift = snapshot["lift"]
        pos, vel = lift.get("pos"), lift.get("vel")
        online = self.link.subsystem_online("", contract.LIFT["command"])
        self._gauge.set(pos, self._lo, self._hi)
        self._big.config(text=f"{pos:.3f}" if pos is not None else "—",
                         fg=LABEL if pos is not None else LABEL3)
        self._rows["height"].config(text=f"{pos:.3f} m" if pos is not None else "—")
        self._rows["velocity"].config(
            text=f"{vel:+.3f} m/s" if vel is not None else "—")
        self._rows["status"].config(
            text="streaming" if pos is not None else ("online" if online else "offline"),
            fg=GREEN if pos is not None else LABEL2)

        # calibration state + measured envelope from the latched /lift/limits
        cal = lift.get("calibrated")
        lo, hi = lift.get("min"), lift.get("max")
        if cal is None:
            self._cal_state.config(text="calibration unknown", fg=LABEL3)
        elif cal:
            self._cal_state.config(text="calibrated", fg=GREEN)
        else:
            self._cal_state.config(text="NOT calibrated", fg=ORANGE)
        if lo is not None and hi is not None:
            self._cal_env.config(text=f"{lo:.3f}–{hi:.3f} m")
            self._rows["range"].config(text=f"{lo:.2f}–{hi:.2f} m")
