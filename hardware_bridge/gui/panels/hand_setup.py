"""hand_setup.py — Hands -> Setup: the joint <-> motor map + calibrated limits.

A bridge-routed (read-only) take on the firmware GUI's Setup -> Joint Limits tab: for the
selected side it lists every joint with its motor ID, the calibrated raw-tick range
[min..max] + center, and the radian range the bridge maps onto. The values come from the
vendored config + ``<side>_calibration.yaml`` (the same files the bridge HAL loads), so
this never opens the bus — editing limits / re-centering lives in the Calibration tab,
which takes the bus directly.
"""

import tkinter as tk

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import contract
from style import (BG, GREEN, LABEL, LABEL2, LABEL3, PANEL_BG, SURFACE, SEP,
                   FONT_MONO_S, FONT_SMALL, FONT_SUB, AppleSegmentedControl)
from widgets import scroll_column

_HDR = [("joint", 13), ("id", 4), ("tick min", 9), ("tick max", 9),
        ("center", 8), ("range", 7), ("angle lo", 9), ("angle hi", 9)]


class HandSetupView(tk.Frame):
    def __init__(self, parent, link=None):
        super().__init__(parent, bg=BG)
        self.side = "left"

        head = tk.Frame(self, bg=BG)
        head.pack(fill="x", padx=20, pady=(12, 6))
        tk.Label(head, text="Setup — joint map & limits", bg=BG, fg=LABEL,
                 font=FONT_SUB).pack(side="left")
        self._seg = AppleSegmentedControl(head, contract.HANDS["sides"], "left",
                                          self._switch_side, width=150, height=32,
                                          parent_bg=BG)
        self._seg.pack(side="left", padx=16)
        self._info = tk.Label(head, text="", bg=BG, fg=LABEL2, font=FONT_MONO_S)
        self._info.pack(side="right")

        tk.Label(self, text="Read-only — from the vendored config + <side>_calibration.yaml "
                            "the bridge loads. Change limits in the Calibration tab.",
                 bg=BG, fg=LABEL3, font=FONT_MONO_S, anchor="w").pack(fill="x", padx=20,
                                                                      pady=(0, 6))

        self._body = tk.Frame(self, bg=BG)
        self._body.pack(fill="both", expand=True, padx=12, pady=(0, 10))
        self._build_table()

    def _switch_side(self, side):
        self.side = side
        self._build_table()

    def _load(self):
        """[(joint, id, tmin, tmax, center, range, alo, ahi)] for the side, or []."""
        try:
            sys.path.insert(0, os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__)))))
            from mabel_hw.drivers.orcahand import HandConfig
            from mabel_hw.drivers.orcahand.calibration import read_mapping
            cfg = HandConfig.load(model=self.side)
        except Exception:
            return []
        calib = {}
        cp = getattr(cfg, "calibration_path", None)
        if cp and os.path.exists(cp):
            try:
                calib = {int(k): v for k, v in read_mapping(cp).items()}
            except Exception:
                calib = {}
        rows = []
        for j in cfg.joints:
            c = calib.get(j.motor_id, {})
            lo, hi = sorted((j.angle_min, j.angle_max))
            rows.append((j.name, j.motor_id, c.get("min", "-"), c.get("max", "-"),
                         c.get("center", "-"), c.get("range", "-"), lo, hi))
        return rows

    def _build_table(self):
        for w in self._body.winfo_children():
            w.destroy()
        rows = self._load()
        self._info.config(text=f"{len(rows)} joints" if rows else "no config")

        card = tk.Frame(self._body, bg=SURFACE)
        card.pack(fill="both", expand=True, padx=8)
        hdr = tk.Frame(card, bg=SURFACE)
        hdr.pack(fill="x", padx=10, pady=(8, 4))
        for name, w in _HDR:
            tk.Label(hdr, text=name, bg=SURFACE, fg=LABEL2, font=FONT_SMALL,
                     width=w, anchor="w").pack(side="left")
        tk.Frame(card, bg=SEP, height=1).pack(fill="x", padx=10)

        outer, inner = scroll_column(card, bg=PANEL_BG)
        outer.pack(fill="both", expand=True, padx=4, pady=4)
        for r in rows:
            row = tk.Frame(inner, bg=PANEL_BG)
            row.pack(fill="x", padx=8, pady=1)
            cells = [r[0], r[1],
                     r[2], r[3], r[4], r[5],
                     f"{r[6]:+.2f}", f"{r[7]:+.2f}"]
            calibrated = r[2] != "-"
            for (name, w), val in zip(_HDR, cells):
                fg = LABEL if name == "joint" else (LABEL2 if calibrated else LABEL3)
                tk.Label(row, text=str(val), bg=PANEL_BG, fg=fg, font=FONT_MONO_S,
                         width=w, anchor="w").pack(side="left")

    def refresh(self, *_):
        pass
