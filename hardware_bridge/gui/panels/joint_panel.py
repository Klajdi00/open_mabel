"""joint_panel.py — generic two-column joint control, shared by Arms and Body.

LEFT: a live telemetry sidebar (TelemetryColumn) for the module — position bar, vel,
torque, like the Overview but scoped. RIGHT: one compact slider per joint plus Send /
Continuous / Sync / Zero. Targets stream to the bridge only on Send, or live while
Continuous is armed (so the panel never fights another commander when left idle).
"""

import tkinter as tk

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import contract
from style import (BG, ELEVATED, GREEN, LABEL, LABEL2, PANEL_BG, RED, SEP,
                   FONT_SMALL, FONT_SUB, accent_btn, btn)
from widgets import CommandRow, TelemetryColumn, scroll_column


class JointControlPanel(tk.Frame):
    def __init__(self, parent, link, *, title, subtitle, joints, state_key, send):
        super().__init__(parent, bg=BG)
        self.link = link
        self._send = send
        self._state_key = state_key
        self._joints = joints
        self._rows = {}
        self._armed = False

        # ── left: live telemetry ────────────────────────────────────────────
        self.tele = TelemetryColumn(self, link, width=330)
        self.tele.pack(side="left", fill="y")
        self.tele.show(title, joints, state_key)
        tk.Frame(self, bg=SEP, width=1).pack(side="left", fill="y")

        # ── right: command ──────────────────────────────────────────────────
        right = tk.Frame(self, bg=BG)
        right.pack(side="left", fill="both", expand=True)
        head = tk.Frame(right, bg=BG)
        head.pack(fill="x", padx=16, pady=(10, 2))
        tk.Label(head, text=title, bg=BG, fg=LABEL, font=FONT_SUB).pack(side="left")
        tk.Label(right, text=subtitle, bg=BG, fg=LABEL2, font=FONT_SMALL,
                 anchor="w").pack(fill="x", padx=16)

        bar = tk.Frame(right, bg=BG)
        bar.pack(fill="x", padx=16, pady=(8, 4))
        self._torque_btn = accent_btn(bar, "Enable Torque", self._toggle_torque,
                                      color=GREEN, parent_bg=BG)
        self._torque_btn.pack(side="left", padx=(0, 8))
        accent_btn(bar, "Send Targets", self._send_once, parent_bg=BG).pack(side="left")
        self._arm_btn = btn(bar, "Continuous: OFF", self._toggle_arm, parent_bg=BG)
        self._arm_btn.pack(side="left", padx=8)
        btn(bar, "Sync", self._sync_to_live, parent_bg=BG).pack(side="left")
        btn(bar, "Zero", self._zero, parent_bg=BG).pack(side="left", padx=8)

        holder = tk.Frame(right, bg=BG)
        holder.pack(fill="both", expand=True, padx=10, pady=(0, 6))
        outer, inner = scroll_column(holder, bg=PANEL_BG)
        outer.pack(fill="both", expand=True)
        for (name, lo, hi) in joints:
            r = CommandRow(inner, name, lo, hi, self._on_change, name_w=12)
            r.pack(fill="x", padx=10, pady=1)
            self._rows[name] = r

    def _targets(self):
        return {n: r.target() for n, r in self._rows.items()}

    def _on_change(self, name, value):
        if self._armed:                         # only stream live when explicitly armed
            self._send({name: value})

    def _send_once(self):
        self._send(self._targets())

    def _toggle_arm(self):
        self._armed = not self._armed
        self._arm_btn.update_text(f"Continuous: {'ON' if self._armed else 'OFF'}")
        self._arm_btn.update_colors(GREEN if self._armed else ELEVATED, LABEL)

    def _sync_to_live(self):
        joints = self.link.snapshot().get(self._state_key, {}).get("joints", {})
        for name, row in self._rows.items():
            if name in joints:
                row.set_target(joints[name][0])

    def _zero(self):
        for row in self._rows.values():
            row.set_target(0.0)
        self._send(self._targets())

    def _toggle_torque(self):
        on = bool(self.link.snapshot().get(self._state_key, {}).get("torque"))
        self.link.joint_torque(self._state_key, not on)

    def refresh(self, snapshot):
        self.tele.refresh(snapshot)
        tq = snapshot.get(self._state_key, {}).get("torque")
        if tq:
            self._torque_btn.update_text("Disable (free-drive)")
            self._torque_btn.update_colors(RED, LABEL)
        else:
            self._torque_btn.update_text("Enable Torque")
            self._torque_btn.update_colors(GREEN, LABEL)
