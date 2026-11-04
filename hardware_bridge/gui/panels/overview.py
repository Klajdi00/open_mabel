"""overview.py - the whole robot's live joint telemetry, compact, in one glance.

No scrolling: every joint the bridge reports is laid out at once -
  Base   : 3 columns, one per swerve module (drive + steer = 6 motors)
  Arms   : two columns, left arm | right arm (7 each)
  Body   : torso + 3-DoF neck    ·   Lift : column height
  Hands  : left hand | right hand, side by side (17 joints each)
Each joint shows a position bar + numeric position, velocity (v), torque/effort (t),
and temperature where the hardware reports it (the hands). All read over DDS from the
bridge; this panel commands nothing.
"""

import math
import tkinter as tk

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import contract
from style import (BG, GREEN, ORANGE, LABEL, LABEL2, LABEL3, PANEL_BG, SURFACE, SEP,
                   FONT_MONO_S, FONT_SMALL, FONT_SUB, StatusDot)
from widgets import TelemetryRow, scroll_column
from sensors import CameraTile, SensorPill

_BASE_RANGE = (-math.pi, math.pi)        # base joints have no URDF limit here


class _Group(tk.Frame):
    """A titled card: header ( dot · title · summary) + one or more columns of rows."""

    def __init__(self, parent, title, topic):
        super().__init__(parent, bg=SURFACE)
        self.topic = topic
        self.rows = {}                   # joint name -> TelemetryRow

        head = tk.Frame(self, bg=SURFACE)
        head.pack(fill="x", padx=12, pady=(5, 2))
        self.dot = StatusDot(head, d=9, bg=SURFACE, color=LABEL3)
        self.dot.pack(side="left", padx=(0, 7))
        tk.Label(head, text=title, bg=SURFACE, fg=LABEL, font=FONT_SMALL).pack(side="left")
        self._head = head
        self._open = tk.Label(head, text="", bg=SURFACE, fg=ORANGE, font=FONT_SMALL)
        self._open.pack(side="right", padx=(8, 0))
        self.summary = tk.Label(head, text="-", bg=SURFACE, fg=LABEL2, font=FONT_MONO_S,
                                anchor="e")
        self.summary.pack(side="right")
        tk.Frame(self, bg=SEP, height=1).pack(fill="x", padx=12)
        self.body = tk.Frame(self, bg=PANEL_BG)
        self.body.pack(fill="both", expand=True, padx=6, pady=(3, 4))

    def add_columns(self, columns, unit="rad", show_vel=True, show_temp=True,
                    bar_w=120, name_w=13):
        for col in columns:
            cf = tk.Frame(self.body, bg=PANEL_BG)
            cf.pack(side="left", fill="both", expand=True, padx=(2, 12), anchor="n")
            for (name, lo, hi) in col:
                r = TelemetryRow(cf, name, lo, hi, unit=unit, bar_w=bar_w, name_w=name_w,
                                 show_vel=show_vel, show_temp=show_temp)
                r.pack(fill="x", pady=0)
                self.rows[name] = r

    def set_nav(self, name, callback):
        """Make the card header a clickable shortcut to that module's own tab."""
        self._open.config(text=f"open {name} ›")
        def go(_e=None):
            callback(name)
        for w in (self._head, self.dot, self.summary, self._open):
            w.bind("<Button-1>", go)
            try:
                w.config(cursor="hand2")
            except Exception:
                pass
        self._head.bind("<Button-1>", go)

    def set_online(self, online):
        self.dot.set(GREEN if online else LABEL3)

    def set_summary(self, text, fg=LABEL2):
        self.summary.config(text=text, fg=fg)


class OverviewPanel(tk.Frame):
    def __init__(self, parent, link):
        super().__init__(parent, bg=BG)
        self.link = link

        tk.Label(self, text="Robot Overview", bg=BG, fg=LABEL,
                 font=FONT_SUB).pack(anchor="w", padx=18, pady=(8, 2))

        outer, inner = scroll_column(self, bg=BG)
        outer.pack(fill="both", expand=True)

        # ── Sensors + Cameras ────────────────────────────────────────────────
        sens = tk.Frame(inner, bg=SURFACE)
        sens.pack(fill="x", padx=10, pady=(0, 8))
        shead = tk.Frame(sens, bg=SURFACE)
        shead.pack(fill="x", padx=12, pady=(7, 2))
        tk.Label(shead, text="Sensors & Cameras", bg=SURFACE, fg=LABEL,
                 font=FONT_SMALL).pack(side="left")
        self.lidar_pill = SensorPill(shead, "LiDAR", bg=SURFACE)
        self.lidar_pill.pack(side="right")
        tk.Frame(sens, bg=SEP, height=1).pack(fill="x", padx=12)
        cams = tk.Frame(sens, bg=PANEL_BG)
        cams.pack(fill="x", padx=8, pady=8)
        self._cam_tiles = []
        for label, url in contract.discover_cameras():
            t = CameraTile(cams, label, url)
            t.pack(side="left", expand=True)        # spread evenly across the width
            self._cam_tiles.append(t)

        # ── Base - 3 module columns (built lazily from telemetry names) ──────--
        self.g_base = _Group(inner, "Base - swerve modules", contract.BASE["cmd_vel"])
        self.g_base.pack(fill="x", padx=10, pady=(0, 6))
        self._base_built = False

        # Arms - left | right.
        self.g_arms = _Group(inner, "Arms - left | right", contract.ARMS["command"])
        left = [j for j in contract.ARMS["joints"] if j[0].startswith("left")]
        right = [j for j in contract.ARMS["joints"] if j[0].startswith("right")]
        self.g_arms.add_columns([left, right], show_temp=False, bar_w=150, name_w=12)
        self.g_arms.pack(fill="x", padx=10, pady=(0, 6))

        # Body + Lift, side by side.
        midrow = tk.Frame(inner, bg=BG)
        midrow.pack(fill="x", padx=10, pady=(0, 6))
        self.g_body = _Group(midrow, "Body - torso + neck", contract.BODY["command"])
        bj = contract.BODY["joints"]
        self.g_body.add_columns([bj[:2], bj[2:]], show_temp=False, bar_w=150, name_w=11)
        self.g_body.pack(side="left", fill="both", expand=True, padx=(0, 8))
        self.g_lift = _Group(midrow, "Lift", contract.LIFT["command"])
        self.g_lift.add_columns([[(contract.LIFT["joint"], contract.LIFT["min_m"],
                                   contract.LIFT["max_m"])]], unit="m",
                                show_vel=False, show_temp=False, bar_w=180, name_w=8)
        self.g_lift.pack(side="left", fill="both", expand=True)

        # Hands - left | right, side by side.
        handrow = tk.Frame(inner, bg=BG)
        handrow.pack(fill="both", expand=True, padx=10, pady=(0, 8))
        self.g_hand = {}
        for i, side in enumerate(contract.HANDS["sides"]):
            g = _Group(handrow, f"{side.title()} Hand",
                       contract.HANDS["command"].format(side=side))
            g.add_columns([contract.hand_joints(side)], bar_w=110, name_w=12)
            g.pack(side="left", fill="both", expand=True, padx=(0, 8) if i == 0 else 0)
            self.g_hand[side] = g

        # let each status card jump to that module's own tab (set by app after build)
        self.on_navigate = None
        self.g_base.set_nav("Base", self._go)
        self.g_arms.set_nav("Arms", self._go)
        self.g_body.set_nav("Body", self._go)
        self.g_lift.set_nav("Lift", self._go)
        for g in self.g_hand.values():
            g.set_nav("Hands", self._go)

    def _go(self, name):
        if callable(getattr(self, "on_navigate", None)):
            self.on_navigate(name)

    # ── live refresh ───────────────────────────────────────────────────────────
    def refresh(self, snap):
        online, hz, n = self.link.lidar_status()
        self.lidar_pill.set(online, f"{hz:.0f} Hz · {n} pts" if online else "offline")
        self._refresh_base(snap["base"]["joints"])
        self._refresh_jointed(self.g_arms, snap["arms"]["joints"])
        self._refresh_jointed(self.g_body, snap["body"]["joints"])
        for side, g in self.g_hand.items():
            self._refresh_hand(g, snap["hands"][side])
        self._refresh_lift(snap)

    def _refresh_base(self, joints):
        g = self.g_base
        g.set_online(self.link.subsystem_online("", g.topic))
        if not self._base_built and joints:
            names = sorted(joints)
            n = max(1, math.ceil(len(names) / 3))
            cols = [[(nm, *_BASE_RANGE) for nm in names[i:i + n]]
                    for i in range(0, len(names), n)]
            g.add_columns(cols, show_temp=False, bar_w=90, name_w=12)
            self._base_built = True
        self._update_rows(g, joints)

    def _refresh_jointed(self, group, joints):
        group.set_online(self.link.subsystem_online("", group.topic))
        self._update_rows(group, joints)

    def _update_rows(self, group, joints):
        n = 0
        for name, row in group.rows.items():
            v = joints.get(name)
            if v is None:
                row.update(None)
            else:
                pos, vel, eff = (list(v) + [0.0, 0.0])[:3]
                row.update(pos, vel, eff)
                n += 1
        group.set_summary(f"{n}/{len(group.rows)}" if group.rows else "no telemetry")

    def _refresh_hand(self, group, d):
        group.set_online(self.link.subsystem_online("", group.topic))
        joints = d.get("joints", {})
        temps = d.get("temps", {})
        live, hot = 0, 0.0
        for name, row in group.rows.items():
            v = joints.get(name)
            t = temps.get(name)
            if v is None:
                row.update(None)
            else:
                pos, vel, eff = (list(v) + [0.0, 0.0])[:3]
                row.update(pos, vel, eff, t)
                live += 1
                if t:
                    hot = max(hot, t)
        tq = d.get("torque")
        tq_s = "torque ON" if tq else ("free" if tq is not None else "?")
        extra = f" · {hot:.0f}°C" if hot else ""
        group.set_summary(f"{live}/{len(group.rows)} · {tq_s}{extra}",
                          fg=GREEN if tq else LABEL2)

    def _refresh_lift(self, snap):
        g = self.g_lift
        g.set_online(self.link.subsystem_online("", g.topic))
        pos = snap["lift"].get("pos")
        row = g.rows.get(contract.LIFT["joint"])
        if row is not None:
            row.update(pos if pos is not None else None)
        g.set_summary(f"{pos:.3f} m" if pos is not None else "-")
