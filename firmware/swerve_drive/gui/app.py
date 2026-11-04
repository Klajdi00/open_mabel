"""MABEL Swerve Studio — main window.

Layout mirrors the ORCA Hand Control Studio:

    ┌──────────────────────── toolbar ────────────────────────┐
    │ title │ conn dot │ panel segmented control │ STOP       │
    ├───────────────┬─────────────────────────────────────────┤
    │ LIVE          │                                         │
    │ TELEMETRY     │            active panel host            │
    │ sidebar       │                                         │
    └───────────────┴─────────────────────────────────────────┘

The sidebar streams telemetry for all 8 controllers (grouped by module).  The
host swaps panels via the segmented control; each panel is self-contained, so
new ones drop into the ``PANELS`` list with zero changes elsewhere.
"""

import math
import tkinter as tk

from . import config as C
from . import theme as T
from .bus import SwerveBus
from .motion import SwerveKinematics
from .widgets import AppleButton, AppleSegmentedControl, StatusDot
from .panels import DrivePanel, CanPanel

# Register additional panels here — they appear in the toolbar automatically.
PANELS = [DrivePanel, CanPanel]

POLL_MS = 120  # UI telemetry refresh period


class SwerveStudio:
    def __init__(self, root: tk.Tk, bus: SwerveBus):
        self.root = root
        self.bus = bus
        self._tel_vars = {}
        self._panels = {}
        self._active = None
        self._kin = SwerveKinematics()        # the motion model the feedback feeds
        self._chassis_vars = {}

        self._build_window()
        self._build_toolbar()

        body = tk.Frame(root, bg=T.BG)
        body.pack(fill="both", expand=True)
        self._build_sidebar(body)

        self._host = tk.Frame(body, bg=T.BG)
        self._host.pack(side="left", fill="both", expand=True)

        for cls in PANELS:
            self._panels[cls.title] = cls(self._host, self)
        self._show_panel(PANELS[0].title)

        self._tick()

    # ── Window chrome ──────────────────────────────────────────────────────────
    def _build_window(self):
        self.root.title("MABEL Swerve Studio")
        self.root.configure(bg=T.BG)
        self.root.minsize(1180, 760)

    def _build_toolbar(self):
        bar = tk.Frame(self.root, bg=T.SURFACE, height=85)
        bar.pack(fill="x", side="top")
        bar.pack_propagate(False)

        title = tk.Frame(bar, bg=T.SURFACE)
        title.pack(side="left", padx=25, pady=16)
        tk.Label(title, text="MABEL Swerve", bg=T.SURFACE, fg=T.LABEL,
                 font=T.FONT_TITLE).pack(anchor="w")
        tk.Label(title, text="Drive Studio", bg=T.SURFACE, fg=T.ORANGE,
                 font=T.FONT_SMALL).pack(anchor="w")

        # Connection state
        conn = tk.Frame(bar, bg=T.SURFACE)
        conn.pack(side="left", padx=(30, 0), pady=26)
        self._conn_dot = StatusDot(conn, bg=T.SURFACE)
        self._conn_dot.pack(side="left", padx=(0, 8))
        self._conn_lbl = tk.Label(conn, bg=T.SURFACE, font=T.FONT_SMALL)
        self._conn_lbl.pack(side="left")
        if self.bus.connected:
            self._conn_dot.set_color(T.GREEN)
            self._conn_lbl.config(text=f"{self.bus.iface} live", fg=T.GREEN)
        else:
            self._conn_dot.set_color(T.YELLOW)
            self._conn_lbl.config(text="SIM mode (no CAN)", fg=T.YELLOW)

        # Global emergency stop
        AppleButton(bar, "STOP", self._estop, bg_color=T.RED, parent_bg=T.SURFACE,
                    width=96, height=42, font=T.FONT_SUB).pack(side="right", padx=25, pady=21)

        # Panel switcher
        names = [cls.title for cls in PANELS]
        self._seg = AppleSegmentedControl(bar, names, names[0], self._show_panel,
                                          width=110 * len(names), height=38,
                                          parent_bg=T.SURFACE)
        self._seg.pack(side="right", padx=15, pady=23)

        tk.Frame(self.root, bg=T.SEP, height=1).pack(fill="x")

    # ── Telemetry sidebar ──────────────────────────────────────────────────────
    def _build_sidebar(self, parent):
        panel = tk.Frame(parent, bg=T.PANEL_BG, width=272)
        panel.pack(side="left", fill="y")
        panel.pack_propagate(False)

        tk.Label(panel, text="LIVE TELEMETRY", bg=T.PANEL_BG, fg=T.LABEL2,
                 font=T.FONT_SMALL).pack(pady=(22, 4))
        self._sim_banner = tk.Label(panel, text="", bg=T.PANEL_BG, fg=T.YELLOW,
                                    font=T.FONT_SMALL)
        self._sim_banner.pack()
        if not self.bus.connected:
            self._sim_banner.config(text="simulated data")

        hdr = tk.Frame(panel, bg=T.PANEL_BG)
        hdr.pack(fill="x", padx=14, pady=(10, 2))
        for txt, w, anc in (("ID", 4, "w"), ("POS", 6, "e"), ("VEL", 6, "e"),
                            ("CUR", 5, "e"), ("TMP", 5, "e")):
            tk.Label(hdr, text=txt, width=w, bg=T.PANEL_BG, fg=T.LABEL2,
                     font=T.FONT_MONO_S, anchor=anc).pack(side="left")

        roles = C.motor_roles()
        for mod in C.MODULES:
            tk.Label(panel, text=f"{mod['abbr']} · {mod['name']}", bg=T.PANEL_BG,
                     fg=T.ORANGE, font=T.FONT_SMALL, anchor="w").pack(
                        fill="x", padx=14, pady=(12, 2))
            for cid in (mod["steer_id"], mod["drive_id"]):
                self._tel_row(panel, cid, roles[cid])

        # ── Chassis velocity estimate (motion model fed by drive feedback) ────
        tk.Frame(panel, bg=T.SEP, height=1).pack(fill="x", padx=14, pady=(16, 0))
        tk.Label(panel, text="CHASSIS · EST", bg=T.PANEL_BG, fg=T.LABEL2,
                 font=T.FONT_SMALL).pack(pady=(12, 2))
        self._chassis_note = tk.Label(panel, text="from drive feedback", bg=T.PANEL_BG,
                                      fg=T.LABEL3, font=T.FONT_TINY)
        self._chassis_note.pack()
        for key, label, unit in (("vx", "vx", "m/s"), ("vy", "vy", "m/s"),
                                 ("w", "ω", "rad/s")):
            row = tk.Frame(panel, bg=T.PANEL_BG)
            row.pack(fill="x", padx=24, pady=1)
            tk.Label(row, text=label, bg=T.PANEL_BG, fg=T.LABEL2,
                     font=T.FONT_MONO_S, anchor="w").pack(side="left")
            var = tk.StringVar(value="--")
            tk.Label(row, textvariable=var, bg=T.PANEL_BG, fg=T.LABEL,
                     font=T.FONT_MONO_S, anchor="e").pack(side="right")
            tk.Label(row, text=unit, bg=T.PANEL_BG, fg=T.LABEL3,
                     font=T.FONT_TINY, anchor="e").pack(side="right", padx=(0, 6))
            self._chassis_vars[key] = var

    def _tel_row(self, parent, can_id, role):
        row = tk.Frame(parent, bg=T.PANEL_BG)
        row.pack(fill="x", padx=14, pady=1)
        tag = "S" if role == C.STEER else "D"
        tag_color = T.PURPLE if role == C.STEER else T.ORANGE
        tk.Label(row, text=f"{tag}{can_id}", width=4, bg=T.PANEL_BG, fg=tag_color,
                 font=T.FONT_MONO_S, anchor="w").pack(side="left")

        v_pos = tk.StringVar(value="----")
        v_vel = tk.StringVar(value="----")
        v_cur = tk.StringVar(value="---")
        v_tmp = tk.StringVar(value="--")
        l_cur = tk.Label(row, textvariable=v_cur, width=5, bg=T.PANEL_BG, fg=T.LABEL,
                         font=T.FONT_MONO_S, anchor="e")
        l_tmp = tk.Label(row, textvariable=v_tmp, width=5, bg=T.PANEL_BG, fg=T.LABEL,
                         font=T.FONT_MONO_S, anchor="e")
        l_pos = tk.Label(row, textvariable=v_pos, width=6, bg=T.PANEL_BG, fg=T.LABEL,
                         font=T.FONT_MONO_S, anchor="e")
        l_vel = tk.Label(row, textvariable=v_vel, width=6, bg=T.PANEL_BG, fg=T.LABEL,
                         font=T.FONT_MONO_S, anchor="e")
        l_pos.pack(side="left")
        l_vel.pack(side="left")
        l_cur.pack(side="left")
        l_tmp.pack(side="left")
        self._tel_vars[can_id] = {"pos": v_pos, "vel": v_vel, "cur": v_cur, "tmp": v_tmp,
                                  "l_pos": l_pos, "l_vel": l_vel, "l_cur": l_cur, "l_tmp": l_tmp}

    # ── Panel switching ────────────────────────────────────────────────────────
    def _show_panel(self, name):
        if self._active is not None:
            self._panels[self._active].unmount()
        self._active = name
        panel = self._panels[name]
        panel.mount()
        self._seg.set_active(name)

    # ── Loops ──────────────────────────────────────────────────────────────────
    def _tick(self):
        snap = self.bus.snapshot()
        for cid, st in snap.items():
            v = self._tel_vars.get(cid)
            if v:
                # POS: steer shows module angle in degrees — live encoder fb when
                # the frame is broadcasting (white), else the commanded angle as a
                # dimmed fallback. Drive shows primary-encoder position when live.
                if st.role == C.STEER:
                    if st.pos_fresh:
                        v["pos"].set(f"{C.encoder_to_deg(cid, st.pos):.0f}°")
                        v["l_pos"].config(fg=T.LABEL)
                    else:
                        v["pos"].set(f"{C.cmd_to_deg(st.setpoint):.0f}°")
                        v["l_pos"].config(fg=T.LABEL3)
                else:
                    fresh = st.pos_fresh and st.pos == st.pos
                    v["pos"].set(f"{st.pos:+.2f}" if fresh else "--")
                    v["l_pos"].config(fg=T.LABEL if fresh else T.LABEL3)
                # VEL: dim to "--" when the velocity frame isn't broadcasting.
                if st.vel_fresh and st.vel == st.vel:
                    v["vel"].set(f"{st.vel:+.0f}")
                    v["l_vel"].config(fg=T.LABEL)
                else:
                    v["vel"].set("--")
                    v["l_vel"].config(fg=T.LABEL3)
                if st.current != st.current:
                    v["cur"].set("---")
                else:
                    v["cur"].set(f"{st.current:.1f}")
                    v["l_cur"].config(fg=T.RED if st.current > 25 else
                                      T.ORANGE if st.current > 12 else T.LABEL)
                if st.temp != st.temp:
                    v["tmp"].set("--")
                else:
                    v["tmp"].set(f"{st.temp:.0f}°")
                    v["l_tmp"].config(fg=T.RED if st.temp > 70 else
                                      T.ORANGE if st.temp > 55 else T.LABEL)
        self._update_chassis_estimate(snap)
        active = self._panels.get(self._active)
        if active is not None:
            active.on_telemetry(snap)
        self.root.after(POLL_MS, self._tick)

    def _update_chassis_estimate(self, snap):
        """Feed live wheel feedback into the motion model → chassis velocity.

        Per module (in MODULES order): measured wheel speed (m/s, from drive
        GetVelocity) + steer angle (rad, from the encoder when its frame is live,
        else the commanded angle). Forward kinematics least-squares → vx/vy/ω."""
        speeds, angles, any_fb = [], [], False
        for mod in C.MODULES:
            d, s = snap.get(mod["drive_id"]), snap.get(mod["steer_id"])
            if d is not None and d.vel_fresh and d.vel == d.vel:
                speeds.append(C.motor_rpm_to_mps(d.vel)); any_fb = True
            else:
                speeds.append(0.0)
            if s is not None and s.pos_fresh:
                angles.append(math.radians(C.encoder_to_deg(mod["steer_id"], s.pos)))
            elif s is not None:
                angles.append(math.radians(C.cmd_to_deg(s.setpoint)))
            else:
                angles.append(0.0)
        vx, vy, w = self._kin.forward(speeds, angles)
        self._chassis_vars["vx"].set(f"{vx:+.2f}")
        self._chassis_vars["vy"].set(f"{vy:+.2f}")
        self._chassis_vars["w"].set(f"{w:+.2f}")
        # Tell the user whether this is real feedback or waiting on frame 2.
        self._chassis_note.config(
            text="from drive feedback" if any_fb else "waiting on vel frame",
            fg=T.GREEN if any_fb else T.LABEL3)

    def _estop(self):
        self.bus.stop_all()
        drive = self._panels.get("Drive")
        if isinstance(drive, DrivePanel):
            drive._estop()

    def on_close(self):
        self.bus.close()
        self.root.destroy()
