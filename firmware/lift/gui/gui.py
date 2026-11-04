#!/usr/bin/env python3
"""MABEL Lift — iOS-style Control GUI.

Matches the exact architecture and design tokens of the ORCA Hand and Swerve GUIs.
"""

import argparse
import sys
import threading
import time
import tkinter as tk
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from style import *
from style import _FONT_BASE  # Explicitly import the hidden variable
from lift_bus import LiftBus
import serial.tools.list_ports

WIN_W      = 1200
WIN_H      = 780
SIDEBAR_W  = 300
RANGE_DEFAULT = 600.0


# ── Vertical Lift Gauge ───────────────────────────────────────────────────────

class LiftGauge(tk.Canvas):
    """A tall rounded track that fills from the bottom to represent height."""
    def __init__(self, parent, width=80, height=260):
        super().__init__(parent, width=width, height=height, bg=PANEL_BG, highlightthickness=0)
        self._cw = width
        self._ch = height
        self._pct = 0.0
        self.bind("<Configure>", lambda e: self._draw())

    def set(self, height_mm, min_mm, max_mm):
        # Fill reflects the ABSOLUTE position within the travel band:
        # empty at the bottom limit, full at the top limit.
        span = (max_mm - min_mm) if (max_mm - min_mm) > 1 else RANGE_DEFAULT
        self._pct = max(0.0, min(1.0, (height_mm - min_mm) / span))
        self._draw()

    def _rrect(self, x0, y0, x1, y1, r, color):
        kw = {"fill": color, "outline": ""}
        self.create_oval(x0, y0, x0 + 2 * r, y0 + 2 * r, **kw)
        self.create_oval(x1 - 2 * r, y0, x1, y0 + 2 * r, **kw)
        self.create_oval(x0, y1 - 2 * r, x0 + 2 * r, y1, **kw)
        self.create_oval(x1 - 2 * r, y1 - 2 * r, x1, y1, **kw)
        self.create_rectangle(x0 + r, y0, x1 - r, y1, **kw)
        self.create_rectangle(x0, y0 + r, x1, y1 - r, **kw)

    def _draw(self):
        self.delete("all")
        w = self.winfo_width() or self._cw
        h = self.winfo_height() or self._ch
        pad = 12
        tw = 24
        x0 = (w - tw) / 2
        x1 = x0 + tw
        top, bot = pad, h - pad
        r = tw / 2
        
        self._rrect(x0, top, x1, bot, r, ELEVATED)
        fill_h = (bot - top) * self._pct
        fy0 = bot - fill_h
        if fill_h > 1:
            self._rrect(x0, max(top, fy0 - r), x1, bot, r, ORANGE)
        
        self.create_line(x0 - 14, fy0, x1 + 14, fy0, fill=LABEL, width=3, capstyle=tk.ROUND)


# ── Telemetry Sidebar (Left Panel) ────────────────────────────────────────────

class TelemetrySidebar(tk.Frame):
    """Matches the ORCA Hand sidebar: Gauge, Status texts, and Event Log."""
    def __init__(self, parent, bus: LiftBus):
        super().__init__(parent, bg=PANEL_BG, width=SIDEBAR_W)
        self.bus = bus
        self.pack_propagate(False)
        self._vars = {}
        self._build()

    def _build(self):
        tk.Label(self, text="LIVE TELEMETRY", bg=PANEL_BG, fg=LABEL2, font=FONT_SMALL).pack(pady=(22, 10))

        self.v_height = tk.StringVar(value="—")
        tk.Label(self, textvariable=self.v_height, bg=PANEL_BG, fg=LABEL, font=(_FONT_BASE, 48, "bold")).pack()
        tk.Label(self, text="HEIGHT (mm)", bg=PANEL_BG, fg=LABEL3, font=F_CAPTION).pack(pady=(0, 15))

        mid = tk.Frame(self, bg=PANEL_BG)
        mid.pack(fill="x", padx=16)

        self.gauge = LiftGauge(mid, width=80, height=260)
        self.gauge.pack(side="left", padx=(0, 10))

        rows = tk.Frame(mid, bg=PANEL_BG)
        rows.pack(side="left", fill="both", expand=True, pady=8)
        
        labels_config = [
            ("status", "Status"), ("mode", "Mode"), 
            ("vel", "Velocity"), ("vset", "Setpoint"), 
            ("effort", "Effort"), ("count", "Count"),
            ("range", "Limits")
        ]
        
        for key, label in labels_config:
            r = tk.Frame(rows, bg=PANEL_BG)
            r.pack(fill="x", pady=5)
            tk.Label(r, text=label, bg=PANEL_BG, fg=LABEL2, font=FONT_SMALL, anchor="w").pack(side="left")
            var = tk.StringVar(value="—")
            self._vars[key] = var
            tk.Label(r, textvariable=var, bg=PANEL_BG, fg=LABEL, font=FONT_MONO, anchor="e").pack(side="right")

        tk.Frame(self, bg=SEP, height=1).pack(fill="x", pady=(15, 0))
        tk.Label(self, text="EVENT LOG", bg=PANEL_BG, fg=LABEL2, font=FONT_SMALL).pack(pady=(12, 4))
        
        self._log = tk.Text(self, bg=PANEL_BG, fg=LABEL2, font=FONT_MONO_S, relief="flat", bd=0, height=8, insertbackground=LABEL, state="disabled")
        self._log.pack(fill="both", expand=True, padx=12, pady=(0, 12))

    def refresh(self):
        s = self.bus.state

        # The event log ALWAYS updates — it's how you see connection and
        # calibration status even before/while no telemetry is flowing.
        self._log.config(state="normal")
        self._log.delete("1.0", "end")
        for entry in self.bus.log_tail(30):
            self._log.insert("end", entry + "\n")
        self._log.see("end")
        self._log.config(state="disabled")

        if not s.connected or not s.has_telemetry:
            self.v_height.set("—")
            for k in self._vars:
                self._vars[k].set("—")
            return

        lo = s.min_height if s.calibrated else 0.0
        hi = s.max_height if s.calibrated else RANGE_DEFAULT
        self.v_height.set(f"{s.height:.1f}")
        self.gauge.set(s.height, lo, hi)

        self._vars["status"].set(s.status_name)
        self._vars["mode"].set(s.mode_name)
        self._vars["vel"].set(f"{s.vel:+.1f}")
        self._vars["vset"].set(f"{s.vset:+.1f}")
        self._vars["effort"].set(f"{s.effort:+.2f}")
        self._vars["count"].set(f"{s.count}")
        rng = f"{s.min_height:.0f}–{s.max_height:.0f}" if s.calibrated else "Uncalibrated"
        self._vars["range"].set(rng)


# ── Card Helper ───────────────────────────────────────────────────────────────

def _build_card(parent, title):
    canvas_bg = AppleCard(parent, radius=16, bg_color=SURFACE, parent_bg=BG)
    inner = tk.Frame(canvas_bg, bg=SURFACE)
    inner.pack(fill="both", expand=True, padx=16, pady=16)
    
    tk.Label(inner, text=title.upper(), bg=SURFACE, fg=ORANGE, font=F_CAPTION, anchor="w").pack(fill="x", pady=(0, 4))
    tk.Frame(inner, bg=SEP, height=1).pack(fill="x", pady=(0, 12))
    
    return canvas_bg, inner


# ── Control Panels ────────────────────────────────────────────────────────────

class PanelPosition(tk.Frame):
    def __init__(self, parent, bus: LiftBus):
        super().__init__(parent, bg=BG)
        self.bus = bus
        self._target_mm = 0.0

        card, c = _build_card(self, "Absolute Positioning")
        card.pack(fill="x", pady=(0, 20))

        # Quick Actions
        tk.Label(c, text="Quick Actions", bg=SURFACE, fg=LABEL2, font=FONT_SMALL, anchor="w").pack(fill="x")
        row_quick = tk.Frame(c, bg=SURFACE)
        row_quick.pack(fill="x", pady=(4, 16))
        
        btn(row_quick, "Return to Park", self.bus.park, parent_bg=SURFACE).pack(side="left")
        btn(row_quick, "▼ Bottom", lambda: self._quick_move("bottom"), parent_bg=SURFACE).pack(side="left", padx=8)
        btn(row_quick, "▲ Top", lambda: self._quick_move("top"), parent_bg=SURFACE).pack(side="left")

        # Precision Slider
        tk.Label(c, text="Target Height", bg=SURFACE, fg=LABEL2, font=FONT_SMALL, anchor="w").pack(fill="x")
        row_slider = tk.Frame(c, bg=SURFACE)
        row_slider.pack(fill="x", pady=(4, 16))
        
        self._pos_val = tk.StringVar(value="0 mm")
        self._pos_slider = AppleSlider(row_slider, 0, int(RANGE_DEFAULT), command=self._on_pos_slider, width=350, height=30)
        self._pos_slider.bind("<ButtonRelease-1>", lambda e: self._do_move(), add="+")
        self._pos_slider.pack(side="left", fill="x", expand=True, padx=(0, 16))
        tk.Label(row_slider, textvariable=self._pos_val, bg=SURFACE, fg=LABEL, font=FONT_SUB, width=8, anchor="e").pack(side="left")

        # Speed Limiter
        tk.Label(c, text="Max Speed Limit", bg=SURFACE, fg=LABEL2, font=FONT_SMALL, anchor="w").pack(fill="x")
        row_spd = tk.Frame(c, bg=SURFACE)
        row_spd.pack(fill="x", pady=(4, 16))
        
        self._spd_val = tk.StringVar(value="40 mm/s")
        self._spd_slider = AppleSlider(row_spd, 5, 100, command=self._on_spd_slider, width=350, height=30)
        self._spd_slider.set(40)
        self._spd_slider.pack(side="left", fill="x", expand=True, padx=(0, 16))
        tk.Label(row_spd, textvariable=self._spd_val, bg=SURFACE, fg=LABEL, font=FONT_SUB, width=8, anchor="e").pack(side="left")

        # Execute
        row_exec = tk.Frame(c, bg=SURFACE)
        row_exec.pack(fill="x", pady=(10, 0))
        accent_btn(row_exec, "Execute Move", self._do_move, parent_bg=SURFACE, width=150).pack(side="left")

    def _quick_move(self, target):
        s = self.bus.state
        tgt = 0.0 if target == "bottom" else (s.max_height if s.calibrated else RANGE_DEFAULT)
        self._pos_slider.set(tgt)
        self._on_pos_slider(tgt)
        self._do_move()

    def _on_pos_slider(self, val):
        self._target_mm = float(val)
        self._pos_val.set(f"{int(val)} mm")

    def _on_spd_slider(self, val):
        self._spd_val.set(f"{int(val)} mm/s")

    def _do_move(self):
        vmax = float(self._spd_slider.get())
        self.bus.move_to(self._target_mm, vmax)

    def refresh(self):
        s = self.bus.state
        if s.calibrated and (abs(self._pos_slider.min_val - s.min_height) > 1 or abs(self._pos_slider.max_val - s.max_height) > 1):
            self._pos_slider.min_val = s.min_height
            self._pos_slider.max_val = max(s.min_height + 1, s.max_height)
            self._pos_slider._redraw()


class PanelVelocity(tk.Frame):
    def __init__(self, parent, bus: LiftBus):
        super().__init__(parent, bg=BG)
        self.bus = bus
        self._vel_mm = 0.0

        card, c = _build_card(self, "Velocity Control")
        card.pack(fill="x")

        # Continuous Velocity
        tk.Label(c, text="Continuous Drive", bg=SURFACE, fg=LABEL2, font=FONT_SMALL, anchor="w").pack(fill="x")
        row_slider = tk.Frame(c, bg=SURFACE)
        row_slider.pack(fill="x", pady=(4, 16))
        
        self._vel_val = tk.StringVar(value="0 mm/s")
        self._vel_slider = AppleSlider(row_slider, -100, 100, command=self._on_vel_slider, width=350, height=30)
        self._vel_slider.set(0)
        self._vel_slider.pack(side="left", fill="x", expand=True, padx=(0, 16))
        tk.Label(row_slider, textvariable=self._vel_val, bg=SURFACE, fg=LABEL, font=FONT_SUB, width=9, anchor="e").pack(side="left")

        row_exec = tk.Frame(c, bg=SURFACE)
        row_exec.pack(fill="x", pady=(0, 20))
        accent_btn(row_exec, "Hold Velocity", self._do_velocity, color=BLUE, parent_bg=SURFACE, width=150).pack(side="left")

        tk.Frame(c, bg=SEP, height=1).pack(fill="x", pady=(0, 16))

        # Manual Jogging
        tk.Label(c, text="Manual Jog", bg=SURFACE, fg=LABEL2, font=FONT_SMALL, anchor="w").pack(fill="x")
        row_jog = tk.Frame(c, bg=SURFACE)
        row_jog.pack(fill="x", pady=(8, 0))
        
        up = btn(row_jog, "▲ Jog Up", lambda: None, parent_bg=SURFACE)
        up.pack(side="left", padx=(0, 8))
        self._bind_jog(up, +0.6)
        
        dn = btn(row_jog, "▼ Jog Down", lambda: None, parent_bg=SURFACE)
        dn.pack(side="left")
        self._bind_jog(dn, -0.6)

    def _bind_jog(self, button, effort):
        button.bind("<ButtonPress-1>", lambda e, ef=effort: self.bus.jog(ef), add="+")
        button.bind("<ButtonRelease-1>", lambda e: self.bus.stop(), add="+")

    def _on_vel_slider(self, val):
        self._vel_mm = float(val)
        self._vel_val.set(f"{int(val)} mm/s")

    def _do_velocity(self):
        self.bus.set_velocity(self._vel_mm)
        
    def refresh(self): pass


class PanelCalibration(tk.Frame):
    def __init__(self, parent, bus: LiftBus):
        super().__init__(parent, bg=BG)
        self.bus = bus

        card, c = _build_card(self, "System Setup & Calibration")
        card.pack(fill="x")

        self._calib_lbl = tk.Label(c, text="Status: Uncalibrated. Run Auto-Calibrate to map physical limits.", bg=SURFACE, fg=LABEL2, font=FONT_BODY, anchor="w")
        self._calib_lbl.pack(fill="x", pady=(0, 16))

        row_cal = tk.Frame(c, bg=SURFACE)
        row_cal.pack(fill="x", pady=(0, 24))
        accent_btn(row_cal, "Run Auto-Calibration", self.bus.calibrate, parent_bg=SURFACE, width=180).pack(side="left")

        tk.Frame(c, bg=SEP, height=1).pack(fill="x", pady=(0, 16))

        # Overrides
        tk.Label(c, text="Manual Overrides", bg=SURFACE, fg=LABEL2, font=FONT_SMALL, anchor="w").pack(fill="x", pady=(0, 8))
        row_ov = tk.Frame(c, bg=SURFACE)
        row_ov.pack(fill="x")
        
        btn(row_ov, "Zero Here", self.bus.zero, parent_bg=SURFACE).pack(side="left")
        btn(row_ov, "Set Home", self.bus.set_home, parent_bg=SURFACE).pack(side="left", padx=8)

        tk.Label(row_ov, text="Soft Limits:", bg=SURFACE, fg=LABEL2, font=FONT_BODY).pack(side="left", padx=(16, 8))
        self._limit_seg = AppleSegmentedControl(row_ov, ["On", "Off"], "On", lambda ch: self.bus.enable_limits(ch == "On"), width=120, height=36, parent_bg=SURFACE)
        self._limit_seg.pack(side="left")

    def refresh(self):
        s = self.bus.state
        if s.calibrated:
            self._calib_lbl.config(text=f"Calibrated — Travel: {s.min_height:.0f} to {s.max_height:.0f} mm", fg=GREEN)


# ── Main App ──────────────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self, port: str | None):
        super().__init__()
        self.bus = LiftBus()
        self._panels = {}
        self._active_nav = "Position"
        self._nav_btns = {}

        self.title("MABEL Lift Studio")
        self.configure(bg=BG)
        self.geometry(f"{WIN_W}x{WIN_H}")
        self.minsize(1000, 650)

        self._build_toolbar()
        self._build_body()

        self.after(80, self._refresh_loop)      # START the live UI update loop
        self.after(200, lambda: self._autostart(port))

    def _autostart(self, port):
        """Connect automatically: use the given port, else auto-detect the Pico.
        Falls back to the picker dialog only if nothing is found / it fails."""
        if not port:
            found = [p.device for p in serial.tools.list_ports.comports()
                     if "ACM" in p.device or "usbmodem" in p.device.lower()]
            port = found[0] if found else None
        if not port:
            self.bus.log("No Pico (ttyACM*) found — open the Port dialog.")
            self._prompt_connect()
            return
        self.bus.log(f"Auto-connecting to {port} …")

        def cb(ok, msg):
            if not ok:
                self.bus.log(f"Auto-connect failed: {msg} — close any REPL/"
                             "console on the port, or use the Port dialog.")
        self._start_connect(port, cb)

    def _build_toolbar(self):
        bar = tk.Frame(self, bg=SURFACE, height=85)
        bar.pack(fill="x", side="top")
        bar.pack_propagate(False)

        # Title block
        title_frame = tk.Frame(bar, bg=SURFACE)
        title_frame.pack(side="left", padx=25, pady=14)
        tk.Label(title_frame, text="MABEL Lift", bg=SURFACE, fg=LABEL, font=FONT_TITLE).pack(anchor="w")
        tk.Label(title_frame, text="Control Studio", bg=SURFACE, fg=ORANGE, font=FONT_SMALL).pack(anchor="w")

        # Navigation tabs (matches ORCA structure)
        nav_f = tk.Frame(bar, bg=SURFACE)
        nav_f.pack(side="left", padx=30, pady=22)
        for name in ["Position", "Velocity", "Calibration"]:
            b = AppleButton(nav_f, name, lambda n=name: self._switch_panel(n),
                            bg_color=ELEVATED, fg_color=LABEL2, font=FONT_BODY,
                            parent_bg=SURFACE, width=120, height=40)
            b.pack(side="left", padx=5)
            self._nav_btns[name] = b
        self._mark_nav("Position")

        # Right Actions (Port & Stop)
        right = tk.Frame(bar, bg=SURFACE)
        right.pack(side="right", padx=25, pady=22)

        self._status_dot = tk.Label(right, text="●", fg=RED, bg=SURFACE, font=("Helvetica", 16))
        self._status_dot.pack(side="left", padx=(0, 8))

        self._port_btn = AppleButton(right, "⏏  Port", self._prompt_connect, bg_color=ELEVATED, fg_color=LABEL, font=FONT_BODY, parent_bg=SURFACE, width=120, height=40)
        self._port_btn.pack(side="left", padx=6)
        
        self._stop_btn = AppleButton(right, "■  E-STOP", self.bus.stop, bg_color=RED, fg_color=LABEL, font=FONT_BODY, parent_bg=SURFACE, width=120, height=40)
        self._stop_btn.pack(side="left", padx=6)

        tk.Frame(self, bg=SEP, height=1).pack(fill="x")

    def _mark_nav(self, name: str):
        for n, b in self._nav_btns.items():
            b.update_colors(ORANGE if n == name else ELEVATED, LABEL if n == name else LABEL2)

    def _build_body(self):
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True)

        self._sidebar = TelemetrySidebar(body, self.bus)
        self._sidebar.pack(side="left", fill="y")
        
        tk.Frame(body, bg=SEP, width=1).pack(side="left", fill="y")

        self._content = tk.Frame(body, bg=BG)
        self._content.pack(side="left", fill="both", expand=True, padx=24, pady=24)
        
        self._panels["Position"] = PanelPosition(self._content, self.bus)
        self._panels["Velocity"] = PanelVelocity(self._content, self.bus)
        self._panels["Calibration"] = PanelCalibration(self._content, self.bus)
        self._switch_panel("Position")

    def _switch_panel(self, name: str):
        self._active_nav = name
        self._mark_nav(name)
        for p in self._panels.values(): p.pack_forget()
        self._panels[name].pack(fill="both", expand=True)

    # ── Connection & Dialogs ──────────────────────────────────────────────────

    def _prompt_connect(self):
        dlg = tk.Toplevel(self)
        dlg.title("")
        dlg.configure(bg=BG)
        dlg.geometry("480x280")
        dlg.resizable(False, False)
        dlg.grab_set()

        hdr = tk.Frame(dlg, bg=SURFACE, height=72)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="Connect to Lift Firmware", bg=SURFACE, fg=LABEL, font=FONT_TITLE).pack(side="left", padx=24, pady=20)
        tk.Frame(dlg, bg=SEP, height=1).pack(fill="x")

        body = tk.Frame(dlg, bg=BG)
        body.pack(fill="both", expand=True, padx=28, pady=20)

        def _scan():
            ports = [p.device for p in serial.tools.list_ports.comports()]
            acm = [p for p in ports if "ACM" in p or "usbmodem" in p.lower()]
            return acm or ports or ["/dev/ttyACM0"]

        ports = _scan()
        tk.Label(body, text="Serial Port", bg=BG, fg=LABEL2, font=FONT_SMALL).pack(anchor="w", pady=(0, 6))
        
        row = tk.Frame(body, bg=BG)
        row.pack(fill="x")
        dd = AppleDropDown(row, ports, default=ports[0], width=340, height=46, parent_bg=BG, font=FONT_BODY)
        dd.pack(side="left")
        AppleButton(row, "↺", lambda: dd.update_options(_scan(), None), bg_color=ELEVATED, fg_color=LABEL2, font=("Helvetica Neue", 16, "bold"), parent_bg=BG, width=46, height=46).pack(side="left", padx=8)

        status = tk.Label(body, text="", bg=BG, fg=LABEL2, font=FONT_BODY)
        status.pack(anchor="w", pady=(16, 0))

        tk.Frame(dlg, bg=SEP, height=1).pack(fill="x")
        btnbar = tk.Frame(dlg, bg=SURFACE, height=72)
        btnbar.pack(fill="x", side="bottom")
        btnbar.pack_propagate(False)

        def _do():
            status.config(text="Connecting…", fg=ORANGE)
            dlg.update()
            def cb(ok, msg):
                if ok:
                    status.config(text="Connected!", fg=GREEN)
                    self.after(500, dlg.destroy)
                else: status.config(text=f"Failed: {msg}", fg=RED)
            
            if self.bus.state.connected:
                self.bus.disconnect()
                time.sleep(0.2)
            self._start_connect(dd.get(), cb)

        AppleButton(btnbar, "Connect", _do, bg_color=ORANGE, fg_color=LABEL, font=FONT_BODY, parent_bg=SURFACE, width=160, height=44).pack(side="right", padx=20, pady=14)
        AppleButton(btnbar, "Cancel", dlg.destroy, bg_color=ELEVATED, fg_color=LABEL2, font=FONT_BODY, parent_bg=SURFACE, width=100, height=44).pack(side="right", pady=14)

    def _connect(self, port):
        self._start_connect(port, lambda ok, msg: self.bus.log(f"Auto-connect failed: {msg}") if not ok else None)

    def _start_connect(self, port, status_cb=None):
        def worker():
            ok, msg = self.bus.connect(port)
            self.after(0, lambda: self._conn_done(ok, msg, port, status_cb))
        threading.Thread(target=worker, daemon=True).start()

    def _conn_done(self, ok, msg, port, status_cb):
        if ok:
            self._status_dot.config(fg=GREEN)
            short = port.split("/")[-1]
            self._port_btn.update_text(f"⏏  {short[:14] + '…' if len(short)>16 else short}")
        if status_cb: status_cb(ok, msg)

    # ── Loop ──────────────────────────────────────────────────────────────────

    def _refresh_loop(self):
        try:
            s = self.bus.state
            if not s.connected: self._status_dot.config(fg=RED)
            elif s.has_telemetry: self._status_dot.config(fg=GREEN)
            else: self._status_dot.config(fg=ORANGE)

            self._sidebar.refresh()
            if self._active_nav in self._panels:
                self._panels[self._active_nav].refresh()
        except Exception as exc:        # never let one bad frame kill the loop
            try:
                self.bus.log(f"UI refresh error: {exc}")
            except Exception:
                pass
        finally:
            # ALWAYS reschedule — this is what keeps telemetry + log alive.
            self.after(80, self._refresh_loop)

def main():
    ap = argparse.ArgumentParser(description="MABEL Lift GUI")
    ap.add_argument("--port", "-p", default=None, help="serial port")
    args = ap.parse_args()
    app = App(port=args.port)
    app.protocol("WM_DELETE_WINDOW", lambda: (app.bus.shutdown(), app.destroy()))
    app.mainloop()

if __name__ == "__main__":
    main()