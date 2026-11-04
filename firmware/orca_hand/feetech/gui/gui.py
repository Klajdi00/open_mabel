#!/usr/bin/env python3
"""MABEL ORCA Hand — Comprehensive Control GUI.

Usage:
    python GUI/gui.py
    python GUI/gui.py --hand right
    python GUI/gui.py --port /dev/ttyUSB0 --hand left
"""

import argparse
import sys
import threading
import time
import tkinter as tk
from pathlib import Path

_HERE      = Path(__file__).resolve().parent
_FEETECH   = _HERE.parent
_ORCA_ROOT = _FEETECH.parent / "orca_core"
for _p in (str(_HERE), str(_FEETECH), str(_ORCA_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from style import *
from hand_bus import HandBus, TOTAL

import serial.tools.list_ports

_PANEL_MODULES = {
    "Control":     ("panels.control",     "ControlPanel"),
    "Setup":       ("panels.setup",       "SetupPanel"),
    "Calibration": ("panels.calibration", "CalibrationPanel"),
    "Poses":       ("panels.poses",       "PosesPanel"),
    "Teleop":      ("panels.teleop",      "TeleopPanel"),
}

WIN_W      = 1350
WIN_H      = 820
SIDEBAR_W  = 270


# ── Telemetry Sidebar ─────────────────────────────────────────────────────────

class TelemetrySidebar(tk.Frame):
    """Left panel: a live position bar per joint (server-monitor style), plus a
    compact temperature read-out and the event log."""

    BAR_W = SIDEBAR_W - 78    # leaves room for the percent read-out

    def __init__(self, parent, bus: HandBus):
        super().__init__(parent, bg=PANEL_BG, width=SIDEBAR_W)
        self.bus = bus
        self.pack_propagate(False)
        self._rows: dict[int, dict] = {}
        self._build()
        bus.subscribe(self._on_state)

    def _build(self):
        head = tk.Frame(self, bg=PANEL_BG)
        head.pack(fill="x", padx=14, pady=(12, 6))
        tk.Label(head, text="LIVE TELEMETRY", bg=PANEL_BG,
                 fg=LABEL2, font=FONT_SMALL).pack(side="left")
        self._hz = tk.Label(head, text="", bg=PANEL_BG, fg=LABEL3, font=FONT_MONO_S)
        self._hz.pack(side="right")
        self._head = head

        # Out-of-range warning banner — hidden until a joint drifts outside its
        # calibrated window. Tells the user exactly where to fix it.
        self._warn = tk.Label(self, bg="#3a2a00", fg=YELLOW, font=FONT_MONO_S,
                              justify="left", anchor="w", padx=8, pady=6,
                              wraplength=SIDEBAR_W - 28)
        self._warn_shown = False

        # Scrollable joint list
        canvas  = tk.Canvas(self, bg=PANEL_BG, highlightthickness=0)
        scroll  = tk.Scrollbar(self, orient="vertical", command=canvas.yview,
                               bg=PANEL_BG, troughcolor=PANEL_BG)
        scroll.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        inner = tk.Frame(canvas, bg=PANEL_BG)
        canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")))
        canvas.configure(yscrollcommand=scroll.set)

        # Two compact lines per joint so all 17 fit without resizing:
        #   line 1:  name ............... ω velocity  τ torque  temp
        #   line 2:  [position bar]  pct%
        for mid in range(1, TOTAL + 1):
            row = tk.Frame(inner, bg=PANEL_BG)
            row.pack(fill="x", padx=14, pady=2)

            top = tk.Frame(row, bg=PANEL_BG)
            top.pack(fill="x")
            name = tk.Label(top, text=f"ID {mid}", bg=PANEL_BG, fg=LABEL,
                            font=FONT_SMALL, anchor="w")
            name.pack(side="left")
            temp = tk.Label(top, text="—", bg=PANEL_BG, fg=LABEL3,
                            font=FONT_MONO_S, anchor="e")
            temp.pack(side="right")
            trq = tk.Label(top, text="τ—", bg=PANEL_BG, fg=LABEL3,
                           font=FONT_MONO_S, anchor="e")
            trq.pack(side="right", padx=(0, 8))
            vel = tk.Label(top, text="ω—", bg=PANEL_BG, fg=LABEL3,
                           font=FONT_MONO_S, anchor="e")
            vel.pack(side="right", padx=(0, 8))

            line2 = tk.Frame(row, bg=PANEL_BG)
            line2.pack(fill="x", pady=(2, 0))
            pct = tk.Label(line2, text="—", bg=PANEL_BG, fg=LABEL2,
                           font=FONT_MONO_S, anchor="e", width=4)
            pct.pack(side="right", padx=(6, 0))
            bar = Bar(line2, width=self.BAR_W, height=7, bg=PANEL_BG)
            bar.pack(side="left")

            self._rows[mid] = {"name": name, "pct": pct, "bar": bar,
                               "vel": vel, "trq": trq, "temp": temp}

        # Separator + event log (compact)
        tk.Frame(self, bg=SEP, height=1).pack(fill="x")
        tk.Label(self, text="EVENT LOG", bg=PANEL_BG,
                 fg=LABEL2, font=FONT_SMALL).pack(pady=(6, 2))
        self._log = tk.Text(self, bg=PANEL_BG, fg=LABEL2, font=FONT_MONO_S,
                             relief="flat", bd=0, height=4,
                             insertbackground=LABEL, state="disabled")
        self._log.pack(fill="x", padx=10, pady=(0, 8))

    def _set_warning(self, text: str):
        if text:
            self._warn.config(text=text)
            if not self._warn_shown:
                self._warn.pack(fill="x", padx=10, pady=(0, 6), after=self._head)
                self._warn_shown = True
        elif self._warn_shown:
            self._warn.pack_forget()
            self._warn_shown = False

    def _on_state(self, state, log):
        n_live = 0
        out_of_range: list[str] = []
        boundary: list[str] = []
        for mid, row in self._rows.items():
            ms = state.motors.get(mid)
            name = self.bus.id_joint.get(mid, f"ID {mid}")
            at_bound = self.bus.at_encoder_boundary(mid)
            if at_bound:
                boundary.append(name)
            if ms and ms.responding and ms.raw is not None:
                n_live += 1
                pct = max(0.0, min(100.0, ms.pct))
                oor = self.bus.out_of_range(mid, ms.raw)
                if oor:
                    out_of_range.append(name)
                flag = oor or at_bound
                row["name"].config(text=("⚠ " + name) if flag else name,
                                   fg=RED if oor else (YELLOW if at_bound else LABEL))
                # bar colour reflects health: out-of-range > hot/high-load > nominal
                if oor:
                    color = RED
                elif ms.temp > 60 or ms.load > 650:
                    color = RED
                elif ms.load > 350:
                    color = ORANGE
                else:
                    color = GREEN
                row["bar"].set(pct / 100.0, color)
                row["pct"].config(text=f"{pct:3.0f}%", fg=RED if oor else LABEL)
                row["vel"].config(
                    text=f"ω {ms.vel:+.2f}",
                    fg=GREEN if abs(ms.vel) > 0.02 else LABEL3)
                row["trq"].config(
                    text=f"τ {int(ms.load):>4d}",
                    fg=RED if ms.load > 650 else ORANGE if ms.load > 350 else LABEL2)
                row["temp"].config(
                    text=f"{int(ms.temp)}°C",
                    fg=RED if ms.temp > 60 else ORANGE if ms.temp > 50 else LABEL2)
            else:
                row["name"].config(text=name, fg=LABEL3)
                row["bar"].set(0.0, LABEL3)
                row["pct"].config(text="—", fg=LABEL3)
                row["vel"].config(text="ω —", fg=LABEL3)
                row["trq"].config(text="τ —", fg=LABEL3)
                row["temp"].config(text="—", fg=LABEL3)
        self._hz.config(text=f"{n_live}/{TOTAL}")

        parts = []
        if boundary:
            parts.append(
                "⚠ At encoder limit (0/4095): " + ", ".join(boundary) +
                " — these can't be read past the wrap. Re-center in "
                "Setup → Joint Limits (↺), then recalibrate.")
        if out_of_range:
            parts.append(
                "⚠ Outside calibrated range: " + ", ".join(out_of_range) +
                " — re-center in Setup → Joint Limits (↺).")
        self._set_warning("\n".join(parts))

        if log:
            self._log.config(state="normal")
            self._log.delete("1.0", "end")
            for entry in log[-25:]:
                self._log.insert("end", entry + "\n")
            self._log.see("end")
            self._log.config(state="disabled")


# ── Main App ──────────────────────────────────────────────────────────────────

class App(tk.Tk):

    def __init__(self, port: str | None, hand: str):
        super().__init__()
        self.bus           = HandBus()
        self._panels: dict = {}
        self._active_nav   = "Control"
        self._nav_btns: dict[str, AppleButton] = {}

        self.title("ORCA Hand Studio")
        self.configure(bg=BG)
        self.geometry(f"{WIN_W}x{WIN_H}")
        self.minsize(1100, 700)

        self._build_toolbar(hand)
        self._build_body()
        self._last_estop_key = None
        self.bus.subscribe(self._on_bus_state)   # keep E-Stop button in sync

        if port:
            self.after(200, lambda: self._connect(port, hand))
        else:
            self.after(200, self._prompt_connect)

    # ── Toolbar ───────────────────────────────────────────────────────────────

    def _build_toolbar(self, hand: str):
        bar = tk.Frame(self, bg=SURFACE, height=85)
        bar.pack(fill="x", side="top")
        bar.pack_propagate(False)

        # Title block
        title_frame = tk.Frame(bar, bg=SURFACE)
        title_frame.pack(side="left", padx=25, pady=14)
        tk.Label(title_frame, text="ORCA Hand", bg=SURFACE,
                 fg=LABEL, font=FONT_TITLE).pack(anchor="w")
        tk.Label(title_frame, text="Control Studio", bg=SURFACE,
                 fg=ORANGE, font=FONT_SMALL).pack(anchor="w")

        # Navigation tabs (AppleButton each)
        nav_f = tk.Frame(bar, bg=SURFACE)
        nav_f.pack(side="left", padx=20, pady=22)
        for name in _PANEL_MODULES:
            b = AppleButton(nav_f, name,
                             lambda n=name: self._switch_panel(n),
                             bg_color=ELEVATED, fg_color=LABEL2,
                             font=FONT_BODY, parent_bg=SURFACE,
                             width=110, height=40)
            b.pack(side="left", padx=5)
            self._nav_btns[name] = b
        self._mark_nav("Control")

        # Right: hand toggle + connection status + connect/disconnect + torque
        right = tk.Frame(bar, bg=SURFACE)
        right.pack(side="right", padx=25, pady=22)

        # Hand selector sits furthest right.
        self._hand_seg = AppleSegmentedControl(
            right, ["left", "right"],
            hand, self._on_hand_toggle,
            width=150, height=38, parent_bg=SURFACE)
        self._hand_seg.pack(side="right", padx=(14, 0))

        # E-Stop / torque toggle: red E-STOP when off, green when armed.
        self._estop_btn = AppleButton(
            right, "E-STOP", self._toggle_torque,
            bg_color=ELEVATED, fg_color=LABEL2, font=FONT_BODY,
            parent_bg=SURFACE, width=140, height=40)
        self._estop_btn.pack(side="right", padx=6)

        # Connect / Disconnect toggle.
        self._conn_btn = AppleButton(
            right, "Connect", self._on_conn_btn,
            bg_color=ORANGE, fg_color=LABEL, font=FONT_BODY,
            parent_bg=SURFACE, width=130, height=40)
        self._conn_btn.pack(side="right", padx=6)

        # Minimal connection-status indicator: a coloured dot + word.
        status = tk.Frame(right, bg=SURFACE)
        status.pack(side="right", padx=(0, 12))
        self._status_dot = tk.Label(status, text="●", fg=RED,
                                     bg=SURFACE, font=("Helvetica", 14))
        self._status_dot.pack(side="left", padx=(0, 6))
        self._conn_label = tk.Label(status, text="Disconnected", fg=LABEL2,
                                    bg=SURFACE, font=FONT_SMALL)
        self._conn_label.pack(side="left")

        tk.Frame(self, bg=SEP, height=1).pack(fill="x")

    def _mark_nav(self, name: str):
        for n, b in self._nav_btns.items():
            if n == name:
                b.update_colors(ORANGE, LABEL)
            else:
                b.update_colors(ELEVATED, LABEL2)

    # ── Body ──────────────────────────────────────────────────────────────────

    def _build_body(self):
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True)

        self._sidebar = TelemetrySidebar(body, self.bus)
        self._sidebar.pack(side="left", fill="y")

        tk.Frame(body, bg=SEP, width=1).pack(side="left", fill="y")

        self._content = tk.Frame(body, bg=BG)
        self._content.pack(side="left", fill="both", expand=True)

        self._switch_panel("Control")

    # ── Panel switching ───────────────────────────────────────────────────────

    def _switch_panel(self, name: str):
        self._active_nav = name
        self._mark_nav(name)
        for p in self._panels.values():
            p.pack_forget()
        if name not in self._panels:
            try:
                mod_name, cls_name = _PANEL_MODULES[name]
                mod   = __import__(mod_name, fromlist=[cls_name])
                panel = getattr(mod, cls_name)(self._content, self.bus)
                self._panels[name] = panel
            except Exception as exc:
                # Show a fallback error label instead of leaving blank
                err = tk.Label(self._content,
                               text=f"Panel '{name}' failed to load:\n{exc}",
                               bg=BG, fg=RED, font=FONT_BODY, justify="left")
                err.pack(padx=40, pady=60, anchor="w")
                self.bus.log(f"Panel error [{name}]: {exc}")
                return
        self._panels[name].pack(fill="both", expand=True)

    # ── Connection ────────────────────────────────────────────────────────────

    def _prompt_connect(self):
        dlg = tk.Toplevel(self)
        dlg.title("")
        dlg.configure(bg=BG)
        dlg.geometry("480x380")
        dlg.resizable(False, False)
        dlg.grab_set()

        # ── Header ────────────────────────────────────────────────────────────
        hdr = tk.Frame(dlg, bg=SURFACE, height=72)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="Connect to ORCA Hand", bg=SURFACE,
                 fg=LABEL, font=FONT_TITLE).pack(side="left", padx=24, pady=20)
        tk.Frame(dlg, bg=SEP, height=1).pack(fill="x")

        body = tk.Frame(dlg, bg=BG)
        body.pack(fill="both", expand=True, padx=28, pady=20)

        # ── Port row ──────────────────────────────────────────────────────────
        def _scan_ports():
            all_ports = serial.tools.list_ports.comports()
            detected  = [p.device for p in all_ports
                         if p.vid in (0x1A86, 0x10C4)
                         or "usb" in (p.description or "").lower()
                         or "usbmodem" in p.device.lower()]
            if not detected:
                detected = [p.device for p in all_ports]
            return detected or ["/dev/ttyUSB0"]

        ports = _scan_ports()

        tk.Label(body, text="Serial Port", bg=BG,
                 fg=LABEL2, font=FONT_SMALL).pack(anchor="w", pady=(0, 6))

        port_row = tk.Frame(body, bg=BG)
        port_row.pack(fill="x")

        port_dd = AppleDropDown(
            port_row, ports, default=ports[0],
            width=340, height=46, parent_bg=BG, font=FONT_BODY,
        )
        port_dd.pack(side="left")

        def _refresh():
            new_ports = _scan_ports()
            port_dd.update_options(new_ports, new_ports[0] if new_ports else None)

        AppleButton(port_row, "↺", _refresh,
                    bg_color=ELEVATED, fg_color=LABEL2,
                    font=("Helvetica Neue", 16, "bold"),
                    parent_bg=BG, width=46, height=46).pack(side="left", padx=8)

        # ── Hand row ──────────────────────────────────────────────────────────
        tk.Label(body, text="Hand", bg=BG,
                 fg=LABEL2, font=FONT_SMALL).pack(anchor="w", pady=(20, 6))

        hand_seg = AppleSegmentedControl(
            body, ["left", "right"],
            self._hand_seg.get(), lambda v: None,
            width=200, height=42, parent_bg=BG,
        )
        hand_seg.pack(anchor="w")

        # ── Status ────────────────────────────────────────────────────────────
        status_lbl = tk.Label(body, text="", bg=BG, fg=LABEL2, font=FONT_BODY)
        status_lbl.pack(anchor="w", pady=(16, 0))

        # ── Connect button ────────────────────────────────────────────────────
        tk.Frame(dlg, bg=SEP, height=1).pack(fill="x")
        btn_bar = tk.Frame(dlg, bg=SURFACE, height=72)
        btn_bar.pack(fill="x", side="bottom")
        btn_bar.pack_propagate(False)

        def _do():
            port = port_dd.get()
            hand = hand_seg.get()
            status_lbl.config(text="Connecting…", fg=ORANGE)
            dlg.update()
            # If already connected (e.g. switching to another ORCA hand), close
            # the current port first so the new one can open cleanly.
            if self.bus.state.connected:
                self.bus.disconnect()
                time.sleep(0.3)
            ok, msg = self.bus.connect(port, hand)
            if ok:
                # sync the main toolbar segmented control
                self._hand_seg.set(hand)
                status_lbl.config(text="Connected!", fg=GREEN)
                self._conn_ok()
                self.after(500, dlg.destroy)
            else:
                status_lbl.config(text=f"Failed: {msg}", fg=RED)

        AppleButton(btn_bar, "Connect", _do,
                    bg_color=ORANGE, fg_color=LABEL,
                    font=FONT_BODY, parent_bg=SURFACE,
                    width=160, height=44).pack(side="right", padx=20, pady=14)
        AppleButton(btn_bar, "Cancel", dlg.destroy,
                    bg_color=ELEVATED, fg_color=LABEL2,
                    font=FONT_BODY, parent_bg=SURFACE,
                    width=100, height=44).pack(side="right", pady=14)

    def _connect(self, port: str, hand: str):
        ok, msg = self.bus.connect(port, hand)
        if ok:
            self._conn_ok()
        else:
            self.bus.log(f"Auto-connect failed: {msg}")

    def _on_conn_btn(self):
        """The Connect/Disconnect toggle in the toolbar."""
        if self.bus.state.connected:
            self._disconnect()
        else:
            self._prompt_connect()

    def _disconnect(self):
        self.bus.disconnect()
        self._set_conn_ui(False)
        self._refresh_panels()

    def _set_conn_ui(self, connected: bool):
        """Single place that reflects connection state into the toolbar."""
        if connected:
            port  = self.bus.port or ""
            short = port.split("/")[-1]
            if len(short) > 18:
                short = short[:16] + "…"
            self._status_dot.config(fg=GREEN)
            self._conn_label.config(text=f"Connected · {short}", fg=LABEL)
            self._conn_btn.update_colors(ELEVATED, LABEL)
            self._conn_btn.update_text("Disconnect")
        else:
            self._status_dot.config(fg=RED)
            self._conn_label.config(text="Disconnected", fg=LABEL2)
            self._conn_btn.update_colors(ORANGE, LABEL)
            self._conn_btn.update_text("Connect")
        self._refresh_estop()

    def _refresh_estop(self):
        """E-Stop button: grey/inert when disconnected, RED 'E-STOP' when torque
        is off, GREEN 'TORQUE ON' once armed."""
        if not self.bus.state.connected:
            self._estop_btn.update_colors(ELEVATED, LABEL2)
            self._estop_btn.update_text("E-STOP")
        elif self.bus.state.torque_on:
            # Torque is live → a red E-STOP that kills it when pressed.
            self._estop_btn.update_colors(RED, LABEL)
            self._estop_btn.update_text("E-STOP")
        else:
            # Torque off → green ENABLE to arm the hand.
            self._estop_btn.update_colors(GREEN, LABEL)
            self._estop_btn.update_text("ENABLE")

    def _on_bus_state(self, state, log):
        """Keep the E-Stop button in sync when torque changes anywhere (e.g. the
        tension routine arming torque). Only schedules a refresh on a real change."""
        key = (state.connected, state.torque_on)
        if key != getattr(self, "_last_estop_key", None):
            self._last_estop_key = key
            self.after(0, self._refresh_estop)

    def _conn_ok(self):
        self._set_conn_ui(True)
        # Panels built before connection have an empty joint map — rebuild them
        # so the Control panel's sliders appear immediately (fixes blank Joints view).
        self._refresh_panels()

    def _refresh_panels(self):
        """Destroy all cached panels and rebuild the active one with fresh data."""
        active = self._active_nav
        for name in list(self._panels.keys()):
            try:
                self._panels[name].destroy()
            except Exception:
                pass
            del self._panels[name]
        self._switch_panel(active)

    # ── Torque toggle ─────────────────────────────────────────────────────────

    def _toggle_torque(self):
        if not self.bus.state.connected:
            self.bus.log("Connect a hand before enabling torque.")
            return
        self.bus.set_torque(not self.bus.state.torque_on)
        self._refresh_estop()

    def _on_hand_toggle(self, choice: str):
        """Switching hand must RELOAD that side's config + calibration so the
        right hand uses the right motor→joint map, the right calibration, and the
        right inversions — never the left's. We reconnect, which loads
        <side>_hand.yaml + <side>_calibration.json from scratch."""
        if choice == self.bus.hand:
            return
        if self.bus.state.connected:
            port = self.bus.port
            self.bus.disconnect()
            time.sleep(0.3)
            ok, msg = self.bus.connect(port, choice)
            if ok:
                self.bus.log(f"Switched to {choice} hand — loaded {choice} "
                             f"config + calibration")
                self._set_conn_ui(True)
            else:
                self.bus.log(f"Switch to {choice} failed: {msg}")
                self._set_conn_ui(False)
        else:
            # Not connected yet — just remember the selection for the next connect.
            self.bus.hand = choice
            self.bus.log(f"Hand set to {choice} (loads on connect)")
        self._refresh_panels()


def main():
    parser = argparse.ArgumentParser(description="MABEL ORCA Hand GUI")
    parser.add_argument("--port", type=str, default=None)
    parser.add_argument("--hand", choices=["left", "right"], default="left")
    parser.add_argument("--mock", action="store_true",
                        help="run with an in-memory hand (no hardware)")
    args = parser.parse_args()
    port = "mock" if args.mock else args.port
    App(port=port, hand=args.hand).mainloop()


if __name__ == "__main__":
    main()