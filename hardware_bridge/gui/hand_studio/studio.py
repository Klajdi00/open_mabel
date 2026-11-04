"""studio.py — the ORCA Hand Studio as an embeddable Frame for the Control Center.

This is the firmware GUI's ``gui.py`` App (telemetry sidebar + nav tabs + connect /
hand-toggle / torque controls) reborn as a ``tk.Frame`` so it lives inside the Control
Center's Hands tab. It owns a bridge-backed :class:`HandBus`, so every panel
(Control / Setup / Calibration / Poses / Teleop) behaves exactly as in the firmware GUI —
the only difference is the hardware path goes through the hand bridge.

Connection is auto + zero-config: on first show it asks the bridge to open the connected
hand (the Feetech adapter is auto-detected). If a ROS hardware bridge is holding the
serial port (on the robot), it is released first so the hand bridge can own it, then
handed back when the studio is hidden — the same handover the standalone calibration tab
used, now covering every tab.
"""

import tkinter as tk

from style import (BG, ELEVATED, GREEN, LABEL, LABEL2, LABEL3, ORANGE, PANEL_BG, RED,
                   SEP, SURFACE, YELLOW, FONT_BODY, FONT_MONO_S, FONT_SMALL, FONT_SUB,
                   FONT_TITLE, AppleButton, AppleSegmentedControl, Bar, StatusDot,
                   accent_btn)
from hand_studio.hand_bus import HandBus, TOTAL
from hand_studio.control import ControlPanel
from hand_studio.setup import SetupPanel
from hand_studio.calibration import CalibrationPanel
from hand_studio.poses import PosesPanel
from hand_studio.teleop import TeleopPanel
from hand_studio.handmodel import HandModelView

_PANELS = {
    "Control": ControlPanel,
    "Setup": SetupPanel,
    "Calibration": CalibrationPanel,
    "Poses": PosesPanel,
    "Teleop": TeleopPanel,
}
SIDEBAR_W = 300
CONTROLS_W = 640         # right dock: calibration + joint control (kept < 3D centre)


# ── Scrollable right dock ────────────────────────────────────────────────────────
class _ScrollDock(tk.Frame):
    """Fixed-width column that scrolls (both axes) so a panel wider or taller than the
    dock is never clipped — the control panels ask for 520-800 px, the dock stays
    narrower than the 3D centre, and the scrollbars appear only when actually needed."""

    def __init__(self, parent, width):
        super().__init__(parent, bg=BG, width=width)
        self.pack_propagate(False)
        self.grid_propagate(False)
        self.canvas = tk.Canvas(self, bg=BG, highlightthickness=0)
        self._vs = tk.Scrollbar(self, orient="vertical", command=self.canvas.yview,
                                bg=PANEL_BG, troughcolor=PANEL_BG)
        self._hs = tk.Scrollbar(self, orient="horizontal", command=self.canvas.xview,
                                bg=PANEL_BG, troughcolor=PANEL_BG)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.inner = tk.Frame(self.canvas, bg=BG)
        self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self._vs.set, xscrollcommand=self._hs.set)
        self.inner.bind("<Configure>", self._on_inner)
        self.canvas.bind("<Configure>", self._on_inner)
        # wheel = vertical scroll while the pointer is over the dock
        for w in (self.canvas, self.inner):
            w.bind("<MouseWheel>", lambda e: self.canvas.yview_scroll(-1 if e.delta > 0 else 1, "units"))
            w.bind("<Button-4>", lambda e: self.canvas.yview_scroll(-1, "units"))
            w.bind("<Button-5>", lambda e: self.canvas.yview_scroll(1, "units"))

    def _on_inner(self, _e=None):
        bbox = self.canvas.bbox("all")
        if not bbox:
            return
        self.canvas.configure(scrollregion=bbox)
        # show a scrollbar only when the content actually overflows that axis
        if bbox[3] > self.canvas.winfo_height():
            self._vs.grid(row=0, column=1, sticky="ns")
        else:
            self._vs.grid_remove()
        if bbox[2] > self.canvas.winfo_width():
            self._hs.grid(row=1, column=0, sticky="ew")
        else:
            self._hs.grid_remove()


# ── Telemetry sidebar (ported verbatim from the firmware GUI) ───────────────────
class TelemetrySidebar(tk.Frame):
    """Left panel: a live position bar per joint plus velocity / torque / temp and the
    event log. Identical to the firmware GUI's sidebar (only the bus is bridge-backed)."""

    BAR_W = SIDEBAR_W - 78

    def __init__(self, parent, bus: HandBus):
        super().__init__(parent, bg=PANEL_BG, width=SIDEBAR_W)
        self.bus = bus
        self.pack_propagate(False)
        self._rows: dict = {}
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

        self._warn = tk.Label(self, bg="#3a2a00", fg=YELLOW, font=FONT_MONO_S,
                              justify="left", anchor="w", padx=8, pady=6,
                              wraplength=SIDEBAR_W - 28)
        self._warn_shown = False

        canvas = tk.Canvas(self, bg=PANEL_BG, highlightthickness=0)
        scroll = tk.Scrollbar(self, orient="vertical", command=canvas.yview,
                              bg=PANEL_BG, troughcolor=PANEL_BG)
        scroll.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        inner = tk.Frame(canvas, bg=PANEL_BG)
        canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.configure(yscrollcommand=scroll.set)

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
        out_of_range, boundary = [], []
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
                row["vel"].config(text=f"ω {ms.vel:+.2f}",
                                  fg=GREEN if abs(ms.vel) > 0.02 else LABEL3)
                row["trq"].config(text=f"τ {int(ms.load):>4d}",
                                  fg=RED if ms.load > 650 else ORANGE if ms.load > 350 else LABEL2)
                row["temp"].config(text=f"{int(ms.temp)}°C",
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
            parts.append("⚠ At encoder limit (0/4095): " + ", ".join(boundary) +
                         " — re-center in Setup → Joint Limits (↺), then recalibrate.")
        if out_of_range:
            parts.append("⚠ Outside calibrated range: " + ", ".join(out_of_range) +
                         " — re-center in Setup → Joint Limits (↺).")
        self._set_warning("\n".join(parts))

        if log:
            self._log.config(state="normal")
            self._log.delete("1.0", "end")
            for entry in log[-25:]:
                self._log.insert("end", entry + "\n")
            self._log.see("end")
            self._log.config(state="disabled")


# ── The studio container ────────────────────────────────────────────────────────
class HandStudio(tk.Frame):
    def __init__(self, parent, link=None):
        super().__init__(parent, bg=BG)
        self.link = link                 # BridgeLink, to release a ROS bridge if present
        self.bus = HandBus()
        self._panels: dict = {}
        self._active_nav = "Control"
        self._nav_btns: dict = {}
        self._connecting = False
        self._auto_done = False
        self._released_ros = False        # did we take the port from a ROS bridge?

        self._build_header()
        self._build_body()
        self._last_estop_key = None
        self.bus.subscribe(self._on_bus_state)
        # Auto-connect once the widget is up (so the connected hand is just detected).
        self.after(400, self._auto_connect)

    # ── header ─────────────────────────────────────────────────────────────────
    def _build_header(self):
        bar = tk.Frame(self, bg=SURFACE, height=64)
        bar.pack(fill="x", side="top")
        bar.pack_propagate(False)

        # Nav tabs (left)
        nav = tk.Frame(bar, bg=SURFACE)
        nav.pack(side="left", padx=(16, 10), pady=12)
        for name in _PANELS:
            b = AppleButton(nav, name, lambda n=name: self._switch_panel(n),
                            bg_color=ELEVATED, fg_color=LABEL2, font=FONT_BODY,
                            parent_bg=SURFACE, width=104, height=38)
            b.pack(side="left", padx=4)
            self._nav_btns[name] = b
        self._mark_nav("Control")

        # Connection controls (right)
        right = tk.Frame(bar, bg=SURFACE)
        right.pack(side="right", padx=16, pady=12)

        self._hand_seg = AppleSegmentedControl(
            right, ["left", "right"], self.bus.hand, self._on_hand_toggle,
            width=140, height=36, parent_bg=SURFACE)
        self._hand_seg.pack(side="right", padx=(12, 0))

        self._estop_btn = AppleButton(
            right, "E-STOP", self._toggle_torque, bg_color=ELEVATED, fg_color=LABEL2,
            font=FONT_BODY, parent_bg=SURFACE, width=120, height=38)
        self._estop_btn.pack(side="right", padx=6)

        self._conn_btn = AppleButton(
            right, "Connect", self._on_conn_btn, bg_color=ORANGE, fg_color=LABEL,
            font=FONT_BODY, parent_bg=SURFACE, width=120, height=38)
        self._conn_btn.pack(side="right", padx=6)

        status = tk.Frame(right, bg=SURFACE)
        status.pack(side="right", padx=(0, 12))
        self._status_dot = StatusDot(status, d=11, bg=SURFACE, color=RED)
        self._status_dot.pack(side="left", padx=(0, 6))
        self._conn_label = tk.Label(status, text="Disconnected", fg=LABEL2,
                                    bg=SURFACE, font=FONT_SMALL)
        self._conn_label.pack(side="left")

        tk.Frame(self, bg=SEP, height=1).pack(fill="x")

    def _mark_nav(self, name):
        for n, b in self._nav_btns.items():
            b.update_colors(ORANGE if n == name else ELEVATED, LABEL if n == name else LABEL2)

    # ── body ───────────────────────────────────────────────────────────────────
    def _build_body(self):
        # Three columns: robot status (left) · 3D model view (centre, main) ·
        # calibration + joint control (right). The nav tabs in the header pick which
        # control panel occupies the right dock; the 3D model always sits in the middle.
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True)

        # LEFT — robot status (live per-motor telemetry)
        self._sidebar = TelemetrySidebar(body, self.bus)
        self._sidebar.pack(side="left", fill="y")
        tk.Frame(body, bg=SEP, width=1).pack(side="left", fill="y")

        # RIGHT — calibration + joint control (the active nav panel), scrollable dock
        tk.Frame(body, bg=SEP, width=1).pack(side="right", fill="y")
        self._dock = _ScrollDock(body, CONTROLS_W)
        self._dock.pack(side="right", fill="y")
        self._content = self._dock.inner        # panels are packed in here

        # CENTRE (main) — the live 3D model view of the hand
        self._model = HandModelView(body, self.bus, self)
        self._model.pack(side="left", fill="both", expand=True)

        self._switch_panel("Control")

    # ── panel switching ─────────────────────────────────────────────────────────
    def _switch_panel(self, name):
        self._active_nav = name
        self._mark_nav(name)
        for p in self._panels.values():
            p.pack_forget()
        if name not in self._panels:
            try:
                self._panels[name] = _PANELS[name](self._content, self.bus)
            except Exception as exc:
                err = tk.Label(self._content, text=f"Panel '{name}' failed to load:\n{exc}",
                               bg=BG, fg=RED, font=FONT_BODY, justify="left")
                err.pack(padx=40, pady=60, anchor="w")
                self.bus.log(f"Panel error [{name}]: {exc}")
                return
        self._panels[name].pack(fill="both", expand=True)

    def _refresh_panels(self):
        active = self._active_nav
        for name in list(self._panels.keys()):
            try:
                self._panels[name].destroy()
            except Exception:
                pass
            del self._panels[name]
        self._switch_panel(active)

    # ── connection ───────────────────────────────────────────────────────────--
    def _auto_connect(self):
        if self._auto_done or self.bus.state.connected:
            return
        self._auto_done = True
        self._connect("")

    def _on_conn_btn(self):
        if self.bus.state.connected:
            self._disconnect()
        else:
            self._connect("")

    def _connect(self, port: str):
        if self._connecting:
            return
        self._connecting = True
        self._conn_label.config(text="Connecting…", fg=ORANGE)
        self.update_idletasks()
        # On the robot a ROS bridge may hold the serial port — release it so the hand
        # bridge can own it (no-op on a Mac bench: nothing is holding the lock).
        if self.link is not None:
            try:
                if self.link.bridge_running():
                    self.link.release_bridge()
                    self._released_ros = True
            except Exception:
                pass
        ok, msg = self.bus.connect(port, self._hand_seg.get())
        self._connecting = False
        if ok:
            self._set_conn_ui(True)
            self._refresh_panels()
        else:
            self._set_conn_ui(False)
            self._conn_label.config(text=f"Failed: {msg[:40]}", fg=RED)
            self.bus.log(f"Connect failed: {msg}")

    def _disconnect(self):
        self.bus.disconnect()
        self._set_conn_ui(False)
        self._refresh_panels()
        # Give the hand back to the ROS bridge only if we took it from one (never spawn
        # a ROS bridge on a Mac bench that never had one).
        if self.link is not None and self._released_ros:
            try:
                self.link.ensure_bridge()
                self._released_ros = False
            except Exception:
                pass

    def _set_conn_ui(self, connected: bool):
        if connected:
            short = (self.bus.port or "auto").split("/")[-1] or "auto"
            self._status_dot.set(GREEN)
            self._conn_label.config(text=f"{self.bus.hand} · {short}", fg=LABEL)
            self._conn_btn.update_colors(ELEVATED, LABEL)
            self._conn_btn.update_text("Disconnect")
        else:
            self._status_dot.set(RED)
            self._conn_label.config(text="Disconnected", fg=LABEL2)
            self._conn_btn.update_colors(ORANGE, LABEL)
            self._conn_btn.update_text("Connect")
        self._refresh_estop()

    def _refresh_estop(self):
        if not self.bus.state.connected:
            self._estop_btn.update_colors(ELEVATED, LABEL2)
            self._estop_btn.update_text("E-STOP")
        elif self.bus.state.torque_on:
            self._estop_btn.update_colors(RED, LABEL)
            self._estop_btn.update_text("E-STOP")
        else:
            self._estop_btn.update_colors(GREEN, LABEL)
            self._estop_btn.update_text("ENABLE")

    def _on_bus_state(self, state, log):
        key = (state.connected, state.torque_on)
        if key != getattr(self, "_last_estop_key", None):
            self._last_estop_key = key
            try:
                self.after(0, self._refresh_estop)
            except Exception:
                pass

    def _toggle_torque(self):
        if not self.bus.state.connected:
            self.bus.log("Connect a hand before enabling torque.")
            return
        self.bus.set_torque(not self.bus.state.torque_on)
        self._refresh_estop()

    def _on_hand_toggle(self, choice: str):
        if choice == self.bus.hand and self.bus.state.connected:
            return
        if self.bus.state.connected:
            self.bus.disconnect()
        self.bus.hand = choice
        self._connect("")

    # ── lifecycle (called by the Control Center container) ─────────────────────
    def refresh(self, snapshot=None):
        # The studio drives its own telemetry over the hand bridge, so the Control
        # Center's DDS snapshot poll is not needed here.
        pass

    def teardown(self):
        try:
            if getattr(self, "_model", None) is not None:
                self._model.close()
        except Exception:
            pass
        try:
            if self.bus.state.connected:
                self.bus.set_torque(False)
            self.bus.disconnect()
        except Exception:
            pass
