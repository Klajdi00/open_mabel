#!/usr/bin/env python3
"""app.py - MABEL Firmware Bench: one GUI for bench-testing the whole hardware
stack — calibration, live visualization, and quick joint-level control.

Combines the per-subsystem GUIs that used to live under firmware/ (orca hand, swerve
base, lift, and arms/body sliders) into a single iOS-dark studio that talks ONLY to the
hardware bridge over DDS - never to a device directly. The bridge is the sole hardware
owner; this app is just another of its clients.

  ┌─ toolbar: title · [Overview Base Arms Body Hands Lift] ·  bridge · GLOBAL E-STOP ─┐
  │  content: the active subsystem panel                                              │
  └───────────────────────────────────────────────────────────────────────────────────┘

Run:  ./gui/control_center.sh      (sources the same ROS env as the bridge)
      python3 -m gui               (if rclpy is already on the path)
"""

import sys
import os
import tkinter as tk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from style import (BG, ELEVATED, GREEN, LABEL, LABEL2, ORANGE, RED, SEP, SURFACE,
                   FONT_SMALL, FONT_TITLE, AppleButton, StatusDot, apply_global_fonts)
from bridge_link import BridgeLink
import panels as P

WIN_W, WIN_H = 1600, 860       # Hands studio: status | 3D model (main) | controls dock
POLL_MS = 80           # telemetry refresh (~12 Hz) - cheap, smooth enough

_NAV = [
    ("Overview", P.OverviewPanel),
    ("Base", P.BasePanel),
    ("Arms", P.ArmsPanel),
    ("Body", P.BodyPanel),
    ("Hands", P.HandsPanel),
    ("Lift", P.LiftPanel),
    ("Monitor", P.MonitorPanel),
]


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        apply_global_fonts(self)
        self.title("MABEL Firmware Bench")
        self.configure(bg=BG)
        self.geometry(f"{WIN_W}x{WIN_H}")
        self.minsize(980, 680)

        self.link = BridgeLink()
        self._panels = {}
        self._nav_btns = {}
        self._active = None
        self._estopped = False

        self._build_toolbar()
        self._content = tk.Frame(self, bg=BG)
        self._content.pack(fill="both", expand=True)

        self._connect_bridge()
        self._switch("Overview")
        self.after(300, self._poll)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        # Panic keybinds: spacebar / Escape = GLOBAL E-STOP from anywhere in the app.
        # Escape = panic E-STOP from anywhere. (Space is NOT bound — it would fire while
        # typing in a port/pose entry; the big red toolbar button is the primary control.)
        self.bind_all("<Escape>", lambda e: self._global_estop(force=True))

    # ── toolbar ────────────────────────────────────────────────────────────--
    def _build_toolbar(self):
        bar = tk.Frame(self, bg=SURFACE, height=78)
        bar.pack(fill="x", side="top")
        bar.pack_propagate(False)

        title = tk.Frame(bar, bg=SURFACE)
        title.pack(side="left", padx=22, pady=12)
        tk.Label(title, text="MABEL", bg=SURFACE, fg=LABEL, font=FONT_TITLE).pack(anchor="w")
        tk.Label(title, text="Firmware Bench", bg=SURFACE, fg=ORANGE,
                 font=FONT_SMALL).pack(anchor="w")

        nav = tk.Frame(bar, bg=SURFACE)
        nav.pack(side="left", padx=18, pady=20)
        for name, _ in _NAV:
            b = AppleButton(nav, name, lambda n=name: self._switch(n),
                            bg_color=ELEVATED, fg_color=LABEL2, parent_bg=SURFACE,
                            width=96, height=38)
            b.pack(side="left", padx=4)
            self._nav_btns[name] = b

        right = tk.Frame(bar, bg=SURFACE)
        right.pack(side="right", padx=20, pady=20)
        self._estop_btn = AppleButton(right, "GLOBAL E-STOP", self._global_estop,
                                      bg_color=RED, fg_color=LABEL, parent_bg=SURFACE,
                                      width=150, height=40)
        self._estop_btn.pack(side="right", padx=6)
        status = tk.Frame(right, bg=SURFACE)
        status.pack(side="right", padx=(0, 14))
        self._dot = StatusDot(status, d=11, bg=SURFACE, color=RED)
        self._dot.pack(side="left", padx=(0, 7))
        self._conn = tk.Label(status, text="bridge: connecting...", fg=LABEL2, bg=SURFACE,
                              font=FONT_SMALL)
        self._conn.pack(side="left")

        tk.Frame(self, bg=SEP, height=1).pack(fill="x")

    def _mark_nav(self, name):
        for n, b in self._nav_btns.items():
            b.update_colors(ORANGE if n == name else ELEVATED, LABEL if n == name else LABEL2)

    # ── bridge ───────────────────────────────────────────────────────────────
    def _connect_bridge(self):
        try:
            self.link.start()
        except Exception as e:
            self._dot.set(RED)
            self._conn.config(text=f"bridge: rclpy unavailable ({type(e).__name__})", fg=RED)
            return
        # Make sure a bridge is actually OWNING the hardware so we detect it - start one
        # if none is running (reuse an existing run_bridge.sh / systemd / real_run one).
        try:
            how = self.link.ensure_bridge()
        except Exception:
            how = "skipped"
        self._dot.set(GREEN)
        msg = {"existing": "bridge: linked (running)",
               "started": "bridge: started - detecting hardware...",
               "skipped": "bridge: client-only (none running)"}.get(how, "bridge: linked")
        self._conn.config(text=msg, fg=LABEL if how != "skipped" else ORANGE)

    def _global_estop(self, force=None):
        """Panic toggle. force=True engages (keybinds always engage, never release);
        force=None toggles. Engaging stops the base + free-drives both hands; releasing
        clears the base latch so the robot is drivable again."""
        engage = True if force is True else (not self._estopped)
        self._estopped = engage
        try:
            self.link.base_estop(engage)
            if engage:
                self.link.base_twist(0.0, 0.0, 0.0)
                for s in ("left", "right"):
                    self.link.hand_torque(s, False)
        except Exception:
            pass
        if engage:
            self._estop_btn.update_text("RELEASE E-STOP")
            self._estop_btn.update_colors(GREEN, LABEL)
            self._conn.config(text="bridge: E-STOP engaged", fg=RED)
        else:
            self._estop_btn.update_text("GLOBAL E-STOP")
            self._estop_btn.update_colors(RED, LABEL)
            self._conn.config(text="bridge: E-STOP released", fg=GREEN)

    # ── panels ───────────────────────────────────────────────────────────────
    def _switch(self, name):
        self._active = name
        self._mark_nav(name)
        for p in self._panels.values():
            p.pack_forget()
        if name not in self._panels:
            cls = dict(_NAV)[name]
            try:
                self._panels[name] = cls(self._content, self.link)
                # let the Overview "system status" cards switch to a module tab
                if hasattr(self._panels[name], "on_navigate"):
                    self._panels[name].on_navigate = self._switch
            except Exception as exc:
                err = tk.Label(self._content, text=f"Panel '{name}' failed:\n{exc}",
                               bg=BG, fg=RED, justify="left")
                err.pack(padx=40, pady=60, anchor="w")
                return
        self._panels[name].pack(fill="both", expand=True)

    def _poll(self):
        if self.link.connected:
            snap = self.link.snapshot()
            panel = self._panels.get(self._active)
            if panel is not None and hasattr(panel, "refresh"):
                try:
                    panel.refresh(snap)
                except Exception:
                    pass
        self.after(POLL_MS, self._poll)

    def _on_close(self):
        # Let panels release hardware first (e.g. the Hand Studio disables torque +
        # closes its bridge connection) before the DDS link is torn down.
        for p in self._panels.values():
            if hasattr(p, "teardown"):
                try:
                    p.teardown()
                except Exception:
                    pass
        try:
            self.link.stop()
        finally:
            self.destroy()


def main():
    App().mainloop()


if __name__ == "__main__":
    main()
