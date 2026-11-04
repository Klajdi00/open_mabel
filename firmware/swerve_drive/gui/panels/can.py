"""CAN panel — bring up, restart, and diagnose the CAN bus from the GUI.

Wraps ``scripts/setup/can_bringup.sh``.  Every action runs in a worker thread
so the UI never freezes, ANSI color codes are stripped, and output streams into
a terminal-style console.  A header strip polls ``status`` and shows the live
link / bitrate / bus-health at a glance, with a colored dot.

Note: bringup/restart/down call ``sudo`` inside the script.  Launch the GUI
from a terminal where sudo is cached (or after ``install-autoup``) so those
actions don't stall waiting for a password.
"""

import queue
import re
import subprocess
import threading
import tkinter as tk

from .. import config as C
from .. import theme as T
from ..widgets import AppleButton, StatusDot
from .base import Panel

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


class CanPanel(Panel):
    title = "CAN Bus"
    accent = T.BLUE

    def __init__(self, host, app):
        super().__init__(host, app)
        self._q = queue.Queue()
        self._busy = False
        self._build()
        self.after(120, self._drain)
        self._refresh_status()

    # ── Construction ───────────────────────────────────────────────────────────
    def _build(self):
        # Status strip
        strip = tk.Frame(self, bg=T.SURFACE, height=78)
        strip.pack(fill="x")
        strip.pack_propagate(False)

        idf = tk.Frame(strip, bg=T.SURFACE)
        idf.pack(side="left", padx=24, pady=14)
        self._dot = StatusDot(idf, color=T.LABEL2, bg=T.SURFACE, size=18)
        self._dot.pack(side="left", padx=(0, 10))
        meta = tk.Frame(idf, bg=T.SURFACE)
        meta.pack(side="left")
        tk.Label(meta, text=C.CAN_IFACE, bg=T.SURFACE, fg=T.LABEL,
                 font=T.FONT_TITLE).pack(anchor="w")
        self._summary = tk.StringVar(value="checking…")
        tk.Label(meta, textvariable=self._summary, bg=T.SURFACE, fg=T.LABEL2,
                 font=T.FONT_SMALL).pack(anchor="w")

        kpis = tk.Frame(strip, bg=T.SURFACE)
        kpis.pack(side="right", padx=24)
        self._kpi_state = self._kpi(kpis, "LINK")
        self._kpi_rate = self._kpi(kpis, "BITRATE")
        self._kpi_bus = self._kpi(kpis, "BUS")

        tk.Frame(self, bg=T.SEP, height=1).pack(fill="x")

        # Action bar
        actions = tk.Frame(self, bg=T.BG)
        actions.pack(fill="x", padx=22, pady=(18, 8))
        self._mk_action(actions, "Bring Up", "up", T.GREEN, fg=T.BG)
        self._mk_action(actions, "Restart", "restart", T.ORANGE, fg=T.BG)
        self._mk_action(actions, "Bring Down", "down", T.ELEVATED)
        self._mk_action(actions, "Status", "status", T.BLUE)
        self._mk_action(actions, "Doctor", "doctor", T.PURPLE)
        self._mk_action(actions, "Scan", "scan", T.ELEVATED)
        self._mk_action(actions, "Auto-Up", "install-autoup", T.ELEVATED)

        # Console
        console_wrap = tk.Frame(self, bg=T.PANEL_BG)
        console_wrap.pack(fill="both", expand=True, padx=22, pady=(8, 22))
        bar = tk.Frame(console_wrap, bg=T.PANEL_BG)
        bar.pack(fill="x", padx=14, pady=(12, 6))
        tk.Label(bar, text="CONSOLE", bg=T.PANEL_BG, fg=T.LABEL2,
                 font=T.FONT_SMALL).pack(side="left")
        AppleButton(bar, "Clear", self._clear, bg_color=T.ELEVATED,
                    parent_bg=T.PANEL_BG, width=70, height=28,
                    font=T.FONT_SMALL, radius=14).pack(side="right")

        self._console = tk.Text(console_wrap, bg="#0A0A0B", fg=T.LABEL,
                                insertbackground=T.LABEL, font=T.FONT_MONO,
                                relief="flat", padx=14, pady=10, wrap="word",
                                height=12, state="disabled")
        self._console.pack(fill="both", expand=True, padx=14, pady=(0, 14))
        for tag, color in (("ok", T.GREEN), ("err", T.RED), ("warn", T.YELLOW),
                           ("cmd", T.BLUE), ("dim", T.LABEL2)):
            self._console.tag_config(tag, foreground=color)

        self._log(f"can_bringup.sh → {C.CAN_BRINGUP}", "dim")
        if not C.CAN_BRINGUP.exists():
            self._log("WARNING: can_bringup.sh not found at the path above.", "err")

    def _kpi(self, parent, label):
        box = tk.Frame(parent, bg=T.SURFACE)
        box.pack(side="left", padx=14)
        var = tk.StringVar(value="—")
        tk.Label(box, textvariable=var, bg=T.SURFACE, fg=T.LABEL,
                 font=T.FONT_SUB).pack()
        tk.Label(box, text=label, bg=T.SURFACE, fg=T.LABEL2,
                 font=T.FONT_SMALL).pack()
        return var

    def _mk_action(self, parent, text, sub, color, fg=T.LABEL):
        AppleButton(parent, text, lambda s=sub: self._run(s), bg_color=color,
                    fg_color=fg, parent_bg=T.BG, width=124, height=40
                    ).pack(side="left", padx=(0, 10))

    # ── Console helpers ────────────────────────────────────────────────────────
    def _log(self, text, tag=None):
        self._console.config(state="normal")
        self._console.insert("end", text + "\n", tag or ())
        self._console.see("end")
        self._console.config(state="disabled")

    def _clear(self):
        self._console.config(state="normal")
        self._console.delete("1.0", "end")
        self._console.config(state="disabled")

    @staticmethod
    def _classify(line):
        """Only color the script's human-readable summary lines; the raw ``ip``
        dump (which legitimately contains words like ``errors`` and the healthy
        ``ERROR-ACTIVE`` state) is left default-white for readability."""
        low = line.lower()
        if "✓" in line or "healthy" in low or "is up at" in low or "is ready" in low:
            return "ok"
        if ("✗" in line or "bus is off" in low or "does not exist" in low
                or "not found" in low or "failed" in low or "cannot" in low):
            return "err"
        if ("degraded" in low or "failing" in low or "administratively down" in low
                or "missing" in low or "no motors" in low or "not loaded" in low):
            return "warn"
        return None

    # ── Command runner ─────────────────────────────────────────────────────────
    def _run(self, sub, *, quiet=False):
        if self._busy:
            if not quiet:
                self._log("Busy — wait for the current command to finish.", "warn")
            return
        if not C.CAN_BRINGUP.exists():
            self._log("can_bringup.sh missing; cannot run.", "err")
            return
        self._busy = True
        if not quiet:
            self._q.put(("line", f"$ can_bringup.sh {sub}", "cmd"))
        threading.Thread(target=self._worker, args=(sub, quiet), daemon=True).start()

    def _worker(self, sub, quiet):
        env = {"CAN_IFACE": C.CAN_IFACE, "CAN_BITRATE": str(C.CAN_BITRATE)}
        try:
            proc = subprocess.run(
                ["bash", str(C.CAN_BRINGUP), sub],
                capture_output=True, text=True, timeout=60,
                env={**__import__("os").environ, **env},
            )
            out = (proc.stdout or "") + (proc.stderr or "")
        except subprocess.TimeoutExpired:
            out = "[timed out after 60s]"
        except Exception as e:
            out = f"[failed to launch: {e}]"

        if not quiet:                       # quiet polls update the header only
            for raw in out.splitlines():
                clean = _ANSI.sub("", raw).rstrip()
                self._q.put(("line", clean, self._classify(clean)))
        self._q.put(("status_text", out, None))
        self._q.put(("done", None, None))

    def _drain(self):
        try:
            while True:
                kind, payload, tag = self._q.get_nowait()
                if kind == "line":
                    self._log(payload, tag)
                elif kind == "status_text":
                    self._update_header(payload)
                elif kind == "done":
                    self._busy = False
        except queue.Empty:
            pass
        self.after(120, self._drain)

    # ── Status header ──────────────────────────────────────────────────────────
    def _refresh_status(self):
        self._run("status", quiet=True)
        self.after(5000, self._refresh_status)

    def _update_header(self, text):
        clean = _ANSI.sub("", text)
        link = self._grep(clean, r"link=(\S+)") or ("UP" if "state UP" in clean else None)
        rate = self._grep(clean, r"bitrate=(\S+)") or self._grep(clean, r"bitrate\s+(\d+)")
        bus = self._grep(clean, r"bus=(\S+)") or self._grep(clean, r"can state (\S+)")

        if link:
            self._kpi_state.set(link)
        if rate:
            try:
                self._kpi_rate.set(f"{int(rate) // 1000}k")
            except ValueError:
                self._kpi_rate.set(rate)
        if bus:
            self._kpi_bus.set(bus)

        if "does not exist" in clean or "MISSING" in clean or not link:
            self._dot.set_color(T.RED)
            self._summary.set("interface missing — plug in adapter / Bring Up")
        elif bus and "ERROR-ACTIVE" in bus:
            self._dot.set_color(T.GREEN)
            self._summary.set("bus healthy")
        elif link == "UP":
            self._dot.set_color(T.YELLOW)
            self._summary.set(f"up · bus {bus or 'unknown'}")
        else:
            self._dot.set_color(T.YELLOW)
            self._summary.set("down — Bring Up to activate")

    @staticmethod
    def _grep(text, pattern):
        m = re.search(pattern, text)
        return m.group(1) if m else None
