"""calibration.py - ORCA hand end-stop calibration tab (direct HAL, NOT ROS).

A faithful port of the firmware GUI's calibration experience into the Control Center.
Unlike the other tabs (DDS clients of the bridge), this one opens the Feetech serial bus
*directly* through :class:`gui.hand_session.HandSession` - so the bridge's hand subsystem
for the selected side must be STOPPED first (the bus is exclusive).

Four sub-modes, mirroring the firmware GUI:

  Tension        drive every finger tendon into its hard-stop and hold, so you can take
                 up the slack side; Stop releases.
  Auto           stall-sweep every joint to both end-stops (the validated Calibrator),
                 with a live per-joint ok grid.
  Select Finger  same sweep, but only the joints you tick (non-destructive merge).
  Manual         torque off; sweep each joint by hand while a coverage bar fills; Save
                 keeps the joints you actually moved.

Results are written to the selected side's ``<side>_calibration.yaml`` - the same file
the bridge HAL loads as its clamp + tick<->radian scale - then reloaded live.
"""

import queue
import threading
import tkinter as tk

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # gui/

from style import (BG, ELEVATED, GREEN, LABEL, LABEL2, LABEL3, ORANGE, PANEL_BG, RED,
                   SURFACE, YELLOW, FONT_BODY, FONT_MONO_S, FONT_SMALL, FONT_SUB,
                   AppleSegmentedControl, ProgressBar, RangeBar, accent_btn, btn, lbl)
from widgets import scroll_column
from hand_session import HandSession

_SUBS = ["Tension", "Auto", "Select Finger", "Manual"]
_STATUS_GLYPH = {"pending": ("o", LABEL3), "running": ("*", ORANGE),
                 "ok": ("ok", GREEN), "warn": ("!", YELLOW), "fail": ("x", RED)}


class HandCalibrationView(tk.Frame):
    def __init__(self, parent, link=None):
        super().__init__(parent, bg=BG)
        self.link = link                 # to hand the serial bus off from the bridge
        self.session = HandSession(log=self._log_line)
        self._q: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._worker = None
        self._chips = {}            # joint name -> {"dot":Label,"info":Label}
        self._manual_rows = {}      # mid -> {"bar":RangeBar,"info":Label,"dot":Label}
        self._obs = {}              # mid -> [lo, hi] observed this manual session
        self._sub = "Tension"

        self._build_header()
        self._sub_seg_holder = tk.Frame(self, bg=BG)
        self._sub_seg_holder.pack(fill="x", padx=20, pady=(4, 2))
        self._sub_seg = AppleSegmentedControl(self._sub_seg_holder, _SUBS, self._sub,
                                              self._switch_sub, height=34, parent_bg=BG)
        self._sub_seg.pack(side="left")

        self._body = tk.Frame(self, bg=BG)
        self._body.pack(fill="both", expand=True, padx=12, pady=(4, 4))

        self._log = lbl(self, "", fg=LABEL3, font=FONT_MONO_S, bg=BG, anchor="w")
        self._log.pack(fill="x", padx=20, pady=(0, 8))

        self._render_sub()
        self.after(120, self._drain)

    # ── header: side + auto-connect (no port field) ───────────────────────────
    def _build_header(self):
        head = tk.Frame(self, bg=BG)
        head.pack(fill="x", padx=20, pady=(14, 4))
        tk.Label(head, text="Calibration", bg=BG, fg=LABEL, font=FONT_SUB).pack(side="left")
        self._conn_lbl = tk.Label(head, text="connecting…", bg=BG, fg=LABEL2,
                                  font=FONT_SMALL)
        self._conn_lbl.pack(side="right")

        bar = tk.Frame(self, bg=BG)
        bar.pack(fill="x", padx=20, pady=(2, 2))
        self._side_seg = AppleSegmentedControl(bar, ["left", "right"], "left",
                                               self._switch_side, width=150, height=34,
                                               parent_bg=BG)
        self._side_seg.pack(side="left")
        tk.Label(bar, text="takes the bus over from the bridge for this side · "
                           "switch back to Control to hand it back",
                 bg=BG, fg=LABEL3, font=FONT_MONO_S, anchor="w").pack(side="left", padx=12)

    # Auto-connect with a bus HANDOVER: stop the GUI-owned bridge (which holds the port
    # exclusively), then open the Feetech bus directly for the selected side.
    def _auto_connect(self):
        if self.session.connected:
            return
        # Default to the hand that is actually live on the bridge (so the connected hand
        # is the one calibrated), THEN take the serial bus over from the bridge.
        if self.link is not None:
            try:
                live = self.link.live_hand_side()
                if live:
                    self._side_seg.set(live)
            except Exception:
                pass
            self._conn_lbl.config(text="taking the bus from the bridge…", fg=ORANGE)
            self.update_idletasks()
            try:
                self.link.release_bridge()
            except Exception:
                pass
        side = self._side_seg.get()
        err = None
        for _ in range(6):                  # the tty may take a moment to free
            try:
                self.session.connect(side, "")          # "" = auto-detect
                err = None
                break
            except Exception as e:
                err = e
                import time as _t; _t.sleep(0.4)
        if err is not None or not self.session.connected:
            self._conn_lbl.config(text="bus busy — stop the bridge first", fg=RED)
            self._log_line(f"connect failed: {err}")
            self._render_sub()
            return
        self._refresh_conn()
        self._render_sub()

    def _switch_side(self, _side):
        self._kill_worker()
        self.session.disconnect()
        self._auto_connect()

    def _refresh_conn(self):
        if self.session.connected:
            miss = f" · missing {self.session.missing}" if self.session.missing else ""
            self._conn_lbl.config(
                text=f"{self.session.side} on {self.session.port or 'auto'}{miss}", fg=GREEN)
        else:
            self._conn_lbl.config(text="not connected", fg=LABEL2)

    # ── show/hide: the container calls these so the bus is handed back ─────────
    def on_show(self):
        self._auto_connect()

    def on_hide(self):
        self._kill_worker()
        self.session.disconnect()
        if self.link is not None:
            try:
                self.link.ensure_bridge()        # give the hand back to the bridge
            except Exception:
                pass

    # ── sub-tab switching ─────────────────────────────────────────────────────
    def _switch_sub(self, sub):
        if self._worker_running():
            self._sub_seg.set(self._sub)        # don't switch mid-run
            return
        self._sub = sub
        self._render_sub()

    def _render_sub(self):
        for w in self._body.winfo_children():
            w.destroy()
        self._chips, self._manual_rows = {}, {}
        if not self.session.connected:
            lbl(self._body, "Connect a hand to calibrate.", fg=LABEL2, bg=BG,
                font=FONT_BODY).pack(padx=20, pady=40, anchor="w")
            return
        {"Tension": self._build_tension, "Auto": self._build_auto,
         "Select Finger": self._build_select, "Manual": self._build_manual}[self._sub]()

    def _ordered_joints(self):
        return [self.session.id_joint[m] for m in self.session.motor_ids
                if m in self.session.id_joint]

    # ── Tension ───────────────────────────────────────────────────────────────
    def _build_tension(self):
        lbl(self._body, "Tendon Tensioning", fg=LABEL, font=FONT_SUB, bg=BG).pack(anchor="w", padx=8)
        lbl(self._body, "Drives every finger motor into its hard-stop and holds it there so "
                        "you can take up the slack side of each tendon. Wrist is skipped. "
                        "Press Stop to release.", fg=LABEL2, font=FONT_SMALL, bg=BG,
            justify="left", wraplength=820).pack(anchor="w", padx=8, pady=(2, 10))
        self._tension_status = lbl(self._body, "ready", fg=LABEL2, font=FONT_BODY, bg=BG)
        self._tension_status.pack(anchor="w", padx=8)
        self._tension_prog = ProgressBar(self._body, width=320)
        self._tension_prog.pack(anchor="w", padx=8, pady=8)
        row = tk.Frame(self._body, bg=BG); row.pack(anchor="w", padx=8, pady=6)
        self._tension_run = accent_btn(row, "Tension", lambda: self._start(self._run_tension),
                                       color=ORANGE, parent_bg=BG)
        self._tension_run.pack(side="left")
        btn(row, "Stop", self._stop_worker, bg=RED, parent_bg=BG).pack(side="left", padx=8)

    def _run_tension(self):
        self.session.run_tension(self._stop, self._q.put)

    # ── Auto ──────────────────────────────────────────────────────────────────
    def _build_auto(self):
        lbl(self._body, "Auto Full Calibration", fg=LABEL, font=FONT_SUB, bg=BG).pack(anchor="w", padx=8)
        lbl(self._body, "Sweeps every joint to both end-stops with stall detection and saves "
                        "the measured limits.", fg=LABEL2, font=FONT_SMALL, bg=BG,
            justify="left", wraplength=820).pack(anchor="w", padx=8, pady=(2, 8))
        self._auto_prog = ProgressBar(self._body, width=320)
        self._auto_prog.pack(anchor="w", padx=8, pady=(0, 8))
        self._build_chip_grid(self._ordered_joints())
        row = tk.Frame(self._body, bg=BG); row.pack(anchor="w", padx=8, pady=8)
        accent_btn(row, "Run Full Calibration",
                   lambda: self._start(lambda: self.session.run_auto(None, self._stop, self._q.put)),
                   color=ORANGE, parent_bg=BG).pack(side="left")
        btn(row, "Stop", self._stop_worker, bg=RED, parent_bg=BG).pack(side="left", padx=8)

    # ── Select Finger ─────────────────────────────────────────────────────────
    def _build_select(self):
        lbl(self._body, "Calibrate Selected Joints", fg=LABEL, font=FONT_SUB, bg=BG).pack(anchor="w", padx=8)
        lbl(self._body, "Sweeps only the ticked joints; others keep their saved limits.",
            fg=LABEL2, font=FONT_SMALL, bg=BG, justify="left", wraplength=820).pack(
            anchor="w", padx=8, pady=(2, 8))
        self._sel_vars = {}
        grid = tk.Frame(self._body, bg=BG); grid.pack(anchor="w", padx=8, fill="x")
        joints = self._ordered_joints()
        cols = 4
        for i, j in enumerate(joints):
            v = tk.BooleanVar(value=False)
            self._sel_vars[j] = v
            tk.Checkbutton(grid, text=j, variable=v, bg=BG, fg=LABEL, selectcolor=ELEVATED,
                           activebackground=BG, activeforeground=LABEL, font=FONT_MONO_S,
                           anchor="w", width=14).grid(row=i // cols, column=i % cols,
                                                      sticky="w", padx=4, pady=2)
        self._auto_prog = ProgressBar(self._body, width=320)
        self._auto_prog.pack(anchor="w", padx=8, pady=(8, 8))
        self._build_chip_grid(joints)
        row = tk.Frame(self._body, bg=BG); row.pack(anchor="w", padx=8, pady=8)
        accent_btn(row, "Calibrate Selected", self._run_selected, color=ORANGE,
                   parent_bg=BG).pack(side="left")
        btn(row, "Select All", lambda: [v.set(True) for v in self._sel_vars.values()],
            parent_bg=BG).pack(side="left", padx=8)
        btn(row, "Clear", lambda: [v.set(False) for v in self._sel_vars.values()],
            parent_bg=BG).pack(side="left")
        btn(row, "Stop", self._stop_worker, bg=RED, parent_bg=BG).pack(side="left", padx=8)

    def _run_selected(self):
        sel = [j for j, v in self._sel_vars.items() if v.get()]
        if not sel:
            self._log_line("no joints selected")
            return
        self._start(lambda: self.session.run_auto(sel, self._stop, self._q.put))

    # shared joint-status chip grid (Auto + Select) -----------------------------
    def _build_chip_grid(self, joints):
        grid = tk.Frame(self._body, bg=PANEL_BG); grid.pack(anchor="w", padx=8, fill="x")
        cols = 3
        for i, j in enumerate(joints):
            cell = tk.Frame(grid, bg=PANEL_BG); cell.grid(row=i // cols, column=i % cols,
                                                          sticky="w", padx=8, pady=3)
            dot = tk.Label(cell, text="o", bg=PANEL_BG, fg=LABEL3, font=FONT_BODY)
            dot.pack(side="left")
            tk.Label(cell, text=j, bg=PANEL_BG, fg=LABEL, font=FONT_MONO_S, width=12,
                     anchor="w").pack(side="left", padx=(4, 4))
            info = tk.Label(cell, text="", bg=PANEL_BG, fg=LABEL3, font=FONT_MONO_S,
                            width=16, anchor="w")
            info.pack(side="left")
            self._chips[j] = {"dot": dot, "info": info}

    # ── Manual ────────────────────────────────────────────────────────────────
    def _build_manual(self):
        lbl(self._body, "Manual Calibration", fg=LABEL, font=FONT_SUB, bg=BG).pack(anchor="w", padx=8)
        lbl(self._body, "Torque is OFF - sweep each joint through its full range. A joint "
                        "turns green past 80% of its range. Save keeps only the joints you moved.",
            fg=LABEL2, font=FONT_SMALL, bg=BG, justify="left", wraplength=820).pack(
            anchor="w", padx=8, pady=(2, 8))
        row = tk.Frame(self._body, bg=BG); row.pack(anchor="w", padx=8, pady=(0, 8))
        accent_btn(row, "Save Calibration", self._manual_save, color=ORANGE, parent_bg=BG).pack(side="left")
        btn(row, "Reset", self._manual_reset, parent_bg=BG).pack(side="left", padx=8)

        outer, inner = scroll_column(self._body, bg=PANEL_BG)
        outer.pack(fill="both", expand=True, padx=4)
        self._manual_view = self.session.calibration_view()
        for m in self.session.motor_ids:
            jn = self.session.id_joint.get(m, f"ID {m}")
            r = tk.Frame(inner, bg=PANEL_BG); r.pack(fill="x", padx=10, pady=3)
            dot = tk.Label(r, text="", bg=PANEL_BG, fg=RED, font=FONT_SMALL); dot.pack(side="left")
            tk.Label(r, text=jn, bg=PANEL_BG, fg=LABEL, font=FONT_MONO_S, width=12,
                     anchor="w").pack(side="left", padx=(4, 6))
            bar = RangeBar(r, width=280, bg=PANEL_BG); bar.pack(side="left")
            info = tk.Label(r, text="0%", bg=PANEL_BG, fg=LABEL3, font=FONT_MONO_S,
                            width=18, anchor="w"); info.pack(side="left", padx=8)
            self._manual_rows[m] = {"dot": dot, "bar": bar, "info": info}
        self._manual_reset()

    def _manual_reset(self):
        if not self.session.connected:
            return
        self.session.set_all_torque(False)
        pos = self.session.positions()
        self._obs = {m: [pos.get(m, self._center(m)), pos.get(m, self._center(m))]
                     for m in self.session.motor_ids}
        self._log_line("manual: torque OFF - sweep each joint now")

    def _center(self, mid):
        v = self._manual_view.get(mid)
        return int(v["center"]) if v else 2048

    def _manual_save(self):
        observed = {m: tuple(self._obs[m]) for m in self._obs
                    if self._obs[m][1] - self._obs[m][0] >= 30}
        if not observed:
            self._log_line("manual: no joint swept far enough to save")
            return
        try:
            n = self.session.save_manual(observed)
        except Exception as e:
            self._log_line(f"manual save failed: {e}")
            return
        self._manual_view = self.session.calibration_view()
        self._log_line(f"manual: saved {n} joint(s) -> {self.session.calib_path}")

    def _update_manual_live(self):
        pos = self.session.positions()
        for m, row in self._manual_rows.items():
            cur = pos.get(m)
            if cur is None:
                continue
            o = self._obs.setdefault(m, [cur, cur])
            o[0], o[1] = min(o[0], cur), max(o[1], cur)
            v = self._manual_view.get(m)
            tlo, thi = (int(v["min"]), int(v["max"])) if v else (o[0], o[1])
            tspan = max(1, thi - tlo)
            cov = (o[1] - o[0]) / tspan
            color = GREEN if cov >= 0.8 else (YELLOW if cov >= 0.5 else RED)
            dom_lo, dom_hi = min(tlo, o[0]), max(thi, o[1])
            row["bar"].set(dom_lo, dom_hi, tlo, thi, o[0], o[1], cur, color=color)
            row["dot"].config(fg=color)
            row["info"].config(text=f"{o[0]}-{o[1]}  {int(cov*100)}%", fg=color)

    # ── worker plumbing ───────────────────────────────────────────────────────
    def _worker_running(self):
        return self._worker is not None and self._worker.is_alive()

    def _start(self, target):
        if self._worker_running():
            return
        self._stop.clear()
        self._worker = threading.Thread(target=target, daemon=True)
        self._worker.start()

    def _stop_worker(self):
        self._stop.set()

    def _kill_worker(self):
        self._stop.set()
        w = self._worker
        if w is not None and w.is_alive():
            w.join(timeout=2.0)
        self._worker = None

    def _set_chip(self, joint, status, info=""):
        chip = self._chips.get(joint)
        if not chip:
            return
        glyph, color = _STATUS_GLYPH.get(status, ("o", LABEL3))
        chip["dot"].config(text=glyph, fg=color)
        if info:
            chip["info"].config(text=info, fg=color)

    def _drain(self):
        try:
            while True:
                ev = self._q.get_nowait()
                self._handle(ev)
        except queue.Empty:
            pass
        if self._sub == "Manual" and self.session.connected and not self._worker_running():
            self._update_manual_live()
        self.after(120, self._drain)

    def _handle(self, ev):
        t = ev.get("type")
        if t == "log":
            self._log_line(ev["msg"])
        elif t == "step":
            idx, total = ev.get("idx", 0), ev.get("total", 1)
            if hasattr(self, "_auto_prog"):
                self._auto_prog.set(100.0 * idx / max(1, total))
            self._log_line(f"step {idx}/{total}: {ev.get('label','')}")
        elif t == "joint":
            info = ""
            if ev.get("min") is not None:
                info = f"{ev['min']}-{ev['max']}"
            self._set_chip(ev["joint"], ev.get("status", "pending"), info)
        elif t == "tension":
            total, n = ev.get("total", 1), ev.get("stalled", 0)
            self._tension_status.config(text=f"holding {n}/{total} at the hard-stop",
                                        fg=GREEN if n == total else ORANGE)
            self._tension_prog.set(100.0 * n / max(1, total))
        elif t == "done":
            mode = ev.get("mode")
            if mode == "auto":
                self._log_line(f"calibration done: {ev.get('count',0)} joint(s) "
                               f"-> {ev.get('path','')}")
            elif mode == "tension":
                self._tension_status.config(text="released", fg=LABEL2)

    def _log_line(self, msg):
        try:
            self._log.config(text=str(msg))
        except Exception:
            pass

    # ── lifecycle (called by the container) ───────────────────────────────────
    def refresh(self, *_):
        pass

    def teardown(self):
        self.on_hide()
