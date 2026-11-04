#!/usr/bin/env python3
"""
MABEL Motor Calibration — Apple-styled GUI
==========================================

A single-window app to calibrate one DaMiao actuator end-to-end:

    Connect  →  watch live telemetry (ID / status / pos / vel / torque)  →
    press **Calibrate**  →  see the friction / inertia plots + identified
    MuJoCo numbers, with every step written to the calibration log.

Runs on macOS over the **DM-USB2FDCAN serial** adapter (the default here) and on
Linux over SocketCAN.  The actual motor drive + identification is the shared
:mod:`mabel_actid.session` engine — this file is only the front-end.

    python gui/motor_calib_gui.py
    python gui/motor_calib_gui.py --motor DM-J4340P --id 1 --port /dev/cu.usbmodem1101

Style matches the ORCA Hand Studio (iOS dark, SF Pro) for a consistent MABEL
firmware look.
"""
from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
import tkinter as tk
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent                       # motor_calibration/
for _p in (str(_HERE), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from style import *                                          # noqa: E402,F403
import plots as gp                                           # noqa: E402
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402

from mabel_actid import session, MOTORS                      # noqa: E402
from mabel_actid.data_logging import CalibrationLog          # noqa: E402
from mabel_actid.serial_ports import list_ports              # noqa: E402

WIN_W, WIN_H = 1320, 880
SIDEBAR_W = 320
MOTOR_NAMES = sorted(MOTORS)
# Default to the bench actuator, not whatever sorts first alphabetically
# (that was DM-J10422P, the 400 N·m torso — a bad default that silently
# rescaled every torque reading when the real motor was a DM-J4340P).
DEFAULT_MOTOR = "DM-J4340P" if "DM-J4340P" in MOTORS else MOTOR_NAMES[0]
PLOT_TABS = ["Friction", "Torque fit", "Staircase", "Chirp", "Torque–speed",
             "Performance"]


# ── connection / calibration controller ──────────────────────────────────────

class Controller:
    """Owns the bus, runs the calibration in a worker thread, and pushes events
    (log lines, progress, phase, done/error) onto a queue the UI drains."""

    def __init__(self):
        self.bus = None
        self.backend = session.default_backend()
        self.motor = DEFAULT_MOTOR        # motor TYPE (a MOTORS key, for ranges)
        self.name = DEFAULT_MOTOR         # instance label (results/<name>/)
        self.can_id = 1
        self.timestamp = ""
        self.events: "queue.Queue[tuple]" = queue.Queue()
        self._abort = threading.Event()
        self._worker: threading.Thread | None = None

    # -- connection ----------------------------------------------------------
    @property
    def connected(self) -> bool:
        return self.bus is not None

    @property
    def running(self) -> bool:
        return self._worker is not None and self._worker.is_alive()

    def connect(self, backend, port, motor, can_id):
        self.disconnect()
        self.backend, self.motor, self.can_id = backend, motor, int(can_id)
        bus = session.make_bus(backend, port=port or None)
        bus.add_motor(self.can_id, MOTORS[motor])
        self.bus = bus
        # non-energizing scan (0xCC) so we know right away whether the motor answers
        found = []
        if hasattr(bus, "scan"):
            try:
                found = bus.scan([self.can_id])
            except Exception:
                pass
        return found

    def disconnect(self):
        if self.bus is not None:
            try:
                self.bus.shutdown()
            except Exception:
                pass
            self.bus = None

    def feedback(self):
        if self.bus is None:
            return None
        # poll the motor without energizing it (serial only); socketcan relies on
        # frames arriving from active commands.
        if not self.running and hasattr(self.bus, "refresh_status"):
            try:
                self.bus.refresh_status(self.can_id)
            except Exception:
                pass
        try:
            return self.bus.get(self.can_id)
        except Exception:
            return None

    # -- calibration ---------------------------------------------------------
    def start_calibration(self, cfg: session.CollectConfig, model: str,
                          name: str | None = None):
        if self.running or self.bus is None:
            return
        self.name = (name or self.motor).strip() or self.motor
        self.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self._abort.clear()
        self._worker = threading.Thread(
            target=self._run, args=(cfg, model), daemon=True)
        self._worker.start()

    def abort(self):
        self._abort.set()

    def _emit(self, kind, payload=None):
        self.events.put((kind, payload))

    def _run(self, cfg, model):
        hooks = session.Hooks(
            log=lambda m: self._emit("log", m),
            progress=lambda f: self._emit("progress", f),
            phase=lambda p: self._emit("phase", p),
            should_abort=self._abort.is_set,
            sample=lambda *a: self._emit("sample", a),
        )
        try:
            self._emit("phase", "starting")
            self._emit("log", f"=== calibration start: '{self.name}' "
                              f"(type {self.motor}) id {self.can_id} ===")
            session.run_collection(self.bus, MOTORS[self.motor], self.can_id, cfg,
                                   hooks, name=self.name)
            self._emit("phase", "identify")
            self._emit("log", "--- identifying friction + inertia + torque-speed ---")
            res = session.identify_and_save(self.name, motor_type=self.motor,
                                            model=model, hooks=hooks,
                                            timestamp=self.timestamp)
            session.render_raw_plots(self.name, hooks)
            self._emit("done", res)
        except session.Aborted:
            self._emit("log", "ABORTED by user — motor disabled.")
            self._emit("error", "aborted")
        except Exception as exc:                       # surface the failure
            self._emit("log", f"ERROR: {exc}")
            self._emit("error", str(exc))


# ── main application ──────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self, backend, port, motor, can_id):
        super().__init__()
        self.ctl = Controller()
        self.ctl.motor = motor              # sync selection -> profile/telemetry
        self.ctl.name = motor               # instance label defaults to the type
        self.ctl.can_id = int(can_id)
        self.title("MABEL Motor Calibration")
        self.configure(bg=BG)
        self.geometry(f"{WIN_W}x{WIN_H}")
        self.minsize(1120, 720)

        self._plot_canvas = None
        self._last_res = None
        self._last_report = None        # full params.json dict (incl. isaaclab block)
        self._active_tab = "Friction"

        # live-plot state: buffers streamed from the worker + a persistent figure
        self._liveplot = gp.LivePlot()
        self._live_mode = False
        self._live_params = None
        self._live_buf = self._empty_buf()

        self._build_toolbar(backend, port, motor, can_id)
        self._build_body()

        self.after(150, self._poll_events)
        self.after(200, self._poll_telemetry)
        self._try_load_existing(motor)

    # ── toolbar ─────────────────────────────────────────────────────────────
    def _build_toolbar(self, backend, port, motor, can_id):
        bar = tk.Frame(self, bg=SURFACE, height=92)
        bar.pack(fill="x", side="top"); bar.pack_propagate(False)

        title = tk.Frame(bar, bg=SURFACE)
        title.pack(side="left", padx=24, pady=14)
        tk.Label(title, text="Motor Calibration", bg=SURFACE, fg=LABEL,
                 font=FONT_TITLE).pack(anchor="w")
        tk.Label(title, text="DaMiao Actuator Studio", bg=SURFACE, fg=ORANGE,
                 font=FONT_SMALL).pack(anchor="w")

        # connection controls
        conn = tk.Frame(bar, bg=SURFACE)
        conn.pack(side="left", padx=18)

        self._backend_seg = AppleSegmentedControl(
            conn, ["serial", "gs_usb", "socketcan", "sim"], backend, lambda v: None,
            width=300, height=34, parent_bg=SURFACE)
        self._backend_seg.grid(row=0, column=0, columnspan=2, pady=(18, 4), sticky="w")

        ports = [p.device for p in list_ports()] or ["(none)"]
        self._port_dd = AppleDropDown(conn, ports, default=port or ports[0],
                                      width=220, height=34, parent_bg=SURFACE,
                                      font=FONT_SMALL)
        self._port_dd.grid(row=1, column=0, pady=2, sticky="w")
        AppleButton(conn, "↺", self._refresh_ports, bg_color=ELEVATED,
                    fg_color=LABEL2, font=("Helvetica Neue", 14, "bold"),
                    parent_bg=SURFACE, width=40, height=34).grid(
                        row=1, column=1, padx=6)

        sel = tk.Frame(bar, bg=SURFACE)
        sel.pack(side="left", padx=10)
        tk.Label(sel, text="NAME (saved as)", bg=SURFACE, fg=LABEL2,
                 font=F_CAPTION).grid(row=0, column=0, sticky="w")
        tk.Label(sel, text="TYPE", bg=SURFACE, fg=LABEL2,
                 font=F_CAPTION).grid(row=0, column=1, sticky="w", padx=(8, 0))
        tk.Label(sel, text="ID", bg=SURFACE, fg=LABEL2,
                 font=F_CAPTION).grid(row=0, column=2, sticky="w", padx=(8, 0))
        # free-text instance label — results are filed under results/<NAME>/
        self._name_var = tk.StringVar(value=motor)
        self._name_entry = tk.Entry(
            sel, textvariable=self._name_var, width=16, bg=ELEVATED, fg=LABEL,
            insertbackground=LABEL, relief="flat", font=FONT_SMALL,
            highlightthickness=1, highlightbackground=SEP, highlightcolor=ORANGE)
        self._name_entry.grid(row=1, column=0, ipady=7, pady=2, sticky="we")
        self._name_entry.bind("<FocusOut>", self._on_name_change)
        self._name_entry.bind("<Return>", self._on_name_change)
        self._motor_dd = AppleDropDown(sel, MOTOR_NAMES, default=motor,
                                       width=160, height=34, parent_bg=SURFACE,
                                       font=FONT_SMALL, command=self._on_motor)
        self._motor_dd.grid(row=1, column=1, pady=2, padx=(8, 0))
        self._id_dd = AppleDropDown(sel, [str(i) for i in range(1, 9)],
                                    default=str(can_id), width=70, height=34,
                                    parent_bg=SURFACE, font=FONT_SMALL)
        self._id_dd.grid(row=1, column=2, padx=(8, 0))

        # right: status + connect
        right = tk.Frame(bar, bg=SURFACE)
        right.pack(side="right", padx=22)
        self._status_dot = tk.Label(right, text="●", fg=RED, bg=SURFACE,
                                    font=("Helvetica", 18))
        self._status_dot.pack(side="left", padx=(0, 8))
        self._connect_btn = AppleButton(
            right, "Connect", self._toggle_connect, bg_color=ORANGE,
            fg_color=LABEL, font=FONT_BODY, parent_bg=SURFACE,
            width=150, height=42)
        self._connect_btn.pack(side="left")

        tk.Frame(self, bg=SEP, height=1).pack(fill="x")

    # ── body ──────────────────────────────────────────────────────────────────
    def _build_body(self):
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True)

        self._build_sidebar(body)
        tk.Frame(body, bg=SEP, width=1).pack(side="left", fill="y")
        self._build_content(body)

    # -- left: live telemetry + motor info -----------------------------------
    def _build_sidebar(self, parent):
        sb = tk.Frame(parent, bg=PANEL_BG, width=SIDEBAR_W)
        sb.pack(side="left", fill="y"); sb.pack_propagate(False)

        tk.Label(sb, text="LIVE MOTOR", bg=PANEL_BG, fg=LABEL2,
                 font=FONT_SMALL).pack(pady=(22, 8))

        # status pill
        self._status_lbl = tk.Label(sb, text="Not connected", bg=ELEVATED,
                                    fg=LABEL2, font=FONT_SUB, width=20, pady=8)
        self._status_lbl.pack(padx=18, fill="x")

        # big readouts: POS / VEL / TORQUE
        self._readout = {}
        for key, unit in [("pos", "rad"), ("vel", "rad/s"), ("tau", "N·m")]:
            c = tk.Frame(sb, bg=SURFACE)
            c.pack(fill="x", padx=18, pady=(12, 0))
            tk.Label(c, text=key.upper(), bg=SURFACE, fg=LABEL2,
                     font=F_CAPTION, anchor="w").pack(anchor="w", padx=14, pady=(10, 0))
            val = tk.Label(c, text="—", bg=SURFACE, fg=LABEL,
                           font=(FONT_MONO[0], 30), anchor="w")
            val.pack(anchor="w", padx=14)
            tk.Label(c, text=unit, bg=SURFACE, fg=LABEL3, font=F_CAPTION,
                     anchor="w").pack(anchor="w", padx=14, pady=(0, 10))
            self._readout[key] = val

        # temps row
        trow = tk.Frame(sb, bg=PANEL_BG)
        trow.pack(fill="x", padx=18, pady=(12, 0))
        self._temp_mos = tk.Label(trow, text="MOS —°", bg=PANEL_BG, fg=LABEL2,
                                  font=FONT_MONO)
        self._temp_mos.pack(side="left")
        self._temp_rot = tk.Label(trow, text="ROTOR —°", bg=PANEL_BG, fg=LABEL2,
                                  font=FONT_MONO)
        self._temp_rot.pack(side="right")

        tk.Frame(sb, bg=SEP, height=1).pack(fill="x", pady=(18, 0))
        tk.Label(sb, text="PROFILE", bg=PANEL_BG, fg=LABEL2,
                 font=FONT_SMALL).pack(pady=(12, 6))
        self._profile_box = tk.Frame(sb, bg=PANEL_BG)
        self._profile_box.pack(fill="x", padx=18)
        self._render_profile(self.ctl.motor)

    def _render_profile(self, motor):
        for w in self._profile_box.winfo_children():
            w.destroy()
        p = MOTORS[motor]
        rows = [("model", p.name), ("role", p.role),
                ("p_max", f"±{p.p_max:.2f} rad"),
                ("v_max", f"±{p.v_max:.1f} rad/s"),
                ("t_max", f"±{p.t_max:.1f} N·m"),
                ("gear", f"{p.gear_ratio:.0f}:1")]
        for k, v in rows:
            r = tk.Frame(self._profile_box, bg=PANEL_BG)
            r.pack(fill="x", pady=1)
            tk.Label(r, text=k, bg=PANEL_BG, fg=LABEL3,
                     font=FONT_MONO_S, anchor="w").pack(side="left")
            tk.Label(r, text=v, bg=PANEL_BG, fg=LABEL,
                     font=FONT_MONO_S, anchor="e").pack(side="right")

    # -- center: calibrate + plots + log -------------------------------------
    def _build_content(self, parent):
        content = tk.Frame(parent, bg=BG)
        content.pack(side="left", fill="both", expand=True)

        # calibrate card
        card_f = tk.Frame(content, bg=SURFACE)
        card_f.pack(fill="x", padx=16, pady=(16, 8))

        top = tk.Frame(card_f, bg=SURFACE)
        top.pack(fill="x", padx=16, pady=14)

        self._calib_btn = AppleButton(
            top, "Calibrate", self._on_calibrate, bg_color=ORANGE,
            fg_color=LABEL, font=FONT_SUB, parent_bg=SURFACE,
            width=200, height=56, radius=16)
        self._calib_btn.pack(side="left")

        opts = tk.Frame(top, bg=SURFACE)
        opts.pack(side="left", padx=24)
        tk.Label(opts, text="SPEED  (fraction of rated v_max)", bg=SURFACE,
                 fg=LABEL2, font=F_CAPTION, anchor="w").pack(anchor="w")
        srow = tk.Frame(opts, bg=SURFACE); srow.pack(anchor="w")
        self._vel_slider = AppleSlider(srow, 5, 40, command=self._on_vel,
                                       width=220, height=28)
        self._vel_slider.set(10)
        self._vel_slider.pack(side="left")
        self._vel_lbl = tk.Label(srow, text="0.10", bg=SURFACE, fg=ORANGE,
                                 font=FONT_MONO)
        self._vel_lbl.pack(side="left", padx=8)
        crow = tk.Frame(opts, bg=SURFACE); crow.pack(anchor="w", pady=(6, 0))
        tk.Label(crow, text="INERTIA SWEEP", bg=SURFACE, fg=LABEL2,
                 font=F_CAPTION).pack(side="left", padx=(0, 8))
        self._chirp_seg = AppleSegmentedControl(
            crow, ["on", "off"], "on", lambda v: None,
            width=110, height=30, parent_bg=SURFACE)
        self._chirp_seg.pack(side="left")
        tk.Label(crow, text="TORQUE–SPEED", bg=SURFACE, fg=LABEL2,
                 font=F_CAPTION).pack(side="left", padx=(16, 8))
        self._ts_seg = AppleSegmentedControl(
            crow, ["on", "off"], "on", lambda v: None,
            width=110, height=30, parent_bg=SURFACE)
        self._ts_seg.pack(side="left")

        prog = tk.Frame(card_f, bg=SURFACE)
        prog.pack(fill="x", padx=16, pady=(0, 14))
        self._phase_lbl = tk.Label(prog, text="idle", bg=SURFACE, fg=LABEL2,
                                   font=FONT_SMALL, anchor="w")
        self._phase_lbl.pack(anchor="w")
        self._progress = ProgressBar(prog, width=WIN_W - SIDEBAR_W - 80,
                                     height=6, fill=ORANGE)
        self._progress.pack(anchor="w", pady=(4, 0))

        # results strip
        self._results_lbl = tk.Label(
            content, text="No calibration yet.", bg=BG, fg=LABEL2,
            font=FONT_MONO_S, anchor="w", justify="left")
        self._results_lbl.pack(fill="x", padx=20, pady=(0, 4))

        # plot tabs + canvas
        tabbar = tk.Frame(content, bg=BG)
        tabbar.pack(fill="x", padx=16)
        self._tab_seg = AppleSegmentedControl(
            tabbar, PLOT_TABS, "Friction", self._on_tab,
            width=660, height=34, parent_bg=BG)
        self._tab_seg.pack(side="left", pady=4)
        AppleButton(tabbar, "Copy IsaacLab", self._copy_isaac, bg_color=ELEVATED,
                    fg_color=LABEL2, font=FONT_SMALL, parent_bg=BG,
                    width=130, height=34).pack(side="right", pady=4)
        AppleButton(tabbar, "Copy MuJoCo", self._copy_mujoco, bg_color=ELEVATED,
                    fg_color=LABEL2, font=FONT_SMALL, parent_bg=BG,
                    width=120, height=34).pack(side="right", padx=6, pady=4)

        self._plot_host = tk.Frame(content, bg=PANEL_BG)
        self._plot_host.pack(fill="both", expand=True, padx=16, pady=8)
        self._show_figure(gp.placeholder())

        # log
        tk.Label(content, text="CALIBRATION LOG", bg=BG, fg=LABEL2,
                 font=F_CAPTION, anchor="w").pack(anchor="w", padx=20)
        self._log = scrolled_text(content, height=8)
        self._log.pack(fill="x", padx=16, pady=(2, 12))
        self._logln("Ready. Connect the DM-USB2FDCAN, pick the motor + ID, then Calibrate.")

    # ── plot helpers ──────────────────────────────────────────────────────────
    def _show_figure(self, fig):
        if self._plot_canvas is not None:
            self._plot_canvas.get_tk_widget().destroy()
        self._plot_canvas = FigureCanvasTkAgg(fig, master=self._plot_host)
        self._plot_canvas.draw()
        self._plot_canvas.get_tk_widget().pack(fill="both", expand=True)

    def _on_tab(self, tab):
        self._active_tab = tab
        if self._live_mode:
            self._enter_live()        # rebuild live artists for the new tab
        else:
            self._refresh_plot()

    # ── live plotting ─────────────────────────────────────────────────────────
    @staticmethod
    def _empty_buf():
        return {ph: {k: [] for k in ("t", "q", "w", "tau", "wcmd")}
                for ph in ("staircase", "chirp", "torque_speed")}

    def _enter_live(self):
        """(Re)mount the persistent live figure for the active tab."""
        self._show_figure(self._liveplot.build(self._active_tab))

    def _on_sample(self, a):
        phase, t, q, w, tau, wcmd = a
        b = self._live_buf.get(phase)
        if b is None:
            return
        b["t"].append(t); b["q"].append(q); b["w"].append(w)
        b["tau"].append(tau); b["wcmd"].append(wcmd)

    def _live_tick(self):
        if not self._live_mode:
            return
        st = self._live_buf["staircase"]
        ch = self._live_buf["chirp"]
        # live regression: friction from the staircase, J from the chirp
        self._live_params = gp.live_friction_fit(st["w"], st["tau"], st["wcmd"])
        if self._live_params is not None and ch["t"]:
            j = gp.live_inertia(ch["t"], ch["w"], ch["tau"], self._live_params)
            if j is not None:
                self._live_params.J = j
        try:
            self._liveplot.update(self._live_buf, self._live_params)
            if self._plot_canvas is not None:
                self._plot_canvas.draw_idle()
        except Exception:
            pass
        if self._live_params is not None:
            p = self._live_params
            self._results_lbl.config(
                text=(f"live fit:  Fc± {p.Fc_pos:.3f}/{p.Fc_neg:.3f} N·m   "
                      f"Fv {p.Fv:.4f}   J {p.J:.5f} kg·m²   (refining…)"),
                fg=ORANGE)
        self.after(120, self._live_tick)

    def _refresh_plot(self):
        motor = self.ctl.name
        logs = session.RESULTS_DIR / motor / "logs"
        params = self._last_res.params if self._last_res else None
        try:
            if self._active_tab == "Friction":
                f = logs / "staircase.csv"
                if not f.exists():
                    return self._show_figure(gp.placeholder("No staircase data yet"))
                log = CalibrationLog.load(f)
                w, tau = log.steady_points()
                self._show_figure(gp.friction_curve(w, tau, params,
                                  title=f"{motor} friction curve"))
            elif self._active_tab == "Torque fit":
                f = logs / "chirp.csv"
                if not (f.exists() and params is not None):
                    return self._show_figure(gp.placeholder("No chirp / fit yet"))
                self._show_figure(gp.torque_fit(CalibrationLog.load(f), params,
                                  title=f"{motor} dynamic torque fit"))
            elif self._active_tab == "Staircase":
                f = logs / "staircase.csv"
                if not f.exists():
                    return self._show_figure(gp.placeholder("No staircase data yet"))
                self._show_figure(gp.staircase(CalibrationLog.load(f),
                                  title=f"{motor} velocity staircase"))
            elif self._active_tab == "Chirp":
                f = logs / "chirp.csv"
                if not f.exists():
                    return self._show_figure(gp.placeholder("No chirp data yet"))
                self._show_figure(gp.chirp(CalibrationLog.load(f),
                                  title=f"{motor} frequency chirp"))
            elif self._active_tab == "Torque–speed":
                f = logs / "torque_speed.csv"
                if not f.exists():
                    return self._show_figure(gp.placeholder("No torque–speed sweep yet"))
                dc = None
                if self._last_report and self._last_report.get("dc_motor"):
                    from mabel_actid.friction_model import DCMotorParams
                    dc = DCMotorParams(**self._last_report["dc_motor"])
                self._show_figure(gp.torque_speed(CalibrationLog.load(f), dc,
                                  title=f"{motor} torque–speed envelope"))
            elif self._active_tab == "Performance":
                # prefer the torque-speed sweep (widest torque range); else staircase
                fts = logs / "torque_speed.csv"; fst = logs / "staircase.csv"
                src = fts if fts.exists() else (fst if fst.exists() else None)
                if src is None:
                    return self._show_figure(gp.placeholder(
                        "No performance data yet (run a torque–speed sweep)"))
                self._show_figure(gp.performance(CalibrationLog.load(src),
                                  title=f"{motor} performance curve"))
        except Exception as exc:
            self._show_figure(gp.placeholder(f"plot error: {exc}"))

    def _try_load_existing(self, name):
        """If a prior run for this NAME exists, show its params + plots."""
        self._last_report = None
        import json
        base = session.RESULTS_DIR / name
        pj = base / "calibration.json"
        if not pj.exists():
            pj = base / "params.json"          # back-compat
        if pj.exists():
            try:
                rep = json.loads(pj.read_text())
                self._last_report = rep
                self._results_lbl.config(
                    text=self._summary_from_report(rep), fg=LABEL)
                self._logln(f"Loaded existing calibration '{name}'.")
            except Exception:
                pass
        self._refresh_plot()

    # ── connection actions ──────────────────────────────────────────────────
    def _refresh_ports(self):
        ports = [p.device for p in list_ports()] or ["(none)"]
        self._port_dd.update_options(ports, ports[0])
        self._logln(f"Ports: {', '.join(ports)}")

    def _on_motor(self, motor):
        if self.ctl.running:
            return
        self.ctl.motor = motor
        self._render_profile(motor)
        # if the NAME field is still just a (default) type name, follow the type
        if self._name_var.get().strip() in MOTOR_NAMES + [""]:
            self._name_var.set(motor)
        self._last_res = None
        self.ctl.name = self._name_var.get().strip() or motor
        self._results_lbl.config(text="No calibration yet.", fg=LABEL2)
        self._try_load_existing(self.ctl.name)

    def _on_name_change(self, *_):
        """User typed an instance name — key results/plots off it and reload."""
        if self.ctl.running:
            return
        name = self._name_var.get().strip() or self.ctl.motor
        if name == self.ctl.name:
            return
        self.ctl.name = name
        self._last_res = None
        self._try_load_existing(name)

    def _toggle_connect(self):
        if self.ctl.connected:
            self.ctl.disconnect()
            self._status_dot.config(fg=RED)
            self._connect_btn.update_colors(ORANGE)
            self._connect_btn.update_text("Connect")
            self._status_lbl.config(text="Not connected", fg=LABEL2, bg=ELEVATED)
            self._logln("Disconnected.")
            return
        backend = self._backend_seg.get()
        port = self._port_dd.get()
        motor = self._motor_dd.get()
        can_id = self._id_dd.get()
        where = {"serial": port, "gs_usb": "USB (CANable/gs_usb)",
                 "socketcan": "can0", "sim": "—"}.get(backend, "")
        self._logln(f"Connecting [{backend}] {where} · {motor} id {can_id} …")
        try:
            found = self.ctl.connect(backend, port, motor, can_id)
        except Exception as exc:
            self._status_dot.config(fg=RED)
            self._status_lbl.config(text="Connect failed", fg=RED, bg=ELEVATED)
            self._logln(f"Connect failed: {exc}")
            return
        self._render_profile(motor)
        self._status_dot.config(fg=GREEN)
        self._connect_btn.update_colors(BLUE)
        self._connect_btn.update_text("Disconnect")
        if found:
            self._logln(f"Connected. Motor id {can_id} answered ✓  telemetry live.")
        elif backend != "sim":
            self._logln(f"Connected to adapter, but motor id {can_id} did NOT answer "
                        f"the 0xCC scan — check 24V power, CAN_H/L wiring, and the ID.")
        else:
            self._logln("Connected (sim). Telemetry is live.")

    # ── calibration actions ─────────────────────────────────────────────────
    def _on_vel(self, val):
        self._vel_lbl.config(text=f"{val/100:.2f}")

    def _on_calibrate(self):
        if self.ctl.running:
            self.ctl.abort()
            self._logln("Abort requested…")
            return
        if not self.ctl.connected:
            self._logln("Connect to a motor first.")
            return
        cfg = session.CollectConfig(
            vel_frac=self._vel_slider.get() / 100.0,
            chirp=(self._chirp_seg.get() == "on"),
            torque_speed=(self._ts_seg.get() == "on"))
        name = self._name_var.get().strip() or self.ctl.motor
        self.ctl.name = name
        self._last_res = None
        self._progress.set(0)
        self._calib_btn.update_colors(RED)
        self._calib_btn.update_text("Abort")
        self._logln(f"Calibrating '{name}' (type {self.ctl.motor}) → results/{name}/")
        # start live plotting: clear buffers, mount the live figure, begin redraws
        self._live_buf = self._empty_buf()
        self._live_params = None
        self._live_mode = True
        self._enter_live()
        self.after(120, self._live_tick)
        self.ctl.start_calibration(cfg, model="stribeck", name=name)

    def _copy_mujoco(self):
        if not self._last_res:
            self._logln("Nothing to copy — run a calibration first.")
            return
        attrs = self._last_res.params.mujoco_attrs()
        txt = " ".join(f'{k}="{v}"' for k, v in attrs.items())
        self.clipboard_clear()
        self.clipboard_append(txt)
        self._logln(f"Copied MuJoCo attrs to clipboard: {txt}")

    def _copy_isaac(self):
        if not (self._last_report and self._last_report.get("isaaclab")):
            self._logln("Nothing to copy — run a calibration first.")
            return
        isaac = self._last_report["isaaclab"]
        def g(k):
            return isaac.get(k, {}).get("value")
        lines = ["DCMotorCfg(",
                 "    joint_names_expr=[\".*\"],"]
        for k in ("effort_limit", "saturation_effort", "velocity_limit",
                  "stiffness", "damping", "armature", "friction"):
            v = g(k)
            src = isaac.get(k, {}).get("source", "")
            val = "{ \".*\": ... }" if (k in ("stiffness", "damping") and v is None) \
                else (v if v is not None else "None  # FILL FROM DATASHEET")
            lines.append(f"    {k}={val},  # {src}")
        lines.append(")")
        txt = "\n".join(lines)
        self.clipboard_clear()
        self.clipboard_append(txt)
        self._logln("Copied Isaac Lab DCMotorCfg to clipboard (see comments for "
                    "which params are measured vs datasheet/policy).")

    # ── periodic UI pumps ─────────────────────────────────────────────────────
    def _poll_telemetry(self):
        fb = self.ctl.feedback()
        if fb is not None and self.ctl.connected:
            self._readout["pos"].config(text=f"{fb.pos:+.3f}")
            self._readout["vel"].config(text=f"{fb.vel:+.3f}")
            self._readout["tau"].config(text=f"{fb.tau:+.3f}")
            self._readout["vel"].config(fg=GREEN if abs(fb.vel) > 0.02 else LABEL)
            self._temp_mos.config(text=f"MOS {fb.t_mos:.0f}°",
                                  fg=RED if fb.t_mos > 70 else LABEL2)
            self._temp_rot.config(text=f"ROTOR {fb.t_rotor:.0f}°",
                                  fg=RED if fb.t_rotor > 80 else LABEL2)
            self._set_status(fb.err)
        self.after(60, self._poll_telemetry)

    def _set_status(self, err):
        if err == 1:
            txt, fg = "ENABLED", GREEN
        elif err == 0:
            txt, fg = "Idle (disabled)", LABEL
        elif err >= 8:
            txt, fg = f"FAULT 0x{err:X}", RED
        else:
            txt, fg = f"state {err}", YELLOW
        self._status_lbl.config(text=txt, fg=fg, bg=ELEVATED)

    def _poll_events(self):
        try:
            while True:
                kind, payload = self.ctl.events.get_nowait()
                if kind == "log":
                    self._logln(payload)
                elif kind == "progress":
                    self._progress.set(payload * 100)
                elif kind == "phase":
                    self._phase_lbl.config(text=f"phase: {payload}")
                elif kind == "sample":
                    self._on_sample(payload)
                elif kind == "done":
                    self._on_done(payload)
                elif kind == "error":
                    self._on_run_end()
        except queue.Empty:
            pass
        self.after(80, self._poll_events)

    def _on_done(self, res):
        self._last_res = res
        self._progress.set(100)
        self._phase_lbl.config(text="phase: done ✓")
        # the full report (with the isaaclab block) is in params.json on disk
        rep = res.report
        import json
        pj = session.RESULTS_DIR / self.ctl.name / "calibration.json"
        try:
            rep = json.loads(pj.read_text())
        except Exception:
            pass
        self._last_report = rep
        self._results_lbl.config(text=self._summary_from_report(rep), fg=LABEL)
        self._on_run_end()        # exits live mode and renders the final full-res plot

    def _on_run_end(self):
        self._live_mode = False        # stop live redraws; show final full-res plot
        self._calib_btn.update_colors(ORANGE)
        self._calib_btn.update_text("Re-calibrate")
        self._refresh_plot()

    @staticmethod
    def _summary_from_report(rep: dict) -> str:
        p = rep.get("params", {})
        mj = rep.get("mujoco", {})
        r2 = rep.get("friction_r2",
                     rep.get("fit_quality", {}).get("friction_r2", float("nan")))
        line1 = (f"Fc± {p.get('Fc_pos',0):.3f}/{p.get('Fc_neg',0):.3f} N·m   "
                 f"Fv {p.get('Fv',0):.4f} N·m·s/rad   "
                 f"J {p.get('J',0):.5f} kg·m²   R² {r2:.4f}")
        line2 = "MuJoCo:  " + "  ".join(f'{k}="{v}"' for k, v in mj.items())
        lines = [line1, line2]
        isaac = rep.get("isaaclab")
        if isaac:
            def tag(k):
                v = isaac.get(k, {}).get("value")
                meas = "MEASURED" in isaac.get(k, {}).get("source", "")
                star = "" if meas else "†"
                return f"{k}={v}{star}"
            line3 = "IsaacLab:  " + "  ".join(
                tag(k) for k in ("velocity_limit", "saturation_effort",
                                 "effort_limit", "armature", "friction"))
            lines.append(line3)
            lines.append("† = datasheet/policy, not bench-measured (see Copy IsaacLab)")
        return "\n".join(lines)

    # ── log ───────────────────────────────────────────────────────────────────
    def _logln(self, msg):
        ts = time.strftime("%H:%M:%S")
        self._log.config(state="normal")
        self._log.insert("end", f"[{ts}] {msg}\n")
        self._log.see("end")
        self._log.config(state="disabled")


def main():
    ap = argparse.ArgumentParser(description="MABEL Motor Calibration GUI")
    ap.add_argument("--backend", choices=["serial", "gs_usb", "socketcan", "sim"],
                    default=None)
    ap.add_argument("--port", default=None)
    ap.add_argument("--motor", default=DEFAULT_MOTOR, choices=MOTOR_NAMES)
    ap.add_argument("--id", type=int, default=1)
    args = ap.parse_args()
    App(backend=args.backend or session.default_backend(),
        port=args.port, motor=args.motor, can_id=args.id).mainloop()


if __name__ == "__main__":
    main()
