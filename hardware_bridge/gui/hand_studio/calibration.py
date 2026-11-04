"""Calibration Panel — Auto full sweep, per-finger, or manual capture."""

import sys
import threading
import time
import tkinter as tk
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from style import *
from orcahand.constants import wrapped_delta, apply_offset, remove_offset, WRAP

CALIBRATION_SEQUENCE = [
    {"step": 1,  "label": "Thumb CMC flex",    "joints": {"thumb_cmc": +1}},
    {"step": 2,  "label": "Thumb CMC extend",  "joints": {"thumb_cmc": -1}},
    {"step": 3,  "label": "Thumb ABD flex",    "joints": {"thumb_abd": +1}},
    {"step": 4,  "label": "Thumb ABD extend",  "joints": {"thumb_abd": -1}},
    {"step": 5,  "label": "Thumb MP flex",     "joints": {"thumb_mp":  +1}},
    {"step": 6,  "label": "Thumb MP extend",   "joints": {"thumb_mp":  -1}},
    {"step": 7,  "label": "Thumb IP flex",     "joints": {"thumb_ip":  +1}},
    {"step": 8,  "label": "Thumb IP extend",   "joints": {"thumb_ip":  -1}},
    {"step": 9,  "label": "All ABD flex",
     "joints": {"index_abd":+1,"middle_abd":+1,"ring_abd":+1,"pinky_abd":+1}},
    {"step": 10, "label": "All ABD extend",
     "joints": {"index_abd":-1,"middle_abd":-1,"ring_abd":-1,"pinky_abd":-1}},
    {"step": 11, "label": "All MCP flex",
     "joints": {"index_mcp":+1,"middle_mcp":+1,"ring_mcp":+1,"pinky_mcp":+1}},
    {"step": 12, "label": "All MCP extend",
     "joints": {"index_mcp":-1,"middle_mcp":-1,"ring_mcp":-1,"pinky_mcp":-1}},
    {"step": 13, "label": "All PIP flex",
     "joints": {"index_pip":+1,"middle_pip":+1,"ring_pip":+1,"pinky_pip":+1}},
    {"step": 14, "label": "All PIP extend",
     "joints": {"index_pip":-1,"middle_pip":-1,"ring_pip":-1,"pinky_pip":-1}},
    {"step": 15, "label": "Wrist flex",        "joints": {"wrist": +1}},
    {"step": 16, "label": "Wrist extend",      "joints": {"wrist": -1}},
]

STEP_RAW    = 5
STEP_DELAY  = 0.02
NUM_STABLE  = 10
STALL_ATOL  = 7
BACKOFF_RAW = 20
CALIB_SPEED = 500
CALIB_TORQUE= 400
# A stall this close to 0 or 4095 is the single-turn encoder running out of
# range (the 0/4095 wrap), NOT a mechanical end-stop. The joint's travel crosses
# the encoder boundary and must be re-centered (Setup → Joint Limits → ↺) so the
# wrap sits a half-turn away before it can be calibrated.
ENC_BOUNDARY = 20

# ── Tensioning ────────────────────────────────────────────────────────────────
# Drive each motor into its hard-stop to take up tendon slack, so the operator
# can secure the slack side of the tendon. ALL finger motors wind the same way.
#   raw-tick direction: +1 increases the encoder count, -1 decreases it.
#   TENSION_SIGN = -1 -> CCW.  Flip to +1 for CW if it tensions the wrong way.
TENSION_IDS  = list(range(1, 17))           # all finger motors; wrist (17) excluded
TENSION_SIGN = -1                           # -1 = CCW
TENSION_DIR  = {m: TENSION_SIGN for m in TENSION_IDS}


class CalibrationPanel(tk.Frame):

    SUB_PANELS = ["Tension", "Auto Calibration", "Select Finger", "Manual"]

    def __init__(self, parent, bus):
        super().__init__(parent, bg=BG)
        self.bus      = bus
        self._thread: threading.Thread | None = None
        self._stop    = threading.Event()
        self._content: tk.Frame | None = None
        self._pending: dict = {}   # {mid: {"min": None, "max": None}}
        self._step_labels: dict[int, tk.Label] = {}   # only the Auto tab fills it;
        #   default {} keeps _mark_step safe when a sweep runs from another tab.
        self._cont: dict[int, dict] = {}   # manual-tab continuous (unwrapped) tracking
        self._saveable: set = set()        # manual-tab joints swept enough to save

        nav = tk.Frame(self, bg=BG, pady=PAD_SM)
        nav.pack(fill="x", padx=PAD)
        AppleSegmentedControl(nav, self.SUB_PANELS, self.SUB_PANELS[0], self._show, parent_bg=BG).pack(side="left")
        # global torque toggle: ON = motors hold position; OFF = joints move freely
        self._torque_btn = accent_btn(nav, "Torque: OFF", self._toggle_torque,
                                      color=ELEVATED)
        self._torque_btn.pack(side="right")
        self._show(self.SUB_PANELS[0])
        self.after(400, self._refresh_torque_btn)

    def _refresh_torque_btn(self):
        if not self.winfo_exists():
            return
        on = bool(getattr(self.bus.state, "torque_on", False))
        try:
            self._torque_btn.update_text("Torque: ON" if on else "Torque: OFF")
            self._torque_btn.update_colors(GREEN if on else ELEVATED, BG if on else LABEL)
        except Exception:
            pass
        self.after(500, self._refresh_torque_btn)

    def _toggle_torque(self):
        on = not bool(getattr(self.bus.state, "torque_on", False))
        try:
            self.bus.set_torque(on)
        except Exception:
            pass
        self._refresh_torque_btn()

    def _show(self, name):
        # leaving the Manual tab: stop its live limit-tracking subscription
        if getattr(self, "_manual_subscribed", False):
            try:
                self.bus.unsubscribe(self._on_manual_state)
            except Exception:
                pass
            self._manual_subscribed = False
        if self._content:
            self._content.destroy()
        self._content = tk.Frame(self, bg=BG)
        self._content.pack(fill="both", expand=True)
        {
            "Tension":          self._build_tension,
            "Auto Calibration": self._build_auto,
            "Select Finger":    self._build_finger,
            "Manual":           self._build_manual,
        }[name]()

    # ── Tensioning ────────────────────────────────────────────────────────────

    def _build_tension(self):
        f = tk.Frame(self._content, bg=BG)
        f.pack(fill="both", expand=True, padx=PAD_LG, pady=PAD)

        lbl(f, "Tendon Tensioning", fg=LABEL, font=F_TITLE2).pack(anchor="w")
        lbl(f, "Drives every motor into its hard-stop to take up tendon slack, so\n"
               "you can secure the slack side of each tendon. Wrist (ID 17) is skipped.",
            fg=LABEL2, font=F_CALLOUT).pack(anchor="w", pady=(2, PAD))

        arrow = "↺" if TENSION_SIGN < 0 else "↻"
        spin  = "Counter-clockwise" if TENSION_SIGN < 0 else "Clockwise"
        cardf = card(f)
        cardf.pack(fill="x", pady=PAD_SM)
        lbl(cardf, f"{arrow}  All motors → {spin}", fg=ORANGE,
            font=F_HEADLINE, bg=SURFACE).pack(anchor="w")
        lbl(cardf, "IDs " + "  ".join(str(m) for m in TENSION_IDS),
            fg=LABEL2, font=F_MONO_SM, bg=SURFACE).pack(anchor="w", pady=(4, 0))

        btn_row = tk.Frame(f, bg=BG)
        btn_row.pack(fill="x", pady=PAD)
        self._tension_btn = accent_btn(btn_row, "▶  Tension Hand",
                                       self._run_tension, color=ORANGE)
        self._tension_btn.pack(side="left", padx=12, pady=6)
        ghost_btn(btn_row, "■  Stop", self._stop_calib, fg=RED).pack(side="left", padx=12, pady=6)

        self._tension_status = lbl(f, "Ready. Press 'Tension Hand'.",
                                   fg=LABEL2, font=F_CALLOUT)
        self._tension_status.pack(anchor="w")

    def _run_tension(self):
        if self._thread and self._thread.is_alive():
            return
        if not self.bus.state.connected:
            self._tension_status.config(text="Connect a hand first.", fg=YELLOW)
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._tension_bg, daemon=True)
        self._thread.start()

    def _set_tension_status(self, text, fg):
        w = getattr(self, "_tension_status", None)
        try:
            if w is not None and w.winfo_exists():
                w.config(text=text, fg=fg)
        except Exception:
            pass

    def _tension_bg(self):
        from collections import deque
        self.after(0, self._set_tension_status, "Tensioning…", ORANGE)

        ids = [m for m in TENSION_DIR if m in self.bus.state.motors]
        buffers = {m: deque(maxlen=NUM_STABLE) for m in ids}
        targets, stalled = {}, set()

        # Torque on and seed each target from the present position.
        for m in ids:
            ms = self.bus.state.motors.get(m)
            targets[m] = ms.raw if (ms and ms.raw is not None) else 2047
            self.bus.write_register(m, 40, 1, 1)   # torque enable
        time.sleep(0.3)

        while len(stalled) < len(ids) and not self._stop.is_set():
            active = [m for m in ids if m not in stalled]
            write_t = {}
            for m in active:
                targets[m] = max(0, min(4095, targets[m] + TENSION_DIR[m] * STEP_RAW))
                write_t[m] = targets[m]
            self.bus.sync_write(write_t, speed=CALIB_SPEED, torque=CALIB_TORQUE)
            time.sleep(STEP_DELAY)

            for m in active:
                ms = self.bus.state.motors.get(m)
                if ms and ms.raw is not None:
                    buffers[m].append(ms.raw)
                    if (len(buffers[m]) == NUM_STABLE and
                            max(buffers[m]) - min(buffers[m]) <= STALL_ATOL):
                        # Stall found: stop driving this motor and HOLD it right
                        # here under torque so the operator can tension against it.
                        stalled.add(m)
                        targets[m] = ms.raw
                        self.bus.sync_write({m: ms.raw}, speed=CALIB_SPEED,
                                            torque=CALIB_TORQUE)
                        jname = self.bus.id_joint.get(m, str(m))
                        self.bus.log(f"  ID {m} ({jname}) stalled — holding at {ms.raw}")
                        self.after(0, self._set_tension_status,
                                   f"Tensioning… {len(stalled)}/{len(ids)} holding at stall",
                                   ORANGE)

        # Re-assert every hold and keep torque ON so the motors stay pinned at
        # their stall while the operator tensions and secures the tendons.
        hold = {m: targets[m] for m in stalled}
        if hold:
            self.bus.sync_write(hold, speed=CALIB_SPEED, torque=CALIB_TORQUE)
        self.bus.state.torque_on = True
        if self._stop.is_set():
            self.after(0, self._set_tension_status, "Stopped.", YELLOW)
        elif not ids:
            self.after(0, self._set_tension_status, "No motors responding.", YELLOW)
        else:
            self.after(0, self._set_tension_status,
                       f"✓ Tensioned {len(stalled)}/{len(ids)} motors — holding at "
                       f"stall. Secure tendons, then disable torque.", GREEN)
            self.bus.log(f"Tensioning done: {len(stalled)}/{len(ids)} motors at stall")

    # ── Auto full calibration ─────────────────────────────────────────────────

    def _build_auto(self):
        f = tk.Frame(self._content, bg=BG)
        f.pack(fill="both", expand=True, padx=PAD_LG, pady=PAD)

        lbl(f, "Auto Full Calibration", fg=LABEL, font=F_TITLE2).pack(anchor="w")
        lbl(f, "Sweeps every motor to both end-stops using stall detection.\n"
               "All joints in each group move simultaneously.",
            fg=LABEL2, font=F_CALLOUT).pack(anchor="w", pady=(2, PAD))

        # Step progress list
        steps_outer = tk.Frame(f, bg=SURFACE, padx=PAD_SM, pady=PAD_SM)
        steps_outer.pack(fill="x", pady=PAD_SM)

        self._step_labels: dict[int, tk.Label] = {}
        cols = 2
        for i, step in enumerate(CALIBRATION_SEQUENCE):
            r, c = divmod(i, cols)
            row = tk.Frame(steps_outer, bg=SURFACE)
            row.grid(row=r, column=c, sticky="w", padx=PAD_SM, pady=1)
            dot = tk.Label(row, text="○", bg=SURFACE, fg=LABEL3,
                           font=F_MONO_SM, width=2)
            dot.pack(side="left")
            tk.Label(row, text=f"Step {step['step']:>2}  {step['label']}",
                     bg=SURFACE, fg=LABEL2, font=F_CAPTION, anchor="w",
                     width=24).pack(side="left")
            self._step_labels[step["step"]] = dot

        btn_row = tk.Frame(f, bg=BG)
        btn_row.pack(fill="x", pady=PAD)
        self._run_btn = accent_btn(btn_row, "▶  Run Full Calibration",
                                    self._run_auto, color=ORANGE)
        self._run_btn.pack(side="left", padx=12, pady=6)
        ghost_btn(btn_row, "■  Stop", self._stop_calib, fg=RED).pack(side="left", padx=12, pady=6)

        self._calib_status = lbl(f, "Ready.", fg=LABEL2, font=F_CALLOUT)
        self._calib_status.pack(anchor="w")

    def _run_auto(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        joints = None   # all joints
        self._thread = threading.Thread(
            target=self._calibrate_bg, args=(joints,),
            kwargs={"status_cb": self._set_calib_status}, daemon=True
        )
        self._thread.start()

    # ── Select finger calibration ─────────────────────────────────────────────

    def _build_finger(self):
        f = tk.Frame(self._content, bg=BG)
        f.pack(fill="both", expand=True, padx=PAD_LG, pady=PAD)

        lbl(f, "Calibrate Specific Joints", fg=LABEL, font=F_TITLE2).pack(anchor="w")
        lbl(f, "Select joints to calibrate, then run.", fg=LABEL2,
            font=F_CALLOUT).pack(anchor="w", pady=(2, PAD))

        grid = tk.Frame(f, bg=BG)
        grid.pack(anchor="w")

        self._joint_checks: dict[str, tk.BooleanVar] = {}
        all_joints = list(self.bus.joint_id.keys()) if self.bus.joint_id else []
        for i, jname in enumerate(all_joints):
            r, c = divmod(i, 4)
            var = tk.BooleanVar(value=False)
            cb  = tk.Checkbutton(
                grid, text=jname, variable=var,
                bg=BG, fg=LABEL, font=F_CALLOUT,
                selectcolor=ELEVATED, activebackground=BG,
                activeforeground=ORANGE,
            )
            cb.grid(row=r, column=c, sticky="w", padx=PAD_SM, pady=2)
            self._joint_checks[jname] = var

        btn_row = tk.Frame(f, bg=BG)
        btn_row.pack(fill="x", pady=PAD)
        accent_btn(btn_row, "▶  Calibrate Selected", self._run_selected,
                   color=ORANGE).pack(side="left", padx=12, pady=6)
        ghost_btn(btn_row, "Select All",   self._select_all).pack(side="left", padx=12, pady=6)
        ghost_btn(btn_row, "Clear",        self._clear_all).pack(side="left", padx=12, pady=6)

        self._finger_status = lbl(f, "Select joints and press Run.", fg=LABEL2, font=F_CALLOUT)
        self._finger_status.pack(anchor="w")

    def _select_all(self):
        for v in self._joint_checks.values():
            v.set(True)

    def _clear_all(self):
        for v in self._joint_checks.values():
            v.set(False)

    def _run_selected(self):
        selected = [j for j, v in self._joint_checks.items() if v.get()]
        if not selected:
            self._finger_status.config(text="No joints selected.", fg=YELLOW)
            return
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._calibrate_bg, args=(selected,),
            kwargs={"status_cb": self._set_finger_status}, daemon=True
        )
        self._thread.start()

    # ── Manual calibration ────────────────────────────────────────────────────

    MANUAL_BAR_W = 300

    def _motor_ids(self):
        return sorted(self.bus.id_joint.keys()) if self.bus.id_joint else list(range(1, 18))

    def _build_manual(self):
        f = tk.Frame(self._content, bg=BG)
        f.pack(fill="both", expand=True, padx=PAD_LG, pady=PAD)

        lbl(f, "Manual Calibration", fg=LABEL, font=F_TITLE2).pack(anchor="w")
        lbl(f, "Torque OFF — sweep every joint by hand all at once; each bar fills as it finds its\n"
               "range (green ≥80 %). Click any joint for its detail: recalibrate · flip · visualise · data.",
            fg=LABEL2, font=F_CALLOUT).pack(anchor="w", pady=(2, PAD_SM))

        btn_row = tk.Frame(f, bg=BG)
        btn_row.pack(fill="x", pady=(0, 2))
        self._save_btn = accent_btn(btn_row, "Save Calibration", self._manual_save, color=ELEVATED)
        self._save_btn.pack(side="left", padx=(0, 10), pady=4)
        accent_btn(btn_row, "↺  Reset", self._manual_reset,
                   color=BLUE).pack(side="left", padx=(0, 10), pady=4)
        accent_btn(btn_row, "⌖  Center / Set Home", self._manual_center_all,
                   color=ELEVATED).pack(side="left", padx=(0, 10), pady=4)
        self._save_dirty_last = None

        self._manual_status = lbl(f, "", fg=LABEL2, font=F_CALLOUT, anchor="w")
        self._manual_status.pack(fill="x", pady=(0, PAD_SM))

        # Theoretical reference = the saved calibration (corrected-frame tick range
        # + its software offset), which maps onto the joint's URDF angle range.
        self._theo = {mid: dict(self.bus.calib.get(mid, {"min": 0, "max": 4095,
                                                         "center": 2047, "offset": 0}))
                      for mid in self._motor_ids()}
        self._manual_data: dict[int, dict] = {}   # live per-joint computed data
        self._manual_rows = {}
        self._detail_mid = None
        # body swaps between the all-joints overview and a single-joint detail
        self._manual_body = tk.Frame(f, bg=BG)
        self._manual_body.pack(fill="both", expand=True)

        self._manual_reset()
        self._manual_show_overview()
        self.bus.subscribe(self._on_manual_state)
        self._manual_subscribed = True

    # ── Manual: all-joints overview (sweep all at once, live position bars) ────
    def _manual_show_overview(self):
        self._detail_mid = None
        for w in self._manual_body.winfo_children():
            w.destroy()
        leg = tk.Frame(self._manual_body, bg=BG)
        leg.pack(fill="x", pady=(0, PAD_SM))
        for txt, col in [("● ≥80 %", GREEN), ("● partial", YELLOW), ("● uncalibrated", RED),
                         ("click a joint for detail ›", LABEL2)]:
            lbl(leg, txt, fg=col, font=F_CAPTION, bg=BG).pack(side="left", padx=(0, 14))
        wrap = tk.Frame(self._manual_body, bg=PANEL_BG)
        wrap.pack(fill="both", expand=True)
        col_l = tk.Frame(wrap, bg=PANEL_BG); col_r = tk.Frame(wrap, bg=PANEL_BG)
        col_l.pack(side="left", fill="both", expand=True, padx=(0, 8))
        col_r.pack(side="left", fill="both", expand=True, padx=(8, 0))
        self._manual_rows = {}
        ids = self._motor_ids(); half = (len(ids) + 1) // 2
        for i, mid in enumerate(ids):
            parent = col_l if i < half else col_r
            jname = self.bus.id_joint.get(mid, f"ID {mid}")
            row = tk.Frame(parent, bg=PANEL_BG); row.pack(fill="x", padx=8, pady=3)
            top = tk.Frame(row, bg=PANEL_BG); top.pack(fill="x")
            dot = tk.Label(top, text="●", fg=LABEL3, bg=PANEL_BG, font=("Helvetica", 10))
            dot.pack(side="left", padx=(0, 5))
            name = tk.Label(top, text=jname, fg=LABEL, bg=PANEL_BG, font=FONT_SMALL, anchor="w")
            name.pack(side="left")
            chev = tk.Label(top, text="›", fg=LABEL3, bg=PANEL_BG, font=FONT_SMALL)
            chev.pack(side="right")
            info = tk.Label(top, text="—", fg=LABEL2, bg=PANEL_BG, font=FONT_MONO_S, anchor="e")
            info.pack(side="right", padx=(8, 0))
            bar = RangeBar(row, width=self.MANUAL_BAR_W, height=12, bg=PANEL_BG)
            bar.pack(fill="x", pady=(2, 0))
            self._manual_rows[mid] = {"dot": dot, "info": info, "bar": bar}
            for w in (row, top, dot, name, chev, info, bar):
                w.bind("<Button-1>", lambda e, m=mid: self._open_joint_detail(m))
                try:
                    w.config(cursor="hand2")
                except Exception:
                    pass
        self._push_data_to_overview()

    # ── Manual: single-joint detail (recalibrate · flip · visualise · data) ────
    def _open_joint_detail(self, mid):
        self._detail_mid = mid
        self._manual_rows = {}
        for w in self._manual_body.winfo_children():
            w.destroy()
        jname = self.bus.id_joint.get(mid, f"ID {mid}")
        head = tk.Frame(self._manual_body, bg=BG); head.pack(fill="x", pady=(0, 8))
        accent_btn(head, "‹ All joints", self._manual_show_overview,
                   color=ELEVATED, font=F_CALLOUT, px=10, py=4).pack(side="left")
        lbl(head, jname, fg=LABEL, font=F_TITLE2, bg=BG).pack(side="left", padx=(14, 0))
        self._detail_flag = tk.Label(head, text="", bg=BG, font=FONT_SMALL)
        self._detail_flag.pack(side="right")

        card = tk.Frame(self._manual_body, bg=SURFACE); card.pack(fill="x", pady=(0, 8))
        lbl(card, "Live position · swept range", fg=LABEL2, font=F_CAPTION,
            bg=SURFACE).pack(anchor="w", padx=12, pady=(9, 3))
        self._detail_bar = RangeBar(card, width=self.MANUAL_BAR_W * 2, height=24, bg=SURFACE)
        self._detail_bar.pack(fill="x", padx=12, pady=(0, 11))

        grid = tk.Frame(self._manual_body, bg=PANEL_BG); grid.pack(fill="x", pady=(0, 8))
        self._detail_vals = {}
        for k, key in [("Motor id", "id"), ("Raw min–max", "raw"),
                       ("Center · offset", "center"), ("Angle range", "angle"),
                       ("Swept range", "swept"), ("Reliability", "cov"), ("Direction", "dir")]:
            r = tk.Frame(grid, bg=PANEL_BG); r.pack(fill="x", padx=12, pady=2)
            tk.Label(r, text=k, fg=LABEL2, bg=PANEL_BG, font=FONT_SMALL, width=15,
                     anchor="w").pack(side="left")
            v = tk.Label(r, text="—", fg=LABEL, bg=PANEL_BG, font=FONT_MONO_S, anchor="w")
            v.pack(side="left")
            self._detail_vals[key] = v

        btns = tk.Frame(self._manual_body, bg=BG); btns.pack(fill="x", pady=(6, 0))
        accent_btn(btns, "↻ Recalibrate", lambda m=mid: self._recalibrate_one(m),
                   color=ELEVATED).pack(side="left", padx=(0, 8))
        accent_btn(btns, "⇄ Flip direction", lambda m=mid: self._detail_flip(m),
                   color=BLUE).pack(side="left", padx=(0, 8))
        accent_btn(btns, "▶ Visualise", lambda m=mid: self._dir_visualize(m),
                   color=GREEN).pack(side="left", padx=(0, 8))
        accent_btn(btns, "↺ Re-center", lambda m=mid: self._manual_recenter(m),
                   color=ORANGE).pack(side="left", padx=(0, 8))
        self._detail_static(mid)
        self._push_data_to_detail()

    def _detail_static(self, mid):
        """Fill the detail fields that don't change frame-to-frame."""
        if self._detail_mid != mid or not hasattr(self, "_detail_vals"):
            return
        t = self._theo.get(mid, {})
        v = self._detail_vals
        try:
            j = self.bus._oh.config.by_id().get(mid) if getattr(self.bus, "_oh", None) else None
        except Exception:
            j = None
        inv = self.bus.is_inverted(mid)
        try:
            v["id"].config(text=str(mid))
            v["raw"].config(text=f"{int(t.get('min', 0))} – {int(t.get('max', 4095))}")
            v["center"].config(text=f"{int(t.get('center', 2047))}  ·  off {int(t.get('offset', 0))}")
            v["angle"].config(text=(f"{j.angle_min:+.2f} … {j.angle_max:+.2f} rad" if j else "—"))
            v["dir"].config(text="flipped (inverted)" if inv else "normal",
                            fg=ORANGE if inv else GREEN)
            self._detail_flag.config(text="↓ flipped" if inv else "↑ normal",
                                     fg=ORANGE if inv else GREEN)
        except Exception:
            pass

    def _detail_flip(self, mid):
        self.bus.set_invert(mid, not self.bus.is_inverted(mid))
        self._detail_static(mid)
        self._dir_visualize(mid)

    def _dir_visualize(self, mid):
        def bg():
            import time
            self.bus.nudge(mid, 72)        # drive toward + (torque auto-on)
            time.sleep(0.6)
            self.bus.nudge(mid, 45)        # relax back so the + swing is clear
        threading.Thread(target=bg, daemon=True).start()

    def _set_manual_status(self, text, fg):
        w = getattr(self, "_manual_status", None)
        try:
            if w is not None and w.winfo_exists():
                w.config(text=text, fg=fg)
        except Exception:
            pass

    def _manual_reset(self):
        """COMPLETELY disable torque (free to move) and restart the session sweep
        at the current pose. The SAVED calibration on disk is left untouched."""
        if self._thread and self._thread.is_alive():
            self._set_manual_status("Wait for the running calibration to finish.", YELLOW)
            return
        self.bus.set_torque(False)                      # broadcast torque off
        try:                                            # belt-and-suspenders per-motor kill
            for mid in self._motor_ids():
                self.bus.write_register(mid, 40, 0, 1)
        except Exception:
            pass
        pos = self.bus.read_positions_direct()
        # Continuous (unwrapped) tracking: seed each joint's running position at the
        # present read so a sweep that crosses the 0/4095 seam integrates cleanly.
        self._cont = {mid: {"pos": float(pos.get(mid, self._theo[mid]["center"])),
                            "lo":  float(pos.get(mid, self._theo[mid]["center"])),
                            "hi":  float(pos.get(mid, self._theo[mid]["center"])),
                            "prev": int(pos.get(mid, self._theo[mid]["center"]))}
                      for mid in self._theo}
        self._saveable = set()
        self._update_save_btn(force=True)
        self._set_manual_status("Reset — min/max cleared. Sweep each joint to find its "
                                "new range. Saved calibration is kept until you Save.",
                                LABEL2)

    def _update_save_btn(self, force: bool = False):
        n = len(getattr(self, "_saveable", set()))
        if not force and n == self._save_dirty_last:
            return
        self._save_dirty_last = n
        try:
            if n:
                self._save_btn.update_colors(ORANGE, LABEL)
                self._save_btn.update_text("Save New Calibration")
            else:
                self._save_btn.update_colors(ELEVATED, LABEL2)
                self._save_btn.update_text("Save Calibration")
        except Exception:
            pass

    def _compute_joint(self, mid, state):
        """Integrate this joint's live swept range + coverage from telemetry."""
        ms = state.motors.get(mid)
        if not (ms and ms.responding and ms.raw is not None):
            return None
        cur = ms.raw
        if not (0 <= cur <= 4095):        # ignore corrupt (out-of-turn) reads
            return None
        c = self._cont.get(mid)
        if c is None:
            c = {"pos": float(cur), "lo": float(cur), "hi": float(cur), "prev": cur}
            self._cont[mid] = c
        else:
            c["pos"] += wrapped_delta(cur - c["prev"])   # shortest-step integrate (seam-safe)
            c["prev"] = cur
            c["lo"] = min(c["lo"], c["pos"]); c["hi"] = max(c["hi"], c["pos"])
        cur_c, olo, ohi = c["pos"], c["lo"], c["hi"]
        t = self._theo[mid]; off = int(t.get("offset", 0))
        ta = remove_offset(min(t["min"], t["max"]), off)
        tb = remove_offset(max(t["min"], t["max"]), off)
        tlo, thi = sorted((cur_c + wrapped_delta(ta - cur_c), cur_c + wrapped_delta(tb - cur_c)))
        tspan = max(1.0, thi - tlo)
        coverage = (ohi - olo) / tspan
        if coverage >= 0.80:
            color = GREEN; self._saveable.add(mid)
        elif coverage >= 0.50:
            color = YELLOW; self._saveable.add(mid)
        else:
            color = RED
        return dict(cur=cur_c, olo=olo, ohi=ohi, tlo=tlo, thi=thi, tspan=tspan,
                    coverage=coverage, cov_pct=coverage * 100, color=color,
                    at_seam=self.bus.at_encoder_boundary(mid))

    def _on_manual_state(self, state, log):
        for mid in self._theo:
            d = self._compute_joint(mid, state)
            if d is not None:
                self._manual_data[mid] = d
        if self._detail_mid is not None:
            self._push_data_to_detail()
        else:
            self._push_data_to_overview()
        self._update_save_btn()

    def _push_data_to_overview(self):
        for mid, row in getattr(self, "_manual_rows", {}).items():
            d = self._manual_data.get(mid)
            if not d:
                continue
            pad = 0.05 * d["tspan"]
            lab = (f"{int(d['olo'])}–{int(d['ohi'])}  {d['cov_pct']:.0f}%"
                   if d["coverage"] >= 0.50 else f"uncalibrated  {d['cov_pct']:.0f}%")
            try:
                row["bar"].set(min(d["tlo"], d["olo"]) - pad, max(d["thi"], d["ohi"]) + pad,
                               d["tlo"], d["thi"], d["olo"], d["ohi"], d["cur"], d["color"])
                row["dot"].config(fg=d["color"])
                row["info"].config(text=lab, fg=d["color"])
            except Exception:
                pass

    def _push_data_to_detail(self):
        mid = self._detail_mid
        d = self._manual_data.get(mid)
        if mid is None or not d or not hasattr(self, "_detail_bar"):
            return
        pad = 0.05 * d["tspan"]
        rel = "good" if d["coverage"] >= 0.80 else "partial" if d["coverage"] >= 0.50 else "poor"
        try:
            self._detail_bar.set(min(d["tlo"], d["olo"]) - pad, max(d["thi"], d["ohi"]) + pad,
                                 d["tlo"], d["thi"], d["olo"], d["ohi"], d["cur"], d["color"])
            self._detail_vals["swept"].config(text=f"{int(d['olo'])} – {int(d['ohi'])}", fg=d["color"])
            self._detail_vals["cov"].config(text=f"{d['cov_pct']:.0f}%  ({rel})", fg=d["color"])
        except Exception:
            pass

    def _manual_save(self):
        """Save the joints swept ≥60 % of their range; joints you didn't sweep keep
        their existing calibration. A swept arc that crosses the 0/4095 seam is
        stored with an auto-computed software offset (no Set Home needed)."""
        saved, wrapped = 0, []
        for mid in sorted(self._saveable):
            c = self._cont.get(mid)
            if c is None:
                continue
            lo, hi = sorted((c["lo"], c["hi"]))
            if hi - lo < 30:
                continue
            stored = self.bus.commit_swept_range(mid, lo, hi)   # picks offset if needed
            if stored.get("offset"):
                wrapped.append(self.bus.id_joint.get(mid, str(mid)))
            saved += 1
        if saved:
            self.bus.save_calib()            # writes + syncs every copy + live driver
            self._theo = {mid: dict(self.bus.calib.get(mid, self._theo[mid]))
                          for mid in self._theo}
            extra = (f" ({len(wrapped)} auto-wrapped: {', '.join(wrapped)})"
                     if wrapped else "")
            self._set_manual_status(f"✓ Saved {saved} joint(s){extra}; others kept "
                                    f"previous calibration.", GREEN)
            self.bus.log(f"Manual calibration: saved {saved} swept joints{extra}")
        else:
            self._set_manual_status("Sweep a joint to ≥60% of its range before saving.",
                                    YELLOW)

    # ── Per-joint actions (re-calibrate under power · re-center) ───────────────

    def _recalibrate_one(self, mid: int):
        """Override and re-measure ONE joint with a powered stall sweep."""
        if self._thread and self._thread.is_alive():
            self._set_manual_status("A calibration is already running.", YELLOW)
            return
        jname = self.bus.id_joint.get(mid)
        if jname is None:
            self._set_manual_status(f"ID {mid}: unknown joint.", RED)
            return
        if not self.bus.ping(mid):
            self._set_manual_status(f"{jname}: not responding.", RED)
            return
        self._stop.clear()
        self._set_manual_status(f"Re-calibrating {jname} under power…", ORANGE)
        self.bus.log(f"Re-calibrate {jname} (single-joint powered sweep)")
        self._thread = threading.Thread(
            target=self._calibrate_bg, args=([jname],),
            kwargs={"status_cb": self._set_manual_status,
                    "done_cb": self._manual_recalib_done}, daemon=True)
        self._thread.start()

    def _manual_recalib_done(self, committed: dict, flagged: set):
        """Main-thread follow-up after a single/selected powered re-calibration:
        refresh the reference range and reseed continuous tracking so the bars
        reflect the new limits, then leave torque off for more hand sweeping."""
        for mid in list(committed) + list(flagged):
            self._theo[mid] = dict(self.bus.calib.get(mid, self._theo.get(mid, {})))
            self._cont.pop(mid, None)
            self._saveable.discard(mid)
        self.bus.set_torque(False)
        if flagged:
            names = ", ".join(self.bus.id_joint.get(m, str(m)) for m in flagged)
            self._set_manual_status(
                f"⚠ {names}: hit the 0/4095 seam under power — hand-sweep it here "
                f"(auto-wraps on Save) or press ↺ to re-center.", YELLOW)
        elif committed:
            names = ", ".join(self.bus.id_joint.get(m, str(m)) for m in committed)
            self._set_manual_status(f"✓ Re-calibrated & saved: {names}.", GREEN)
        else:
            self._set_manual_status("Re-calibrate produced no usable range.", YELLOW)

    def _manual_recenter(self, mid: int):
        """Hardware re-center one joint: burn its CURRENT pose as 2048. Hold the
        joint at its mechanical mid-travel first, then press ↻ to re-sweep."""
        if self._thread and self._thread.is_alive():
            self._set_manual_status("Wait for the running calibration to finish.", YELLOW)
            return
        ok, msg = self.bus.recenter(mid)
        if ok:
            jname = self.bus.id_joint.get(mid, str(mid))
            self._theo[mid] = dict(self.bus.calib.get(mid, self._theo.get(mid, {})))
            self._cont.pop(mid, None)
            self._set_manual_status(f"{msg} Now press ↻ on {jname} (or hand-sweep) "
                                    f"to set its true range.", GREEN)
        else:
            self._set_manual_status(msg, RED)

    def _manual_center_all(self):
        """Latch every responding motor's present pose as 2048 and rebuild its
        range around it (Setup's Center All, on this screen)."""
        if self._thread and self._thread.is_alive():
            self._set_manual_status("Wait for the running calibration to finish.", YELLOW)
            return
        ids = [m for m in self._motor_ids() if self.bus.ping(m)]
        if not ids:
            self._set_manual_status("No motors responding.", YELLOW)
            return
        done = 0
        for m in ids:
            c = self.bus.calib.get(m, {"min": 0, "max": 4095})
            span = max(100, min(3800, abs(int(c.get("max", 4095)) - int(c.get("min", 0))) or 2000))
            if self.bus.center_offset(m):
                lo, hi = 2048 - span // 2, 2048 + span // 2
                self.bus.calib[m] = {"min": lo, "max": hi, "center": 2048, "offset": 0}
                self._theo[m] = dict(self.bus.calib[m])
                self._cont.pop(m, None)
                done += 1
            time.sleep(0.02)
        self.bus.save_calib()
        self._set_manual_status(f"⌖ Centred / homed {done}/{len(ids)} motors at 2048 "
                                f"(saved).", GREEN if done == len(ids) else YELLOW)
        self.bus.log(f"Manual: Center/Set-Home {done}/{len(ids)}")

    # ── Background calibration thread ─────────────────────────────────────────

    def _stop_calib(self):
        self._stop.set()

    def _set_calib_status(self, text, fg):
        w = getattr(self, "_calib_status", None)
        try:
            if w is not None and w.winfo_exists():
                w.config(text=text, fg=fg)
        except Exception:
            pass

    def _set_finger_status(self, text, fg):
        w = getattr(self, "_finger_status", None)
        try:
            if w is not None and w.winfo_exists():
                w.config(text=text, fg=fg)
        except Exception:
            pass

    def _calibrate_bg(self, joints_filter: list | None,
                      status_cb=None, done_cb=None):
        """Powered stall-detection sweep, runnable from any tab.

        Progress goes to ``status_cb(text, color)`` (defaults to the Auto tab's
        status label); a ``done_cb(committed, flagged)`` fires on the main thread
        when finished. A powered goal is clamped to 0–4095, so this sweep cannot
        drive across the single-turn seam: a joint whose travel straddles 0/4095
        stalls there and is *flagged* (hand-sweep it in Manual to auto-wrap, or
        ↺ re-center). Every committed range is routed through
        ``commit_swept_range`` (offset 0 here, identical to the legacy result)."""
        from collections import deque

        def report(text, color):
            self.after(0, status_cb or self._set_calib_status, text, color)

        report("Calibration running…", ORANGE)

        jid      = self.bus.joint_id
        pending  = {mid: {"min": None, "max": None} for mid in range(1, 18)}
        committed= {}
        flagged  = set()        # joints whose powered sweep hit the encoder seam

        sequence = CALIBRATION_SEQUENCE
        if joints_filter:
            jset = set(joints_filter)
            sequence = [
                {**s, "joints": {j: d for j, d in s["joints"].items() if j in jset}}
                for s in sequence
            ]
            sequence = [s for s in sequence if s["joints"]]

        # Power only the motors this sweep touches, so a single-joint re-calibrate
        # never stiffens the whole hand.
        involved = {jid[j] for s in sequence for j in s["joints"] if j in jid}
        self.bus.state.torque_on = bool(involved)

        for step_def in sequence:
            if self._stop.is_set():
                break
            step_n = step_def["step"]
            label  = step_def["label"]

            self.after(0, lambda n=step_n: self._mark_step(n, "running"))
            report(f"Step {step_n}: {label}…", ORANGE)

            mid_dir = {jid[j]: d for j, d in step_def["joints"].items() if j in jid}
            buffers = {mid: deque(maxlen=NUM_STABLE) for mid in mid_dir}
            targets = {}
            stalled = set()

            # Enable torque and read start positions
            for mid in mid_dir:
                ms = self.bus.state.motors.get(mid)
                targets[mid] = ms.raw if (ms and ms.raw is not None) else 2047
                self.bus.write_register(mid, 40, 1, 1)  # torque on

            time.sleep(0.3)

            while len(stalled) < len(mid_dir) and not self._stop.is_set():
                active = [m for m in mid_dir if m not in stalled]
                write_t = {}
                for mid in active:
                    direction = mid_dir[mid]
                    targets[mid] = max(0, min(4095, targets[mid] + direction * STEP_RAW))
                    write_t[mid] = targets[mid]
                self.bus.sync_write(write_t, speed=CALIB_SPEED, torque=CALIB_TORQUE)
                time.sleep(STEP_DELAY)

                for mid in active:
                    ms = self.bus.state.motors.get(mid)
                    if ms and ms.raw is not None and 0 <= ms.raw <= 4095:
                        buffers[mid].append(ms.raw)
                        if (len(buffers[mid]) == NUM_STABLE and
                                max(buffers[mid]) - min(buffers[mid]) <= STALL_ATOL):
                            stalled.add(mid)
                            direction = mid_dir[mid]
                            jname = self.bus.id_joint.get(mid, str(mid))
                            # A stall right at 0/4095 is the single-turn seam, not a
                            # mechanical stop: the powered goal clamps there so we
                            # can't cross it. Flag it for a hand-sweep / re-center
                            # instead of recording a bogus limit.
                            if (ms.raw <= ENC_BOUNDARY or
                                    ms.raw >= 4095 - ENC_BOUNDARY):
                                flagged.add(mid)
                                self.bus.log(f"  ⚠ ID {mid} ({jname}) hit the "
                                             f"encoder seam at {ms.raw} — hand-sweep "
                                             f"to auto-wrap or ↺ re-center.")
                                continue
                            limit = max(0, min(4095, ms.raw - direction * BACKOFF_RAW))
                            key   = "max" if direction > 0 else "min"
                            pending[mid][key] = limit
                            self.bus.log(f"  ID {mid} ({jname}) {key}={limit}")

            # Disable torque for this group
            for mid in mid_dir:
                self.bus.write_register(mid, 40, 0, 1)

            self.after(0, lambda n=step_n: self._mark_step(n, "done"))

        # Commit each fully-swept, non-flagged joint through the wrap-aware path.
        for mid in range(1, 18):
            if mid in flagged:
                continue
            p = pending[mid]
            if p["min"] is not None and p["max"] is not None:
                committed[mid] = self.bus.commit_swept_range(mid, p["min"], p["max"])

        if committed:
            self.bus.save_calib()
        # Every touched motor is de-energised now; reflect that in the E-Stop UI.
        self.bus.state.torque_on = False
        self.bus._notify()

        names = sorted(self.bus.id_joint.get(m, str(m)) for m in flagged)
        if flagged:
            report("⚠ " + ", ".join(names) + " hit the 0/4095 seam — hand-sweep them "
                   "in Manual (auto-wraps on Save) or ↺ re-center. " +
                   (f"Other {len(committed)} saved." if committed else "Nothing saved."),
                   YELLOW)
            self.bus.log(f"Calibration: {len(committed)} saved, {len(flagged)} at seam "
                         f"({', '.join(names)})")
        elif committed:
            report(f"Complete — {len(committed)} motors calibrated.", GREEN)
            self.bus.log(f"Calibration done: {len(committed)} joints")
        else:
            report("Stopped or no data.", YELLOW)

        if done_cb is not None:
            self.after(0, done_cb, dict(committed), set(flagged))

    def _mark_step(self, step_n: int, status: str):
        dot = self._step_labels.get(step_n)
        if dot:
            if status == "running":
                dot.config(text="◉", fg=ORANGE)
            elif status == "done":
                dot.config(text="✓", fg=GREEN)
