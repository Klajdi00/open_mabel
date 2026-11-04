"""Control Panel — Joint sliders, per-finger control, whole-hand control.

Matches the old SliderApp._build_columns() layout exactly:
  Joints  — per-joint AppleSlider cards, organised by finger column
  Fingers — one macro AppleSlider per finger (0-100 %)
  Hand    — single wide AppleSlider for all joints

The active mode is driven from the toolbar (gui.py sets it via set_mode()).
"""

import sys
import threading
import time
import tkinter as tk
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from style import *

# ── Layout (tip → base) ───────────────────────────────────────────────────────
LAYOUT = {
    "left": [
        ("Thumb",  ["thumb_ip", "thumb_mp", "thumb_abd", "thumb_cmc"]),
        ("Index",  ["index_pip", "index_mcp", "index_abd"]),
        ("Middle", ["middle_pip", "middle_mcp", "middle_abd"]),
        ("Ring",   ["ring_pip", "ring_mcp", "ring_abd"]),
        ("Pinky",  ["pinky_pip", "pinky_mcp", "pinky_abd"]),
        ("Wrist",  ["wrist"]),
    ],
    "right": [
        ("Thumb",  ["thumb_ip", "thumb_mp", "thumb_abd", "thumb_cmc"]),
        ("Index",  ["index_pip", "index_mcp", "index_abd"]),
        ("Middle", ["middle_pip", "middle_mcp", "middle_abd"]),
        ("Ring",   ["ring_pip", "ring_mcp", "ring_abd"]),
        ("Pinky",  ["pinky_pip", "pinky_mcp", "pinky_abd"]),
        ("Wrist",  ["wrist"]),
    ],
}

POSES = {
    "Flat":  {j: 5  for j in ["index_pip","index_mcp","middle_pip","middle_mcp",
                               "ring_pip","ring_mcp","pinky_pip","pinky_mcp",
                               "thumb_ip","thumb_mp","thumb_cmc"]},
    "Fist":  {j: 90 for j in ["index_pip","index_mcp","middle_pip","middle_mcp",
                               "ring_pip","ring_mcp","pinky_pip","pinky_mcp",
                               "thumb_ip","thumb_mp","thumb_cmc"]},
    "Pinch": {"index_pip":60,"index_mcp":60,"middle_pip":5,"middle_mcp":5,
              "ring_pip":5,"ring_mcp":5,"pinky_pip":5,"pinky_mcp":5,
              "thumb_ip":55,"thumb_mp":60,"thumb_cmc":55},
    "Point": {"index_pip":5,"index_mcp":5,"middle_pip":90,"middle_mcp":85,
              "ring_pip":90,"ring_mcp":85,"pinky_pip":90,"pinky_mcp":85,
              "thumb_ip":65,"thumb_mp":65,"thumb_cmc":65},
    "Peace": {"index_pip":5,"index_mcp":5,"middle_pip":5,"middle_mcp":5,
              "ring_pip":90,"ring_mcp":85,"pinky_pip":90,"pinky_mcp":85,
              "thumb_ip":65,"thumb_mp":65,"thumb_cmc":65},
}

SEND_HZ = 40


class ControlPanel(tk.Frame):

    def __init__(self, parent, bus):
        super().__init__(parent, bg=BG)
        self.bus         = bus
        self._ctrl_mode  = "Joints"
        self._sliders:   dict[str, AppleSlider]  = {}
        self._lbl_vars:  dict[str, tk.StringVar] = {}
        self._pending:   dict[int, int]          = {}
        self._lock       = threading.Lock()
        self._last_drag  = 0.0
        self._is_reading = False
        self._cols_frame: tk.Frame | None = None
        self._mode_seg = None

        # Pack order matters in Tkinter: fixed top + bottom bars FIRST so they
        # reserve their space, then the expanding middle fills what's left.

        # 1. Mode toggle bar  (Joints / Fingers / Hand)  — top
        self._build_mode_bar()

        # 2. Preset / pose bar  — bottom
        self._build_pose_bar()

        # 3. Slider area  — fills the middle (packed LAST)
        self._main = tk.Frame(self, bg=BG)
        self._main.pack(fill="both", expand=True, side="top")

        # Initial columns
        self._build_columns()

        # Background send thread
        threading.Thread(target=self._send_loop, daemon=True).start()

        # Subscribe to state updates for slider readback
        bus.subscribe(self._on_state)

    # ── Mode toggle bar ───────────────────────────────────────────────────────

    def _build_mode_bar(self):
        bar = tk.Frame(self, bg=BG, height=66)
        bar.pack(fill="x", side="top")
        bar.pack_propagate(False)

        inner = tk.Frame(bar, bg=BG)
        inner.pack(side="left", padx=30, pady=14)

        tk.Label(inner, text="CONTROL MODE", bg=BG, fg=LABEL2,
                 font=FONT_SMALL).pack(side="left", padx=(0, 16))

        self._mode_seg = AppleSegmentedControl(
            inner, ["Joints", "Fingers", "Hand"],
            self._ctrl_mode, self._on_mode_change,
            width=300, height=40, parent_bg=BG)
        self._mode_seg.pack(side="left")

        tk.Frame(self, bg=SEP, height=1).pack(fill="x", side="top")

    def _on_mode_change(self, mode: str):
        self._ctrl_mode = mode
        self._build_columns()

    # ── Public API ────────────────────────────────────────────────────────────

    def set_mode(self, mode: str):
        if mode == self._ctrl_mode:
            return
        self._ctrl_mode = mode
        if self._mode_seg is not None:
            self._mode_seg.set(mode)
        self._build_columns()

    def set_hand(self, hand: str):
        self._build_columns()

    def rebuild(self):
        """Rebuild the slider columns — call after connecting so the now-populated
        joint map produces sliders (fixes blank Joints view on first connect)."""
        self._build_columns()

    def destroy(self):
        # Remove our bus callback so a rebuilt panel doesn't leave a dangling
        # subscription pointing at destroyed widgets.
        try:
            self.bus.unsubscribe(self._on_state)
        except Exception:
            pass
        super().destroy()

    # ── Column builder ────────────────────────────────────────────────────────

    def _build_columns(self):
        if self._cols_frame:
            self._cols_frame.destroy()

        outer = tk.Frame(self._main, bg=BG)
        outer.pack(fill="both", expand=True, padx=25, pady=25)
        self._cols_frame = outer

        self._sliders.clear()
        self._lbl_vars.clear()

        jid  = self.bus.joint_id or {}
        hand = self.bus.hand or "left"

        if self._ctrl_mode == "Joints":
            self._build_joints(outer, jid, hand)
        elif self._ctrl_mode == "Fingers":
            self._build_fingers(outer, jid, hand)
        else:
            self._build_hand(outer, jid, hand)

    # ── Joints mode ───────────────────────────────────────────────────────────

    def _build_joints(self, outer, jid, hand):
        for col_idx, (finger, joints) in enumerate(LAYOUT[hand]):
            col = tk.Frame(outer, bg=BG)
            col.grid(row=0, column=col_idx, sticky="ns", padx=12)

            tk.Label(col, text=finger.upper(), bg=BG, fg=LABEL2,
                     font=FONT_SUB, anchor="w").pack(fill="x", pady=(0, 12))

            for jname in joints:
                mid = jid.get(jname)
                if mid is None:
                    continue

                c = self.bus.calib.get(mid, {"min": 0, "max": 4095, "center": 2047})
                calibrated = not (c["min"] == 0 and c["max"] == 4095)

                card = tk.Frame(col, bg=SURFACE, padx=14, pady=14)
                card.pack(fill="x", pady=6)

                name_row = tk.Frame(card, bg=SURFACE)
                name_row.pack(fill="x")
                tk.Label(name_row,
                         text=jname.replace("_", " ").title(),
                         bg=SURFACE, fg=LABEL, font=FONT_BODY,
                         anchor="w").pack(side="left")
                id_bg = ORANGE if calibrated else SEP
                tk.Label(name_row, text=f"ID {mid}",
                         bg=id_bg, fg=LABEL, font=FONT_SMALL,
                         padx=5, pady=3).pack(side="right")

                sl = AppleSlider(
                    card, from_=c["min"], to=c["max"],
                    width=150, height=28,
                    command=lambda v, j=jname, m=mid: self._on_joint_slide(j, m, v),
                )
                sl.set(c["center"])
                sl.pack(fill="x", pady=(10, 8))
                self._sliders[jname] = sl

                val_row = tk.Frame(card, bg=SURFACE)
                val_row.pack(fill="x")
                lv = tk.StringVar(value=f"{c['center']:4d} | 50%")
                tk.Label(val_row, textvariable=lv, bg=SURFACE,
                         fg=LABEL, font=FONT_MONO, anchor="w").pack(side="left")
                tk.Label(val_row, text=f"[{c['min']} ↔ {c['max']}]",
                         bg=SURFACE, fg=LABEL2, font=FONT_SMALL,
                         anchor="e").pack(side="right")
                self._lbl_vars[jname] = lv

        outer.grid_columnconfigure(tuple(range(len(LAYOUT[hand]))), weight=1)

    # ── Fingers mode ──────────────────────────────────────────────────────────

    def _build_fingers(self, outer, jid, hand):
        for col_idx, (finger, joints) in enumerate(LAYOUT[hand]):
            col = tk.Frame(outer, bg=BG)
            col.grid(row=0, column=col_idx, sticky="ns", padx=16)

            mids = [jid.get(j) for j in joints if jid.get(j)]
            if not mids:
                continue

            card = tk.Frame(col, bg=SURFACE, padx=20, pady=20)
            card.pack(fill="x", pady=40)

            name_row = tk.Frame(card, bg=SURFACE)
            name_row.pack(fill="x", pady=(0, 18))
            tk.Label(name_row, text=finger.upper(), bg=SURFACE,
                     fg=LABEL, font=FONT_TITLE, anchor="w").pack(side="left")
            id_str = ",".join(str(m) for m in mids)
            tk.Label(name_row, text=f"IDs {id_str}", bg=SEP,
                     fg=LABEL, font=FONT_SMALL, padx=8, pady=4).pack(side="right")

            sl = AppleSlider(
                card, from_=0, to=100, width=170, height=36,
                command=lambda v, n=finger, m=mids: self._on_macro_slide(n, m, v),
            )
            sl.set(50)
            sl.pack(fill="x", pady=(16, 12))
            self._sliders[finger] = sl

            val_row = tk.Frame(card, bg=SURFACE)
            val_row.pack(fill="x")
            lv = tk.StringVar(value="50%")
            tk.Label(val_row, textvariable=lv, bg=SURFACE,
                     fg=BLUE, font=FONT_MONO, anchor="w").pack(side="left")
            tk.Label(val_row, text="[0 ↔ 100]", bg=SURFACE,
                     fg=LABEL2, font=FONT_SMALL, anchor="e").pack(side="right")
            self._lbl_vars[finger] = lv

        outer.grid_columnconfigure(tuple(range(len(LAYOUT[hand]))), weight=1)

    # ── Hand mode ─────────────────────────────────────────────────────────────

    def _build_hand(self, outer, jid, hand):
        col  = tk.Frame(outer, bg=BG)
        col.pack(expand=True)

        card = tk.Frame(col, bg=SURFACE, padx=40, pady=40)
        card.pack(pady=60)

        name_row = tk.Frame(card, bg=SURFACE)
        name_row.pack(fill="x", pady=(0, 20))
        tk.Label(name_row, text="WHOLE HAND CONTROL", bg=SURFACE,
                 fg=LABEL, font=FONT_TITLE, anchor="w").pack(side="left")
        tk.Label(name_row, text=f"IDs 1-17", bg=ORANGE,
                 fg=LABEL, font=FONT_SMALL, padx=12, pady=6).pack(side="right")

        all_mids = [m for m in jid.values()]
        sl = AppleSlider(
            card, from_=0, to=100, width=460, height=44,
            command=lambda v, m=all_mids: self._on_macro_slide("Hand", m, v),
        )
        sl.set(50)
        sl.pack(fill="x", pady=(30, 20))
        self._sliders["Hand"] = sl

        val_row = tk.Frame(card, bg=SURFACE)
        val_row.pack(fill="x")
        lv = tk.StringVar(value="50%")
        tk.Label(val_row, textvariable=lv, bg=SURFACE,
                 fg=BLUE, font=("Menlo", 18, "bold"), anchor="w").pack(side="left")
        tk.Label(val_row, text="[0 ↔ 100]", bg=SURFACE,
                 fg=LABEL2, font=FONT_SUB, anchor="e").pack(side="right")
        self._lbl_vars["Hand"] = lv

    # ── Pose bar ──────────────────────────────────────────────────────────────

    def _build_pose_bar(self):
        tk.Frame(self, bg=SEP, height=1).pack(fill="x", side="bottom")
        bar = tk.Frame(self, bg=SURFACE, height=95)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)

        inner = tk.Frame(bar, bg=SURFACE)
        inner.pack(side="left", padx=30, pady=25)
        tk.Label(inner, text="Presets:", bg=SURFACE,
                 fg=LABEL2, font=FONT_SUB).pack(side="left", padx=(0, 20))

        # Using the `btn` wrapper directly instead of `AppleButton` + `_btn_w`
        for name in POSES:
            btn(inner, name,
                lambda n=name: self._apply_pose(n),
                bg=ELEVATED, parent_bg=SURFACE,
                px=20, height=40, radius=12).pack(side="left", padx=10)

        tk.Frame(bar, bg=SEP, width=1).pack(side="right", padx=25,
                                             fill="y", pady=20)
        
        btn(bar, "Close All", self._close_all,
            bg=RED, parent_bg=SURFACE,
            width=130, height=44, radius=14).pack(side="right", padx=(10, 30), pady=25)
        btn(bar, "Open All",  self._open_all,
            bg=BLUE, parent_bg=SURFACE,
            width=130, height=44, radius=14).pack(side="right", padx=10, pady=25)

    # ── Slider callbacks ──────────────────────────────────────────────────────

    def _disp_raw(self, mid: int, raw: int) -> int:
        """Map actual raw tick <-> slider display value. For an inverted joint the
        slider is mirrored so it tracks the SAME direction as the telemetry bar
        and the percent read-out (both invert-aware). Self-inverse."""
        c = self.bus.calib.get(mid, {"min": 0, "max": 4095})
        lo, hi = int(c["min"]), int(c["max"])
        return (lo + hi - int(raw)) if self.bus._inverted(mid) else int(raw)

    def _on_joint_slide(self, jname: str, mid: int, v: int):
        if self._is_reading:
            return
        self._last_drag = time.time()
        raw = self._disp_raw(mid, v)            # slider display value -> actual tick
        pct = self.bus.raw_to_pct(mid, raw)
        if jname in self._lbl_vars:
            self._lbl_vars[jname].set(f"{raw:4d} | {pct:3.0f}%")
        with self._lock:
            self._pending[mid] = raw

    def _on_macro_slide(self, name: str, mids: list, pct: int):
        self._last_drag = time.time()
        if name in self._lbl_vars:
            self._lbl_vars[name].set(f"{pct}%")
        with self._lock:
            for mid in mids:
                self._pending[mid] = self.bus.pct_to_raw(mid, pct)

    # ── Poses ─────────────────────────────────────────────────────────────────

    def _apply_pose(self, name: str):
        pose = POSES.get(name, {})
        jid  = self.bus.joint_id
        self._last_drag = time.time()

        if self._ctrl_mode == "Joints":
            for jname, pct in pose.items():
                if jname in self._sliders and jid.get(jname):
                    mid = jid[jname]
                    self._sliders[jname].set(
                        self._disp_raw(mid, self.bus.pct_to_raw(mid, pct)))
        elif self._ctrl_mode in ("Fingers", "Hand"):
            for s in self._sliders.values():
                s.set(50)

        with self._lock:
            for jname, pct in pose.items():
                mid = jid.get(jname)
                if mid:
                    self._pending[mid] = self.bus.pct_to_raw(mid, pct)

    def _open_all(self):
        self._apply_pose("Flat")

    def _close_all(self):
        self._apply_pose("Fist")

    # ── Send loop ─────────────────────────────────────────────────────────────

    def _send_loop(self):
        dt = 1.0 / SEND_HZ
        while True:
            with self._lock:
                pending = dict(self._pending)
                self._pending.clear()
            if pending:
                # Moving a control implies intent to move the hand — arm torque
                # automatically so the joint actually responds (it is off by
                # default, and after an E-Stop). E-Stop still kills it instantly.
                if not self.bus.state.torque_on:
                    self.bus.set_torque(True)
                try:
                    self.bus.sync_write(pending, speed=2000, torque=500)
                except Exception:
                    pass
            time.sleep(dt)

    # ── State subscription ────────────────────────────────────────────────────

    def _on_state(self, state, log):
        if time.time() - self._last_drag < 1.0:
            return   

        if self._ctrl_mode != "Joints":
            return

        jid = self.bus.joint_id
        if not jid:
            return

        id_joint = {v: k for k, v in jid.items()}
        for mid, ms in state.motors.items():
            if not ms.responding or ms.raw is None:
                continue
            jname = id_joint.get(mid)
            if not jname or jname not in self._sliders:
                continue
            sl   = self._sliders[jname]
            disp = self._disp_raw(mid, ms.raw)
            if abs(sl.get() - disp) > 15:
                self._is_reading = True
                sl.set(disp)
                pct = self.bus.raw_to_pct(mid, ms.raw)
                if jname in self._lbl_vars:
                    self._lbl_vars[jname].set(f"{ms.raw:4d} | {pct:3.0f}%")
                self._is_reading = False