#!/usr/bin/env python3
"""MABEL ORCA Hand — Advanced Control Studio.

Features:
- Telemetry Sidebar (Pos, Vel, Load, Temp)
- Tri-Mode Control: Individual Joints, Whole Fingers, or Entire Hand
- True iOS Dark Mode Aesthetics with Vector Sliders, Segmented Controls, and Pill Buttons
"""

import argparse
import sys
import threading
import time
from pathlib import Path
import tkinter as tk

_FEETECH   = Path(__file__).resolve().parent.parent
_ORCA_ROOT = _FEETECH.parent / "orca_core"
for _p in (str(_FEETECH), str(_ORCA_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from orca_core.hardware.feetech import (
    PortHandler, sms_sts, GroupSyncRead, COMM_SUCCESS,
)
from orca_core.hardware.feetech.sms_sts import SMS_STS_PRESENT_POSITION_L
from utils.hand_config import DATA_DIR, detect_port


# ── Per-hand joint layouts (Tip to Base) ──────────────────────────────────────
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

JOINT_ID = {
    "left": {
        "middle_pip": 1,  "pinky_mcp":  2,  "pinky_pip":  3,  "pinky_abd":  4,
        "middle_abd": 5,  "index_abd":  6,  "index_mcp":  7,  "index_pip":  8,
        "thumb_mp":   9,  "thumb_ip":  10,  "thumb_abd": 11,  "thumb_cmc": 12,
        "ring_abd":  13,  "ring_pip":  14,  "ring_mcp":  15,  "middle_mcp":16,  "wrist": 17,
    },
    "right": {
        "index_pip":  1,  "index_mcp":  2,  "index_abd":  3,  "middle_abd": 4,
        "pinky_abd":  5,  "pinky_pip":  6,  "pinky_mcp":  7,  "middle_pip": 8,
        "middle_mcp": 9,  "ring_mcp":  10,  "ring_pip":  11,  "ring_abd":  12,
        "thumb_cmc": 13,  "thumb_abd": 14,  "thumb_mp":  15,  "thumb_ip":  16,  "wrist": 17,
    },
}

POSES = {
    "Flat":  {j: 5  for j in ["index_pip","index_mcp","middle_pip","middle_mcp","ring_pip","ring_mcp","pinky_pip","pinky_mcp","thumb_ip","thumb_mp","thumb_cmc"]},
    "Fist":  {j: 90 for j in ["index_pip","index_mcp","middle_pip","middle_mcp","ring_pip","ring_mcp","pinky_pip","pinky_mcp","thumb_ip","thumb_mp","thumb_cmc"]},
    "Pinch": {"index_pip": 60,"index_mcp": 60, "middle_pip":5, "middle_mcp":5, "ring_pip":5, "ring_mcp":5, "pinky_pip":5, "pinky_mcp":5, "thumb_ip": 55, "thumb_mp":60,"thumb_cmc":55},
    "Point": {"index_pip":5,"index_mcp":5, "middle_pip":90,"middle_mcp":85, "ring_pip":90, "ring_mcp":85, "pinky_pip":90, "pinky_mcp":85, "thumb_ip":65, "thumb_mp":65,"thumb_cmc":65},
    "Peace": {"index_pip":5,"index_mcp":5, "middle_pip":5,"middle_mcp":5, "ring_pip":90,"ring_mcp":85, "pinky_pip":90,"pinky_mcp":85, "thumb_ip":65,"thumb_mp":65,"thumb_cmc":65},
}

BAUDRATE = 1_000_000
TOTAL    = 17
SEND_HZ  = 40
POLL_HZ  = 15
TORQUE   = 400
SPEED    = 300
ACC      = 30

CALIB_PATH = DATA_DIR / "calibration" / "joint_limits.json"

# ── iOS App Dark Mode Aesthetic ───────────────────────────────────────────────
BG          = "#000000"   
PANEL_BG    = "#121214"   
SURFACE     = "#1C1C1E"   
ELEVATED    = "#2C2C2E"   
SEP         = "#3A3A3C"   
LABEL       = "#FFFFFF"   
LABEL2      = "#8E8E93"   
ORANGE      = "#FF9F0A"   
DARK_ORANGE = "#CC7A00"   
GREEN       = "#30D158"   
RED         = "#FF453A"   
BLUE        = "#0A84FF"   

FONT_TITLE  = ("Helvetica Neue", 18, "bold")
FONT_SUB    = ("Helvetica Neue", 14, "bold")
FONT_BODY   = ("Helvetica Neue", 13, "bold")
FONT_SMALL  = ("Helvetica Neue", 11, "bold")
FONT_MONO   = ("Menlo", 12)


# ── Bus ───────────────────────────────────────────────────────────────────────

def _load_calibration() -> dict:
    import json
    if CALIB_PATH.exists():
        with open(CALIB_PATH) as f:
            raw = json.load(f)
        return {int(k): {"min": int(v["min"]), "max": int(v["max"]), "center": int(v["center"])} for k, v in raw.items()}
    return {mid: {"min": 0, "max": 4095, "center": 2047} for mid in range(1, TOTAL + 1)}

class Bus:
    def __init__(self, port: str):
        self.ph = PortHandler(port)
        self.ph.baudrate = BAUDRATE
        if not self.ph.openPort():
            raise OSError(f"Cannot open {port}")
        self.pkt = sms_sts(self.ph)
        self._lock = threading.Lock()

    def close(self):
        try:
            with self._lock:
                self.pkt.write1ByteTxRx(0xFE, 40, 0)
                self.ph.closePort()
        except Exception: pass

    def torque(self, ids: list, on: bool):
        with self._lock:
            for mid in ids:
                self.pkt.write1ByteTxRx(mid, 40, int(on))

    def sync_write(self, targets: dict):
        with self._lock:
            self.pkt.groupSyncWrite.clearParam()
            for mid, raw in targets.items():
                self.pkt.SyncWritePosEx(mid, int(raw), SPEED, ACC, TORQUE)
            self.pkt.groupSyncWrite.txPacket()
            self.pkt.groupSyncWrite.clearParam()

    def sync_read_all(self, ids: list) -> dict:
        with self._lock:
            sr = GroupSyncRead(self.pkt, SMS_STS_PRESENT_POSITION_L, 8)
            for mid in ids: sr.addParam(mid)
            if sr.txRxPacket() != COMM_SUCCESS: return {}
            
            out = {}
            for mid in ids:
                if sr.isAvailable(mid, SMS_STS_PRESENT_POSITION_L, 8)[0]:
                    r_pos = sr.getData(mid, 56, 2)
                    r_vel = sr.getData(mid, 58, 2)
                    r_load= sr.getData(mid, 60, 2)
                    r_temp= sr.getData(mid, 63, 1)

                    out[mid] = {
                        'pos': self.pkt.scs_tohost(r_pos, 15),
                        'vel': self.pkt.scs_tohost(r_vel, 15),
                        'load': r_load & 0x3FF,
                        'temp': r_temp
                    }
            return out


# ── Helpers ───────────────────────────────────────────────────────────────────

def pct_to_raw(mid: int, pct: float, calib: dict) -> int:
    c = calib.get(mid, {"min": 0, "max": 4095})
    lo, hi = c["min"], c["max"]
    return int(lo + (max(0.0, min(100.0, pct)) / 100.0) * (hi - lo))

def raw_to_pct(mid: int, raw: int, calib: dict) -> float:
    c = calib.get(mid, {"min": 0, "max": 4095})
    lo, hi = c["min"], c["max"]
    if hi == lo: return 0.0
    pct = (raw - lo) / (hi - lo) * 100.0
    return max(0.0, min(100.0, pct))


# ── Custom iOS UI Elements ────────────────────────────────────────────────────

class AppleSlider(tk.Canvas):
    def __init__(self, parent, from_, to, command=None, width=160, height=28, **kwargs):
        super().__init__(parent, width=width, height=height, bg=SURFACE, highlightthickness=0, **kwargs)
        self.min_val, self.max_val = min(from_, to), max(from_, to)
        self.val = self.min_val
        self.command = command
        self.padding = 14
        self.track_y = height // 2

        self.id_trough = self.create_line(self.padding, self.track_y, width-self.padding, self.track_y, fill=ELEVATED, width=8, capstyle=tk.ROUND)
        self.id_fill = self.create_line(self.padding, self.track_y, self.padding, self.track_y, fill=ORANGE, width=8, capstyle=tk.ROUND)
        self.id_shadow = self.create_oval(0, 0, 0, 0, fill="#000000", outline="")
        self.id_thumb = self.create_oval(0, 0, 0, 0, fill=LABEL, outline="")

        self.bind("<Button-1>", self._on_drag)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<Configure>", lambda e: self._redraw())

    def set(self, val):
        self.val = max(self.min_val, min(self.max_val, val))
        self._redraw()

    def get(self): return self.val

    def _redraw(self):
        w = self.winfo_width()
        if w <= 1: w = int(self.cget('width'))
        pct = 0 if self.max_val == self.min_val else (self.val - self.min_val) / (self.max_val - self.min_val)
        x = self.padding + pct * (w - 2 * self.padding)
        
        self.coords(self.id_trough, self.padding, self.track_y, w-self.padding, self.track_y)
        self.coords(self.id_fill, self.padding, self.track_y, x, self.track_y)
        
        r = 13 
        self.coords(self.id_shadow, x-r, self.track_y-r+2, x+r, self.track_y+r+2)
        self.coords(self.id_thumb, x-r, self.track_y-r, x+r, self.track_y+r)

    def _on_drag(self, event):
        w = self.winfo_width()
        if w <= 1: w = int(self.cget('width'))
        x_clamped = max(self.padding, min(w - self.padding, event.x))
        pct = (x_clamped - self.padding) / (w - 2 * self.padding)
        self.val = int(self.min_val + pct * (self.max_val - self.min_val))
        self._redraw()
        if self.command: self.command(self.val)


class AppleButton(tk.Canvas):
    def __init__(self, parent, text, cmd, bg_color, fg_color=LABEL, font=FONT_BODY, parent_bg=SURFACE, radius=18, width=100, height=38, **kwargs):
        super().__init__(parent, width=width, height=height, bg=parent_bg, highlightthickness=0, **kwargs)
        self.cmd = cmd
        self.text_val = text
        self.default_bg = bg_color
        self.fg_color = fg_color
        self.font = font
        self.radius = radius
        
        def calc_hover(hex_color):
            if hex_color.startswith("#"):
                try:
                    r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
                    return f"#{min(255, int(r*1.15+15)):02x}{min(255, int(g*1.15+15)):02x}{min(255, int(b*1.15+15)):02x}"
                except Exception: return hex_color
            return hex_color

        self.hover_bg = calc_hover(self.default_bg)
        self.current_bg = self.default_bg
        self.current_fg = self.fg_color
        
        self.shapes = []
        self.text_id = None
        
        self.bind("<Configure>", self._draw)
        self.bind("<Enter>", lambda e: self._recolor(self.hover_bg, self.fg_color))
        self.bind("<Leave>", lambda e: self._recolor(self.default_bg, self.fg_color))
        self.bind("<ButtonPress-1>", lambda e: self._recolor(self.fg_color, self.default_bg)) 
        self.bind("<ButtonRelease-1>", self._on_release)
        
    def _draw(self, event=None):
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        if w <= 1: w = int(self.cget("width"))
        if h <= 1: h = int(self.cget("height"))
        r = min(self.radius, h//2, w//2)
        
        kwargs = {"fill": self.current_bg, "outline": ""}
        self.shapes = [
            self.create_oval(0, 0, 2*r, 2*r, **kwargs),
            self.create_oval(w-2*r, 0, w, 2*r, **kwargs),
            self.create_oval(0, h-2*r, 2*r, h, **kwargs),
            self.create_oval(w-2*r, h-2*r, w, h, **kwargs),
            self.create_rectangle(r, 0, w-r, h, **kwargs),
            self.create_rectangle(0, r, w, h-r, **kwargs)
        ]
        self.text_id = self.create_text(w//2, h//2, text=self.text_val, fill=self.current_fg, font=self.font)

    def _recolor(self, bg_col, fg_col):
        self.current_bg = bg_col
        self.current_fg = fg_col
        for s in self.shapes: self.itemconfig(s, fill=self.current_bg)
        if self.text_id: self.itemconfig(self.text_id, fill=self.current_fg)
            
    def _on_release(self, e):
        self._recolor(self.hover_bg, self.fg_color)
        if -15 <= e.x <= self.winfo_width()+15 and -15 <= e.y <= self.winfo_height()+15:
            if self.cmd: self.cmd()

    def update_colors(self, new_bg, new_fg=LABEL):
        self.default_bg = new_bg
        self.fg_color = new_fg
        try:
            r, g, b = int(new_bg[1:3], 16), int(new_bg[3:5], 16), int(new_bg[5:7], 16)
            self.hover_bg = f"#{min(255, int(r*1.15+15)):02x}{min(255, int(g*1.15+15)):02x}{min(255, int(b*1.15+15)):02x}"
        except Exception: self.hover_bg = new_bg
        self._recolor(self.default_bg, self.fg_color)

    def update_text(self, new_text):
        self.text_val = new_text
        if self.text_id: self.itemconfig(self.text_id, text=self.text_val)


class AppleSegmentedControl(tk.Canvas):
    """Vector-drawn iOS Segmented Control (Smooth Pill with active highlight)."""
    def __init__(self, parent, items, default_item, command, width=220, height=38, radius=10, parent_bg=SURFACE, **kwargs):
        super().__init__(parent, width=width, height=height, bg=parent_bg, highlightthickness=0, **kwargs)
        self.items = items
        self.active_item = default_item
        self.command = command
        self.radius = radius
        self.bg_color = ELEVATED
        self.active_color = ORANGE

        self.bind("<Configure>", self._draw)
        self.bind("<Button-1>", self._on_click)

    def _draw(self, event=None):
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        if w <= 1: w = int(self.cget("width"))
        if h <= 1: h = int(self.cget("height"))
        
        self._draw_rounded_rect(0, 0, w, h, self.radius, self.bg_color)

        idx = self.items.index(self.active_item)
        seg_w = w / len(self.items)
        pad = 3
        
        ax0 = idx * seg_w + pad
        ay0 = pad
        ax1 = (idx + 1) * seg_w - pad
        ay1 = h - pad
        self._draw_rounded_rect(ax0, ay0, ax1, ay1, self.radius - 2, self.active_color)

        for i, item in enumerate(self.items):
            cx = i * seg_w + seg_w / 2
            cy = h / 2
            self.create_text(cx, cy, text=item.capitalize() if item != "Joints" else item, fill=LABEL, font=FONT_BODY)

    def _draw_rounded_rect(self, x0, y0, x1, y1, r, color):
        self.create_oval(x0, y0, x0+2*r, y0+2*r, fill=color, outline="")
        self.create_oval(x1-2*r, y0, x1, y0+2*r, fill=color, outline="")
        self.create_oval(x0, y1-2*r, x0+2*r, y1, fill=color, outline="")
        self.create_oval(x1-2*r, y1-2*r, x1, y1, fill=color, outline="")
        self.create_rectangle(x0+r, y0, x1-r, y1, fill=color, outline="")
        self.create_rectangle(x0, y0+r, x1, y1-r, fill=color, outline="")

    def _on_click(self, event):
        w = self.winfo_width()
        if w <= 1: w = int(self.cget("width"))
        seg_w = w / len(self.items)
        idx = int(event.x // seg_w)
        if 0 <= idx < len(self.items):
            selected = self.items[idx]
            if selected != self.active_item:
                self.active_item = selected
                self._draw()
                self.command(selected)


# ── Main application ──────────────────────────────────────────────────────────

class SliderApp:
    def __init__(self, root: tk.Tk, bus: Bus, calib: dict, initial_hand: str):
        self.root   = root
        self.bus    = bus
        self.calib  = calib
        self.ids    = list(range(1, TOTAL + 1))
        self._hand  = initial_hand
        self._torque_on = True
        self._ctrl_mode = "Joints" 

        self._sliders:  dict[str, AppleSlider]  = {}
        self._lbl_vars: dict[str, tk.StringVar] = {}
        self._tel_vars: dict[int, dict]         = {}
        self._pending:  dict[int, int]          = {}
        self._lock = threading.Lock()

        self._last_user_interaction = 0.0
        self._is_reading = False

        self._build_window()
        self._build_toolbar()
        
        self.main_container = tk.Frame(self.root, bg=BG)
        self.main_container.pack(fill="both", expand=True)
        
        self._build_telemetry()
        
        self.controls_container = tk.Frame(self.main_container, bg=BG)
        self.controls_container.pack(side="left", fill="both", expand=True)

        self._cols_frame = None
        self._build_columns()
        self._build_pose_bar()
        self._start_threads()

    def _build_window(self):
        self.root.title("ORCA Hand Studio")
        self.root.configure(bg=BG)
        self.root.resizable(True, True)
        self.root.minsize(1250, 750)

    def _build_toolbar(self):
        bar = tk.Frame(self.root, bg=SURFACE, height=85)
        bar.pack(fill="x", side="top")
        bar.pack_propagate(False)

        title_frame = tk.Frame(bar, bg=SURFACE)
        title_frame.pack(side="left", padx=25, pady=16)
        tk.Label(title_frame, text="ORCA Hand", bg=SURFACE, fg=LABEL, font=FONT_TITLE).pack(side="top", anchor="w")
        tk.Label(title_frame, text="Control Studio", bg=SURFACE, fg=ORANGE, font=FONT_SMALL).pack(side="top", anchor="w")

        right = tk.Frame(bar, bg=SURFACE)
        right.pack(side="right", padx=25, pady=22)

        self._status_dot = tk.Label(right, text="●", fg=GREEN, bg=SURFACE, font=("Helvetica", 16))
        self._status_dot.pack(side="left", padx=(0, 8))

        self._torque_btn = AppleButton(right, "Torque is ON", self._toggle_torque, bg_color=GREEN, parent_bg=SURFACE, width=150, height=40)
        self._torque_btn.pack(side="left")

        # 🌟 TRUE iOS SEGMENTED CONTROLS 🌟
        self._hand_seg = AppleSegmentedControl(bar, ["left", "right"], self._hand, self._switch_hand, width=150, height=36, parent_bg=SURFACE)
        self._hand_seg.pack(side="left", padx=(50, 15), pady=24)

        self._mode_seg = AppleSegmentedControl(bar, ["Joints", "Fingers", "Hand"], self._ctrl_mode, self._switch_mode, width=280, height=36, parent_bg=SURFACE)
        self._mode_seg.pack(side="left", padx=15, pady=24)

        tk.Frame(self.root, bg=SEP, height=1).pack(fill="x")

    def _build_telemetry(self):
        tel_panel = tk.Frame(self.main_container, bg=PANEL_BG, width=260)
        tel_panel.pack(side="left", fill="y")
        tel_panel.pack_propagate(False)

        tk.Label(tel_panel, text="LIVE TELEMETRY", bg=PANEL_BG, fg=LABEL2, font=FONT_SMALL).pack(pady=(25, 12))
        
        hdr = tk.Frame(tel_panel, bg=PANEL_BG)
        hdr.pack(fill="x", padx=12, pady=(0, 6))
        tk.Label(hdr, text="ID", width=3, bg=PANEL_BG, fg=LABEL2, font=FONT_MONO, anchor="w").pack(side="left")
        tk.Label(hdr, text="POS", width=5, bg=PANEL_BG, fg=LABEL2, font=FONT_MONO, anchor="e").pack(side="left")
        tk.Label(hdr, text="VEL", width=4, bg=PANEL_BG, fg=LABEL2, font=FONT_MONO, anchor="e").pack(side="left")
        tk.Label(hdr, text="LD", width=4, bg=PANEL_BG, fg=LABEL2, font=FONT_MONO, anchor="e").pack(side="left")
        tk.Label(hdr, text="TMP", width=4, bg=PANEL_BG, fg=LABEL2, font=FONT_MONO, anchor="e").pack(side="left")

        for mid in self.ids:
            row = tk.Frame(tel_panel, bg=PANEL_BG)
            row.pack(fill="x", padx=12, pady=3)
            
            tk.Label(row, text=f"{mid:02d}", width=3, bg=PANEL_BG, fg=ORANGE, font=FONT_MONO, anchor="w").pack(side="left")
            
            v_pos = tk.StringVar(value="----")
            v_vel = tk.StringVar(value="---")
            v_ld  = tk.StringVar(value="---")
            v_tmp = tk.StringVar(value="--")
            
            l_pos = tk.Label(row, textvariable=v_pos, width=5, bg=PANEL_BG, fg=LABEL, font=FONT_MONO, anchor="e")
            l_vel = tk.Label(row, textvariable=v_vel, width=4, bg=PANEL_BG, fg=LABEL, font=FONT_MONO, anchor="e")
            l_ld  = tk.Label(row, textvariable=v_ld, width=4, bg=PANEL_BG, fg=LABEL, font=FONT_MONO, anchor="e")
            l_tmp = tk.Label(row, textvariable=v_tmp, width=4, bg=PANEL_BG, fg=LABEL, font=FONT_MONO, anchor="e")
            
            l_pos.pack(side="left")
            l_vel.pack(side="left")
            l_ld.pack(side="left")
            l_tmp.pack(side="left")
            
            self._tel_vars[mid] = {'pos': v_pos, 'vel': v_vel, 'ld': v_ld, 'tmp': v_tmp, 
                                   'l_vel': l_vel, 'l_ld': l_ld, 'l_tmp': l_tmp}

        tk.Frame(self.main_container, bg=SEP, width=1).pack(side="left", fill="y")

    def _build_columns(self):
        if self._cols_frame: self._cols_frame.destroy()

        outer = tk.Frame(self.controls_container, bg=BG)
        outer.pack(fill="both", expand=True, padx=25, pady=25)
        self._cols_frame = outer

        self._sliders.clear()
        self._lbl_vars.clear()

        jid = JOINT_ID[self._hand]

        if self._ctrl_mode == "Joints":
            for col_idx, (finger, joints) in enumerate(LAYOUT[self._hand]):
                col = tk.Frame(outer, bg=BG)
                col.grid(row=0, column=col_idx, sticky="ns", padx=8)
                tk.Label(col, text=finger.upper(), bg=BG, fg=LABEL2, font=FONT_SUB, anchor="w").pack(fill="x", pady=(0, 12))

                for jname in joints:
                    mid = jid.get(jname)
                    if mid is None: continue

                    c = self.calib.get(mid, {"min": 0, "max": 4095, "center": 2047})
                    calibrated = not (c["min"] == 0 and c["max"] == 4095)
                    card = tk.Frame(col, bg=SURFACE, padx=12, pady=12)
                    card.pack(fill="x", pady=6)
                    
                    name_row = tk.Frame(card, bg=SURFACE)
                    name_row.pack(fill="x")
                    tk.Label(name_row, text=jname.replace("_", " ").title(), bg=SURFACE, fg=LABEL, font=FONT_BODY, anchor="w").pack(side="left")
                    
                    id_bg = ORANGE if calibrated else SEP
                    tk.Label(name_row, text=f"ID {mid}", bg=id_bg, fg=LABEL, font=FONT_SMALL, padx=5, pady=3).pack(side="right")

                    sl = AppleSlider(card, from_=c["min"], to=c["max"], width=150, height=26,
                                     command=lambda v, j=jname, m=mid: self._on_joint_slider(j, m, v))
                    sl.set(c["center"])
                    sl.pack(fill="x", pady=(10, 6))
                    self._sliders[jname] = sl

                    val_row = tk.Frame(card, bg=SURFACE)
                    val_row.pack(fill="x")
                    lbl_var = tk.StringVar(value=f"{c['center']:4d} | 50%")
                    tk.Label(val_row, textvariable=lbl_var, bg=SURFACE, fg=LABEL, font=FONT_MONO, anchor="w").pack(side="left")
                    tk.Label(val_row, text=f"[{c['min']} ↔ {c['max']}]", bg=SURFACE, fg=LABEL2, font=FONT_SMALL, anchor="e").pack(side="right")
                    self._lbl_vars[jname] = lbl_var

            outer.grid_columnconfigure(tuple(range(6)), weight=1)

        elif self._ctrl_mode == "Fingers":
            for col_idx, (finger, joints) in enumerate(LAYOUT[self._hand]):
                col = tk.Frame(outer, bg=BG)
                col.grid(row=0, column=col_idx, sticky="ns", padx=12)
                
                mids = [jid.get(j) for j in joints if jid.get(j)]
                if not mids: continue

                card = tk.Frame(col, bg=SURFACE, padx=16, pady=16)
                card.pack(fill="x", pady=40)
                
                name_row = tk.Frame(card, bg=SURFACE)
                name_row.pack(fill="x", pady=(0, 18))
                tk.Label(name_row, text=finger.upper(), bg=SURFACE, fg=LABEL, font=FONT_SUB, anchor="w").pack(side="left")
                
                id_str = ",".join(str(m) for m in mids)
                tk.Label(name_row, text=f"IDs {id_str}", bg=SEP, fg=LABEL, font=FONT_SMALL, padx=8, pady=3).pack(side="right")

                sl = AppleSlider(card, from_=0, to=100, width=170, height=30,
                                 command=lambda v, n=finger, m=mids: self._on_macro_slider(n, m, v))
                sl.set(50)
                sl.pack(fill="x", pady=(14, 8))
                self._sliders[finger] = sl

                val_row = tk.Frame(card, bg=SURFACE)
                val_row.pack(fill="x")
                v_lbl = tk.StringVar(value="50%")
                tk.Label(val_row, textvariable=v_lbl, bg=SURFACE, fg=LABEL, font=FONT_MONO, anchor="w").pack(side="left")
                tk.Label(val_row, text="[0 ↔ 100]", bg=SURFACE, fg=LABEL2, font=FONT_SMALL, anchor="e").pack(side="right")
                self._lbl_vars[finger] = v_lbl

            outer.grid_columnconfigure(tuple(range(6)), weight=1)

        elif self._ctrl_mode == "Hand":
            col = tk.Frame(outer, bg=BG)
            col.pack(expand=True)
            
            card = tk.Frame(col, bg=SURFACE, padx=40, pady=40)
            card.pack(pady=40)
            
            name_row = tk.Frame(card, bg=SURFACE)
            name_row.pack(fill="x", pady=(0, 15))
            tk.Label(name_row, text="WHOLE HAND CONTROL", bg=SURFACE, fg=LABEL, font=FONT_TITLE, anchor="w").pack(side="left")
            tk.Label(name_row, text=f"IDs 1-{TOTAL}", bg=ORANGE, fg=LABEL, font=FONT_SMALL, padx=10, pady=5).pack(side="right")
            
            all_mids = [m for m in jid.values()]
            sl = AppleSlider(card, from_=0, to=100, width=420, height=40,
                             command=lambda v, n="Hand", m=all_mids: self._on_macro_slider(n, m, v))
            sl.set(50)
            sl.pack(fill="x", pady=(25, 15))
            self._sliders["Hand"] = sl
            
            val_row = tk.Frame(card, bg=SURFACE)
            val_row.pack(fill="x")
            v_lbl = tk.StringVar(value="50%")
            tk.Label(val_row, textvariable=v_lbl, bg=SURFACE, fg=LABEL, font=("Menlo", 16, "bold"), anchor="w").pack(side="left")
            tk.Label(val_row, text="[0 ↔ 100]", bg=SURFACE, fg=LABEL2, font=FONT_SUB, anchor="e").pack(side="right")
            self._lbl_vars["Hand"] = v_lbl


    def _build_pose_bar(self):
        tk.Frame(self.controls_container, bg=SEP, height=1).pack(fill="x")
        bar = tk.Frame(self.controls_container, bg=SURFACE, height=90)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)

        inner = tk.Frame(bar, bg=SURFACE)
        inner.pack(side="left", padx=30, pady=25)
        tk.Label(inner, text="Presets:", bg=SURFACE, fg=LABEL2, font=FONT_SUB).pack(side="left", padx=(0, 18))

        for name in POSES:
            AppleButton(inner, name, lambda n=name: self._apply_pose(n), bg_color=ELEVATED, parent_bg=SURFACE, width=90, height=38).pack(side="left", padx=8)

        tk.Frame(bar, bg=SEP, width=1).pack(side="right", padx=25, fill="y", pady=20)
        AppleButton(bar, "Open All",  self._open_all,  bg_color=BLUE, parent_bg=SURFACE, width=110, height=42).pack(side="right", padx=10, pady=24)
        AppleButton(bar, "Close All", self._close_all, bg_color=RED,  parent_bg=SURFACE, width=110, height=42).pack(side="right", padx=10, pady=24)

    # ── Input Handlers ──
    def _on_joint_slider(self, jname: str, mid: int, raw: int):
        if getattr(self, '_is_reading', False): return
        self._last_user_interaction = time.time()
        
        pct = raw_to_pct(mid, raw, self.calib)
        if jname in self._lbl_vars:
            self._lbl_vars[jname].set(f"{raw:4d} | {pct:3.0f}%")
        with self._lock: self._pending[mid] = raw

    def _on_macro_slider(self, name: str, mids: list, pct: int):
        self._last_user_interaction = time.time()
        self._lbl_vars[name].set(f"{pct}%")
        with self._lock:
            for mid in mids:
                self._pending[mid] = pct_to_raw(mid, pct, self.calib)

    def _toggle_torque(self):
        self._torque_on = not self._torque_on
        self.bus.torque(self.ids, self._torque_on)
        if self._torque_on:
            self._torque_btn.update_colors(GREEN)
            self._torque_btn.update_text("Torque is ON")
            self._status_dot.config(fg=GREEN)
        else:
            self._torque_btn.update_colors(RED)
            self._torque_btn.update_text("Torque is OFF")
            self._status_dot.config(fg=RED)

    def _switch_hand(self, hand: str):
        if hand == self._hand: return
        self._hand = hand
        self._build_columns()

    def _switch_mode(self, mode: str):
        if mode == self._ctrl_mode: return
        self._ctrl_mode = mode
        self._build_columns()

    def _apply_pose(self, name: str):
        pose = POSES.get(name, {})
        jid  = JOINT_ID[self._hand]
        
        if self._ctrl_mode == "Joints":
            for jname, pct in pose.items():
                if jname in self._sliders and jid.get(jname):
                    self._sliders[jname].set(pct_to_raw(jid[jname], pct, self.calib))
        elif self._ctrl_mode == "Fingers":
            for s in self._sliders.values(): s.set(50)
        
        self._last_user_interaction = time.time()
        with self._lock:
            for jname, pct in pose.items():
                if mid := jid.get(jname): self._pending[mid] = pct_to_raw(mid, pct, self.calib)

    def _open_all(self): self._apply_pose("Flat")
    def _close_all(self): self._apply_pose("Fist")

    # ── Background Loops ──
    def _start_threads(self):
        threading.Thread(target=self._send_loop, daemon=True).start()
        threading.Thread(target=self._read_loop, daemon=True).start()

    def _send_loop(self):
        dt = 1.0 / SEND_HZ
        while True:
            with self._lock:
                pending = dict(self._pending)
                self._pending.clear()
            if pending and self._torque_on:
                try: self.bus.sync_write(pending)
                except Exception: pass
            time.sleep(dt)

    def _read_loop(self):
        dt = 1.0 / POLL_HZ
        while True:
            try:
                data = self.bus.sync_read_all(self.ids)
                user_is_active = (time.time() - self._last_user_interaction) < 1.0

                for mid, metrics in data.items():
                    if mid in self._tel_vars:
                        v = self._tel_vars[mid]
                        v['pos'].set(f"{metrics['pos']:4d}")
                        v['vel'].set(f"{metrics['vel']:3d}")
                        v['ld'].set(f"{metrics['load']:3d}")
                        v['tmp'].set(f"{metrics['temp']:2d}°")
                        
                        v['l_vel'].config(fg=GREEN if abs(metrics['vel']) > 0 else LABEL)
                        v['l_ld'].config(fg=RED if metrics['load'] > 600 else ORANGE if metrics['load'] > 300 else LABEL)
                        v['l_tmp'].config(fg=RED if metrics['temp'] > 65 else ORANGE if metrics['temp'] > 50 else LABEL)

                    if self._ctrl_mode == "Joints" and not user_is_active:
                        jid_rev = {v: k for k, v in JOINT_ID[self._hand].items()}
                        jname = jid_rev.get(mid)
                        if jname and jname in self._sliders:
                            if abs(self._sliders[jname].get() - metrics['pos']) > 15:
                                self._is_reading = True
                                self._sliders[jname].set(metrics['pos'])
                                pct = raw_to_pct(mid, metrics['pos'], self.calib)
                                self._lbl_vars[jname].set(f"{metrics['pos']:4d} | {pct:3.0f}%")
                                self._is_reading = False

            except Exception: pass
            time.sleep(dt)

    def on_close(self):
        self.bus.close()
        self.root.destroy()

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=str)
    parser.add_argument("--hand", choices=["left", "right"], default="left")
    args = parser.parse_args()

    port  = args.port or detect_port()
    calib = _load_calibration()
    
    bus = Bus(port)
    root = tk.Tk()
    app  = SliderApp(root, bus, calib, args.hand)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())