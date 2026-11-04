"""Setup Panel — Motor IDs, Joint Limits, EEPROM Registers, Set Home."""

import sys
import time
import tkinter as tk
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from style import *

# Common register table for quick access
REGISTERS = [
    ("ID",              5,  1, "Motor bus ID (1–253)"),
    ("Baud Rate",       6,  1, "0=1M 1=500K 2=250K"),
    ("Min Angle Limit", 9,  2, "Raw position lower limit"),
    ("Max Angle Limit", 11, 2, "Raw position upper limit"),
    ("Max Torque",      16, 2, "Startup torque limit (0–1000)"),
    ("Operating Mode",  33, 1, "0=Servo 1=Wheel"),
    ("Torque Enable",   40, 1, "0=Off 1=On"),
    ("Present Position",56, 2, "Current raw position (read-only)"),
    ("Present Temp",    63, 1, "Temperature °C (read-only)"),
]


class SetupPanel(tk.Frame):

    SUB_PANELS = ["Motor IDs", "Joint Limits", "Center", "Servo Limits",
                  "Registers", "Set Home"]

    def __init__(self, parent, bus):
        super().__init__(parent, bg=BG)
        self.bus     = bus
        self._content: tk.Frame | None = None

        nav = tk.Frame(self, bg=BG, pady=PAD_SM)
        nav.pack(fill="x", padx=PAD)
        self._seg = AppleSegmentedControl(nav, self.SUB_PANELS, self.SUB_PANELS[0], self._show, parent_bg=BG)
        self._seg.pack(side="left")
        self._show(self.SUB_PANELS[0])

    def _show(self, name):
        if self._content:
            self._content.destroy()
        self._content = tk.Frame(self, bg=BG)
        self._content.pack(fill="both", expand=True)
        {
            "Motor IDs":    self._build_ids,
            "Joint Limits": self._build_limits,
            "Center":       self._build_center,
            "Servo Limits": self._build_servo_limits,
            "Registers":    self._build_registers,
            "Set Home":     self._build_home,
        }[name]()

    # ── Shared helpers ────────────────────────────────────────────────────────

    def _motor_ids(self) -> list[int]:
        """Configured motor IDs (all 17), or 1..17 before a hand is connected."""
        if self.bus.id_joint:
            return sorted(self.bus.id_joint.keys())
        return list(range(1, 18))

    def _two_col(self, parent):
        """Two side-by-side columns for compact per-motor lists; returns the
        (left, right) frames. Callers put the first half of the rows in left."""
        wrap = tk.Frame(parent, bg=BG)
        wrap.pack(fill="both", expand=True, pady=PAD_SM)
        left = tk.Frame(wrap, bg=BG)
        right = tk.Frame(wrap, bg=BG)
        left.pack(side="left", fill="both", expand=True, padx=(0, 8))
        right.pack(side="left", fill="both", expand=True, padx=(8, 0))
        return left, right

    def _scroll_table(self, parent, headers):
        """A scrollable SURFACE table with a header row; returns the inner frame."""
        frame = tk.Frame(parent, bg=SURFACE)
        frame.pack(fill="both", expand=True, pady=PAD_SM)
        hdr = tk.Frame(frame, bg=ELEVATED)
        hdr.pack(fill="x")
        for text, w in headers:
            tk.Label(hdr, text=text, bg=ELEVATED, fg=LABEL2, font=F_CAPTION,
                     width=w, anchor="w").pack(side="left", padx=4, pady=4)
        canvas = tk.Canvas(frame, bg=SURFACE, highlightthickness=0)
        scroll = tk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scroll.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        inner = tk.Frame(canvas, bg=SURFACE)
        canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")))
        canvas.configure(yscrollcommand=scroll.set)
        return inner

    # ── Motor IDs ─────────────────────────────────────────────────────────────

    def _build_ids(self):
        f = tk.Frame(self._content, bg=BG)
        f.pack(fill="both", expand=True, padx=PAD_LG, pady=PAD)

        lbl(f, "Motor ID Assignment", fg=LABEL, font=F_TITLE2).pack(anchor="w")
        lbl(f, "Every motor and its current bus ID. Type a new ID and press Change.\n"
               "Duplicate or already-in-use IDs are rejected. Power-cycle + reconnect after.",
            fg=LABEL2, font=F_CALLOUT).pack(anchor="w", pady=(2, PAD))

        self._id_status = lbl(f, "", fg=LABEL2, font=F_CALLOUT)
        self._id_status.pack(anchor="w", pady=(0, PAD_SM))

        self._id_new_vars: dict[int, tk.StringVar] = {}
        left, right = self._two_col(f)
        ids = self._motor_ids()
        half = (len(ids) + 1) // 2
        for i, mid in enumerate(ids):
            col = left if i < half else right
            jname = self.bus.id_joint.get(mid, f"ID {mid}")
            row = tk.Frame(col, bg=SURFACE)
            row.pack(fill="x", pady=2, padx=2)
            tk.Label(row, text=str(mid), bg=SURFACE, fg=ORANGE, font=F_MONO,
                     width=3, anchor="w").pack(side="left", padx=4)
            tk.Label(row, text=jname, bg=SURFACE, fg=LABEL, font=F_CALLOUT,
                     width=11, anchor="w").pack(side="left", padx=4)
            var = tk.StringVar(value=str(mid))
            tk.Entry(row, textvariable=var, width=5, bg=ELEVATED, fg=LABEL,
                     font=F_MONO, relief="flat", insertbackground=LABEL,
                     justify="center").pack(side="left", padx=4)
            accent_btn(row, "Change", lambda m=mid, v=var: self._change_id(m, v),
                       color=ORANGE, font=F_CALLOUT, px=8, py=4).pack(side="left", padx=6)
            self._id_new_vars[mid] = var

    def _change_id(self, cur: int, var: tk.StringVar):
        s = var.get().strip()
        try:
            new = int(s)
        except ValueError:
            self._id_status.config(text=f"Motor {cur}: '{s}' is not a valid ID.", fg=RED)
            return
        if new == cur:
            self._id_status.config(text=f"Motor {cur} already has ID {cur}.", fg=YELLOW)
            return
        if not (1 <= new <= 253):
            self._id_status.config(text="New ID must be 1–253.", fg=RED)
            return
        # Reject IDs already assigned to another motor in the chain…
        if new in (set(self._motor_ids()) - {cur}):
            owner = self.bus.id_joint.get(new, f"ID {new}")
            self._id_status.config(
                text=f"ID {new} is already used by {owner} — pick a different ID.", fg=RED)
            return
        # …and any motor physically answering at that ID right now.
        if self.bus.ping(new):
            self._id_status.config(
                text=f"ID {new} is already live on the bus — pick a different ID.", fg=RED)
            return
        if not self.bus.ping(cur):
            self._id_status.config(text=f"Motor {cur} is not responding.", fg=RED)
            return
        self.bus.write_register(cur, 55, 0, 1)         # unlock EEPROM
        time.sleep(0.02)
        ok = self.bus.write_register(cur, 5, new, 1)   # reg 5 = ID
        time.sleep(0.05)
        self.bus.write_register(new, 55, 1, 1)         # lock EEPROM
        if ok:
            var.set(str(new))
            self._id_status.config(
                text=f"✓ Motor {cur} → ID {new}. Power-cycle, then reconnect.", fg=GREEN)
            self.bus.log(f"Motor ID changed: {cur} → {new}")
        else:
            self._id_status.config(text="Change failed. Check power and wiring.", fg=RED)

    # ── Joint Limits ──────────────────────────────────────────────────────────

    def _build_limits(self):
        f = tk.Frame(self._content, bg=BG)
        f.pack(fill="both", expand=True, padx=PAD_LG, pady=PAD)

        lbl(f, "Joint Limits (Calibration)", fg=LABEL, font=F_TITLE2).pack(anchor="w")
        lbl(f, "View and edit calibrated min/max for each motor.  Save writes joint_limits.json.\n"
               "↺ re-centers the joint at its current pose (→2048) keeping the same range; "
               "values must stay in 0–4095.",
            fg=LABEL2, font=F_CALLOUT).pack(anchor="w", pady=(2, PAD_SM))

        # Scrollable table
        frame = tk.Frame(f, bg=SURFACE)
        frame.pack(fill="both", expand=True)

        hdr = tk.Frame(frame, bg=ELEVATED)
        hdr.pack(fill="x")
        for col, w, text in [("ID", 4, "ID"), ("Joint", 14, "Joint"),
                               ("Min", 7, "Min"), ("Max", 7, "Max"),
                               ("Center", 7, "Center"), ("", 6, "")]:
            tk.Label(hdr, text=text, bg=ELEVATED, fg=LABEL2,
                     font=F_CAPTION, width=w, anchor="center").pack(side="left")

        self._limit_entries: dict[int, dict] = {}
        canvas = tk.Canvas(frame, bg=SURFACE, highlightthickness=0)
        scroll = tk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scroll.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        inner = tk.Frame(canvas, bg=SURFACE)
        canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")))
        canvas.configure(yscrollcommand=scroll.set)

        for mid in range(1, 18):
            jname = self.bus.id_joint.get(mid, f"ID {mid}")
            c     = self.bus.calib.get(mid, {"min": 0, "max": 4095, "center": 2047})
            row   = tk.Frame(inner, bg=SURFACE)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=str(mid), bg=SURFACE, fg=LABEL2,
                     font=F_MONO_SM, width=4, anchor="center").pack(side="left")
            tk.Label(row, text=jname, bg=SURFACE, fg=LABEL,
                     font=F_CALLOUT, width=14, anchor="w").pack(side="left")
            entries = {}
            for key, val in [("min", c.get("min", 0)),
                              ("max", c.get("max", 4095)),
                              ("center", c.get("center", 2047))]:
                var = tk.StringVar(value=str(val))
                e = tk.Entry(row, textvariable=var, width=7,
                             bg=ELEVATED, fg=LABEL, font=F_MONO_SM,
                             relief="flat", insertbackground=LABEL, justify="center")
                e.pack(side="left", padx=1)
                entries[key] = var
            self._limit_entries[mid] = entries
            accent_btn(row, "↺", lambda m=mid: self._recenter_from_motor(m),
                       color=BLUE, font=F_MONO_SM, px=8, py=2).pack(side="left", padx=4)

        self._limits_status = lbl(f, "", fg=LABEL2, font=F_CALLOUT)
        self._limits_status.pack(anchor="w", pady=(PAD_SM, 0))

        btn_row = tk.Frame(f, bg=BG)
        btn_row.pack(fill="x", pady=PAD_SM)
        accent_btn(btn_row, "Save to joint_limits.json", self._save_limits,
                   color=ORANGE).pack(side="left", padx=12, pady=6)
        ghost_btn(btn_row, "Reload from file", self._reload_limits).pack(side="left", padx=12, pady=6)

    def _recenter_from_motor(self, mid: int):
        """Re-center this joint at its current pose WITHOUT changing its range.

        Latches the present pose as the servo midpoint (2048) on the hardware,
        then re-clips the calibrated limits to ``2048 ± span/2`` keeping the
        SAME span — so the toggle range the sliders / poses use is unchanged,
        just re-anchored on the new center and always inside the 0–4095 servo
        limits (2048 is dead-centre, so any span ≤ 4095 fits). This is the same
        convention as the Center tab; the result is saved immediately because
        the encoder zero has physically moved.
        """
        if not self.bus.ping(mid):
            self._set_limits_status(f"ID {mid} not responding — connect the "
                                    f"hand first.", RED)
            return
        # Keep the existing toggle range. Prefer the (possibly edited) entry
        # span; fall back to the saved calibration; never invent a new width.
        span = None
        try:
            mn = int(self._limit_entries[mid]["min"].get())
            mx = int(self._limit_entries[mid]["max"].get())
            if 1 <= abs(mx - mn) <= 4095:
                span = abs(mx - mn)
        except (ValueError, KeyError):
            pass
        if span is None:
            c = self.bus.calib.get(mid, {"min": 0, "max": 4095})
            span = abs(int(c["max"]) - int(c["min"])) or 4095
        # 1) latch present pose as the servo midpoint 2048
        if not self.bus.center_offset(mid):
            self._set_limits_status(f"ID {mid}: failed to set midpoint.", RED)
            return
        # 2) rebuild the calibrated range symmetric around 2048 (span preserved)
        lo, hi = 2048 - span // 2, 2048 + span // 2
        self.bus.calib[mid] = {"min": lo, "max": hi, "center": 2048}
        self.bus.save_calib()
        self._limit_entries[mid]["min"].set(str(lo))
        self._limit_entries[mid]["max"].set(str(hi))
        self._limit_entries[mid]["center"].set("2048")
        jname = self.bus.id_joint.get(mid, mid)
        self._set_limits_status(
            f"↺ {jname}: centred at 2048 → range {lo}–{hi} "
            f"(span {span} unchanged, saved).", GREEN)

    def _set_limits_status(self, text, fg):
        w = getattr(self, "_limits_status", None)
        try:
            if w is not None and w.winfo_exists():
                w.config(text=text, fg=fg)
        except Exception:
            pass

    def _save_limits(self):
        bad = []
        for mid, entries in self._limit_entries.items():
            try:
                mn = int(entries["min"].get())
                mx = int(entries["max"].get())
                ct = int(entries["center"].get())
            except ValueError:
                bad.append(mid)
                continue
            # Reject physically-impossible limits so a typo / stale value can't
            # re-poison the file. The single-turn encoder is strictly 0–4095.
            if not (0 <= mn <= 4095 and 0 <= mx <= 4095 and mn < mx
                    and mn <= ct <= mx):
                bad.append(mid)
                continue
            self.bus.calib[mid] = {"min": mn, "max": mx, "center": ct}
        if bad:
            names = ", ".join(f"{self.bus.id_joint.get(m, m)}" for m in bad)
            self._set_limits_status(
                f"Skipped invalid rows ({names}); must be 0–4095 with "
                f"min<center<max. Use ↺ to re-center from the motor.", RED)
        self.bus.save_calib()
        if not bad:
            self._set_limits_status("✓ Saved to joint_limits.json.", GREEN)

    def _reload_limits(self):
        self.bus.reload_calib()
        self._show("Joint Limits")

    # ── Servo angle limits (the hard EEPROM ceiling) ────────────────────────────

    ADDR_MIN_ANGLE = 9    # 2 bytes
    ADDR_MAX_ANGLE = 11   # 2 bytes
    ADDR_LOCK = 55        # 1 byte: 0 = unlock EEPROM, 1 = lock

    def _build_servo_limits(self):
        f = tk.Frame(self._content, bg=BG)
        f.pack(fill="both", expand=True, padx=PAD_LG, pady=PAD)

        lbl(f, "Servo Angle Limits", fg=LABEL, font=F_TITLE2).pack(anchor="w")
        lbl(f, "The servo's built-in hard min/max position (EEPROM reg 9 / 11). Widen these\n"
               "to remove a ceiling that clips a joint. 0–4095 = full single turn. Write\n"
               "unlocks EEPROM, sets both, and re-locks. Power-cycle is NOT required.",
            fg=LABEL2, font=F_CALLOUT).pack(anchor="w", pady=(2, PAD))

        top = tk.Frame(f, bg=BG)
        top.pack(fill="x", pady=(0, PAD_SM))
        accent_btn(top, "↻  Read All", self._read_all_servo_limits,
                   color=BLUE).pack(side="left", padx=(0, 10), pady=4)
        accent_btn(top, "Set All Full (0–4095)", self._servo_limits_full,
                   color=ORANGE).pack(side="left", padx=10, pady=4)
        self._servo_status = lbl(top, "", fg=LABEL2, font=F_CALLOUT)
        self._servo_status.pack(side="left", padx=12)

        self._servo_min_vars: dict[int, tk.StringVar] = {}
        self._servo_max_vars: dict[int, tk.StringVar] = {}
        left, right = self._two_col(f)
        ids = self._motor_ids()
        half = (len(ids) + 1) // 2
        for i, mid in enumerate(ids):
            col = left if i < half else right
            jname = self.bus.id_joint.get(mid, f"ID {mid}")
            row = tk.Frame(col, bg=SURFACE)
            row.pack(fill="x", pady=2, padx=2)
            tk.Label(row, text=str(mid), bg=SURFACE, fg=ORANGE, font=F_MONO,
                     width=3, anchor="w").pack(side="left", padx=4)
            tk.Label(row, text=jname, bg=SURFACE, fg=LABEL, font=F_CALLOUT,
                     width=11, anchor="w").pack(side="left", padx=4)
            minv, maxv = tk.StringVar(value="—"), tk.StringVar(value="—")
            for var in (minv, maxv):
                tk.Entry(row, textvariable=var, width=6, bg=ELEVATED, fg=LABEL,
                         font=F_MONO, relief="flat", insertbackground=LABEL,
                         justify="center").pack(side="left", padx=3)
            accent_btn(row, "Write", lambda m=mid: self._write_servo_limit(m),
                       color=ELEVATED, font=F_CALLOUT, px=8, py=4).pack(side="left", padx=6)
            self._servo_min_vars[mid] = minv
            self._servo_max_vars[mid] = maxv
        self._read_all_servo_limits()

    def _read_all_servo_limits(self):
        n = 0
        for mid in self._motor_ids():
            mn = self.bus.read_register(mid, self.ADDR_MIN_ANGLE, 2)
            mx = self.bus.read_register(mid, self.ADDR_MAX_ANGLE, 2)
            if mn is not None:
                self._servo_min_vars[mid].set(str(mn)); n += 1
            if mx is not None:
                self._servo_max_vars[mid].set(str(mx))
        self._servo_status.config(text=f"Read limits from {n} motor(s).", fg=LABEL2)

    def _write_servo_limit(self, mid: int):
        try:
            mn = int(self._servo_min_vars[mid].get())
            mx = int(self._servo_max_vars[mid].get())
        except ValueError:
            self._servo_status.config(text=f"ID {mid}: limits must be numbers.", fg=RED)
            return
        if not (0 <= mn <= 4095 and 0 <= mx <= 4095):
            self._servo_status.config(text="Limits must be 0–4095.", fg=RED)
            return
        if not self.bus.ping(mid):
            self._servo_status.config(text=f"ID {mid} not responding.", fg=RED)
            return
        ok = self._apply_servo_limit(mid, mn, mx)
        self._servo_status.config(
            text=(f"✓ ID {mid} limits → {mn}–{mx}." if ok else f"ID {mid}: write failed."),
            fg=GREEN if ok else RED)
        if ok:
            self.bus.log(f"Servo limits ID {mid}: {mn}–{mx}")

    def _apply_servo_limit(self, mid: int, mn: int, mx: int) -> bool:
        try:
            self.bus.write_register(mid, self.ADDR_LOCK, 0, 1)       # unlock EEPROM
            time.sleep(0.02)
            ok1 = self.bus.write_register(mid, self.ADDR_MIN_ANGLE, mn, 2)
            time.sleep(0.02)
            ok2 = self.bus.write_register(mid, self.ADDR_MAX_ANGLE, mx, 2)
            time.sleep(0.02)
            self.bus.write_register(mid, self.ADDR_LOCK, 1, 1)       # re-lock EEPROM
            return bool(ok1 and ok2)
        except Exception:
            return False

    def _servo_limits_full(self):
        ids = [m for m in self._motor_ids() if self.bus.ping(m)]
        done = 0
        for m in ids:
            if self._apply_servo_limit(m, 0, 4095):
                self._servo_min_vars[m].set("0")
                self._servo_max_vars[m].set("4095")
                done += 1
            time.sleep(0.02)
        self._servo_status.config(
            text=f"✓ Set {done}/{len(ids)} motors to full range 0–4095.",
            fg=GREEN if done == len(ids) else YELLOW)
        self.bus.log(f"Servo limits: {done}/{len(ids)} set to full 0–4095")

    # ── Registers ─────────────────────────────────────────────────────────────

    def _build_registers(self):
        f = tk.Frame(self._content, bg=BG)
        f.pack(fill="both", expand=True, padx=PAD_LG, pady=PAD)

        lbl(f, "EEPROM / SRAM Register Access", fg=LABEL, font=F_TITLE2).pack(anchor="w")
        lbl(f, "Read or write any register on a specific motor.",
            fg=LABEL2, font=F_CALLOUT).pack(anchor="w", pady=(2, PAD))

        ctrl = card(f)
        ctrl.pack(fill="x", pady=PAD_SM)

        row1 = tk.Frame(ctrl, bg=SURFACE)
        row1.pack(fill="x", pady=PAD_SM)

        for label_text, width, default in [
            ("Motor ID", 5, "1"), ("Register", 5, "56"), ("Bytes", 3, "2")
        ]:
            col = tk.Frame(row1, bg=SURFACE)
            col.pack(side="left", padx=12, pady=6)
            tk.Label(col, text=label_text, bg=SURFACE,
                     fg=LABEL2, font=F_CAPTION).pack(anchor="w")
        self._reg_mid = tk.StringVar(value="1")
        self._reg_addr= tk.StringVar(value="56")
        self._reg_n   = tk.StringVar(value="2")
        self._reg_val = tk.StringVar(value="")
        self._reg_out = tk.StringVar(value="")

        # Rebuild properly
        for child in row1.winfo_children():
            child.destroy()

        fields = [("Motor ID", self._reg_mid, 5),
                  ("Register", self._reg_addr, 6),
                  ("Bytes (1/2)", self._reg_n, 4)]
        for label_text, var, w in fields:
            col = tk.Frame(row1, bg=SURFACE)
            col.pack(side="left", padx=12, pady=6)
            tk.Label(col, text=label_text, bg=SURFACE,
                     fg=LABEL2, font=F_CAPTION).pack(anchor="w")
            tk.Entry(col, textvariable=var, width=w, bg=ELEVATED,
                     fg=LABEL, font=F_MONO, relief="flat",
                     insertbackground=LABEL).pack()

        btn_row = tk.Frame(ctrl, bg=SURFACE)
        btn_row.pack(fill="x", pady=(0, PAD_SM))
        accent_btn(btn_row, "Read", self._read_reg, color=BLUE).pack(side="left", padx=12, pady=6)
        col = tk.Frame(btn_row, bg=SURFACE)
        col.pack(side="left", padx=12, pady=6)
        tk.Label(col, text="Write value", bg=SURFACE,
                 fg=LABEL2, font=F_CAPTION).pack(anchor="w")
        tk.Entry(col, textvariable=self._reg_val, width=8, bg=ELEVATED,
                 fg=LABEL, font=F_MONO, relief="flat",
                 insertbackground=LABEL).pack()
        accent_btn(btn_row, "Write", self._write_reg, color=ORANGE).pack(side="left", padx=12, pady=6)

        out = tk.Frame(f, bg=SURFACE, padx=PAD_SM, pady=PAD_SM)
        out.pack(fill="x", pady=4)
        tk.Label(out, textvariable=self._reg_out, bg=SURFACE,
                 fg=GREEN, font=F_MONO, anchor="w").pack(fill="x")

        # Quick reference table
        divider(f).pack(fill="x", pady=PAD)
        lbl(f, "Common Registers", fg=LABEL2, font=F_FOOTNOTE).pack(anchor="w")
        for name, addr, nbytes, desc in REGISTERS:
            r = tk.Frame(f, bg=BG)
            r.pack(fill="x", pady=1)
            tk.Label(r, text=f"  {addr:>3}", bg=BG, fg=ORANGE,
                     font=F_MONO_SM, width=6).pack(side="left")
            tk.Label(r, text=f"{name:<20}", bg=BG, fg=LABEL,
                     font=F_CALLOUT, width=20, anchor="w").pack(side="left")
            tk.Label(r, text=desc, bg=BG, fg=LABEL3,
                     font=F_CAPTION, anchor="w").pack(side="left")

    def _read_reg(self):
        try:
            mid  = int(self._reg_mid.get())
            addr = int(self._reg_addr.get())
            n    = int(self._reg_n.get())
        except ValueError:
            self._reg_out.set("Invalid values"); return
        val = self.bus.read_register(mid, addr, n)
        if val is None:
            self._reg_out.set(f"ID {mid} reg {addr}: no response")
        else:
            self._reg_out.set(f"ID {mid}  reg {addr}  = {val}  (0x{val:04X})")

    def _write_reg(self):
        try:
            mid  = int(self._reg_mid.get())
            addr = int(self._reg_addr.get())
            n    = int(self._reg_n.get())
            val  = int(self._reg_val.get())
        except ValueError:
            self._reg_out.set("Invalid values"); return
        ok = self.bus.write_register(mid, addr, val, n)
        self._reg_out.set(f"Write {'OK' if ok else 'FAILED'}: ID {mid} reg {addr} = {val}")
        self.bus.log(f"Write reg {addr}={val} on ID {mid}: {'OK' if ok else 'FAIL'}")

    # ── Center joint range ──────────────────────────────────────────────────────

    def _build_center(self):
        f = tk.Frame(self._content, bg=BG)
        f.pack(fill="both", expand=True, padx=PAD_LG, pady=PAD)

        lbl(f, "Center Joint Range", fg=LABEL, font=F_TITLE2).pack(anchor="w")
        lbl(f, "Servos have a hard tick limit (0/4095). If a joint's range sits near it, the\n"
               "motion gets clipped. Move a joint to its MECHANICAL CENTER, then Center: the\n"
               "servo midpoint (2048) is set at that pose and the calibrated range is rebuilt\n"
               "as 2048 ± span/2, so the joint's min/max always stay inside the servo limits.",
            fg=LABEL2, font=F_CALLOUT).pack(anchor="w", pady=(2, PAD))

        top = tk.Frame(f, bg=BG)
        top.pack(fill="x", pady=(0, PAD_SM))
        accent_btn(top, "⌖  Center All Motors", self._center_all,
                   color=RED).pack(side="left", padx=(0, 12), pady=4)
        self._center_status = lbl(top, "", fg=LABEL2, font=F_CALLOUT)
        self._center_status.pack(side="left")

        self._center_pos_lbls: dict[int, tk.StringVar] = {}
        self._center_rng_lbls: dict[int, tk.StringVar] = {}
        left, right = self._two_col(f)
        ids = self._motor_ids()
        half = (len(ids) + 1) // 2
        for i, mid in enumerate(ids):
            col = left if i < half else right
            jname = self.bus.id_joint.get(mid, f"ID {mid}")
            span = self._joint_span(mid)
            row = tk.Frame(col, bg=SURFACE)
            row.pack(fill="x", pady=2, padx=2)
            tk.Label(row, text=str(mid), bg=SURFACE, fg=ORANGE, font=F_MONO,
                     width=3, anchor="w").pack(side="left", padx=4)
            tk.Label(row, text=jname, bg=SURFACE, fg=LABEL, font=F_CALLOUT,
                     width=11, anchor="w").pack(side="left", padx=4)
            posv = tk.StringVar(value="—")
            tk.Label(row, textvariable=posv, bg=SURFACE, fg=LABEL2, font=F_MONO_SM,
                     width=5, anchor="w").pack(side="left", padx=2)
            rngv = tk.StringVar(value=f"{2048 - span // 2}–{2048 + span // 2}")
            tk.Label(row, textvariable=rngv, bg=SURFACE, fg=LABEL3, font=F_MONO_SM,
                     width=10, anchor="w").pack(side="left", padx=2)
            accent_btn(row, "Center", lambda m=mid: self._center_motor(m),
                       color=ELEVATED, font=F_CALLOUT, px=8, py=4).pack(side="left", padx=6)
            self._center_pos_lbls[mid] = posv
            self._center_rng_lbls[mid] = rngv
        self._refresh_center_positions()

    def _joint_span(self, mid: int) -> int:
        """Theoretical tick span for a joint (from its current calibration)."""
        c = self.bus.calib.get(mid, {"min": 0, "max": 4095})
        span = abs(int(c["max"]) - int(c["min"]))
        return span if span > 0 else 4095

    def _refresh_center_positions(self):
        pos = self.bus.read_positions_direct()
        for m, v in self._center_pos_lbls.items():
            v.set(str(pos.get(m, "—")))

    def _center_motor(self, mid: int) -> bool:
        if not self.bus.ping(mid):
            self._center_status.config(text=f"ID {mid} not responding.", fg=RED)
            return False
        span = self._joint_span(mid)
        try:
            # 1) latch the present pose as the servo midpoint (2048)
            if not self.bus.center_offset(mid):
                self._center_status.config(text=f"ID {mid}: failed to set midpoint.", fg=RED)
                return False
            # 2) rebuild the calibrated range symmetric around 2048
            lo, hi = 2048 - span // 2, 2048 + span // 2
            self.bus.calib[mid] = {"min": lo, "max": hi, "center": 2048}
            self.bus.save_calib()
            self._center_status.config(
                text=f"✓ ID {mid} centred at 2048 → range {lo}–{hi}.", fg=GREEN)
            self.bus.log(f"Centred ID {mid}: range {lo}–{hi} (span {span})")
            self._refresh_center_positions()
            return True
        except Exception as e:
            self._center_status.config(text=f"Error: {e}", fg=RED)
            return False

    def _center_all(self):
        ids = [m for m in self._motor_ids() if self.bus.ping(m)]
        if not ids:
            self._center_status.config(text="No motors responding.", fg=YELLOW)
            return
        done = 0
        for m in ids:
            span = self._joint_span(m)
            try:
                if self.bus.center_offset(m):
                    self.bus.calib[m] = {"min": 2048 - span // 2,
                                         "max": 2048 + span // 2, "center": 2048}
                    done += 1
            except Exception:
                pass
            time.sleep(0.02)
        self.bus.save_calib()
        self._refresh_center_positions()
        self._center_status.config(
            text=f"✓ Centred {done}/{len(ids)} motors at 2048.",
            fg=GREEN if done == len(ids) else YELLOW)
        self.bus.log(f"Center All: {done}/{len(ids)} motors")

    # ── Set Home ──────────────────────────────────────────────────────────────

    def _build_home(self):
        f = tk.Frame(self._content, bg=BG)
        f.pack(fill="both", expand=True, padx=PAD_LG, pady=PAD)

        lbl(f, "Set Home / Zero Position", fg=LABEL, font=F_TITLE2).pack(anchor="w")
        lbl(f, "Move each motor to its home position, then burn the EEPROM offset so\n"
               "the current position reads as 2048 (center). Home one, or all at once.",
            fg=LABEL2, font=F_CALLOUT).pack(anchor="w", pady=(2, PAD))

        top = tk.Frame(f, bg=BG)
        top.pack(fill="x", pady=(0, PAD_SM))
        accent_btn(top, "⌂  Set Home — All Motors", self._set_home_all,
                   color=RED).pack(side="left", padx=(0, 12), pady=4)
        self._home_status = lbl(top, "", fg=LABEL2, font=F_CALLOUT)
        self._home_status.pack(side="left")

        inner = self._scroll_table(
            f, [("ID", 4), ("Joint", 14), ("Pos", 7), ("", 10)])
        self._home_pos_lbls: dict[int, tk.StringVar] = {}
        for mid in self._motor_ids():
            jname = self.bus.id_joint.get(mid, f"ID {mid}")
            row = tk.Frame(inner, bg=SURFACE)
            row.pack(fill="x", pady=2)
            tk.Label(row, text=str(mid), bg=SURFACE, fg=ORANGE, font=F_MONO,
                     width=4, anchor="w").pack(side="left", padx=4)
            tk.Label(row, text=jname, bg=SURFACE, fg=LABEL, font=F_CALLOUT,
                     width=14, anchor="w").pack(side="left", padx=4)
            posv = tk.StringVar(value="—")
            tk.Label(row, textvariable=posv, bg=SURFACE, fg=LABEL2, font=F_MONO_SM,
                     width=7, anchor="w").pack(side="left", padx=4)
            accent_btn(row, "Set Home", lambda m=mid: self._set_home(m),
                       color=ELEVATED, font=F_CALLOUT, px=10, py=4).pack(side="left", padx=8)
            self._home_pos_lbls[mid] = posv
        self._refresh_home_positions()

    def _refresh_home_positions(self):
        pos = self.bus.read_positions_direct()
        for m, v in self._home_pos_lbls.items():
            v.set(str(pos.get(m, "—")))

    def _set_home(self, mid: int):
        if not self.bus.ping(mid):
            self._home_status.config(text=f"ID {mid} not responding.", fg=RED)
            return
        try:
            if self.bus.center_offset(mid):
                self._home_status.config(text=f"✓ ID {mid} homed → 2048.", fg=GREEN)
                self.bus.log(f"Home set for ID {mid}")
                self._refresh_home_positions()
            else:
                self._home_status.config(text=f"ID {mid}: failed to set home.", fg=RED)
        except Exception as e:
            self._home_status.config(text=f"Error: {e}", fg=RED)

    def _set_home_all(self):
        ids = [m for m in self._motor_ids() if self.bus.ping(m)]
        if not ids:
            self._home_status.config(text="No motors responding.", fg=YELLOW)
            return
        done = 0
        for m in ids:
            try:
                if self.bus.center_offset(m):
                    done += 1
            except Exception:
                pass
            time.sleep(0.02)
        self._refresh_home_positions()
        self._home_status.config(
            text=f"✓ Homed {done}/{len(ids)} motors → 2048.",
            fg=GREEN if done == len(ids) else YELLOW)
        self.bus.log(f"Set Home All: {done}/{len(ids)} motors")
