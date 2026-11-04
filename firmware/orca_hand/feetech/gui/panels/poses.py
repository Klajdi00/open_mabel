"""Poses & Animations Panel — Record, replay, delete saved poses and animations."""

import sys
import threading
import time
import tkinter as tk
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from style import *


class PosesPanel(tk.Frame):

    SUB_PANELS = ["Poses", "Animations"]

    def __init__(self, parent, bus):
        super().__init__(parent, bg=BG)
        self.bus      = bus
        self._content: tk.Frame | None = None
        self._poses: dict   = {}
        self._anims: dict   = {}
        self._recording     = False
        self._anim_frames: list = []
        self._rec_thread: threading.Thread | None = None
        self._stop_rec      = threading.Event()

        nav = tk.Frame(self, bg=BG, pady=PAD_SM)
        nav.pack(fill="x", padx=PAD)
        AppleSegmentedControl(nav, self.SUB_PANELS, self.SUB_PANELS[0], self._show, parent_bg=BG).pack(side="left")
        self._show("Poses")

    def _show(self, name):
        if self._content:
            self._content.destroy()
        self._content = tk.Frame(self, bg=BG)
        self._content.pack(fill="both", expand=True)
        if name == "Poses":
            self._build_poses()
        else:
            self._build_anims()

    # ── Poses ─────────────────────────────────────────────────────────────────

    def _build_poses(self):
        self._poses = self.bus.load_poses()
        f = tk.Frame(self._content, bg=BG)
        f.pack(fill="both", expand=True, padx=PAD_LG, pady=PAD)

        lbl(f, "Saved Poses", fg=LABEL, font=F_TITLE2).pack(anchor="w")
        lbl(f, "Record the hand's current position. Replay moves smoothly to that pose.",
            fg=LABEL2, font=F_CALLOUT).pack(anchor="w", pady=(2, PAD_SM))

        # Record controls
        rec = card(f)
        rec.pack(fill="x", pady=4)

        rrow = tk.Frame(rec, bg=SURFACE)
        rrow.pack(fill="x")
        lbl(rrow, "Pose name:", fg=LABEL2, font=F_CALLOUT).pack(side="left", padx=PAD_SM)
        self._pose_name = tk.StringVar(value="")
        tk.Entry(rrow, textvariable=self._pose_name, width=18,
                 bg=ELEVATED, fg=LABEL, font=F_BODY,
                 relief="flat", insertbackground=LABEL).pack(side="left", padx=PAD_SM)
        accent_btn(rrow, "Capture Current Position",
                   self._record_pose, color=ORANGE).pack(side="left", padx=PAD_SM)
        lbl(rrow, "(torque off → pose by hand → capture)",
            fg=LABEL3, font=F_CAPTION).pack(side="left", padx=PAD_SM)

        self._pose_msg = lbl(f, "", fg=LABEL2, font=F_CALLOUT)
        self._pose_msg.pack(anchor="w", pady=2)

        divider(f).pack(fill="x", pady=PAD_SM)

        # Pose list
        lbl(f, f"Saved poses ({len(self._poses)})", fg=LABEL2,
            font=F_FOOTNOTE).pack(anchor="w")

        self._pose_list_frame = tk.Frame(f, bg=BG)
        self._pose_list_frame.pack(fill="both", expand=True, pady=PAD_SM)
        self._refresh_pose_list()

    def _refresh_pose_list(self):
        for w in self._pose_list_frame.winfo_children():
            w.destroy()
        for i, (name, data) in enumerate(self._poses.items()):
            row = tk.Frame(self._pose_list_frame, bg=SURFACE, pady=PAD_SM, padx=PAD_SM)
            row.pack(fill="x", pady=2)
            lbl(row, name, fg=LABEL, font=F_BODY).pack(side="left", padx=PAD_SM)
            lbl(row, f"{len(data)} joints", fg=LABEL3, font=F_CAPTION).pack(side="left")
            accent_btn(row, "▶  Play",
                       lambda n=name: self._play_pose(n),
                       color=BLUE, font=F_CALLOUT, px=10, py=5).pack(side="right", padx=4)
            ghost_btn(row, "Delete",
                      lambda n=name: self._delete_pose(n),
                      fg=RED, font=F_CAPTION, px=8, py=4).pack(side="right", padx=4)

    def _record_pose(self):
        name = self._pose_name.get().strip()
        if not name:
            self._pose_msg.config(text="Enter a pose name first.", fg=YELLOW)
            return
        if not self.bus.state.connected:
            self._pose_msg.config(text="Not connected.", fg=RED)
            return

        # Key poses by JOINT NAME (not motor id) in the semantic 0..1 scale
        # (0 = open, 100 % = closed, invert-aware). This makes a pose portable
        # across hands: the same "peace" plays on left or right because each side
        # maps the joint name to its own motor + calibration (a mirror).
        state = self.bus.state
        captured = {}
        for mid, ms in state.motors.items():
            if ms.raw is None:
                continue
            jname = self.bus.id_joint.get(mid)
            if not jname:
                continue
            captured[jname] = round(self.bus.raw_to_pct(mid, ms.raw) / 100.0, 4)

        self._poses[name] = captured
        self.bus.save_poses(self._poses)
        self._pose_msg.config(text=f"✓ Saved '{name}' ({len(captured)} joints).", fg=GREEN)
        self.bus.log(f"Pose recorded: {name}")
        self._refresh_pose_list()

    def _play_pose(self, name: str):
        pose = self._poses.get(name, {})
        targets = {}
        for key, norm in pose.items():
            if str(key).lstrip("-").isdigit():
                # Legacy pose keyed by motor id (same-hand only): raw-based map.
                mid = int(key)
                c = self.bus.calib.get(mid, {"min": 0, "max": 4095})
                lo, hi = int(c["min"]), int(c["max"])
                if mid in self.bus.state.motors:
                    targets[mid] = max(min(lo, hi),
                                       min(max(lo, hi), int(lo + norm * (hi - lo))))
            else:
                # Portable pose keyed by joint name: resolve to THIS hand's motor
                # and use the invert-aware percent so the physical pose matches.
                mid = self.bus.joint_id.get(key)
                if mid is not None and mid in self.bus.state.motors:
                    targets[mid] = self.bus.pct_to_raw(mid, norm * 100.0)
        if targets:
            self.bus.set_torque(True)
            self.bus.sync_write(targets, speed=2000, torque=1200)
            self.bus.log(f"Pose played: {name}")

    def _delete_pose(self, name: str):
        del self._poses[name]
        self.bus.save_poses(self._poses)
        self.bus.log(f"Pose deleted: {name}")
        self._refresh_pose_list()

    # ── Animations ────────────────────────────────────────────────────────────

    def _build_anims(self):
        self._anims = self.bus.load_animations()
        f = tk.Frame(self._content, bg=BG)
        f.pack(fill="both", expand=True, padx=PAD_LG, pady=PAD)

        lbl(f, "Animations", fg=LABEL, font=F_TITLE2).pack(anchor="w")
        lbl(f, "Record timestamped motion by moving the hand freely. Replays at original speed.",
            fg=LABEL2, font=F_CALLOUT).pack(anchor="w", pady=(2, PAD_SM))

        rec = card(f)
        rec.pack(fill="x", pady=4)

        rrow = tk.Frame(rec, bg=SURFACE)
        rrow.pack(fill="x")
        lbl(rrow, "Name:", fg=LABEL2, font=F_CALLOUT).pack(side="left", padx=PAD_SM)
        self._anim_name = tk.StringVar(value="")
        tk.Entry(rrow, textvariable=self._anim_name, width=18,
                 bg=ELEVATED, fg=LABEL, font=F_BODY,
                 relief="flat", insertbackground=LABEL).pack(side="left", padx=PAD_SM)
        self._rec_btn = accent_btn(rrow, "⏺  Start Recording",
                                    self._toggle_record, color=RED)
        self._rec_btn.pack(side="left", padx=PAD_SM)

        self._anim_status = lbl(f, "Torque OFF to begin recording.", fg=LABEL2, font=F_CALLOUT)
        self._anim_status.pack(anchor="w", pady=4)

        self._rec_bar = tk.Frame(f, bg=ELEVATED, height=4)
        self._rec_bar.pack(fill="x", pady=2)
        self._rec_fill = tk.Frame(self._rec_bar, bg=RED, height=4)
        self._rec_fill.place(x=0, y=0, relwidth=0.0, relheight=1.0)

        divider(f).pack(fill="x", pady=PAD_SM)
        lbl(f, f"Saved animations ({len(self._anims)})", fg=LABEL2,
            font=F_FOOTNOTE).pack(anchor="w")

        self._anim_list_frame = tk.Frame(f, bg=BG)
        self._anim_list_frame.pack(fill="both", expand=True, pady=PAD_SM)
        self._refresh_anim_list()

    def _toggle_record(self):
        if self._recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self):
        name = self._anim_name.get().strip()
        if not name:
            self._safe_status("Enter animation name first.", YELLOW)
            return
        if not self.bus.state.connected:
            self._safe_status("Not connected to hardware.", RED)
            return

        self._recording = True
        self._anim_frames.clear()
        self._stop_rec.clear()

        # Disable torque so the hand can be moved freely
        self.bus.set_torque(False)

        # Safe widget updates — buttons may be destroyed if panel is rebuilt
        self._safe_update_btn("■  Stop", YELLOW)
        self._safe_status("Recording…  move the hand freely", RED)

        self._rec_thread = threading.Thread(target=self._record_loop, daemon=True)
        self._rec_thread.start()

    def _stop_recording(self):
        self._stop_rec.set()
        self._recording = False
        name = self._anim_name.get().strip() if self._is_widget_alive("_anim_name") else ""
        if self._anim_frames and name:
            self._anims[name] = self._anim_frames[:]
            self.bus.save_animations(self._anims)
            dur = self._anim_frames[-1]["time"]
            msg = f"✓ '{name}' — {len(self._anim_frames)} frames  {dur:.1f}s"
            self.after(0, lambda: self._safe_status(msg, GREEN))
            self.after(0, self._safe_refresh_list)
            self.bus.log(msg)
        else:
            self.after(0, lambda: self._safe_status("No frames captured.", YELLOW))
        self.after(0, lambda: self._safe_update_btn("⏺  Start Recording", RED))
        self.after(0, lambda: self._safe_fill(0.0))

    def _record_loop(self):
        """Read positions directly at 30 Hz — independent of background poll.

        Uses bus.read_positions_direct() so recording works reliably even when:
        - torque is off
        - the 10 Hz background poll is slow or lagging
        - no prior state has been populated
        """
        THRESHOLD = 5
        FPS       = 30
        dt        = 1.0 / FPS
        last_pos: dict = {}
        t0 = time.perf_counter()
        MAX_DUR   = 60.0

        while not self._stop_rec.is_set():
            elapsed = time.perf_counter() - t0
            if elapsed > MAX_DUR:
                self.after(0, self._stop_recording)
                break

            # Direct read — fresh data every frame, works with torque off
            cur = self.bus.read_positions_direct()

            if not cur:
                time.sleep(dt)
                continue

            # First frame: always capture to initialise last_pos
            if not last_pos:
                last_pos = dict(cur)

            moved = any(
                abs(cur.get(m, 0) - last_pos.get(m, cur.get(m, 0))) > THRESHOLD
                for m in cur
            )
            if moved:
                frame = {str(m): v for m, v in cur.items()}
                self._anim_frames.append({"time": round(elapsed, 3), "positions": frame})
                last_pos = dict(cur)
                n        = len(self._anim_frames)
                progress = min(elapsed / MAX_DUR, 1.0)
                self.after(0, lambda p=progress, c=n: self._update_rec_ui(p, c))

            time.sleep(dt)

    def _update_rec_ui(self, progress: float, count: int):
        self._safe_fill(progress)
        self._safe_status(
            f"Recording…  {count} frames  ({progress*100:.0f}% of 60 s max)", RED)

    # ── Safe widget helpers (guards against destroyed widgets) ────────────────

    def _is_widget_alive(self, attr: str) -> bool:
        try:
            w = getattr(self, attr, None)
            return w is not None and w.winfo_exists()
        except Exception:
            return False

    def _safe_status(self, text: str, color: str):
        try:
            if self._is_widget_alive("_anim_status"):
                self._anim_status.config(text=text, fg=color)
        except Exception:
            pass

    def _safe_update_btn(self, text: str, color: str):
        try:
            if self._is_widget_alive("_rec_btn"):
                self._rec_btn.update_text(text)
                self._rec_btn.update_colors(color)
        except Exception:
            pass

    def _safe_fill(self, progress: float):
        try:
            if self._is_widget_alive("_rec_fill"):
                self._rec_fill.place(relwidth=max(0.0, min(1.0, progress)))
        except Exception:
            pass

    def _safe_refresh_list(self):
        try:
            if self._is_widget_alive("_anim_list_frame"):
                self._refresh_anim_list()
        except Exception:
            pass

    def _refresh_anim_list(self):
        for w in self._anim_list_frame.winfo_children():
            w.destroy()
        for name, frames in self._anims.items():
            dur = frames[-1]["time"] if frames else 0
            row = tk.Frame(self._anim_list_frame, bg=SURFACE, pady=PAD_SM, padx=PAD_SM)
            row.pack(fill="x", pady=2)
            lbl(row, name, fg=LABEL, font=F_BODY).pack(side="left", padx=PAD_SM)
            lbl(row, f"{len(frames)} fr  {dur:.1f}s", fg=LABEL3,
                font=F_CAPTION).pack(side="left")
            accent_btn(row, "▶  Play",
                       lambda n=name: self._play_anim(n),
                       color=BLUE, font=F_CALLOUT, px=10, py=5).pack(side="right", padx=4)
            ghost_btn(row, "Delete",
                      lambda n=name: self._delete_anim(n),
                      fg=RED, font=F_CAPTION, px=8, py=4).pack(side="right", padx=4)

    def _play_anim(self, name: str):
        frames = self._anims.get(name, [])
        if not frames:
            return
        self.bus.set_torque(True)
        threading.Thread(target=self._playback_thread,
                         args=(frames,), daemon=True).start()
        self.bus.log(f"Animation play: {name}")

    def _playback_thread(self, frames: list):
        # Crawl to first frame
        first = {int(k): v for k, v in frames[0]["positions"].items()}
        goal_time = 2500
        self.bus.sync_write(first, speed=2000, torque=goal_time)
        time.sleep(2.5)

        t0 = time.perf_counter()
        for frame in frames:
            target_t = frame["time"]
            while (time.perf_counter() - t0) < target_t:
                pass
            targets = {int(k): v for k, v in frame["positions"].items()}
            self.bus.sync_write(targets, speed=2000, torque=100)

    def _delete_anim(self, name: str):
        del self._anims[name]
        self.bus.save_animations(self._anims)
        self.bus.log(f"Animation deleted: {name}")
        self._refresh_anim_list()
