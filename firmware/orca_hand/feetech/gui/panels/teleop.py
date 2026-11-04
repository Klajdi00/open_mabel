"""Teleop Panel — Webcam (MediaPipe) or Apple Vision Pro retargeting."""

import sys
import threading
import time
import types
import tkinter as tk
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from style import *


# ── MediaPipe loader ─────────────────────────────────────────────────────────
# mediapipe's top-level __init__ imports the optional `tasks` API, which drags
# in TensorFlow. TensorFlow built against numpy 1.x crashes on numpy 2.x with
# "TypeError: Unable to convert function return value to a Python type".
# The hand-tracking *solution* runs on mediapipe's own C++ graph and never uses
# TensorFlow, so we register harmless stubs for `mediapipe.tasks` first — this
# lets the hand solution import cleanly regardless of the numpy version.

_MP_CACHE: dict = {}

def load_mediapipe():
    """Return (cv2, mp_hands, mp_drawing). Raises on genuine missing/broken deps.

    Result is cached so repeated calls are cheap.
    """
    if _MP_CACHE:
        return _MP_CACHE["cv2"], _MP_CACHE["hands"], _MP_CACHE["draw"]

    import cv2  # ModuleNotFoundError here means opencv-python is missing

    # Stub the tasks API so `import mediapipe` skips the TensorFlow chain.
    for _m in ("mediapipe.tasks", "mediapipe.tasks.python"):
        if _m not in sys.modules:
            stub = types.ModuleType(_m)
            stub.__path__ = []          # mark as a package so submodule access works
            sys.modules[_m] = stub

    import mediapipe  # noqa: F401
    from mediapipe.python.solutions import hands as mp_hands
    from mediapipe.python.solutions import drawing_utils as mp_draw

    _MP_CACHE.update(cv2=cv2, hands=mp_hands, draw=mp_draw)
    return cv2, mp_hands, mp_draw


# ── Controller hand retargeter (the EXACT one the whole-body teleop uses) ─────
# We feed MediaPipe hand landmarks into mabel.hand_vector_retarget /
# hand_ik_retarget — the same DexPilot-style keypoint-IK solver the production
# TeleopController drives from the Vision Pro — so the GUI webcam teleop maps the
# hand identically to whole-body teleop, instead of the old curl approximation.

_RT_CACHE: dict = {}
TELEOP_TORQUE = 150          # gentle goal-time while streaming retargeted poses

# MediaPipe Hands landmark index -> the ARKit wire-name the retargeter expects.
_MP_TO_WIRE = {
    "wrist": 0,
    "index_knuckle": 5,   "index_intermediate_base": 6,   "index_tip": 8,
    "middle_knuckle": 9,  "middle_intermediate_base": 10, "middle_tip": 12,
    "ring_knuckle": 13,   "ring_intermediate_base": 14,   "ring_tip": 16,
    "little_knuckle": 17, "little_intermediate_base": 18, "little_tip": 20,
    "thumb_intermediate_base": 2, "thumb_intermediate_tip": 3, "thumb_tip": 4,
}

# MuJoCo actuator name "<Side> <Finger> <Seg>" -> ORCA joint name.
_FINGER_SEG = {"Abd": "abd", "MCP": "mcp", "PIP": "pip"}
_THUMB_SEG = {"MCP": "cmc", "Abd": "abd", "PIP": "mp", "DIP": "ip"}


def _act_to_orca(act_name: str):
    parts = act_name.split()                 # e.g. "Left Index MCP"
    if len(parts) != 3:
        return None
    _, finger, seg = parts
    finger = finger.lower()
    if finger == "thumb":
        return f"thumb_{_THUMB_SEG.get(seg, seg.lower())}"
    return f"{finger}_{_FINGER_SEG.get(seg, seg.lower())}"


def _repo_root() -> Path:
    for p in Path(__file__).resolve().parents:
        if (p / "controller").is_dir() and (p / "simulation").is_dir():
            return p
    raise RuntimeError("could not locate MABEL repo root (needs controller/ + simulation/)")


def load_retargeter():
    """Build (and cache) the SAME hand retargeter the whole-body controller uses:
    mabel.hand_vector_retarget.make_hand_retargeter over the canonical
    mabel_full.xml. The MuJoCo model load is the slow part, hence the cache."""
    if "rt" in _RT_CACHE:
        return _RT_CACHE["rt"]
    import mujoco
    root = _repo_root()
    ctrl = str(root / "controller")
    if ctrl not in sys.path:
        sys.path.insert(0, ctrl)
    xml = root / "simulation" / "mabel_mujoco" / "models" / "mabel_full.xml"
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    from mabel.hand_vector_retarget import make_hand_retargeter
    rt = make_hand_retargeter(model, data)
    _RT_CACHE.update(rt=rt, model=model, data=data)
    return rt


class TeleopPanel(tk.Frame):

    SUB_PANELS = ["Webcam", "Vision Pro"]

    def __init__(self, parent, bus):
        super().__init__(parent, bg=BG)
        self.bus      = bus
        self._content: tk.Frame | None = None
        self._teleop_thread: threading.Thread | None = None
        self._teleop_stop   = threading.Event()
        self._teleop_running= False
        # Latest camera frame (RGB numpy array) shared from the worker thread;
        # the main thread reads it to build the Tk image.
        self._latest_frame  = None
        self._frame_lock    = threading.Lock()
        self._rt            = None     # controller hand retargeter (lazy-loaded)

        nav = tk.Frame(self, bg=BG, pady=PAD_SM)
        nav.pack(fill="x", padx=PAD)
        AppleSegmentedControl(nav, self.SUB_PANELS, self.SUB_PANELS[0], self._show, parent_bg=BG).pack(side="left")
        self._show("Webcam")

    def _show(self, name):
        # Stop any running camera before tearing down its preview widget
        if self._teleop_running:
            self._teleop_stop.set()
            self._teleop_running = False
        if self._content:
            self._content.destroy()
        self._content = tk.Frame(self, bg=BG)
        self._content.pack(fill="both", expand=True)
        if name == "Webcam":
            self._build_webcam()
        else:
            self._build_avp()

    # ── Webcam / MediaPipe ───────────────────────────────────────────────────

    def _build_webcam(self):
        f = tk.Frame(self._content, bg=BG)
        f.pack(fill="both", expand=True, padx=PAD, pady=PAD_SM)

        # Dependency check — load the hand-tracking solution (tensorflow bypassed)
        deps_ok, err = True, None
        try:
            load_mediapipe()
        except Exception as e:
            deps_ok, err = False, e

        # ── Compact top bar: title + live controls on one row ─────────────────
        bar = tk.Frame(f, bg=BG)
        bar.pack(fill="x")

        title_box = tk.Frame(bar, bg=BG)
        title_box.pack(side="left")
        lbl(title_box, "Webcam Retargeting", fg=LABEL, font=F_TITLE2).pack(anchor="w")
        lbl(title_box, "MediaPipe → controller keypoint-IK (same as whole-body teleop)",
            fg=LABEL3, font=F_CAPTION).pack(anchor="w")
        self._wc_status = lbl(title_box, "Ready." if deps_ok else "Dependencies unavailable.",
                              fg=LABEL2, font=F_CALLOUT)
        self._wc_status.pack(anchor="w")

        ctrl = tk.Frame(bar, bg=BG)
        ctrl.pack(side="right")
        lbl(ctrl, "Cam", fg=LABEL2, font=F_CALLOUT).pack(side="left", padx=(0, 4))
        self._cam_idx = tk.StringVar(value="0")
        tk.Entry(ctrl, textvariable=self._cam_idx, width=3, bg=ELEVATED,
                 fg=LABEL, font=F_BODY, relief="flat",
                 insertbackground=LABEL, justify="center").pack(side="left", padx=(0, 12))
        lbl(ctrl, "Scale", fg=LABEL2, font=F_CALLOUT).pack(side="left", padx=(0, 4))
        self._scale = tk.StringVar(value="1.0")
        tk.Entry(ctrl, textvariable=self._scale, width=4, bg=ELEVATED,
                 fg=LABEL, font=F_BODY, relief="flat",
                 insertbackground=LABEL, justify="center").pack(side="left", padx=(0, 12))
        self._mirror_var = tk.BooleanVar(value=True)
        tk.Checkbutton(ctrl, text="Mirror", variable=self._mirror_var,
                       bg=BG, fg=LABEL, font=F_CALLOUT, selectcolor=ELEVATED,
                       activebackground=BG, activeforeground=LABEL).pack(side="left", padx=(0, 12))
        self._wc_btn = accent_btn(ctrl, "▶  Start", self._toggle_webcam, color=ORANGE)
        self._wc_btn.pack(side="left")

        # ── Missing-dependency warning (only when load failed) ────────────────
        if not deps_ok:
            warn = card(f)
            warn.pack(fill="x", pady=PAD_SM)
            msg = str(err)
            if isinstance(err, ModuleNotFoundError):
                missing = "opencv-python" if "cv2" in msg else \
                          "mediapipe"     if "mediapipe" in msg else "dependency"
                lbl(warn, f"⚠  {missing} is not installed",
                    fg=YELLOW, font=F_HEADLINE, bg=SURFACE).pack(anchor="w")
                lbl(warn, f"pip install {missing if missing!='dependency' else 'mediapipe opencv-python'}",
                    fg=ORANGE, font=F_MONO, bg=SURFACE).pack(anchor="w", pady=4)
            else:
                lbl(warn, "⚠  MediaPipe failed to load",
                    fg=YELLOW, font=F_HEADLINE, bg=SURFACE).pack(anchor="w")
                lbl(warn, (msg[:90] + "…") if len(msg) > 90 else msg,
                    fg=LABEL2, font=F_MONO_SM, bg=SURFACE).pack(anchor="w", pady=2)
                lbl(warn, "Try:  pip install 'numpy<2'   (TensorFlow/numpy conflict)",
                    fg=ORANGE, font=F_MONO, bg=SURFACE).pack(anchor="w", pady=4)
            accent_btn(warn, "↻  Retry", self._retry_deps,
                       color=BLUE, parent_bg=SURFACE).pack(anchor="w", pady=(6, 2))

        # ── Large camera display — fills all remaining space ──────────────────
        self._preview = tk.Label(
            f, bg="#000000", fg=LABEL3,
            text="Press  ▶ Start  to begin the camera feed",
            font=F_TITLE2)
        self._preview.pack(fill="both", expand=True, pady=(PAD_SM, 0))

    def _retry_deps(self):
        # Clear any partial cache and rebuild the webcam view (re-runs the check)
        _MP_CACHE.clear()
        self._show("Webcam")

    def _toggle_webcam(self):
        if self._teleop_running:
            self._stop_teleop()
        else:
            self._start_webcam()

    def _start_webcam(self):
        try:
            load_mediapipe()
        except Exception as e:
            self._wc_status.config(text=f"MediaPipe unavailable: {str(e)[:50]}", fg=RED)
            return

        try:
            cam   = int(self._cam_idx.get())
            scale = float(self._scale.get())
        except ValueError:
            self._wc_status.config(text="Invalid camera or scale.", fg=RED)
            return

        # Build the controller's hand retargeter (loads the MuJoCo model once).
        self._wc_status.config(text="Loading retargeter…", fg=ORANGE); self.update_idletasks()
        try:
            self._rt = load_retargeter()
        except Exception as e:
            self._wc_status.config(text=f"Retargeter load failed: {str(e)[:60]}", fg=RED)
            return

        self._teleop_stop.clear()
        self._teleop_running = True
        self._wc_btn.update_text("■  Stop"); self._wc_btn.update_colors(RED)
        self.bus.set_torque(True)
        self._teleop_thread = threading.Thread(
            target=self._webcam_loop,
            args=(cam, scale, self._mirror_var.get()),
            daemon=True,
        )
        self._teleop_thread.start()

    def _stop_teleop(self):
        self._teleop_stop.set()
        self._teleop_running = False
        with self._frame_lock:
            self._latest_frame = None
        def _reset_ui():
            try:
                self._wc_btn.update_text("▶  Start"); self._wc_btn.update_colors(ORANGE)
                self._wc_status.config(text="Stopped.", fg=LABEL2)
                self._preview.config(image="", text="Press  ▶ Start  to begin the camera feed")
                self._preview.image = None
            except Exception:
                pass
        self.after(0, _reset_ui)

    def _webcam_loop(self, cam_idx: int, scale: float, mirror: bool):
        cv2, mp_hands, mp_draw = load_mediapipe()

        hands = mp_hands.Hands(max_num_hands=1,
                               min_detection_confidence=0.6,
                               min_tracking_confidence=0.5)
        cap = cv2.VideoCapture(cam_idx)
        # Request a high-res feed for a crisp, large display
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not cap.isOpened():
            self.after(0, lambda: self._wc_status.config(
                text=f"Cannot open camera {cam_idx}.", fg=RED))
            self._stop_teleop()
            return

        self.after(0, lambda: self._wc_status.config(text="Running…", fg=GREEN))
        # Kick off the main-thread display refresh loop
        self.after(0, self._refresh_preview)

        # Landmark styling: blue dots, orange connections (iOS palette)
        dot_spec  = mp_draw.DrawingSpec(color=(255, 159, 10), thickness=-1, circle_radius=4)
        line_spec = mp_draw.DrawingSpec(color=(10, 132, 255), thickness=2)

        try:
            while not self._teleop_stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.01)
                    continue
                if mirror:
                    frame = cv2.flip(frame, 1)
                rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                # Retarget with the controller's EXACT solver: MediaPipe world
                # landmarks -> ARKit skeleton -> keypoint-IK -> ORCA joint angles.
                if results.multi_hand_world_landmarks and self._rt is not None:
                    world = results.multi_hand_world_landmarks[0].landmark
                    side = "Left" if (self.bus.hand or "left") == "left" else "Right"
                    try:
                        q = self._rt.apply(side, self._skeleton(world), synchronous=True)
                    except Exception:
                        q = None
                    if q and self.bus.state.connected:
                        targets = self._q_to_targets(q)
                        if targets:
                            self.bus.sync_write(targets, speed=2000, torque=TELEOP_TORQUE)

                # Draw the skeleton onto the frame (image landmarks, for display).
                if results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame, results.multi_hand_landmarks[0],
                        mp_hands.HAND_CONNECTIONS, dot_spec, line_spec)

                # Publish the latest RGB frame for the main thread to display.
                # The whole feed is shown continuously, hand present or not.
                with self._frame_lock:
                    self._latest_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                time.sleep(1.0 / 30)
        finally:
            cap.release()
            hands.close()
            self._stop_teleop()

    def _refresh_preview(self):
        """Main-thread: display the latest camera frame, scaled to fill the
        preview area while preserving aspect ratio.  Reschedules at ~30 fps."""
        if not self._teleop_running:
            return
        try:
            from PIL import Image, ImageTk
            with self._frame_lock:
                frame = None if self._latest_frame is None else self._latest_frame
            if frame is not None and self._preview.winfo_exists():
                pw = max(self._preview.winfo_width(),  640)
                ph = max(self._preview.winfo_height(), 360)
                fh, fw = frame.shape[:2]
                s = min(pw / fw, ph / fh)
                nw, nh = max(1, int(fw * s)), max(1, int(fh * s))
                img = Image.fromarray(frame)
                if (nw, nh) != (fw, fh):
                    img = img.resize((nw, nh), Image.BILINEAR)
                photo = ImageTk.PhotoImage(img)
                self._preview.config(image=photo, text="")
                self._preview.image = photo   # keep a reference
        except Exception:
            pass
        self.after(33, self._refresh_preview)

    def _skeleton(self, world_lms) -> dict:
        """MediaPipe world landmarks -> {ARKit wire-name: 4x4 world} the
        retargeter consumes (only translation is used; canonical frame is built
        internally, so the absolute scale/orientation don't matter)."""
        import numpy as np
        sk = {}
        for name, idx in _MP_TO_WIRE.items():
            lm = world_lms[idx]
            T = np.eye(4)
            T[0, 3], T[1, 3], T[2, 3] = lm.x, lm.y, lm.z
            sk[name] = T
        return sk

    def _q_to_targets(self, q: dict) -> dict:
        """Solved {actuator_name: angle_rad} -> {motor_id: raw_tick}, mapping
        each MuJoCo actuator to its ORCA joint and applying that joint's
        calibration affine (invert + clamp) — identical to every other command
        path, so teleop stays in sync with the calibration."""
        oh = getattr(self.bus, "_oh", None)
        if oh is None:
            return {}
        by_name = oh.config.by_name()
        active = self.bus.state.motors
        targets = {}
        for act, val in q.items():
            orca = _act_to_orca(act)
            j = by_name.get(orca) if orca else None
            if j is not None and j.motor_id in active:
                targets[j.motor_id] = j.angle_to_raw(float(val))
        return targets

    # ── Vision Pro ────────────────────────────────────────────────────────────

    def _build_avp(self):
        f = tk.Frame(self._content, bg=BG)
        f.pack(fill="both", expand=True, padx=PAD_LG, pady=PAD)

        lbl(f, "Apple Vision Pro Teleop", fg=LABEL, font=F_TITLE2).pack(anchor="w")
        lbl(f, "Receives joint angle commands from bringup.py via ZMQ.\n"
               "Start bringup.py on the host and teleop_avp.py on the Vision Pro.",
            fg=LABEL2, font=F_CALLOUT).pack(anchor="w", pady=(2, PAD))

        info = card(f)
        info.pack(fill="x", pady=PAD_SM)
        steps = [
            "1.  python control/bringup.py           ← runs on this machine",
            "2.  python control/teleop_avp.py         ← runs on Vision Pro",
            "3.  bringup.py reads from ZMQ and commands motors directly.",
        ]
        for s in steps:
            lbl(info, s, fg=LABEL2, font=F_CALLOUT, bg=SURFACE).pack(anchor="w", padx=PAD_SM, pady=3)

        lbl(f, "ZMQ endpoint:", fg=LABEL2, font=F_CALLOUT).pack(anchor="w", pady=(PAD, 0))
        self._zmq_ep = tk.StringVar(value="tcp://localhost:5555")
        tk.Entry(f, textvariable=self._zmq_ep, width=30,
                 bg=ELEVATED, fg=LABEL, font=F_MONO,
                 relief="flat", insertbackground=LABEL).pack(anchor="w")

        accent_btn(f, "Launch bringup.py in terminal",
                   self._launch_bringup, color=ORANGE).pack(anchor="w", pady=PAD_SM)

    def _launch_bringup(self):
        import subprocess, os
        script = str(Path(__file__).resolve().parent.parent.parent / "bringup.py")
        try:
            subprocess.Popen(["python3", script], cwd=os.path.dirname(script))
            self.bus.log("bringup.py launched in background")
        except Exception as e:
            self.bus.log(f"Failed to launch bringup.py: {e}")
