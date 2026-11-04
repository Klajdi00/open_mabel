"""handmodel.py — the hand studio's MAIN view: a live 3D model of the ORCA hand.

This is the centre panel of the Firmware Bench hand studio. It renders the *real
robot model* — the very MJCF that ships with the ORCA firmware
(``firmware/orca_hand/feetech``) — offscreen with MuJoCo, and articulates it in
real time from the hand bridge's live joint telemetry. Drag to orbit, scroll to
zoom, double-click to reset the view.

The model is driven purely from ``bus.state.motors[mid].pct`` (0-100 %, already
calibration- and invert-corrected by :meth:`HandBus.raw_to_pct`), mapped through
the joint-name table below onto each MJCF hinge's own range. So the 3D hand shows
exactly the joint fractions the studio believes the physical hand is at — move a
joint (calibration sweep, slider, teleop) and it moves here too.

Degrades gracefully: if MuJoCo / Pillow / numpy or the model assets are missing,
the widget shows a labelled placeholder and the rest of the studio keeps working.
"""

import tkinter as tk
from pathlib import Path

from style import (BG, PANEL_BG, SURFACE, ELEVATED, LABEL, LABEL2, LABEL3, ORANGE,
                   GREEN, RED, SEP, FONT_SMALL, FONT_MONO_S, FONT_SUB, AppleButton)

# ── optional heavy deps: the studio must import even without them ────────────────
try:
    import mujoco
    import numpy as np
    from PIL import Image, ImageTk
    _DEPS_OK = True
    _DEPS_ERR = ""
except Exception as _e:                       # pragma: no cover - stripped env
    _DEPS_OK = False
    _DEPS_ERR = f"{type(_e).__name__}: {_e}"

# repo_root / firmware / orca_hand / feetech  (this file: …/hardware_bridge/gui/hand_studio/)
_FEETECH = Path(__file__).resolve().parents[3] / "firmware" / "orca_hand" / "feetech"

# bus joint name (ORCA) -> MJCF joint base name (URDF). Only the thumb differs:
# ORCA cmc/mp/ip == URDF mcp/pip/dip (see right_hand.yaml). Everything else matches.
_THUMB_REMAP = {"thumb_cmc": "thumb_mcp", "thumb_mp": "thumb_pip", "thumb_ip": "thumb_dip"}

_RENDER_MS = 66                # ~15 fps render tick
_MIN_W, _MIN_H = 240, 240      # renderer size clamps
_MAX_W, _MAX_H = 1400, 1400


def _compose_scene(side: str) -> str:
    # A large offscreen framebuffer lets us render at the widget's size (the model's
    # default is only 640x480, which caps the render resolution otherwise).
    return (f'<mujoco model="orca {side} studio">\n'
            f'  <include file="assets/mjcf/scene.xml"/>\n'
            f'  <include file="assets/mjcf/options.xml"/>\n'
            f'  <include file="models/mjcf/orcahand_{side}.mjcf"/>\n'
            f'  <visual><global offwidth="{_MAX_W}" offheight="{_MAX_H}"/></visual>\n'
            f'</mujoco>\n')


class HandModelView(tk.Frame):
    """Live MuJoCo render of the current hand, articulated from bridge telemetry."""

    def __init__(self, parent, bus, studio):
        super().__init__(parent, bg=BG)
        self.bus = bus
        self.studio = studio                 # to read the current hand side (bus.hand)

        self._models: dict = {}              # side -> (model, data, jindex)
        self._renderer = None
        self._rsize = (0, 0)                 # renderer's current (w, h)
        self._side_loaded = None
        self._cam = None
        self._img_ref = None                 # keep a ref so Tk doesn't GC the PhotoImage
        self._latest: dict = {}              # joint name -> pct, written from the bus thread
        self._connected = False
        self._load_err = ""
        self._drag = None
        self._last_sig = None                # dirty check: skip render if nothing changed
        self._want = (0, 0)                  # target render size from <Configure>
        self._alive = True

        self._build()
        if _DEPS_OK:
            bus.subscribe(self._on_state)
            self.after(300, self._tick)

    # ── UI scaffold ─────────────────────────────────────────────────────────────
    def _build(self):
        head = tk.Frame(self, bg=BG)
        head.pack(fill="x", padx=14, pady=(10, 4))
        tk.Label(head, text="3D MODEL", bg=BG, fg=LABEL2, font=FONT_SMALL).pack(side="left")
        self._badge = tk.Label(head, text="", bg=BG, fg=LABEL3, font=FONT_MONO_S)
        self._badge.pack(side="left", padx=(8, 0))
        tk.Label(head, text="drag · orbit    scroll · zoom    dbl-click · reset",
                 bg=BG, fg=LABEL3, font=FONT_MONO_S).pack(side="right")

        # the render surface
        self._view = tk.Label(self, bg=BG, bd=0, highlightthickness=0)
        self._view.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        if not _DEPS_OK:
            self._view.config(
                text="3D model view unavailable\n\n" + _DEPS_ERR +
                     "\n\ninstall mujoco + pillow to enable",
                fg=LABEL3, font=FONT_MONO_S, justify="center")
            return

        self._view.bind("<Configure>", self._on_resize)
        self._view.bind("<ButtonPress-1>", self._on_press)
        self._view.bind("<B1-Motion>", self._on_motion)
        self._view.bind("<Double-Button-1>", lambda e: self._reset_view())
        # scroll-zoom: Windows/Mac (<MouseWheel>) + X11 (Button-4/5)
        self._view.bind("<MouseWheel>", self._on_wheel)
        self._view.bind("<Button-4>", lambda e: self._zoom(0.9))
        self._view.bind("<Button-5>", lambda e: self._zoom(1.1))

    # ── telemetry (bus thread): snapshot joint fractions, no Tk calls here ───────
    def _on_state(self, state, log):
        d = {}
        for mid, ms in state.motors.items():
            if ms and ms.responding and ms.raw is not None:
                name = self.bus.id_joint.get(mid)
                if name:
                    d[name] = ms.pct
        self._latest = d                     # atomic replace — read on the main thread
        self._connected = bool(state.connected)

    # ── model loading (main thread, lazy per side) ──────────────────────────────
    def _ensure_model(self, side: str):
        if side in self._models:
            return self._models[side]
        model = self._load_model(side)
        if model is None:
            return None
        data = mujoco.MjData(model)
        jindex = self._joint_index(model, side)
        self._models[side] = (model, data, jindex)
        return self._models[side]

    def _load_model(self, side: str):
        # MuJoCo resolves <include>/mesh paths relative to the XML file's dir, so the
        # composed scene has to live inside the feetech root. Write a dot-file, parse,
        # remove it. Fall back to the committed main.xml for the right hand.
        tmp = _FEETECH / f".studio_scene_{side}.xml"
        try:
            tmp.write_text(_compose_scene(side))
            try:
                return mujoco.MjModel.from_xml_path(str(tmp))
            finally:
                try:
                    tmp.unlink()
                except OSError:
                    pass
        except Exception as e:
            main = _FEETECH / "main.xml"               # main.xml composes the RIGHT hand
            if side == "right" and main.exists():
                try:
                    return mujoco.MjModel.from_xml_path(str(main))
                except Exception as e2:
                    e = e2
            self._load_err = f"{type(e).__name__}: {e}"
            return None

    def _joint_index(self, model, side: str) -> dict:
        """base joint name (side prefix stripped) -> (qpos_adr, lo, hi, limited)."""
        idx = {}
        pre = side + "_"
        for jid in range(model.njnt):
            jn = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            if not jn:
                continue
            base = jn[len(pre):] if jn.startswith(pre) else jn
            lo, hi = model.jnt_range[jid]
            idx[base] = (int(model.jnt_qposadr[jid]), float(lo), float(hi),
                         bool(model.jnt_limited[jid]))
        return idx

    # ── camera ──────────────────────────────────────────────────────────────────
    def _init_cam(self, model):
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, cam)
        cam.azimuth += 35.0
        cam.elevation = -18.0
        self._cam = cam

    def _reset_view(self):
        model_pack = self._models.get(self._side_loaded)
        if model_pack:
            self._init_cam(model_pack[0])
            self._last_sig = None

    def _on_press(self, e):
        if self._cam is not None:
            self._drag = (e.x, e.y, self._cam.azimuth, self._cam.elevation)

    def _on_motion(self, e):
        if self._cam is None or self._drag is None:
            return
        x0, y0, az0, el0 = self._drag
        self._cam.azimuth = az0 - (e.x - x0) * 0.4
        self._cam.elevation = max(-89.0, min(89.0, el0 + (e.y - y0) * 0.4))
        self._last_sig = None                # force a redraw

    def _on_wheel(self, e):
        self._zoom(0.9 if e.delta > 0 else 1.1)

    def _zoom(self, factor):
        if self._cam is not None:
            self._cam.distance = max(0.08, min(2.0, self._cam.distance * factor))
            self._last_sig = None

    # ── resize ──────────────────────────────────────────────────────────────────
    def _on_resize(self, e):
        self._want = (max(_MIN_W, min(_MAX_W, e.width)),
                      max(_MIN_H, min(_MAX_H, e.height)))

    def _ensure_renderer(self, model, w, h):
        if self._renderer is not None and self._rsize == (w, h) and \
                self._side_loaded == self.studio.bus.hand:
            return self._renderer
        if self._renderer is not None:
            try:
                self._renderer.close()
            except Exception:
                pass
        self._renderer = mujoco.Renderer(model, h, w)
        self._rsize = (w, h)
        return self._renderer

    # ── render loop (main thread) ───────────────────────────────────────────────
    def _tick(self):
        if not self._alive or not self.winfo_exists():
            return
        try:
            if self.winfo_ismapped():
                self._render_once()
            self._fails = 0
        except Exception as e:
            # A render exception almost always means no GL context (headless); latch
            # after a few tries so we don't churn the error path ~15x/second forever.
            self._fails = getattr(self, "_fails", 0) + 1
            self._show_error(f"{type(e).__name__}: {e}")
            if self._fails >= 3:
                return
        if self._alive:
            self.after(_RENDER_MS, self._tick)

    def _render_once(self):
        side = self.studio.bus.hand
        pack = self._ensure_model(side)
        if pack is None:
            self._show_error(self._load_err or "model unavailable")
            return
        model, data, jindex = pack
        if self._side_loaded != side or self._cam is None:
            self._side_loaded = side
            self._init_cam(model)

        # target render size (falls back to the widget's current size), clamped to the
        # model's offscreen framebuffer so the Renderer never overflows it.
        w, h = self._want
        if w < _MIN_W or h < _MIN_H:
            w = max(_MIN_W, min(_MAX_W, self._view.winfo_width()))
            h = max(_MIN_H, min(_MAX_H, self._view.winfo_height()))
        w = max(_MIN_W, min(w, int(model.vis.global_.offwidth)))
        h = max(_MIN_H, min(h, int(model.vis.global_.offheight)))

        latest = self._latest                 # single read of the shared ref
        sig = self._signature(side, latest, w, h)
        if sig == self._last_sig:
            return                            # nothing changed — skip the GPU work
        self._last_sig = sig

        # articulate: joint fraction -> hinge range (neutral/open when no telemetry)
        for jname, ent in jindex.items():
            adr, lo, hi, lim = ent
            if not lim:
                continue
            data.qpos[adr] = lo               # default: joint at its minimum (open)
        for jname, pct in latest.items():
            base = _THUMB_REMAP.get(jname, jname)
            ent = jindex.get(base)
            if ent is None:
                continue
            adr, lo, hi, lim = ent
            if not lim:
                continue
            f = max(0.0, min(1.0, pct / 100.0))
            data.qpos[adr] = lo + f * (hi - lo)
        mujoco.mj_forward(model, data)

        r = self._ensure_renderer(model, w, h)
        r.update_scene(data, self._cam)
        rgb = r.render()

        img = ImageTk.PhotoImage(Image.fromarray(rgb))
        self._img_ref = img                   # prevent GC
        self._view.config(image=img, text="")
        self._badge.config(
            text=f"{side} hand · {'live' if latest else 'no telemetry'}",
            fg=GREEN if latest else LABEL3)

    def _signature(self, side, latest, w, h):
        cam = self._cam
        camsig = (round(cam.azimuth, 1), round(cam.elevation, 1),
                  round(cam.distance, 3)) if cam else ()
        joints = tuple(sorted((k, round(v, 1)) for k, v in latest.items()))
        return (side, w, h, camsig, joints, self._connected)

    def _show_error(self, msg):
        self._view.config(image="", text="3D model view error\n\n" + msg,
                          fg=LABEL3, font=FONT_MONO_S, justify="center")
        self._img_ref = None

    # ── lifecycle ───────────────────────────────────────────────────────────────
    def close(self):
        self._alive = False
        if self._renderer is not None:
            try:
                self._renderer.close()
            except Exception:
                pass
            self._renderer = None

    def destroy(self):
        self.close()
        super().destroy()
