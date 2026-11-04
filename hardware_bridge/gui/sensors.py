"""sensors.py — live camera tiles + sensor-status chips for the Control Center.

Cameras are read as MJPEG over HTTP (the same streams the web teleop uses, served by the
sensors stack / mjpeg_server on :8080) — no ROS image subscription needed. Each tile runs
a tiny background reader thread and shows the latest frame, with a status dot + fps; an
absent stream just shows "offline" and keeps retrying. LiDAR / IMU status come from the
bridge telemetry snapshot (see bridge_link).
"""

import io
import threading
import time
import urllib.request

import tkinter as tk

from style import (ELEVATED, GREEN, LABEL, LABEL2, LABEL3, ORANGE, RED, SURFACE,
                   FONT_MONO_S, FONT_SMALL, StatusDot)

try:
    from PIL import Image, ImageTk
    _PIL = True
except Exception:
    _PIL = False


class MjpegStream:
    """Background reader of a multipart/x-mixed-replace MJPEG URL. ``latest()`` returns
    (PIL.Image|None, ok, fps). Reconnects forever; cheap when the server is absent."""

    def __init__(self, url):
        self.url = url
        self._frame = None
        self._ok = False
        self._fps = 0.0
        self._stop = threading.Event()
        self._t = None

    def start(self):
        if self._t and self._t.is_alive():
            return
        self._stop.clear()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def stop(self):
        self._stop.set()

    def latest(self):
        return self._frame, self._ok, self._fps

    def _run(self):
        while not self._stop.is_set():
            try:
                req = urllib.request.urlopen(self.url, timeout=3.0)
                buf = b""
                cnt, t0 = 0, time.monotonic()
                while not self._stop.is_set():
                    chunk = req.read(8192)
                    if not chunk:
                        break
                    buf += chunk
                    a = buf.find(b"\xff\xd8")
                    b = buf.find(b"\xff\xd9", a + 2) if a >= 0 else -1
                    if a >= 0 and b > a:
                        jpg, buf = buf[a:b + 2], buf[b + 2:]
                        if _PIL:
                            try:
                                im = Image.open(io.BytesIO(jpg)); im.load()
                                self._frame, self._ok = im, True
                            except Exception:
                                pass
                        cnt += 1
                        now = time.monotonic()
                        if now - t0 >= 1.0:
                            self._fps = cnt / (now - t0)
                            cnt, t0 = 0, now
                    if len(buf) > 4_000_000:
                        buf = b""
                try:
                    req.close()
                except Exception:
                    pass
            except Exception:
                pass
            self._frame, self._ok, self._fps = None, False, 0.0
            self._stop.wait(2.0)        # back off before retrying
        self._ok = False


class CameraTile(tk.Frame):
    """A fixed-size live thumbnail with a caption (name · status dot · fps)."""

    def __init__(self, parent, label, url, w=212, h=128):
        super().__init__(parent, bg=SURFACE)
        self.w, self.h = w, h
        self._stream = MjpegStream(url) if (url and _PIL) else None
        self._tkimg = None

        holder = tk.Frame(self, bg="#000000", width=w, height=h)
        holder.pack(padx=4, pady=(4, 0))
        holder.pack_propagate(False)
        msg = "not connected" if _PIL else "no Pillow"
        self._img = tk.Label(holder, bg="#000000", fg=LABEL3, text=msg, font=FONT_SMALL)
        self._img.pack(fill="both", expand=True)

        cap = tk.Frame(self, bg=SURFACE)
        cap.pack(fill="x", padx=4, pady=(2, 4))
        self._dot = StatusDot(cap, d=8, bg=SURFACE, color=LABEL3)
        self._dot.pack(side="left", padx=(2, 5))
        tk.Label(cap, text=label, bg=SURFACE, fg=LABEL, font=FONT_SMALL).pack(side="left")
        self._stat = tk.Label(cap, text="", bg=SURFACE, fg=LABEL3, font=FONT_MONO_S)
        self._stat.pack(side="right")

        if self._stream:
            self._stream.start()
        self._tick()

    def _tick(self):
        if self._stream is not None:
            frame, ok, fps = self._stream.latest()
            if ok and frame is not None:
                im = frame.copy()
                im.thumbnail((self.w, self.h))
                self._tkimg = ImageTk.PhotoImage(im)
                self._img.config(image=self._tkimg, text="")
                self._dot.set(GREEN)
                self._stat.config(text=f"{fps:.0f} fps", fg=LABEL2)
            else:
                self._img.config(image="", text="not connected", fg=LABEL3)
                self._dot.set(LABEL3)
                self._stat.config(text="")
        self.after(150, self._tick)

    def stop(self):
        if self._stream is not None:
            self._stream.stop()


class SensorPill(tk.Frame):
    """A compact status chip: ● dot + name + detail (e.g. "LiDAR  12 Hz")."""

    def __init__(self, parent, label, bg=ELEVATED):
        super().__init__(parent, bg=bg)
        self._bg = bg
        self._dot = StatusDot(self, d=9, bg=bg, color=LABEL3)
        self._dot.pack(side="left", padx=(10, 6), pady=5)
        tk.Label(self, text=label, bg=bg, fg=LABEL, font=FONT_SMALL).pack(side="left")
        self._detail = tk.Label(self, text="—", bg=bg, fg=LABEL3, font=FONT_MONO_S)
        self._detail.pack(side="left", padx=(7, 11))

    def set(self, state, detail=""):
        """state: True/'ok' (green) · 'warn' (orange) · False/None (offline)."""
        color = {True: GREEN, "ok": GREEN, "warn": ORANGE,
                 "err": RED, False: LABEL3, None: LABEL3}.get(state, LABEL3)
        self._dot.set(color)
        self._detail.config(text=detail or ("ok" if color == GREEN else "—"),
                            fg=LABEL2 if color != LABEL3 else LABEL3)
