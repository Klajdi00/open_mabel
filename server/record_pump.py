"""record_pump.py — off-loop camera capture so recording never stalls the sim.

Rendering every robot camera costs a fixed GPU stall per frame (tens of ms on
macOS); doing it INLINE in the control loop drops the sim to a fraction of real
time the instant recording starts. This pump renders on a DEDICATED thread from
a cheap MjData snapshot, so the control loop only pays a memcpy (`mj_copyData`).

Back-pressure: if the renderer can't keep up with the target fps the pump simply
DROPS the frame (the episode stays temporally consistent via per-frame
timestamps) — the sim keeps running smoothly instead of grinding to slow-motion.

The renderers live on the worker thread because a GL context is thread-bound;
the EpisodeRecorder is driven entirely from that one thread (start on spawn,
_write_frame per capture, stop on join), so there is no cross-thread contention.
"""
from __future__ import annotations

import logging
import threading

import mujoco

from sim_sensor_source import SimSensorSource

log = logging.getLogger("recorder")


class RecordPump:
    """Threaded camera capture feeding an EpisodeRecorder. One pump per episode."""

    def __init__(self, model, recorder, *, get_control, get_desired, chassis_body,
                 fps=10.0, rgb_res=(424, 240), depth_res=(424, 240)):
        self.model = model
        self.recorder = recorder
        self.fps = max(1.0, float(fps))
        self._cfg = dict(get_control=get_control, get_desired=get_desired,
                         chassis_body=chassis_body, rgb_res=rgb_res, depth_res=depth_res)
        self.source = None
        self._snap = mujoco.MjData(model)        # private copy the worker renders from
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._t = 0.0
        self._has = False                        # a fresh snapshot is waiting
        self._ctl = None                         # control state AT the snapshot instant
        self._busy = False                       # worker is mid-render
        self._last_t = -1e9
        self._run = False
        self._thread = None

    @property
    def active(self) -> bool:
        return self._run

    def meta(self) -> dict:
        return self.source.meta() if self.source else {}

    # ── lifecycle ──────────────────────────────────────────────────────────────
    def start(self, task, scene) -> Exception | None:
        """Spawn the worker; it builds the renderers + opens the episode. Blocks
        until that's done so the caller can publish an accurate recording status.
        Returns an exception if the source/renderers failed to build, else None."""
        self._run = True
        self._task, self._scene = task, scene
        self._ready = threading.Event()
        self._start_err: list = [None]
        self._thread = threading.Thread(target=self._loop, name="record-pump", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=15)
        if self._start_err[0] is not None:
            self._run = False
        return self._start_err[0]

    def stop(self):
        """Finish the worker and finalise the episode (returns the episode dir)."""
        self._run = False
        self._wake.set()
        if self._thread:
            self._thread.join(timeout=10)
        if self.source:
            self.source.close()
        return self.recorder.stop()

    # ── capture path ───────────────────────────────────────────────────────────
    def maybe_capture(self, t: float, data):
        """Called from the control loop. Snapshots `data` (cheap) if a frame is due
        and the worker is free; otherwise returns at once so the sim never blocks."""
        if not self._run or (t - self._last_t) < (1.0 / self.fps):
            return
        with self._lock:
            if self._busy or self._has:          # worker behind → drop this frame
                return
            mujoco.mj_copyData(self._snap, self.model, data)
            # The operator's control state (which body parts are tracked, scope,
            # method, engaged…) captured at the SAME instant as the physics —
            # read later on the worker it could straddle a toggle and label a
            # frame with the wrong condition (policy-training labels).
            try:
                self._ctl = self._cfg["get_control"]()
            except Exception:                    # never let capture crash teleop
                self._ctl = None
            self._t = t
            self._has = True
            self._last_t = t
        self._wake.set()

    def _loop(self):
        try:
            self.source = SimSensorSource(
                self.model, self._snap, self._cfg["chassis_body"],
                self._cfg["get_control"], get_desired=self._cfg["get_desired"],
                rgb_res=self._cfg["rgb_res"], depth_res=self._cfg["depth_res"])
            m = self.source.meta()
            m["scene"] = self._scene
            m["capture"] = "threaded"
            self.recorder.start(self._task, m)
        except Exception as e:                    # renderer/source failed to build
            self._start_err[0] = e
            self._ready.set()
            return
        self._ready.set()

        while self._run:
            if not self._wake.wait(timeout=0.2):
                continue
            self._wake.clear()
            with self._lock:
                if not self._has:
                    continue
                self._has = False
                t = self._t
                ctl = self._ctl
                self._busy = True
            try:
                s = self.source.sample()                              # renders from _snap
                if ctl is not None:
                    s["control"] = ctl                                # snapshot-synced
                self.recorder._write_frame(t, s)
            except Exception as e:                # never let capture crash teleop
                log.warning("REC frame skipped: %s", e)
            finally:
                with self._lock:
                    self._busy = False
