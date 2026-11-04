"""
The retargeter pool: one worker PROCESS per active variant (see variants.py
for why a process and not a thread), fanned the same frames, latest result
kept per variant for the UI and the headset's robot_state mirror.

ONE WRITER PER PIPE. Each worker has a dedicated sender thread, the only code
that ever writes its Connection; callers (the gateway's frame callback, the
replay player, API requests) only post into a latest-wins frame slot or an
ordered control queue and never block. Both rules were learned the hard way:
  * two threads sending on one Connection interleave their bytes; the worker
    then reads a bogus length and blocks forever (clicking Replay did it: the
    API thread's reset raced the player thread's seek) — a permanent freeze;
  * a worker rebuilding its engine on reset (~2.5 s) lets its pipe fill, and a
    blocking send stalled the replay player for seconds on every loop.
A reset clears the slot, so no frame posted before it is delivered after it.
"""
from __future__ import annotations

import multiprocessing as mp
import threading
import time

from . import variants as V
from .worker import run_worker


class Pool:
    def __init__(self):
        self._ctx = mp.get_context("spawn")
        self._lock = threading.Lock()
        self.workers = {}          # vid -> {proc, conn, ready, error, latest, n, t_rx}
        self._seq = 0
        from . import config as _CFG
        self.settings = {"scope": _CFG.SCOPE if _CFG.BASE_FOLLOW else "local", "robot": _CFG.BASE_PROFILE}

    # ── lifecycle ──────────────────────────────────────────────────────────
    def set_active(self, vids):
        vids = [v for v in vids if v in V.VARIANTS]
        with self._lock:
            for v in [v for v in self.workers if v not in vids]:
                self._stop(v)
            for v in vids:
                if v not in self.workers:
                    self._start(v)
        return self.active()

    def restart(self, vids):
        """Fresh workers for `vids` (their spec changed on a catalogue reload)."""
        with self._lock:
            for v in [v for v in vids if v in self.workers]:
                self._stop(v)
                self._start(v)

    def active(self):
        return list(self.workers)

    def _start(self, vid):
        a, b = self._ctx.Pipe()
        p = self._ctx.Process(target=run_worker, args=(vid, b), daemon=True,
                              name=f"rtl-{vid}")
        p.start()
        w = {"proc": p, "conn": a, "ready": False, "error": None, "latest": None,
             "n": 0, "t_rx": None, "rate": 0.0, "since": time.time(),
             "cv": threading.Condition(), "slot": None, "ctl": [], "alive": True,
             "dropped": 0}
        self.workers[vid] = w
        threading.Thread(target=self._reader, args=(vid, w), daemon=True).start()
        threading.Thread(target=self._sender, args=(vid, w), daemon=True).start()
        self._post_ctl(w, ("set", dict(self.settings)))

    def _stop(self, vid):
        w = self.workers.pop(vid, None)
        if not w:
            return
        self._post_ctl(w, ("stop",))
        w["proc"].join(timeout=1.5)
        if w["proc"].is_alive():
            w["proc"].terminate()

    def shutdown(self):
        with self._lock:
            for v in list(self.workers):
                self._stop(v)

    def _reader(self, vid, w):
        conn = w["conn"]
        while True:
            try:
                msg = conn.recv()
            except (EOFError, OSError):
                if self.workers.get(vid) is w and not w["error"]:
                    w["error"] = "worker exited"
                return
            kind = msg[0]
            if kind == "ready":
                w["ready"] = True
                w["info"] = msg[2] if len(msg) > 2 else {}
            elif kind == "error":
                w["error"] = msg[2].strip().splitlines()[-1][:300]
            elif kind == "rec":
                now = time.time()
                if w["t_rx"] is not None:
                    dt = now - w["t_rx"]
                    if dt > 0:
                        w["rate"] = 0.9 * w["rate"] + 0.1 / dt
                w["t_rx"] = now
                w["latest"] = (msg[2], msg[3])
                w["n"] += 1

    def _sender(self, vid, w):
        """The ONLY writer of this worker's pipe."""
        conn, cv = w["conn"], w["cv"]
        while True:
            with cv:
                while w["alive"] and not w["ctl"] and w["slot"] is None:
                    cv.wait(0.5)
                ctl, w["ctl"] = w["ctl"], []
                frame, w["slot"] = w["slot"], None
            try:
                for m in ctl:
                    conn.send(m)
                    if m[0] == "stop":
                        return
                if frame is not None:
                    conn.send(frame)
            except Exception:
                return
            if not w["alive"]:
                return

    def _post_ctl(self, w, msg):
        with w["cv"]:
            w["slot"] = None                     # nothing from before a reset/stop
            if msg[0] in ("reset", "reanchor") and w["ctl"] and w["ctl"][-1][0] == msg[0]:
                w["cv"].notify()                 # a reset already pending covers this one (each is an
                return                           # engine rebuild, ~1 s: queued ones were a backlog)
            w["ctl"].append(msg)
            if msg[0] == "stop":
                w["alive"] = False
            w["cv"].notify()

    # ── frames ─────────────────────────────────────────────────────────────
    def submit(self, t, frame):
        """Post one frame {kind, p} at engine clock t [s] to every worker.
        Never blocks: an undelivered older frame is replaced (counted)."""
        self._seq += 1
        msg = ("frame", self._seq, t, frame)
        for w in list(self.workers.values()):
            if not w["ready"]:
                continue
            with w["cv"]:
                if w["slot"] is not None:
                    w["dropped"] += 1
                w["slot"] = msg
                w["cv"].notify()
        return self._seq

    def configure(self, **kw):
        """Live switches for every worker (and every worker started later):
        scope "global" (whole body) / "local" (upper body), robot "mabel" / "nova",
        ablation channels torso / neck / lift (config.CHANNELS), end effector "orca" / "gripper"."""
        self.settings.update({k: v for k, v in kw.items() if v is not None})
        for w in list(self.workers.values()):
            with w["cv"]:                         # not _post_ctl: a switch keeps the pending frame
                w["ctl"].append(("set", dict(self.settings)))
                w["cv"].notify()
        return dict(self.settings)

    def reset(self):
        cb = getattr(self, "on_reset", None)
        if cb:
            cb()
        for w in list(self.workers.values()):
            self._post_ctl(w, ("reset",))
            w["latest"] = None

    def reanchor(self):
        """Fast live re-anchor; discard frames queued before the request."""
        cb = getattr(self, "on_reset", None)
        if cb:
            cb()
        for w in list(self.workers.values()):
            self._post_ctl(w, ("reanchor",))
            w["latest"] = None

    def snapshot(self):
        out = {}
        for vid, w in list(self.workers.items()):
            lt = w["latest"]
            out[vid] = {"ready": w["ready"], "error": w["error"], "n": w["n"],
                        "ee_mount": (w.get("info") or {}).get("ee_mount"),
                        "dropped": w["dropped"] + ((w["latest"] or (0, {}))[1].get("dropped") or 0),
                        "hz": round(w["rate"], 1),
                        "seq": lt[0] if lt else None,
                        "rec": lt[1] if lt else None}
        return out
