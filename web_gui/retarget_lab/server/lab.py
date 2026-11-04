"""
The lab: one operator stream, many retargeters.

Sources (exactly one drives the retargeters at a time):
  live    frames from the Vision Pro gateway (:9090), real time
  replay  a recorded episode (or an EgoDex clip) played back on its own clock —
          pause / seek / speed / loop

Recording is independent of the source: while recording, every live frame the
headset sends is written raw (episodes.Recorder), whether or not the live
stream is the one being retargeted.
"""
from __future__ import annotations

import threading
import time

from . import episodes, human
from .gateway import Gateway
from .pool import Pool
from . import variants as V
from .variants import DEFAULT_ACTIVE
from .worker import pose_from_frame



def _recording_status(self):
    """robot_state.recording in the shape the apps read (NOVA's dataset/recorder.status): {isRecording, task,
    elapsed, ...}. Until 2026-10-28 this was a bare bool, which the Vision Pro app cannot parse
    (`p["recording"] as? [String: Any]`), so its REC button never followed the lab: a take started in the studio
    left the headset showing "not recording", its first Stop tap sent START (a no-op while recording) and only
    the second one stopped — "stop recording won't respond for a long time"."""
    rec = self.recorder
    st = rec.status() if hasattr(rec, "status") else {"active": bool(getattr(rec, "active", False))}
    active = bool(st.get("active"))
    meta = getattr(rec, "meta", None) or {}
    return {"isRecording": active,
            "task": (meta.get("name") if active else None) or getattr(self, "_last_task", ""),
            "elapsed": float(st.get("duration_s", 0.0)) if active else 0.0,
            "episodeId": st.get("id"), "error": None}

class Player:
    """Plays a loaded episode into the pool on the episode's own clock."""

    def __init__(self, lab):
        self.lab = lab
        self.meta = None
        self.frames = []
        self.i = 0
        self.playing = False
        self.speed = 1.0
        self.loop = True
        self._wake = threading.Event()
        self._jump = None
        threading.Thread(target=self._run, daemon=True, name="rtl-player").start()

    def load(self, eid):
        meta, frames = episodes.load(eid)
        self.playing = False
        self.meta, self.frames, self.i = meta, frames, 0
        self._jump = 0
        self._wake.set()
        return meta

    def play(self, on=True):
        if self.frames:
            self.playing = bool(on)
            self._wake.set()

    def seek(self, frac):
        if self.frames:
            self._jump = int(max(0.0, min(1.0, frac)) * (len(self.frames) - 1))
            self._wake.set()

    def status(self):
        n = len(self.frames)
        return {"loaded": self.meta is not None, "id": (self.meta or {}).get("id"),
                "name": (self.meta or {}).get("name"), "i": self.i, "n": n,
                "t": self.frames[self.i]["t"] if n else 0.0,
                "duration": self.frames[-1]["t"] if n else 0.0,
                "playing": self.playing, "speed": self.speed, "loop": self.loop}

    def _run(self):
        # engine clock: monotone across loops/seeks so no engine ever sees time
        # run backwards (a seek also RESETS the engines — a jump in the operator
        # is not a motion the retargeter should smooth across)
        clock = 0.0
        while True:
            if self._jump is not None:
                self.i, self._jump = self._jump, None
                self.lab.pool.reset()
                clock += 1.0
                if self.lab.source == "replay" and self.frames:
                    fr = self.frames[self.i]
                    self.lab._submit(clock, fr)
            if not (self.playing and self.frames and self.lab.source == "replay"):
                self._wake.wait(0.1)
                self._wake.clear()
                continue
            fr = self.frames[self.i]
            self.lab._submit(clock, fr)
            if self.i + 1 >= len(self.frames):
                if self.loop:
                    self._jump = 0
                else:
                    self.playing = False
                continue
            dt = self.frames[self.i + 1]["t"] - fr["t"]
            dt = min(max(dt, 0.0), 0.5)
            clock += dt
            self.i += 1
            time.sleep(dt / max(self.speed, 0.05))


class Lab:
    def __init__(self):
        self.pool = Pool()
        self._sk_anchor = None
        from .physics import PhysicsLink
        self.physics = PhysicsLink(self)

        def _on_reset():
            self._sk_anchor = None                  # re-anchor the drawn operator with the engines
            self.physics.on_reset()                 # and NOVA's gateway (physics on)
        self.pool.on_reset = _on_reset
        self.recorder = episodes.Recorder()
        self.gateway = Gateway(self)
        self.player = Player(self)
        self.source = "live"
        self.mirror = DEFAULT_ACTIVE[0]          # whose joints the headset sees
        self.latest_frame = None                 # {kind, p, t} that drove the pool
        self.live_frames = 0
        self._ts_off = None                      # headset stamp -> lab clock
        self._live_t = -1.0
        self._t0 = time.monotonic()
        self.status_msg = ""

    def start(self, active=None):
        self.set_active(active or DEFAULT_ACTIVE)    # mirror falls back to the first active variant
        self.gateway.start()
        from . import config as _C
        if _C.PHYSICS_PREWARM:
            self.physics.prewarm()

    def shutdown(self):
        self.recorder.stop()
        self.physics.shutdown()
        self.pool.shutdown()
        self.gateway.stop()

    # ── frames ─────────────────────────────────────────────────────────────
    def on_frame(self, payload, rt, device):
        """Gateway callback, one per teleop_frame from the active headset."""
        self.live_frames += 1
        self._watch_head(payload, rt)
        if self.recorder.active:
            self.recorder.add(payload, rt)
        if self.source == "live":
            self._submit(self._live_clock(payload), {"kind": "wire", "p": payload})

    def _watch_head(self, payload, rt):
        """Flag a STUCK head: the same transform, state "normal", for > 2 s while a hand is tracked. Whole body
        (base, lean, lift, the arms' body frame) follows the head, so a stuck head parks the base and skews the arms.
        Seen 2026-10-23: the VP app froze the head after Stop/Home (fixed in RobotController.makeFrameSource)."""
        h = payload.get("head") or {}
        m = (h.get("transform") or {}).get("matrix")
        hands = any((payload.get(k) or {}).get("isTracked") for k in ("leftHand", "rightHand"))
        if m is None or h.get("trackingState") != "normal" or not hands:
            return
        if m != getattr(self, "_head_m", None):
            self._head_m, self._head_since = m, rt
            if self.status_msg.startswith("headset head is frozen"):
                self.status_msg = ""
        elif rt - self._head_since > 2.0 and not self.status_msg.startswith("headset head is frozen"):
            self.status_msg = "headset head is frozen (identical pose > 2 s) — whole body cannot follow; restart the VP app"

    def _live_clock(self, payload):
        """Engine time [s] for a live frame: the headset's CAPTURE stamp, mapped
        onto the lab clock — never the arrival time. Wi-Fi delivers the 60 Hz
        stream in bursts (measured on a 69 s take: 75 arrival gaps of ~0.10 s
        while the headset stamps stayed at 16.7 ms), so on the arrival clock the
        frames of a burst read as ~1 ms apart and the arm's trust region
        (rate x dt) lets the arm barely move on them. The offset is re-latched on
        a reconnect / clock jump (|drift| > 0.5 s); the result is strictly
        increasing either way."""
        now = time.monotonic() - self._t0
        try:
            ts = float(payload.get("timestamp"))
        except (TypeError, ValueError):
            ts = None
        if ts is None:
            t = now
        else:
            t = ts + self._ts_off if self._ts_off is not None else None
            if t is None or abs(t - now) > 0.5:
                self._ts_off = now - ts
                t = now
        t = max(t, self._live_t + 1e-4)
        self._live_t = t
        return t

    def _submit(self, t, frame):
        self.latest_frame = frame
        self.pool.submit(t, {"kind": frame["kind"], "p": frame["p"]})
        if frame["kind"] == "wire":                 # NOVA's gateway speaks the headset wire only
            self.physics.forward(frame["p"])

    # ── commands ───────────────────────────────────────────────────────────
    def set_source(self, src):
        if src not in ("live", "replay"):
            raise ValueError(src)
        if src != self.source:
            self.source = src
            self.player.playing = False
            self.pool.reset()
        return src

    def set_active(self, vids):
        act = self.pool.set_active(vids)
        if self.mirror not in act and act:
            self.mirror = act[0]
        return act

    def record(self, on, name="", device="", notes=""):
        if name:
            self._last_task = name                       # robot_state.recording.task keeps the headset's name
        if on:
            dev = device or next((c["device"] for c in self.gateway.status()["clients"]
                                  if c["active"]), "")
            return self.recorder.start(name=name, device=dev, notes=notes)
        meta = self.recorder.stop()
        if meta:
            self.status_msg = f"saved episode {meta['id']} · {meta['frames']} frames"
        return meta

    def reanchor(self):
        self.pool.reanchor()

    def home(self):
        """Every retargeter back to the HOME keyframe (base at the origin, lift half-extended, arms, torso and
        neck home), engines rebuilt; live tracking re-anchors on the next stable head frame."""
        self.pool.reset()
        self.status_msg = "reset to home"

    # ── views ──────────────────────────────────────────────────────────────
    def robot_state(self):
        """The robot_state the headset gets: the mirror variant's joints."""
        snap = self.pool.workers.get(self.mirror)
        q, base = {}, None
        if snap and snap["latest"]:
            rec = snap["latest"][1]
            q, base = rec["q"], rec.get("base")
            # the gripper on the wire = ONE finger's opening (0..0.044 m), as NOVA streams it; the engine's
            # opening is both jaws (0..0.088). The headset draws it on the rig hello.hands names.
            if self.pool.settings.get("ee") == "gripper" and rec.get("gripper"):
                q = dict(q)
                for s, o in rec["gripper"].items():
                    q[f"{s}_gripper"] = round(0.5 * float(o), 5)
        # `base` = the chassis world pose, in the gateway's shape (x, y, z, yaw,
        # quat [w,x,y,z], steer): without it the headset parks the robot at the
        # origin and the whole-body motion (walk, turn) never shows.
        return {"timestamp": time.time(), "mode": "teleop", "jointPositions": q,
                "base": base or {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0,
                                 "quat": [1.0, 0.0, 0.0, 0.0]},
                "latencyMs": 0.0, "recording": _recording_status(self),
                "battery": {"percentage": 100.0, "voltage": 0.0, "isCharging": False},
                "diagnostics": []}

    def state(self):
        fr = self.latest_frame
        sk = None
        if fr is not None:
            try:
                pose = pose_from_frame(fr)
                if getattr(self, "_sk_anchor", None) is None:
                    self._sk_anchor = human.anchor_of(pose)    # = where the engines calibrate
                sk = human.skeleton(pose, self._sk_anchor)
            except Exception:
                sk = None
        return {"type": "state", "t": time.time(), "source": self.source,
                "mirror": self.mirror, "gateway": self.gateway.status(),
                "recording": self.recorder.status(), "player": self.player.status(),
                "liveFrames": self.live_frames, "human": sk,
                "variants": self.pool.snapshot(), "msg": self.status_msg,
                "settings": dict(self.pool.settings),
                "physics": self.physics.snapshot(),
                "catalog": list(V.VARIANTS)}
