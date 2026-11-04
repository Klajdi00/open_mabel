"""
Vision Pro teleop node for the Control Lab.

When enabled it binds the STANDARD teleop port (:9090, ws path /teleop) and
speaks just enough of the bridge wire protocol that the unmodified MABEL
Vision Pro / iOS apps connect straight to this Mac and stream whole-body
tracking (head, both hand skeletons incl. wrists, pinch, elbows when the app
sends them). Frames are converted at ingest (mabel.sources.wire →
OperatorPose) and handed to SimWorld.

The main sim bridge also uses :9090 — stop it first; bind failure is reported
to the UI instead of crashing the studio.

A MOCK source (mabel.sources.mock at 60 Hz) drives the same ingest path for
headset-free testing, and the operator calibration profile
(whole_body_retargeting.OperatorProfile JSON cache) is loaded on demand.
"""
from __future__ import annotations

import asyncio
import json
import threading
import time

from . import config as CFG

import sys
sys.path.insert(0, str(CFG.CONTROLLER_DIR))

from mabel.types import OperatorPose                       # noqa: E402
from mabel.sources import mock_frame                       # noqa: E402
from mabel.whole_body_retargeting import OperatorProfile   # noqa: E402


class VPNode:
    """Owns the operator-frame sources: 'headset' (WS server) or 'mock'."""

    def __init__(self, world):
        self.world = world
        self.mode = "off"            # off | headset | mock
        self.error = None
        self._server_thread = None
        self._mock_thread = None
        self._stop = threading.Event()
        self._loop = None
        self.clients = 0
        self._roster = {}            # ws -> {device, ua, since, frames, last, body}
        self._active = None          # the ws whose frames drive the bench

    # ── control ────────────────────────────────────────────────────────────
    def set_source(self, mode):
        if mode not in ("off", "headset", "mock", "avp"):
            raise ValueError(f"unknown VP source {mode!r}")
        self.stop()
        self.mode = mode
        self.error = None
        if mode == "headset":
            self._stop.clear()
            self._server_thread = threading.Thread(
                target=self._serve, daemon=True, name="vp-node")
            self._server_thread.start()
        elif mode == "mock":
            self._stop.clear()
            self._mock_thread = threading.Thread(
                target=self._mock, daemon=True, name="vp-mock")
            self._mock_thread.start()
        elif mode == "avp":
            self._stop.clear()
            self._mock_thread = threading.Thread(
                target=self._avp, daemon=True, name="vp-zmq")
            self._mock_thread.start()
        self.world.vp_status["source"] = mode

    def stop(self):
        self._stop.set()
        if self._loop is not None:
            try:
                self._loop.call_soon_threadsafe(lambda: None)
            except RuntimeError:
                pass
        for th in (self._server_thread, self._mock_thread):
            if th is not None:
                th.join(timeout=5.0)
                if th.is_alive():
                    self.error = ("previous teleop server is still shutting "
                                  "down — retry in a moment")
        self._server_thread = self._mock_thread = None
        self._roster.clear()
        self._active = None
        self._loop = None
        self.mode = "off"
        self.world.vp_status["connected"] = False
        self.world.vp_status["source"] = "off"

    @staticmethod
    def _device_of(ua):
        u = (ua or "").lower()
        if "visionos" in u or "vision" in u or "mabelteleop" in u:
            return "Vision Pro"
        if "iphone" in u or "ios" in u or "mabel mobile" in u:
            return "iPhone"
        if "mozilla" in u or "chrome" in u or "safari" in u:
            return "Web client"
        return "Device"

    def status(self):
        import time as _t
        now = _t.time()
        roster = [{"device": m["device"], "frames": m["frames"],
                   "ageS": round(now - m["last"], 1) if m["last"] else None,
                   "body": m["body"], "active": ws is self._active}
                  for ws, m in list(self._roster.items())]
        active = next((r for r in roster if r["active"]), None)
        return {"mode": self.mode, "error": self.error, "clients": self.clients,
                "port": CFG.VP_PORT, "roster": roster,
                "activeDevice": active["device"] if active else None}

    # ── headset source: minimal bridge-protocol WS server ──────────────────
    def _serve(self):
        import websockets

        def _arbitrate(ws, body, now):
            """Auto-select the driving device: first-come, body-tracking beats
            nav-only, and a stale active device is replaced."""
            act = self._active
            if act is not None and act not in self._roster:
                act = self._active = None
            if act is None:
                self._active = ws
                self.world.status_msg = (
                    f"input: {self._roster[ws]['device']} connected")
                return
            if ws is act:
                return
            am = self._roster[act]
            stale = (now - am["last"]) > 1.5
            richer = body and not am["body"]
            if stale or richer:
                self._active = ws
                self.world.status_msg = (
                    f"input: switched to {self._roster[ws]['device']}"
                    + (" (previous stale)" if stale else " (body tracking)"))

        async def handler(ws):
            ua = ""
            try:                        # all UA values (proxies/tools may add one)
                ua = ", ".join(ws.request_headers.get_all("User-Agent"))
            except Exception:
                try:
                    ua = ws.request_headers.get("User-Agent", "")
                except Exception:
                    pass
            meta = {"device": self._device_of(ua), "ua": ua,
                    "since": time.time(), "frames": 0, "last": 0.0,
                    "body": False}
            self._roster[ws] = meta
            self.clients += 1
            try:
                await ws.send(json.dumps({"type": "hello", "payload": {
                    "server": "control_lab", "role": "test-bench"}}))
                async for raw in ws:
                    try:
                        msg = json.loads(raw)
                    except (TypeError, ValueError):
                        continue
                    t = msg.get("type")
                    p = msg.get("payload") or {}
                    if t == "ping":
                        await ws.send(json.dumps(
                            {"type": "pong", "payload": p}))
                    elif t == "teleop_frame":
                        now = time.time()
                        body = bool(p.get("head") or p.get("leftHand")
                                    or p.get("rightHand"))
                        meta["frames"] += 1
                        meta["last"] = now
                        meta["body"] = meta["body"] or body
                        _arbitrate(ws, body, now)
                        if ws is not self._active:
                            continue        # only the active device drives
                        nav = p.get("navJoystick")
                        if nav:
                            self.world.nav.update(
                                {k: float(nav.get(k, 0.0))
                                 for k in ("lx", "ly", "rx", "ry")})
                        if body:
                            try:
                                pose = OperatorPose.from_teleop_frame(p)
                            except Exception:
                                continue
                            self.world.ingest_pose(pose, meta["device"])
            finally:
                self.clients -= 1
                self._roster.pop(ws, None)
                if self._active is ws:
                    self._active = None

        async def main():
            server = None
            for attempt in range(4):        # ride out a slow prior release
                try:
                    server = await websockets.serve(
                        handler, "0.0.0.0", CFG.VP_PORT,
                        max_size=8 * 1024 * 1024, close_timeout=1.0)
                    break
                except OSError as e:
                    if attempt == 3:
                        self.error = (f"port {CFG.VP_PORT} busy ({e}). Stop "
                                      "the sim bridge (start_teleop.sh) or "
                                      "wait a moment and retry.")
                        self.mode = "off"
                        return
                    await asyncio.sleep(0.7)
            try:
                while not self._stop.is_set():
                    await asyncio.sleep(0.2)
            finally:
                server.close()
                try:
                    await asyncio.wait_for(server.wait_closed(), timeout=3.0)
                except asyncio.TimeoutError:
                    pass

        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(main())
        finally:
            self._loop.close()

    # ── mock source: synthetic operator @ 60 Hz ─────────────────────────────
    def _mock(self):
        # Head + circling wrists from mock_frame; NAMED hand skeletons (which
        # the finger retargeters require) from the calibration synthesizer —
        # cycled so the fingers open/close over time.
        from mabel.whole_body_retargeting.calibration import (synthesize_pose,
                                                              POSE_HANDS)
        hand_frames = synthesize_pose(POSE_HANDS, n=48, noise=0.0005)
        t0 = time.perf_counter()
        i = 0
        while not self._stop.is_set():
            t = time.perf_counter() - t0
            pose = OperatorPose.from_avframe(mock_frame(t))
            hf = hand_frames[i % len(hand_frames)]
            pose.left_joints = hf.left_joints
            pose.right_joints = hf.right_joints
            i += 1
            self.world.ingest_pose(pose, "Mock operator")
            time.sleep(1.0 / 60.0)

    def _avp(self):
        # DIRECT Vision-Pro ZMQ ingest: subscribe to the stream published by
        # simulation/mabel_mujoco/scripts/vision_pro_stream.py (PUB, conflated,
        # tcp://127.0.0.1:5556 = mabel.config.AVP_ZMQ_ADDR; override with
        # MABEL_CTS_AVP_ADDR). Same OperatorPose seam as the :9090 headset path
        # — the lab then retargets AND overlays the operator skeleton, so the
        # whole-body retargeting is debuggable against the live stream.
        import os
        try:
            from mabel.sources import AVPReceiver
        except Exception as e:                              # zmq missing etc.
            self.error = f"AVP stream source unavailable: {e}"
            self.mode = "off"
            self.world.vp_status["source"] = "off"
            return
        addr = os.environ.get("MABEL_CTS_AVP_ADDR", None)
        try:
            rx = AVPReceiver(addr) if addr else AVPReceiver()
        except Exception as e:
            self.error = f"AVP stream connect failed: {e}"
            self.mode = "off"
            self.world.vp_status["source"] = "off"
            return
        last_ts = None
        try:
            while not self._stop.is_set():
                frame = rx.poll()
                ts = getattr(frame, "timestamp", None) if frame is not None else None
                if frame is not None and ts != last_ts:
                    last_ts = ts
                    self.world.ingest_pose(OperatorPose.from_avframe(frame),
                                           "Vision Pro (ZMQ stream)")
                time.sleep(1.0 / 120.0)         # poll ≥ stream rate; conflated
        finally:
            rx.close()

    # ── calibration ─────────────────────────────────────────────────────────
    def load_calibration(self, user=CFG.DEFAULT_USER):
        """Try the OperatorProfile JSON caches; report what was found."""
        for d in CFG.PROFILE_DIRS:
            try:
                prof = OperatorProfile.load(user, str(d))
            except Exception:
                prof = None
            if prof is not None:
                self.world.apply_profile(prof)
                return {"ok": True, "kind": "operator_profile",
                        "dir": str(d), "user": user,
                        "arm_length_m": getattr(prof, "arm_length_m", None)}
        # fall back to the AVP reach profile (scalar reach only — informative)
        try:
            reach = json.loads(CFG.AVP_REACH_PROFILE.read_text())
            self.world.vp_status["calibration"] = (
                f"reach-only ({reach.get('human_reach')})")
            return {"ok": False, "kind": "reach_only", "profile": reach,
                    "note": "no OperatorProfile JSON found; retargeting uses "
                            "the default RETARGET_SCALE"}
        except Exception:
            return {"ok": False, "kind": "none",
                    "note": "no calibration found; defaults in use"}
