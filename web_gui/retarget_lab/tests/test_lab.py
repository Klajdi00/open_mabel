"""Retarget Lab contracts.

  * the kinematic retargeter is DETERMINISTIC (same clip, same clock -> same q)
    and its ideal servo really makes qpos == the command;
  * a recorded episode round-trips losslessly and replays on a zero-based,
    strictly increasing clock;
  * wire frames and EgoDex payloads produce the same OperatorPose (so the lab's
    recordings are as good an input as the corpus);
  * the gateway speaks enough of the app protocol (hello caps, pong,
    robot_state, record from the headset) — exercised with the fake headset;
  * variants are process-isolated: configuring one never leaks into another.
"""
import asyncio
import gzip
import json
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from retarget_lab.server import config as CFG

CLIPS = sorted(CFG.EGODEX_CLIPS.glob("*.json.gz"))
needs_clips = pytest.mark.skipif(not CLIPS, reason="EgoDex benchmark clips not present (local-only corpus)")


def _clip():
    return json.load(gzip.open(CLIPS[0]))["payloads"]


@needs_clips
def test_kinematic_retargeter_deterministic_and_ideal(monkeypatch):
    from retarget_lab.server import config as CFG
    from retarget_lab.server.worker import KinematicRetargeter, pose_from_frame
    monkeypatch.setattr(CFG, "SERVO", "ideal")
    pay = _clip()[:40]
    outs = []
    for _ in range(2):
        kr = KinematicRetargeter("ours")
        recs = [kr.step(pose_from_frame({"kind": "avp", "p": p}), p["timestamp"]) for p in pay]
        outs.append(np.array([[r["q"][k] for k in sorted(r["q"])] for r in recs]))
    assert np.allclose(outs[0], outs[1]), "same clip + same clock must give the same joints"
    # ideal servo: every position-servo joint sits exactly on its (clipped) command
    d = kr.data
    for a, qa, lo, hi, lim in kr._servo:
        c = float(d.ctrl[a])
        assert d.qpos[qa] == pytest.approx(min(max(c, lo), hi) if lim else c)
    assert recs[-1]["anchored"], "engine never anchored on a steady clip"


@needs_clips
def test_reset_equals_fresh():
    """reset() reuses the loaded model; it must reproduce a fresh engine exactly."""
    from retarget_lab.server.worker import KinematicRetargeter, pose_from_frame
    pay = _clip()[:30]
    run = lambda kr: np.array([[r["q"][k] for k in sorted(r["q"])] for r in
                               (kr.step(pose_from_frame({"kind": "avp", "p": p}), p["timestamp"]) for p in pay)])
    kr = KinematicRetargeter("ours")
    a = run(kr)
    kr.reset()
    assert np.array_equal(a, run(kr))


@needs_clips
def test_wire_and_payload_give_same_pose():
    sys.path.insert(0, str(CFG.PKG / "tools"))
    from fake_vp import to_wire
    from retarget_lab.server.worker import pose_from_frame
    p = _clip()[10]
    a = pose_from_frame({"kind": "avp", "p": p})
    b = pose_from_frame({"kind": "wire", "p": to_wire(p, 1, 0.0)["payload"]})
    assert np.allclose(a.head, b.head)
    for s in ("left", "right"):
        ja, jb = getattr(a, f"{s}_joints"), getattr(b, f"{s}_joints")
        for k in ja:
            assert np.allclose(ja[k], jb[k]), k


def test_episode_roundtrip(tmp_path, monkeypatch):
    from retarget_lab.server import episodes
    monkeypatch.setattr(CFG, "EPISODES_DIR", tmp_path)
    rec = episodes.Recorder()
    meta = rec.start(name="unit test!")
    assert meta["id"].endswith("unit_test")
    frames = [{"sequence": i, "timestamp": 100.0 + (i // 2) * 0.016,     # duplicate stamps on purpose
               "head": {"transform": {"matrix": list(np.eye(4).ravel())}, "trackingState": "normal"}}
              for i in range(10)]
    for i, f in enumerate(frames):
        rec.add(f, 5000.0 + i * 0.016)
    m = rec.stop()
    assert m["frames"] == 10
    meta2, loaded = episodes.load(m["id"])
    assert [f["p"] for f in loaded] == frames                      # lossless
    t = [f["t"] for f in loaded]
    assert t[0] == 0.0 and all(b > a for a, b in zip(t, t[1:]))   # zero-based, strictly increasing
    assert any(e["id"] == m["id"] for e in episodes.list_episodes())
    episodes.delete(m["id"])
    with pytest.raises(FileNotFoundError):
        episodes.delete("../../etc")                               # never escapes the folder


def test_variant_isolation():
    """configure() of a 'published' variant rewrites module knobs — only in ITS process."""
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    with ProcessPoolExecutor(2, mp_context=mp.get_context("spawn")) as ex:
        a = ex.submit(_knobs_after, "gmr").result()
        b = ex.submit(_knobs_after, "ours").result()
    assert a["ARM_WS_ASSIST"] is False and a["table"] is False
    assert b["ARM_WS_ASSIST"] is True and b["table"] is True


def _knobs_after(vid):
    sys.path.insert(0, str(CFG.CONTROLLER_DIR)); sys.path.insert(0, str(CFG.PKG.parent))
    from retarget_lab.server.worker import KinematicRetargeter
    import mabel.teleop_engine.engine as ENG
    kr = KinematicRetargeter(vid)
    return {"ARM_WS_ASSIST": bool(ENG.ARM_WS_ASSIST), "table": kr.eng.arm["left"].table is not None}


def _free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


@needs_clips
def test_gateway_with_fake_headset(tmp_path, monkeypatch):
    """The fake headset connects, gets hello{caps}, pongs, robot_state, and its
    record button produces an episode; every retargeter receives the frames."""
    monkeypatch.setattr(CFG, "VP_PORT", _free_port())
    monkeypatch.setattr(CFG, "BONJOUR", False)
    monkeypatch.setattr(CFG, "EPISODES_DIR", tmp_path)
    monkeypatch.setattr(CFG, "PHYSICS_PREWARM", False)
    from retarget_lab.server.lab import Lab
    lab = Lab()
    lab.start(active=["ours", "direct"])
    try:
        t0 = time.time()
        while time.time() - t0 < 60 and not all(w["ready"] for w in lab.pool.workers.values()):
            time.sleep(0.2)
        sys.path.insert(0, str(CFG.PKG / "tools"))
        import fake_vp

        class A:
            host, port, clip, seconds, record = "127.0.0.1", CFG.VP_PORT, None, 4.0, 2.0
        asyncio.run(fake_vp.main(A()))
        time.sleep(0.5)
        snap = lab.pool.snapshot()
        assert all(v["n"] > 100 for v in snap.values()), snap
        assert snap["ours"]["rec"]["anchored"]
        eps = list(tmp_path.iterdir())
        assert len(eps) == 1 and json.loads((eps[0] / "meta.json").read_text())["frames"] > 60
        st = lab.robot_state()
        assert "left_arm_1" in st["jointPositions"]
    finally:
        lab.shutdown()


@needs_clips
def test_pool_survives_concurrent_reset_and_submit():
    """Regression: the API thread's reset raced the player's frames on ONE pipe and
    corrupted it (permanent freeze). Hammer it from three threads; the worker must
    keep answering and the callers must never block."""
    import threading
    from retarget_lab.server.pool import Pool
    pay = _clip()
    pool = Pool()
    pool.set_active(["direct"])
    try:
        w = pool.workers["direct"]
        t0 = time.time()
        while not w["ready"] and time.time() - t0 < 60:
            time.sleep(0.1)
        stop = threading.Event()
        worst = [0.0]

        def feed():
            i = 0
            while not stop.is_set():
                a = time.perf_counter()
                pool.submit(i / 60, {"kind": "avp", "p": pay[i % len(pay)]})
                worst[0] = max(worst[0], time.perf_counter() - a)
                i += 1
                time.sleep(1 / 120)

        def resets():
            while not stop.is_set():
                pool.reset()
                time.sleep(0.3)
        ths = [threading.Thread(target=feed), threading.Thread(target=feed), threading.Thread(target=resets)]
        for t in ths:
            t.start()
        time.sleep(4)
        stop.set()
        for t in ths:
            t.join()
        n0 = w["n"]
        for i in range(60):                              # calm traffic afterwards
            pool.submit(100 + i / 30, {"kind": "avp", "p": pay[i % len(pay)]})
            time.sleep(1 / 30)
        time.sleep(1.5)
        assert w["n"] > n0, "worker stopped answering after concurrent reset/submit"
        assert worst[0] < 0.05, f"submit blocked for {worst[0]*1e3:.0f} ms"
        assert w["error"] is None
    finally:
        pool.shutdown()


@needs_clips
def test_slew_servo_never_exceeds_the_robot_slews(monkeypatch):
    """Physical servo (default): no joint moves faster than its robot's setpoint slew,
    lift included (0.03 m/s), for both robots."""
    from retarget_lab.server import config as CFG
    from retarget_lab.server.worker import KinematicRetargeter, pose_from_frame
    pay = _clip()[:90]
    for robot in ("mabel", "nova"):
        kr = KinematicRetargeter("ours")
        kr.apply_settings({"robot": robot})
        sl = CFG.SLEWS[robot]
        prev, prev_t = None, None
        for p in pay:
            kr.step(pose_from_frame({"kind": "avp", "p": p}), p["timestamp"])
            d = kr.data
            q = {qa: float(d.qpos[qa]) for qa in kr._slew_key}
            lift = sum(float(d.qpos[x]) for x in kr._lift_q)
            if prev is not None:
                dt = kr.eng._frame_dt
                for qa, (k, i) in kr._slew_key.items():
                    v = sl["arm"][i] if k == "arm" else sl[k]
                    assert abs(q[qa] - prev[0][qa]) <= v * dt + 1e-9, (robot, k, i)
                assert abs(lift - prev[1]) <= sl["lift"] * dt + 1e-9
            prev = (q, lift)


def test_end_effector_reaches_the_headset():
    """The End effector toggle reaches the apps the way NOVA's gateway reports it: hello.hands per side
    ("v1" ORCA / "gripper"), and with the gripper robot_state carries `<side>_gripper` = ONE finger's opening
    (the engine's 0..0.088 m jaw opening / 2), so the headset draws the same gripper as the studio."""
    from types import SimpleNamespace
    from retarget_lab.server.gateway import Gateway
    from retarget_lab.server.lab import Lab
    rec = {"q": {"left_arm_1": 0.1}, "gripper": {"left": 0.088, "right": 0.02}}
    pool = SimpleNamespace(settings={"ee": "orca"}, workers={"m": {"latest": (0, rec)}})
    lab = SimpleNamespace(pool=pool, mirror="m", recorder=SimpleNamespace(active=False))
    gw = Gateway(lab)
    meta = {"device": "visionpro"}
    assert gw._hello(meta)["payload"]["hands"] == {"left": "v1", "right": "v1"}
    assert "left_gripper" not in Lab.robot_state(lab)["jointPositions"]
    pool.settings["ee"] = "gripper"
    assert gw._hello(meta)["payload"]["hands"] == {"left": "gripper", "right": "gripper"}
    q = Lab.robot_state(lab)["jointPositions"]
    assert q["left_gripper"] == 0.044 and q["right_gripper"] == 0.01 and q["left_arm_1"] == 0.1
    assert "left_gripper" not in rec["q"]                    # the worker's record is not mutated


def test_robot_state_recording_is_the_apps_shape():
    """2026-10-28: robot_state.recording must be NOVA's {isRecording, task, ...} dict — a bare bool is unparseable to
    the Vision Pro app, whose REC button then never follows the lab (first Stop tap after a studio start = START)."""
    from types import SimpleNamespace
    from retarget_lab.server.lab import Lab
    pool = SimpleNamespace(settings={"ee": "orca"}, workers={})
    for active in (False, True):
        rec = SimpleNamespace(active=active, meta={"name": "floor test"} if active else None,
                              status=lambda a=active: {"active": a, "id": "x" if a else None, "duration_s": 2.5 if a else 0})
        lab = SimpleNamespace(pool=pool, mirror="m", recorder=rec, _last_task="floor test")
        r = Lab.robot_state(lab)["recording"]
        assert isinstance(r, dict) and r["isRecording"] is active and r["task"] == "floor test"
        assert r["elapsed"] == (2.5 if active else 0.0)
