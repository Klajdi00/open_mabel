"""Whole-body lab fixes (2026-10-15, live Vision Pro session):

  1. "the frames are not moving relative to the world" — the base never moved
     and robot_state carried no `base`, so the headset parked the robot at the
     origin. Now the engine runs GLOBAL and its base twist is integrated.
  2. "the retargeting ... sometimes freezes" — a double pinch while grasping
     toggled the engine's clutch (teleop_engine/safety.py) and nothing released
     it: 9.4 s of a 69 s take frozen. The lab runs with the clutch off.
  3. "loses track" (in part) — live frames were clocked by ARRIVAL; Wi-Fi bursts
     made them ~1 ms apart, and the arm trust region (rate x dt) froze the arm on
     them. Live frames are now clocked by the headset's capture stamp.
"""
import copy
import gzip
import json
import time
from types import SimpleNamespace

import numpy as np
import pytest

from retarget_lab.server import config as CFG

CLIPS = sorted(CFG.EGODEX_CLIPS.glob("*.json.gz"))
needs_clips = pytest.mark.skipif(not CLIPS, reason="EgoDex benchmark clips not present (local-only corpus)")
TAKE = CFG.EPISODES_DIR / "20261015_103700_Oct_15th_test1"
needs_take = pytest.mark.skipif(not (TAKE / "frames.jsonl.gz").exists(),
                                reason="the 2026-10-15 live take is local data (gitignored)")


def _shift(mat16, dxyz):
    """Translate one flat 4x4 (either storage order) by dxyz in the world."""
    M = np.asarray(mat16, float).reshape(4, 4)
    if np.allclose(M[3], [0, 0, 0, 1]):          # row-major: translation in column 3
        M[:3, 3] += dxyz
    else:                                          # column-major: translation in row 3
        M[3, :3] += dxyz
    return M.reshape(-1).tolist()


def _walk_payload(p, dxyz):
    q = copy.deepcopy(p)
    for k in ("head_mat", "left_elbow", "right_elbow"):
        if q.get(k):
            q[k] = _shift(q[k], dxyz)
    for k in ("left_hand_mats", "right_hand_mats"):
        if q.get(k):
            q[k] = [_shift(m, dxyz) for m in q[k]]
    for k in ("left_joints", "right_joints"):
        if q.get(k):
            q[k] = {n: _shift(m, dxyz) for n, m in q[k].items()}
    return q


@needs_clips
def _backward(p):
    """Unit vector (ARKit) opposite the operator's horizontal facing in payload p."""
    H = np.asarray(p["head_mat"], float).reshape(4, 4)
    f = -H[:3, 2]
    f[1] = 0.0
    return -f / np.linalg.norm(f)


def test_operator_walking_back_moves_the_base_and_reaches_robot_state():
    """The whole operator steps 0.6 m BACK (ARKit +Z) over 3 s: the robot must
    back up too (chassis front = MJ -X, so backing up = +x), and the headset's
    robot_state must carry that pose."""
    from retarget_lab.server.lab import Lab
    from retarget_lab.server.worker import KinematicRetargeter, pose_from_frame
    p0 = json.load(gzip.open(CLIPS[0]))["payloads"][0]
    kr = KinematicRetargeter("ours")
    rec = None
    bk = _backward(p0)                                     # the operator's OWN backward direction
    for i in range(300):                                   # 1 s still, then 3 s walk... (5 s)
        t = i / 60.0
        back = 0.0 if t < 1.0 else min(0.6, 0.2 * (t - 1.0))
        rec = kr.step(pose_from_frame({"kind": "avp", "p": _walk_payload(p0, back * bk)}), t)
    for i in range(300, 480):                              # 3 s to settle at 0.6 m back
        rec = kr.step(pose_from_frame({"kind": "avp", "p": _walk_payload(p0, 0.6 * bk)}), i / 60.0)
    b = rec["base"]
    assert b["x"] > 0.3, b                                 # followed (1:1 past the 2 cm back box)
    assert abs(b["y"]) < 0.05 and abs(b["yaw"]) < 0.05, b  # straight back, no turn
    # robot_state carries it, in the gateway's shape
    lab = SimpleNamespace(mirror="ours", recorder=SimpleNamespace(active=False),
                          pool=SimpleNamespace(workers={"ours": {"latest": (1, rec)}}))
    st = Lab.robot_state(lab)
    assert st["base"]["x"] == pytest.approx(b["x"]) and len(st["base"]["quat"]) == 4


def test_robot_state_has_a_base_before_any_frame():
    from retarget_lab.server.lab import Lab
    lab = SimpleNamespace(mirror="ours", recorder=SimpleNamespace(active=False),
                          pool=SimpleNamespace(workers={}))
    assert Lab.robot_state(lab)["base"]["quat"] == [1.0, 0.0, 0.0, 0.0]


@needs_take
def test_double_pinch_while_grasping_does_not_freeze():
    """The live take froze at t = 59.95 s (the clutch toggled on a grasp and was
    never released). Replay 55-63 s: the arms must keep moving."""
    from retarget_lab.server import episodes
    from retarget_lab.server.worker import KinematicRetargeter, pose_from_frame
    _, frames = episodes.load(TAKE.name)
    frames = [f for f in frames if 55.0 <= f["t"] <= 63.0]
    kr = KinematicRetargeter("ours")
    prev, still = None, 0.0
    longest = 0.0
    for a, b in zip(frames, frames[1:] + [None]):
        rec = kr.step(pose_from_frame(a), a["t"], collide=False)
        assert not rec["clutched"]
        arm = np.array([v for k, v in sorted(rec["q"].items()) if "arm" in k])
        if prev is not None and np.abs(arm - prev).max() < 1e-6 and b is not None:
            still += b["t"] - a["t"]
            longest = max(longest, still)
        else:
            still = 0.0
        prev = arm
    assert longest < 0.5, f"arms frozen for {longest:.1f} s"


def test_live_clock_follows_headset_stamps_through_a_wifi_burst(monkeypatch):
    """Six frames captured 16.7 ms apart arrive as one burst after a 100 ms
    gap: the engine must see 16.7 ms steps, not ~0."""
    from retarget_lab.server import lab as L
    now = [10.0]
    monkeypatch.setattr(L.time, "monotonic", lambda: now[0])
    lab = SimpleNamespace(_t0=0.0, _ts_off=None, _live_t=-1.0)
    ts = 5000.0
    out = []
    for i in range(12):
        if i == 6:
            now[0] += 0.100                                # the gap, then a burst
        elif i > 6:
            now[0] += 0.0005
        else:
            now[0] += 1 / 60
        out.append(L.Lab._live_clock(lab, {"timestamp": ts + i / 60}))
    dt = np.diff(out)
    assert np.allclose(dt, 1 / 60, atol=1e-6), dt
    # a reconnect (headset clock jumps) re-latches instead of stalling
    t_jump = L.Lab._live_clock(lab, {"timestamp": 10.0})
    assert t_jump > out[-1]
    # no stamp -> arrival clock, still strictly increasing
    assert L.Lab._live_clock(lab, {}) > t_jump


@needs_clips
def test_scope_and_robot_switch_live():
    """The studio's Whole body / Upper body and MABEL / NOVA toggles reach a running
    retargeter without a rebuild: upper body holds the base, NOVA swaps the slews."""
    from retarget_lab.server import config as CFG
    from retarget_lab.server.worker import KinematicRetargeter, pose_from_frame
    p0 = json.load(gzip.open(CLIPS[0]))["payloads"][0]
    kr = KinematicRetargeter("ours")
    st = kr.apply_settings({"scope": "local", "robot": "nova"})
    assert (st["scope"], st["robot"]) == ("local", "nova")
    assert kr.slews is CFG.SLEWS["nova"]
    bk = _backward(p0)
    for i in range(240):                                    # walk 0.6 m back: upper body must not follow
        back = min(0.6, i / 120 * 0.6)
        rec = kr.step(pose_from_frame({"kind": "avp", "p": _walk_payload(p0, back * bk)}), i / 60.0)
    assert abs(rec["base"]["x"]) < 1e-6 and rec["settings"]["scope"] == "local"
    kr.apply_settings({"scope": "global"})
    for i in range(240, 600):                               # 0.6 m more, ramped (a one-frame jump re-anchors)
        back = 0.6 + min(0.6, (i - 240) / 120 * 0.6)
        rec = kr.step(pose_from_frame({"kind": "avp", "p": _walk_payload(p0, back * bk)}), i / 60.0)
    assert rec["base"]["x"] > 0.2                            # whole body: it walks (from where it was)
    kr.reset()
    assert (kr.settings["scope"], kr.settings["robot"]) == ("global", "nova")   # a reset keeps the switches


def test_operator_figure_walks_in_the_world():
    """The drawn operator: anchored at calibration, so walking moves it (it used to be
    re-centred on the head every frame and stood still at the origin)."""
    from retarget_lab.server import human
    from mabel.types import OperatorPose
    H0 = np.eye(4); H0[:3, 3] = [0.0, 1.6, 0.0]
    H1 = np.eye(4); H1[:3, 3] = [0.0, 1.6, 0.7]               # 0.7 m back (ARKit +z)
    p0, p1 = OperatorPose(), OperatorPose()
    p0.head, p1.head = H0, H1
    a = human.anchor_of(p0)
    s0, s1 = human.skeleton(p0, a), human.skeleton(p1, a)
    assert np.allclose(s0["head"][:2], 0.0, atol=1e-9)
    assert s1["head"][0] == pytest.approx(0.7, abs=1e-3)       # MJ +x = robot back
    assert human.skeleton(p1)["head"][0] == pytest.approx(0.0, abs=1e-9)   # the old pose view stays still


def test_headset_scope_switch_reaches_the_pool():
    """The headset's Whole body / Upper body button (control_mode.scope) switches the lab
    and is passed on to NOVA's gateway when physics is on."""
    import asyncio
    from retarget_lab.server.gateway import Gateway
    got = {}
    lab = SimpleNamespace(pool=SimpleNamespace(configure=lambda **kw: got.update(kw)),
                          physics=SimpleNamespace(control=lambda t, p: got.setdefault("phys", (t, p))))
    gw = Gateway.__new__(Gateway)
    gw.lab = lab
    asyncio.run(gw._dispatch(None, {"device": "Vision Pro"}, "control_mode", {"scope": "local"}))
    assert got.get("scope") == "local" and got["phys"] == ("control_mode", {"scope": "local"})
    got.clear()
    asyncio.run(gw._dispatch(None, {"device": "Vision Pro"}, "control_mode", {"method": "retarget"}))
    assert not got                                          # no scope key: nothing switches


def test_physics_lag_estimator_recovers_a_known_delay():
    """PhysicsLink.lag(): physics = kinematic delayed by 0.30 s -> ~300 ms, tiny residual."""
    from collections import deque
    from retarget_lab.server.physics import PhysicsLink, LAG_JOINTS
    pl = PhysicsLink.__new__(PhysicsLink)
    pl.hist = deque(maxlen=900)
    t = np.arange(0, 12, 0.02)
    sig = lambda tt: np.stack([np.sin(1.3 * tt + k) for k in range(len(LAG_JOINTS))], axis=1)
    K, P = sig(t), sig(t - 0.30)
    for i in range(len(t)):
        pl.hist.append((t[i], K[i], P[i]))
    lg = pl.lag()
    assert lg["time_lag_ms"] == pytest.approx(300, abs=25) and lg["residual_rms"] < 0.05 and not lg["lag_at_cap"]


@needs_take
def test_ablation_channels_fix_torso_neck_lift_and_nova_has_no_channel():
    """Studio Torso / Neck / Lift switches (config.CHANNELS): "fixed" holds each joint at
    home after the solve while the same retargeter runs; there is no "nova" channel any
    more (it falls back to free, like any unknown mode) — robot "nova" puts NOVA's own
    windows on the engine instead, so its free neck can look 53 deg down."""
    from retarget_lab.server import episodes
    from retarget_lab.server.worker import KinematicRetargeter, pose_from_frame
    _, rows = episodes.load(TAKE.name)
    fr = [r for r in rows if r.get("kind") == "wire"][:600]
    kr = KinematicRetargeter("ours")
    assert kr.apply_settings({"torso": "bogus"})["torso"] == "free"
    assert kr.apply_settings({"neck": "nova"})["neck"] == "free"
    kr.apply_settings({"robot": "nova", "torso": "fixed", "neck": "fixed", "lift": "fixed"})
    for f in fr:
        rec = kr.step(pose_from_frame(f), f["t"], collide=False, frame=f)
        if f["t"] - fr[0]["t"] > 3.0:                       # past the 2.5 s park ease
            q = rec["q"]
            assert abs(q["torso"]) < 1e-6
            assert max(abs(q[k]) for k in ("neck_yaw", "neck_pitch", "neck_roll")) < 1e-6
            assert abs(q["lift_lower"] + q["lift_upper"] - kr.lift_home) < 2e-3
    kr.apply_settings({"torso": "free", "neck": "free", "lift": "free"})
    assert tuple(kr.eng.neck_rng[1]) == CFG.NOVA_NECK_RANGE["neck_pitch"]
    assert (kr.eng.wbc.torso.lo, kr.eng.wbc.torso.hi) == CFG.NOVA_TORSO_RANGE
    kr.apply_settings({"robot": "mabel"})
    assert abs(kr.eng.neck_rng[1][1] - 0.785) < 1e-3          # MABEL keeps its MJCF window
