"""The torso-frame estimator attributes each canonical motion to the right cause.

Written against the synthetic forward operator model in
`controller/experiments/exp_body_frame.py`: given KNOWN torso yaw, neck yaw, root
translation and arm angles, the estimator must recover the torso and leave the
rest as neck / arm. The bounds are loose enough for the low-pass lag and tight
enough that the failures that motivated the rewrite cannot pass:

  * a look-around read as a body turn (the head-only baseline's 42 deg error),
  * a one-arm reach read as a body turn (why the old estimator used elbows).
"""
import os
import sys

import numpy as np
import pytest

# the experiment module carries the forward operator model + scenarios
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "experiments"))
import exp_body_frame as X  # noqa: E402

from mabel.whole_body_retargeting.body_frame_estimator import (  # noqa: E402
    BodyFrameEstimator, head_yaw, _wrap, _bearing, _shared, UP)


def _run(gen):
    frames = list(gen())
    est = BodyFrameEstimator()
    _, h0, wl0, wr0, _ = frames[0]
    est.calibrate_facing(h0, wl0, wr0)
    psi0 = frames[0][4]["psi"]
    torso, neck_est, neck_true, origins = [], [], [], []
    for _, h, wl, wr, gt in frames:
        origin, _ = est.update(h, wl, wr, X.DT)
        torso.append(est.torso_yaw())
        neck_est.append(est.neck_yaw(h[:3, :3]))
        neck_true.append(gt["neck"])
        origins.append(origin)
    truth = np.array([_wrap(f[4]["psi"] - psi0) for f in frames])
    return (np.array(torso), truth, np.array(neck_est), np.array(neck_true),
            np.array(origins), frames)


def _rms(a):
    return float(np.sqrt(np.mean(np.square(a))))


# ── the four canonical motions ────────────────────────────────────────────────
def test_whole_body_turn_is_tracked():
    torso, truth, *_ = _run(X.s1_whole_body_turn)
    err = np.degrees([_wrap(a - b) for a, b in zip(torso, truth)])
    assert _rms(err) < 4.0, f"torso yaw RMS {_rms(err):.2f} deg (should track the turn)"
    assert abs(np.degrees(torso[-1]) - 60.0) < 6.0, (
        f"a 60 deg body turn ended at {np.degrees(torso[-1]):.1f} deg")


def test_look_around_is_NOT_a_body_turn():
    """The head-only baseline's 42 deg failure. Neck-only must give ~0 torso."""
    torso, truth, neck_est, neck_true, *_ = _run(X.s2_look_around)
    assert np.max(np.abs(np.degrees(torso))) < 3.0, (
        f"look-around leaked {np.max(np.abs(np.degrees(torso))):.1f} deg into torso")
    # ...and all of it must show up as NECK
    nerr = np.degrees([_wrap(a - b) for a, b in zip(neck_est, neck_true)])
    assert _rms(nerr) < 3.0, f"neck residual RMS {_rms(nerr):.2f} deg (should equal head)"


def test_one_arm_reach_is_NOT_a_body_turn():
    """A one-arm reach moves one wrist; it must not be mistaken for a turn."""
    torso, *_ = _run(X.s3_one_arm_reach)
    assert np.max(np.abs(np.degrees(torso))) < 3.0, (
        f"one-arm reach leaked {np.max(np.abs(np.degrees(torso))):.1f} deg into torso")


def test_dynamic_walk_turn_reach():
    """The whole point: a simultaneous walk + turn + reach is decomposed."""
    torso, truth, _, _, origins, frames = _run(X.s4_walk_turn_reach)
    err = np.degrees([_wrap(a - b) for a, b in zip(torso, truth)])
    assert _rms(err) < 3.0, f"dynamic torso yaw RMS {_rms(err):.2f} deg"
    # the chest origin must track the operator's walk
    walked = origins - origins[0]
    root = np.array([f[4]["root"] for f in frames])
    hxz = np.linalg.norm((walked - root)[:, [0, 2]], axis=1) * 1000.0
    assert _rms(hxz) < 30.0, f"root (walk) tracking RMS {_rms(hxz):.1f} mm"


# ── the two hard cases the wrist-only estimator could not resolve ─────────────
def test_both_arms_parallel_head_still_is_NOT_a_turn():
    """S3b: BOTH arms sweep the SAME way while the body (and head) are still. The
    wrist bearings rotate together, so a bearing-consensus reads it as a turn --
    but with the head still it is a two-arm reach, and the base must NOT move.
    The anti-correlation of the bearing vs orientation cues + the head gate catch
    it. This is the false-drive case that matters for safety."""
    torso, *_ = _run(X.s3b_both_arms_parallel)
    assert np.max(np.abs(np.degrees(torso))) < 3.0, (
        f"both-arms reach leaked {np.max(np.abs(np.degrees(torso))):.1f} deg "
        "into the base (false drive)")


def test_gaze_fixed_turn_IS_a_body_turn():
    """S5: the torso turns while the neck counter-rotates so the head stays fixed
    on a target (head yaw ~ 0). The head cannot corroborate, yet the arms orbit
    the spine rigidly -- the wrist cues AGREE -- so the base must still turn. The
    head-only baseline loses this entirely (head is stationary)."""
    torso, truth, *_ = _run(X.s5_gaze_fixed_turn)
    err = np.degrees([_wrap(a - b) for a, b in zip(torso, truth)])
    assert _rms(err) < 4.0, f"gaze-fixed turn RMS {_rms(err):.2f} deg (lost the turn)"
    assert abs(np.degrees(torso[-1]) - 50.0) < 8.0, (
        f"a 50 deg gaze-fixed turn ended at {np.degrees(torso[-1]):.1f} deg")


def test_case3_arms_and_head_together_IS_a_body_turn():
    """S6 (the operator's case 3): the arms sweep AND the head follows the SAME
    way, torso otherwise still -> the base should absorb the shared rotation. The
    arm motion is IDENTICAL to S3b; only the head differs, so this pair isolates
    the head-corroboration mechanism -- S3b (head still) stays put, S6 (head
    follows) drives the base."""
    torso, truth, *_ = _run(X.s6_point_and_look)
    err = np.degrees([_wrap(a - b) for a, b in zip(torso, truth)])
    assert _rms(err) < 4.0, f"case-3 arms+head RMS {_rms(err):.2f} deg (base did not follow)"


def test_engine_body_yaw_convention():
    """Lock in the live wiring contract (teleop/engine.py:_headset_body_yaw): the
    base heading is `atan2(d_x, -d_y)` on `AVP_TO_ROBOT @ B[:,0]` (body's right in
    the robot frame) -- the IDENTICAL formula the elbow estimator uses, so
    BASE_YAW_SIGN and the downstream delta stay valid. A body turn must produce a
    same-sign, right-magnitude DELTA; a look-around and a two-arm reach must not."""
    from mabel.whole_body_retargeting.body_frame_estimator import BodyFrameEstimator
    AVP_TO_ROBOT = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])  # matches hyperparams

    def body_yaw_delta(gen):
        fr = list(gen())
        est = BodyFrameEstimator()
        _, h0, wl0, wr0, _ = fr[0]
        est.calibrate_facing(h0, wl0, wr0)

        def yaw(B):
            r = AVP_TO_ROBOT @ B[:, 0]
            return np.arctan2(r[0], -r[1])
        y0 = None
        for _, h, wl, wr, _ in fr:
            _, B = est.update(h, wl, wr, X.DT)
            y = yaw(B)
            if y0 is None:
                y0 = y
        return np.degrees(_wrap(y - y0))

    # a +60 deg body turn -> a positive delta near 60 (same sign as the true turn)
    turn = body_yaw_delta(X.s1_whole_body_turn)
    assert turn > 30.0, f"body_yaw delta {turn:.1f} deg has the wrong sign/magnitude for a +60 turn"
    assert abs(turn - 60.0) < 15.0, f"body_yaw delta {turn:.1f} deg should track the 60 deg turn"
    # a look-around and a two-arm reach (head still) must not move the base heading
    assert abs(body_yaw_delta(X.s2_look_around)) < 3.0, "look-around drove the base heading"
    assert abs(body_yaw_delta(X.s3b_both_arms_parallel)) < 3.0, "two-arm reach drove the base heading"


def test_base_translation_only_when_the_head_moves():
    """The head is the translation ground truth: an arm reach with a still head
    leaves the base velocity ZERO (local manipulation, T1); a walk moves it (T2).
    """
    from mabel.whole_body_retargeting.body_frame_estimator import BodyFrameEstimator
    # T1: one-arm reach, head still -> no base translation
    est = BodyFrameEstimator()
    fr = list(X.s3_one_arm_reach())
    _, h0, wl0, wr0, _ = fr[0]
    est.calibrate_facing(h0, wl0, wr0)
    vmax = 0.0
    for _, h, wl, wr, _ in fr:
        est.update(h, wl, wr, X.DT)
        vmax = max(vmax, float(np.linalg.norm(est.base_velocity())))
    assert vmax < 0.05, f"a still-head reach commanded {vmax*1000:.0f} mm/s of base drive"
    # T2: walk -> base velocity tracks the operator's forward speed (0.6 m / 5 s)
    est = BodyFrameEstimator()
    fr = list(X.s4_walk_turn_reach())
    _, h0, wl0, wr0, _ = fr[0]
    est.calibrate_facing(h0, wl0, wr0)
    speeds = []
    for _, h, wl, wr, _ in fr:
        est.update(h, wl, wr, X.DT)
        speeds.append(float(np.linalg.norm(est.base_velocity())))
    assert max(speeds) > 0.05, "a 0.6 m walk produced no base translation"


def test_beats_the_head_only_baseline_on_look_around():
    """Quantify the improvement that motivated the work."""
    from mabel.whole_body_retargeting.calibration import body_frame
    est_torso, truth, *_ = _run(X.s2_look_around)
    frames = list(X.s2_look_around())
    _, B0 = body_frame(frames[0][1])
    base = []
    for _, h, _, _, _ in frames:
        _, B = body_frame(h)
        base.append(_bearing(B[:, 2]) - _bearing(B0[:, 2]))
    base_err = _rms(np.degrees([_wrap(b - t) for b, t in zip(base, truth)]))
    est_err = _rms(np.degrees([_wrap(a - t) for a, t in zip(est_torso, truth)]))
    assert base_err > 20.0, "the head-only baseline should fail look-around badly"
    assert est_err < base_err / 5.0, (
        f"estimator {est_err:.1f} deg vs head-only {base_err:.1f} deg -- "
        "should be a >5x improvement")


# ── the decomposition identity + robustness ──────────────────────────────────
def test_head_yaw_equals_torso_plus_neck():
    """The contract: head yaw = torso yaw + neck yaw, always."""
    for gen in (X.s1_whole_body_turn, X.s2_look_around, X.s4_walk_turn_reach):
        frames = list(gen())
        est = BodyFrameEstimator()
        _, h0, wl0, wr0, _ = frames[0]
        est.calibrate_facing(h0, wl0, wr0)
        for _, h, wl, wr, _ in frames:
            est.update(h, wl, wr, X.DT)
            hy = head_yaw(h[:3, :3])
            recon = _wrap(est.psi + est.neck_yaw(h[:3, :3]))
            assert abs(_wrap(hy - recon)) < 1e-9, "head != torso + neck"


def test_survives_tracker_noise():
    """5 mm + 1 deg jitter/frame -- heavier than real AVP -- must not break it."""
    rng = np.random.default_rng(0)
    torso, truth = [], []
    frames = list(X.s4_walk_turn_reach())
    est = BodyFrameEstimator()
    _, h0, wl0, wr0, _ = frames[0]
    est.calibrate_facing(h0, wl0, wr0)
    psi0 = frames[0][4]["psi"]
    for _, h, wl, wr, gt in frames:
        hn, wln, wrn = (X._jitter(x, rng) for x in (h, wl, wr))
        est.update(hn, wln, wrn, X.DT)
        torso.append(est.torso_yaw())
        truth.append(_wrap(gt["psi"] - psi0))
    err = np.degrees([_wrap(a - b) for a, b in zip(torso, truth)])
    assert _rms(err) < 6.0, f"noisy torso yaw RMS {_rms(err):.2f} deg"


def test_uses_no_simulator_and_only_numpy():
    """It must run on the real headset stream: numpy + config only."""
    import ast
    import mabel.whole_body_retargeting.body_frame_estimator as m
    tree = ast.parse(__import__("pathlib").Path(m.__file__).read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert "mujoco" not in imported, "estimator imports a simulator"
    assert imported - {"numpy", "__future__", ""} <= {"", }, (
        f"estimator imports more than numpy/config: {imported}")


# ── the fusion primitive, in isolation ───────────────────────────────────────
def test_shared_rejects_differential_motion():
    """`_shared` is the heart of it: agree -> the common value; disagree -> 0."""
    # both rotate the same way -> the shared rotation, high weight
    v, w = _shared(0.20, 0.22)
    assert 0.19 < v < 0.23 and w > 0.8
    # opposite signs (a two-arm reach) -> nothing
    v, w = _shared(0.20, -0.20)
    assert abs(v) < 1e-9 and w == 0.0
    # one still, one moving (a one-arm reach) -> mostly rejected
    v, w = _shared(0.30, 0.0)
    assert w < 0.05


# ── orientation-blind two-arm sweep (2026-07-31 bimanual cross-talk) ──────────
def _sweep_frames(n=400, dt=1.0 / 60.0, rotate_wrists=False, turn_head=False):
    """Both hands sweeping laterally in front of a STATIC head and torso.

    The hands TRANSLATE without rotating unless asked, which is the case every
    other guard in the estimator misses: `anti` is defined from the wrist
    ORIENTATION common mode, so with no wrist rotation it collapses to 0 and the
    bearing cue alone — two wrists swinging the same way about the chest, i.e.
    exactly what a turn looks like — is trusted ungated.
    """
    out = []
    for i in range(n):
        t = i * dt
        head = np.eye(4)
        head[:3, 3] = [0.0, 1.55, 0.0]
        if turn_head:
            th = 0.5 * np.sin(2 * np.pi * 0.25 * t)
            c, s = np.cos(th), np.sin(th)
            head[:3, :3] = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        x = 0.13 * np.sin(2 * np.pi * 0.4 * t)
        wl, wr = np.eye(4), np.eye(4)
        wl[:3, 3] = [-0.26 + x, 1.15, -0.34]
        wr[:3, 3] = [+0.26 + x, 1.15, -0.34]
        if rotate_wrists:
            th = 0.5 * np.sin(2 * np.pi * 0.4 * t)
            c, s = np.cos(th), np.sin(th)
            R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
            wl[:3, :3] = R
            wr[:3, :3] = R
        out.append((head, wl, wr))
    return out


def _torso_swing(frames):
    est = BodyFrameEstimator()
    est.calibrate_facing(*frames[0])
    ys = []
    for h, wl, wr in frames:
        est.update(h, wl, wr, 1.0 / 60.0)
        ys.append(est.torso_yaw())
    ys = np.degrees(np.asarray(ys[60:], float))
    return float(np.max(ys) - np.min(ys))


def test_two_arm_sweep_with_still_head_is_NOT_a_body_turn():
    """THE bimanual cross-talk bug: a two-hand lateral sweep in front of a
    static head invented a body turn, and because the arm targets are
    decomposed in that frame it rotated BOTH arms — each arm then tracked the
    OTHER hand better than its own (r = -0.84 vs +0.36), which is what the
    operator saw as "only one hand works at a time". Measured 20.2 deg before
    the fix."""
    swing = _torso_swing(_sweep_frames())
    assert swing < 2.0, f"orientation-blind two-arm sweep leaked {swing:.1f} deg of torso yaw"


def test_two_arm_sweep_with_TURNING_head_still_turns_the_body():
    """The gate is a CONJUNCTION (no orientation evidence AND a still head), so
    a real turn — where the head goes with the body — must survive it."""
    swing = _torso_swing(_sweep_frames(turn_head=True))
    assert swing > 5.0, f"a corroborating head turn was suppressed ({swing:.1f} deg)"


# ── three-frame forward consensus (2026-07-31) ───────────────────────────────
def _Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _consensus():
    from mabel.whole_body_retargeting.body_frame_estimator import ForwardConsensus
    fc = ForwardConsensus()
    # Learn with the operator squared up, but with the wrists carrying an
    # ARBITRARY convention offset — the point is that nothing assumes ARKit's
    # hand-anchor axis layout, which is the trap this design avoids.
    fc.learn(_Ry(0.0), _Ry(0.9), _Ry(-0.6))
    return fc


def test_squared_up_is_detected_whatever_the_wrist_convention():
    fc = _consensus()
    yaw, conf = fc.update(_Ry(0.0), _Ry(0.9), _Ry(-0.6))
    assert conf > 0.95, f"squared up should be certain, got {conf:.2f}"
    assert abs(yaw) < np.deg2rad(2), f"consensus heading {np.degrees(yaw):.1f} deg"


def test_whole_body_turn_keeps_the_consensus_and_tracks_it():
    """Turning with the hands is still 'facing forward' — relative to the body."""
    fc = _consensus()
    yaw, conf = fc.update(_Ry(0.52), _Ry(0.52 + 0.9), _Ry(0.52 - 0.6))
    assert conf > 0.95
    assert abs(yaw - 0.52) < np.deg2rad(3), f"tracked {np.degrees(yaw):.1f} vs 30 deg"


def test_one_hand_across_the_body_is_NOT_a_facing_cue():
    fc = _consensus()
    _, conf = fc.update(_Ry(0.0), _Ry(0.9 - 0.7), _Ry(-0.6))
    assert conf == 0.0, f"a reach must not set the datum, conf {conf:.2f}"


def test_look_around_is_NOT_a_facing_cue():
    """Head turns, hands stay — the classic false positive for a head-only datum
    (14.5 deg mean error, sigma 18.9, Ponton et al. CGF 2022)."""
    fc = _consensus()
    _, conf = fc.update(_Ry(0.6), _Ry(0.9), _Ry(-0.6))
    assert conf == 0.0, f"a look-around must not set the datum, conf {conf:.2f}"


def test_set_forward_learns_the_offsets():
    """Set Forward is the operator declaring they are squared up, so it is also
    where the wrist offsets are learned."""
    est = BodyFrameEstimator()
    assert not est.facing.off
    h, wl, wr = np.eye(4), np.eye(4), np.eye(4)
    wl[:3, :3] = _Ry(0.9); wr[:3, :3] = _Ry(-0.6)
    est.calibrate_facing(h, wl, wr)
    assert set(est.facing.off) == {"left", "right"}
    _, conf = est.facing.update(np.eye(3), _Ry(0.9), _Ry(-0.6))
    assert conf > 0.95


# --- Set Forward must LAND, and land soon ------------------------------------
#
# The full datum wants a head and BOTH wrists in one frame, because that is what
# lets the consensus learn its wrist offsets. But the operator presses the button
# with their hands wherever they are, and a hand outside the headset's field of
# view is simply not there — so "wait for the good frame" could wait
# indefinitely: "the recenter forward function — it is too slow" (operator,
# 2026-08-01). Past `FORWARD_DEADLINE` the engine zeroes from the head alone.

def test_head_only_zero_gives_the_SAME_datum():
    """Dropping the wrists must not move the datum: the full path's `psi` IS the
    head yaw, so the deadline fallback has to agree with it exactly."""
    h = np.eye(4)
    h[:3, :3] = _Ry(0.7)
    h[:3, 3] = [0.2, 1.5, -0.3]
    wl, wr = np.eye(4), np.eye(4)
    wl[:3, 3] = [-0.25, 1.1, -0.35]
    wr[:3, 3] = [0.25, 1.1, -0.35]

    full = BodyFrameEstimator(); full.calibrate_facing(h, wl, wr)
    head_only = BodyFrameEstimator(); head_only.calibrate_facing_head(h)

    assert head_only.psi == pytest.approx(full.psi, abs=1e-9)
    assert head_only.psi == pytest.approx(head_yaw(h[:3, :3]), abs=1e-9)


def test_head_only_zero_keeps_previously_learned_offsets():
    """What the fallback CANNOT do is learn the wrist offsets — so it must leave
    the ones it has alone rather than clearing them, and the next full
    calibrate_facing relearns them."""
    est = BodyFrameEstimator()
    h, wl, wr = np.eye(4), np.eye(4), np.eye(4)
    wl[:3, :3] = _Ry(0.9); wr[:3, :3] = _Ry(-0.6)
    est.calibrate_facing(h, wl, wr)
    learned = dict(est.facing.off)

    est.calibrate_facing_head(_Ry4(0.3))
    assert est.facing.off == learned, "a head-only zero must not forget the offsets"


def test_head_only_zero_reseeds_cleanly_on_the_next_full_frame():
    """`_prev` is stale after a head-only zero (it belongs to another datum), so
    `ready` is cleared and `update` re-seeds through its own not-ready branch —
    otherwise the next frame integrates the whole gap as one huge increment."""
    est = BodyFrameEstimator()
    h, wl, wr = np.eye(4), np.eye(4), np.eye(4)
    wl[:3, 3] = [-0.25, 1.1, -0.35]
    wr[:3, 3] = [0.25, 1.1, -0.35]
    est.calibrate_facing(h, wl, wr)
    est.calibrate_facing_head(h)
    assert not est.ready and est._prev is None

    before = est.torso_yaw()
    est.update(h, wl, wr, 1.0 / 60.0)      # the resuming frame
    assert est.ready
    assert est.torso_yaw() == pytest.approx(before, abs=1e-9), \
        "re-seeding must not integrate a jump"


def _Ry4(a):
    m = np.eye(4)
    m[:3, :3] = _Ry(a)
    return m
