"""Contracts for the EgoDex -> MABEL operator-pose converter.

The conversion depends on four conventions that were measured against the data
rather than read off the column names (see `egodex_source` module docstring).
Three of them are silent when wrong -- they produce a run that completes and
looks plausible -- so they are pinned here.

The tests that need the corpus SKIP when it is absent: the cache is gitignored
(EgoDex is CC-by-NC-ND) and a fresh checkout will not have it.
"""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(os.path.dirname(HERE), "experiments")
sys.path.insert(0, EXP)

import egodex_source as ES                                      # noqa: E402

CACHE = os.path.join(EXP, "data", "egodex")
needs_data = pytest.mark.skipif(
    not os.path.isdir(os.path.join(CACHE, "pour")),
    reason="EgoDex cache absent (gitignored); run exp_egodex_retarget.py --fetch")


# ── the wire contract ────────────────────────────────────────────────────────
def test_names_21_matches_the_engine():
    """The converter mirrors _NAMES_21 so it can stay MuJoCo-free. If the engine
    renames a landmark, this must fail rather than silently mis-order joints."""
    from mabel.teleop_engine.engine import _NAMES_21
    assert ES.NAMES_21 == list(_NAMES_21)


def test_finger_index_covers_every_landmark_but_the_wrist():
    assert set(ES.FINGER_INDEX) == set(ES.NAMES_21) - {"wrist"}
    assert len(ES.FINGER_INDEX) == 20
    # indices are distinct and inside the 25-joint block
    idx = sorted(ES.FINGER_INDEX.values())
    assert len(set(idx)) == len(idx)
    assert 0 <= idx[0] and idx[-1] < ES.FOREARM_INDEX


def test_metacarpals_are_the_dropped_indices():
    """MABEL has 21 landmarks, ARKit 26. The five dropped ones must be exactly
    the metacarpals -- 4, 9, 14, 19 (the thumb has none) -- not an off-by-one."""
    used = set(ES.FINGER_INDEX.values())
    assert set(range(ES.FOREARM_INDEX)) - used == {4, 9, 14, 19}


def test_forearm_name_is_what_the_wire_looks_for():
    from mabel.sources.wire import _FOREARM_ARM
    assert ES.FOREARM_NAME == _FOREARM_ARM


def test_opencv_to_arkit_is_an_involution_and_preserves_position():
    """It relabels the camera's own axes: flipping twice is identity, the
    translation column is untouched, and it must not mirror handedness."""
    M = ES.OPENCV_TO_ARKIT
    assert np.allclose(M @ M, np.eye(4))
    assert np.isclose(np.linalg.det(M[:3, :3]), 1.0)     # rotation, not a mirror
    T = np.eye(4)
    T[:3, 3] = [1.0, 2.0, 3.0]
    assert np.allclose((T @ M)[:3, 3], T[:3, 3])


# ── the conversion, on synthetic data ────────────────────────────────────────
def _fake_episode(n=4):
    rng = np.random.default_rng(0)
    head = np.tile(np.eye(4), (n, 1, 1))
    head[:, :3, 3] = [0.0, 1.2, 0.0]
    wrist, fingers = {}, {}
    for side in ("left", "right"):
        W = np.tile(np.eye(4), (n, 1, 1))
        W[:, :3, 3] = rng.normal(0, 0.1, (n, 3))
        wrist[side] = W
        F = np.tile(np.eye(4), (n, 25, 1, 1))
        F[:, :, :3, 3] = rng.normal(0, 0.05, (n, 25, 3))
        fingers[side] = F
    return ES.Episode(task="t", index=0, instruction="i",
                      head=head, wrist=wrist, fingers=fingers)


def test_payload_has_the_shape_the_avp_receiver_parses():
    p = _fake_episode().payload(1)
    assert len(p["head_mat"]) == 16
    for side in ("left", "right"):
        assert len(p[f"{side}_hand_mats"]) == 21
        assert all(len(m) == 16 for m in p[f"{side}_hand_mats"])
        # 21 landmarks + the forearm the elbow objective needs
        assert len(p[f"{side}_joints"]) == 22
        assert ES.FOREARM_NAME in p[f"{side}_joints"]
        assert len(p[f"{side}_elbow"]) == 16
        assert p[f"{side}_pinch"] >= 0.0


def test_hand_mats_index_zero_is_the_wrist():
    """_NAMES_21[0] is 'wrist' and the arm IK reads h_mats[0] directly
    (engine.py: `WR = rh[0]  # [0] = wrist`). An ordering slip here silently
    feeds a fingertip to the body-frame estimator."""
    ep = _fake_episode()
    p = ep.payload(2)
    for side, s in (("left", "l"), ("right", "r")):
        got = np.asarray(p[f"{side}_hand_mats"][0], float).reshape(4, 4)
        assert np.allclose(got, ep.wrist[side][2])


def test_payload_feeds_operator_pose():
    from mabel.types import OperatorPose
    op = OperatorPose.from_payload(_fake_episode().payload(0))
    assert op.head.shape == (4, 4)
    assert len(op.right) == 21 and len(op.left) == 21
    assert op.right_elbow.shape == (4, 4)
    assert op.right_pinch is not None


def test_timestamps_advance_at_the_frame_rate():
    ep = _fake_episode(n=4)
    ts = [p["timestamp"] for p in ep.payloads()]
    assert np.allclose(np.diff(ts), 1.0 / ES.FPS)


# ── the measured conventions, against the real corpus ────────────────────────
@needs_data
def test_rotations_are_row_major():
    """The middle metacarpal is nearly rigid to the wrist, so read correctly it
    sits at a fixed point in the wrist frame. Row-major: local +x, ~86 mm,
    a few mm of scatter. Column-major scatters it over ~25 mm."""
    ep = ES.load_task(os.path.join(CACHE, "pour"), max_episodes=3)[0]
    W = ep.wrist["left"]
    P = ep.fingers["left"][:, 10, :3, 3]                 # middle metacarpal
    loc = np.einsum('nji,nj->ni', W[:, :3, :3], P - W[:, :3, 3])
    assert loc[:, 0].mean() > 0.05                       # along the fingers
    assert np.linalg.norm(loc.std(0)) < 0.012            # rigid to the wrist


@needs_data
def test_head_gaze_points_at_the_hands_and_below_the_horizon():
    """Convention (2). These are seated tabletop tasks: the operator looks down,
    and roughly toward their own hands. The pre-fix reading had the gaze 142 deg
    away -- behind the operator -- which is the failure this pins."""
    ep = ES.load_task(os.path.join(CACHE, "pour"), max_episodes=3)[0]
    g = ep.gaze()
    mid = 0.5 * (ep.wrist["left"][:, :3, 3] + ep.wrist["right"][:, :3, 3])
    v = mid - ep.head[:, :3, 3]
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    ang = np.degrees(np.arccos(np.clip((g * v).sum(1), -1, 1)))
    assert ang.mean() < 75.0                             # measured ~38 deg
    assert np.degrees(np.arcsin(-g[:, 1])).mean() > 5.0  # pitched DOWN


@needs_data
def test_head_is_at_head_height():
    ep = ES.load_task(os.path.join(CACHE, "pour"), max_episodes=3)[0]
    y = ep.head[:, 1, 3]
    assert 0.8 < y.min() and y.max() < 1.9               # a seated/standing person


@needs_data
def test_forearm_joint_is_a_forearm_away_from_the_wrist():
    """Index 24 is the forearm's elbow end -- the joint that feeds the arm IK's
    elbow objective. If the layout shifted, this lands on a fingertip."""
    ep = ES.load_task(os.path.join(CACHE, "pour"), max_episodes=3)[0]
    d = np.linalg.norm(ep.fingers["left"][:, ES.FOREARM_INDEX, :3, 3]
                       - ep.wrist["left"][:, :3, 3], axis=1)
    assert 0.15 < np.median(d) < 0.40                    # measured ~0.265 m


@needs_data
def test_palm_normal_points_mostly_down_on_a_tabletop_task():
    """Convention: outward palm normal is +y local on the left, -y on the right.
    Over a table it must point mostly DOWN; the opposite sign would have both
    palms turned to the ceiling."""
    ep = ES.load_task(os.path.join(CACHE, "pour"), max_episodes=6)[0]
    for side in ("left", "right"):
        assert (ep.palm_normal(side)[:, 1] < 0).mean() > 0.6


@needs_data
def test_finger_chain_lengths_follow_human_anatomy():
    """Pins the ARKit finger ORDER: middle is the longest chain, little the
    shortest. A permuted order would keep every shape test passing."""
    ep = ES.load_task(os.path.join(CACHE, "pour"), max_episodes=3)[0]
    W = ep.wrist["left"][0, :3, 3]
    tip = {n: np.linalg.norm(ep.fingers["left"][0, ES.FINGER_INDEX[f"{n}_tip"], :3, 3] - W)
           for n in ("index", "middle", "ring", "little")}
    assert tip["middle"] == max(tip.values())
    assert tip["little"] == min(tip.values())
