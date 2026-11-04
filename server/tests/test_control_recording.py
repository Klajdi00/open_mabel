"""Every recorded frame carries the operator's control state (2026-10-16) —
which body parts drove the robot, scope, method, flags, engaged — so policy
training and ablation analysis can condition on / filter by it.

Pins: the encoding (codes + legend), and that an H5 written through the real
CurationRecorder → IL EpisodeRecorder path actually contains obs/control/*
per frame and decodes back to what was set — including a MID-EPISODE toggle."""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from curation_recorder import (CONTROL_LEGEND, CONTROL_METHODS, CONTROL_TRACKING,  # noqa: E402
                               CurationRecorder, encode_control)

SPEC = {"method": "retarget", "scope": "global", "gravity_comp": True,
        "whole_body_retarget": False, "estop": False, "engaged": True,
        "tracking": {"neck": True, "arms": True, "hands": True, "torso": True, "lift": True}}


def test_encode_whole_body_all_live():
    o = encode_control(SPEC)
    assert o["obs/control/tracking"].tolist() == [1, 1, 1, 1, 1]
    assert int(o["obs/control/scope"]) == 1
    assert int(o["obs/control/method"]) == CONTROL_METHODS.index("retarget")
    assert o["obs/control/flags"].tolist() == [1, 0, 0]
    assert int(o["obs/control/engaged"]) == 1


def test_encode_frozen_parts_and_upper_body():
    s = json.loads(json.dumps(SPEC))
    s["scope"] = "local"
    s["tracking"].update(neck=False, torso=False, lift=False)
    o = encode_control(s)
    order = list(CONTROL_TRACKING)
    tr = o["obs/control/tracking"].tolist()
    assert tr[order.index("neck")] == 0 and tr[order.index("torso")] == 0
    assert tr[order.index("lift")] == 0 and tr[order.index("arms")] == 1
    assert int(o["obs/control/scope"]) == 0


def test_unknowns_are_explicit_not_guessed():
    o = encode_control({"method": "something_new", "tracking": {"neck": True}})
    assert int(o["obs/control/method"]) == -1
    assert int(o["obs/control/scope"]) == -1
    assert o["obs/control/tracking"].tolist()[1:] == [-1, -1, -1, -1]
    assert encode_control(None) == {}                 # old bridges: no keys at all


def test_legend_covers_every_key():
    keys = set(encode_control(SPEC))
    assert keys == set(CONTROL_LEGEND)
    assert CONTROL_LEGEND["obs/control/tracking"]["order"] == list(CONTROL_TRACKING)


def test_h5_round_trip_with_mid_episode_toggle(tmp_path):
    import h5py
    rec = CurationRecorder(str(tmp_path), fps=1000.0)
    rec.start("ablation_check", {"nq": 3, "nv": 3, "nu": 2})
    for i in range(6):
        s = json.loads(json.dumps(SPEC))
        if i >= 3:                                    # the operator freezes the neck mid-take
            s["tracking"]["neck"] = False
        rec._write_frame(0.1 * i, {"qpos": np.zeros(3), "qvel": np.zeros(3),
                                   "ctrl": np.zeros(2), "control": s})
    path = rec.stop()
    assert path
    deadline = time.time() + 10
    while time.time() < deadline and not os.path.exists(path):
        time.sleep(0.05)                              # the save runs on a worker thread
    time.sleep(0.3)
    with h5py.File(path, "r") as f:
        tr = f["obs/control/tracking"][()]
        assert tr.shape == (6, len(CONTROL_TRACKING))
        neck = tr[:, list(CONTROL_TRACKING).index("neck")].tolist()
        assert neck == [1, 1, 1, 0, 0, 0]
        schema = json.loads(f.attrs["schema_json"])
        assert schema["control_encoding"]["obs/control/tracking"]["order"] == list(CONTROL_TRACKING)
