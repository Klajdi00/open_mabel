"""Recordings scan: episode task folders + rosbag sessions on disk."""
import json
import os

from control_center.server import recordings


def _mk(p, content=b"x" * 100):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "wb") as f:
        f.write(content)


def test_episodes_by_task(tmp_path):
    root = tmp_path / "datasets"
    _mk(str(root / "pick_cup" / "episode_0001_20260101" / "meta.json"))
    _mk(str(root / "pick_cup" / "episode_0002_20260102" / "meta.json"))
    _mk(str(root / "pour_water" / "episode_0001_20260103" / "qpos.npy"))
    _mk(str(root / "h5_style" / "episode_0001.h5"))
    (root / "empty_task").mkdir()
    _mk(str(root / "stray_file.txt"))            # non-dir at root: ignored

    out = recordings.episodes(str(root))
    by = {t["task"]: t for t in out["tasks"]}
    assert by["pick_cup"]["episodes"] == 2
    assert by["pour_water"]["episodes"] == 1
    assert by["h5_style"]["episodes"] == 1
    assert "empty_task" not in by                # no episodes -> hidden
    assert out["total"] == 4
    assert by["pick_cup"]["bytes"] >= 200
    assert by["pick_cup"]["latest"] > 0


def test_bags_sessions(tmp_path):
    root = tmp_path / "recordings"
    s = root / "20260629_070111_default"
    _mk(str(s / "default_0.mcap"), b"m" * 500)
    _mk(str(s / "metadata.yaml"))
    with open(s / "session.json", "w") as f:
        json.dump({"label": "default", "topics": 42}, f)
    _mk(str(root / "20260630_120000_grasp_test" / "grasp_0.mcap"))

    out = recordings.bags(str(root))
    assert out["total"] == 2
    names = [x["name"] for x in out["sessions"]]
    assert names == sorted(names, reverse=True)  # newest first
    first = out["sessions"][1]
    assert first["label"] == "default"
    assert first["mcaps"] == 1
    assert first["meta"]["topics"] == 42


def test_missing_roots_are_empty():
    assert recordings.episodes("/nonexistent/x")["total"] == 0
    assert recordings.bags("/nonexistent/x")["total"] == 0


def test_summary_shape(tmp_path, monkeypatch):
    from control_center.server import config as CFG
    monkeypatch.setattr(CFG, "EPISODES_DIR", str(tmp_path / "e"))
    monkeypatch.setattr(CFG, "RECORDINGS_DIR", str(tmp_path / "r"))
    out = recordings.summary()
    assert "episodes" in out and "bags" in out
