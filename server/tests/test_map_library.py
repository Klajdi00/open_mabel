"""map_library: the saved-room store behind LIST_MAPS / LOAD_MAP / SAVE_MAP /
DELETE_MAP. Pure file IO + numpy — no ROS, so these run anywhere.

server/ modules import each other flat (`from map_stream import ...`), so the
server dir goes on sys.path the same way the studios' contract tests locate it:
walk up until wire_protocol.py is in sight rather than hard-coding a depth.
"""
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))


def _find_server():
    d = _HERE
    for _ in range(6):
        if os.path.isfile(os.path.join(d, "wire_protocol.py")):
            return d
        d = os.path.dirname(d)
    raise RuntimeError("server/ (wire_protocol.py) not found above " + _HERE)


sys.path.insert(0, _find_server())

import map_library                                    # noqa: E402


@pytest.fixture()
def store(tmp_path, monkeypatch):
    """An isolated map dir. Set before each call so map_dirs()/save_dir() — which
    read the env every time — never touch the real ~/mabel_maps."""
    monkeypatch.setenv("MABEL_MAP_DIR", str(tmp_path))
    # never shell out to nvblox from a test: point the env script at nothing
    monkeypatch.setenv("MABEL_NVBLOX_SAVE_ENV", str(tmp_path / "no_such_env.sh"))
    return tmp_path


def _grid(w=4, h=3):
    """A MapStream._grid_cache-shaped live map: (w,h,res,ox,oy,oyaw,cells)."""
    cells = np.full((h, w), 255, np.uint8)      # unknown
    cells[0, :] = 0                             # free row (bottom, y-up)
    cells[h - 1, :] = 100                       # occupied row (top)
    return (w, h, 0.05, -1.0, -2.0, 0.0, cells)


# ── name validation: the only thing between a client string and open() ─────────
@pytest.mark.parametrize("bad", [
    "../etc/passwd", "..", "a/b", "/abs", "n\0ul", "", "  ", "dot.ted",
    "sp ace", "semi;colon", "x" * 65,
])
def test_bad_names_rejected(store, bad):
    with pytest.raises(ValueError):
        map_library.save_map(bad, _grid())
    with pytest.raises(ValueError):
        map_library.delete_map(bad)


@pytest.mark.parametrize("ok", ["room", "Room_1", "living-room", "a", "x" * 64])
def test_good_names_accepted(store, ok):
    map_library.save_map(ok, _grid())
    assert ok in [m["name"] for m in map_library.list_maps()]


def test_traversal_cannot_escape_the_map_dir(store, tmp_path):
    """The allowlist must stop a name from writing outside the store, not merely
    sanitize it into a different file inside."""
    victim = tmp_path.parent / "escaped.yaml"
    with pytest.raises(ValueError):
        map_library.save_map("../escaped", _grid())
    assert not victim.exists()


# ── roundtrip ─────────────────────────────────────────────────────────────────
def test_save_list_delete_roundtrip(store):
    assert map_library.list_maps() == []            # empty store

    info = map_library.save_map("kitchen", _grid())
    assert info["name"] == "kitchen"
    assert (store / "kitchen.pgm").exists() and (store / "kitchen.yaml").exists()
    assert info["has_mesh"] is False                 # no nvblox in a test env
    assert info["size_bytes"] > 0 and info["saved_at"] > 0

    maps = map_library.list_maps()
    assert [m["name"] for m in maps] == ["kitchen"]
    assert maps[0]["size_bytes"] == info["size_bytes"]

    out = map_library.delete_map("kitchen")
    assert sorted(os.path.basename(p) for p in out["removed"]) == \
        ["kitchen.pgm", "kitchen.yaml"]
    assert map_library.list_maps() == []


def test_saved_map_loads_back_through_the_live_wire_encoder(store):
    """save_map → load_grid_frame must round-trip: what we write is what the /map
    clients get back, in the SAME GRID 0x04 frame the live stream sends."""
    map_library.save_map("den", _grid())
    frame = map_library.load_grid_frame("den")
    assert frame is not None and frame[0] == 0x04

    import struct
    w, h = struct.unpack_from("<II", frame, 4)
    res, ox, oy, _oz, _yaw = struct.unpack_from("<fffff", frame, 12)
    assert (w, h) == (4, 3)
    assert abs(res - 0.05) < 1e-6
    assert abs(ox + 1.0) < 1e-5 and abs(oy + 2.0) < 1e-5
    cells = np.frombuffer(frame[32:], np.uint8).reshape(h, w)
    # trinary + the PGM row flip must survive: free row stays at the BOTTOM
    assert list(cells[0]) == [0] * w
    assert list(cells[h - 1]) == [100] * w
    assert list(cells[1]) == [255] * w              # unknown stays unknown


def test_list_maps_newest_first_and_ignores_yaml_without_pgm(store):
    map_library.save_map("older", _grid())
    map_library.save_map("newer", _grid())
    os.utime(store / "older.yaml", (1000, 1000))
    os.utime(store / "older.pgm", (1000, 1000))
    (store / "notamap.yaml").write_text("image: nope.pgm\n")   # no .pgm beside it

    names = [m["name"] for m in map_library.list_maps()]
    assert names == ["newer", "older"]              # newest first
    assert "notamap" not in names


# ── graceful failure ──────────────────────────────────────────────────────────
def test_delete_missing_map_raises(store):
    with pytest.raises(ValueError, match="no such map"):
        map_library.delete_map("ghost")


def test_load_missing_map_is_none_not_an_error(store):
    assert map_library.load_grid_frame("ghost") is None
    assert map_library.load_mesh_chunks("ghost") == []


def test_save_without_a_live_map_raises(store):
    with pytest.raises(ValueError, match="no live map"):
        map_library.save_map("empty", None)


def test_save_empty_grid_raises(store):
    with pytest.raises(ValueError, match="empty"):
        map_library.save_map("empty", (0, 0, 0.05, 0.0, 0.0, 0.0,
                                       np.zeros((0, 0), np.uint8)))
