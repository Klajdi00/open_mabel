"""store.py — one loader for MABEL's as-deployed calibration.

Read policy (calibration_data FIRST, legacy fallback — a stale upstream
checkout or missing mirror must never be fatal):

  mabel.json     <bridge>/calibration_data/mabel.json
                 -> <MABEL>/the upstream arm stack  (read-only)
  hands/<side>   <bridge>/calibration_data/hands/<side>_calibration.json
                 -> <MABEL>/firmware/orca_hand/feetech/orcahand/configs/
                 -> <MABEL>/ros/src/mabel_hand_hardware/config/
                 -> <bridge>/mabel_hw/drivers/orcahand/configs/             (vendored)

Slices are addressed exactly like the upstream calib service ("arm/openarm-dual",
"neck/rssr", "hand/orca-left"), so studio-written data needs no translation.
Writers (the studios) dual-write into calibration_data; ``save_slice`` is the
bridge-local writer with the same read-modify-write + atomic-replace semantics
as the upstream server. ``watch()`` mtime-polls the RESOLVED sources so a live
save reaches running drivers without a restart.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence

from .. import paths
from ..errors import CalibrationError
from .schema import JointCalib

__all__ = ["CalibrationStore"]


def _default_root() -> Path:
    return Path(paths.bridge_dir()) / "calibration_data"


class CalibrationStore:
    """Loads once, answers many; hot-reloadable via watch(). Thread-safe."""

    def __init__(self, root: Optional[os.PathLike] = None, *,
                 mabel_root: Optional[os.PathLike] = None, logger=None):
        self._root = Path(root) if root else _default_root()
        self._mabel = Path(mabel_root) if mabel_root else Path(paths.mabel_root())
        self._log = logger or (lambda *a, **k: None)
        self._lock = threading.Lock()
        self._tree: dict = {}
        self._source: Dict[str, Path] = {}      # logical name -> file that won

    # ── candidate chains ──────────────────────────────────────────────────────
    def _tree_candidates(self) -> Sequence[Path]:
        return (self._root / "mabel.json",
                self._mabel / "upstream" / "MABEL" / "modules" / "calib"
                / "config" / "mabel.json")

    def _hand_candidates(self, side: str) -> Sequence[Path]:
        fname = f"{side}_calibration.json"
        return (self._root / "hands" / fname,
                self._mabel / "firmware" / "orca_hand" / "feetech" / "orcahand"
                / "configs" / fname,
                self._mabel / "ros" / "src" / "mabel_hand_hardware"
                / "config" / fname,
                Path(__file__).resolve().parents[1] / "drivers" / "orcahand"
                / "configs" / fname)

    @staticmethod
    def _first_existing(cands: Sequence[Path]) -> Optional[Path]:
        for p in cands:
            if p.is_file():
                return p
        return None

    # ── loading ───────────────────────────────────────────────────────────────
    def load(self) -> "CalibrationStore":
        src = self._first_existing(self._tree_candidates())
        with self._lock:
            self._source.clear()
            if src is None:
                self._tree = {}
                self._log("[calib] NO calibration tree found "
                          f"(looked at {[str(p) for p in self._tree_candidates()]}) "
                          "— drivers run uncalibrated raw")
            else:
                try:
                    self._tree = json.loads(src.read_text())
                except Exception as e:
                    raise CalibrationError(f"unparsable calibration tree {src}: {e}") from e
                self._source["mabel.json"] = src
                self._log(f"[calib] tree <- {src}")
            for side in ("left", "right"):
                hp = self._first_existing(self._hand_candidates(side))
                if hp is not None:
                    self._source[f"hands/{side}"] = hp
        return self

    @property
    def source(self) -> Dict[str, Path]:
        """{logical name: the file that won the fallback race} — for logs/tests."""
        with self._lock:
            return dict(self._source)

    # ── reads (upstream-style slash addressing) ───────────────────────────────
    def slice(self, path: str, default=None):
        """Walk the tree by 'family/id[/...]' — e.g. slice('arm/openarm-dual')."""
        node = self._tree_snapshot()
        for part in filter(None, path.strip("/").split("/")):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def _tree_snapshot(self) -> dict:
        with self._lock:
            return self._tree

    def joint(self, slice_path: str, joint: str) -> Optional[JointCalib]:
        if joint.startswith("_"):
            return None                       # _task/_doc blocks are metadata
        entry = self.slice(f"{slice_path}/calib/{joint}")
        if not isinstance(entry, dict):
            return None
        return JointCalib.from_dict(entry, joint=joint)

    def group(self, slice_path: str, joints: Sequence[str],
              *, strict: bool = False) -> Dict[str, JointCalib]:
        """{joint: JointCalib} for every calibrated joint in ``joints``.
        strict=True raises if any joint is missing (bring-up checks)."""
        out: Dict[str, JointCalib] = {}
        missing = []
        for name in joints:
            jc = self.joint(slice_path, name)
            if jc is None:
                missing.append(name)
            else:
                out[name] = jc
        if missing and strict:
            raise CalibrationError(
                f"{slice_path}: no calibration for {missing} "
                f"(have: {sorted(out)})")
        return out

    def task_limits(self, slice_path: str) -> Optional[dict]:
        """The '_task' block (e.g. the neck's pitch/roll/yaw min/max clamps)."""
        block = self.slice(f"{slice_path}/calib/_task")
        return dict(block) if isinstance(block, dict) else None

    def connection(self, slice_path: str) -> dict:
        """Non-calib identity keys of a slice (teensy_ip/port/stream_hz/device/
        baud...) — how upstream nodes resolve their transport."""
        node = self.slice(slice_path)
        if not isinstance(node, dict):
            return {}
        return {k: v for k, v in node.items()
                if k != "calib" and not k.startswith("_")}

    def hand_calibration_path(self, side: str) -> Optional[Path]:
        """The WINNING tick-clamp file for a hand — passed into OrcaHand so the
        driver keeps its own proven loader."""
        return self._first_existing(self._hand_candidates(side))

    # ── write (bridge-local mirror writer) ────────────────────────────────────
    def save_slice(self, path: str, value) -> Path:
        """Read-modify-write calibration_data/mabel.json (creates it from the
        current fallback tree when absent). Atomic tmp+replace, matching the
        upstream server's persist. Returns the written path."""
        target = self._root / "mabel.json"
        with self._lock:
            tree = json.loads(target.read_text()) if target.is_file() else \
                dict(self._tree)
            node = tree
            parts = [p for p in path.strip("/").split("/") if p]
            if not parts:
                raise CalibrationError("save_slice needs a non-empty path")
            for part in parts[:-1]:
                node = node.setdefault(part, {})
                if not isinstance(node, dict):
                    raise CalibrationError(f"{path}: {part} is not a subtree")
            node[parts[-1]] = value
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(tree, indent=1))
            os.replace(tmp, target)
            self._tree = tree
            self._source["mabel.json"] = target
        self._log(f"[calib] saved {path} -> {target}")
        return target

    # ── hot reload ────────────────────────────────────────────────────────────
    def watch(self, callback: Callable[[str], None],
              *, period_s: float = 2.0) -> Callable[[], None]:
        """mtime-poll the resolved sources on a daemon thread; on change,
        reload and call ``callback(logical_name)``. Returns a stop function."""
        stop = threading.Event()

        def _mtimes() -> Dict[str, float]:
            out = {}
            for name, p in self.source.items():
                try:
                    out[name] = p.stat().st_mtime
                except OSError:
                    pass
            return out

        def _loop():
            seen = _mtimes()
            while not stop.wait(period_s):
                now = _mtimes()
                changed = [n for n, t in now.items() if seen.get(n) != t]
                # a mirror appearing where a fallback used to win is also a change
                if self._first_existing(self._tree_candidates()) != \
                        self.source.get("mabel.json"):
                    changed.append("mabel.json")
                if changed:
                    self.load()
                    seen = _mtimes()
                    for name in dict.fromkeys(changed):
                        try:
                            callback(name)
                        except Exception as e:      # noqa: BLE001
                            self._log(f"[calib] watch callback failed: {e}")

        threading.Thread(target=_loop, daemon=True,
                         name="mabel-hw-calib-watch").start()
        return stop.set
