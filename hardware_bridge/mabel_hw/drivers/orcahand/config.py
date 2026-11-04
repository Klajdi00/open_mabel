"""Hand configuration: the joint <-> motor map, angle ranges, and calibration.

A :class:`HandConfig` is the static description of one physical hand. It is
loaded from a YAML file (the ``<side>_hand.yaml`` format shared with the ROS
package) plus its companion calibration JSON, and produces the ordered list of
:class:`~orcahand.joints.Joint` objects the :class:`~orcahand.hand.OrcaHand`
drives.

YAML schema::

    side: right
    calibration_file: right_calibration.json   # relative to the yaml, or absolute
    joints:
      index_mcp: { id: 2, angle_min: -0.3491, angle_max: 1.6581, invert: false }
      ...

Canonical configs for the MABEL left/right hands ship inside the package
(``orcahand/configs/``), so ``HandConfig.load("right")`` works out of the box;
pass an explicit path to use a project-specific config (e.g. the ROS share dir).

Calibration source of truth: the SETUP STUDIO master ``mabel.json``
(``hand/orca-<side>/calib``, read through
:class:`~mabel_hw.calibration.CalibrationStore`). ``HandConfig.load`` maps and
commands each joint with the studio's own unit-gain map
(``angle = model_mid + dir·wrap_π(raw − raw_mid)``) so the driver, every teleop
device, and the studio's 3D viewport render the SAME pose from the SAME numbers —
there is no second file to keep in sync. The legacy tick files
(``<side>_calibration.json``, ROM-normalised map) remain a per-joint FALLBACK for
a stale upstream checkout or a joint the master hasn't calibrated. Precedence:

  explicit ``calibration_path`` (bench/test)  >  studio master mabel.json
  >  tick file  >  full-range identity.
"""

from __future__ import annotations

import os

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import yaml

from .calibration import Calibration
from .constants import TICK_TO_RAD
from .joints import Joint


def _model_joint_kwargs(entry: "Optional[dict]") -> "Optional[dict]":
    """Validate ONE studio ``mabel.json`` calib entry and normalise it to the
    fields :class:`~orcahand.joints.Joint` needs, or None if unusable (so the
    joint falls back to the tick map). Mirrors the studio's own guard
    (``hand_bridge._calib_target_ticks`` refuses an entry with half-span ≤ 0)."""
    if not isinstance(entry, dict):
        return None
    if any(k not in entry for k in ("raw_min", "raw_max", "model_mid")):
        return None
    try:
        rmin = float(entry["raw_min"])
        rmax = float(entry["raw_max"])
        model_mid = float(entry["model_mid"])
        d = int(entry.get("dir", 1))
    except (TypeError, ValueError):
        return None
    if rmax <= rmin:                    # no usable travel — refuse, don't guess
        return None
    return {"raw_min_rad": rmin, "raw_max_rad": rmax,
            "model_mid": model_mid, "invert": d < 0}

_CONFIG_DIR = Path(__file__).resolve().parent / "configs"


def _mabel_root() -> Optional[Path]:
    """The MABEL repo root: MABEL_ROOT env, else walk up from this file looking
    for the repo markers (depth-independent, so vendored moves can't break it)."""
    env = os.environ.get("MABEL_ROOT")
    if env:
        return Path(env)
    for parent in Path(__file__).resolve().parents:
        if (parent / "firmware").is_dir() and (parent / "ros").is_dir():
            return parent
    return None


def _studio_hand_calib(side: str, slice_path: Optional[str] = None) -> "tuple[dict, Optional[Path]]":
    """The STUDIO's master hand calibration for ``side`` from ``mabel.json``.

    Returns ``({joint_name: {raw_min, raw_max (rad), dir, model_mid}}, source_path)``
    read through :class:`~mabel_hw.calibration.CalibrationStore` (the bridge's one
    view of ``hand/orca-<side>/calib``, dual-written by the setup studio on every
    save). This is the SINGLE MASTER: the driver renders and commands from the
    exact numbers the studio's 3D viewport uses, so teleop devices and the studio
    can never disagree. Empty dict (never fatal) when the calibration package or
    the tree is absent — the tick-file fallback then applies."""
    try:
        from ...calibration import CalibrationStore   # lazy: keep orcahand standalone
    except Exception:
        return {}, None
    try:
        store = CalibrationStore().load()
        node = store.slice(f"{slice_path or f'hand/orca-{side}'}/calib")
        src = store.source.get("mabel.json")
    except Exception:
        return {}, None
    if not isinstance(node, dict):
        return {}, None
    calib = {k: v for k, v in node.items()
             if not k.startswith("_") and isinstance(v, dict)}
    return calib, src


def _firmware_calibration(side: str) -> Optional[Path]:
    """The FALLBACK tick calibration file for ``side``, or None if absent.

    Used only when the studio master (mabel.json, see ``_studio_hand_calib``) has
    no entry for a joint. Resolution order (the dual-write topology in
    calibration_data/README.md): the bridge's read-first mirror
    ``hardware_bridge/calibration_data/hands``, then the firmware GUI folder."""
    root = _mabel_root()
    if root is None:
        return None
    mirror = root / "hardware_bridge" / "calibration_data" / "hands"
    fw = root / "firmware" / "orca_hand" / "feetech" / "orcahand" / "configs"
    for cfg_dir, exts in ((mirror, (".json",)), (fw, (".yaml", ".json"))):
        for ext in exts:
            p = cfg_dir / f"{side}_calibration{ext}"
            if p.exists():
                return p
    return None


@dataclass(frozen=True)
class HandConfig:
    side: str                 # "left" | "right"
    joints: List[Joint]       # ordered; the canonical joint order for this hand
    config_path: Optional[Path] = None
    calibration_path: Optional[Path] = None
    generation: str = "v1"    # "v1" = MABEL's own ORCA hand, "v2" = NOVA's (swapped in)

    # ------------------------------------------------------------------ lookups
    @property
    def joint_names(self) -> List[str]:
        return [j.name for j in self.joints]

    @property
    def motor_ids(self) -> List[int]:
        return [j.motor_id for j in self.joints]

    def joint(self, name: str) -> Joint:
        for j in self.joints:
            if j.name == name:
                return j
        raise KeyError(f"unknown joint {name!r} for {self.side} hand")

    def by_name(self) -> Dict[str, Joint]:
        return {j.name: j for j in self.joints}

    def by_id(self) -> Dict[int, Joint]:
        return {j.motor_id: j for j in self.joints}

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(
        cls,
        model: Optional[str] = None,
        config_path: Optional[Union[str, Path]] = None,
        calibration_path: Optional[Union[str, Path]] = None,
        use_model_calib: bool = True,
    ) -> "HandConfig":
        """Load a hand config.

        Parameters
        ----------
        model
            ``"left"`` or ``"right"`` -- loads the packaged canonical config.
        config_path
            explicit path to a ``*_hand.yaml`` (overrides ``model``).
        calibration_path
            explicit calibration JSON (overrides the yaml's ``calibration_file``
            AND the studio master -- use for a bench file or a deterministic test).
        use_model_calib
            when True (default) and no explicit ``calibration_path`` is given, the
            joint mapping comes from the STUDIO MASTER (``mabel.json``
            ``hand/orca-<side>/calib``) via the unit-gain map, so the driver
            renders/commands identically to the setup studio. Set False (the mock
            does) to force the packaged tick-file ROM map for determinism.
        """
        if config_path is None:
            if model is None:
                raise ValueError("pass either model='left'/'right' or config_path=...")
            config_path = _CONFIG_DIR / f"{model}_hand.yaml"
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"hand config not found: {config_path}")

        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        side = cfg.get("side", model or "unknown")
        # SWAPPABLE HANDS: MABEL's own ORCA v1 (left_hand.yaml / right_hand.yaml) or NOVA's ORCA v2
        # (<side>_hand_v2.yaml, written by NOVA's servos/orca_hands.py export-v2). A v2 hand has its own
        # calibration slice and NEVER falls back to the v1 tick files.
        generation = str(cfg.get("generation", "v1"))
        calib_slice = cfg.get("calib_slice") or f"hand/orca-{side}"

        # Resolve the calibration file:
        #   explicit arg > firmware GUI folder (the calibration source of truth,
        #   see module docstring) > yaml's calibration_file > none.
        calib_path = calibration_path
        if calib_path is None and generation == "v1":
            calib_path = _firmware_calibration(side)
        if calib_path is None:
            cf = cfg.get("calibration_file")
            if cf:
                cf = Path(cf)
                calib_path = cf if cf.is_absolute() else config_path.parent / cf
        if calib_path is not None and Path(calib_path).exists():
            calib = Calibration.from_file(calib_path)
        else:
            calib = Calibration.empty()
            calib_path = None

        # Studio master (mabel.json): the single source of truth. Consulted only
        # when no explicit tick file was requested; per joint it overrides the tick
        # ROM map with the studio's unit-gain map. Absent joints fall back to ticks.
        model_calib: dict = {}
        model_src: Optional[Path] = None
        if use_model_calib and calibration_path is None:
            model_calib, model_src = (_studio_hand_calib(side) if calib_slice == f"hand/orca-{side}"
                                      else _studio_hand_calib(side, calib_slice))

        joints: List[Joint] = []
        used_model = False
        for name, spec in cfg["joints"].items():
            mid = int(spec["id"])
            mc = calib.get(mid)
            m = model_calib.get(name)
            model_kw = _model_joint_kwargs(m)
            if model_kw is not None:
                used_model = True
                # Direction rides `dir` from the studio; invert carries it so the
                # velocity/torque sign logic (shared with the tick map) stays right.
                rmin_t = int(round(model_kw["raw_min_rad"] / TICK_TO_RAD)) % 4096
                rmax_t = int(round(model_kw["raw_max_rad"] / TICK_TO_RAD)) % 4096
                joints.append(Joint(
                    name=name,
                    motor_id=mid,
                    angle_min=float(spec.get("angle_min", 0.0)),
                    angle_max=float(spec.get("angle_max", 1.5708)),
                    invert=model_kw["invert"],
                    raw_min=rmin_t,
                    raw_max=rmax_t,
                    center=(rmin_t + rmax_t) // 2,
                    offset=0,
                    model_mid=model_kw["model_mid"],
                    raw_min_rad=model_kw["raw_min_rad"],
                    raw_max_rad=model_kw["raw_max_rad"],
                ))
                continue
            joints.append(Joint(
                name=name,
                motor_id=mid,
                angle_min=float(spec.get("angle_min", 0.0)),
                angle_max=float(spec.get("angle_max", 1.5708)),
                invert=bool(spec.get("invert", False)),
                raw_min=mc.raw_min,
                raw_max=mc.raw_max,
                center=mc.center,
                offset=mc.offset,
            ))

        # calibration_path reports the file that actually drove the mapping (the
        # studio master when it won, else the tick file) — this is what the bridge
        # logs on connect.
        src_path = model_src if used_model else calib_path
        return cls(
            side=side,
            joints=joints,
            config_path=config_path,
            calibration_path=Path(src_path) if src_path else None,
            generation=generation,
        )

    @staticmethod
    def available_models() -> List[str]:
        return sorted(p.stem.replace("_hand", "") for p in _CONFIG_DIR.glob("*_hand.yaml"))
