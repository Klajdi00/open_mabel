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
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import yaml

from .calibration import Calibration
from .joints import Joint

_CONFIG_DIR = Path(__file__).resolve().parent / "configs"


@dataclass(frozen=True)
class HandConfig:
    side: str                 # "left" | "right"
    joints: List[Joint]       # ordered; the canonical joint order for this hand
    config_path: Optional[Path] = None
    calibration_path: Optional[Path] = None

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
    ) -> "HandConfig":
        """Load a hand config.

        Parameters
        ----------
        model
            ``"left"`` or ``"right"`` -- loads the packaged canonical config.
        config_path
            explicit path to a ``*_hand.yaml`` (overrides ``model``).
        calibration_path
            explicit calibration JSON (overrides the yaml's ``calibration_file``).
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

        # Resolve the calibration file (explicit arg > yaml field > none).
        calib_path = calibration_path
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

        joints: List[Joint] = []
        for name, spec in cfg["joints"].items():
            mid = int(spec["id"])
            mc = calib.get(mid)
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

        return cls(
            side=side,
            joints=joints,
            config_path=config_path,
            calibration_path=Path(calib_path) if calib_path else None,
        )

    @staticmethod
    def available_models() -> List[str]:
        return sorted(p.stem.replace("_hand", "") for p in _CONFIG_DIR.glob("*_hand.yaml"))
