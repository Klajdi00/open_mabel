"""control_params — the canonical reader for controller/control_param.json.

`controller/control_param.json` is the single editable source of the arm control-law
parameters (per-joint PD gains + gravity-feedforward config). It lives with the control code
because gains and gravity compensation ARE the control law — the hardware bridge only executes
the motor commands that fall out of them. Every consumer reads the SAME file through a loader
like this one so the setup studio, the real-robot gravity FF, and mabel_hw's command-default
gains can never silently disagree (they did: mabel_hw defaulted the shoulders to kp=30, above
the damping ceiling, while the derived value is 14.9).

Resolution order:
  1. $MABEL_CONTROL_PARAM (explicit path override)
  2. controller/control_param.json  (this file is controller/mabel/control_params.py, so the
     controller dir is parents[1])

Returns {} when nothing is found, so every caller degrades to its own built-in fallback rather
than crash — a missing control-param file must never take the arm command path down.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

__all__ = ["control_param_path", "load_control_params", "arm_gains", "arm_gravity_ff",
           "arm_soft_float", "arm_soft_drag", "arm_soft_gains", "torso_gravity_ff"]


def control_param_path() -> "Path | None":
    env = os.environ.get("MABEL_CONTROL_PARAM")
    cands = [Path(env)] if env else []
    cands.append(Path(__file__).resolve().parents[1] / "control_param.json")
    for c in cands:
        if c.is_file():
            return c
    return None


def load_control_params() -> dict:
    p = control_param_path()
    if p is None:
        return {}
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return {}


def arm_gains(params: "dict | None" = None) -> dict:
    """{joint: {kp, kd, vmax}} from control_param.json (arm.gains). Empty -> caller's fallback."""
    p = params if params is not None else load_control_params()
    g = (p.get("arm") or {}).get("gains")
    return g if isinstance(g, dict) else {}


def arm_gravity_ff(params: "dict | None" = None) -> dict:
    """{enabled, clamp, scale, ramp_s, pose_pad} from control_param.json (arm.gravity_feedforward)."""
    p = params if params is not None else load_control_params()
    ff = (p.get("arm") or {}).get("gravity_feedforward")
    return ff if isinstance(ff, dict) else {}


def arm_soft_drag(params: "dict | None" = None) -> dict:
    """control_param.json arm.soft_drag — the position-based SOFT drag-and-hold
    tunables (band, qd gate, follow rate). Empty -> drag_follow.DRAG_DEFAULTS."""
    p = params if params is not None else load_control_params()
    d = (p.get("arm") or {}).get("soft_drag")
    return d if isinstance(d, dict) else {}


def arm_soft_gains(params: "dict | None" = None) -> dict:
    """control_param.json arm.soft_gains — how the app stiffness slider maps to
    the SOFT-mode gain scales: {kp_scale_min, kp_scale_max, kd_scale}."""
    p = params if params is not None else load_control_params()
    d = (p.get("arm") or {}).get("soft_gains")
    return d if isinstance(d, dict) else {}


def arm_soft_float(params: "dict | None" = None) -> dict:
    """control_param.json arm.soft_float — the FLOATING soft-mode admittance
    tunables (per-class deadbands, virtual mass/damping, auto-tare rates).
    Empty -> arm_control.admittance.FLOAT_DEFAULTS."""
    p = params if params is not None else load_control_params()
    d = (p.get("arm") or {}).get("soft_float")
    return d if isinstance(d, dict) else {}


def torso_gravity_ff(params: "dict | None" = None) -> dict:
    """{scale, offset_nm} from control_param.json (torso.gravity_feedforward) — the
    live-measured trim on the posture-aware MJCF torso hold torque (fitted by
    hardware_bridge/experiments/exp_torso_gravity_sysid.py)."""
    p = params if params is not None else load_control_params()
    ff = (p.get("torso") or {}).get("gravity_feedforward")
    return ff if isinstance(ff, dict) else {}
