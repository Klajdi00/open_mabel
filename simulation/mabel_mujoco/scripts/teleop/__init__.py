"""BACK-COMPAT SHIM — the teleop engine moved into the shipped controller
package: ``mabel.teleop_engine`` (controller/mabel/teleop_engine, 2026-07-22).

Why it moved: the engine is the retargeting + whole-body control stack that
BOTH the sim and the REAL robot run (the MabelTeleop facade composes it for
every dispatch target), so it belongs in the installable `mabel` library —
one code path, tested in sim, deployed unchanged to the robot. A directory
named "simulation/..." serving the real robot was the same trap as the
"sim_teleop_bridge serves the real robot too" naming collision.

This shim keeps every historical import working unchanged:

    sys.path.insert(0, ".../simulation/mabel_mujoco/scripts")
    from teleop import TeleopEngine            # scripts, bridge (pre-move)
    from teleop.hyperparams import REACH_SCALE # tools, tests
    from teleop.safety import EngagementSupervisor

New code should import ``mabel.teleop_engine`` directly. Do NOT add real code
here — edit controller/mabel/teleop_engine/.
"""
import os
import sys

# Make the controller package importable for legacy callers that only put the
# scripts dir on sys.path (deps.py used to do this from inside the package).
_CONTROLLER = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "..", "..", "controller"))
if os.path.isdir(_CONTROLLER) and _CONTROLLER not in sys.path:
    sys.path.insert(0, _CONTROLLER)

from mabel.teleop_engine import *              # noqa: F401,F403,E402
# star-import skips underscore names some legacy scripts rely on
from mabel.teleop_engine import _HAVE_SWERVE, _HAVE_SELFCOL     # noqa: F401,E402
from mabel.teleop_engine.hyperparams import _GROUP              # noqa: F401,E402
from mabel.teleop_engine.hud import _A         # noqa: F401,E402

# Alias every submodule so `import teleop.arm` / `from teleop.engine import X`
# resolve to the moved package (one module object, no duplicate state).
# importlib, not getattr: lazily-imported submodules (hand_keypoint) are not
# package attributes until something imports them.
import importlib                               # noqa: E402
for _name in ("hyperparams", "logsetup", "deps", "geometry", "hand",
              "hand_anatomical", "hand_keypoint", "arm", "collision",
              "motion_model", "wholebody", "safety", "hud", "overlay",
              "model_utils", "engine"):
    sys.modules[f"{__name__}.{_name}"] = importlib.import_module(
        f"mabel.teleop_engine.{_name}")
del _name
