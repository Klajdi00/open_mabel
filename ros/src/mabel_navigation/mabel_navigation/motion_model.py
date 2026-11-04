"""MABEL base motion model -- tip-safe swerve velocity smoothing.

SINGLE SOURCE OF TRUTH: this module is a thin re-export of the tip-over / ZMP
motion model that lives in the whole-body controller library,
``controller/mabel/config.py``. It used to be a verbatim hand-port of that math,
which silently drifted whenever the controller's CoM/lever constants were
re-measured. Importing the one implementation makes drift impossible -- the ROS
navigation stack and the teleop controller now exercise byte-identical logic
(verified by a parity sweep in the package tests).

The motion model turns a *desired* body twist (vx, vy, omega) into a physically
realizable, tip-safe one whose speed/acceleration envelope shrinks ~1/CoM-height
so a raised lift cannot brake or turn hard enough to tip the robot. It is pure
math (numpy only) -- no ROS, no MuJoCo, no sim privilege -- so it runs identically
on the real robot; the only external input is the lift height ``z_lift`` (metres),
read on the robot from ``/joint_states`` (the ``lift_lower`` joint).

`mabel.config` is imported lazily-friendly: it pulls in only numpy (the heavy
MuJoCo/scipy parts of the controller stay unloaded), so this stays a lightweight
dependency for the navigation node.
"""
import os
import sys


def _ensure_mabel_importable():
    """Make the standalone `mabel` controller library importable.

    Prefer an installed / already-on-PYTHONPATH `mabel`. Otherwise fall back to a
    source checkout (env MABEL_ROOT, then the usual repo locations) -- the same
    "installed-or-in-repo" resolution mabel_base_hardware uses for maxswerve-host,
    so a dev box without a pip install still resolves the one true source.
    """
    try:
        import mabel  # noqa: F401
        return
    except ImportError:
        pass
    candidates = []
    root = os.environ.get("MABEL_ROOT")
    if root:
        candidates.append(os.path.join(root, "controller"))
    candidates += [
        os.path.expanduser("~/open_mabel/controller"),
        os.path.expanduser("~/open_mabel/controller"),
    ]
    for path in candidates:
        if os.path.isdir(os.path.join(path, "mabel")) and path not in sys.path:
            sys.path.insert(0, path)
            return


_ensure_mabel_importable()

try:
    # The one true implementation + its constants. Re-exported below so existing
    # callers (`from mabel_navigation import motion_model as mm; mm.safe_base_step`)
    # keep working unchanged.
    from mabel.config import (  # noqa: F401
        # constants
        MAX_LIN_SPEED, MAX_ANG_SPEED, MAX_LIN_ACCEL, MAX_ANG_ACCEL, LIFT_RANGE,
        GRAVITY, TIP_LEVER_FWD, TIP_LEVER_LAT, TIP_LEVER_BACK,
        TIP_SF_FWD, TIP_SF_AGILE, STOP_TIME_FWD, STOP_TIME_AGILE,
        COM_Z0, COM_DZDLIFT, V_SPEED_FLOOR, V_FWD_CEIL, V_AGILE_CEIL,
        SUPPORT_PTS, COM_XY,
        # functions
        com_height, max_decel_dir, max_speed_dir, max_decel, max_speed,
        com_accel, tip_usage, accel_safe_scale, safe_base_step,
    )
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "mabel_navigation.motion_model now re-exports the tip-over model from the "
        "MABEL controller library (`mabel.config`), the single source of truth. It "
        "could not be imported: " + str(e) + ".\nPut the controller on the ROS "
        "Python path -- pip-install `controller/` (package `mabel`), or set "
        "MABEL_ROOT=/path/to/open_mabel, or check it out at ~/open_mabel."
    ) from e
