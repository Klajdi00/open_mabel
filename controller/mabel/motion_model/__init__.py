"""
mabel.motion_model — the tip-safety motion model of the mobile base.

Every base twist, from every input device (joystick, iOS/VisionPro nav sticks,
web GUI, ROS nav, admittance glide), MUST pass through `safe_base_step(...)`
before it reaches the swerve IK — never call
`mabel.whole_body_controller.base_controller.swerve_ik` on a raw twist.

Modules
-------
tip_over       lift-height-aware per-direction caps + coupled ZMP envelope
               (`safe_base_step`, `tip_usage`, `max_speed_dir`, ...).
               NumPy-only: importable on the robot without MuJoCo.
com_support    live CoM / support-polygon measurement from the MuJoCo model
               (`robot_com`, `support_margin`, `CoMBaseSupport`) — used to
               verify/re-derive the constants baked into tip_over.
base_response  calibrated forward model of the swerve base's twist response
               (`BaseMotionModel`, `predict_twist`, `integrate`) — used by the
               ROS command-odometry node.

`tip_over` is imported eagerly (NumPy-only); the MuJoCo-backed helpers load
lazily so `import mabel.config` stays lightweight on the robot.
"""
import importlib

from .tip_over import *          # noqa: F401,F403  (numpy-only, eager)
from . import tip_over           # noqa: F401

_LAZY = {
    # com_support (MuJoCo-backed)
    "robot_com": "com_support", "support_polygon": "com_support",
    "support_margin": "com_support", "com_offset_base": "com_support",
    "angular_tip_margin": "com_support", "live_tip_usage": "com_support",
    "CoMBaseSupport": "com_support",
    # base_response (calibrated forward model)
    "BaseMotionModel": "base_response", "predict_twist": "base_response",
    "integrate": "base_response",
    # swerve_modules (per-module drive/steer limits, NumPy-only)
    "SwerveModules": "swerve_modules", "check_limits": "swerve_modules",
}


def __getattr__(name):                      # PEP 562 lazy attribute
    mod = _LAZY.get(name)
    if mod is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f".{mod}", __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY.keys()))
