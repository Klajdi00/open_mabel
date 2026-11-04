"""
MABEL whole-body teleoperation & control library.

Frontend-agnostic: every frontend (Apple Vision Pro, iOS, web GUI, webcam
studio, ROS, keyboard/joystick) only produces an `OperatorPose` (or joint/nav
commands) and consumes a `RobotCommand` via the `MabelTeleop` facade — it never
touches MuJoCo, ZMQ, or the retargeting internals. Every input path goes
through the SAME motion model, dynamic models, retargeting, and whole-body
controllers before anything reaches an actuator.

Package layout (each subpackage has its own docstring):

    sources/                 device ingest -> OperatorPose (wire / AVP / mock)
    hand_retargeting/        finger skeleton -> ORCA hands (position, decoupled)
    whole_body_retargeting/  head+wrists -> arms, neck (decoupled), followers,
                             per-operator calibration
    motion_model/            tip-over safety envelope (lift-aware ZMP), CoM
                             measurement, calibrated base response
    dynamic_models/          gravity/bias/inertia + external-torque estimation
                             per region: arms/ upper_body/ whole_body/
    whole_body_controller/   ik, position_controller (Day-1 PD),
                             gravity_compensation, collision_avoidance,
                             admittance (force estimation + base glide),
                             compliance, base_controller (swerve), engine
    safety/                  teleop-link fail-safe: EngagementSupervisor + clutch
    runtime/                 loop instrumentation: PerfMonitor / Dashboard / recorder
    plant.py                 the ONE sim/real seam (SimPlant / RealPlant)
    facade.py                MabelTeleop — the one-call entry the bridges drive

High level
----------
MabelTeleop      one-call facade: OperatorPose -> RobotCommand (composes everything)
OperatorPose     device-agnostic operator pose input (types.py)
RobotCommand     device-agnostic actuator-target output (types.py)

Building blocks (each hides its complexity behind a verb)
---------------------------------------------------------
TeleopController production controller: retarget + whole-body IK + swerve + neck
Retargeter       operator wrist/head poses -> robot palm SE(3) targets
HandIKRetargeter operator finger skeleton -> ORCA finger joint targets
WholeBodyIK      palm SE(3) targets -> lift/torso/arm/wrist joint targets
NeckController   gaze direction -> 3-DoF neck
WBCEngine        torque runtime: impedance/compliance x arms/upper/whole body
JointPositionController   Day-1 plain PD (no system ID needed)
GravityCompController     region gravity feedforward (back-drivable float)
ComplianceController      gravity-comp'd region a human can move by hand
BaseAdmittance / WrenchEstimator / ContactEstimator   push -> base glide
swerve_ik(vx,vy,omega,data)       body twist -> steer/drive wheel commands
config.safe_base_step(...)        tip-safe base motion model (lift-height aware)
SelfCollisionGuard                capsule self-collision avoidance (CBF / QP)
EngagementSupervisor              tracking-loss fail-safe (ease HOME / ease back)
ClutchController                  single-hand double-pinch re-indexing
PerfMonitor                       live per-stage rate/latency HUD
AVPReceiver / AVPFrame            ZMQ ingest of the Vision Pro stream

Tunables live in `config` (config.SAFE_*, config.CLUTCH_*, config.SC_*, ...);
the tip-over motion model itself is `mabel.motion_model` (config re-exports it).

Import model
------------
`config` is imported eagerly (it needs only numpy), so lightweight consumers --
e.g. the ROS navigation node that reuses the tip-safe motion model -- can do
`from mabel.config import safe_base_step` WITHOUT pulling in MuJoCo/scipy. Every
other public symbol is loaded lazily on first attribute access (PEP 562), so
`import mabel` stays cheap and `from mabel import MabelTeleop` still works exactly
as before -- it just defers the heavy MuJoCo import to that moment.

Old flat module paths (mabel.wbc, mabel.swerve, mabel.self_collision, mabel.hud,
...) are resolved to their new subpackage homes by a lazy meta-path finder (see
`_LegacyModuleFinder` below) — no file-per-shim clutter, and still MuJoCo-free until
first use. The contract is `tests/test_imports_compat.py`; extend it for new consumers.
"""
import importlib
import importlib.abc
import importlib.machinery
import sys

from . import config  # noqa: F401  (numpy-only; safe to import eagerly)

# Public symbol -> submodule that defines it. Resolved lazily by __getattr__ so
# that importing `mabel` (or `mabel.config`) does not eagerly import MuJoCo/scipy.
_LAZY = {
    # frontend-agnostic facade + contracts
    "OperatorPose": "types", "RobotCommand": "types",
    "MabelTeleop": "facade",
    "app_joystick": "sources.wire",
    # production control stack
    "TeleopController": "teleop_controller",
    "Retargeter": "whole_body_retargeting.wrist_clutch",
    "ArmRetargeter": "whole_body_retargeting.arm_cosine",
    "HandIKRetargeter": "hand_retargeting.keypoint_ik",
    "make_hand_retargeter": "hand_retargeting.vector",
    "WholeBodyIK": "whole_body_controller.ik",
    "NeckController": "whole_body_retargeting.neck",
    "swerve_ik": "whole_body_controller.base_controller.swerve",
    "SelfCollisionGuard": "whole_body_controller.collision_avoidance.capsule_guard",
    "solve_qp": "whole_body_controller.collision_avoidance.capsule_guard",
    "CoMBaseSupport": "motion_model.com_support",
    "robot_com": "motion_model.com_support",
    "support_margin": "motion_model.com_support",
    # torque-domain control
    "WBCEngine": "whole_body_controller.engine",
    "JointPositionController": "whole_body_controller.position_controller.pd",
    "GravityCompController": "whole_body_controller.gravity_compensation.gravity_comp",
    "ComplianceController": "whole_body_controller.compliance.compliance",
    "BaseAdmittance": "whole_body_controller.admittance.base_admittance",
    "external_base_wrench": "whole_body_controller.admittance.base_admittance",
    "WrenchEstimator": "whole_body_controller.admittance.wrench_estimation",
    "ContactEstimator": "whole_body_controller.admittance.wrench_estimation",
    # dynamics + plant seam
    "MujocoDynamics": "dynamic_models.model",
    "MomentumObserver": "dynamic_models.observers",
    "ArmsDynamicModel": "dynamic_models.arms",
    "UpperBodyDynamicModel": "dynamic_models.upper_body",
    "WholeBodyDynamicModel": "dynamic_models.whole_body",
    "Plant": "plant", "SimPlant": "plant",
    # safety + telemetry
    "EngagementSupervisor": "safety", "ClutchController": "safety",
    "position_actuators": "safety",
    "PerfMonitor": "runtime.hud",
    # operator-stream ingest
    "AVPReceiver": "sources.avp", "AVPFrame": "sources.avp",
    # per-operator body/hand calibration
    "OperatorProfile": "whole_body_retargeting.calibration",
    "BodyCalibrator": "whole_body_retargeting.calibration",
}


def __getattr__(name):                      # PEP 562 module-level lazy attribute
    mod = _LAZY.get(name)
    if mod is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f".{mod}", __name__), name)
    globals()[name] = value                 # cache so __getattr__ runs once
    return value


# ── Legacy flat-module compatibility ─────────────────────────────────────────
# Pre-2026-07 code imports flat paths like `mabel.wbc`, `mabel.swerve`,
# `mabel.self_collision`, `mabel.hud`. The real code now lives in the subpackages
# below; rather than keep a file per old name cluttering the package root, a single
# lazy meta-path finder aliases each legacy name onto its new home ON FIRST IMPORT.
# Laziness matters: `import mabel` / `import mabel.config` must stay MuJoCo-free (the
# ROS nav node imports config on the Jetson without MuJoCo), so nothing here is
# imported until a consumer actually asks for the legacy module. Contract lives in
# tests/test_imports_compat.py. Most old names are a 1:1 rename onto one module;
# `dynamics` aggregated several homes, so its body is in `_compat/dynamics.py`.
_LEGACY_MODULES = {
    "admittance":          "mabel.whole_body_controller.admittance.base_admittance",
    "arm_retarget":        "mabel.whole_body_retargeting.arm_cosine",
    "avp_receiver":        "mabel.sources.avp",
    "base_motion":         "mabel.motion_model.base_response",
    "calibration":         "mabel.whole_body_retargeting.calibration",
    "com_support":         "mabel.motion_model.com_support",
    "dynamics":            "mabel._compat.dynamics",
    "force_estimation":    "mabel.whole_body_controller.admittance.wrench_estimation",
    "hand_ik_retarget":    "mabel.hand_retargeting.keypoint_ik",
    "hand_retarget":       "mabel.hand_retargeting.anatomical",
    "hands":               "mabel.hand_retargeting.grasp",
    "hand_vector_retarget": "mabel.hand_retargeting.vector",
    "head":                "mabel.whole_body_retargeting.neck",
    "lazy_base":           "mabel.whole_body_retargeting.followers",
    "mock_avp":            "mabel.sources.mock",
    "retarget":            "mabel.whole_body_retargeting.wrist_clutch",
    "self_collision":      "mabel.whole_body_controller.collision_avoidance.capsule_guard",
    "swerve":              "mabel.whole_body_controller.base_controller.swerve",
    "wbc":                 "mabel.whole_body_controller.ik",
    "wbc_runtime":         "mabel.whole_body_controller.engine",
    "wire":                "mabel.sources.wire",
    # real modules that moved into subpackages (dashboard/hud/recorder -> runtime/):
    "dashboard":           "mabel.runtime.dashboard",
    "hud":                 "mabel.runtime.hud",
    "recorder":            "mabel.runtime.recorder",
    # NB: `mabel.safety` is now a package (safety/) at the same path, so it resolves
    # natively — no alias needed.
}


class _LegacyModuleFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Resolve a legacy flat `mabel.<name>` to its new home, lazily and without a
    file per shim. Appended (not prepended) to sys.meta_path, so real submodules
    (config, facade, runtime, …) are found first and only genuinely-gone names hit
    this finder."""

    _PREFIX = __name__ + "."                 # "mabel."

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(self._PREFIX):
            return None
        leaf = fullname[len(self._PREFIX):]
        if "." in leaf or leaf not in _LEGACY_MODULES:
            return None
        return importlib.machinery.ModuleSpec(fullname, self)

    def create_module(self, spec):
        leaf = spec.name[len(self._PREFIX):]
        module = importlib.import_module(_LEGACY_MODULES[leaf])
        sys.modules[spec.name] = module      # alias the old dotted path onto it
        return module

    def exec_module(self, module):           # already executed under its real name
        pass


if not any(isinstance(_f, _LegacyModuleFinder) for _f in sys.meta_path):
    sys.meta_path.append(_LegacyModuleFinder())


def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY.keys()))


__all__ = [
    "config", "OperatorPose", "RobotCommand", "MabelTeleop", "app_joystick",
    "TeleopController", "Retargeter", "ArmRetargeter", "HandIKRetargeter",
    "make_hand_retargeter", "WholeBodyIK", "NeckController", "swerve_ik",
    "SelfCollisionGuard", "solve_qp",
    "WBCEngine", "JointPositionController", "GravityCompController",
    "ComplianceController", "BaseAdmittance", "external_base_wrench",
    "WrenchEstimator", "ContactEstimator",
    "MujocoDynamics", "MomentumObserver", "ArmsDynamicModel",
    "UpperBodyDynamicModel", "WholeBodyDynamicModel", "Plant", "SimPlant",
    "EngagementSupervisor", "ClutchController", "position_actuators",
    "PerfMonitor", "AVPReceiver", "AVPFrame",
    "OperatorProfile", "BodyCalibrator",
]
