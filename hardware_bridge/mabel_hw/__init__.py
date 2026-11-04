"""mabel_hw — the MABEL hardware access layer, as an importable library.

The hardware bridge process is the ONE process allowed to touch MABEL's devices
(flock-enforced). This package is what that process is built from — and the API
any in-process consumer (the DDS adapters, the hand RPC, experiments, tests)
programs against:

    from mabel_hw import MabelHardware

    with MabelHardware() as hw:                    # flock -> calibration -> connect
        state = hw.read()                          # RobotState frozen snapshot
        with hw.left_arm.claim("teleop") as arm:   # exclusive writer lease
            arm.set_joint_positions({"l_j1": 0.3})
        hw.estop()                                 # lease-free, always allowed

Layering (each layer depends only on the one below):

    api.MabelHardware      facade: read()/subscribe(), handles, estop     (api.py)
      subsystems.*         typed + calibrated + lease-gated per subsystem
        arbitration        exactly-one-writer leases                (arbitration.py)
        calibration        calibration_data/ store + anchor math    (calibration/)
        drivers.*          ROS-free device protocols (CAN, UDP, serial)
    types                  frozen state/command contracts               (types.py)

Out-of-process clients (server, ROS nodes, GUIs, apps) do NOT import this — they
keep speaking the DDS topics / video ports /:8779 hand RPC, which the thin
adapters in mabel_hw.ros / mabel_hw.rpc serve on top of this same API.

Import weight: ``import mabel_hw`` is cheap (PEP 562 lazy exports). rclpy is only
imported inside mabel_hw.ros / mabel_hw.cameras; numpy only inside drivers.
"""
from __future__ import annotations

import importlib

# Eager: config is os/env-only, and it is the one module lightweight consumers
# (launch scripts, the GUI contract) reach for first.
from . import config                                             # noqa: F401

# name -> submodule that defines it (resolved on first attribute access)
_LAZY = {
    # facade
    "MabelHardware": "api", "Subscription": "api",
    # drivers (heavy deps stay lazy: numpy/serial/protobuf load on first touch)
    "SwerveBase": "drivers.swerve",
    "SwerveClient": "drivers.maxswerve_host",
    "TeensyLink": "drivers.teensy_link", "TeensyJointGroup": "drivers.teensy_link",
    "CanLink": "drivers.damiao_can",
    "build_joint_driver": "drivers",
    "RobotClient": "drivers.robot_host",
    "DamiaoGroup": "drivers.damiao_can", "DamiaoBus": "drivers.damiao_can",
    "HandsController": "drivers.hands", "Hand": "drivers.hands",
    "OrcaHand": "drivers.orcahand",
    "LiftDriver": "drivers.lift",
    "RPLidar": "drivers.rplidar",
    "AlsaAudio": "drivers.audio",
    # contracts
    "RobotState": "types", "JointGroupState": "types", "NeckState": "types",
    "SwerveModuleState": "types", "BaseHealth": "types", "BaseState": "types",
    "HandState": "types", "LiftState": "types", "LidarScan": "types",
    "CameraStreamInfo": "types", "SensorsState": "types",
    "AudioChunk": "types", "AudioState": "types",
    "JointCommand": "types", "TwistCommand": "types", "GazeCommand": "types",
    # arbitration
    "LeaseBroker": "arbitration", "Lease": "arbitration",
    "LeaseInfo": "arbitration", "GROUPS": "arbitration",
    # driver seam
    "JointGroupDriver": "drivers.seam", "GroupFeedback": "drivers.seam",
    "MockJointGroupDriver": "drivers.seam",
    # errors
    "HardwareError": "errors", "SubsystemUnavailable": "errors",
    "GroupBusy": "errors", "NotLeaseHolder": "errors",
    "InfeasiblePose": "errors", "CalibrationError": "errors",
}

__all__ = sorted(_LAZY)


def __getattr__(name):
    try:
        module_path = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module 'mabel_hw' has no attribute {name!r}") from None
    module = importlib.import_module(f".{module_path}", __name__)
    value = getattr(module, name)
    globals()[name] = value            # cache: next access skips __getattr__
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
