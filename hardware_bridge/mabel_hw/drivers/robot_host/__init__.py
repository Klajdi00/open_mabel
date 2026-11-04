"""robot_host — generic Python client for the Teensy robot UDP control link.

VENDORED from the upstream arm stack

Self-describing: the client learns its actuator layout from the firmware's
RobotInfo handshake at connect(), so ONE client serves any robot defined by the
Teensy's /robot.json — the openArm dual (l_j1..r_j7) and the 3-DOF RSSR neck
ride the same firmware + this same client, differing only in config.

SAFETY: connect() ENABLES the motors (keepalive => firmware damping hold);
close()/silence => limp within ~0.25 s. See client.py's module docstring.
"""
from .client import (RobotClient, RobotState, RobotHealth, RobotInfo,
                     ActuatorInfo)
from . import robot_pb2

__all__ = ["RobotClient", "RobotState", "RobotHealth", "RobotInfo",
           "ActuatorInfo", "robot_pb2"]
