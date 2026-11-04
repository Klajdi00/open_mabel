"""subsystems — typed, calibrated, ROS-free subsystem cores.

The middle layer of mabel_hw: each class owns ONE driver, applies calibration
(raw<->model via mabel_hw.calibration), and exposes the typed state/command
surface the facade's handles gate with write leases. The rclpy adapters in
mabel_hw.ros are thin plumbing over these same objects — sim/real/ROS-less
consumers all share this one implementation.

  JointGroupSubsystem   torso and any other MIT joint group, any backend
  ArmsSubsystem         + the calibrated commandable range per joint
  NeckSubsystem         serial direct-drive neck + task space (set_gaze /
                        pitch-roll-yaw == the motor angles, 1:1)
  BaseSubsystem         MAXSwerve twist-in, per-module state + health out
  HandSubsystem         one ORCA hand side
  LiftSubsystem         Pico standing-desk column
  LidarSubsystem        RPLIDAR capture (single owner of the serial port)
  AudioSubsystem        USB speaker + mic (single owner of the sound card)
  CameraSuite           camera stream directory + liveness (frames stay on the
                        network video plane — never through this API)
"""
from .joint_group import JointGroupSubsystem
from .neck import NeckSubsystem
from .arms import ArmsSubsystem
from .base import BaseSubsystem
from .hands import HandSubsystem
from .lift import LiftSubsystem
from .lidar import LidarSubsystem
from .audio import AudioSubsystem
from .cameras import CameraSuite

__all__ = ["JointGroupSubsystem", "ArmsSubsystem", "NeckSubsystem",
           "BaseSubsystem", "HandSubsystem", "LiftSubsystem", "LidarSubsystem",
           "AudioSubsystem", "CameraSuite"]
