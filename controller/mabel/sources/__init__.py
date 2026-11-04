"""
mabel.sources — operator-input ingest: every device lands on `OperatorPose`.

The retargeting stack is device-agnostic: whatever produces the operator frame
(Vision Pro ARKit skeleton over ZMQ, the iOS / web / VisionPro apps' teleop
wire format over WebSocket, a webcam skeleton from the browser studios, or a
synthetic mock), it is converted HERE into the one `mabel.types.OperatorPose`
contract, and everything downstream (whole_body_retargeting, hand_retargeting,
the facade) is identical.

    wire   the apps' teleop wire format (iOS / Vision Pro / web GUI / webcam
           studios) -> OperatorPose  (`operator_pose_from_teleop_frame`),
           plus the nav-joystick decode (`app_joystick`).
    avp    direct Vision Pro ZMQ stream ingest (`AVPReceiver`, `AVPFrame`).
    mock   synthetic operator frames for tests/benchmarks (`mock_frame`).
"""
from .wire import operator_pose_from_teleop_frame, app_joystick  # noqa: F401
from .avp import AVPReceiver, AVPFrame                           # noqa: F401
from .mock import mock_frame                                     # noqa: F401

__all__ = ["operator_pose_from_teleop_frame", "app_joystick",
           "AVPReceiver", "AVPFrame", "mock_frame"]
