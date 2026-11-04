"""rpc — ROS-free network surfaces of the hardware bridge.

  hand_service  HandBridgeService: the transport-agnostic ORCA-hand owner
                (full low-level surface: raw ticks, registers, OFSCAL, calib save)
  hand_server   TCP newline-JSON server/client on 127.0.0.1:8779 + ensure_daemon
                (the macOS-capable path — no rclpy anywhere here)

Public stable entry: ``python3 -m mabel_hw.rpc.hand_server``.
"""
