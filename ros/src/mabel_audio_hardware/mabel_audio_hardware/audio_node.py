#!/usr/bin/env python3
"""Audio bridge CLIENT — links the ROS graph to the hardware_bridge audio subsystem.

The speaker + mic are owned by the standalone ``~/open_mabel/hardware_bridge`` (the
single HAL): it owns the USB sound card (ALSA capture/playback, the mixer, and
the VID:PID device resolution that finds the bar on ANY port). This node NEVER
opens a device — it is the thin translator/monitor between the MABEL teleop
contract and the bridge, exactly like ``mabel_arms_hardware/arms_node.py`` is
for the arms:

    /audio/speaker       (UInt8MultiArray)  gateway/apps -> bridge  (pass-through:
                                            the bridge subscribes it directly —
                                            no relay hop)
    /mabel_cmd           (MabelSwerveCmd.hardware_state)  e-stop mirror:
        engage  -> /audio/enable false   (mic muted: an e-stopped robot must not
                                          keep streaming the room's audio)
        release -> /audio/enable true
    bridge -> /audio/mic            (UInt8MultiArray)  raw S16_LE PCM, sensor QoS
    bridge -> /audio/state          (Bool, LATCHED)    mic enabled/muted
    bridge -> /audio/device_health  (String, JSON)     {speaker:{...}, mic:{...}} @2 Hz

Liveness is measured on /audio/device_health, NOT /audio/mic: mic capture is
DEMAND-DRIVEN (the bridge releases the card when nobody is subscribed), so an
absent mic stream is the normal idle state and would report a healthy speaker as
offline. device_health publishes at a steady 2 Hz whenever the subsystem is up.
"""

import json
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String

try:
    from mabel_interfaces.msg import MabelSwerveCmd
    _HAVE_CMD = True
except Exception:                                     # bench without the interfaces pkg
    MabelSwerveCmd = None
    _HAVE_CMD = False


class AudioBridgeClient(Node):

    def __init__(self):
        super().__init__("audio_hardware_node")

        # Bridge contract topics (must match hardware_bridge/config.py AUDIO).
        self.declare_parameter("health_topic", "/audio/device_health")
        self.declare_parameter("enable_topic", "/audio/enable")
        self.declare_parameter("state_topic", "/audio/state")
        self.declare_parameter("follow_estop", True)
        self.declare_parameter("liveness_period", 3.0)   # s without health = offline

        self._health_topic = self.get_parameter("health_topic").value
        self._enable_topic = self.get_parameter("enable_topic").value
        self._follow_estop = bool(self.get_parameter("follow_estop").value)
        self._liveness = float(self.get_parameter("liveness_period").value)

        self._last_health_t = None
        self._online = False
        self._estop = None                            # None = never heard /mabel_cmd
        self._card = ""

        self.pub_enable = self.create_publisher(Bool, self._enable_topic, 10)
        self.create_subscription(String, self._health_topic, self._on_health, 5)

        if self._follow_estop and _HAVE_CMD:
            self.create_subscription(MabelSwerveCmd, "/mabel_cmd", self._on_cmd, 10)
        elif self._follow_estop:
            self.get_logger().warn(
                "follow_estop requested but mabel_interfaces is not importable — "
                "the mic will NOT mute on e-stop this run.")

        self.create_timer(1.0, self._check_liveness)
        self.get_logger().info(
            f"audio bridge client up: watching {self._health_topic}"
            + (" · /mabel_cmd e-stop -> mic mute" if self._follow_estop and _HAVE_CMD
               else " · e-stop mirror OFF"))

    # --- liveness ---------------------------------------------------------------
    def _on_health(self, msg: String):
        self._last_health_t = time.monotonic()
        try:
            self._card = (json.loads(msg.data) or {}).get("card", "")
        except Exception:                             # noqa: BLE001
            pass

    def _check_liveness(self):
        online = (self._last_health_t is not None and
                  time.monotonic() - self._last_health_t < self._liveness)
        if online and not self._online:
            self.get_logger().info(
                f"bridge audio ONLINE (card {self._card!r}, "
                f"{self._health_topic} streaming).")
        elif not online and self._online:
            self.get_logger().warn(
                f"bridge audio OFFLINE (no {self._health_topic}) — is the hardware "
                "bridge up with MABEL_HW_AUDIO=true and the USB audio card plugged in? "
                "(hardware_bridge/experiments/exp_audio.py diagnoses the card)")
        self._online = online

    # --- e-stop mirror ----------------------------------------------------------
    def _on_cmd(self, msg):
        engaged = int(getattr(msg, "hardware_state", 0)) == 1
        if engaged == self._estop:
            return                                    # EDGE-triggered: a level-
        self._estop = engaged                         # triggered mirror would spam
        self.pub_enable.publish(Bool(data=not engaged))
        self.get_logger().info(
            f"e-stop {'ENGAGED' if engaged else 'released'} -> mic "
            f"{'muted' if engaged else 'live'}.")


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = AudioBridgeClient()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
