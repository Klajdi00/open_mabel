#!/usr/bin/env python3
"""diag_neck_path.py — pinpoint WHY the neck slider doesn't move the real robot.

Both the web control center and the iOS app are clients of the SAME gateway (ws :9090),
so if BOTH fail the break is downstream of the gateway, not in a client. The command chain
for a neck slider is:

    joint_command{neck_*}  →  gateway facade (data.ctrl["Neck Pitch"])
                           →  command_extract  →  /body/command {neck_pitch,…}
                           →  HAL body_node.set_gaze  →  NeckSubsystem  →  servo
    torque enable (estop-off) → gateway ROSPlant.set_neck_torque → /body/torque_enable (Bool)
                           →  HAL body_node._on_torque → neck writer.set_torque(True)

This probes the two ROS seams the gateway and the HAL meet at and prints a verdict:

    ros2 run … OR:  python3 server/tools/diag_neck_path.py
    (needs ROS sourced + the robot's DDS domain — run it where real_run.sh runs.)

It is READ-ONLY: it only subscribes/inspects topics; it never commands a motor.
Move a neck slider in the studio while it runs so it can see /body/command change.
"""
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from std_msgs.msg import Bool
from sensor_msgs.msg import JointState

_LATCHED = QoSProfile(depth=1)
_LATCHED.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
_LATCHED.reliability = QoSReliabilityPolicy.RELIABLE

TORQUE_EN = "/body/torque_enable"
TORQUE_ST = "/body/torque_state"
BODY_CMD = "/body/command"
BODY_JS = "/body/joint_states"
NECK = ("neck_pitch", "neck_roll", "neck_yaw")


def main():
    rclpy.init()
    n = Node("diag_neck_path")
    seen = {"torque_state": None, "cmd_neck": {}, "js_neck": {}, "cmd_count": 0}

    n.create_subscription(Bool, TORQUE_ST,
                          lambda m: seen.__setitem__("torque_state", bool(m.data)), _LATCHED)

    def _cmd(m):
        d = dict(zip(m.name, m.position))
        nk = {k: round(d[k], 4) for k in NECK if k in d}
        if nk:
            seen["cmd_neck"] = nk
            seen["cmd_count"] += 1
    n.create_subscription(JointState, BODY_CMD, _cmd, 10)

    def _js(m):
        d = dict(zip(m.name, m.position))
        seen["js_neck"] = {k: round(d[k], 4) for k in NECK if k in d}
    n.create_subscription(JointState, BODY_JS, _js, 10)

    print("Probing the neck command path for ~6 s — MOVE A NECK SLIDER now …\n")
    t0 = time.time()
    while time.time() - t0 < 6.0:
        rclpy.spin_once(n, timeout_sec=0.1)

    def count(topic):
        return n.count_publishers(topic), n.count_subscribers(topic)

    tep, tes = count(TORQUE_EN)
    cmp_, cms = count(BODY_CMD)
    jsp, jss = count(BODY_JS)

    print("=== ROS seams ===")
    print(f"{TORQUE_EN:22s} publishers={tep} subscribers={tes}   "
          f"(want: pub≥1 = gateway ROSPlant, sub≥1 = HAL body_node)")
    print(f"{BODY_CMD:22s} publishers={cmp_} subscribers={cms}   "
          f"(want: pub≥1 = gateway, sub≥1 = HAL body_node)")
    print(f"{BODY_JS:22s} publishers={jsp} subscribers={jss}   (HAL neck telemetry)")
    print("\n=== live values ===")
    print(f"neck torque_state (latched): {seen['torque_state']}   (want True to move)")
    print(f"/body/command neck targets seen: {seen['cmd_neck']}  ({seen['cmd_count']} msgs)")
    print(f"/body/joint_states neck now:     {seen['js_neck']}")

    print("\n=== VERDICT ===")
    if tep == 0:
        print("✗ NOBODY publishes /body/torque_enable → the gateway ROSPlant has no")
        print("  set_neck_torque (STALE gateway process). Restart real_run.sh / mabel-bringup")
        print("  so the gateway re-imports the current mabel_control.ros_plant.")
    elif tes == 0:
        print("✗ /body/torque_enable has a publisher but NO subscriber → the HAL body_node")
        print("  isn't up / didn't claim the neck (MABEL_HW_NECK, or the bus is busy —")
        print("  another owner holds /dev/ttyUSB0). Check the HAL bring-up log.")
    elif seen["torque_state"] is not True:
        print("✗ Torque is not ON (torque_state != True). Disengage E-stop / press Torque ON")
        print("  in the studio; the neck boots limp and /body/command is inert until energised.")
    elif seen["cmd_count"] == 0:
        print("✗ Torque is on, but NO neck target reached /body/command while you moved the")
        print("  slider → the break is UPSTREAM in the gateway (control mode = retarget pops")
        print("  the neck; use Joint space, or the client isn't sending joint_command).")
    elif not seen["js_neck"]:
        print("~ Commands flow + torque on, but the HAL publishes no neck joint_states →")
        print("  the neck servos aren't answering (bus/power). Check exp_neck.py on the HAL.")
    else:
        print("✓ Torque on, /body/command carries neck targets, and the HAL streams neck")
        print("  state. If the head still isn't moving, it's mechanical / calibration")
        print("  (check the neck/serial _task limits aren't clamping to ~0).")

    n.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
