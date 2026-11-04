#!/usr/bin/env bash
# USER-REQUESTED torso gravity sweep: hold at angles well inside the travel and
# record the motor's measured torque. Sole commander (gateway stopped).
source /home/mabel/open_mabel/ros/scripts/mabel_env.sh >/dev/null 2>&1 || \
  source /opt/ros/jazzy/setup.bash
python3 - <<'PY'
import json, math, time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool

# Hold points (model rad, 0 = upright). Travel usable window is [-1.035, +0.215]
# (stops minus margin); we stay well inside it.
DOWN = [0.10, 0.0, -0.15, -0.30, -0.45, -0.60, -0.75]
ANGLES = DOWN + DOWN[::-1][1:]        # down then back up: friction cancels in the mean

rclpy.init()
n = Node("torso_grav_probe")
state = {}
def on_js(m):
    d = dict(zip(m.name, zip(m.position, m.effort)))
    if "torso" in d:
        state["pos"], state["eff"] = d["torso"]
        state["t"] = time.time()
n.create_subscription(JointState, "/body/joint_states", on_js, 10)
cmd = n.create_publisher(JointState, "/body/command", 10)
tq = n.create_publisher(Bool, "/body/torque_enable", 10)

def spin(sec):
    end = time.time() + sec
    while time.time() < end:
        rclpy.spin_once(n, timeout_sec=0.1)

def send(target):
    m = JointState()
    m.header.stamp = n.get_clock().now().to_msg()
    m.name, m.position, m.effort = ["torso"], [float(target)], [0.0]  # NO feedforward:
    cmd.publish(m)                                # the PD's own effort IS the measurement

spin(2.0)
assert "pos" in state, "no /body/joint_states — is the bridge up?"
print(f"start: torso at {state['pos']:+.3f} rad, enabling torque (safe hold)...", flush=True)
tq.publish(Bool(data=True))
spin(1.5)

rows = []
for a in ANGLES:
    for _ in range(30):                           # keep the target latched while gliding
        send(a); spin(0.1)
    # settle, then average 1.5 s of feedback
    ps, es = [], []
    end = time.time() + 1.5
    while time.time() < end:
        send(a); spin(0.1)
        if time.time() - state.get("t", 0) < 0.5:
            ps.append(state["pos"]); es.append(state["eff"])
    if ps:
        pm, em = sum(ps)/len(ps), sum(es)/len(es)
        rows.append((a, pm, em))
        print(f"  target {a:+.2f} -> held {math.degrees(pm):+6.1f} deg,"
              f" measured torque {em:+6.2f} N.m", flush=True)

# return upright, then free (balanced there per the operator)
for _ in range(40):
    send(0.0); spin(0.1)
tq.publish(Bool(data=False))
spin(0.5)
print("torque OFF — sweep done", flush=True)
json.dump(rows, open("/tmp/torso_grav_rows.json", "w"))
rclpy.shutdown()
PY
