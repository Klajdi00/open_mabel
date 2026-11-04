"""Auto-detect whether MABEL is running in simulation or on real hardware.

The MABEL sim (``mabel_sim`` mujoco_node) publishes ``/clock``; the real robot
does NOT. So we probe the live ROS graph for a ``/clock`` publisher to decide
sim-vs-real -- the SLAM launches need no manual ``use_sim_time`` / sim flag, they
set the clock and map the right camera topics automatically.

The probe runs in a SUBPROCESS so importing/initialising rclpy never pollutes the
launch process (and a probe failure just falls back to 'real' instead of breaking
the launch). Override the discovery window with MABEL_SIM_PROBE_SECS.
"""
import os
import subprocess
import sys

_PROBE = r'''
import os, time, rclpy
rclpy.init()
node = rclpy.create_node("_mabel_slam_sim_probe")
secs = float(os.environ.get("MABEL_SIM_PROBE_SECS", "2.0"))
deadline = time.time() + secs
found = False
while time.time() < deadline:
    if node.count_publishers("/clock") > 0:   # mujoco_node publishes /clock in sim
        found = True
        break
    rclpy.spin_once(node, timeout_sec=0.1)
print("SIM" if found else "REAL")
node.destroy_node()
rclpy.shutdown()
'''


def detect_is_sim(probe_secs=5.0):
    """True if a /clock publisher (the MuJoCo sim) is on the graph, else False.

    The window is generous (5 s) ON PURPOSE: a MISSED /clock detection silently
    cascades -- use_sim_time stays False, so nvblox/cuVSLAM stamp TF lookups in
    wall-clock while the sim publishes sim-time TF, every lookup fails, nothing
    integrates and NO mesh is ever built. Across the conda(sim)<->apt(SLAM) DDS
    boundary discovery can take a couple of seconds, so a 2 s probe was flaky.
    When in doubt pass use_sim_time:=true explicitly to skip the probe entirely."""
    env = dict(os.environ, MABEL_SIM_PROBE_SECS=str(probe_secs))
    try:
        out = subprocess.run([sys.executable, '-c', _PROBE], env=env,
                             capture_output=True, text=True, timeout=probe_secs + 8)
        return 'SIM' in out.stdout
    except Exception:       # noqa: BLE001 -- a probe failure must never break launch
        return False        # default to real hardware (the safe wall-clock choice)


_TOPIC_PROBE = r'''
import os, time, rclpy
rclpy.init()
node = rclpy.create_node("_mabel_slam_topic_probe")
topic = os.environ["MABEL_TOPIC_PROBE"]
deadline = time.time() + float(os.environ.get("MABEL_SIM_PROBE_SECS", "3.0"))
found = False
while time.time() < deadline:
    if node.count_publishers(topic) > 0:
        found = True
        break
    rclpy.spin_once(node, timeout_sec=0.1)
print("FOUND" if found else "ABSENT")
node.destroy_node()
rclpy.shutdown()
'''


def detect_has_publisher(topic, probe_secs=3.0):
    """True if `topic` has at least one live publisher on the graph.

    Same subprocess-probe pattern as detect_is_sim (never pollutes / breaks the
    launch process). Used e.g. to auto-enable cuVSLAM IMU fusion only when the
    camera driver actually publishes /camera/front/imu."""
    env = dict(os.environ, MABEL_SIM_PROBE_SECS=str(probe_secs),
               MABEL_TOPIC_PROBE=topic)
    try:
        out = subprocess.run([sys.executable, '-c', _TOPIC_PROBE], env=env,
                             capture_output=True, text=True, timeout=probe_secs + 8)
        return 'FOUND' in out.stdout
    except Exception:       # noqa: BLE001 -- a probe failure must never break launch
        return False


def resolve_is_sim(value, probe_secs=5.0):
    """Map a launch-arg string to an is_sim bool.

    value: 'auto' (default) -> probe /clock; 'true'/'sim' -> force sim;
    'false'/'real' -> force real. Use the explicit forms to skip the probe (e.g.
    when sim + SLAM launch together and /clock may not be up yet)."""
    v = (value or 'auto').strip().lower()
    if v in ('true', '1', 'yes', 'sim'):
        return True
    if v in ('false', '0', 'no', 'real'):
        return False
    return detect_is_sim(probe_secs)
