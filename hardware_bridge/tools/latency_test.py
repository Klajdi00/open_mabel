#!/usr/bin/env python3
"""latency_test.py — measure base command latency through the bridge (mock Teensy).

NO real hardware: a mock Teensy on 127.0.0.1 timestamps each DISTINCT command it
receives (RXSTEP <monotonic> drive0=..). We publish step commands with distinct vx,
record the publish time, and correlate by the resulting drive value (CLOCK_MONOTONIC is
system-wide on Linux, so the two processes' timestamps are directly comparable).

Two paths:
  bridge : /mabel/hw/base/cmd_vel  -> IK+calib -> Teensy        (the pure bridge hop)
  full   : /mabel_cmd -> base_bridge_client -> cmd_vel -> bridge -> Teensy

Run (in the mabel ROS env):  python3 tools/latency_test.py
"""

import os
import re
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
HWB = os.path.dirname(HERE)
TEENSY = "127.0.0.1"
FAKE_LOG = "/tmp/lat_fake.log"
BR_LOG = "/tmp/lat_bridge.log"
CL_LOG = "/tmp/lat_client.log"
LOCK = "/tmp/lat_hwb.lock"
procs = []


def _spawn(cmd, log, **kw):
    p = subprocess.Popen(cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT, **kw)
    procs.append(p)
    return p


def _cleanup():
    for p in procs:
        p.send_signal(2)
    time.sleep(1)
    for p in procs:
        if p.poll() is None:
            p.kill()


def _stats(name, lat):
    if not lat:
        print(f"  {name:<7} NO SAMPLES correlated")
        return
    lat.sort()
    p95 = lat[min(len(lat) - 1, int(0.95 * len(lat)))]
    print(f"  {name:<7} n={len(lat):<3} min={min(lat):.2f}  median={statistics.median(lat):.2f}  "
          f"p95={p95:.2f}  max={max(lat):.2f}   (ms)")


def main():
    os.environ.setdefault("ROS_DOMAIN_ID", "45")
    os.environ["MABEL_HW_LOCK"] = LOCK
    print("starting mock Teensy + bridge + client (127.0.0.1, no real hardware)...")
    _spawn([sys.executable, os.path.join(HERE, "fake_teensy.py"),
            "--host", TEENSY, "--port", "5800"], FAKE_LOG)
    env = dict(os.environ, MABEL_HW_BASE="true", MABEL_HW_BASE_TEENSY_IP=TEENSY,
               MABEL_HW_ARMS="false", MABEL_HW_BODY="false", MABEL_HW_HANDS="false",
               MABEL_HW_LIFT="false", MABEL_HW_CAMERAS="false")
    _spawn(["bash", os.path.join(HWB, "run_bridge.sh")], BR_LOG, cwd=HWB, env=env)
    _spawn(["ros2", "run", "mabel_base_hardware", "swerve_node",
            "--ros-args", "-r", "__node:=lat_client"], CL_LOG, env=env)
    time.sleep(6)

    import rclpy
    from geometry_msgs.msg import Twist, Vector3
    from mabel_interfaces.msg import MabelSwerveCmd

    rclpy.init()
    node = rclpy.create_node("lat_pub")
    pub_cmd_vel = node.create_publisher(Twist, "/mabel/hw/base/cmd_vel", 10)
    pub_mabel = node.create_publisher(MabelSwerveCmd, "/mabel_cmd", 10)
    time.sleep(1.5)   # discovery

    def run_phase(publish, vx_of):
        pubs = {}
        for k in range(1, 31):
            v = vx_of(k)
            expected = round(-v, 3)        # drive0 at Teensy (back module, DRIVE_DIR=-1)
            t0 = time.monotonic()
            publish(v)
            pubs[expected] = t0
            end = time.monotonic() + 0.12
            while time.monotonic() < end:
                rclpy.spin_once(node, timeout_sec=0.005)
        time.sleep(0.4)
        return pubs

    # Phase 1: bridge-only (publish Twist directly to the bridge contract).
    bridge_pubs = run_phase(
        lambda v: pub_cmd_vel.publish(Twist(linear=Vector3(x=float(v)))),
        lambda k: 0.05 + 0.025 * k)
    # Phase 2: full path through /mabel_cmd -> client -> bridge.
    full_pubs = run_phase(
        lambda v: pub_mabel.publish(MabelSwerveCmd(vx=float(v))),
        lambda k: 1.0 + 0.025 * k)         # distinct, non-overlapping vx range

    node.destroy_node()
    rclpy.shutdown()
    time.sleep(0.3)

    # Correlate RXSTEP lines by drive0 value.
    bridge_lat, full_lat = [], []
    for line in open(FAKE_LOG):
        m = re.search(r"RXSTEP ([\d.]+) drive0=(-?[\d.]+)", line)
        if not m:
            continue
        t_rx = float(m.group(1)); d0 = round(float(m.group(2)), 3)
        if d0 in bridge_pubs and t_rx >= bridge_pubs[d0]:
            bridge_lat.append((t_rx - bridge_pubs[d0]) * 1000.0)
        elif d0 in full_pubs and t_rx >= full_pubs[d0]:
            full_lat.append((t_rx - full_pubs[d0]) * 1000.0)

    print("\n=== command latency (publish -> Teensy receives it) ===")
    _stats("bridge", bridge_lat)
    _stats("full", full_lat)
    print("\nbridge = cmd_vel->IK->UDP ; full = /mabel_cmd->client->bridge->UDP")


if __name__ == "__main__":
    try:
        main()
    finally:
        _cleanup()
