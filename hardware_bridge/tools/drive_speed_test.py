#!/usr/bin/env python3
"""drive_speed_test.py — verify the DRIVE VELOCITY commanded through the bridge.

NO real hardware: a mock Teensy on 127.0.0.1 records every command. We check that the
keyboard's nav-mode forward command reaches the Teensy at the FULL commanded velocity
(not shrunk), that the I/K keys raise the max speed live (nav mode), and that a direct
cmd_vel is passed through uncapped. This pins down whether "drive too tiny" is a command
bug or just a too-low max-speed setting.

  keystroke -> keyboard_teleop -> /mabel_cmd -> base_bridge_client -> bridge(IK) -> mock
"""
import os
import pty
import re
import statistics
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
HWB = os.path.dirname(HERE)
TEENSY = "127.0.0.1"
FAKE_LOG = "/tmp/ds_fake.log"
procs = []


def spawn(cmd, log, **kw):
    p = subprocess.Popen(cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT, **kw)
    procs.append(p)
    return p


def cleanup():
    for p in procs:
        try:
            p.send_signal(2)
        except Exception:
            pass
    time.sleep(1)
    for p in procs:
        if p.poll() is None:
            p.kill()
    # PID-only (no name-based pkill — that would also kill a REAL bridge).


def fake_drive_mag(t0, t1):
    """|drive0| values the mock logged within [t0,t1] (RXSTEP + per-second status)."""
    out = []
    for line in open(FAKE_LOG):
        m = re.search(r"RXSTEP ([\d.]+) drive0=(-?[\d.]+)", line)
        if m and t0 <= float(m.group(1)) <= t1:
            out.append(abs(float(m.group(2))))
    return out


def main():
    os.environ.setdefault("ROS_DOMAIN_ID", "48")
    os.environ["MABEL_HW_LOCK"] = "/tmp/ds_hwb.lock"
    env = dict(os.environ, MABEL_HW_BASE="true", MABEL_HW_BASE_TEENSY_IP=TEENSY,
               MABEL_HW_ARMS="false", MABEL_HW_BODY="false", MABEL_HW_HANDS="false",
               MABEL_HW_LIFT="false", MABEL_HW_CAMERAS="false")
    print("mock Teensy + bridge + base_bridge_client (127.0.0.1)...")
    spawn([sys.executable, os.path.join(HERE, "fake_teensy.py"),
           "--host", TEENSY, "--port", "5800"], FAKE_LOG)
    spawn(["bash", os.path.join(HWB, "run_bridge.sh")], "/tmp/ds_bridge.log", cwd=HWB, env=env)
    spawn(["ros2", "run", "mabel_base_hardware", "swerve_node",
           "--ros-args", "-r", "__node:=base_bridge_client"], "/tmp/ds_client.log", env=env)
    time.sleep(6)

    import rclpy
    from geometry_msgs.msg import Twist, Vector3
    from mabel_interfaces.msg import MabelSwerveCmd
    rclpy.init()
    mon = rclpy.create_node("ds_mon")
    vx_samples = []
    mon.create_subscription(MabelSwerveCmd, "/mabel_cmd",
                            lambda m: vx_samples.append((time.monotonic(), m.vx)), 50)
    pub_cmd_vel = mon.create_publisher(Twist, "/mabel/hw/base/cmd_vel", 10)
    threading.Thread(target=lambda: rclpy.spin(mon), daemon=True).start()

    master, slave = pty.openpty()
    kt = subprocess.Popen(
        ["ros2", "run", "mabel_teleop", "keyboard_teleop", "--ros-args",
         "-p", "model:=/nonexistent", "-p", "rate:=20.0"],
        stdin=slave, stdout=slave, stderr=slave, env=env, preexec_fn=os.setsid)
    procs.append(kt)
    os.close(slave)
    threading.Thread(target=lambda: _drain(master), daemon=True).start()
    time.sleep(5)

    def hold(keys, dur):
        end = time.time() + dur
        while time.time() < end:
            os.write(master, keys)
            time.sleep(0.08)

    def vx_window(t0, t1):
        return [v for t, v in vx_samples if t0 <= t <= t1]

    print("\n=== 1. keyboard FORWARD held — is the velocity SUSTAINED at the cap? ===")
    t0 = time.monotonic(); hold(b"\x1b[A", 4.0); t1 = time.monotonic()
    time.sleep(0.3)
    vw = [abs(v) for v in vx_window(t0 + 1.0, t1)]   # after the ramp
    dw = fake_drive_mag(t0 + 1.0, t1 + 0.3)
    if vw:
        print(f"  /mabel_cmd |vx|: median={statistics.median(vw):.3f} min={min(vw):.3f} "
              f"max={max(vw):.3f}  (default cap 0.8)")
    if dw:
        print(f"  mock Teensy |drive|: median={statistics.median(dw):.3f} max={max(dw):.3f} m/s")
    sustained = vw and statistics.median(vw) > 0.7
    print(f"  -> {'SUSTAINED at the cap (command faithful)' if sustained else 'NOT sustained (decay?)'}")

    print("\n=== 2. raise max speed live with 'i' (nav mode), then forward ===")
    for _ in range(8):
        os.write(master, b"i"); time.sleep(0.35)   # edge-triggered; >HOLD_TIMEOUT apart
    time.sleep(0.3)
    t0 = time.monotonic(); hold(b"\x1b[A", 3.0); t1 = time.monotonic()
    time.sleep(0.3)
    vw = [abs(v) for v in vx_window(t0 + 1.2, t1)]
    dw = fake_drive_mag(t0 + 1.2, t1 + 0.3)
    print(f"  after 8×'i': /mabel_cmd |vx| median={statistics.median(vw):.3f}" if vw else "  no vx")
    print(f"  mock Teensy |drive| max={max(dw):.3f} m/s" if dw else "  no drive")
    boosted = vw and statistics.median(vw) > 0.9
    print(f"  -> {'I/K raises the speed in nav mode (works)' if boosted else 'I/K did not raise speed'}")

    print("\n=== 3. direct cmd_vel = 2.0 m/s — bridge must pass it through uncapped ===")
    t0 = time.monotonic()
    end = t0 + 2.5
    while time.monotonic() < end:
        pub_cmd_vel.publish(Twist(linear=Vector3(x=2.0)))
        time.sleep(0.05)
    time.sleep(0.3)
    dw = fake_drive_mag(t0, end + 0.3)
    print(f"  mock Teensy |drive| for cmd_vel 2.0: max={max(dw):.3f} m/s" if dw else "  no drive")
    print(f"  -> {'bridge passes commanded velocity 1:1 (no shrink/cap)' if dw and max(dw) > 1.8 else 'velocity was shrunk!'}")

    mon.destroy_node()
    rclpy.shutdown()


def _drain(fd):
    while True:
        try:
            if not os.read(fd, 4096):
                break
        except OSError:
            break


if __name__ == "__main__":
    if subprocess.run(["pgrep", "-f", "[h]ardware_bridge.py"],
                      capture_output=True).returncode == 0:
        print("✗ ABORT: a hardware bridge is already running; stop it first.")
        sys.exit(1)
    try:
        main()
    finally:
        cleanup()
