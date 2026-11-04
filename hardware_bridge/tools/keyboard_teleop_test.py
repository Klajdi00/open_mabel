#!/usr/bin/env python3
"""keyboard_teleop_test.py — prove ROS keyboard teleop drives the hardware bridge.

End-to-end, NO real hardware: a mock Teensy on 127.0.0.1 stands in for the base. We
launch the REAL `mabel_teleop keyboard_teleop` node attached to a pseudo-terminal and
type real keystrokes into it, then assert the motion reaches the mock Teensy:

    keystroke -> keyboard_teleop -> /mabel_cmd -> base_bridge_client -> /mabel/hw/base/cmd_vel
              -> hardware bridge (IK+calib) -> mock Teensy

Checks: forward (Up) drives the wheels; rotate (A) drives them; SPACE e-stop zeros
them. We watch BOTH ends — /mabel_cmd (teleop output) and the mock Teensy (hardware
input) — so a break anywhere in the chain is caught.
"""

import os
import pty
import re
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
HWB = os.path.dirname(HERE)
TEENSY = "127.0.0.1"
FAKE_LOG = "/tmp/kt_fake.log"
BR_LOG = "/tmp/kt_bridge.log"
CL_LOG = "/tmp/kt_client.log"
procs = []
PASS = 0
FAIL = 0


def ok(m):
    global PASS
    PASS += 1
    print(f"  ✓ {m}")


def no(m):
    global FAIL
    FAIL += 1
    print(f"  ✗ {m}")


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
    # PID-only: run_bridge.sh execs python at the Popen PID, so killing procs above kills
    # the bridge. We deliberately do NOT pkill by name (that would kill a REAL bridge too).


def fake_drives_between(t0, t1):
    """|drive0| values the mock logged (RXSTEP <mono> drive0=..) within [t0,t1]."""
    out = []
    for line in open(FAKE_LOG):
        m = re.search(r"RXSTEP ([\d.]+) drive0=(-?[\d.]+)", line)
        if m and t0 <= float(m.group(1)) <= t1:
            out.append(abs(float(m.group(2))))
    return out


def main():
    os.environ.setdefault("ROS_DOMAIN_ID", "46")
    os.environ["MABEL_HW_LOCK"] = "/tmp/kt_hwb.lock"
    env = dict(os.environ, MABEL_HW_BASE="true", MABEL_HW_BASE_TEENSY_IP=TEENSY,
               MABEL_HW_ARMS="false", MABEL_HW_BODY="false", MABEL_HW_HANDS="false",
               MABEL_HW_LIFT="false", MABEL_HW_CAMERAS="false")

    print("starting mock Teensy + bridge + base_bridge_client (127.0.0.1)...")
    spawn([sys.executable, os.path.join(HERE, "fake_teensy.py"),
           "--host", TEENSY, "--port", "5800"], FAKE_LOG)
    spawn(["bash", os.path.join(HWB, "run_bridge.sh")], BR_LOG, cwd=HWB, env=env)
    spawn(["ros2", "run", "mabel_base_hardware", "swerve_node",
           "--ros-args", "-r", "__node:=base_bridge_client"], CL_LOG, env=env)
    time.sleep(6)

    # ---- watch /mabel_cmd (the teleop's OUTPUT) in a background rclpy thread ----
    import rclpy
    from mabel_interfaces.msg import MabelSwerveCmd
    rclpy.init()
    mon = rclpy.create_node("kt_monitor")
    seen = {"min_vx": 0.0, "max_omega": 0.0, "estop": False, "n": 0}

    def on_cmd(msg):
        seen["n"] += 1
        seen["min_vx"] = min(seen["min_vx"], msg.vx)
        seen["max_omega"] = max(seen["max_omega"], abs(msg.omega))
        if msg.hardware_state == 1:
            seen["estop"] = True
    mon.create_subscription(MabelSwerveCmd, "/mabel_cmd", on_cmd, 10)
    threading.Thread(target=lambda: rclpy.spin(mon), daemon=True).start()

    # ---- launch the REAL keyboard_teleop on a PTY so it uses the termios backend ----
    master, slave = pty.openpty()
    kt = subprocess.Popen(
        ["ros2", "run", "mabel_teleop", "keyboard_teleop",
         "--ros-args", "-p", "model:=/nonexistent", "-p", "rate:=20.0"],
        stdin=slave, stdout=slave, stderr=slave, env=env, preexec_fn=os.setsid)
    procs.append(kt)
    os.close(slave)
    # drain the teleop's dashboard so its stdout never blocks on a full PTY buffer
    threading.Thread(target=lambda: _drain(master), daemon=True).start()
    time.sleep(5)
    if not os.path.exists("/tmp/kt_started"):
        pass

    def hold(keys, dur):
        end = time.time() + dur
        while time.time() < end:
            os.write(master, keys)
            time.sleep(0.08)        # < HOLD_TIMEOUT (0.25 s) so it reads as "held"

    def check(cond, yes, nope):
        ok(yes) if cond else no(nope)

    print("== A. FORWARD (Up arrow held) =="); seen.update(min_vx=0.0)
    t0 = time.monotonic(); hold(b"\x1b[A", 1.8); t1 = time.monotonic()
    time.sleep(0.3)
    fwd = fake_drives_between(t0, t1 + 0.3)
    check(seen["min_vx"] < -0.2,
          f"teleop published forward (/mabel_cmd vx min={seen['min_vx']:.2f})",
          f"teleop did NOT publish forward (vx min={seen['min_vx']:.2f})")
    check(bool(fwd) and max(fwd) > 0.3,
          f"mock Teensy driven forward (|drive| up to {max(fwd):.2f})" if fwd else "x",
          f"mock saw no forward drive ({fwd[-3:] if fwd else 'none'})")

    time.sleep(1.2)   # release: keys decay, ramp back to 0

    print("== B. ROTATE (A held) ==")
    t0 = time.monotonic(); hold(b"a", 1.5); t1 = time.monotonic()
    time.sleep(0.3)
    rot = fake_drives_between(t0, t1 + 0.3)
    check(seen["max_omega"] > 0.2,
          f"teleop published rotation (/mabel_cmd |omega| max={seen['max_omega']:.2f})",
          "teleop did NOT publish rotation")
    check(bool(rot) and max(rot) > 0.1,
          f"mock Teensy driven in rotation (|drive| up to {max(rot):.2f})" if rot else "x",
          "mock saw no rotation drive")

    time.sleep(1.2)

    print("== C. E-STOP (SPACE) while commanding forward ==")
    os.write(master, b" ")           # toggle e-stop ON
    time.sleep(0.4)
    t0 = time.monotonic(); hold(b"\x1b[A", 1.2); t1 = time.monotonic()  # try to drive
    time.sleep(0.3)
    est = fake_drives_between(t0, t1 + 0.3)
    check(seen["estop"], "teleop published e-stop (hardware_state=1)",
          "no e-stop seen on /mabel_cmd")
    check((not est) or max(est) < 0.05,
          "mock Teensy drive STAYED ZERO under e-stop (safe)",
          f"mock moved under e-stop! (|drive| {max(est) if est else 0:.2f}) — UNSAFE")

    mon.destroy_node()
    rclpy.shutdown()
    print(f"\n================  KEYBOARD TELEOP: {PASS} passed, {FAIL} failed  ================")
    print("ALL GREEN — keyboard teleop maps through the bridge to (mock) hardware."
          if FAIL == 0 else "FAILURES above.")
    return FAIL


def _drain(fd):
    while True:
        try:
            if not os.read(fd, 4096):
                break
        except OSError:
            break


if __name__ == "__main__":
    # GUARD: never run if a real bridge is up (we'd disturb the real robot).
    if subprocess.run(["pgrep", "-f", "[h]ardware_bridge.py"],
                      capture_output=True).returncode == 0:
        print("✗ ABORT: a hardware bridge is already running; stop it first "
              "(sudo systemctl stop mabel-hardware-bridge).")
        sys.exit(1)
    rc = 1
    try:
        rc = main()
    finally:
        cleanup()
    sys.exit(rc)
