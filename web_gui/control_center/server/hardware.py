"""
Hard e-stop: beyond the bridge's instant zero-drive + torque-off, ENGAGING
kills the hardware bring-up (`ros2 launch mabel_bringup hardware.launch.py`)
so every subsystem closes its serial/CAN link and the motors are truly
released. RE-ENGAGING relaunches the exact same command with the exact same
environment (captured at kill time) — bring-up takes a while, so the GUI
polls /api/hw/status until joint telemetry flows again.
"""
import os
import signal
import subprocess
import threading
import time

_SAVED = {"cmdline": None, "env": None, "cwd": None}
_LAUNCH_MARK = "hardware.launch.py"


def _ros_broadcast(estop, torque):
    """FAST actuation kill/clear over the hardware bridge's OWN ROS contract:
    /mabel/hw/base/estop (latched, priority gate: estop > pad > cmd_vel) and
    the per-subsystem torque_enable topics. Lands in tens of milliseconds —
    no process needs to die for the robot to go limp."""
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import (QoSProfile, ReliabilityPolicy, DurabilityPolicy)
    from std_msgs.msg import Bool

    node = Node("mabel_cc_estop")
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
    pubs = [node.create_publisher(Bool, "/mabel/hw/base/estop", latched)]
    tpubs = [node.create_publisher(Bool, t, 1) for t in
             ("/arms/torque_enable", "/left_hand/torque_enable",
              "/right_hand/torque_enable", "/body/torque_enable")]
    try:
        t0 = time.time()
        while time.time() - t0 < 0.6:              # let discovery match
            ex.spin_once(timeout_sec=0.1)
        pubs[0].publish(Bool(data=bool(estop)))
        if torque is not None:
            for p in tpubs:
                p.publish(Bool(data=bool(torque)))
        t0 = time.time()
        while time.time() - t0 < 0.4:              # flush
            ex.spin_once(timeout_sec=0.1)
    finally:
        ex.remove_node(node)
        node.destroy_node()


def _find_pid():
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cmd = f.read().split(b"\0")
        except OSError:
            continue
        if any(_LAUNCH_MARK.encode() in part for part in cmd):
            return int(pid), [c.decode() for c in cmd if c]
    return None, None


def engage():
    """E-STOP, layered by speed:
      1. INSTANT (~50 ms): latched base e-stop + torque_enable=False for arms,
         both hands, and the neck (/body) over the bridge's ROS topics → limp.
      2. BACKGROUND: SIGINT the hardware bring-up so every subsystem closes
         its serial/CAN link (full release; does not block the stop)."""
    fast = {"ros_kill": False}
    try:
        _ros_broadcast(estop=True, torque=False)
        fast["ros_kill"] = True
    except Exception as e:
        fast["ros_kill_error"] = str(e)

    pid, cmd = _find_pid()
    if pid is None:
        return {**fast, "found": False, "note": "hardware.launch not running"}
    try:
        with open(f"/proc/{pid}/environ", "rb") as f:
            env = {}
            for kv in f.read().split(b"\0"):
                if b"=" in kv:
                    k, v = kv.split(b"=", 1)
                    env[k.decode()] = v.decode()
        _SAVED.update(cmdline=cmd, env=env,
                      cwd=os.readlink(f"/proc/{pid}/cwd"))
    except OSError:
        pass

    def _reap():
        os.kill(pid, signal.SIGINT)
        for _ in range(60):
            if not os.path.exists(f"/proc/{pid}"):
                return
            time.sleep(0.25)
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    threading.Thread(target=_reap, daemon=True).start()
    return {**fast, "found": True, "killing": pid,
            "restartable": bool(_SAVED["cmdline"])}


def release(telemetry_state=None):
    """RE-ENGAGE: relaunch the hardware bring-up captured at engage time, and
    once joint telemetry flows again, clear the latched base e-stop so the
    robot is actually BACK (torque re-energize stays an explicit operator
    action — the hands hold-in-place when it happens)."""
    pid, _ = _find_pid()
    started = False
    if pid is None:
        if not _SAVED["cmdline"]:
            return {"started": False,
                    "note": "no captured launch (engage was never issued here)"}
        log = open("/tmp/mabel_cc_hw_bringup.log", "ab")
        p = subprocess.Popen(_SAVED["cmdline"], cwd=_SAVED["cwd"] or "/",
                             env=_SAVED["env"], stdout=log, stderr=log,
                             start_new_session=True)
        pid, started = p.pid, True

    def _clear_when_up():
        t0 = time.time()
        while time.time() - t0 < 90:
            fresh = (telemetry_state is not None and telemetry_state.stamp
                     and time.time() - telemetry_state.stamp < 2.0)
            if fresh:
                try:
                    _ros_broadcast(estop=False, torque=None)
                except Exception:
                    pass
                return
            time.sleep(1.0)
    threading.Thread(target=_clear_when_up, daemon=True).start()
    return {"started": started, "pid": pid}


def status(telemetry_state):
    """up = joint telemetry flowing again (fresh /joint_states samples)."""
    pid, _ = _find_pid()
    fresh = (time.time() - telemetry_state.stamp) < 2.0 if telemetry_state.stamp else False
    return {"launch_running": pid is not None, "telemetry_fresh": fresh,
            "up": pid is not None and fresh}
