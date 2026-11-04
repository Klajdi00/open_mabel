"""bridge_link.py - the control center's ONLY link to the robot: the hardware bridge.

A single rclpy node, spun on a background thread, that speaks the bridge's generic DDS
contract (see contract.py). The GUI publishes commands through it and reads a
thread-safe telemetry snapshot from it. It NEVER touches hardware - exactly like every
other bridge client, it goes through DDS to the one hardware owner.

Threading model: rclpy spins on its own thread and updates ``self._state`` under a lock;
Tk reads ``snapshot()`` from a periodic ``after`` poll on the main thread. Publishers are
created once and are safe to call from the Tk thread. This keeps all Tk calls on the Tk
thread (Tk is not thread-safe) and all ROS callbacks on the ROS thread.
"""

from __future__ import annotations

import fcntl
import os
import signal
import subprocess
import threading
import time
from copy import deepcopy

import contract

# Where the bridge lives + its singleton lock (must match hardware_bridge.py).
_GUI_DIR = os.path.dirname(os.path.abspath(__file__))
_BRIDGE_DIR = os.path.dirname(_GUI_DIR)
_LOCK_PATH = os.environ.get("MABEL_HW_LOCK", "/tmp/mabel_hardware_bridge.lock")

# command topic -> (state key, hand side) so an 'online' check can be resolved to that
# subsystem's live-telemetry freshness (see BridgeLink.subsystem_online).
_TOPIC_TO_KEY = {
    contract.BASE["cmd_vel"]: ("base", None),
    contract.ARMS["command"]: ("arms", None),
    contract.BODY["command"]: ("body", None),
    contract.LIFT["command"]: ("lift", None),
}
for _s in contract.HANDS["sides"]:
    _TOPIC_TO_KEY[contract.HANDS["command"].format(side=_s)] = ("hands", _s)

# /<side>_hand/joint_states now carries URDF-native names (the bridge translates the
# ORCA thumb at the source — hands_subsystem._THUMB_TO_URDF). The GUI's per-joint table
# is keyed by the ORCA firmware names, so map the URDF thumb names back on ingestion.
_URDF_TO_ORCA_THUMB = {"thumb_mcp": "thumb_cmc", "thumb_dip": "thumb_mp",
                       "thumb_pip": "thumb_ip"}


class BridgeLink:
    def __init__(self, domain: int | None = None):
        self._domain = domain
        self._node = None
        self._exec = None
        self._thread = None
        self._lock = threading.Lock()
        self._pubs = {}
        self._running = False
        self._state = self._blank_state()
        self._bridge_proc = None       # set if WE started the bridge (we then own it)
        self._bridge_log = None

    # ── state shape ──────────────────────────────────────────────────────────
    @staticmethod
    def _blank_state():
        return {
            "base": {"stamp": 0.0, "joints": {}},               # name -> (pos, vel)
            "arms": {"stamp": 0.0, "joints": {}, "torque": None},   # name -> (pos, vel, eff)
            "body": {"stamp": 0.0, "joints": {}, "torque": None},
            "lift": {"stamp": 0.0, "pos": None, "vel": None,
                     "min": None, "max": None, "calibrated": None},
            "hands": {s: {"stamp": 0.0, "joints": {}, "torque": None,
                          "temps": {}, "volts": {}}
                      for s in contract.HANDS["sides"]},
            "sensors": {"lidar": {"stamp": 0.0, "hz": 0.0, "n": 0}},
        }

    # ── lifecycle ──────────────────────────────────────────────────────────--
    def start(self):
        if self._running:
            return
        import rclpy
        from rclpy.executors import SingleThreadedExecutor

        if not rclpy.ok():
            rclpy.init()
        from rclpy.node import Node
        self._node = Node("mabel_control_center")
        self._build_io()
        self._exec = SingleThreadedExecutor()
        self._exec.add_node(self._node)
        self._running = True
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def _spin(self):
        try:
            self._exec.spin()
        except Exception:
            pass

    def stop(self):
        self._running = False
        # If WE started the bridge, shut it down (SIGINT = clean: disables torque,
        # releases the serial bus). A bridge we merely connected to is left running.
        if self._bridge_proc is not None:
            try:
                self._bridge_proc.send_signal(signal.SIGINT)
                self._bridge_proc.wait(timeout=5)
            except Exception:
                try: self._bridge_proc.kill()
                except Exception: pass
            self._bridge_proc = None
        try:
            if self._exec is not None:
                self._exec.shutdown()
            if self._node is not None:
                self._node.destroy_node()
        except Exception:
            pass
        try:
            import rclpy
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass

    # ── bridge ownership ─────────────────────────────────────────────────────
    @staticmethod
    def bridge_running() -> bool:
        """True if a hardware bridge already owns the hardware (its singleton flock is
        held). Lets the GUI reuse a bridge started by run_bridge.sh / systemd / real_run
        instead of starting a second one (which would refuse on the same lock)."""
        try:
            fd = os.open(_LOCK_PATH, os.O_RDWR | os.O_CREAT, 0o644)
        except OSError:
            return False
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(fd, fcntl.LOCK_UN)      # acquired => nobody owns it
            return False
        except OSError:
            return True                          # locked => a bridge is running
        finally:
            os.close(fd)

    def ensure_bridge(self) -> str:
        """Make sure a bridge is up so the GUI actually detects hardware.

        Returns: 'existing' (one was already running - reuse it), 'started' (we launched
        one and now own it), or 'skipped' (auto-start disabled). Set
        MABEL_GUI_NO_AUTOBRIDGE=1 to never auto-start (pure-client mode). Subsystems and
        ports come from the usual MABEL_HW_* env (we only fill in sensible defaults so a
        bare ./gui.sh detects the connected hand): hands ON + free-drive, right hand
        pinned by USB serial, base ON (idles if no Teensy)."""
        if self.bridge_running():
            return "existing"
        if os.environ.get("MABEL_GUI_NO_AUTOBRIDGE") == "1":
            return "skipped"
        run = os.path.join(_BRIDGE_DIR, "run_bridge.sh")
        if not os.access(run, os.X_OK):
            return "skipped"
        env = dict(os.environ)
        env.setdefault("MABEL_HW_HANDS", "true")
        env.setdefault("MABEL_HW_HANDS_AUTO_TORQUE", "false")   # never command on launch
        env.setdefault("MABEL_HW_HANDS_LEFT_PORT", "none")
        env.setdefault("MABEL_HW_HANDS_RIGHT_PORT",
                       "/dev/serial/by-id/usb-1a86_USB_Single_Serial_0000000002-if00")
        # HANDS-ONLY by default. The base/arms/body/lift are NOT enabled here: with no
        # Teensy/CAN on a hand bench the base subsystem otherwise churns reconnects to an
        # unreachable IP forever (flooding the log and destabilising the process - that
        # was the "lost connection" cause). To drive the whole robot, bring the bridge up
        # yourself (run_bridge.sh / real_run.sh with those subsystems on) and the GUI will
        # detect+reuse it. Each is still overridable via its MABEL_HW_* env var.
        for _k in ("MABEL_HW_BASE", "MABEL_HW_ARMS", "MABEL_HW_BODY",
                   "MABEL_HW_LIFT", "MABEL_HW_CAMERAS"):
            env.setdefault(_k, "false")
        self._bridge_log = open("/tmp/mabel_gui_bridge.log", "w")
        self._bridge_proc = subprocess.Popen(
            [run], cwd=_BRIDGE_DIR, env=env,
            stdout=self._bridge_log, stderr=subprocess.STDOUT)
        return "started"

    def release_bridge(self) -> bool:
        """Free the serial bus so the Calibration tab can open the Feetech port directly
        (it is exclusive). Stops the bridge WE started if we own it; otherwise stops the
        running bridge by the PID in its singleton lock file (hardware_bridge.py writes
        its pid there). Then waits until the lock actually clears. Returns True if a
        bridge was running and is now stopped."""
        stopped = False
        if self._bridge_proc is not None:                  # one we own
            try:
                self._bridge_proc.send_signal(signal.SIGINT)
                self._bridge_proc.wait(timeout=6)
            except Exception:
                try: self._bridge_proc.kill()
                except Exception: pass
            self._bridge_proc = None
            stopped = True
        elif self.bridge_running():                        # a foreign bridge holds the bus
            try:
                pid = int(open(_LOCK_PATH).read().strip())
                os.kill(pid, signal.SIGINT)
                stopped = True
            except Exception:
                return False
        if stopped:
            for _ in range(40):                            # wait up to ~4s for it to exit
                if not self.bridge_running():
                    break
                time.sleep(0.1)
            time.sleep(0.5)                                # OS releases the tty
        return stopped

    @property
    def connected(self) -> bool:
        return self._running and self._node is not None

    # ── publishers / subscribers ───────────────────────────────────────────--
    def _build_io(self):
        from geometry_msgs.msg import Twist
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Bool, Float64, Float64MultiArray

        n = self._node
        P = self._pubs
        # Base
        P["base_cmd"] = n.create_publisher(Twist, contract.BASE["cmd_vel"], 10)
        P["base_estop"] = n.create_publisher(Bool, contract.BASE["estop"], 10)
        n.create_subscription(JointState, contract.BASE["joint_states"],
                              lambda m: self._on_js("base", m), 10)
        # Arms / Body (+ torque enable/state)
        P["arms_cmd"] = n.create_publisher(JointState, contract.ARMS["command"], 10)
        P["arms_torque"] = n.create_publisher(Bool, contract.ARMS["torque_enable"], 10)
        n.create_subscription(JointState, contract.ARMS["joint_states"],
                              lambda m: self._on_js("arms", m), 10)
        n.create_subscription(Bool, contract.ARMS["torque_state"],
                              lambda m: self._set_field("arms", "torque", bool(m.data)), 10)
        P["body_cmd"] = n.create_publisher(JointState, contract.BODY["command"], 10)
        P["body_torque"] = n.create_publisher(Bool, contract.BODY["torque_enable"], 10)
        n.create_subscription(JointState, contract.BODY["joint_states"],
                              lambda m: self._on_js("body", m), 10)
        n.create_subscription(Bool, contract.BODY["torque_state"],
                              lambda m: self._set_field("body", "torque", bool(m.data)), 10)
        # Lift
        P["lift_cmd"] = n.create_publisher(Float64, contract.LIFT["command"], 10)
        P["lift_calibrate"] = n.create_publisher(Bool, contract.LIFT["calibrate"], 10)
        n.create_subscription(JointState, contract.LIFT["joint_states"],
                              self._on_lift, 10)
        # /lift/limits is latched (TRANSIENT_LOCAL) — match it so the panel learns the
        # measured min/max/calibrated even when it joins after the last calibrate.
        from rclpy.qos import DurabilityPolicy, QoSProfile
        from std_msgs.msg import String
        _latched = QoSProfile(depth=1)
        _latched.durability = DurabilityPolicy.TRANSIENT_LOCAL
        n.create_subscription(String, contract.LIFT["limits"], self._on_lift_limits,
                              _latched)
        # LiDAR liveness (RPLIDAR /scan, sensor-data QoS = BEST_EFFORT)
        try:
            from sensor_msgs.msg import LaserScan
            from rclpy.qos import qos_profile_sensor_data
            n.create_subscription(LaserScan, contract.LIDAR["scan"], self._on_scan,
                                  qos_profile_sensor_data)
        except Exception:
            pass
        # Hands (per side)
        for s in contract.HANDS["sides"]:
            P[f"hand_cmd_{s}"] = n.create_publisher(
                JointState, contract.HANDS["command"].format(side=s), 10)
            P[f"hand_torque_{s}"] = n.create_publisher(
                Bool, contract.HANDS["torque_enable"].format(side=s), 10)
            n.create_subscription(JointState, contract.HANDS["joint_states"].format(side=s),
                                  lambda m, s=s: self._on_hand_js(s, m), 10)
            n.create_subscription(Bool, contract.HANDS["torque_state"].format(side=s),
                                  lambda m, s=s: self._set_hand(s, "torque", bool(m.data)), 10)
            n.create_subscription(Float64MultiArray, contract.HANDS["temperatures"].format(side=s),
                                  lambda m, s=s: self._on_hand_marray(s, "temps", m), 10)
            n.create_subscription(Float64MultiArray, contract.HANDS["voltages"].format(side=s),
                                  lambda m, s=s: self._on_hand_marray(s, "volts", m), 10)

    # ── subscription callbacks (ROS thread) ─────────────────────────────────--
    def _on_js(self, key, msg):
        with self._lock:
            d = self._state[key]
            d["stamp"] = time.monotonic()
            for i, name in enumerate(msg.name):
                pos = msg.position[i] if i < len(msg.position) else 0.0
                vel = msg.velocity[i] if i < len(msg.velocity) else 0.0
                eff = msg.effort[i] if i < len(msg.effort) else 0.0
                d["joints"][name] = (pos, vel, eff)

    def _on_scan(self, msg):
        now = time.monotonic()
        with self._lock:
            d = self._state["sensors"]["lidar"]
            dt = now - d["stamp"] if d["stamp"] else 0.0
            if 0.0 < dt < 5.0:
                inst = 1.0 / dt
                d["hz"] = inst if d["hz"] == 0.0 else 0.8 * d["hz"] + 0.2 * inst
            d["stamp"] = now
            d["n"] = len(msg.ranges)

    def _on_lift(self, msg):
        with self._lock:
            d = self._state["lift"]
            d["stamp"] = time.monotonic()
            if msg.position:
                d["pos"] = msg.position[0]
            if msg.velocity:
                d["vel"] = msg.velocity[0]

    def _on_lift_limits(self, msg):
        import json
        try:
            lim = json.loads(msg.data)
        except (ValueError, TypeError):
            return
        with self._lock:
            d = self._state["lift"]
            d["min"] = lim.get("min")
            d["max"] = lim.get("max")
            d["calibrated"] = lim.get("calibrated")

    def _on_hand_js(self, side, msg):
        with self._lock:
            d = self._state["hands"][side]
            d["stamp"] = time.monotonic()
            for i, name in enumerate(msg.name):
                pos = msg.position[i] if i < len(msg.position) else 0.0
                vel = msg.velocity[i] if i < len(msg.velocity) else 0.0
                eff = msg.effort[i] if i < len(msg.effort) else 0.0
                bare = _bare(name, side)
                d["joints"][_URDF_TO_ORCA_THUMB.get(bare, bare)] = (pos, vel, eff)

    def _on_hand_marray(self, side, field, msg):
        # The bridge packs joint names into the dim label as CSV (see hands_subsystem).
        names = []
        try:
            if msg.layout.dim:
                names = [n for n in msg.layout.dim[0].label.split(",") if n]
        except Exception:
            pass
        with self._lock:
            d = self._state["hands"][side][field]
            for i, v in enumerate(msg.data):
                key = _bare(names[i], side) if i < len(names) else str(i)
                d[key] = v

    def _set_hand(self, side, field, value):
        with self._lock:
            self._state["hands"][side][field] = value

    def _set_field(self, key, field, value):
        with self._lock:
            self._state[key][field] = value

    # ── telemetry read (Tk thread) ──────────────────────────────────────────--
    def snapshot(self) -> dict:
        with self._lock:
            return deepcopy(self._state)

    def lidar_status(self, max_age: float = 2.0):
        """(online, hz, n_points) for the RPLIDAR /scan."""
        with self._lock:
            d = dict(self._state["sensors"]["lidar"])
        online = d["stamp"] > 0.0 and (time.monotonic() - d["stamp"]) < max_age
        return online, (d["hz"] if online else 0.0), d["n"]

    def live(self, key: str, side: str | None = None, max_age: float = 2.0) -> bool:
        """True only if real telemetry for this subsystem ARRIVED recently.

        This is the honest 'hardware connected' test. The bridge subscribes to every
        enabled subsystem's command topic even when the device is absent (e.g. the left
        hand shares the right hand's node; the base subscribes with no Teensy), and it
        ONLY publishes joint_states when the hardware actually responds - so freshness of
        the feedback, not the presence of a command subscriber, is what tells us a device
        is really there and streaming."""
        with self._lock:
            st = (self._state["hands"][side]["stamp"] if side is not None
                  else self._state.get(key, {}).get("stamp", 0.0))
        return st > 0.0 and (time.monotonic() - st) < max_age

    def live_hand_side(self):
        """The hand side currently streaming telemetry (so Calibration can default to the
        actually-connected hand), or None."""
        for s in contract.HANDS["sides"]:
            if self.live("hands", s):
                return s
        return None

    def subsystem_online(self, cmd_pub_key: str, cmd_topic: str) -> bool:
        """'Online' == the bridge is publishing LIVE telemetry for this subsystem (the
        device is connected and streaming), resolved from its command topic. NOT merely
        'the bridge advertises the topic' - that is true even with no hardware."""
        key_side = _TOPIC_TO_KEY.get(cmd_topic)
        if key_side is None:
            return False
        return self.live(key_side[0], key_side[1])

    # ── commands (Tk thread -> DDS) ──────────────────────────────────────────--
    def base_twist(self, vx, vy, wz):
        """Publish a base twist, TIP-GATED.

        This goes straight to the bridge's base intake -- past /mabel_cmd, past the
        gateway, past controller/ -- so the GUI is the last place the tip model can
        run. It used to publish the raw slider values: at full lift the sliders could
        ask for ~2x the tip-safe speed with nothing to stop them.

        One-shot target, not a ramp loop, so this applies the two STEADY parts of
        safe_base_step (tip_over.py: the per-axis ceilings, then the combined-target
        ZMP projection) rather than the whole ramping step. Accel is bounded by how
        fast a human moves a slider.

        Lift unknown -> assume FULLY RAISED. The one direction that is never safe to
        guess is "it's probably down".
        """
        from geometry_msgs.msg import Twist
        vx, vy, wz = self._tip_gate(float(vx), float(vy), float(wz))
        m = Twist()
        m.linear.x, m.linear.y, m.angular.z = vx, vy, wz
        self._pubs["base_cmd"].publish(m)

    def _tip_gate(self, tvx, tvy, tw):
        """Clamp a base target into the tip-safe envelope at the current lift height.

        Returns the target unchanged (and warns once) if controller/ is unavailable --
        this is a bench GUI, and failing to open is worse than failing to clamp; but
        it must never be silent about it.
        """
        try:
            from mabel.motion_model import (LIFT_RANGE, max_speed_dir,
                                            tip_speed_scale)
        except Exception as e:                      # pragma: no cover
            if not getattr(self, "_tip_warned", False):
                self._tip_warned = True
                print(f"[bridge_link] WARNING: tip model unavailable ({e}); base "
                      f"sliders are UNGATED — drive with the lift DOWN.")
            return tvx, tvy, tw

        pos = self._state.get("lift", {}).get("pos")
        z = float(pos) if pos is not None else LIFT_RANGE[1]   # unknown -> worst case

        # Per-axis ceilings. NOTE the frame: forward is -x in base_link, so v_fwd
        # bounds the NEGATIVE side and v_back the positive one (same as tip_over).
        v_fwd, v_back, v_lat = max_speed_dir(z)
        tvx = max(-v_fwd, min(v_back, tvx))
        tvy = max(-v_lat, min(v_lat, tvy))
        # Then project the COMBINED target into the steady ZMP envelope: each axis
        # can be legal alone yet tip the robot together.
        s = tip_speed_scale(tvx, tvy, tw, z)
        return tvx * s, tvy * s, tw * s

    def base_estop(self, on: bool):
        from std_msgs.msg import Bool
        self._pubs["base_estop"].publish(Bool(data=bool(on)))

    def _joint_cmd(self, pub_key, targets: dict):
        from sensor_msgs.msg import JointState
        m = JointState()
        m.header.stamp = self._node.get_clock().now().to_msg()
        m.name = list(targets.keys())
        m.position = [float(v) for v in targets.values()]
        self._pubs[pub_key].publish(m)

    def arms_command(self, targets: dict):
        self._joint_cmd("arms_cmd", targets)

    def body_command(self, targets: dict):
        self._joint_cmd("body_cmd", targets)

    def joint_torque(self, key, on: bool):
        """Engage / free-drive the arms or body (key='arms'|'body')."""
        from std_msgs.msg import Bool
        pub = self._pubs.get(f"{key}_torque")
        if pub is not None:
            pub.publish(Bool(data=bool(on)))

    def hand_command(self, side, targets: dict):
        self._joint_cmd(f"hand_cmd_{side}", targets)

    def hand_torque(self, side, on: bool):
        from std_msgs.msg import Bool
        self._pubs[f"hand_torque_{side}"].publish(Bool(data=bool(on)))

    def lift_command(self, metres: float):
        from std_msgs.msg import Float64
        self._pubs["lift_cmd"].publish(Float64(data=float(metres)))

    def lift_calibrate(self):
        """Trigger the on-Pico stall-seek (floor+ceiling); the bridge measures and
        persists the min/max travel envelope and re-publishes /lift/limits."""
        from std_msgs.msg import Bool
        self._pubs["lift_calibrate"].publish(Bool(data=True))


def _bare(name: str, side: str) -> str:
    p = f"{side}_"
    return name[len(p):] if name.startswith(p) else name
