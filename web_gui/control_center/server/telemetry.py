"""
Telemetry hub: one background source (ROS tap or mock plant) feeding a shared
latest-state snapshot that the WS endpoint fans out at TELEMETRY_HZ.

The bridge's 9090 `robot_state` mirror only carries joint POSITIONS; this hub
adds velocity + effort (from /joint_states), per-hand actuator temperatures
(/​<side>_hand/temperatures), torque + e-stop flags, and last-commanded targets
(tapped off the command topics) — the data the 59-DOF panel grades against.
"""
import math
import threading
import time

from . import config as CFG
from .joints import JOINT_NAMES, JOINT_GROUP


class TelemetryState:
    """Thread-safe latest-value store keyed by canonical joint name."""

    def __init__(self):
        self._lock = threading.Lock()
        self.pos = {}
        self.vel = {}
        self.eff = {}
        self.target = {}
        self.temp = {}
        self.estop = False
        self.torque = {"arms": None, "handL": None, "handR": None}
        self.scan = None                 # 360-bin LiDAR ranges (m), body frame
        self.caminfo = {}                # ns -> {w,h,fx,fy,cx,cy,d[5]} (live)
        self.source = "none"
        self.stamp = 0.0

    def update_joints(self, names, pos=None, vel=None, eff=None):
        with self._lock:
            for i, n in enumerate(names):
                if n not in JOINT_GROUP:
                    continue
                if pos is not None and i < len(pos): self.pos[n] = float(pos[i])
                if vel is not None and i < len(vel): self.vel[n] = float(vel[i])
                if eff is not None and i < len(eff): self.eff[n] = float(eff[i])
            self.stamp = time.time()

    def update_targets(self, names, positions):
        with self._lock:
            for n, p in zip(names, positions):
                if n in JOINT_GROUP:
                    self.target[n] = float(p)

    def update_temps(self, names, temps):
        with self._lock:
            for n, t in zip(names, temps):
                if n in JOINT_GROUP:
                    self.temp[n] = float(t)

    def snapshot(self):
        """Parallel arrays over the canonical 60-joint order (null = unknown)."""
        with self._lock:
            g = lambda d: [d.get(n) for n in JOINT_NAMES]
            return {
                "t": self.stamp, "source": self.source, "estop": self.estop,
                "torque": dict(self.torque),
                "p": g(self.pos), "v": g(self.vel), "e": g(self.eff),
                "tgt": g(self.target), "temp": g(self.temp),
                "scan": self.scan,
            }


class MockSource:
    """Synthetic 59-DOF plant so the whole GUI runs (and is testable) with no
    robot: smooth per-joint sines, targets leading measurements (a few joints
    lag on purpose to exercise the yellow/red grading), warm hand motors."""

    def __init__(self, state, hz=50.0):
        self.state = state
        self.hz = hz
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.state.source = "mock"
        self.state.torque = {"arms": True, "handL": True, "handR": True}
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _run(self):
        t0 = time.time()
        lag_joints = {"left_arm_4", "right_arm_2", "right_index_pip"}
        while not self._stop.is_set():
            t = time.time() - t0
            names, pos, vel, eff, tgt = [], [], [], [], []
            for i, n in enumerate(JOINT_NAMES):
                w = 0.3 + 0.05 * (i % 7)
                a = 0.02 if JOINT_GROUP[n] == "base" and n.endswith("_drive") else 0.35
                target = a * math.sin(w * t + i)
                lag = 0.28 if n in lag_joints else 0.02
                names.append(n)
                tgt.append(target)
                pos.append(target - lag * math.cos(w * t + i))
                vel.append(a * w * math.cos(w * t + i))
                eff.append(0.4 * math.sin(w * t * 0.5 + i))
            self.state.update_joints(names, pos, vel, eff)
            self.state.update_targets(names, tgt)
            hands = [n for n in JOINT_NAMES if JOINT_GROUP[n] in ("left_hand", "right_hand")]
            self.state.update_temps(
                hands, [42 + 30 * (0.5 + 0.5 * math.sin(0.05 * t + i))
                        for i, _ in enumerate(hands)])
            # synthetic rectangular room (5×4 m) around the robot
            scan = []
            for i in range(360):
                th = math.radians(i)
                c, s = math.cos(th), math.sin(th)
                rx = 2.5 / abs(c) if abs(c) > 1e-6 else 1e9
                ry = 2.0 / abs(s) if abs(s) > 1e-6 else 1e9
                scan.append(round(min(rx, ry), 3))
            self.state.scan = scan
            self._stop.wait(1.0 / self.hz)


class RosSource:
    """rclpy tap on the live graph. Read-only except nothing — commands are the
    browser's job (straight to the 9090 bridge)."""

    SUBS_HINT = ["/joint_states", "/left_hand/temperatures",
                 "/right_hand/temperatures", "/mabel/hw/base/estop"]

    def __init__(self, state):
        self.state = state
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        import rclpy                                      # raises if absent
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.state.source = "ros"
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _run(self):
        # Respawn loop: a real_run.sh restart / SIGINT reaching rclpy's signal
        # handling shuts the context down under us (ExternalShutdownException)
        # and used to kill this thread permanently — HTTP kept serving while
        # telemetry AND the /ws/stream clouds went silent (2026-07-23). Each
        # spin gets its own Context so a dead one is simply replaced.
        import logging
        import rclpy
        from rclpy.context import Context
        from rclpy.executors import ExternalShutdownException
        log = logging.getLogger("mabel_cc.telemetry")
        while not self._stop.is_set():
            ctx = Context()
            try:
                rclpy.init(args=None, context=ctx)
                self._spin(ctx)
            except ExternalShutdownException:
                log.warning("ROS tap: context externally shut down — respawning")
            except Exception as e:                             # noqa: BLE001
                log.warning("ROS tap died: %r — respawning", e)
            finally:
                try:
                    rclpy.shutdown(context=ctx)
                except Exception:
                    pass
            self._stop.wait(2.0)

    def _spin(self, ctx):
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
        from sensor_msgs.msg import JointState, LaserScan
        from std_msgs.msg import Bool, Float64MultiArray

        node = rclpy.create_node("mabel_control_center_tap", context=ctx)
        try:                       # selectable cloud/grid streams ride this node
            from . import streams
            streams.HUB.attach(node)
        except Exception:
            pass
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        st = self.state

        def on_js_aggregate(msg):
            # /joint_states is assembled by joint_state_publisher, which
            # ZERO-FILLS velocity/effort — take positions only from it
            st.update_joints(list(msg.name), list(msg.position))

        def on_js_sub(msg):
            # per-subsystem topics carry the real pos/vel/effort feedback
            st.update_joints(list(msg.name), list(msg.position),
                             list(msg.velocity), list(msg.effort))

        def on_cmd(msg):
            st.update_targets(list(msg.name), list(msg.position))

        def on_temp(side):
            def cb(msg):
                names = (msg.layout.dim[0].label.split(",")
                         if msg.layout.dim else [])
                pref = [n if n.startswith(side) else f"{side}_{n}" for n in names]
                st.update_temps(pref, list(msg.data))
            return cb

        def on_estop(msg):
            st.estop = bool(msg.data)

        def on_torque(key):
            def cb(msg):
                st.torque[key] = bool(msg.data)
            return cb

        def on_scan(msg):
            # resample to 360 bins so the payload stays small and uniform
            n = len(msg.ranges)
            if not n:
                return
            st.scan = [round(float(msg.ranges[int(i * n / 360)]), 3)
                       for i in range(360)]

        node.create_subscription(JointState, "/joint_states", on_js_aggregate, 20)
        for t in ("/base/joint_states", "/arms/joint_states",
                  "/body/joint_states", "/left_hand/joint_states",
                  "/right_hand/joint_states"):
            node.create_subscription(JointState, t, on_js_sub, 20)
        try:
            from rclpy.qos import qos_profile_sensor_data
            node.create_subscription(LaserScan, "/scan", on_scan,
                                     qos_profile_sensor_data)
        except Exception:
            node.create_subscription(LaserScan, "/scan", on_scan, 5)
        for t in ("/arms/command", "/left_hand/command", "/right_hand/command",
                  "/body/command"):
            node.create_subscription(JointState, t, on_cmd, 10)
        for side in ("left", "right"):
            node.create_subscription(Float64MultiArray,
                                     f"/{side}_hand/temperatures",
                                     on_temp(side), 10)
        # live intrinsics + distortion for the 3D camera projections
        from sensor_msgs.msg import CameraInfo

        def on_info(ns):
            def cb(msg):
                self.state.caminfo[ns] = {
                    "w": msg.width, "h": msg.height,
                    "fx": msg.k[0], "fy": msg.k[4],
                    "cx": msg.k[2], "cy": msg.k[5],
                    "d": list(msg.d[:5]) + [0.0] * max(0, 5 - len(msg.d)),
                }
            return cb
        for ns, topics in {
            "front": ["/camera/front/color/camera_info"],
            "zed": ["/camera/zed/color/camera_info",
                    "/camera/zed/left/color/camera_info"],
            "left_wrist": ["/camera/left_wrist/color/camera_info"],
            "right_wrist": ["/camera/right_wrist/color/camera_info"],
        }.items():
            for t in topics:
                node.create_subscription(CameraInfo, t, on_info(ns), 5)

        # /odom topic pose → StreamHub: the cloud fallback frame while SLAM
        # publishes no odom TF (same trick as the gateway's CloudFramer)
        try:
            from nav_msgs.msg import Odometry
            from . import streams as _streams

            def on_odom(msg):
                p = msg.pose.pose.position
                q = msg.pose.pose.orientation
                _streams.HUB.set_odom_pose(p.x, p.y, p.z, q.x, q.y, q.z, q.w)

            node.create_subscription(Odometry, "/odom", on_odom,
                                     qos_profile_sensor_data)
        except Exception:
            pass

        node.create_subscription(Bool, "/mabel/hw/base/estop", on_estop, latched)
        node.create_subscription(Bool, "/arms/torque_state",
                                 on_torque("arms"), latched)
        node.create_subscription(Bool, "/left_hand/torque_state",
                                 on_torque("handL"), latched)
        node.create_subscription(Bool, "/right_hand/torque_state",
                                 on_torque("handR"), latched)
        node.create_subscription(Bool, "/body/torque_state",
                                 on_torque("neck"), latched)

        ex = SingleThreadedExecutor(context=ctx)
        ex.add_node(node)
        try:
            while not self._stop.is_set() and ctx.ok():
                ex.spin_once(timeout_sec=0.2)
        finally:
            ex.remove_node(node)
            node.destroy_node()


def make_source(state, kind=None):
    """auto → ROS if rclpy imports, else mock; explicit kinds honored."""
    kind = kind or CFG.TELEMETRY_SOURCE
    if kind in ("ros", "auto"):
        try:
            src = RosSource(state)
            src.start()
            return src
        except Exception:
            if kind == "ros":
                raise
    src = MockSource(state)
    src.start()
    return src
