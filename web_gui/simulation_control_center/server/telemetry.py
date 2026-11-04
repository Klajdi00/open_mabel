"""
Telemetry hub: one background source feeding a shared latest-state snapshot
that the WS endpoint fans out at TELEMETRY_HZ.

Sim twin: the primary source here is SimSource — a passive websocket client of
the SIM BRIDGE itself (ws :9090/teleop robot_state @50 Hz + the read-only
/monitor plane), so every number in the panel is the actual MuJoCo state, not a
synthetic plant. Velocities are finite-differenced from the 50 Hz positions;
joint targets come from the bridge's fused actuation (monitor plane); efforts +
temperatures stay null (the sim publishes neither — honest blanks beat fakes).
RosSource still exists for the Ubuntu sim-with-ROS stack, MockSource for CI.
"""
import json
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
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
        from sensor_msgs.msg import JointState, LaserScan
        from std_msgs.msg import Bool, Float64MultiArray

        rclpy.init(args=None)
        node = Node("mabel_control_center_tap")
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

        node.create_subscription(Bool, "/mabel/hw/base/estop", on_estop, latched)
        node.create_subscription(Bool, "/arms/torque_state",
                                 on_torque("arms"), latched)
        node.create_subscription(Bool, "/left_hand/torque_state",
                                 on_torque("handL"), latched)
        node.create_subscription(Bool, "/right_hand/torque_state",
                                 on_torque("handR"), latched)

        while not self._stop.is_set() and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.2)
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


class SimSource:
    """Live tap on the SIM BRIDGE: a passive client of ws :9090/teleop (the
    same 50 Hz robot_state every app mirrors) + the read-only /monitor plane
    (fused e-stop / joint targets). Never sends teleop frames, so it never
    claims the driver arbitration; it is also the studio's uplink for the
    sim-only commands (reset / estop) via send()."""

    # robot_state.base carries the swerve modules as parallel [FL, FR, B]
    # steer-position and drive-velocity arrays; map them onto canonical names.
    _STEER = ["fl_steer", "fr_steer", "b_steer"]
    _DRIVE = ["fl_drive", "fr_drive", "b_drive"]

    def __init__(self, state, host=None, port=None, token=None):
        self.state = state
        self.host = host or CFG.UPSTREAM_HOST
        self.port = port or CFG.TELEOP_PORT
        self.token = CFG.APP_TOKEN if token is None else token
        self.connected = False
        self.hello = None                    # bridge hello payload (robot identity)
        self.last_state = None               # latest raw robot_state payload
        self._loop = None
        self._ws = None                      # live /teleop socket (send channel)
        self._stop = threading.Event()
        self._started = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="scc-sim-source")
        self._prev = {}                      # name -> (t, pos) for finite-diff vel
        self._vel = {}                       # name -> smoothed velocity
        self._drive_pos = {}                 # integrated drive-wheel angles

    # ── lifecycle ────────────────────────────────────────────────────────────
    def start(self):
        import websockets                              # noqa: F401 — fail fast
        self.state.source = "sim"
        self.state.torque = {"arms": True, "handL": True, "handR": True}
        self._thread.start()
        self._started.wait(timeout=2.0)
        return self

    def stop(self):
        self._stop.set()
        if self._loop is not None:
            try:
                self._loop.call_soon_threadsafe(lambda: None)   # wake the loop
            except Exception:
                pass
        self._thread.join(timeout=3.0)

    def send(self, msg_type, payload=None):
        """Send one wire envelope up the live /teleop socket (reset, estop…).
        Returns False when the bridge is unreachable."""
        import asyncio
        if self._loop is None or self._ws is None:
            return False
        raw = json.dumps({"type": msg_type, "payload": payload or {}})

        async def _send():
            await self._ws.send(raw)
        try:
            fut = asyncio.run_coroutine_threadsafe(_send(), self._loop)
            fut.result(timeout=3.0)
            return True
        except Exception:
            return False

    # ── the client loops (own asyncio loop on the thread) ───────────────────
    def _url(self, path):
        q = f"?client=sim-control-center"
        if self.token:
            q += f"&key={self.token}"
        return f"ws://{self.host}:{self.port}{path}{q}"

    def _run(self):
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._started.set()

        async def _main():
            await asyncio.gather(self._teleop_loop(), self._monitor_loop())
        try:
            loop.run_until_complete(_main())
        finally:
            loop.close()

    async def _teleop_loop(self):
        import asyncio
        import websockets
        while not self._stop.is_set():
            try:
                async with websockets.connect(
                        self._url("/teleop"), max_size=8 * 1024 * 1024,
                        open_timeout=3, close_timeout=1) as ws:
                    self._ws = ws
                    self.connected = True
                    async for raw in ws:
                        if self._stop.is_set():
                            break
                        try:
                            msg = json.loads(raw)
                        except Exception:
                            continue
                        t, p = msg.get("type"), msg.get("payload") or {}
                        if t == "robot_state":
                            self._on_state(p)
                        elif t == "hello":
                            self.hello = p
            except Exception:
                pass
            finally:
                self._ws = None
                self.connected = False
            if not self._stop.is_set():
                await asyncio.sleep(1.5)

    async def _monitor_loop(self):
        import asyncio
        import websockets
        while not self._stop.is_set():
            try:
                async with websockets.connect(
                        self._url("/monitor"), max_size=8 * 1024 * 1024,
                        open_timeout=3, close_timeout=1) as ws:
                    async for raw in ws:
                        if self._stop.is_set():
                            break
                        try:
                            msg = json.loads(raw)
                        except Exception:
                            continue
                        self._on_monitor(msg)
            except Exception:
                pass
            if not self._stop.is_set():
                await asyncio.sleep(2.0)

    # ── mapping the bridge planes onto the telemetry store ──────────────────
    def _on_state(self, p):
        now = time.time()
        self.last_state = p
        names, pos, vel = [], [], []
        for n, v in (p.get("jointPositions") or {}).items():
            if n not in JOINT_GROUP:
                continue
            v = float(v)
            names.append(n)
            pos.append(v)
            vel.append(self._fd_vel(n, now, v))
        base = p.get("base") or {}
        steer = base.get("steer") or []
        for n, v in zip(self._STEER, steer):
            names.append(n)
            pos.append(float(v))
            vel.append(self._fd_vel(n, now, float(v)))
        drive = base.get("drive") or []
        for n, w in zip(self._DRIVE, drive):
            w = float(w)
            # the bridge sends drive-wheel VELOCITY; integrate a relative angle
            # so the position band still shows the wheels turning
            t_prev, a_prev = self._drive_pos.get(n, (now, 0.0))
            a = a_prev + w * min(0.1, max(0.0, now - t_prev))
            self._drive_pos[n] = (now, a)
            names.append(n)
            pos.append(a)
            vel.append(w)
        self.state.update_joints(names, pos, vel)

    def _fd_vel(self, name, now, v):
        t0, p0 = self._prev.get(name, (None, None))
        self._prev[name] = (now, v)
        if t0 is None or now <= t0:
            return 0.0
        raw = (v - p0) / (now - t0)
        sm = 0.6 * self._vel.get(name, raw) + 0.4 * raw
        self._vel[name] = sm
        return sm

    def _on_monitor(self, msg):
        g = msg.get("global") or {}
        status = g.get("status") or {}
        if "estop" in status:
            estop = bool(status.get("estop"))
            self.state.estop = estop
            # sim hand torque rides the e-stop edge (bridge convention)
            self.state.torque = {"arms": not estop, "handL": not estop,
                                 "handR": not estop}
        act = g.get("actuation") or {}
        joints = act.get("joints") or {}
        if joints:
            names = [n for n in joints if n in JOINT_GROUP]
            self.state.update_targets(names, [float(joints[n]) for n in names])
        wheels = act.get("wheels") or {}
        wnames, wvals = [], []
        for mod, canon in (("FL", "fl_steer"), ("FR", "fr_steer"),
                           ("B", "b_steer")):
            if mod in wheels and "steer" in wheels[mod]:
                wnames.append(canon)
                wvals.append(float(wheels[mod]["steer"]))
        if wnames:
            self.state.update_targets(wnames, wvals)


def make_source(state, kind=None):
    """auto → ROS if rclpy imports, else the SIM bridge tap (retries until the
    bridge is up — never silently degrades to mock); explicit kinds honored.
    `demo` REPLAYS a recorded episode (offline demo with no sim/bridge)."""
    kind = kind or CFG.TELEMETRY_SOURCE
    if kind == "demo":
        from . import demo
        path = demo.find_demo_episode()
        if path:
            ep = demo.DemoEpisode(path)
            state.demo_episode = ep                 # camera fallback reads this
            return demo.DemoSource(state, ep).start()
        # no episode on disk → fall back to the synthetic mock so it still opens
        src = MockSource(state); src.start(); return src
    if kind in ("ros", "auto"):
        try:
            src = RosSource(state)
            src.start()
            return src
        except Exception:
            if kind == "ros":
                raise
    if kind in ("sim", "auto"):
        try:
            return SimSource(state).start()
        except Exception:
            if kind == "sim":
                raise
    src = MockSource(state)
    src.start()
    return src
