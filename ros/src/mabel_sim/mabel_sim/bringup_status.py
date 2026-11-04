#!/usr/bin/env python3
"""bringup_status -- a pinned, colour terminal dashboard for the MABEL HAL.

Both bring-ups otherwise tell you they are alive only through scattered per-driver
logger lines ("SwerveClient connected", "Neck online", ...). This node gives them
the same at-a-glance "is everything talking?" feel, but as a single tidy panel
that STAYS PINNED to the top of the terminal while the node logs scroll underneath
it -- so you can watch liveness without losing the launch output.

ONE node serves both backends via the ``mode`` parameter, so the panel design
stays identical between them:
  * ``mode:=sim``      -- the MuJoCo HAL view (``mabel_sim sim.launch.py``):
                          physics ``/clock`` + a real-time-factor read, and the
                          simulated sensor topics (front/back D435i, ZED head,
                          body/wrist cams, ``/scan``).
  * ``mode:=hardware`` -- the real-robot view (``mabel_bringup
                          hardware.launch.py``): the actual driver topics from the
                          interface_config contract (per-subsystem joint_states,
                          ORCA hands) and the real mabel_sensors camera stack.
                          The robot runs on wall time (no ``/clock``), so the
                          banner shows a live "N/M online" count instead of RTF.

How the pin works (no curses, no extra window):
  * We write the panel straight to the controlling terminal ``/dev/tty`` rather
    than stdout, so ``ros2 launch``'s line-prefixing/buffering never mangles our
    cursor-control escapes.
  * We set a DEC scroll region (CSI ``t;b r``) BELOW the panel. The terminal then
    confines all scrolling -- i.e. every other node's log output -- to that lower
    region, leaving the top rows (the panel) untouched. A timer redraws the panel
    in place a few times a second, so it self-heals against the occasional
    interleaved write from another process sharing the tty.
  * On shutdown we reset the scroll region and restore the cursor.

If there is no controlling terminal (headless / piped / no tty -- e.g. a cron or
``nohup`` run) we fall back to emitting a one-line summary via the ROS logger every
few seconds, so the node is still useful (and never crashes) without a tty.

Liveness is measured from the WALL clock (``time.monotonic``), independent of sim
time, so a slow real-time-factor never makes the panel look frozen. Each signal is
tracked via a tiny EMA of its inter-message period -> a live Hz read plus an
OK/WAIT/STALE state. Cameras/clouds are watched through their lightweight
``camera_info`` topics (published in lockstep with the image/cloud) rather than the
heavy image/PointCloud2 payloads, so the monitor itself costs almost nothing.

Subscriptions use a BEST_EFFORT QoS, which is compatible with both reliable and
best-effort publishers, so the panel sees every topic regardless of how each node
chose to publish it.
"""

import json
import os
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState, LaserScan, CameraInfo, Image
from nav_msgs.msg import Odometry
from tf2_msgs.msg import TFMessage
from std_msgs.msg import String


# --------------------------------------------------------------------------- #
#  Terminal / ANSI helpers
# --------------------------------------------------------------------------- #
CSI = "\033["
RESET = CSI + "0m"


def _sgr(*codes):
    return CSI + ";".join(str(c) for c in codes) + "m"


BOLD = _sgr(1)
DIM = _sgr(2)
CYAN = _sgr(36)
GREY = _sgr(90)
GREEN = _sgr(32)
YELLOW = _sgr(33)
RED = _sgr(31)
WHITE = _sgr(97)

# Status -> (dot colour, label colour, text)
_STATE_STYLE = {
    "OK":   (GREEN,  WHITE, "online"),
    "WAIT": (YELLOW, DIM,   "waiting"),
    "STALE": (RED,   DIM,   "STALE"),
    "OFF":  (GREY,   GREY,  "offline"),
    "TODO": (GREY,   DIM,   "n/a"),     # hardware: detection not wired up yet
}


def _vislen(s):
    """Printed width of a string ignoring ANSI SGR sequences."""
    out, i = 0, 0
    n = len(s)
    while i < n:
        if s[i] == "\033":
            j = s.find("m", i)
            i = (j + 1) if j != -1 else n
            continue
        out += 1
        i += 1
    return out


# --------------------------------------------------------------------------- #
#  Per-signal liveness + rate tracker
# --------------------------------------------------------------------------- #
class _Signal:
    """Tracks last-seen time and an EMA Hz for one monitored topic.

    Hardware-level detection (set via the sig() helper):
      * ``validate(msg)`` -- a content check; a message only counts as a "tick"
        if it is actually POPULATED (e.g. a JointState that carries names +
        positions, a CameraInfo with a non-zero resolution). An empty/placeholder
        message therefore never makes a row look alive.
      * ``real_only`` -- the row is OK only when a NON-simulation node is
        publishing the topic. The sim publishes the very same contract topics via
        its ``mujoco_*`` nodes, so this is what stops a running sim (or a driver
        whose hardware is unplugged and silent) from reading as online. The live
        publisher set is refreshed by the node's _update_links timer.
      * ``dummy`` -- a placeholder row whose detection is not wired up yet; always
        renders as "n/a" (TODO) so it is never mistaken for a real reading.
      * ``rate_of(msg)`` -- report the rate of the DATA the row is about, when
        that differs from the rate of the topic being watched. Needed when a row
        is proved by a slow heartbeat but describes a fast stream: the audio rows
        watch /audio/device_health at 2 Hz (subscribing to /audio/mic would hold
        the sound card open — see _build_groups_hardware), and rendering that 2
        in the same column as "arms 50.4 Hz" reads as "the mic runs at 2 Hz",
        which for audio is nonsense. Return the true Hz, or None to fall back to
        the inter-message EMA.
    """

    STALE_AFTER = 2.0   # s without a message -> STALE

    def __init__(self, label, enabled=True):
        self.label = label
        self.enabled = enabled
        self.count = 0
        self.hz = 0.0
        self._last = None      # monotonic time of last message
        self._prev = None      # monotonic time of the previous message
        # hardware-level detection (defaults = legacy/sim behaviour)
        self.topic = None
        self.validate = None       # msg -> bool: is this message populated?
        self.real_only = False     # require a non-mujoco publisher
        self.dummy = False         # detection not implemented yet
        self.rate_of = None        # msg -> float|None: the DATA's rate, not the topic's
        self._has_real_pub = False  # set by BringupStatus._update_links
        self._real_streak = 0      # consecutive cycles a real pub was seen (debounce)

    def _as_dummy(self):
        """Mark this as a placeholder row (renders 'n/a') and return self."""
        self.dummy = True
        return self

    def tick(self, msg=None):
        # Drop empty / not-yet-populated messages so they never count as live.
        if self.validate is not None and msg is not None:
            try:
                if not self.validate(msg):
                    return
            except Exception:
                return
        now = time.monotonic()
        if self.rate_of is not None:
            try:
                r = self.rate_of(msg)
            except Exception:                       # noqa: BLE001
                r = None
            if r is not None:
                self.hz = float(r)
        elif self._prev is not None:
            dt = now - self._prev
            if dt > 1e-6:
                inst = 1.0 / dt
                self.hz = inst if self.hz == 0.0 else 0.7 * self.hz + 0.3 * inst
        self._prev = now
        self._last = now
        self.count += 1

    def state(self):
        if self.dummy:
            return "TODO"
        if not self.enabled:
            return "OFF"
        # Real data must come from an actual hardware driver, not the simulator.
        if self.real_only and not self._has_real_pub:
            return "OFF"
        if self._last is None:
            return "WAIT"
        if (time.monotonic() - self._last) > self.STALE_AFTER:
            return "STALE"
        return "OK"


# --------------------------------------------------------------------------- #
#  Dashboard node
# --------------------------------------------------------------------------- #
class BringupStatus(Node):
    def __init__(self):
        super().__init__("bringup_status")

        # mode selects WHICH signal set + banner the panel shows. "sim" is the
        # original MuJoCo HAL view (physics /clock + RTF, sim sensor topics);
        # "hardware" is the real-robot view used by mabel_bringup -- same panel,
        # but the real subsystem/driver topics and (since the robot runs on wall
        # time, with no /clock) a live "online" count in place of the RTF.
        self.declare_parameter("mode", "sim")          # "sim" | "hardware"
        self.declare_parameter("title", "")            # "" -> per-mode default
        self.declare_parameter("subtitle", "")         # "" -> per-mode default
        self.declare_parameter("sensors", True)        # camera (+sim LiDAR) bridge up
        self.declare_parameter("wheel_odom", True)     # swerve_odometry running
        self.declare_parameter("front_only", False)    # only the front camera is up
        self.declare_parameter("foxglove", False)      # foxglove bridge running
        self.declare_parameter("arms", True)           # (hardware) arms driver up
        self.declare_parameter("body", True)           # (hardware) lift/torso/neck driver up
        self.declare_parameter("hands", True)          # (hardware) ORCA hand drivers up
        self.declare_parameter("lidar", True)          # (hardware) RPLIDAR /scan up
        self.declare_parameter("audio", True)          # (hardware) USB sound card up
        self.declare_parameter("refresh_rate", 5.0)    # panel redraws / s

        self.mode = str(self.get_parameter("mode").value).lower()
        self._hw = (self.mode == "hardware")

        if self._hw:
            def_title, def_sub = "MABEL  ·  HARDWARE BRING-UP", "real-time robot HAL"
        else:
            def_title, def_sub = "MABEL  ·  SIMULATION BRING-UP", "sim-time robot HAL"
        self.title = str(self.get_parameter("title").value) or def_title
        self.subtitle = str(self.get_parameter("subtitle").value) or def_sub

        sensors = bool(self.get_parameter("sensors").value)
        wheel_odom = bool(self.get_parameter("wheel_odom").value)
        front_only = bool(self.get_parameter("front_only").value)
        foxglove = bool(self.get_parameter("foxglove").value)
        arms = bool(self.get_parameter("arms").value)
        body = bool(self.get_parameter("body").value)
        hands = bool(self.get_parameter("hands").value)
        lidar = bool(self.get_parameter("lidar").value)
        audio = bool(self.get_parameter("audio").value)
        refresh = float(self.get_parameter("refresh_rate").value)

        # ---- the signals we monitor, grouped for display --------------------
        # (group_header, [(key, Signal), ...]);  self._sigs holds the
        # (Signal, topic, msg_type) tuples that get a subscription.
        self._groups = []
        self._sigs = []

        def sig(label, topic, mtype, enabled=True, validate=None,
                real_only=False, rate_of=None):
            s = _Signal(label, enabled)
            s.topic = topic
            s.validate = validate
            s.real_only = real_only
            s.rate_of = rate_of
            self._sigs.append((s, topic, mtype))
            return s

        if self._hw:
            self._build_groups_hardware(sig, sensors, wheel_odom, front_only,
                                        arms, body, hands, lidar, audio, foxglove)
        else:
            self._build_groups_sim(sig, sensors, wheel_odom, front_only, foxglove)

        # ---- wire the subscriptions -----------------------------------------
        qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST)
        for s, topic, mtype in self._sigs:
            self.create_subscription(
                mtype, topic, (lambda m, sg=s: sg.tick(m)), qos)

        # ---- real-time-factor from /clock vs the wall clock -----------------
        # Sim only: real hardware has no /clock, so the banner shows an online
        # count there instead (see _build_lines).
        self._rtf = 0.0
        self._clk_prev = None       # (wall_monotonic, sim_seconds)
        if not self._hw:
            self.create_subscription(Clock, "/clock", self._on_clock, qos)

        self._t0 = time.monotonic()

        # ---- terminal setup -------------------------------------------------
        self._tty = None
        self._region = None         # (header_lines, rows) currently programmed
        try:
            self._tty = open("/dev/tty", "w")
            self._tty_fd = self._tty.fileno()
            self._w(CSI + "?25l")   # hide cursor while we own the top rows
        except OSError:
            self._tty = None
            self.get_logger().info(
                "no /dev/tty -- bringup_status falls back to periodic log lines.")

        period = 1.0 / max(refresh, 0.5)
        self.create_timer(period, self._render)
        self._log_accum = 0.0       # for the no-tty fallback cadence

        # Hardware: refresh which topics have a REAL (non-sim) publisher, so the
        # panel can tell an actual driver apart from the simulator publishing the
        # same contract topics. Cheap local graph query, ~1 Hz.
        if self._hw:
            self._update_links()
            self.create_timer(0.5, self._update_links)   # 2-cycle debounce ~= 1 s

    # ------------------------------------------------------------- group specs
    def _build_groups_sim(self, sig, sensors, wheel_odom, front_only, foxglove):
        """The original MuJoCo-HAL signal set (sim.launch.py)."""
        core = [
            ("physics", sig("MuJoCo /clock", "/clock", Clock)),
            ("state",   sig("/joint_states", "/joint_states", JointState)),
            ("odom",    sig("/odom  (swerve)", "/odom", Odometry, wheel_odom)),
            ("tf",      sig("/tf", "/tf", TFMessage)),
        ]
        self._groups.append(("ROBOT", core))

        if sensors:
            lidar = [("scan", sig("/scan  (RPLIDAR A3)", "/scan", LaserScan))]
            self._groups.append(("LiDAR", lidar))

            cams = [
                ("front", sig("front  D435i  rgb",
                              "/camera/front/color/camera_info", CameraInfo)),
                ("ir",    sig("front  D435i  IR stereo",
                              "/camera/front/infra1/camera_info", CameraInfo)),
            ]
            if not front_only:
                cams += [
                    ("back",  sig("back   D435i  rgb",
                                  "/camera/back/color/camera_info", CameraInfo)),
                    ("zedl",  sig("zed/left   (head)",
                                  "/camera/zed/left/camera_info", CameraInfo)),
                    ("zedr",  sig("zed/right  (head)",
                                  "/camera/zed/right/camera_info", CameraInfo)),
                    ("body",  sig("body",
                                  "/camera/body/camera_info", CameraInfo)),
                    ("lw",    sig("left_wrist",
                                  "/camera/left_wrist/camera_info", CameraInfo)),
                    ("rw",    sig("right_wrist",
                                  "/camera/right_wrist/camera_info", CameraInfo)),
                ]
            self._groups.append(("CAMERAS  (rgb)", cams))

            clouds = [
                ("fpc", sig("front  depth cloud",
                            "/camera/front/depth/camera_info", CameraInfo)),
            ]
            if not front_only:
                clouds += [
                    ("zpc", sig("zed    head cloud",
                                "/camera/zed/depth/camera_info", CameraInfo)),
                    ("bpc", sig("back   depth cloud",
                                "/camera/back/depth/camera_info", CameraInfo)),
                ]
            self._groups.append(("POINT CLOUDS", clouds))

        if foxglove:
            self._groups.append(
                ("BRIDGES", [("fox", _Signal("foxglove ws://:8765", True))]))

    def _build_groups_hardware(self, sig, sensors, wheel_odom, front_only,
                               arms, body, hands, lidar, audio, foxglove):
        """The real-robot signal set (mabel_bringup hardware.launch.py).

        Detection is done at the HARDWARE level: a row is ONLINE only when a driver
        is actually publishing POPULATED data on its topic -- encoder joint states,
        real CameraInfo / Image (non-zero resolution), a LaserScan with ranges. A
        driver whose hardware is unplugged advertises its topic but goes silent, so
        it correctly reads as WAIT/offline. real_only additionally requires a
        NON-sim publisher (the sim publishes the same contract topics), so a running
        sim can never read as online.

        Layout (mirrors the physical robot):
          BASE     swerve drive
          ARMS     left + right arm        (one bimanual driver, /arms/joint_states)
          BODY     lift + torso + neck     (one body driver,      /body/joint_states)
          HANDS    left + right hand       (ORCA)
          SENSORS  base-front D435i · front-head ZED-M · L/R wrist · RPLIDAR /scan
          AUDIO    speaker + mic           (one USB sound card, duplex)
                   (RGB only -- if the RGB of a depth cam is up, its depth is up
                   too, so no separate depth rows)
        """
        # Content checks -- a message only counts if it actually carries data.
        def js_ok(m):
            return bool(m.name) and bool(m.position)

        def cam_ok(m):
            return m.width > 0 and m.height > 0          # CameraInfo / Image

        def scan_ok(m):
            return len(m.ranges) > 0

        def has_joint(*subs):
            """True iff the JointState names ANY joint whose name contains one of
            `subs`. Lets ONE /body/joint_states topic drive per-BUS rows: each HAL
            node only names a joint while its motor streams live feedback (dead bus
            -> joint absent), so a substring test tells the neck's Dynamixel bus
            (pitch/roll) apart from its DaMiao bus (yaw) and from the torso DaMiao."""
            def _v(m):
                return bool(m.name) and any(
                    any(s in n for s in subs) for n in m.name)
            return _v

        # -- BASE: swerve_node streams encoder JointState ONLY while the Teensy
        #    link is up. Watch /base/joint_states (the base driver's dedicated
        #    topic), NOT /joint_states (joint_state_publisher fills that with home
        #    defaults -> would read as a permanently-"connected" base).
        self._groups.append(("BASE", [
            ("base", sig("swerve drive  (Teensy encoders)", "/base/joint_states",
                         JointState, validate=js_ok, real_only=True)),
        ]))

        # -- ARMS: one bimanual driver publishes BOTH arms on /arms/joint_states, but
        #    the two arms share ONE openarm-dual link — so a generic "any joint" check
        #    would read the DEAD arm as connected off the LIVE arm's data. Gate each row
        #    on ITS OWN side's joints; the HAL only names a joint whose motor shows live
        #    measurement (per-motor liveness in drivers/openarm.py), so a disconnected
        #    arm drops out of the stream and its row goes WAIT/STALE honestly.
        if arms:
            self._groups.append(("ARMS", [
                ("la", sig("left arm   (DM-J4310 CAN)", "/arms/joint_states",
                           JointState, validate=has_joint("left_arm"))),
                ("ra", sig("right arm  (DM-J4310 CAN)", "/arms/joint_states",
                           JointState, validate=has_joint("right_arm"))),
            ]))

        # -- BODY: FOUR independent connections, NOT one "body driver". Each row must
        #    prove ITS OWN hardware is streaming, or an unplugged bus reads "connected"
        #    off a sibling's data:
        #      lift          Pico over serial      -> its OWN topic /lift/joint_states
        #                    (watching /body/joint_states made it read connected off the
        #                     neck even with no lift attached).
        #      torso         DaMiao on the CAN bus -> the `torso` joint must be present.
        #      neck dynamixel pitch/roll, serial   -> a neck_pitch/neck_roll (or
        #                     motor_* fallback) joint must be present.
        #      neck damiao    yaw, CAN             -> the `neck_yaw` joint must be present.
        #    (Each HAL node names a joint only while its motor streams live feedback, so
        #     a dead bus simply drops out of /body/joint_states -> row goes WAIT/STALE.)
        if body:
            self._groups.append(("BODY", [
                ("lift",     sig("lift            (Pico serial)", "/lift/joint_states",
                                 JointState, validate=js_ok)),
                ("torso",    sig("torso           (DaMiao serial)", "/body/joint_states",
                                 JointState, validate=has_joint("torso"))),
                ("neck_dxl", sig("neck dynamixel  (pitch/roll)", "/body/joint_states",
                                 JointState,
                                 validate=has_joint("neck_pitch", "neck_roll", "motor_"))),
                ("neck_dm",  sig("neck damiao     (yaw)", "/body/joint_states",
                                 JointState, validate=has_joint("neck_yaw"))),
            ]))

        if hands:
            self._groups.append(("HANDS", [
                ("lh", sig("left_hand   (ORCA)", "/left_hand/joint_states",
                           JointState, validate=js_ok)),
                ("rh", sig("right_hand  (ORCA)", "/right_hand/joint_states",
                           JointState, validate=js_ok)),
            ]))

        if sensors:
            # ONE row per camera, watched on its RGB stream -- if the RGB is up the
            # depth (for the D435/ZED/D405) is up too, so no separate depth rows.
            # The robot's sensor suite: base-front D435i + front-head ZED-M + the
            # two wrist cams + the RPLIDAR (all owned by the hardware_bridge).
            cams = [
                ("front", sig("base front  D435i", "/camera/front/color/camera_info",
                              CameraInfo, validate=cam_ok, real_only=True)),
                ("head",  sig("front head  ZED-M", "/camera/zed/left/camera_info",
                              CameraInfo, validate=cam_ok, real_only=True)),
            ]
            if not front_only:
                cams += [
                    # Real-robot wrist cams publish /camera/wrist_{left,right}/* (the
                    # hardware_bridge contract; BOTH wrists = RealSense D405, sides
                    # pinned by librealsense serial in hardware_bridge/config.py).
                    # NOTE the sim uses the OPPOSITE order /camera/{left,right}_wrist/*
                    # (see _build_groups_sim + sensor_config.py) -- the two domains
                    # differ, so watch the hardware names here or the panel reads
                    # "waiting" forever.
                    ("lw",   sig("left_wrist   D405", "/camera/wrist_left/image_raw",
                                 Image, validate=cam_ok)),
                    ("rw",   sig("right_wrist  D405", "/camera/wrist_right/image_raw",
                                 Image, validate=cam_ok)),
                ]
            if lidar:
                cams.append(
                    ("scan", sig("RPLIDAR A2M12  /scan", "/scan", LaserScan,
                                 validate=scan_ok)))
            self._groups.append(("SENSORS", cams))

        # -- AUDIO: one USB sound card is BOTH the speaker and the mic (one
        #    duplex ALSA card), but they fail independently -- a pulled mic plug
        #    (or, on the old bar, its physical mute button) silences the mic while
        #    the speaker stays perfect -- so show a row each, like the arms share
        #    /arms/joint_states.
        #
        #    Both rows watch /audio/device_health, NOT /audio/mic. Mic capture is
        #    DEMAND-DRIVEN: the HAL opens the sound card only while something is
        #    subscribed. A panel that watched the mic topic would therefore HOLD
        #    THE CARD OPEN for as long as the panel runs -- a status display must
        #    never change the thing it measures (and it would block a calibration
        #    studio / speaker-test from ever claiming the device). device_health
        #    publishes at a steady 2 Hz whenever the subsystem is up, listener or
        #    not, which is exactly the liveness question this panel asks.
        if audio:
            def audio_ok(m):
                try:
                    return bool((json.loads(m.data) or {}).get("connected"))
                except Exception:                       # noqa: BLE001
                    return False

            def frame_rate(side):
                """Differentiate device_health's monotonic frame counter -> the
                REAL audio rate (~50/s while flowing, 0 while idle).

                Without this the row would print the HEARTBEAT's 2 Hz next to
                "arms 50.4 Hz" and read as "the mic runs at 2 Hz" — nonsense for
                audio, and exactly the kind of number someone reasonably panics
                about. 0.0 Hz here is honest and useful: the card is up and
                nothing is flowing (nobody is listening — capture is
                demand-driven — or nobody is talking)."""
                prev = {"n": None, "t": None}

                def _rate(m):
                    try:
                        n = int((json.loads(m.data) or {})[side]["frames"])
                    except Exception:                   # noqa: BLE001
                        return None
                    now = time.monotonic()
                    n0, t0 = prev["n"], prev["t"]
                    prev["n"], prev["t"] = n, now
                    if n0 is None or now - t0 < 1e-6:
                        return 0.0                      # first sample: no delta yet
                    return max(0.0, (n - n0) / (now - t0))
                return _rate

            self._groups.append(("AUDIO", [
                ("spk", sig("speaker  (JMTek USB card)", "/audio/device_health",
                            String, validate=audio_ok, rate_of=frame_rate("speaker"))),
                ("mic", sig("mic      (JMTek USB card)", "/audio/device_health",
                            String, validate=audio_ok, rate_of=frame_rate("mic"))),
            ]))

        if foxglove:
            self._groups.append(
                ("BRIDGES", [("fox", _Signal("foxglove ws://:8765", True))]))

    def _update_links(self):
        """Refresh, per real_only row, whether a NON-sim node publishes its topic.

        A cheap local ROS-graph query; the result gates each row's ONLINE state so
        the simulator (which publishes the same contract topics via mujoco_* nodes)
        is never mistaken for real hardware."""
        for s, topic, _m in self._sigs:
            if not s.real_only:
                continue
            try:
                infos = self.get_publishers_info_by_topic(topic)
            except Exception:
                infos = []
            # A real driver has a proper, non-sim node name. Require the name to
            # be NON-EMPTY (a still-discovering endpoint can momentarily report
            # "", which would otherwise read as a spurious real publisher) and to
            # not be one of the simulator's mujoco_* nodes.
            real = any(
                (getattr(i, "node_name", "") or "") and
                "mujoco" not in i.node_name
                for i in infos)
            # HARDWARE PROFILE: there is NO simulator on the graph, so the whole
            # point of this query (exclude mujoco_* pubs) is moot — and it actively
            # HURTS. A camera worker is its own subprocess (`mabel_hw_cam_*`), and
            # cross-process / cross-RMW discovery can leave its endpoint's node_name
            # momentarily blank, which stranded a STREAMING camera at "offline"
            # (its /camera/*/camera_info flows in lockstep with the MJPEG the GUI
            # shows, but the name check failed → the row never went ONLINE). On real
            # hardware, ANY publisher — or a freshly VALIDATED message — is proof of a
            # real device; the row's own validator + staleness already gate OK/WAIT.
            if self._hw and (infos or (
                    s._last is not None
                    and (time.monotonic() - s._last) <= s.STALE_AFTER)):
                real = True
            # Debounce: only trust a real publisher after it has been seen for two
            # consecutive cycles, so a transient discovery-time endpoint (which can
            # briefly appear while DDS is still resolving node names) never flashes
            # a row "online" for a tick.
            s._real_streak = (s._real_streak + 1) if real else 0
            s._has_real_pub = s._real_streak >= 2

    def _online_counts(self):
        """(#OK, #enabled) across all monitored signals -- the hardware banner
        metric in place of the sim RTF."""
        ok = tot = 0
        for _h, items in self._groups:
            for _k, s in items:
                if not s.enabled or s.dummy:
                    continue
                tot += 1
                if s.state() == "OK":
                    ok += 1
        return ok, tot

    # ------------------------------------------------------------------ clock
    def _on_clock(self, msg):
        sim = msg.clock.sec + msg.clock.nanosec * 1e-9
        wall = time.monotonic()
        if self._clk_prev is not None:
            w0, s0 = self._clk_prev
            dw = wall - w0
            if dw > 0.05:           # update a few times a second, smoothed
                inst = (sim - s0) / dw
                self._rtf = inst if self._rtf == 0.0 else 0.8 * self._rtf + 0.2 * inst
                self._clk_prev = (wall, sim)
        else:
            self._clk_prev = (wall, sim)

    # ------------------------------------------------------------------ tty io
    def _w(self, s):
        if self._tty is not None:
            self._tty.write(s)

    def _term_size(self):
        try:
            sz = os.get_terminal_size(self._tty_fd)
            return sz.lines, sz.columns
        except OSError:
            return 24, 80

    # ----------------------------------------------------------------- render
    def _build_lines(self, cols):
        """Return the list of panel lines (with ANSI), each <= cols wide."""
        now_dot = GREEN  # default
        lines = []
        inner = max(40, cols)

        # --- banner ---
        title = f"{BOLD}{WHITE}{self.title}{RESET}"
        if self._hw:
            # No sim clock on the robot -> show a live online count instead of RTF.
            n_ok, n_tot = self._online_counts()
            col = GREEN if n_tot and n_ok == n_tot else (YELLOW if n_ok else RED)
            metric = f"{col}{n_ok}/{n_tot} live{RESET}"
        else:
            metric = (f"{CYAN}RTF {self._rtf:4.2f}{RESET}" if self._rtf
                      else f"{DIM}RTF --{RESET}")
        el = time.monotonic() - self._t0
        elapsed = f"{DIM}up {int(el // 60):02d}:{int(el % 60):02d}{RESET}"
        bar = "─" * inner
        lines.append(f"{CYAN}{bar}{RESET}")
        right = f"{metric}   {elapsed}"
        pad = inner - _vislen(title) - _vislen(right) - 2
        pad = max(1, pad)
        lines.append(f" {title}{' ' * pad}{right} ")
        lines.append(f"{CYAN}{bar}{RESET}")

        # --- groups ---
        for header, items in self._groups:
            lines.append(f" {CYAN}{header}{RESET}")
            for _key, s in items:
                st = s.state()
                dot_c, lab_c, word = _STATE_STYLE[st]
                dot = f"{dot_c}●{RESET}"
                label = f"{lab_c}{s.label}{RESET}"
                if st == "OK" and s.hz > 0.0:
                    val = f"{GREEN}{word:>7}{RESET} {DIM}{s.hz:5.1f} Hz{RESET}"
                else:
                    valcol = {"OK": GREEN, "WAIT": YELLOW, "STALE": RED,
                              "OFF": GREY, "TODO": GREY}[st]
                    val = f"{valcol}{word:>7}{RESET}"
                left = f"   {dot} {label}"
                fill = inner - _vislen(left) - _vislen(val) - 2
                fill = max(1, fill)
                lines.append(f"{left} {GREY}{'.' * fill}{RESET} {val}")

        foot = (f"{DIM}{len(self._sigs)} signals · panel pinned · "
                f"logs scroll below{RESET}")
        lines.append(f"{CYAN}{bar}{RESET}")
        lines.append(f" {foot}")
        return lines

    def _render(self):
        if self._tty is None:
            self._render_fallback()
            return
        rows, cols = self._term_size()
        lines = self._build_lines(cols)
        hh = len(lines)

        # Need at least a few rows of scroll area below the panel; if the window
        # is too short, skip the scroll region (just repaint in place) so we
        # never wedge the terminal.
        use_region = (hh + 3) <= rows
        if use_region and self._region != (hh, rows):
            # (re)program the DEC scroll region to start just below the panel.
            self._w(f"{CSI}{hh + 1};{rows}r")    # DECSTBM: scroll only rows below
            self._w(f"{CSI}{rows};1H")           # park cursor at the bottom
            self._region = (hh, rows)

        out = [CSI + "7"]                        # DECSC: save cursor
        for i, line in enumerate(lines):
            out.append(f"{CSI}{i + 1};1H")       # absolute move, row i+1
            out.append(CSI + "2K")               # clear the line
            out.append(line)
            out.append(RESET)
        out.append(CSI + "8")                    # DECRC: restore cursor
        self._w("".join(out))
        try:
            self._tty.flush()
        except (OSError, ValueError):
            pass

    def _render_fallback(self):
        """No tty: emit a compact one-line status through the ROS logger."""
        self._log_accum += 1
        if self._log_accum < 5:                  # ~ every 5 refreshes
            return
        self._log_accum = 0
        parts = []
        for _h, items in self._groups:
            for _k, s in items:
                mark = {"OK": "+", "WAIT": "?", "STALE": "!",
                        "OFF": "-", "TODO": "."}[s.state()]
                parts.append(f"{mark}{s.label.split()[0]}")
        self.get_logger().info("bringup: " + " ".join(parts) +
                               f"  (RTF {self._rtf:.2f})")

    # ---------------------------------------------------------------- shutdown
    def restore_terminal(self):
        if self._tty is None:
            return
        try:
            self._w(f"{CSI}r")          # reset scroll region to full screen
            self._w(f"{CSI}?25h")       # show cursor
            self._w("\n")
            self._tty.flush()
            self._tty.close()
        except (OSError, ValueError):
            pass
        self._tty = None

    def destroy_node(self):
        self.restore_terminal()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = BringupStatus()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.restore_terminal()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
