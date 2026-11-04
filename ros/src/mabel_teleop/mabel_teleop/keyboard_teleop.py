#!/usr/bin/env python3
"""MABEL keyboard whole-body teleop (TAB cycles through the modes).

Port of the MuJoCo sim's teleop_keyboard.py to ROS 2.

  Mode 1 - DIRECT NAV:  drive the 3-module swerve base + jog the lift.
  Mode 2 - SE(3) IK:    move 6-DOF palm targets; the IK tracks them with the arms +
                        wrists while the LIFT and TORSO hold exactly where nav left
                        them (same hold as nav -> no jump on handover) and the lazy
                        base creeps to keep the hands reachable. Two GREEN balls in
                        RViz mark the live left/right wrist (palm) targets.
  Mode 3 - COMPLIANCE:  no keyboard driving -- the operator DRAGS the robot in the
                        MuJoCo viewer (Ctrl+drag). The sim runs the whole-body
                        compliance law: arms + torso are gravity-compensated and
                        back-drivable, the base admits the drag wrench and glides,
                        and the lift follows a vertical pull. RViz mirrors the
                        state. This mode just tells the sim to engage (it publishes
                        /mabel/control_mode); all the physics happens sim-side.

Mode 1 controls (base -- arrows mirror the iOS app's LEFT joystick)
  Arrows  : translate  (Up/Down = forward/back,  Left/Right = strafe)
  A / D   : rotate left / right       W / S : lift up / down
Mode 2 controls
  W/S A/D R/F : LEFT  palm  X-/X+  Y-/Y+  Z+/Z-
  Z/X C/V B/N : LEFT  palm  rotate about X / Y / Z
  Arrows + T/G: RIGHT palm  X(Up/Dn) Y(L/R) Z(T/G)
  Y/H         : LEFT  hand  close / open       U/J : RIGHT hand close / open
Mode 3 controls
  (none -- drag the robot in the MuJoCo viewer; SPACE freezes / holds the pose)
All modes
  I / K : max linear speed +/-       O / L : max angular speed +/-
  TAB   : cycle mode                 SPACE : toggle E-STOP    Ctrl-C : quit

Mode 1 publishes ``mabel_interfaces/MabelSwerveCmd`` on ``/mabel_cmd`` by default, OR
``geometry_msgs/Twist`` on ``/cmd_vel`` (REP-103) when ``output:=cmd_vel`` -- the latter
runs through cmd_vel_bridge's tip-safe motion model, the same path Nav2 uses (this
subsumes the old standalone ``drive_keyboard`` node).

BOTH paths are tip-gated by the SAME model (``mabel.motion_model.safe_base_step``):
``/cmd_vel`` gets it downstream in cmd_vel_bridge, ``/mabel_cmd`` gets it HERE, because
that topic goes straight on to swerve_node -> the bridge -> the wheels and this node is
its last gate. That satisfies the rule in ``mabel/motion_model/__init__.py``: every base
twist from every input device passes ``safe_base_step`` before the swerve IK. Losing the
model (no controller/ checkout) is logged as an ERROR, never silently downgraded. Mode 2 additionally publishes joint
targets on ``/arms/command``, ``/body/command`` and ``/<side>_hand/command`` (wrist + the
grip-driven finger joints, Y/H + U/J), plus the lazy-base twist on ``/mabel_cmd`` -- the
same topics the real hardware consumes.

Mode 2 needs MuJoCo + the model as a kinematics engine (synced from
``/joint_states`` + ``/odom``); if the model cannot be loaded only Mode 1 runs.

Keyboard backend: uses `pynput` if installed (true hold-to-move). Otherwise
falls back to a pure-stdlib terminal reader (keys auto-decay ~0.25 s after the
last repeat, so holding still drives) -- no extra dependency, run it as-is.
Run with the terminal focused:  ros2 run mabel_teleop keyboard_teleop
"""

import os
import select
import sys
import termios
import tty

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy

from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
from mabel_interfaces.msg import MabelSwerveCmd

# Mode 2 (whole-body IK) reuses the SERVER's controller -- the canonical `mabel`
# package (controller/mabel), the SAME code /server runs -- so the ROS teleop and the
# server never drift. We load only the light controller modules (config, wbc,
# lazy_base) by registering a synthetic `mabel` package that bypasses the package's
# heavy AVP/zmq __init__. MuJoCo here is purely a kinematics / Jacobian engine.
try:
    import math
    import types
    import numpy as np
    import mujoco

    _CTRL = os.path.expanduser(os.environ.get("MABEL_CONTROLLER", "~/open_mabel/controller"))
    if "mabel" not in sys.modules:
        _pkg = types.ModuleType("mabel")
        _pkg.__path__ = [os.path.join(_CTRL, "mabel")]
        sys.modules["mabel"] = _pkg
    from mabel import config as _C
    from mabel.wbc import WholeBodyIK
    from mabel.lazy_base import LazyBase
    IK_ACTUATORS = list(_C.IK_ACTUATORS)
    _WBC_OK = True
except Exception as _e:                       # pragma: no cover
    _WBC_OK = False
    _WBC_ERR = _e


# ---------------------------------------------------------------------------
# THE TIP-SAFETY GATE (mabel.motion_model.safe_base_step) — imported SEPARATELY
# from the WBC block above on purpose. That block needs MuJoCo/numpy as a
# kinematics engine; the tip model needs neither. Sharing their `try` meant a
# machine without MuJoCo silently lost the SAFETY gate along with Mode 2 --
# exactly the input the gate exists to bound.
#
# The rule this satisfies (mabel/motion_model/__init__.py): "Every base twist,
# from every input device ... MUST pass through safe_base_step(...) before it
# reaches the swerve IK". Publishing a raw ramped twist on /mabel_cmd -- as this
# node did -- reaches swerve_node -> the bridge -> the wheels, ungated: fixed
# 2 m/s^2 / 5 rad/s^2 ramps that do not shrink as the lift rises.
# ---------------------------------------------------------------------------
try:
    import os as _os
    import sys as _sys
    import types as _types
    _CTRL_P = _os.path.expanduser(
        _os.environ.get("MABEL_CONTROLLER", "~/open_mabel/controller"))
    if "mabel" not in _sys.modules:
        _p = _types.ModuleType("mabel")
        _p.__path__ = [_os.path.join(_CTRL_P, "mabel")]
        _sys.modules["mabel"] = _p
    from mabel.motion_model import safe_base_step as _safe_base_step
    from mabel.motion_model import LIFT_RANGE as _MODEL_LIFT_RANGE
    _TIP_OK = True
except Exception as _te:                      # pragma: no cover
    _TIP_OK = False
    _TIP_ERR = _te
    _MODEL_LIFT_RANGE = None


def rotate_mat(mat, axis, angle):
    """Rotate a 3x3 orientation by `angle` about world `axis` (Rodrigues).

    A local keyboard-input helper for nudging the palm targets; the IK itself is
    the server's ``mabel.wbc.WholeBodyIK``."""
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return (np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)) @ mat

DEFAULT_MODEL = "~/open_mabel/simulation/mabel_mujoco/models/mabel_full.xml"
IK_POS_RATE = 0.5       # m/s   palm translate rate (Mode 2)
IK_ROT_RATE = 1.0       # rad/s palm rotate rate (Mode 2)
GRIP_RATE = 1.5         # 1/s   hand open<->close rate (Mode 2 Y/H + U/J); grip in [0,1]
WRIST_VIZ_DIAM = 0.06   # m     diameter of the Mode-2 green wrist-target balls (RViz)
# Mode-2 keys that move a palm target (position or orientation). The lazy base stays
# disarmed until one of these is pressed, so handing IK control back never moves the
# robot until the operator actually commands something.
M2_POSE_KEYS = frozenset({"w", "s", "a", "d", "r", "f", "z", "x", "c", "v", "b", "n",
                          "up", "down", "left", "right", "t", "g"})

# ORCA bare finger joint -> MuJoCo actuator suffix (mirror of
# mabel_sim.interface_config.ORCA_TO_MJ_SUFFIX). The teleop publishes the BARE names on
# /<side>_hand/command and the sim maps them to "Left Index MCP" etc.; we read each
# actuator's ctrlrange from the model to interpolate open(extended)->closed(fisted).
ORCA_TO_MJ_SUFFIX = {
    "index_abd": "Index Abd", "index_mcp": "Index MCP", "index_pip": "Index PIP",
    "middle_abd": "Middle Abd", "middle_mcp": "Middle MCP", "middle_pip": "Middle PIP",
    "ring_abd": "Ring Abd", "ring_mcp": "Ring MCP", "ring_pip": "Ring PIP",
    "pinky_abd": "Pinky Abd", "pinky_mcp": "Pinky MCP", "pinky_pip": "Pinky PIP",
    "thumb_cmc": "Thumb MCP", "thumb_abd": "Thumb Abd",
    "thumb_mp": "Thumb PIP", "thumb_ip": "Thumb DIP",
}
# Joints that curl on close (grip 0->1 = extended->flexed). The rest (abduction /
# thumb opposition) stay at a neutral spread so the open pose is a flat hand.
HAND_FLEXION = {"index_mcp", "index_pip", "middle_mcp", "middle_pip",
                "ring_mcp", "ring_pip", "pinky_mcp", "pinky_pip",
                "thumb_mp", "thumb_ip"}

# ── Limits / smoothing (mirror simulation/.../mabel/config.py) ───────────────
# Drive notes (measured on the real loaded base, single publisher):
#   * the SPARK velocity loop UNDER-REACHES to ~50% of commanded (the kF=0 steady-state
#     error), so cmd 0.4 -> ~0.21 m/s actual, cmd 1.0 -> ~0.34 m/s. It moves cleanly down
#     to cmd ~0.3 even under the 20 kg payload.
#   * ABOVE ~1.5 m/s commanded it COLLAPSES (wheels stall at high current) — so the cap
#     stays well below that. Real fix for faster/slower control = SPARK drive kFF +
#     current limit in REV Hardware Client. See docs/hardware_bridge.tex.
# Default is SLOW for safe driving; I/K trim it live up to LIN_SPEED_MAX.
MAX_LIN_SPEED = 0.4     # m/s   commanded (~0.21 m/s actual) — slow default
MAX_ANG_SPEED = 1.0     # rad/s
LIN_SPEED_STEP = 0.1    # m/s   per I/K press
ANG_SPEED_STEP = 0.2    # rad/s per O/L press
LIN_SPEED_MAX = 1.3     # m/s   HARD cap — stay below the ~1.5 m/s drive-collapse point
ANG_SPEED_MAX = 2.5     # rad/s
# Fallback ramp rates — used ONLY on the /cmd_vel path (cmd_vel_bridge applies the
# real tip model downstream) and if the tip model is unimportable. The /mabel_cmd
# path uses safe_base_step, whose caps SHRINK as the lift rises; these constants
# do not, which is why they must never be the gate on a path that reaches wheels.
MAX_LIN_ACCEL = 2.0     # m/s^2
MAX_ANG_ACCEL = 5.0     # rad/s^2
# Cascaded lift travel [m]. Single-sourced from the tip model so this node and the
# model cannot disagree about what "fully raised" means (they must agree: z_lift is
# the model's input). Only falls back to a literal if the controller is absent.
LIFT_RANGE = _MODEL_LIFT_RANGE if _MODEL_LIFT_RANGE is not None else (0.0, 0.635)
# Capped at the real-hardware cruise ceiling (60 mm/s = 0.06 m/s); see
# firmware/lift/firmware/config.py: MAX_VELOCITY_MM_S = 60.0.
LIFT_SPEED = 0.06           # m/s manual jog (hardware-limited)
HOLD_TIMEOUT = 0.25         # s; termios fallback "key still held" window


# ─────────────────────────────────────────────────────────────────────────────
# Terminal UI -- a fixed, in-place dashboard instead of a scrolling status line.
# All output redraws over the same screen region each tick (cursor-home + clear-
# to-EOL per line), so nothing "moves up and disappears". Plain ANSI only (no
# curses) so it works over SSH / inside `ros2 run`. Falls back to the old single
# `\r` line when stdout is not a TTY (e.g. piped logs under `ros2 launch`).
# ─────────────────────────────────────────────────────────────────────────────
class C:
    RESET = "\033[0m"; BOLD = "\033[1m"; DIM = "\033[2m"
    RED = "\033[91m"; GREEN = "\033[92m"; YELLOW = "\033[93m"
    BLUE = "\033[94m"; MAGENTA = "\033[95m"; CYAN = "\033[96m"; WHITE = "\033[97m"
    GREY = "\033[90m"


def _bar(frac, width=16, color=C.GREEN):
    """A filled/empty block bar for a 0..1 fraction."""
    frac = 0.0 if frac != frac else max(0.0, min(1.0, frac))
    n = int(round(frac * width))
    return color + "█" * n + C.GREY + "░" * (width - n) + C.RESET


def _sval(x, fmt="{:+5.2f}"):
    """A signed value, dimmed at ~zero so only what's moving stands out."""
    s = fmt.format(x)
    return (C.GREY + s + C.RESET) if abs(x) < 1e-3 else C.WHITE + s + C.RESET


# ─────────────────────────────────────────────────────────────────────────────
# Keyboard backends -> both expose .active() -> set of canonical key tokens
# tokens: 'up' 'down' 'left' 'right' 'space' 'tab' and single chars 'w','e',...
# ─────────────────────────────────────────────────────────────────────────────
class NullBackend:
    """No keyboard available (no TTY, no pynput) -- publishes idle commands."""
    name = "none (no TTY; idle only)"

    def active(self):
        return set()

    def close(self):
        pass


class PynputBackend:
    name = "pynput (hold-to-move)"

    def __init__(self):
        from pynput import keyboard
        self._k = keyboard
        self._pressed = set()
        self._special = {
            keyboard.Key.up: "up", keyboard.Key.down: "down",
            keyboard.Key.left: "left", keyboard.Key.right: "right",
            keyboard.Key.space: "space", keyboard.Key.tab: "tab",
        }
        self._listener = keyboard.Listener(on_press=self._on_press,
                                           on_release=self._on_release)
        self._listener.start()

    def _tok(self, key):
        if key in self._special:
            return self._special[key]
        try:
            return key.char.lower()
        except AttributeError:
            return None

    def _on_press(self, key):
        t = self._tok(key)
        if t:
            self._pressed.add(t)

    def _on_release(self, key):
        self._pressed.discard(self._tok(key))

    def active(self):
        return set(self._pressed)

    def close(self):
        self._listener.stop()


class TermiosBackend:
    name = "termios (stdlib fallback)"

    def __init__(self, clock):
        self._clock = clock                 # callable -> seconds (float)
        self._fd = sys.stdin.fileno()
        self._old = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        self._seen = {}                     # token -> last-seen time

    def _read_tokens(self):
        toks = []
        while select.select([sys.stdin], [], [], 0)[0]:
            buf = os.read(self._fd, 32)
            i = 0
            while i < len(buf):
                b = buf[i]
                if b == 0x03:               # Ctrl-C
                    raise KeyboardInterrupt
                if b == 0x1B and i + 2 < len(buf) and buf[i + 1] == ord("["):
                    toks.append({"A": "up", "B": "down",
                                 "C": "right", "D": "left"}.get(chr(buf[i + 2])))
                    i += 3
                    continue
                if b == ord(" "):
                    toks.append("space")
                elif b == ord("\t"):
                    toks.append("tab")
                elif 32 < b < 127:
                    toks.append(chr(b).lower())
                i += 1
        return [t for t in toks if t]

    def active(self):
        now = self._clock()
        for t in self._read_tokens():
            self._seen[t] = now
        return {t for t, ts in self._seen.items() if now - ts < HOLD_TIMEOUT}

    def close(self):
        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)


# ─────────────────────────────────────────────────────────────────────────────
class KeyboardTeleop(Node):

    def __init__(self):
        super().__init__("keyboard_teleop")

        self.declare_parameter("rate", 20.0)
        self.declare_parameter("model", DEFAULT_MODEL)
        # Mode-1 base output: "mabel_cmd" drives the base/sim directly (default), or
        # "cmd_vel" publishes Twist through cmd_vel_bridge's tip-safe model (Nav2 path).
        self.declare_parameter("output", "mabel_cmd")
        self.declare_parameter("cmd_topic", "/cmd_vel")
        # RViz fixed frame the Mode-2 wrist-target balls are drawn in. The IK targets
        # live in the model's world frame, which is the sim's /odom frame.
        self.declare_parameter("viz_frame", "odom")
        rate = float(self.get_parameter("rate").value)
        self.dt = 1.0 / rate
        self.output = str(self.get_parameter("output").value).lower()
        cmd_topic = str(self.get_parameter("cmd_topic").value)
        self.viz_frame = str(self.get_parameter("viz_frame").value)

        # /mabel_cmd is always created (Mode 2's lazy base uses it); the Twist publisher
        # is added only when driving the base via /cmd_vel.
        self.pub = self.create_publisher(MabelSwerveCmd, "/mabel_cmd", 10)
        self.pub_twist = (self.create_publisher(Twist, cmd_topic, 10)
                          if self.output == "cmd_vel" else None)
        # Mode-2 wrist-target balls for RViz (two green spheres at the palm targets).
        self.pub_wrist_viz = self.create_publisher(MarkerArray, "/wrist_targets", 10)
        self._wrist_viz_cleared = True
        # Whole-body control mode for the sim (Mode 3 compliance). Latched so a sim
        # that starts after the teleop still receives the current mode.
        _latched = QoSProfile(depth=1)
        _latched.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.pub_ctrl_mode = self.create_publisher(String, "/mabel/control_mode", _latched)

        # state
        self.max_lin = MAX_LIN_SPEED
        self.max_ang = MAX_ANG_SPEED
        self.vx = self.vy = self.omega = 0.0
        self.lift = LIFT_RANGE[0]
        self.estop = False
        self.prev_active = set()
        self.mode = 1
        self.grip_l = self.grip_r = 0.0     # Mode 2 hand close fraction [0..1] (0 = open)
        # Each hand's fingers stay UNCOMMANDED (held by the sim at their nav pose)
        # until the operator first presses that hand's grip key. Otherwise entering
        # Mode 2 would slam the fingers to the grip=0 "open" pose even though the user
        # never touched the hand -- the targets must not change on a mode switch alone.
        self._hand_armed = {"left": False, "right": False}
        self._mode_name = {1: "DIRECT NAV", 2: "SE(3) IK", 3: "COMPLIANCE"}
        # Mode-2 lazy base stays disarmed (no creep) until the operator gives input.
        self._m2_base_armed = False

        # dashboard state
        self._fancy = sys.stdout.isatty()
        self._drawn = False
        self._event = ""                    # transient banner (mode/e-stop changes)
        self._event_t = -1e9

        # ---- Mode 2 (whole-body IK) setup ---------------------------------
        self._ik_ready = self._init_wbc()
        if self._ik_ready:
            self.pub_arms = self.create_publisher(JointState, "/arms/command", 10)
            self.pub_body = self.create_publisher(JointState, "/body/command", 10)
            self.pub_lhand = self.create_publisher(JointState, "/left_hand/command", 10)
            self.pub_rhand = self.create_publisher(JointState, "/right_hand/command", 10)
            self._last_js = {}          # urdf joint name -> position
            self._last_odom = None      # (pos[3], quat_wxyz[4])
            self.create_subscription(JointState, "/joint_states", self._on_js, 10)
            self.create_subscription(Odometry, "/odom", self._on_odom, 10)

        # Modes available for TAB cycling. Mode 2 (whole-body IK) and Mode 3
        # (compliance) are native-only -- they drive /mabel_cmd + the sim, not /cmd_vel.
        self._modes = [1]
        if self.output == "mabel_cmd":
            if self._ik_ready:
                self._modes.append(2)
            self._modes.append(3)        # compliance is sim-side; needs no local model

        self.kb = self._make_backend()
        self.get_logger().info(f"keyboard backend: {self.kb.name}")
        # The /mabel_cmd twist reaches the wheels from here, so losing the tip model
        # on that path is a real safety downgrade (fixed ramps that ignore the lift),
        # not a missing nicety. Never silent.
        if self.output != "cmd_vel":
            if _TIP_OK:
                self.get_logger().info(
                    "tip-safe motion model ACTIVE on /mabel_cmd (lift-aware caps)")
            else:
                self.get_logger().error(
                    f"tip-safe motion model UNAVAILABLE ({_TIP_ERR}) — /mabel_cmd is "
                    f"driven by FIXED accel ramps that do NOT shrink as the lift "
                    f"rises. Set MABEL_CONTROLLER to the controller/ checkout, or "
                    f"run with output:=cmd_vel so cmd_vel_bridge gates the twist.")
        self._print_help()

        self.timer = self.create_timer(self.dt, self._tick)

    # ---- Mode 2 kinematics engine -------------------------------------------
    def _init_wbc(self):
        if not _WBC_OK:
            self.get_logger().warn(
                f"Mode 2 disabled (no MuJoCo/numpy: {_WBC_ERR}); Mode 1 only")
            return False
        path = os.path.expanduser(self.get_parameter("model").value)
        if not os.path.exists(path):
            self.get_logger().warn(f"Mode 2 disabled (model not found: {path})")
            return False
        try:
            self.model = mujoco.MjModel.from_xml_path(path)
            self.data = mujoco.MjData(self.model)
            mujoco.mj_forward(self.model, self.data)
            self.ik = WholeBodyIK(self.model, self.data)
            self.lazy = LazyBase()
            # URDF joint name -> MuJoCo joint name, for syncing /joint_states
            self._urdf2mj = {}
            for j in range(self.model.njnt):
                n = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
                if n:
                    self._urdf2mj[n.lower().replace(" ", "_")] = n
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "base_root")
            self._base_qadr = self.model.jnt_qposadr[bid] if bid >= 0 else None
            # Per-hand finger open/close targets (read from actuator ctrlrange) so
            # Y/H + U/J interpolate each finger extended<->fisted.
            self._hand_jt = self._build_hand_targets()
            return True
        except Exception as e:
            self.get_logger().warn(f"Mode 2 disabled (model load failed: {e})")
            return False

    def _build_hand_targets(self):
        """{side: [(orca_bare, open_val, closed_val), ...]} from the actuator ctrlrange.

        Flexion joints map grip 0->1 to (extended lo -> fisted hi); abduction /
        opposition joints stay at a neutral spread (clamped 0). The teleop publishes
        the BARE ORCA names; mujoco_node maps them to the "<Side> <Joint>" actuators."""
        out = {}
        for side in ("left", "right"):
            prefix = "Left" if side == "left" else "Right"
            rows = []
            for bare, suffix in ORCA_TO_MJ_SUFFIX.items():
                aid = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{prefix} {suffix}")
                if aid < 0:
                    continue
                lo, hi = (float(x) for x in self.model.actuator_ctrlrange[aid])
                if bare in HAND_FLEXION:
                    rows.append((bare, lo, hi))           # open = extended, close = fisted
                else:
                    neutral = min(max(0.0, lo), hi)       # neutral spread at both ends
                    rows.append((bare, neutral, neutral))
            out[side] = rows
        return out

    def _hand_cmd(self, side, wrist, grip, armed):
        """(name, value) list for /<side>_hand/command: the wrist (always, it's an IK
        DOF that holds its synced angle) plus the grip-driven fingers ONLY once the
        operator has actually gripped this hand -- until then the fingers are left out
        so the sim keeps holding them where nav left them (no snap on mode entry)."""
        out = [("wrist", wrist)]
        if armed:
            for bare, lo, hi in getattr(self, "_hand_jt", {}).get(side, ()):
                out.append((bare, lo + grip * (hi - lo)))
        return out

    def _on_js(self, msg: JointState):
        self._last_js = dict(zip(msg.name, msg.position))

    def _on_odom(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self._last_odom = ((p.x, p.y, p.z), (q.w, q.x, q.y, q.z))

    def _measured_lift(self):
        """The robot's CURRENT lift height in cascaded units (Lift Lower + Lift
        Upper, the same units the sim's `Lift` tendon actuator and `lift_pos` use),
        read from /joint_states -- or None if not yet known.

        `self.lift` is the SINGLE lift target carried across every mode. Snapping it
        to this on each mode switch mirrors the iOS/Vision Pro app's anti-jump rule
        ("the current state IS the new target"): the command the next mode publishes
        then equals where the lift already is, so the height never steps -- not on
        nav->wrist, and not when handing back after a compliance drag moved it."""
        js = getattr(self, "_last_js", None)
        if not js or "lift_lower" not in js:
            return None
        z = js["lift_lower"] + js.get("lift_upper", js["lift_lower"])
        return float(min(max(z, LIFT_RANGE[0]), LIFT_RANGE[1]))

    def _js_qpos(self, urdf):
        """Absolute MuJoCo qpos for a urdf joint from /joint_states, or None.

        The sim publishes (qpos - qpos0): URDF joint-zero is the assembled / ref
        pose, so a joint with a non-zero ``ref`` (the wrists carry ref=+-0.15, plus
        the thumb / abductions) reports its angle SHIFTED by that ref. Feeding the
        reported value back as a raw qpos therefore seeds the kinematic model 0.15
        rad off at the wrist -- so ik.sync_to_current() captures an offset wrist and
        the hand twitches (~8.6 deg) the instant Mode 2 takes over. Add qpos0 back to
        reconstruct the TRUE model qpos, so the seeded pose matches the real robot."""
        mj = self._urdf2mj.get(urdf)
        if mj is None or urdf not in self._last_js:
            return None
        qadr = self.model.joint(mj).qposadr[0]
        return self._last_js[urdf] + float(self.model.qpos0[qadr])

    def _sync_kin(self, feedforward=True):
        """Update the kinematic MjData and run FK.

        feedforward=True (per IK tick): the upper-body joints are set to the
        COMMANDED targets (ik.target_q), not the lagged /joint_states. The IK is
        then a resolved-rate controller on its own internal state -- stable
        regardless of the round-trip latency through the bridge. The base
        (from /odom) and lift (from /joint_states) are slow and stay measured.

        feedforward=False (on mode entry): set everything from the real measured
        state so ik.sync_to_current() captures the true pose (no jump)."""
        # base pose from odom (always measured)
        if self._last_odom is not None and self._base_qadr is not None:
            (px, py, pz), (qw, qx, qy, qz) = self._last_odom
            a = self._base_qadr
            self.data.qpos[a:a + 3] = (px, py, pz)
            self.data.qpos[a + 3:a + 7] = (qw, qx, qy, qz)

        if feedforward:
            for name in IK_ACTUATORS:
                if name == "Lift":
                    continue                       # lift stays measured (below)
                self.data.joint(name).qpos[0] = self.ik.target_q[name]
            for jn in ("lift_lower", "lift_upper"):   # measured lift
                q = self._js_qpos(jn)
                if q is not None:
                    self.data.joint(self._urdf2mj[jn]).qpos[0] = q
        else:
            for urdf in self._last_js:
                mj = self._urdf2mj.get(urdf)
                if mj is not None:
                    self.data.joint(mj).qpos[0] = self._js_qpos(urdf)
        mujoco.mj_forward(self.model, self.data)

    # ---- backend selection ----
    def _make_backend(self):
        # A real terminal (you ran `ros2 run` and are typing here) is the most reliable
        # input: read its stdin directly in cbreak mode. Only when there is NO
        # controlling TTY -- typically `ros2 launch`, which does not wire a terminal to
        # the node -- do we fall back to pynput's GLOBAL X listener, which only sees keys
        # the X server delivers to a focused window (so it misses keys typed into a
        # launch/SSH terminal). That is why interactive keyboard teleop wants `ros2 run`.
        if sys.stdin.isatty():
            return TermiosBackend(lambda: self.get_clock().now().nanoseconds * 1e-9)
        try:
            be = PynputBackend()
            self.get_logger().warn(
                "no TTY (ros2 launch): using pynput's GLOBAL listener -- it only "
                "captures keys when a focused X window receives them. If keys do "
                "nothing, run instead:  ros2 run mabel_teleop keyboard_teleop")
            return be
        except Exception as e:
            self.get_logger().error(
                "KEYBOARD DISABLED (idle only): launched without a TTY and pynput "
                f"unavailable ({e}). Run in a focused terminal:  "
                "ros2 run mabel_teleop keyboard_teleop")
            return NullBackend()

    def _print_help(self):
        # In fancy (TTY) mode the controls live in the live dashboard footer, so
        # don't scroll a static block that the first redraw would wipe anyway.
        if self._fancy:
            self._set_event(f"backend: {self.kb.name}", C.CYAN)
            return
        if self.output == "cmd_vel":
            m2 = "N/A (cmd_vel output)"
        else:
            m2 = "available (TAB)" if self._ik_ready else "DISABLED (no model)"
        m3 = "available (TAB)" if self.output == "mabel_cmd" else "N/A (cmd_vel output)"
        iface = "/cmd_vel (Twist)" if self.output == "cmd_vel" else "/mabel_cmd"
        print("\n=== MABEL KEYBOARD TELEOP ===")
        print(f"  Mode 1 base -> {iface}   |  Mode 2 SE(3) IK: {m2}   |  Mode 3 compliance: {m3}")
        print("  M1  Arrows=translate  A/D=rotate L/R  W/S=lift up/down")
        print("  M2  WSAD/RF=L-palm xyz  ZXCVBN=L-palm rot  Arrows+T/G=R-palm xyz")
        print("  M2  Y/H=L-hand close/open   U/J=R-hand close/open  (green balls = wrist targets)")
        print("  M3  drag the robot in the MuJoCo viewer (Ctrl+drag); no keyboard driving")
        print("  All  I/K=lin spd  O/L=ang spd  TAB=cycle mode  SPACE=E-STOP  Ctrl-C=quit")
        print("=============================\n")

    # ---- dashboard rendering -------------------------------------------------
    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _set_event(self, msg, color=C.WHITE):
        """Flash a transient banner under the panel (mode switch, e-stop, ...)."""
        self._event = color + msg + C.RESET
        self._event_t = self._now()

    def _draw(self, body):
        """Redraw the whole panel in place: header + body + footer + event.

        `body` is the per-mode list of pre-coloured strings. Each line is padded
        with a clear-to-EOL so shorter new lines don't leave old text behind, and
        the region below is cleared so the panel never scrolls."""
        backend = self.kb.name.split()[0]
        if self.output == "cmd_vel":
            m2 = "n/a"
        else:
            m2 = "ready" if self._ik_ready else "no model"
        head = [
            "",
            f"  {C.BOLD}{C.CYAN}M A B E L{C.RESET}  {C.DIM}keyboard teleop"
            f"{C.RESET}   {C.GREY}backend {backend} · mode-2 {m2}{C.RESET}",
            f"  {C.GREY}{'─' * 56}{C.RESET}",
        ]
        foot = [
            f"  {C.GREY}{'─' * 56}{C.RESET}",
        ] + self._footer()
        # transient event banner (fades after ~2.5 s)
        if self._event and self._now() - self._event_t < 2.5:
            foot.append(f"  {C.DIM}›{C.RESET} {self._event}")
        else:
            foot.append("")

        lines = head + body + foot
        out = []
        if not self._drawn:
            out.append("\033[2J\033[?25l")        # clear screen, hide cursor
            self._drawn = True
        out.append("\033[H")                       # cursor home
        for ln in lines:
            out.append(ln + "\033[K\n")            # line + clear to EOL
        out.append("\033[J")                       # clear everything below
        sys.stdout.write("".join(out))
        sys.stdout.flush()

    def _footer(self):
        if self.mode == 3:
            return [
                f"  {C.DIM}Ctrl+drag in the MuJoCo viewer{C.RESET} — no keyboard driving",
                f"  {C.DIM}SPACE{C.RESET} freeze/resume   {C.DIM}TAB{C.RESET} mode"
                f"   {C.DIM}^C{C.RESET} quit",
            ]
        if self.mode == 2:
            return [
                f"  {C.DIM}WSAD/RF{C.RESET} L-palm xyz   {C.DIM}ZXCVBN{C.RESET} L-palm rot"
                f"   {C.DIM}arrows/TG{C.RESET} R-palm xyz",
                f"  {C.DIM}Y/H U/J{C.RESET} hands   {C.DIM}I/K O/L{C.RESET} speed"
                f"   {C.DIM}TAB{C.RESET} mode   {C.DIM}SPACE{C.RESET} e-stop"
                f"   {C.DIM}^C{C.RESET} quit",
            ]
        return [
            f"  {C.DIM}arrows{C.RESET} move   {C.DIM}A/D{C.RESET} turn"
            f"   {C.DIM}W/S{C.RESET} lift   {C.DIM}I/K{C.RESET} lin   {C.DIM}O/L{C.RESET} ang",
            f"  {C.DIM}TAB{C.RESET} switch mode   {C.DIM}SPACE{C.RESET} e-stop"
            f"   {C.DIM}^C{C.RESET} quit",
        ]

    def _state_row(self):
        if self.estop:
            return (f"  {C.GREY}state{C.RESET}  {C.BOLD}{C.RED}▣ E-STOP{C.RESET}")
        return (f"  {C.GREY}state{C.RESET}  {C.BOLD}{C.GREEN}● RUN{C.RESET}")

    # ---- ramp a value toward target at max accel ----
    @staticmethod
    def _ramp(cur, target, accel, dt):
        if cur < target:
            return min(target, cur + accel * dt)
        return max(target, cur - accel * dt)

    def _tick(self):
        active = self.kb.active()
        new = active - self.prev_active            # edge-triggered keys
        self.prev_active = active
        dt = self.dt

        # edge events: speed trim + e-stop toggle
        if "space" in new:
            self.estop = not self.estop
            if self.estop:
                self._set_event("E-STOP ENGAGED", C.RED)
            else:
                self._set_event("E-STOP cleared", C.GREEN)
            if not self._fancy:
                print(f"\n[STATE] E-STOP {'ENGAGED' if self.estop else 'CLEARED'}")
        # I/K trim the MAX LINEAR speed live (works in every mode, incl. nav Mode 1);
        # O/L trim the max angular speed. The new cap takes effect on the next key-hold.
        if "i" in new:
            self.max_lin = min(LIN_SPEED_MAX, self.max_lin + LIN_SPEED_STEP)
            self._set_event(f"max linear speed → {self.max_lin:.1f} m/s", C.CYAN)
        if "k" in new:
            self.max_lin = max(0.1, self.max_lin - LIN_SPEED_STEP)
            self._set_event(f"max linear speed → {self.max_lin:.1f} m/s", C.CYAN)
        if "o" in new:
            self.max_ang = min(ANG_SPEED_MAX, self.max_ang + ANG_SPEED_STEP)
            self._set_event(f"max angular speed → {self.max_ang:.1f} rad/s", C.CYAN)
        if "l" in new:
            self.max_ang = max(0.2, self.max_ang - ANG_SPEED_STEP)
            self._set_event(f"max angular speed → {self.max_ang:.1f} rad/s", C.CYAN)

        # TAB: cycle to the next available mode (edge-triggered).
        if "tab" in new and len(self._modes) > 1:
            i = self._modes.index(self.mode) if self.mode in self._modes else 0
            self.mode = self._modes[(i + 1) % len(self._modes)]
            # Carry the lift across the switch: snap the one shared lift target to the
            # robot's ACTUAL height so the next mode re-commands exactly where the lift
            # already is (no step). This is the lift's half of the iOS "current state
            # IS the new target" handover -- it also picks up a height the operator
            # changed by dragging in compliance (Mode 3), so handing back to nav/IK
            # continues from there instead of snapping to a stale jog value.
            ml = self._measured_lift()
            if ml is not None:
                self.lift = ml
            if self.mode == 2:
                # Adopt the CURRENT (possibly compliance-dragged) pose as the desired
                # pose: sync the kinematic state to the measured robot, snap the palm
                # targets there, and freeze the null-space posture to it -- so the
                # robot holds exactly where it is and does not move until the operator
                # presses a key. The lazy base is likewise disarmed until first input.
                self._sync_kin(feedforward=False)   # measured state (incl. dragged pose)
                self.ik.sync_to_current(freeze_posture=True)
                self._m2_base_armed = False
            if self.mode != 2:
                self._clear_wrist_viz()             # drop the Mode-2 balls
            self._set_event(f"switched to MODE {self.mode} · {self._mode_name[self.mode]}",
                            C.MAGENTA)
            if not self._fancy:
                print(f"\n[MODE] {self.mode}  {self._mode_name[self.mode]}")

        if self.mode == 3:
            self._tick_mode3(active, dt)
        elif self.mode == 2 and self._ik_ready:
            self._tick_mode2(active, dt)
        else:
            self._tick_mode1(active, dt)

        # Tell the sim which whole-body control mode to run. Compliance only while
        # in Mode 3 AND not e-stopped; otherwise the sim holds position (and, on
        # e-stop in Mode 3, freezes the dragged pose).
        cm = String()
        cm.data = "compliance" if (self.mode == 3 and not self.estop) else "position"
        self.pub_ctrl_mode.publish(cm)

    # ---- Mode-2 wrist-target balls (RViz) -----------------------------------
    def _publish_wrist_viz(self, p_l, p_r):
        """Publish two green spheres at the left/right palm (wrist) IK targets."""
        arr = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        for i, p in enumerate((p_l, p_r)):
            m = Marker()
            m.header.frame_id = self.viz_frame
            m.header.stamp = stamp
            m.ns = "wrist_targets"
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(p[0])
            m.pose.position.y = float(p[1])
            m.pose.position.z = float(p[2])
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = WRIST_VIZ_DIAM
            m.color.r, m.color.g, m.color.b, m.color.a = 0.10, 0.90, 0.20, 0.85
            arr.markers.append(m)
        self.pub_wrist_viz.publish(arr)
        self._wrist_viz_cleared = False

    def _clear_wrist_viz(self):
        """Delete the wrist balls (called when leaving Mode 2) so they don't linger."""
        if self._wrist_viz_cleared:
            return
        arr = MarkerArray()
        for i in range(2):
            m = Marker()
            m.header.frame_id = self.viz_frame
            m.ns = "wrist_targets"
            m.id = i
            m.action = Marker.DELETE
            arr.markers.append(m)
        self.pub_wrist_viz.publish(arr)
        self._wrist_viz_cleared = True

    # ---------------------------------------------------------- Mode 1: nav
    def _tick_mode1(self, active, dt):
        # Base mapping (internal twist is in the sim frame: forward = vx<0, strafe-left =
        # vy<0, omega CCW+). Arrows mirror the iOS app's LEFT joystick; A/D rotate; W/S lift.
        tvx = tvy = tw = 0.0
        if not self.estop:
            if "up" in active:    tvx -= self.max_lin   # forward
            if "down" in active:  tvx += self.max_lin   # back
            if "left" in active:  tvy -= self.max_lin   # strafe left
            if "right" in active: tvy += self.max_lin   # strafe right
            if "a" in active:     tw += self.max_ang    # rotate left (CCW)
            if "d" in active:     tw -= self.max_ang    # rotate right (CW)
            if "w" in active:
                self.lift = min(LIFT_RANGE[1], self.lift + LIFT_SPEED * dt)  # lift up
            if "s" in active:
                self.lift = max(LIFT_RANGE[0], self.lift - LIFT_SPEED * dt)  # lift down

        # Smooth toward the target (zeros out when keys released / estopped).
        #
        # /mabel_cmd path: the twist goes straight to swerve_node -> the bridge ->
        # the wheels, so THIS node is the last place the tip model can run. Use it:
        # safe_base_step's per-axis caps shrink with lift height and it projects the
        # combined target into the ZMP envelope. The old fixed-accel _ramp did
        # neither, so a raised keyboard-driven robot could be commanded to tip.
        #
        # /cmd_vel path: cmd_vel_bridge applies the same model downstream, so a plain
        # ramp here is correct — gating twice would just double-ramp the response.
        if self.output != "cmd_vel" and _TIP_OK:
            self.vx, self.vy, self.omega = _safe_base_step(
                self.vx, self.vy, self.omega, tvx, tvy, tw, dt, float(self.lift))
        else:
            self.vx = self._ramp(self.vx, tvx, MAX_LIN_ACCEL, dt)
            self.vy = self._ramp(self.vy, tvy, MAX_LIN_ACCEL, dt)
            self.omega = self._ramp(self.omega, tw, MAX_ANG_ACCEL, dt)

        if self.output == "cmd_vel":
            # REP-103 Twist (forward=+x, left=+y, CCW+): flip the sim-frame twist.
            # cmd_vel_bridge applies the tip-safe model; lift has no /cmd_vel channel.
            tw_msg = Twist()
            tw_msg.linear.x = -float(self.vx)
            tw_msg.linear.y = -float(self.vy)
            tw_msg.angular.z = float(self.omega)
            self.pub_twist.publish(tw_msg)
        else:
            msg = MabelSwerveCmd()
            msg.vx, msg.vy, msg.omega = float(self.vx), float(self.vy), float(self.omega)
            msg.lift_pos = float(self.lift)
            msg.hardware_state = 1 if self.estop else 0
            self.pub.publish(msg)

        iface = "/cmd_vel" if self.output == "cmd_vel" else "/mabel_cmd"
        if self._fancy:
            lo, hi = LIFT_RANGE
            lift_frac = (self.lift - lo) / (hi - lo) if hi > lo else 0.0
            body = [
                f"  {C.GREY}mode {C.RESET} {C.BOLD}{C.WHITE}❶ DIRECT NAV{C.RESET}"
                f"   {C.GREY}→ {iface}{C.RESET}",
                self._state_row(),
                "",
                f"  {C.GREY}speed{C.RESET}  lin {C.WHITE}{self.max_lin:4.1f}{C.RESET} m/s"
                f"   ang {C.WHITE}{self.max_ang:4.1f}{C.RESET} rad/s",
                f"  {C.GREY}twist{C.RESET}  vx {_sval(self.vx)}   vy {_sval(self.vy)}"
                f"   ω {_sval(self.omega)}",
                f"  {C.GREY}lift {C.RESET}  {C.WHITE}{self.lift:5.3f}{C.RESET} m  "
                f"{_bar(lift_frac)} {C.GREY}{hi:.3f}{C.RESET}",
            ]
            self._draw(body)
        else:
            flag = "E-STOP" if self.estop else "  RUN "
            sys.stdout.write(
                f"\r[M1 {flag} {iface}] lin{self.max_lin:4.1f} ang{self.max_ang:4.1f} | "
                f"vx{self.vx:+5.2f} vy{self.vy:+5.2f} w{self.omega:+5.2f} | "
                f"lift{self.lift:5.3f}m   ")
            sys.stdout.flush()

    # ---------------------------------------------------- Mode 2: SE(3) IK
    def _tick_mode2(self, active, dt):
        self._sync_kin()                       # FK/Jacobian at the real state
        ik = self.ik

        if not self.estop:
            s = IK_POS_RATE * dt
            r = IK_ROT_RATE * dt
            # LEFT palm position
            if "w" in active: ik.target_l_pos[0] -= s
            if "s" in active: ik.target_l_pos[0] += s
            if "a" in active: ik.target_l_pos[1] -= s
            if "d" in active: ik.target_l_pos[1] += s
            if "r" in active: ik.target_l_pos[2] += s
            if "f" in active: ik.target_l_pos[2] -= s
            # LEFT palm orientation
            if "z" in active: ik.target_l_mat = rotate_mat(ik.target_l_mat, [1, 0, 0],  r)
            if "x" in active: ik.target_l_mat = rotate_mat(ik.target_l_mat, [1, 0, 0], -r)
            if "c" in active: ik.target_l_mat = rotate_mat(ik.target_l_mat, [0, 1, 0],  r)
            if "v" in active: ik.target_l_mat = rotate_mat(ik.target_l_mat, [0, 1, 0], -r)
            if "b" in active: ik.target_l_mat = rotate_mat(ik.target_l_mat, [0, 0, 1],  r)
            if "n" in active: ik.target_l_mat = rotate_mat(ik.target_l_mat, [0, 0, 1], -r)
            # RIGHT palm position
            if "up" in active:    ik.target_r_pos[0] -= s
            if "down" in active:  ik.target_r_pos[0] += s
            if "left" in active:  ik.target_r_pos[1] -= s
            if "right" in active: ik.target_r_pos[1] += s
            if "t" in active:     ik.target_r_pos[2] += s
            if "g" in active:     ik.target_r_pos[2] -= s
            # hands: Y/H = LEFT close/open, U/J = RIGHT close/open (grip 0..1)
            gs = GRIP_RATE * dt
            if active & {"y", "h"}: self._hand_armed["left"] = True
            if active & {"u", "j"}: self._hand_armed["right"] = True
            if "y" in active: self.grip_l = min(1.0, self.grip_l + gs)
            if "h" in active: self.grip_l = max(0.0, self.grip_l - gs)
            if "u" in active: self.grip_r = min(1.0, self.grip_r + gs)
            if "j" in active: self.grip_r = max(0.0, self.grip_r - gs)

        # Track the palms with the ARMS + WRISTS only; the lift and torso are HELD at
        # their entry pose -- the very same parts the robot kept still during nav. So
        # the controller holding lift/torso/neck is identical in nav and in wrist
        # tracking, and handing control back and forth never steps them. The lift uses
        # the shared self.lift (cascaded units, carried from nav); reach beyond the
        # arms' range still comes from the lazy base creeping below.
        ik.apply(ik.solve(arms_only=True), dt, arms_only=True)
        # Arm the lazy base only once the operator actually commands a palm target;
        # until then the base holds (no creep on handover from nav / compliance).
        if active & M2_POSE_KEYS:
            self._m2_base_armed = True
        tvx, tvy, tw = self.lazy.command(ik.target_l_pos, ik.target_r_pos, self.data) \
            if (self._m2_base_armed and not self.estop) else (0.0, 0.0, 0.0)
        self._publish_mode2(tvx, tvy, tw)
        # Two green balls in RViz at the live left/right wrist (palm) targets.
        self._publish_wrist_viz(ik.target_l_pos, ik.target_r_pos)

        el = float(np.linalg.norm(ik.e_pos_l))
        er = float(np.linalg.norm(ik.e_pos_r))
        if self._fancy:
            lo, hi = LIFT_RANGE
            lift = float(self.lift)             # held lift target (carried from nav)
            lift_frac = (lift - lo) / (hi - lo) if hi > lo else 0.0
            ecol = lambda e: (C.RED if e > 0.02 else C.GREEN)
            body = [
                f"  {C.GREY}mode {C.RESET} {C.BOLD}{C.WHITE}❷ SE(3) IK{C.RESET}",
                self._state_row(),
                "",
                f"  {C.GREY}base {C.RESET}  vx {_sval(tvx)}   vy {_sval(tvy)}"
                f"   ω {_sval(tw)}",
                f"  {C.GREY}lift {C.RESET}  {C.WHITE}{lift:5.3f}{C.RESET} m  "
                f"{_bar(lift_frac)} {C.GREY}{hi:.3f}{C.RESET}",
                f"  {C.GREY}error{C.RESET}  L {ecol(el)}{el:.3f}{C.RESET} m"
                f"   R {ecol(er)}{er:.3f}{C.RESET} m",
                f"  {C.GREY}grip {C.RESET}  L {_bar(self.grip_l, 10, C.CYAN)} "
                f"{C.WHITE}{self.grip_l:3.0%}{C.RESET}   "
                f"R {_bar(self.grip_r, 10, C.CYAN)} {C.WHITE}{self.grip_r:3.0%}{C.RESET}",
            ]
            self._draw(body)
        else:
            flag = "E-STOP" if self.estop else "  RUN "
            sys.stdout.write(
                f"\r[M2 {flag}] base({tvx:+.2f},{tvy:+.2f},{tw:+.2f}) "
                f"lift{self.lift:5.3f}m | Lerr{el:.3f} Rerr{er:.3f} | "
                f"grip L{self.grip_l:3.0%} R{self.grip_r:3.0%}   ")
            sys.stdout.flush()

    def _publish_mode2(self, tvx, tvy, tw):
        ik = self.ik
        # base twist (lazy) + lift on /mabel_cmd
        base = MabelSwerveCmd()
        base.vx, base.vy, base.omega = float(tvx), float(tvy), float(tw)
        # The lift is HELD at the shared cascaded target (carried from nav), never the
        # IK's lower-joint value -- which is half the tendon's units and would drop the
        # lift to half height on the nav->wrist handover.
        base.lift_pos = float(self.lift)
        base.hardware_state = 1 if self.estop else 0
        self.pub.publish(base)

        def js(names_vals):
            m = JointState()
            m.header.stamp = self.get_clock().now().to_msg()
            for n, v in names_vals:
                m.name.append(n)
                m.position.append(float(v))
            return m

        # arms (urdf-style names accepted by the bridge / arms hardware)
        arms = [(f"left_arm_{i}", ik.target_q[f"Left Arm {i}"]) for i in range(1, 8)] + \
               [(f"right_arm_{i}", ik.target_q[f"Right Arm {i}"]) for i in range(1, 8)]
        self.pub_arms.publish(js(arms))
        self.pub_body.publish(js([("torso", ik.target_q["Torso"])]))
        # wrists + grip-driven fingers on the ORCA hand topics (bare ORCA names)
        self.pub_lhand.publish(js(self._hand_cmd(
            "left", ik.target_q["Left Wrist"], self.grip_l, self._hand_armed["left"])))
        self.pub_rhand.publish(js(self._hand_cmd(
            "right", ik.target_q["Right Wrist"], self.grip_r, self._hand_armed["right"])))

    # ------------------------------------------------- Mode 3: compliance
    def _tick_mode3(self, active, dt):
        """Compliance: no keyboard driving. The operator drags the robot in the
        MuJoCo viewer (Ctrl+drag); the sim runs the whole-body compliance law
        (gravity-comp + back-drivable arms/torso, base + lift admittance) and RViz
        mirrors the state. Here we only park the base command and render the panel;
        the engage signal goes out on /mabel/control_mode from _tick()."""
        self._clear_wrist_viz()
        # Park the base twist so nothing stale drives the wheels while the sim's
        # compliance engine owns the base. hardware_state carries the e-stop.
        stop = MabelSwerveCmd()
        stop.hardware_state = 1 if self.estop else 0
        self.pub.publish(stop)

        if self._fancy:
            estop_row = (f"  {C.GREY}estop{C.RESET}  {C.BOLD}{C.RED}FROZEN{C.RESET}"
                         f" {C.GREY}— SPACE to resume{C.RESET}") if self.estop else \
                        (f"  {C.GREY}estop{C.RESET}  {C.DIM}SPACE freezes & holds the pose{C.RESET}")
            body = [
                f"  {C.GREY}mode {C.RESET} {C.BOLD}{C.WHITE}❸ COMPLIANCE{C.RESET}"
                f"   {C.GREY}→ /mabel/control_mode{C.RESET}",
                self._state_row(),
                "",
                f"  {C.GREY}drag {C.RESET}  {C.WHITE}Ctrl+drag{C.RESET} any arm/body in the "
                f"{C.WHITE}MuJoCo viewer{C.RESET}",
                f"  {C.GREY}feel {C.RESET}  arms + torso {C.GREEN}gravity-comp & back-drivable{C.RESET}",
                f"  {C.GREY}body {C.RESET}  base glides under the drag · lift follows a vertical pull",
                f"  {C.GREY}rviz {C.RESET}  mirrors the live state",
                estop_row,
            ]
            self._draw(body)
        else:
            flag = "E-STOP" if self.estop else "  RUN "
            sys.stdout.write(
                f"\r[M3 {flag} COMPLIANCE] drag the robot in the MuJoCo viewer "
                f"(Ctrl+drag); RViz mirrors the state   ")
            sys.stdout.flush()

    def close(self):
        # stop the base, hand the sim back to position control, clear the balls,
        # then restore the terminal.
        try:
            stop = MabelSwerveCmd()
            self.pub.publish(stop)
            pos = String()
            pos.data = "position"
            self.pub_ctrl_mode.publish(pos)
            self._clear_wrist_viz()
        except Exception:
            pass
        self.kb.close()
        if self._fancy and self._drawn:
            # show the cursor again and drop below the panel so the shell prompt
            # returns on a clean line instead of overwriting the dashboard.
            sys.stdout.write("\033[?25h\n")
            sys.stdout.flush()


def main(args=None):
    rclpy.init(args=args)
    node = KeyboardTeleop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print()


if __name__ == "__main__":
    main()
