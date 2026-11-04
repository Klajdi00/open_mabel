#!/usr/bin/env python3
"""Live single-hand status dashboard -- the keyboard_teleop look, for one hand.

Subscribes to a hand's feedback (``/<side>_hand/joint_states`` + ``temperatures``
+ ``voltages``) and paints a compact, in-place terminal panel of all 17 motors:
motor ID, joint, position (with a bar inside its calibrated range), velocity,
torque (normalised effort), and temperature -- so you see at a glance exactly
what the hardware is doing.

It writes straight to ``/dev/tty`` (like mabel_sim's bringup_status), so the
in-place redraw works cleanly even under ``ros2 launch`` where stdout is piped.
No hardware access -- it only reads the driver's ROS topics and the YAML config
(for joint order + angle ranges).
"""

import os
import termios
import time
import tty

import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64MultiArray

from orcahand.config import HandConfig

CSI = "\033["


class C:
    RESET = "\033[0m"; BOLD = "\033[1m"; DIM = "\033[2m"
    RED = "\033[91m"; GREEN = "\033[92m"; YELLOW = "\033[93m"
    BLUE = "\033[94m"; MAGENTA = "\033[95m"; CYAN = "\033[96m"; WHITE = "\033[97m"
    GREY = "\033[90m"
    ON_GREEN = "\033[42m"; ON_YELLOW = "\033[43m"; ON_GREY = "\033[100m"; BLACK = "\033[30m"


def _bar(frac, width=10, color=C.CYAN):
    frac = 0.0 if frac != frac else max(0.0, min(1.0, frac))
    n = int(round(frac * width))
    return color + "█" * n + C.GREY + "░" * (width - n) + C.RESET


def _temp_col(t):
    return C.GREEN if t < 45 else (C.YELLOW if t < 60 else C.RED)


def _tor_col(x):
    a = abs(x)
    return C.GREY if a < 0.05 else (C.WHITE if a < 0.3 else (C.YELLOW if a < 0.6 else C.RED))


def _dimzero(x, fmt="{:+5.2f}"):
    s = fmt.format(x)
    return (C.GREY + s + C.RESET) if abs(x) < 1e-3 else C.WHITE + s + C.RESET


class HandStatus(Node):

    def __init__(self):
        super().__init__("hand_status")
        self.declare_parameter("side", "left")
        self.declare_parameter("config_file", "")
        self.declare_parameter("refresh", 12.0)

        self.side = self.get_parameter("side").value
        self.prefix = f"{self.side}_"
        cfg_path = self.get_parameter("config_file").value
        if not cfg_path:
            share = get_package_share_directory("mabel_hand_hardware")
            cfg_path = os.path.join(share, "config", f"{self.side}_hand.yaml")

        cfg = HandConfig.load(config_path=cfg_path)
        # ordered (name, id, angle_min, angle_max) -- the canonical joint order
        self.joints = [(j.name, j.motor_id, j.angle_min, j.angle_max) for j in cfg.joints]
        self.n = len(self.joints)

        self.js = {}        # name -> (pos, vel, eff) from the latest message
        self.temp = {}      # name -> degC
        self.volt = {}      # name -> V
        self._last = 0.0    # monotonic of last joint_states
        self._hz = 0.0      # EMA of joint_states rate
        self._prev_cb = None

        self.create_subscription(JointState, f"/{self.side}_hand/joint_states",
                                 self._on_js, 10)
        self.create_subscription(Float64MultiArray, f"/{self.side}_hand/temperatures",
                                 lambda m: self._on_arr(m, self.temp), 10)
        self.create_subscription(Float64MultiArray, f"/{self.side}_hand/voltages",
                                 lambda m: self._on_arr(m, self.volt), 10)

        # Torque mode: latched state in, toggle command out (Tab key).
        latched = QoSProfile(depth=1)
        latched.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self._torque_on = None      # None = unknown until first torque_state
        self.create_subscription(Bool, f"/{self.side}_hand/torque_state",
                                 self._on_torque, latched)
        self.pub_torque = self.create_publisher(
            Bool, f"/{self.side}_hand/torque_enable", 10)

        self._tty = None
        self._kfd = None            # raw keyboard fd on the controlling terminal
        self._termios_saved = None
        try:
            self._tty = open("/dev/tty", "w")
            self._w(CSI + "?25l")        # hide cursor while we own the screen
            self._w(CSI + "2J")          # clear once
            # cbreak keeps ISIG, so ^C still quits; we only steal raw keystrokes.
            self._kfd = os.open("/dev/tty", os.O_RDONLY | os.O_NONBLOCK)
            self._termios_saved = termios.tcgetattr(self._kfd)
            tty.setcbreak(self._kfd)
        except OSError:
            self.get_logger().info("no /dev/tty -- status dashboard disabled "
                                   "(driver still runs; echo the topics instead).")
        self.create_timer(1.0 / max(float(self.get_parameter("refresh").value), 1.0),
                          self._render)
        self.create_timer(0.02, self._poll_keys)   # snappy Tab response

    # ------------------------------------------------------------------ inputs
    def _bare(self, name):
        return name[len(self.prefix):] if name.startswith(self.prefix) else name

    def _on_js(self, msg: JointState):
        d = {}
        for i, name in enumerate(msg.name):
            p = msg.position[i] if i < len(msg.position) else 0.0
            v = msg.velocity[i] if i < len(msg.velocity) else 0.0
            e = msg.effort[i] if i < len(msg.effort) else 0.0
            d[self._bare(name)] = (p, v, e)
        self.js = d
        now = time.monotonic()
        if self._prev_cb is not None:
            dt = now - self._prev_cb
            if dt > 0:
                inst = 1.0 / dt
                self._hz = inst if self._hz == 0 else 0.8 * self._hz + 0.2 * inst
        self._prev_cb = now
        self._last = now

    def _on_arr(self, msg: Float64MultiArray, store):
        dims = msg.layout.dim
        for i, v in enumerate(msg.data):
            label = dims[i].label if i < len(dims) else str(i)
            store[self._bare(label)] = v

    def _on_torque(self, msg: Bool):
        self._torque_on = bool(msg.data)

    # ------------------------------------------------------------------ keys
    def _poll_keys(self):
        if self._kfd is None:
            return
        try:
            data = os.read(self._kfd, 64)
        except (BlockingIOError, OSError):
            return
        if b"\t" in data:                       # Tab -> flip torque mode
            target = not bool(self._torque_on)  # default OFF if state unknown
            self.pub_torque.publish(Bool(data=target))

    def _mode_banner(self):
        """A full-width colored bar showing the live torque mode + Tab hint."""
        if self._torque_on is None:
            tag = (f"{C.BOLD}{C.BLACK}{C.ON_GREY}  …  waiting for hand  {C.RESET}")
            hint = f"{C.GREY}[Tab] toggle{C.RESET}"
        elif self._torque_on:
            tag = (f"{C.BOLD}{C.BLACK}{C.ON_GREEN}  ⚙  TORQUE ENGAGED  —  motors "
                   f"holding position  {C.RESET}")
            hint = f"{C.GREY}[Tab] → free-drive{C.RESET}"
        else:
            tag = (f"{C.BOLD}{C.BLACK}{C.ON_YELLOW}  ✋ FREE-DRIVE  —  torque off, move "
                   f"joints freely by hand  {C.RESET}")
            hint = f"{C.GREY}[Tab] → engage{C.RESET}"
        return f"{tag}   {hint}"

    # ------------------------------------------------------------------ render
    def _w(self, s):
        if self._tty:
            self._tty.write(s); self._tty.flush()

    def _render(self):
        if not self._tty:
            return
        age = time.monotonic() - self._last
        live = len(self.js)
        fresh = (live > 0 and age < 0.5)
        if fresh:
            state = f"{C.GREEN}● LIVE{C.RESET}"
        elif live > 0:
            state = f"{C.YELLOW}○ STALE{C.RESET}"
        else:
            state = f"{C.RED}○ WAITING for /{self.side}_hand …{C.RESET}"
        vbus = (sum(self.volt.values()) / len(self.volt)) if self.volt else 0.0

        L = []
        L.append("")
        L.append(f"  {C.BOLD}{C.CYAN}M A B E L{C.RESET}  {C.BOLD}{self.side.upper()} hand"
                 f"{C.RESET}   {C.GREY}{live}/{self.n} motors · {self._hz:4.0f} Hz · "
                 f"{vbus:4.1f} V{C.RESET}   {state}")
        L.append(f"  {C.GREY}{'─' * 60}{C.RESET}")
        L.append("  " + self._mode_banner())
        L.append(f"  {C.GREY}{'─' * 60}{C.RESET}")
        L.append(f"  {C.DIM}ID  joint        position"
                 f"{'':9}vel    tor    temp{C.RESET}")
        for name, mid, amin, amax in self.joints:
            if name in self.js:
                p, v, e = self.js[name]
                frac = (p - amin) / (amax - amin) if amax != amin else 0.0
                t = self.temp.get(name)
                tcol = _temp_col(t) if t is not None else C.GREY
                tstr = f"{tcol}{t:4.0f}°{C.RESET}" if t is not None else f"{C.GREY}  --{C.RESET}"
                L.append(
                    f"  {C.GREY}{mid:>2}{C.RESET}  {name:<11}"
                    f" {C.WHITE}{p:+5.2f}{C.RESET} {_bar(frac)}"
                    f"  {_dimzero(v)}  {_tor_col(e)}{e:+5.2f}{C.RESET}"
                    f"  {tstr}")
            else:
                L.append(f"  {C.GREY}{mid:>2}  {name:<11} {'· no response':<36}{C.RESET}")
        L.append(f"  {C.GREY}{'─' * 60}{C.RESET}")
        L.append(f"  {C.BOLD}[Tab]{C.RESET}{C.DIM} engage ⇄ free-drive · pos bar = within "
                 f"calibrated range · tor = norm torque −1..1 · ^C quit{C.RESET}")

        out = [CSI + "H"]                 # cursor home
        for ln in L:
            out.append(ln + CSI + "K\n")  # clear to EOL
        out.append(CSI + "J")            # clear below
        self._w("".join(out))

    def shutdown(self):
        if self._kfd is not None:
            try:
                if self._termios_saved is not None:
                    termios.tcsetattr(self._kfd, termios.TCSADRAIN, self._termios_saved)
                os.close(self._kfd)
            except OSError:
                pass
        if self._tty:
            try:
                self._w(CSI + "?25h\n")   # restore cursor
                self._tty.close()
            except OSError:
                pass


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = HandStatus()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.shutdown()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
