"""lift_node — vertical-column lift DDS adapter (Pico DC motor over USB serial).

Thin plumbing over :class:`mabel_hw.subsystems.lift.LiftSubsystem` (which wraps
the proven ``firmware/lift/host`` metres API; the Pico runs the closed loop).
Contract:

  Subscribed:  <command>       std_msgs/Float64        target height [m] (STROKE)
               <calibrate>     std_msgs/Bool           true = stall-seek floor+ceiling
  Published:   <joint_states>  sensor_msgs/JointState  lift position [m] (STROKE) + velocity [m/s]
               <limits>        std_msgs/String         {min,max,calibrated} (latched)

STROKE vs ABSOLUTE datum — the conversion this node exists to hide
-----------------------------------------------------------------
The ROS ``/lift/*`` contract is the URDF's STROKE convention: 0 m == the floor,
``span`` m (~0.635) == fully raised. That is what the single-source-of-truth MJCF/URDF
models (two prismatic stages summing to the stroke), and therefore what the gateway,
the web console, and the iOS / Vision Pro apps all command and render.

The lift FIRMWARE, however, speaks ABSOLUTE height: the floor datum is
``LIFT_MIN_HEIGHT_MM`` (0.3175 m) and the ceiling ``+LIFT_TRAVEL_MM`` (0.9525 m), and
``lift_to_height`` CLAMPS every target to ``[min_pos, max_pos]`` in that absolute frame.
Sending a stroke command straight through therefore breaks it: a stroke of 0.20 m is
below the absolute floor, so the firmware clamps it up to 0.3175 m and the column never
moves — the "lift stalls from the web GUI / apps but works from the setup studio" bug
(the setup studio talks to the driver DIRECTLY, in absolute metres, so it is immune).

This adapter is the seam between the device (absolute) and the ROS contract (stroke), so
it converts BOTH directions here — command ``+floor`` on the way down, feedback ``-floor``
on the way up — keeping every ``/lift/*`` consumer in one consistent stroke frame while
the driver and the setup studio keep their native absolute units. ``floor`` is the
firmware's live ``min_pos`` (LiftState), so a recalibration that moves the datum is
tracked automatically.

With no Pico on the serial port the node advertises the contract and stays idle.
"""

import json
import math

from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import Bool, Float64, String
from sensor_msgs.msg import JointState

from ..errors import HardwareError, SubsystemUnavailable
from ..subsystems.lift import LiftSubsystem as CoreLift
from ._util import add_publisher_watchdog


def build(cfg, hw=None, **_):
    return LiftNode(cfg, hw=hw)


class LiftNode(Node):
    def __init__(self, cfg, hw=None):
        super().__init__("mabel_hw_lift")
        self.cfg = cfg
        self.joint = cfg["joint_name"]
        self._writer = None
        log = self.get_logger()

        if hw is not None and hw.lift.available:
            self.core = hw.lift._sub
            self._writer = hw.lift.claim(f"dds:{cfg['topic_command']}",
                                         steal=False)
        else:
            self.core = CoreLift(cfg, logger=lambda m: log.info(m))
            try:
                self.core.connect()
                log.info(f"lift connected on {cfg['port']} @ {cfg['baud']}")
            except SubsystemUnavailable as e:
                log.warn(f"lift connect failed ({e}); lift IDLE.")
        if cfg.get("auto_home") and self.core.connected():
            log.info("homing lift (stall-seek floor)...")
            try:
                (self._writer.home if self._writer else self.core.home)(wait=False)
            except HardwareError as e:
                log.warn(f"lift home failed: {e}")

        latched = QoSProfile(depth=1)
        latched.durability = DurabilityPolicy.TRANSIENT_LOCAL   # late joiners get last

        self.create_subscription(Float64, cfg["topic_command"], self._on_cmd, 10)
        self.create_subscription(Float64, cfg.get("topic_velocity", "/lift/velocity"),
                                 self._on_velocity, 10)
        self.create_subscription(Bool, cfg["topic_calibrate"], self._on_calibrate, 10)
        self._pub = self.create_publisher(JointState, cfg["topic_joint_states"], 10)
        self._pub_limits = self.create_publisher(String, cfg["topic_limits"], latched)
        self._limits = None
        # Absolute floor datum [m] the stroke<->absolute conversion pivots on. Seeded
        # 0 (no offset) until the first telemetry frame reports the real min_pos, then
        # tracked live from LiftState so a recalibration is picked up automatically.
        self._floor_m = 0.0
        # Per-move torque, command-on-change (see _on_cmd). The column is non-back-drivable,
        # so a resting lift needs NO torque — we energize only for the duration of a move
        # and release again on arrival (exactly what the setup studio does by hand).
        self._span_m = None          # travel span [m] (max_pos-min_pos); clamps the target
        self._target = None          # last STROKE setpoint commanded (None = none yet)
        self._moving = False         # True between a new target and its arrival
        # Joystick JOG (velocity): the drive stick moves the column up/down at a velocity
        # WHILE held (firmware velocity mode) and stops + releases torque on release. Kept
        # separate from the position path above — a held stick would otherwise ramp a
        # position setpoint that floods the Pico's position loop. See _on_velocity.
        self._jogging = False        # True while a velocity jog owns the lift
        self._last_vel = 0.0         # last velocity forwarded (dedup — set_velocity resets the loop)
        self._last_stroke = 0.0      # newest measured stroke [m] (for a bumpless jog->hold handoff)
        # GROUNDING gate: no position command is accepted (or even latched into
        # _target) until the Pico is connected AND at least one telemetry frame has
        # arrived. Before that we have no floor datum, no travel span, and no idea
        # where the column IS — executing (or remembering) a setpoint then is how a
        # stale gateway stream drove the lift to the floor on bring-up.
        self._have_fb = False        # True after the first good telemetry frame
        add_publisher_watchdog(self, cfg["topic_command"])
        self._set_torque(False)      # boot into the HOLD state: no torque, worm gear holds
        self.create_timer(1.0 / max(cfg["publish_rate"], 1.0), self._publish_state)
        self.create_timer(1.0, self.core.service)   # hot-plug reconnect
        self.get_logger().info(
            f"lift online: cmd={cfg['topic_command']} (Float64 m) -> Pico.")

    def _set_torque(self, on):
        """Energize (True) / release (False) the lift motor. HOLD == torque OFF: the
        column is non-back-drivable so it holds mechanically with the H-bridge released,
        drawing no current and generating no heat. We turn torque ON only for the span of
        a move and back OFF on arrival — the setup-studio behaviour. Idempotent; a failure
        is logged, never fatal (the firmware boots torque-on and coasts when idle anyway)."""
        if not self.core.connected():
            return
        try:
            self.core.set_torque(bool(on))
        except Exception as e:                       # never let this sink the node
            self.get_logger().warn(f"lift set_torque({on}) failed: {e}",
                                   throttle_duration_sec=5.0)

    # A stroke target that moves less than this (metres) counts as UNCHANGED. The gateway
    # re-publishes a held setpoint at its ~50 Hz control rate; re-sending the SAME target
    # would re-plan the Pico's trapezoidal profile every tick (so the motor never
    # accelerates) AND re-energize a column meant to be resting. 0.5 mm is far below the
    # slider's 1 mm step, so every real move still starts a fresh command. Arrival =
    # within _ARRIVE_TOL_M of the (clamped) target AND slower than _ARRIVE_VEL_MPS.
    _CMD_EPS_M = 0.0005
    _ARRIVE_TOL_M = 0.005
    _ARRIVE_VEL_MPS = 0.006

    # Below this speed a jog counts as RELEASED (stop + release torque). Above the vel-eps
    # a fresh velocity is forwarded to the firmware (which resets its loop on each call, so
    # a held stick must NOT re-stream the same speed — dedup like the position path).
    _JOG_DEADBAND_MPS = 0.003
    _VEL_EPS_MPS = 0.004

    def _on_velocity(self, msg: Float64):
        """Joystick jog: drive the column at a VELOCITY while the stick is held, stop and
        release torque when it returns to centre. The firmware's velocity mode holds the
        speed and clamps itself at the end stops; on release the worm gear holds. Only a
        CHANGED speed is forwarded (set_velocity resets the firmware loop every call)."""
        v = float(msg.data)
        if not math.isfinite(v):
            return
        if abs(v) > self._JOG_DEADBAND_MPS:                       # stick deflected -> jog
            if (not self._jogging) or abs(v - self._last_vel) > self._VEL_EPS_MPS:
                self._set_torque(True)
                try:
                    if self._writer is not None:
                        self._writer.set_velocity(v)
                    else:
                        self.core.set_velocity(v)
                    self._last_vel = v
                except Exception as e:                            # noqa: BLE001
                    self.get_logger().warn(f"lift jog failed: {e}",
                                           throttle_duration_sec=2.0)
            self._jogging = True
        elif self._jogging:                                       # stick released -> STOP
            self._jogging = False
            self._last_vel = 0.0
            self.core.stop()                                      # halt now (lease-free)
            self._set_torque(False)                               # release; worm gear holds
            # Reseed the position path to where we actually stopped so the (still-arriving)
            # /lift/command setpoint resumes holding HERE without a correction lurch.
            self._target = self._last_stroke
            self._moving = False
            self.get_logger().info(
                f"lift: JOG stop at stroke {self._last_stroke:.3f} m, torque released.")

    def _on_cmd(self, msg: Float64):
        # A velocity jog owns the lift while active: ignore the position setpoint stream
        # (it is a stale ramp) until the stick is released and _on_velocity hands back.
        if self._jogging:
            return
        stroke = float(msg.data)
        # NaN = the gateway's "no lift command" sentinel (its measured-height seed
        # hasn't grounded yet). Also GROUND every real setpoint: with no connection or
        # no telemetry we have no datum/span and no measured height — drop the command
        # WITHOUT latching it into _target, so a stale stream received while the Pico
        # is away can never become "the last target" the column dives to on connect.
        if not math.isfinite(stroke):
            return
        if not self.core.connected() or not self._have_fb:
            self.get_logger().info(
                "lift: position command ignored — no telemetry yet (column holds).",
                throttle_duration_sec=5.0)
            return
        # Command-on-CHANGE, like the setup studio: a held setpoint is sent ONCE, not
        # streamed. The Pico runs its profile to completion and coasts on arrival, so the
        # repeat carries no information — and re-sending it would reset the move and
        # re-energize a resting column.
        if self._span_m is not None:                 # clamp to travel so a >range target
            stroke = min(max(stroke, 0.0), self._span_m)   # can still register as ARRIVED
        if self._target is not None and abs(stroke - self._target) < self._CMD_EPS_M:
            return
        # First grounded setpoint that already matches the measured height (the
        # gateway's seed) is a HOLD, not a move: adopt it silently instead of
        # energizing the motor to travel zero millimetres.
        if (self._target is None
                and abs(stroke - self._last_stroke) < self._ARRIVE_TOL_M):
            self._target = stroke
            return
        self._target = stroke
        self._moving = True
        self._set_torque(True)                       # energize for the move
        try:
            # /lift/command is STROKE (0 == floor); the firmware wants ABSOLUTE height.
            target_abs = stroke + self._floor_m
            self.get_logger().info(
                f"lift: MOVE -> stroke {stroke:.3f} m (abs {target_abs:.3f} m), torque ON",
                throttle_duration_sec=0.5)         # a slider DRAG changes fast; don't spam
            if self._writer is not None:
                self._writer.set_height(target_abs)  # non-blocking; the Pico runs the loop
            else:
                self.core.set_height(target_abs)
        except HardwareError as e:
            self.get_logger().warn(f"lift move refused: {e}",
                                   throttle_duration_sec=2.0)
            self._moving = False
            self._set_torque(False)  # nothing is running the profile — don't stay hot
        except Exception as e:
            self.get_logger().warn(f"lift move failed: {e}",
                                   throttle_duration_sec=2.0)
            self._moving = False
            self._set_torque(False)

    def _on_calibrate(self, msg: Bool):
        if not msg.data:
            return
        try:
            # Non-blocking; the Pico runs the floor+ceiling stall-seek and
            # persists the measured min/max envelope to its own flash.
            self.get_logger().info("lift calibrate: stall-seek floor + ceiling...")
            if self._writer is not None:
                self._writer.calibrate(wait=False)
            else:
                self.core.calibrate(wait=False)
        except HardwareError as e:
            self.get_logger().warn(f"lift calibrate refused: {e}",
                                   throttle_duration_sec=2.0)
        except Exception as e:
            self.get_logger().warn(f"lift calibrate failed: {e}",
                                   throttle_duration_sec=2.0)

    def _publish_state(self):
        st = self.core.state()
        if not st.connected or st.height is None:
            # Connection gap: re-ground before accepting any command, and forget the
            # dead session's setpoint — the Pico may have power-cycled, so "the last
            # target" no longer describes anything real. The first grounded setpoint
            # after rejoin then either adopts silently (matches measured) or is a
            # deliberate fresh move.
            self._have_fb = False
            self._target = None
            self._moving = False
            return
        self._have_fb = True         # telemetry live: commands are grounded again
        # Track the firmware's live floor datum + travel span so both directions of the
        # conversion (command here + feedback below) share ONE offset, and _on_cmd can
        # clamp a target into reach. Keep the last good values if an uncalibrated board
        # momentarily reports 0.
        if st.min_pos and st.min_pos > 0:
            self._floor_m = float(st.min_pos)
        if st.max_pos and st.max_pos > st.min_pos:
            self._span_m = float(st.max_pos) - float(st.min_pos)
        # ABSOLUTE height -> STROKE (0 == floor). Floor-clamp so encoder noise just
        # below the datum never publishes a small negative stroke.
        pos_stroke = max(0.0, float(st.height) - self._floor_m)
        self._last_stroke = pos_stroke               # for the jog->hold handoff
        # Arrival -> release torque: the move is done once we are within tolerance of the
        # target and effectively stopped. The worm gear then holds with the motor released
        # (no current, no heat) until the next changed command re-energizes it.
        if (self._moving and self._target is not None
                and abs(pos_stroke - self._target) < self._ARRIVE_TOL_M
                and abs(float(st.velocity)) < self._ARRIVE_VEL_MPS):
            self._moving = False
            self._set_torque(False)
            self.get_logger().info(
                f"lift: ARRIVED at stroke {pos_stroke:.3f} m, torque released (hold).")
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = [self.joint]
        js.position = [pos_stroke]
        js.velocity = [float(st.velocity)]           # a rate — frame-independent
        self._pub.publish(js)
        self._publish_limits(st)

    def _publish_limits(self, st):
        """Latched min/max envelope — republished ONLY when it changes (e.g. after
        a calibrate), so a late subscriber always sees the current travel range.

        Reported in the ROS STROKE frame: min == 0 (the floor), max == the measured
        travel span (max_pos - min_pos), matching the /lift/command + /lift/joint_states
        convention above and the URDF's 0..stroke prismatic pair."""
        span = max(0.0, float(st.max_pos) - float(st.min_pos))
        lim = {"min": 0.0,
               "max": round(span, 4),
               "calibrated": bool(st.calibrated)}
        if lim != self._limits:
            self._limits = lim
            self._pub_limits.publish(String(data=json.dumps(lim)))

    def destroy_node(self):
        try:
            if self._writer is not None:
                self._writer.release()
            else:
                self.core.close()      # standalone: free-wheel + close the Pico
        finally:
            super().destroy_node()


# legacy class name (pre-mabel_hw shims + old importers)
LiftSubsystem = LiftNode
