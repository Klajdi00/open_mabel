"""robot_command.py — THE standardized MABEL robot actuation command.

ONE canonical schema that **both** the MuJoCo simulator and the real robot
consume, so every client speaks the same language and the bridge dispatches
identically to sim or hardware.

  client INTENT  (wire_protocol.py: teleop_frame / joint_command / control_mode)
        │  resolved by the bridge (retarget · WBC · IK · base follower)
        ▼
  RobotCommand   (this file — the ACTUATION contract)
        ├──► sim:  applied to MuJoCo actuators
        └──► real: published to ROS, one field per topic:

      base      → /mabel_cmd      mabel_interfaces/MabelSwerveCmd  {vx,vy,omega,lift,estop}
      joints    → /arms/command, /body/command        sensor_msgs/JointState
      grippers  → /<side>_hand/command                sensor_msgs/JointState

Because the SAME command object drives both, a demonstration recorded in sim and
a deployment on hardware are byte-identical at this seam — which is exactly the
property the imitation-learning policies need.

Units (SI, explicit so no client guesses):
  vx, vy = m/s   ·   omega = rad/s   ·   lift = m   ·   joints = rad   ·   grip = 0..1

The monitor renders this object live so a human can verify the control commands
are dispatched correctly; nothing here applies anything — it is a pure value
type plus presentation helpers.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field


# Canonical limb grouping, by substring of the joint name (robust to the exact
# MuJoCo names, e.g. "Left Arm 3", "Lift Lower", "FL Steer", "Neck Yaw").
def joint_group(name: str) -> str:
    n = name.lower()
    if "arm" in n and "left" in n:   return "left_arm"
    if "arm" in n and "right" in n:  return "right_arm"
    if "wrist" in n and "left" in n: return "left_arm"
    if "wrist" in n and "right" in n: return "right_arm"
    if "torso" in n:                 return "torso"
    if "neck" in n:                  return "neck"
    if "lift" in n:                  return "lift"
    if "steer" in n or "drive" in n or "wheel" in n: return "base"
    # Orca-hand finger joints ("Left Thumb MCP", "Right Pinky Abd", …) + grippers.
    if any(f in n for f in ("grip", "finger", "hand", "thumb", "index",
                            "middle", "ring", "pinky")): return "gripper"
    return "other"


# Order limbs read top-to-bottom in the monitor.
GROUP_ORDER = ["base", "lift", "torso", "neck", "left_arm", "right_arm",
               "gripper", "other"]

# How each field of this command maps onto the real robot's ROS interface — shown
# in the monitor and used by the (future) hardware bridge so the mapping lives in
# exactly one place.
ROS_MAPPING = {
    "base":     ("/mabel_cmd",         "mabel_interfaces/MabelSwerveCmd"),
    "joints":   ("/arms/command · /body/command", "sensor_msgs/JointState"),
    "grippers": ("/<side>_hand/command", "sensor_msgs/JointState"),
    "estop":    ("/mabel_cmd.estop",   "bool"),
}


@dataclass
class BaseTwist:
    vx: float = 0.0       # m/s   forward (+) / back (−)
    vy: float = 0.0       # m/s   left (+) / right (−)  (holonomic strafe)
    omega: float = 0.0    # rad/s yaw, CCW (+)
    lift: float = 0.0     # m     column height target


@dataclass
class RobotCommand:
    """One actuation command, dispatched to sim or real identically."""
    seq: int = 0
    stamp: float = field(default_factory=time.time)
    target: str = "sim:mujoco"               # "sim:mujoco" | "real:ros"
    source: dict = field(default_factory=dict)   # {device, peer} that drove it
    mode: dict = field(default_factory=dict)     # {method, controlType, region, stiffness}
    estop: bool = False
    base: BaseTwist = field(default_factory=BaseTwist)
    joints: dict = field(default_factory=dict)   # {joint_name: radians}
    grippers: dict = field(default_factory=dict) # {"left": 0..1, "right": 0..1}
    # Per swerve module the resolved {steer rad, drive rad/s} — the swerve-IK
    # output of the base twist, shown so the wheels can be debugged individually.
    wheels: dict = field(default_factory=dict)   # {"FL": {"steer", "drive"}, …}

    def to_dict(self) -> dict:
        d = asdict(self)
        # round floats so the wire/JSON view is readable, not 17 digits
        d["base"] = {k: round(v, 4) for k, v in d["base"].items()}
        d["joints"] = {k: round(float(v), 4) for k, v in self.joints.items()}
        d["grippers"] = {k: round(float(v), 3) for k, v in self.grippers.items()}
        d["wheels"] = {m: {a: round(float(x), 4) for a, x in v.items()}
                       for m, v in self.wheels.items()}
        return d

    def grouped_joints(self) -> dict[str, dict]:
        """Joint targets bucketed by limb, in GROUP_ORDER, for display."""
        out: dict[str, dict] = {}
        for name, val in self.joints.items():
            out.setdefault(joint_group(name), {})[name] = round(float(val), 4)
        return {g: out[g] for g in GROUP_ORDER if g in out}
