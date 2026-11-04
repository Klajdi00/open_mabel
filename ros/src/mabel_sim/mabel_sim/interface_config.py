"""Canonical hardware <-> simulation interface contract for MABEL.

This file is the single source of truth that lets the MuJoCo simulation be a
*drop-in replacement* for the real hardware bring-up.  It defines, per
subsystem:

  * which MuJoCo actuators/joints belong to that subsystem,
  * the ROS topic names and the joint names that travel on them,
  * the name translations between the three naming worlds we have to bridge:

      MuJoCo name      e.g.  "Left Index MCP"   (spaces, Title Case)
      URDF  name       e.g.  "left_index_mcp"   (snake_case, used by RViz /
                                                  robot_state_publisher)
      ORCA  name       e.g.  "index_mcp"        (bare finger name on the
                                                  /left_hand topics)

The real hardware nodes (mabel_base_hardware, mabel_arms_hardware,
mabel_body_hardware, mabel_hand_hardware) publish/subscribe these same topics, so
switching between `mabel_sim sim.launch.py` and `mabel_bringup hardware.launch.py`
changes nothing for any downstream node (teleop, planners, RViz).
"""

import math

# --------------------------------------------------------------------------- #
#  Naming helpers
# --------------------------------------------------------------------------- #

def urdfize(mj_name: str) -> str:
    """MuJoCo joint/actuator name  ->  URDF joint name.

    The MuJoCo model was authored so that this exact transform reproduces the
    URDF joint names (verified against mabel_description/urdf/mabel_full.urdf).
    """
    return mj_name.lower().replace(" ", "_")


# --------------------------------------------------------------------------- #
#  Model / scene
# --------------------------------------------------------------------------- #

# Default MuJoCo scene (overridable via the `scene` ROS parameter).
# Furnished home (kitchen/dining room + graspable objects) so the cameras and
# 360 deg LiDAR have a realistic environment to perceive out of the box.
DEFAULT_SCENE = "~/open_mabel/simulation/mabel_mujoco/scenes/mabel_home.xml"

# Free joint of the mobile base inside the MuJoCo model.
BASE_FREE_JOINT = "base_root"


# --------------------------------------------------------------------------- #
#  BASE  (swerve drive + lift)  --  topic: /mabel_cmd (mabel_interfaces/MabelSwerveCmd)
# --------------------------------------------------------------------------- #

CMD_BASE_TOPIC = "/mabel_cmd"
ODOM_TOPIC = "/odom"
ODOM_FRAME = "odom"
BASE_FRAME = "base_link"

WHEEL_RADIUS = 0.0378        # m  (drive actuator is velocity -> rad/s = v / r; measured from model)
# NOTE: the sim wheel mesh + contact actually ROLLS at an effective ~0.020 m and
# slips ~40-50%, so the base reaches ~60% of the commanded speed. Changing this
# value shifts the slip regime nonlinearly, so it's left at the nominal; the
# wheel-odometry node calibrates its OWN radius (~0.020) to the effective rolling
# instead. Tuning the wheel collision (a cylinder geom of known radius + higher
# friction) in models/swerve_module.xml is the real fix for clean rolling.
MAX_LIN_ACCEL = 2.0          # m/s^2  command ramping
MAX_ANG_ACCEL = 5.0          # rad/s^2
STEER_DEADBAND = 0.52        # rad; above this steering error, drive power is choked

# Lift: position actuator on the "Total Lift" tendon, ctrlrange 0 .. 0.635 m.
LIFT_ACTUATOR = "Lift"
LIFT_MIN, LIFT_MAX = 0.0, 0.635

# Swerve modules (3-module delta base). Values copied verbatim from the MuJoCo
# project's own IK (simulation/mabel_mujoco/scripts/mabel/config.py) — the
# module positions AND the per-column mechanical yaw offsets are both required,
# otherwise the three wheels point in conflicting directions and lock the base.
#   pos  = module [x, y] offset from chassis centre (m)
#   yaw  = mechanical mount yaw of the steering column (rad); the +pi/2 term
#          compensates for the wheel's native coordinate frame.
SWERVE_MODULES = {
    "fl": {"steer": "FL Steer", "drive": "FL Drive",
           "pos": (-0.17, -0.17), "yaw": 1.570795 + math.pi / 2},
    "fr": {"steer": "FR Steer", "drive": "FR Drive",
           "pos": (-0.17,  0.17), "yaw": 0.0 + math.pi / 2},
    "b":  {"steer": "B Steer",  "drive": "B Drive",
           "pos": ( 0.15,  0.0),  "yaw": 3.9269926 + math.pi / 2},
}


# --------------------------------------------------------------------------- #
#  ARMS  --  topics: /arms/command  ->  /arms/joint_states  (sensor_msgs/JointState)
# --------------------------------------------------------------------------- #
#  7-DOF per arm.  The wrist roll belongs to the HAND subsystem (it is ORCA
#  motor id 17), so it is intentionally NOT listed here.

CMD_ARMS_TOPIC = "/arms/command"
ARMS_STATE_TOPIC = "/arms/joint_states"

ARM_ACTUATORS = (
    [f"Left Arm {i}" for i in range(1, 8)] +
    [f"Right Arm {i}" for i in range(1, 8)]
)


# --------------------------------------------------------------------------- #
#  BODY  (torso + 3-DOF neck)  --  /body/command -> /body/joint_states
# --------------------------------------------------------------------------- #

CMD_BODY_TOPIC = "/body/command"
BODY_STATE_TOPIC = "/body/joint_states"

BODY_ACTUATORS = ["Torso", "Neck Yaw", "Neck Pitch", "Neck Roll"]


# --------------------------------------------------------------------------- #
#  HANDS  --  /<side>_hand/command -> /<side>_hand/{joint_states,temperatures,voltages}
# --------------------------------------------------------------------------- #
#  Matches mabel_hand_hardware/hand_node.py exactly: one node-equivalent per
#  hand, living under the left_hand / right_hand namespace, joint names are the
#  ORCA bare names ("index_mcp") or side-prefixed ("left_index_mcp").
#
#  ORCA bare name  ->  MuJoCo actuator suffix.  Note the thumb naming differs
#  between ORCA (cmc/mp/ip) and the MuJoCo/URDF model (mcp/pip/dip).

HAND_SIDES = ("left", "right")

# ORCA bare joint name  ->  MuJoCo actuator name fragment (without side prefix).
ORCA_TO_MJ_SUFFIX = {
    "index_abd":  "Index Abd",  "index_mcp":  "Index MCP",  "index_pip":  "Index PIP",
    "middle_abd": "Middle Abd", "middle_mcp": "Middle MCP", "middle_pip": "Middle PIP",
    "ring_abd":   "Ring Abd",   "ring_mcp":   "Ring MCP",   "ring_pip":   "Ring PIP",
    "pinky_abd":  "Pinky Abd",  "pinky_mcp":  "Pinky MCP",  "pinky_pip":  "Pinky PIP",
    "thumb_cmc":  "Thumb MCP",  "thumb_abd":  "Thumb Abd",
    "thumb_mp":   "Thumb PIP",  "thumb_ip":   "Thumb DIP",
    "wrist":      "Wrist",
}

# Fake but realistic feedback for the temperatures / voltages topics.
HAND_FAKE_TEMP_C = 31.0
HAND_FAKE_VOLT_V = 7.4


def hand_joint_map(side: str):
    """Return {orca_bare_name: mujoco_actuator_name} for one hand side."""
    prefix = "Left" if side == "left" else "Right"
    return {bare: f"{prefix} {suffix}" for bare, suffix in ORCA_TO_MJ_SUFFIX.items()}


# --------------------------------------------------------------------------- #
#  AUDIO  --  topics: /audio/speaker -> the card -> /audio/mic
#             (std_msgs/UInt8MultiArray, raw S16_LE PCM, one frame per message)
# --------------------------------------------------------------------------- #
#  The hardware_bridge owns the USB sound card; ROS never opens it. Format is
#  fixed by hardware_bridge/config.py AUDIO (16 kHz mono S16_LE, 20 ms frames =
#  640 B) and frozen in its contract.json — it is NOT carried per-message.
#  audio_common_msgs is deliberately not used: it is not installed on thor and
#  the wire is a flat byte run either way.

CMD_AUDIO_TOPIC = "/audio/speaker"
AUDIO_MIC_TOPIC = "/audio/mic"
AUDIO_ENABLE_TOPIC = "/audio/enable"        # Bool: mic live (false = muted)
AUDIO_STATE_TOPIC = "/audio/state"          # Bool, LATCHED: realized mic state
AUDIO_HEALTH_TOPIC = "/audio/device_health"  # String (JSON) @2 Hz


# --------------------------------------------------------------------------- #
#  Publish rates (Hz)
# --------------------------------------------------------------------------- #

RATE_JOINT_STATES = 50.0     # global /joint_states + /arms + /body
RATE_HANDS = 50.0            # per-hand feedback (orca default)
RATE_ODOM = 50.0
RATE_TELEMETRY = 2.0         # human-readable terminal dashboard
RATE_IMU = 200.0             # front D435i IMU (sensor_msgs/Imu); real BMI055 ~200-400 Hz
RATE_AUDIO = 50.0            # mic frames/s (20 ms @ 16 kHz); device-paced, not timed
RATE_AUDIO_HEALTH = 2.0      # /audio/device_health (mirrors motor_health)

# --------------------------------------------------------------------------- #
#  Front D435i IMU  --  sensor_msgs/Imu on IMU_TOPIC, frame_id = IMU_FRAME.
#  Read from the MuJoCo accelerometer/gyro/framequat sensors on the `front_imu`
#  site (see mabel_full.xml). The accelerometer already reports specific force
#  (gravity included); the node only adds sensor NOISE + BIAS on top.
# --------------------------------------------------------------------------- #
IMU_TOPIC = "/camera/front/imu"
IMU_FRAME = "front_imu_optical_frame"
# MuJoCo sensor names (must match mabel_full.xml).
IMU_ACC_SENSOR = "front_imu_acc"
IMU_GYRO_SENSOR = "front_imu_gyro"
IMU_QUAT_SENSOR = "front_imu_quat"
IMU_SITE = "front_imu"
# Noise model defaults, tuned to the D435i's Bosch BMI055 (overridable via ROS
# params on mujoco_node). White-noise std is derived from the datasheet noise
# density at the IMU_RATE bandwidth; bias is a fixed per-axis offset (drawn once)
# plus a slow random walk -- exactly what cuVSLAM's IMU calibration estimates out.
IMU_ACC_NOISE = 0.0147       # m/s^2   white-noise std  (150 ug/rtHz @ 100 Hz BW)
IMU_GYRO_NOISE = 0.0024      # rad/s   white-noise std  (0.014 dps/rtHz @ 100 Hz BW)
IMU_ACC_BIAS = 0.02          # m/s^2   std of the fixed accel bias offset
IMU_GYRO_BIAS = 0.005        # rad/s   std of the fixed gyro bias offset
IMU_ACC_WALK = 0.0010        # m/s^2 per sample  accel bias random-walk step std
IMU_GYRO_WALK = 0.0002       # rad/s per sample  gyro  bias random-walk step std
