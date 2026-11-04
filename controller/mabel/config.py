"""
Central configuration for MABEL whole-body control.

All physical constants, kinematic geometry, and MuJoCo name lists live here so
the control modules (swerve, wbc, head, retarget) never hard-code magic numbers.

The tip-over safety motion model (its measured CoM/lever constants AND the
`safe_base_step` / `tip_usage` / `max_speed_dir` family) now lives in
`mabel.motion_model.tip_over` — the single source — and is re-exported here so
every existing `from mabel.config import safe_base_step` / `config.COM_XY`
consumer keeps working unchanged. Re-derive those constants with
`experiments/wbc_stability.py` (E4) and edit them in motion_model/tip_over.py.
"""
import math
import os
import numpy as np

# Tip-over motion model + base/lift motion limits (single source; numpy-only).
from .motion_model.tip_over import *          # noqa: F401,F403
from .motion_model.tip_over import _axis_decel, _support_edges, _SUPPORT_EDGES  # noqa: F401

# ── Scene / model ────────────────────────────────────────────────────────────
SCENE_XML = "../scenes/mabel_sim.xml"   # relative to the scripts/ directory

# ── Mobile base (3-module delta swerve) ──────────────────────────────────────
WHEEL_RADIUS = 0.0381   # 3in REV MAXSwerve wheel (radius = 1.5in)

# Module positions relative to the chassis centre [x, y] (metres), from the XML.
MODULE_POSITIONS = {
    "fl": np.array([-0.17, -0.17]),
    "fr": np.array([-0.17,  0.17]),
    "b":  np.array([ 0.15,  0.0]),
}

# Mechanical mount yaw of each steering column. The +pi/2 compensates for the
# wheel's native coordinate frame (see legacy/motion_model.py derivation).
MODULE_YAW_OFFSETS = {
    "fl": 1.570795 + (math.pi / 2),
    "fr": 0.0      + (math.pi / 2),
    "b":  3.9269926 + (math.pi / 2),
}

# Base motion limits / smoothing: MAX_LIN_SPEED, MAX_ANG_SPEED, MAX_LIN_ACCEL,
# MAX_ANG_ACCEL now live in motion_model/tip_over.py (re-exported above).
STEER_DEADBAND  = 0.52    # rad; above this steering error, drive power is choked

# ── Lift / torso ─────────────────────────────────────────────────────────────
# LIFT_RANGE + LIFT_HW_MAX_SPEED (the firmware 60 mm/s ceiling) live in
# motion_model/tip_over.py (re-exported above); the teleop tunables derive here.
LIFT_SPEED        = LIFT_HW_MAX_SPEED   # m/s manual jog (hardware-limited)
MAX_LIFT_IK_SPEED = LIFT_HW_MAX_SPEED   # m/s cap when the lift is driven by IK
# Torso rate cap when driven by the task-space IK. The torso swings the entire
# upper-body inertia: an uncapped DLS slams it ~1.4 rad/s toward a far target,
# and the angular momentum carries the CoM past the static support edge before
# the stability guard's stop line can hold (measured face-plant on the bench).
MAX_TORSO_IK_SPEED = 0.5                # rad/s

# ── Tip-over safety: moved to motion_model/tip_over.py (re-exported above) ────
# GRAVITY, TIP_LEVER_*, TIP_SF_*, STOP_TIME_*, COM_Z0/COM_DZDLIFT, V_*_CEIL,
# SUPPORT_PTS, COM_XY and the com_height/max_*/tip_usage/safe_base_step family
# all come from mabel.motion_model.tip_over now.
# ── Whole-body force compliance (admittance base) ─────────────────────────────
# The base behaves as a virtual planar mass with viscous + Coulomb "virtual
# friction", driven by the external drag wrench. A SMALL comfort box (referenced
# to the neutral hand pose, not the chassis origin) makes the base engage quickly
# so the arm never reaches its limit and pulls the body over. Tuned by the rigorous
# drag test tools/compliance_drag_test.py.
DRAG_BOX        = 0.03   # m; hand excursion from neutral before the base follows
DRAG_BOX_RAMP   = 0.04   # m; extra excursion to reach full engagement
ADMIT_MASS      = 4.5    # kg;  virtual base mass (lower = snappier engage)
ADMIT_INERTIA   = 2.5    # kg m^2
ADMIT_LIN_DAMP  = 22.0   # N/(m/s);  viscous virtual friction (v_ss = F/this)
ADMIT_ANG_DAMP  = 18.0   # Nm/(rad/s);  high -> a one-hand push yaws only gently
ADMIT_F_STATIC  = 3.0    # N;  static (Coulomb) deadband on force
ADMIT_M_STATIC  = 8.0    # Nm; static deadband on moment (ignore incidental yaw)
ADMIT_V_STOP    = 0.02   # m/s; settle-to-rest threshold
ADMIT_W_STOP    = 0.05   # rad/s

# ── Upper-body IK chain (18 DOF: lift + torso + 8 per arm) ────────────────────
# Names match the <actuator> and <joint> entries in models/mabel_full.xml.
IK_ACTUATORS = (
    ["Lift", "Torso"]
    + [f"Left Arm {i}" for i in range(1, 8)] + ["Left Wrist"]
    + [f"Right Arm {i}" for i in range(1, 8)] + ["Right Wrist"]
)
# The Lift actuator drives a tendon; its underlying joint for qpos reads is:
LIFT_JOINT = "Lift Lower"

# Differential-IK tuning.
IK_KP_POS      = 6.0      # end-effector position gain
IK_KP_ROT      = 2.0      # end-effector orientation gain
IK_DAMPING     = 0.05     # DLS lambda
IK_POSTURE_GAIN = 0.5     # null-space pull toward the rest posture
IK_VEL_CAP     = 1.5      # m/s   cap on commanded EE linear tracking speed
IK_ANGVEL_CAP  = 3.0      # rad/s cap on commanded EE angular tracking speed
# Hold-still deadband: ignore sub-threshold tracking error so the arm does not
# creep/chase gravity sag when the operator is not commanding anything.
IK_POS_DEADBAND = 0.002   # m     EE position error below this -> no command
IK_ROT_DEADBAND = 0.0087  # rad   EE orientation error below this (~0.5 deg) -> no command
# Reach-assist followers (lazy base/lift/torso) only engage while the operator is
# actively moving the hands; below this per-tick target motion the body holds still.
LAZY_CMD_EPS    = 1.5e-4  # m     per-substep hand-target motion to count as "commanding"
IK_SETTLE_STEPS = 800     # physics steps to settle to gravity equilibrium at startup
# Per-DOF weights for the weighted DLS (higher = "lazier" joint).
# HUMAN-LIKE priority (operator directive, 2026-07-18): the MOBILE BASE does the
# walking; the LIFT and TORSO are the LAZIEST sections — a person steps toward
# out-of-reach work instead of contorting. Torso 2.0 measured as "unnatural":
# a 0.6 m forward drag slammed it to a full −1.05 rad lean before the base
# (comfort-box law) ever engaged.
IK_WEIGHT_LIFT  = 50.0
IK_WEIGHT_TORSO = 30.0

# ── Task-space base follow: position+heading servo (whole_body wrist) ─────────
# The base parks the chassis at its HOME offset from the palm-target centroid
# (the geometry the arms work at, ~75-80% of usable reach — the "80% rule"
# generalised to the plane; ARM_MAX_REACH = 0.672 m measured mount→palm with the
# arm saturated) and turns to FACE the centroid. Both servos are HYSTERETIC
# snap-stop state machines: engage above _ON, drive at a decisive minimum
# speed, stop dead below _OFF — a plain P-law creeps asymptotically near its
# deadband ("always floating"). Scalar reach-band laws could not represent
# LATERAL misalignment (side drags produced forward-crowding hunts, no strafe,
# no turn); the position error can.
ARM_MAX_REACH       = 0.67    # m usable mount→palm reach (measured, both arms)
FOLLOW_KP_REACH     = 2.5     # 1/s position error -> base speed
FOLLOW_POS_ON       = 0.06    # m  position error that ENGAGES the base
FOLLOW_POS_OFF      = 0.02    # m  error below which it STOPS (snap)
FOLLOW_MIN_V        = 0.08    # m/s minimum speed while engaged (no imperceptible creep)
FOLLOW_YAW_ON       = 0.30    # rad centroid bearing that engages the turn (~17°)
FOLLOW_YAW_OFF      = 0.08    # rad bearing below which the turn stops
FOLLOW_MIN_W        = 0.15    # rad/s minimum turn rate while engaged
FOLLOW_YAW_MIN_DIST = 0.30    # m  never yaw with the centroid on top of the base
                              #    (bearing ill-conditioned: the −3.3 rad/s spin)

# ── Task-space static-stability guard (WholeBodyIK.apply) ────────────────────
# Chasing an out-of-reach forward target, the 18-DOF IK contorts the torso +
# arms until the true CoM leaves the wheel support triangle and the robot
# face-plants (measured on the Control Lab bench: ~88 deg pitch-pole at ANY
# base speed — a posture problem, not a braking one; the tip-over motion model
# only bounds base accelerations against the FIXED home CoM). apply() therefore
# evaluates the PLANNED pose's live CoM support margin (motion_model.com_support)
# and decelerates the posture advance to stop AT the margin — the same
# stop-at-the-wall law as self-collision. The base follower then restores reach.
COM_GUARD_ENABLE = True
# The guard is a LAST-DITCH net, deliberately placed BELOW the natural operating
# band: with MAX_TORSO_IK_SPEED in force the natural task-space trajectories
# bottom out at +0.014..+0.031 m of live margin (measured across far-forward /
# fwd+up / fwd+lateral drags, both regions, mabel_bare; home margin ~0.094 m)
# WITHOUT falling — so the line sits at 0.010 m where it never engages in
# normal reach. Near the line the guard PROJECTS dq onto the margin-preserving
# tangent (CBF damper) rather than freezing it: a uniform freeze — and even an
# ACTIVE projection line at 0.055 — ratchets the posture into folded-arm
# attractors (measured: 0.6–0.9 m of lost reach).
COM_GUARD_MARGIN = 0.010  # m; the planned CoM may not push past this line
COM_GUARD_GAIN   = 2.0    # 1/s damper: max margin-consumption rate toward the line

# ── Impedance REFERENCE smoothing (WBC joint/wrist tracking) ───────────────────
# The task-space wrist target arrives as a MULTI-JOINT IK solution (green-ball
# planner, wrist drag, or a streamed cartesian goal). Its null-space/redundancy
# resolution can flip frame-to-frame, so the raw q_des jumps between wire frames.
# Fed straight into the impedance law τ = bias + K(q_des − q) − D q̇ (engine.py),
# every jump steps the spring and — with the modest joint damping D=0.6 — the arm
# BUZZES ("very jittery" vs the smooth joint slider, whose q_des moves slowly and
# one joint at a time). These filter the REFERENCE the impedance law tracks, at
# the single choke point, so the arm follows a wrist pose "slower but surely":
#   * a first-order low-pass (WBC_REF_TAU) rejects the high-frequency IK wiggle;
#   * a slew cap (WBC_REF_SLEW) bounds how fast the reference may move, so a large
#     jump becomes a deliberate ramp instead of a torque impulse.
# A slow, smooth command (joint slider, the client-eased green-ball Execute) sits
# below both the LPF corner and the slew cap, so it passes through with only a few
# tens of ms of lag. dt-correct (frame-rate independent). Set WBC_REF_TAU=0 and a
# huge WBC_REF_SLEW to restore the old verbatim behaviour. Verified by
# controller/experiments/exp_wbc_ref_smoothing.py.
WBC_REF_TAU  = float(os.getenv("MABEL_WBC_REF_TAU",  "0.08"))  # s,     LPF time constant
WBC_REF_SLEW = float(os.getenv("MABEL_WBC_REF_SLEW", "3.0"))   # rad/s, reference slew cap
# Stiffness-scaled damping: the impedance spring τ = … + K(q_des−q) − D q̇ is a
# second-order system whose damping ratio ζ = D / (2√(K·M_eff)) FALLS as the
# operator raises stiffness K — so a stiff arm rings and OVERSHOOTS a target
# ("track surely" fails). Adding a term ∝ √K holds ζ roughly constant across the
# whole stiffness range: D_eff = WBC_DAMP_BASE + WBC_DAMP_RATIO·√K in IMPEDANCE
# only. Compliance is K=0, so the √K term vanishes and Soft keeps exactly its
# back-drivable viscous feel (WBC_DAMP_BASE). Tuned to ζ≈0.8 (a few % overshoot,
# still snappy) at the app's typical K≈30; verified in \S refsmooth + V4.
WBC_DAMP_BASE  = float(os.getenv("MABEL_WBC_DAMP_BASE",  "0.6"))   # Nm·s/rad, base viscous damping (== the old constant)
WBC_DAMP_RATIO = float(os.getenv("MABEL_WBC_DAMP_RATIO", "0.24"))  # Nm·s/rad per √(Nm/rad), impedance-only
# Discrete-stability guard on the per-joint damping. The −D q̇ term is applied with a
# zero-order hold each control step; the explicit (forward-Euler) update is stable only
# while D < 2·M_j/dt (M_j = the joint's effective inertia). The stiffness-scaled D_eff
# above is tuned for the ~0.2 kg·m² arm links, but the ORCA WRIST is ~100× lighter
# (M≈2.5e−3), so that D rings the wrist UP and the sim/CAN torque diverges (measured:
# qacc 1.3e4 rad/s² at K=30, dt=4 ms). Each managed joint's D is therefore capped at
# WBC_DAMP_DISC_MARGIN·M_j/dt (β=1 → the deadbeat-optimal value, a factor 2 below the
# hard limit). Only the wrists are ever near their limit; the arm/torso are untouched.
WBC_DAMP_DISC_MARGIN = float(os.getenv("MABEL_WBC_DAMP_DISC_MARGIN", "1.0"))  # β ∈ (0, 2)

# ── Self-collision avoidance (toggleable safety layer) ────────────────────────
# A capsule model keeps the moving links (forearms/hands) out of the torso, head,
# and opposite arm. The avoidance is RUNTIME-TOGGLEABLE (TeleopController /
# WBCEngine .set_self_collision) and ON BY DEFAULT — teleop must never be able to
# drive a hand through the robot's own body. This flag sets the initial state of
# every controller (WholeBodyIK, WBCEngine, TeleopEngine); opt OUT explicitly per
# run (bridge: --no-self-collision / MABEL_SELF_COLLISION=0; API:
# set_self_collision(False)).  See self_collision.py for the geometry. Validated
# by controller/experiments/exp_wbc_validation.py (guard ON: capsule gap held at
# the margin; OFF: interpenetration).
SELF_COLLISION_ENABLE = True    # safety default ON; explicit opt-out only

# Avoidance LAW (how the guard alters the motion when a link nears another):
#   "stop" — velocity damper (Faverjon–Tournassoud): take the operator's tracking
#            solution and remove ONLY its gap-closing component, so the hand
#            decelerates to a stop AT the margin along the intended path and the
#            tangential motion the operator commanded is preserved. No sideways
#            push / redirection — it feels like a wall, not a force field.
#   "cbf"  — legacy CBF tracking-QP: re-optimises tracking subject to the barrier,
#            which slides the hand around obstacles (the "gravity field" feel).
# SC_LAG_PAD: extra margin (m) on the MEASURED-pose barrier rows of the "project"
# law. The commanded velocity reaches the metal through a position servo, so the
# physical link overshoots a commanded stop by its following error; the measured-
# pose rows brake this much earlier to absorb it (the arms-crossing scenario
# measured ~4 mm of hand-hand overshoot without it).
SC_LAG_PAD = float(os.getenv("MABEL_SC_LAG_PAD", "0.020"))
# SC_TARGET_VMAX: plan glide rate (rad/s) for the manipulate-target CBF filter
# (collision_avoidance/target_filter.py) — how fast a wrist/joint plan flows toward
# the raw client target. 1.0 matches the studios' safe-blend velocity budget.
SC_TARGET_VMAX = float(os.getenv("MABEL_SC_TARGET_VMAX", "1.0"))
# "project" (default 2026-07-29, operator-requested): CBF velocity projection —
# remove ONLY each active barrier's normal (gap-closing) velocity excess, keep the
# tangential component, so a hand commanded into the body SLIDES along it toward
# the target. "stop" = uniform slow-down that halts ON the approach path (a wall);
# "cbf" = legacy tracking-QP.
SC_METHOD = os.getenv("MABEL_SC_METHOD", "project")

SC_INFLUENCE = 0.06       # m   influence distance d_i: start reacting below this gap
# THIN LAYER, 2026-08-23 (operator: "have a thin layer of collision avoidance,
# 1 cm ish"). 0.025 until then. A 25 mm shell around every capsule is a lot of
# workspace to give up — it means the robot cannot bring a hand within 25 mm of
# its own chest — and the margin only ever needs to be thick enough that the
# discrete CBF cannot overshoot through it between ticks. At the teleop frame
# rate and SC_ALPHA the measured worst-case overshoot is well under a
# millimetre, so 10 mm is ~10x the overshoot and still reads as "nearly
# touching". Guarded by tests/test_collision_body.py, which asserts no
# interpenetration over the whole front-workspace posture table.
SC_MARGIN    = 0.010      # m   safety distance d_s: never let the surfaces get closer
SC_ALPHA     = 6.0        # 1/s barrier gain: allowed approach rate ḋ ≥ −α(d − d_s)
SC_STOP_DECEL = 2.0       # m/s² ("stop" law) braking-distance cap: approach speed is
                          # held ≤ √(2·decel·gap) so a fast motion stops before the margin
SC_POSTURE_REG = 1e-3     # μ   null-space posture weight folded into the (cbf) QP cost
SC_QP_VEL_CAP = 4.0       # rad/s (and m/s for the lift) per-DOF velocity box in the QP
SC_JOINT_LIMIT_GAIN = 8.0 # 1/s velocity-damper gain pulling q̇ off the joint limits
SC_ADJ_HOPS = 1           # never check a pair whose anchor bodies are within this many
                          # tree hops (parent/child) -- adjacent links can't self-collide
# Compliance "soft wall" (torque-domain barrier used by WBCEngine / ComplianceController
# when self-collision is enabled): brake the closing velocity and spring out of the
# margin so a dragged arm stops at the boundary instead of pushing through itself.
SC_BARRIER_KV = 40.0      # Nm·s  damping on the excess closing rate (ḋ below the limit)
SC_BARRIER_KP = 600.0     # Nm/m  stiffness pushing back out once inside the margin
# Capsule radii [m] of the bounding volumes (tuned so the rest pose is collision-free).
SC_RADII = {"forearm": 0.035, "hand": 0.055, "torso": 0.09, "head": 0.12,
            # Elbow sphere (the upper-arm guard, "l/r_upper" pairs). The
            # SETTLED right elbow rests at only 0.136 m from the torso axis:
            # r 0.035 + torso 0.09 leaves +11 mm at rest, over the pair's 3 mm
            # margin (capsule_guard._build_pairs) with 8 mm of settling slack.
            # A physical-size 0.045 sphere left +1 mm and the global SC_MARGIN
            # spring-loaded the natural rest pose (measured: 15° of fake
            # impedance "sag"; an upper-arm SEGMENT sits 6 cm inside the torso
            # capsule at rest and is unusable outright).
            "upper": 0.035}

# ── Active perception: 3-DOF neck + head ─────────────────────────────────────
NECK_ACTUATORS = ["Neck Yaw", "Neck Pitch", "Neck Roll"]
NECK_JOINTS    = ["Neck Yaw", "Neck Pitch", "Neck Roll"]
HEAD_BODY      = "head"
# Head local axes at the rest pose (probed from the model):
#   local -Z  -> world -X  (robot front)   => forward / optical axis
#   local +Y  -> world +Z  (up)
# These are tunable; flip a sign if the head looks the wrong way.
HEAD_FORWARD_LOCAL = np.array([0.0, 0.0, -1.0])
HEAD_UP_LOCAL      = np.array([0.0, 1.0,  0.0])
NECK_KP            = 4.0     # look-at pointing gain
NECK_FALLBACK_DIST = 1.5     # m; gaze-ray length when no scene hit is found
NECK_RAY_OFFSET    = 0.20    # m; start the gaze ray ahead of the head to clear the body
NECK_MIN_HIT_DIST  = 0.25    # m; closer hits are treated as self-hits (use fallback)

# ── Manipulation frames ──────────────────────────────────────────────────────
CHASSIS_BODY = "mabel_chassis"
LEFT_PALM    = "left_palm"
RIGHT_PALM   = "right_palm"

# ── Vision Pro retargeting (apple_avp -> MABEL) ──────────────────────────────
AVP_ZMQ_ADDR = "tcp://127.0.0.1:5556"   # matches the PUB in dexterous_hand/vision_pro
AVP_ZMQ_TOPIC = ""                        # PUB sends raw JSON (no topic prefix)

# Axis remap from the Vision Pro / ARKit frame (X right, Y up, -Z forward) into
# MABEL's world frame (robot front = -X, robot left = -Y, up = +Z). Applied to
# *relative* motion, so a sign error here is easy to spot and flip:
#   +X_avp (right) -> +Y_robot   |   +Y_avp (up) -> +Z_robot   |   +Z_avp (back) -> +X_robot
AVP_TO_ROBOT = np.array([
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
])

# Cartesian scaling of hand motion (GMR non-uniform local scaling). >1 amplifies
# the operator's reach to cover MABEL's larger workspace; per-axis [fwd,left,up]
# in robot frame.
#
# This is the DEFAULT / fallback gain used when no per-operator calibration is
# available. With a calibrated OperatorProfile (see calibration.py) the Retargeter
# replaces this with a per-user scalar gamma = robot_arm_reach / human_arm_reach so a
# tall and a short operator both map 1:1 onto MABEL's workspace (ACE-style workspace
# remap, x_e = gamma*(x_h - c) + c_t; Yang et al. CoRL 2024, arXiv 2408.11805).
RETARGET_SCALE = np.array([1.3, 1.3, 1.4])

# Reasonable bounds on the per-user scale so a noisy calibration can't produce a
# runaway or inverted gain (clamped around the fixed default above).
RETARGET_GAMMA_MIN = 0.7
RETARGET_GAMMA_MAX = 2.2

# The clutch mapping's amplification (RETARGET_SCALE > 1) is deliberate: the operator
# moves the hands in a small comfortable box and the lazy base/torso/lift followers
# carry the large reaches, so the gain is NOT a literal arm-reach match. Per-user
# calibration therefore CORRECTS the tuned gain rather than replacing it -- it is
# normalised to a reference operator's arm length so a reference-sized operator keeps
# the exact hand-tuned feel and only larger/smaller arms are adjusted:
#   clutch  mode:  scale = RETARGET_SCALE * clamp(ANTHRO_ARM_REF / L_arm_user)
#   shoulder mode: gamma = clamp(robot_arm_reach / L_arm_user)   (absolute 1:1 reach)
# ANTHRO_ARM_REF is the shoulder->wrist length of the operator the gain was tuned for
# (Winter arm ratio at ~1.75 m stature); set it to your own measured arm length to
# anchor "unchanged feel" to you.
ANTHRO_ARM_REF = 0.581   # m; reference operator shoulder->wrist length (~1.75 m stature)

# ARKit camera looks along local -Z; this is the operator's gaze axis.
AVP_GAZE_LOCAL = np.array([0.0, 0.0, -1.0])

# ── Cosine-directional arm retargeting (parity with scripts/avp_teleop.py) ────
# The app's arm IK matches, scale-independently, the DIRECTIONS of the operator's
# arm/palm to the robot's (cosine), plus the hand CONTACT point (~middle fingertip)
# position and a per-joint posture regularization — solved with SLSQP. This is the
# SAME algorithm as the modular research stack (scripts/teleop/arm.py), ported into
# this package (mabel/arm_retarget.py) so the Vision Pro app (bridge -> MabelTeleop)
# and avp_teleop.py produce matching arm + finger motion. Two paths:
#   "cosine" — the ported directional contact-point IK (default; matches avp_teleop)
#   "dls"    — the legacy clutch-relative palm SE(3) target + DLS WholeBodyIK
# The whole-body followers (lazy base/torso/lift) are unchanged in both — the app's
# sparse ARKit input has no body odometry, so it keeps the bounding-box followers.
# NOTE: the WebSocket bridge (apps) now runs the FULL avp_teleop stack via
# server/avp_engine.py, so this flag only affects the desktop viewer's MabelTeleop;
# kept at the legacy "dls" so that path is unchanged.
ARM_RETARGET = "dls"

ARM_W_WRIST  = 600.0     # CONTACT-point position weight (precise; lets the hands meet)
ARM_W_ELBOW  = 30.0      # elbow direction from the shoulder (only when an elbow is sent;
                         # ARKit gives none -> term auto-disables, like avp_teleop.py)
ARM_W_ORIENT = 20.0      # palm orientation cosine (forward + normal axes)
ARM_W_SMOOTH = 1.0       # temporal smoothness (lower = snappier wrist tracking; 2.0 read as lag)
ARM_MAXITER  = 35        # SLSQP iterations per frame
ARM_OPTIMIZE_WRIST = True  # ORCA wrist solved as a jointly-optimised 8th DoF (natural flip)
ARM_REG_WRIST = 0.03     # light reg on the optimised ORCA wrist (free to flip, returns to rest)
# Per-arm-joint posture regularization toward rest (Arm1..7): light shoulder position,
# HARD shoulder-roll (Arm3) + wrist-yaw (Arm7) so roll/orientation are done by the
# forearm (Arm5) + wrist pitch (Arm6), not by spinning the shoulder/last joint.
ARM_REG_W = np.array([0.08, 0.08, 0.40, 0.08, 0.02, 0.08, 0.60])
ARM_J7_YAW_LIM = 0.45    # rad (~26 deg); clip Arm7's optimisation bound about rest
ARM_WRIST_FRONT = -0.05  # m; chassis-x of the body front line — targets are clamped in front
HAND_REACH = 0.14        # m; ORCA wrist -> contact point (~middle fingertip) along palm forward
# Human->robot reach scale for the contact-point mapping (head->fingertip). Used when
# no per-operator profile is set; with calibration the per-user gamma takes over. This
# is avp_teleop.py's REACH_SCALE (auto-calibrated there; the OperatorProfile here).
REACH_SCALE = 1.2
# Keypoint bodies the arm IK reads (shoulder/elbow), per side.
ARM_SHOULDER = {"l": "l_arm_mount", "r": "r_arm_mount"}
ARM_ELBOW    = {"l": "l_elbow",     "r": "r_elbow"}

# Pinch -> grasp mapping (thumb tip = MP joint 4, index tip = MP joint 8).
PINCH_OPEN_DIST  = 0.08   # m, fingers apart  -> grasp = 0
PINCH_CLOSE_DIST = 0.02   # m, fingers together -> grasp = 1

# ── Lazy-base reactive follower (hand-centroid comfort box, robot frame) ──────
# The base only moves when the hand centroid leaves this comfort box, keeping
# the targets inside the arms' reachable workspace (MABEL's analogue of GMR /
# HoMMI root-translation handling). Distances in metres, robot-local frame.
# e_fwd is measured from the chassis centre, which sits under the torso, so
# e_fwd == 0 is the body line. FWD_MAX is the max forward arm reach; FWD_MIN is
# pinned to the body line so the base retreats the instant a wrist crosses
# behind the torso, preventing the arms from reaching backwards and twisting.
LAZY_FWD_MIN, LAZY_FWD_MAX = 0.0, 0.80
LAZY_LEFT_HALFSPAN         = 0.25
LAZY_YAW_DEADBAND          = 0.30
LAZY_KP_LIN                = 1.5
LAZY_KP_ANG                = 1.5

# ── Lazy lift (vertical reach follower) ──────────────────────────────────────
# In manipulation mode the arms are solved as 8 DOF (lift+torso excluded), so the
# lift creeps to keep the hand centroid near a comfortable height instead of
# being part of the fast IK. Height is measured relative to the chassis; the
# Lift actuator (slide, ctrlrange [0, 0.635]) raises the whole upper body.
LAZY_LIFT_REF       = 0.63    # m above chassis = comfortable hand height (rest palm)
LAZY_LIFT_HALFSPAN  = 0.12    # m deadband before the lift reacts
LAZY_LIFT_KP        = 1.2     # 1/s
LAZY_LIFT_MAX_SPEED = LIFT_HW_MAX_SPEED    # m/s cap on lift retarget travel (hardware-limited)

# ── Lazy torso (forward-reach lean follower) ─────────────────────────────────
# The Torso hinge (range [-1.047, 0.262] rad; negative = lean forward) leans the
# body forward for moderate forward reach before the base has to translate.
LAZY_TORSO_FWD_MAX   = 0.45   # m forward (chassis frame) before the torso leans
LAZY_TORSO_KP        = 1.5    # 1/s  (applied toward a forward lean, i.e. -angle)
LAZY_TORSO_MAX_SPEED = 0.8    # rad/s cap on torso pitch rate

# ── Navigation mode (drone-style virtual joysticks) ──────────────────────────
# Joystick axes are normalised [-1, 1]. Left stick = translate, right stick =
# yaw (x) + lift (y). These set the full-deflection command magnitudes.
NAV_MAX_VX     = 0.8    # m/s   forward/back at full left-stick Y
NAV_MAX_VY     = 0.8    # m/s   strafe at full left-stick X
NAV_MAX_OMEGA  = 1.2    # rad/s yaw at full right-stick X
NAV_MAX_LIFT   = LIFT_HW_MAX_SPEED   # m/s   lift travel at full right-stick Y (hardware-limited)
NAV_DEADZONE   = 0.08   # ignore small stick noise around centre

# ── Wrist-frame alignment (operator wrist -> robot palm) ─────────────────────
# Relative-rotation retargeting maps operator wrist rotations 1:1 into the robot
# palm. If the headset's ARKit wrist-joint axes differ from this assumption,
# replace identity with a fixed 3x3 rotation so the palm orientation matches the
# operator's hand exactly. Identity = pass the relative rotation through unchanged.
WRIST_ALIGN = np.eye(3)


# ── Operator anthropometry & per-user calibration ────────────────────────────
# A one-time, session-start calibration measures the operator's body/hand
# dimensions (see calibration.py) and feeds them into the retargeting scale, the
# shoulder anchor, and the per-finger hand retargeter, so teleop maps 1:1 across
# operators of different size. The math is grounded in the canonical
# Winter / Drillis-Contini segment-length anthropometry (fractions of stature H)
# and the sparse-3-point VR-teleop literature (EgoPoser T-pose ratios 2308.06493;
# ACE workspace scaling 2408.11805; AnyTeleop hand-size alpha 2307.04577).
#
# Anthropometric ratios (Winter, "Biomechanics and Motor Control of Human
# Movement", Fig 4.1; fractions of stature H unless noted):
ANTHRO_ARM_RATIO       = 0.332   # shoulder->wrist length / H  (upper arm + forearm)
ANTHRO_UPPER_ARM_FRAC  = 0.56    # upper-arm fraction of the shoulder->wrist length
ANTHRO_FOREARM_FRAC    = 0.44    # forearm fraction of the shoulder->wrist length
ANTHRO_BIACROMIAL      = 0.259   # shoulder (biacromial) width / H
ANTHRO_ACROMION_BELOW_EYE = 0.118  # eye-line -> acromion drop / H (head above shoulder)
ANTHRO_EYE_HEIGHT      = 0.936   # eye height / H  (HMD floor->eye, for H from the headset)

# Robot-side reference bodies for the per-user scale gamma. The robot arm reach is
# the rest-pose distance from the shoulder body to the palm body (computed live by
# FK in retarget.py, so it tracks the model, not a hard-coded number).
LEFT_SHOULDER  = "l_shoulder_1"
RIGHT_SHOULDER = "r_shoulder_1"

# Arm/EE anchor for the retargeting scale:
#   "clutch"   — keep the existing UMI-style relative mapping, just swap the fixed
#                RETARGET_SCALE for the per-user scalar gamma (low-risk default).
#   "shoulder" — anchor the palm target at the calibrated shoulder centre,
#                p_ee = c_sh_robot + gamma*(p_wrist - c_sh_human) (more anatomically
#                faithful reach; falls back to "clutch" when c_sh is missing/noisy).
RETARGET_ANCHOR = "clutch"

# Calibration capture: each guided pose is accepted once the relevant joints hold
# still for a short dwell, then averaged over a window to denoise. Stillness is judged
# by the POSITIONAL SPREAD over a short window (max deviation of the tracked points
# from their window mean), NOT frame-to-frame speed -- the latter is dominated by
# tracker jitter (a few mm at 30-90 Hz reads as ~0.1 m/s) and would never settle.
CALIB_HOLD_JITTER  = 0.02    # m; max point spread over the window to count as "still"
CALIB_STILL_WINDOW = 0.25    # s; trailing window the spread is measured over
CALIB_POSE_NOVELTY = 0.08    # m; a new still-hold this far from the last captured pose
                             # re-arms the next capture (robust to discrete frame jumps
                             # where the motion transient itself is never observed)
CALIB_HOLD_TIME    = 0.5     # s; required continuous still-hold before a pose is captured
CALIB_AVG_WINDOW   = 0.8     # s; trailing window averaged once a hold is detected
CALIB_MIN_SAMPLES  = 8       # frames; minimum samples in the window for a valid capture
CALIB_PROFILE_DIR  = "profiles"   # where OperatorProfile JSONs are cached (per user)

# Finger retargeter selection (benchmarked head-to-head in the experiments doc):
#   "keypoint" — the existing DexPilot-style keypoint IK (hand_ik_retarget.py),
#                now fed CALIBRATED per-finger lengths instead of a live estimate.
#   "vector"   — AnyTeleop-style scale-invariant vector optimization with the
#                per-user hand-size alpha (hand_vector_retarget.py).
FINGER_RETARGET = "keypoint"
# Solver for the keypoint retargeter. "slsqp" uses the EXACT reference HandOptimizer
# cost (cosine shape + relative pinch + smoothness + reg) from scripts/avp_teleop.py
# (parity); "gn" is the faster damped-Gauss-Newton minimizing the same shape/pinch
# terms (~0.5 ms vs ~5 ms). "gn" is the desktop-viewer default; the apps use the full
# avp_teleop HandOptimizer via server/avp_engine.py.
HAND_SOLVER = "gn"


# ── Teleop fail-safe (tracking loss -> ease HOME -> ease back) ────────────────
# When the operator removes the headset (the stream stalls) or a hand/arm drops out,
# the EngagementSupervisor slowly eases the whole upper body to a safe HOME pose and
# freezes the base; when tracking resumes it eases back from HOME to the operator's
# CURRENT pose (no snap). See mabel/safety.py.
SAFE_STREAM_TIMEOUT = 0.40   # s with NO operator frame -> headset off / link down
SAFE_SIGNAL_TIMEOUT = 0.50   # s a required signal (head / hand / elbow) stale -> lost
SAFE_RETREAT_TIME   = 4.0    # s to ease from the live pose -> HOME (slow, gentle)
SAFE_ENGAGE_TIME    = 1.5    # s to ease from HOME -> live pose on re-acquire (2.5 felt sluggish)
# SAFE PARK POSE (operator request 2026-07-27): HOME must be the pose where cutting
# power is harmless — lift at HALF its 0.635 m travel (the worm gear holds unpowered
# at any height), torso at its +15 deg lean-BACK end stop (a torque cut settles the
# chest into the stop instead of flopping it forward).
SAFE_HOME_LIFT      = 0.3175  # m; lift height at HOME (half of the 0..0.635 travel)
SAFE_HOME_TORSO_DEG = 15.0    # deg; torso lean-BACK at HOME (+ = back; clipped to the
                              # actuator ctrlrange, which caps at exactly +15 deg)
SAFE_REQUIRE_BOTH_HANDS = False  # True: both hands required; False: either hand keeps live

# ── Clutch / re-indexing (single-hand DOUBLE-PINCH toggles a whole-body freeze) ─
# Pinch EITHER hand twice quickly (a double-click) to FREEZE the robot and recenter;
# double-pinch again to RELEASE. A single pinch is left free for grasping. See
# mabel/safety.py (ClutchController).
CLUTCH_ENABLE        = True
CLUTCH_PINCH_ON      = 0.030  # m; thumb-index gap below this = pinched (closing edge)
CLUTCH_PINCH_OFF     = 0.055  # m; above this = released (hysteresis, no chatter)
CLUTCH_DOUBLE_WINDOW = 0.60   # s; the two pinches must land within this to count
CLUTCH_COOLDOWN      = 0.80   # s; lockout after a toggle (one gesture = one toggle)


# ── PURE PD (the default torque law, 2026-08-23) ─────────────────────────────
# Two numbers, not twenty-eight gains. Each joint's kp and kd are derived from
# its OWN reflected inertia:  kp_j = M_j w_n^2,  kd_j = 2 zeta M_j w_n, so the
# tracking lag kd/kp = 2 zeta / w_n is identical across the arm while the damping
# ratio is right for every joint individually.
#
# Chosen by sweep against the MuJoCo model (controller/experiments/exp_pd_tuning.py,
# phase P2) — lowest tracking lag that does not ring, parks accurately, and does
# not pump torque into the retargeting's own command jitter:
#
#                        shipped uniform 500/45     per-joint w60 zeta1.0
#     tracking lag             89.5 ms                    32.1 ms
#     rise (10-90%)           190   ms                    66   ms
#     settle (2%)             364   ms                   120   ms
#     overshoot                 0.0 %                      2.3 %
#     steady-state error        1.0 mrad                   1.9 mrad
#     torque on 2 mrad         16.3 % of F_lim            10.3 % of F_lim
#       of command jitter
#
# The last row is the "not too stiff" check and it goes the RIGHT way: normalised
# by each joint's own actuator, the per-joint set is quieter, because one uniform
# kp = 500 was 14 % of full scale on a 7 N.m wrist and 2.5 % on the 40 N.m
# shoulder. Doc: controller/docs/arm_pd_tuning.tex.
PD_WN = 60.0      # rad/s  (9.5 Hz closed-loop bandwidth)
PD_ZETA = 1.0     # critically damped: fastest response that does not overshoot
