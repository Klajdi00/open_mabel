"""
Retarget Lab configuration — ports, paths, env overrides.

Studio-family conventions (see control_lab/server/config.py): walk-up repo
root, env-overridable ports. Ports:
  * 8120  the studio (HTTP + /ws/state)                MABEL_RTL_PORT
  * 9090  the Vision Pro gateway (/teleop)             MABEL_RTL_VP_PORT
          — the apps hard-code :9090, so the lab serves it there. The sim/real
            gateway uses the same port: stop it first.
"""
import os
from pathlib import Path

_HERE = Path(__file__).resolve().parent          # server/
PKG = _HERE.parent                               # retarget_lab/
REPO = PKG.parent.parent                         # MABEL/

HOST = os.environ.get("MABEL_RTL_HOST", "0.0.0.0")
PORT = int(os.environ.get("MABEL_RTL_PORT", "8120"))
VP_PORT = int(os.environ.get("MABEL_RTL_VP_PORT", "9090"))
# Bonjour service name. Deliberately NOT "MABEL … Sim": the VP app folds
# names that start with "mabel" and contain "sim" into its built-in sim entry
# (RobotStore.mergeDiscovered); a distinct name gets its own robot row.
BONJOUR = os.environ.get("MABEL_RTL_BONJOUR", "1") not in ("0", "false", "no")
BONJOUR_NAME = os.environ.get("MABEL_RTL_BONJOUR_NAME", "Retarget Lab Sim")

STATE_HZ = float(os.environ.get("MABEL_RTL_STATE_HZ", "30"))
ROBOT_STATE_HZ = float(os.environ.get("MABEL_RTL_ROBOT_STATE_HZ", "30"))

# Bare robot + floor, `home` keyframe (lift half-extended, base upright).
SCENE = os.environ.get(
    "MABEL_RTL_SCENE",
    str(REPO / "simulation" / "mabel_mujoco" / "scenes" / "mabel_bare.xml"))

WEB_DIR = PKG / "web"
WEBSITE_ASSETS = REPO / "web_gui" / "shared_assets"     # mabel_rig.glb + mabel_joints.json
OPENARM_MESHES = REPO / "simulation" / "mabel_mujoco" / "models" / "meshes"   # OpenArm v1 gripper (end_effector.py)
CONTROLLER_DIR = REPO / "controller"
EXPERIMENTS_DIR = CONTROLLER_DIR / "experiments"

# Recorded human episodes (raw teleop_frame wire payloads, JSONL.gz).
DATA_DIR = Path(os.environ.get("MABEL_RTL_DATA", str(PKG / "data")))
EPISODES_DIR = DATA_DIR / "episodes"
EVAL_DIR = DATA_DIR / "evaluations"
# Read-only extra corpus: the EgoDex benchmark clips (CC-by-NC-ND, local only).
EGODEX_CLIPS = EXPERIMENTS_DIR / "data" / "egodex" / "benchmark" / "clips"

# WHOLE-BODY settings every variant runs under (kinematic: nothing here can
# move hardware). Base: GLOBAL retarget scope — the base follows the head
# overflow / body heading and is integrated kinematically, so the robot walks
# and turns in the world (and on the headset). Lean: the hip-pivot trunk-angle
# torso law (hyperparams TORSO_LEAN_MODE) instead of the box-clipped one that
# saturates at 17 deg. Clutch: the double-pinch freeze is OFF — in a lab whose
# operators grasp things, two quick pinches froze every retargeter until the
# end of the take (measured: 9.4 s of a 69 s recording).
BASE_FOLLOW = os.environ.get("MABEL_RTL_BASE", "1") not in ("0", "false", "no")
TORSO_LEAN_MODE = os.environ.get("MABEL_RTL_TORSO_LEAN", "angle")
CLUTCH = os.environ.get("MABEL_RTL_CLUTCH", "0") not in ("0", "false", "no")
# Swerve: the base twist is realised through the per-module actuator model
# (controller/mabel/motion_model/swerve_modules.py) — wheel-speed and drive-torque
# caps with margin, the steer at full NEO 550 speed, the controller's drive choke
# while a wheel is still turning. Off = the twist is realised exactly (holonomic).
SWERVE_MODEL = os.environ.get("MABEL_RTL_SWERVE", "1") not in ("0", "false", "no")
# Whose base: "mabel" (the controller's swerve law, MABEL's triangle) or "nova"
# (NOVA's driver: its triangle, open-loop duty top speed, 0.75 m/s^2 slew, twist
# low-pass). NOVA's gateway runs this same engine with "nova".
BASE_PROFILE = os.environ.get("MABEL_RTL_ROBOT", "nova").strip().lower()   # the lab IS NOVA's deployment (2026-10-22)
# Whole body vs upper body (the studio's toggle; the headset's control_mode.scope on
# NOVA): "global" = the base follows the operator, "local" = it holds.
SCOPE = os.environ.get("MABEL_RTL_SCOPE", "global").strip().lower()
# Heading + base follow (teleop_engine/hyperparams.py HEADING_MODE / BASE_TRANSLATION_MODE):
# the drift-free lazy-neck head heading and the 1:1 body-frame base follow, measured in
# controller/experiments/exp_heading.py and E6. "bodyframe"/"shipped" = the robot defaults.
HEADING_MODE = os.environ.get("MABEL_RTL_HEADING", "head").strip().lower()
BASE_TRANSLATION_MODE = os.environ.get("MABEL_RTL_BASE_TRANSLATION", "body").strip().lower()
# PHYSICAL SERVO: every joint follows its command at the robot's own setpoint SLEW
# (rad/s, m/s) instead of arriving at once — the HAL's slew is a hard rate limit on the
# real robot, so a map that asks for more than it gets lags exactly like this. "ideal"
# = the old instant servo. Slews per robot, from the robots' own configs:
#   MABEL  hardware_bridge/calibration_data/mabel.json arm gains vmax 0.8 (all 14),
#          mabel_hw/config.py TORSO slew 0.8, NECK yaw_slew 1.5
#   NOVA   its calibration tree: arm j1-j2 2.5, j3-j4 3.0, j5-j7 4.0; torso 1.5; neck 1.5
#   lift   0.03 m/s both (operator, 2026-10-15)
SERVO = os.environ.get("MABEL_RTL_SERVO", "slew").strip().lower()
# ABLATION CHANNELS (studio switches; settings "torso" / "neck" / "lift"): the same
# retargeter runs, then each channel is
#   "free"   the engine drives it (the lab's "Ours") — and that IS what the robot runs;
#   "fixed"  parked at HOME after the solve (torso 0, neck level, lift at the home
#            keyframe 0.3175 m) with the engine's own ease (safety.ChannelPark, 2.5 s).
# There is no NOVA channel (2026-10-18, operator: "the lab testing is the NOVA deployed one").
# Robot "nova" puts NOVA's own windows on the engine instead (worker._robot_limits), and
# ../NOVA/controller/lab_retarget.py sends the engine's arms, torso, NECK and (whole body)
# lift as they are: torso within NOVA_TORSO_RANGE (= its TORSO_CMD_RANGE), neck within
# NOVA_NECK_RANGE (= its calibrated servo windows; pitch + = look down). lab_retarget.py's
# offline check asserts both equal NOVA's own numbers.
CHANNELS = {"torso": ("free", "fixed"), "neck": ("free", "fixed"), "lift": ("free", "fixed")}
# END EFFECTOR (studio toggle, setting "ee"): MABEL's ORCA hand v1, or the ORIGINAL OpenArm v1 parallel gripper
# on J7 (teleop_engine/end_effector.py: the arm then tracks the gripper base face, 89.6 mm out from the wrist centre J5 x J6 x J7, and the operator's hand
# closure sets the opening). Applies to every variant; the RL policies keep their own arm law (only the fingers
# and the drawing change).
EE_KINDS = ("orca", "gripper")
NOVA_TORSO_RANGE = (-1.0, 0.20)
NOVA_NECK_RANGE = {"neck_yaw": (-1.570796, 1.570796), "neck_pitch": (-1.570796, 0.925025),
                   "neck_roll": (-0.785398, 0.785398)}
NOVA_DIR = Path(os.environ.get("MABEL_RTL_NOVA", str(REPO.parent / "NOVA")))
# Physics: pre-start NOVA's sim stack with the lab so the Physics toggle is ~1 s, not ~10 s.
PHYSICS_PREWARM = os.environ.get("MABEL_RTL_PHYSICS_PREWARM", "1") not in ("0", "false", "no")
SLEWS = {
    "mabel": {"arm": [0.8] * 7, "torso": 0.8, "neck": 1.5, "lift": 0.03},
    "nova":  {"arm": [3.5, 3.5, 6.0, 6.0, 10.0, 10.0, 10.0], "torso": 1.5, "neck": 1.5, "lift": 0.03},
}


def _nova_tree_slews():
    """NOVA's arm slews from ITS calibration tree (arm/openarm-dual/gains <j>.vmax, what its HAL enforces), so the lab
    never drifts from the robot; the table above is the fallback."""
    import json
    try:
        g = json.load(open(NOVA_DIR / "hardware_bridge" / "calibration_data" / "nova.json"))["arm"]["openarm-dual"]["gains"]
        return [float(g[f"r_j{i}"]["vmax"]) for i in range(1, 8)]
    except Exception:                                            # noqa: BLE001 — no NOVA checkout: the fallback
        return None


SLEWS["nova"]["arm"] = _nova_tree_slews() or SLEWS["nova"]["arm"]
# NOVA's IK speed caps (worker._robot_limits): the retargeter must be allowed to ask what the robot can now do.
# MEASURED 2026-10-22 (a sibling robot's whole teleop path in the sim fitted to its arms, every
# motor's commanded torque checked against its peak): self-collision QP 4 -> 10 rad/s and J5-J7 trust 5 -> 10, with
# NOVA's new slews / gains. J3 stays at the proximal 2 rad/s ON PURPOSE: at 4 the palm flip finished faster (p90 1.10
# vs 1.46 s) but the upper arm whipped the loaded hand, and J7's own damping then asked 0.92 of the J4310's peak
# (0.74 at 2, today's level) — the current spike that trips the motor's over-current protection. MABEL keeps the
# engine's defaults.
NOVA_IK_CAPS = dict(qp_vmax=10.0, distal=10.0, override={})

# Synthetic clock for offline evaluation (the engine is driven at the
# episode's own frame times; this is only the fallback when stamps are absent).
EVAL_DT = 1.0 / 60.0


def lan_ip():
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"
