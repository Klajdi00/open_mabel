"""
wire_protocol.py — Python mirror of the Swift WireEnvelope / TeleopFrame types.

JSON field names exactly match the Swift Codable defaults (camelCase). The
WireCodec helpers encode/decode the same two-layer envelope the Swift app uses:

  { "type": "teleop_frame", "payload": { ...TeleopFrame... } }

The codec is intentionally dependency-free (stdlib only) so it loads fast
and can be imported in ROS2 nodes without extra packages.
"""
from __future__ import annotations
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Enums (raw values must match Swift enum rawValues exactly)
# ---------------------------------------------------------------------------

class WireMessageType(str, Enum):
    TELEOP_FRAME = "teleop_frame"
    ROBOT_STATE  = "robot_state"
    PING         = "ping"
    PONG         = "pong"
    ERROR        = "error"
    HELLO        = "hello"
    RESET        = "reset"          # client → sim: reset the simulation to home
    RECALIBRATE  = "recalibrate"    # client → sim: re-anchor the retargeting clutch
    # Input-device-agnostic commands (iOS app, future controllers, web):
    JOINT_COMMAND = "joint_command"  # direct joint targets {"joints": {id: rad}}
    CONTROL_MODE  = "control_mode"   # structured control spec — see below.
                                     # Additive booleans (both optional, both
                                     # MERGED not replaced, so a client may send
                                     # one and leave the other alone):
                                     #   "gravityComp": bool — the upper-body
                                     #     gravity feed-forward. Default TRUE;
                                     #     OFF lets the arm sag on servo
                                     #     stiffness alone (diagnostics/demos).
                                     #   "wholeBodyRetarget": bool — which
                                     #     retargeting profile the arms run.
                                     #     TRUE = the ORIGINAL whole-body map
                                     #     (follows the operator's whole arm,
                                     #     elbow included); FALSE (default) =
                                     #     the front-workspace tabular profile
                                     #     that prioritises WRIST position and
                                     #     palm direction.
    EXTERNAL_FORCE = "external_force" # {"forces": {body_name: [fx,fy,fz]}} operator push anywhere
    RECORD        = "record"         # {"active": bool, "name": str?} start/stop an episode
    ESTOP         = "estop"          # {"active": bool} operator emergency stop / release.
                                     # Also gates HAND TORQUE on the real robot: release
                                     # (active=False) energises the hands (/<side>_hand/
                                     # torque_enable=True); engage releases them to
                                     # free-drive. Edge-driven, so hands never auto-
                                     # energise on bring-up.
    SET_ROBOT_NAME = "set_robot_name" # {"name": str} rename the robot (persists + syncs)
    SET_SPEED_CAPS = "set_speed_caps" # {"vmax", "wmax"} live base speed caps
    BODY_CALIBRATION = "body_calibration"  # {"action": "start"|"cancel", "user": str} per-operator anthropometry
    NAV_GOAL      = "nav_goal"        # {"x","y","yaw"?} Nav2 goal in the map/odom frame (→ /goal_pose, NavFn A* + MPPI)
    NAV_GOAL_PREVIEW = "nav_goal_preview"  # {"x","y","yaw"?} STAGED goal: show on RViz (/goal_pose_preview) + reachability check, NO drive
    NAV_CANCEL    = "nav_cancel"      # cancel the active Nav2 goal (operator stick takeover / clear)
    NAV_RESULT    = "nav_result"      # server → client: {"status": "reached"|"unreachable"|"aborted", "msg": str} goal outcome
    DATA_COLLECT  = "data_collect"    # {"active": bool, "label": str?} start/stop a rosbag (MCAP) data-collection session
    LIST_MAPS     = "list_maps"       # client → server: request the saved-room list
    MAP_LIST      = "map_list"        # server → client: {"maps": [{"name","has_mesh","has_reloc"}], "loaded": str?}
    LOAD_MAP      = "load_map"        # client → server: {"name": str} load a saved room ("" / "live" → back to live SLAM)
    SAVE_MAP      = "save_map"        # client → server: {"name": str} serialize the LIVE map into the map dir as <name> (2D pgm/yaml + nvblox mesh) → MAP_LIST
    DELETE_MAP    = "delete_map"      # client → server: {"name": str} delete a saved room's whole <name>.* set → MAP_LIST
    POLICY_SELECT    = "policy_select"     # client → server: {"name": str|null} select (not run) a learned
                                           # policy; echoed in robot_state.policy + :9600 status. A runner
                                           # plugs into the RobotCommand seam later (robot_command.py).
    ANIMATION_SELECT = "animation_select"  # client → server: {"name": str|null} select a scripted
                                           # movement/animation; echoed like policy_select.
    # ── Animation demos (animation_demo/) ────────────────────────────────────
    # Record a human demonstration and replay it through the SAME controller.
    # Deliberately NOT folded into RECORD: that one drives CurationRecorder and
    # writes a policy-training .h5 of the robot's realized state. A demo clip
    # stores the controller's INPUTS (`_joint_cmd` + base twist) so replay can
    # push them back through `MabelTeleop.joint_targets`. Two different payloads
    # for two different jobs; the app's Record menu picks between them.
    DEMO_RECORD  = "demo_record"    # client → server: {"active": bool, "name": str,
                                    #   "countdown": float=3.0, "notes": str?}
                                    #   active=true ARMS the take — capture starts
                                    #   when the countdown expires, timed by the
                                    #   server clock (an app-side countdown drifts
                                    #   by the link latency).
    DEMO_REPLAY  = "demo_replay"    # client → server: {"active": bool, "name": str,
                                    #   "repeats": int=1, "speed": float=1.0}
                                    #   active=false winds the player down through
                                    #   its return-home ramp; it is not a freeze.
    DEMO_DELETE  = "demo_delete"    # client → server: {"name": str}
    DEMO_LIST    = "demo_list"      # client → server: request the clip library
    DEMO_CLIPS   = "demo_clips"     # server → client: {"clips": [...]} — also
                                    #   mirrored in robot_state.animation so a
                                    #   client that missed the reply still sees it.
    OPERATOR_POINTS  = "operator_points"   # bi-directional retarget-debug overlay:
                                           # client → server {"on": bool} opts this client in;
                                           # server → client {"t", "sides": {side: {"target","elbow",
                                           # "fwd","up","hand":[[xyz]x21]}}} — the EXACT wrist
                                           # target / elbow hint / palm axes / hand skeleton the
                                           # arm IK consumed, in ROBOT/ODOM world (the engine's
                                           # own retargeting map — never a re-derived transform),
                                           # at VP frame rate while whole-body retarget runs.
    AUDIO_CONFIG     = "audio_config"      # client → server: {"micEnabled": bool} mute/unmute the
                                           # ROBOT's mic; echoed in robot_state.audio. CONTROL only —
                                           # the PCM itself rides the dedicated duplex :9610 plane
                                           # (AUDIO 0x07), never this socket: a continuous 32 kB/s
                                           # firehose through the 50 Hz control loop is the exact
                                           # starvation perception_server.py was split out to fix.
                                           # See docs/data_streams.md + audio_streams.py.
    # Server → client additions ride EXISTING message types only (robot_state
    # gained additive keys: estop, controlMode, dataCollection, policy,
    # animation, battery.available) so older app builds keep working unchanged.
    # The 96xx read-only data plane is documented in docs/data_streams.md.


# ── The three orthogonal teleop axes (INPUT → METHOD → CONTROLLER → ROBOT) ────
# A CONTROL_MODE payload now carries a structured spec rather than one flat enum:
#   {"method": <TeleopMethod>, "controlType": <ControlType>,
#    "region": <ControlRegion>, "stiffness": K, "scope": <RetargetScope>}
# Legacy payloads with just {"mode": ...} are still accepted (see LEGACY_MODES).
#
# `scope` is ADDITIVE and orthogonal to `region`. Region scopes which JOINTS the
# torque engine owns; scope decides whether the MOBILE BASE is in the retargeting
# follow loop at all:
#   "local"   (default) arms + hands + torso + lift + neck follow the operator;
#             the base stays put and the operator drives it on the joystick.
#   "global"  the base joins the loop, taking the overflow past the comfort box
#             (translation) and past the yaw deadband (orientation).
# Absent or unrecognised => "local": a moving base is the consequential state, so
# it is asked for, never inherited from a stale or silent client.
#
# (Not to be confused with the retired `frame` key below — that was about which
# MEASUREMENTS the estimator could believe. `scope` is about which DoFs are in
# the optimisation.)
# There is NO `frame` key. A local/global retargeting-mode axis shipped briefly
# (2026-07-28) and was removed the same day by operator directive: retargeting
# is GLOBAL only — walking and turning drive the base. The offline study lives
# in controller/mabel/retargeting/ + controller/docs/retargeting_modes.pdf; do
# not reintroduce a `frame` key on the wire.

class ControlType(str, Enum):
    """How the robot realises a reference — the control law on the joints."""
    IMPEDANCE  = "impedance"    # rigid: track the target with a spring (gravity-comp)
    COMPLIANCE = "compliance"   # soft: yield to external force (admittance)


class ControlRegion(str, Enum):
    """Which part of the body the controller owns."""
    ARM        = "arm"          # arms only
    UPPER_BODY = "upper_body"   # arms + torso (+ lift held)
    WHOLE_BODY = "whole_body"   # + mobile base (admittance glide in compliance)


class RetargetScope(str, Enum):
    """Which DoFs the whole-body retargeting follow loop owns.

    Orthogonal to ControlRegion: region is about which joints the torque engine
    holds, this is about whether the mobile base follows the operator.
    """
    LOCAL  = "local"    # arms+hands+torso+lift+neck; base is joystick-driven
    GLOBAL = "global"   # + the mobile base, past its translation/yaw deadbands


class TeleopMethod(str, Enum):
    """How the human's intent is mapped to a robot reference."""
    JOINT      = "joint"        # per-joint angle targets (sliders / toggles)
    NAVIGATION = "navigation"   # base twist + lift rate (drive joysticks)
    WRIST      = "wrist"        # palm SE(3) target → arm IK (ball drag / joystick)
    RETARGET   = "retarget"     # whole-body IK from hand/elbow/torso tracking (headset)


# Legacy flat mode string → (method, control type, region). Keeps older app
# builds working while the structured spec rolls out.
LEGACY_MODES: dict[str, tuple] = {
    "navigation":   (TeleopMethod.NAVIGATION, ControlType.IMPEDANCE,  ControlRegion.WHOLE_BODY),
    "gravity_comp": (TeleopMethod.WRIST,      ControlType.COMPLIANCE, ControlRegion.ARM),
    "compliance":   (TeleopMethod.WRIST,      ControlType.COMPLIANCE, ControlRegion.WHOLE_BODY),
    "joints":       (TeleopMethod.JOINT,      ControlType.IMPEDANCE,  ControlRegion.ARM),
}


class TrackingState(str, Enum):
    NORMAL        = "normal"
    LIMITED       = "limited"
    NOT_AVAILABLE = "notAvailable"


class Chirality(str, Enum):
    LEFT  = "left"
    RIGHT = "right"


class TeleopMode(str, Enum):
    MANIPULATION   = "Arms & Hands"
    BASE_DRIVING   = "Base Driving"


class RobotMode(str, Enum):
    IDLE        = "idle"
    TELEOP      = "teleop"
    AUTONOMOUS  = "autonomous"
    ESTOPPED    = "estopped"
    RECOVERING  = "recovering"


class DiagLevel(str, Enum):
    INFO    = "info"
    WARNING = "warning"
    ERROR   = "error"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Transform4x4:
    """4×4 row-major transform matrix (16 floats)."""
    matrix: list[float]  # len == 16

    @classmethod
    def identity(cls) -> "Transform4x4":
        m = [0.0] * 16
        m[0] = m[5] = m[10] = m[15] = 1.0
        return cls(matrix=m)

    def position_xyz(self) -> tuple[float, float, float]:
        """Extract translation: column 3 of the matrix (row-major index 3,7,11)."""
        return (self.matrix[3], self.matrix[7], self.matrix[11])

    def to_numpy(self):
        """Return a (4,4) numpy array — only imported when called."""
        import numpy as np
        return np.array(self.matrix, dtype=float).reshape(4, 4)


@dataclass
class HeadPose:
    transform: Transform4x4
    trackingState: TrackingState = TrackingState.NORMAL


@dataclass
class HandJointPose:
    joint: str                   # HandJoint raw value string
    localTransform: Transform4x4
    isTracked: bool


@dataclass
class HandPose:
    chirality: Chirality
    anchorTransform: Transform4x4
    joints: list[HandJointPose]
    isTracked: bool


@dataclass
class NavJoystick:
    """Drone-style virtual joysticks for navigation mode, normalised [-1, 1].
    Left stick = translate (x=strafe, y=fwd/back); right stick = yaw (x) + lift (y)."""
    lx: float = 0.0
    ly: float = 0.0
    rx: float = 0.0
    ry: float = 0.0


@dataclass
class TeleopFrame:
    sequence: int
    timestamp: float
    head: HeadPose
    leftHand: HandPose | None = None
    rightHand: HandPose | None = None
    mode: TeleopMode = TeleopMode.MANIPULATION
    navJoystick: NavJoystick | None = None


@dataclass
class Battery:
    percentage: float = 0.0
    voltage: float = 0.0
    isCharging: bool = False


@dataclass
class Diagnostic:
    level: DiagLevel
    component: str
    message: str


@dataclass
class RobotState:
    timestamp: float = field(default_factory=time.time)
    mode: RobotMode = RobotMode.IDLE
    battery: Battery = field(default_factory=Battery)
    jointPositions: dict[str, float] = field(default_factory=dict)
    diagnostics: list[Diagnostic] = field(default_factory=list)
    latencyMs: float = 0.0
    # Per-operator body calibration (see controller/mabel/calibration.py):
    #   calibrationStatus: {"active", "pose", "index", "total", "done"} while a capture
    #     is running (null when idle); the app shows live pose prompts + progress.
    #   operatorProfile: the measured OperatorProfile dict (height_m, arm_length_m, ...)
    #     so the app can display + persist the result. Null until a profile exists.
    calibrationStatus: dict | None = None
    operatorProfile: dict | None = None
    # Sparse per-joint dynamics (~10 Hz, stamp-gated; additive — apps that don't
    # know the fields ignore them). Keys are the SAME iOS joint ids as
    # jointPositions (URDF snake_case: left_arm_1..right_arm_7, hands, ...);
    # only joints the HAL actually measured appear (absent ≠ zero).
    #   jointVelocities: rad/s        jointTorques: N·m
    #   jointTemps: °C (DaMiao MOSFET / Feetech servo temperature)
    #   motorHealth: merged */motor_health topics {joint: {torqueNm, tempC}}
    jointVelocities: dict[str, float] | None = None
    jointTorques: dict[str, float] | None = None
    jointTemps: dict[str, float] | None = None
    motorHealth: dict | None = None
    # Per-joint SAFE COMMANDABLE range, streamed alongside the low-freq hardware
    # status (module connection roster) so a teleop slider clamps to the REAL,
    # calibrated limits — not the stale baked rig. Keys are the SAME URDF joint ids
    # as jointPositions; values are [lo, hi] in RADIANS, PROCESSED/model frame,
    # already pulled in by the end-stop safety margin (the "gap" between the
    # theoretical range and the true measured limits). The app shows these as
    # DEGREES and never sees the raw encoder frame. Additive: apps that don't know
    # the field ignore it. Idempotent — the full map rides every status tick, so a
    # dropped packet self-corrects and a recalibration flows out automatically.
    jointLimits: dict[str, list[float]] | None = None
    # ECHOED OPERATOR SWITCHES (additive). The app's toggles are the SOURCE, but
    # they must not be the only copy: a client that reconnects, or a second
    # client watching the same robot, has no other way to learn the live state,
    # and a toggle whose UI can disagree with the robot is worse than no toggle.
    # Mirrors the control_mode keys of the same names.
    #   gravityComp:        is the upper-body gravity feed-forward feeding?
    #   wholeBodyRetarget:  TRUE = original whole-body arm map;
    #                       FALSE = front-workspace wrist-focused profile.
    gravityComp: bool | None = None
    wholeBodyRetarget: bool | None = None


# ---------------------------------------------------------------------------
# JSON codec (mirrors WireCodec.swift)
# ---------------------------------------------------------------------------

class WireCodec:
    """Encode/decode the two-layer envelope the Swift app uses."""

    # ---- encode ----------------------------------------------------------

    @staticmethod
    def encode(payload: Any, msg_type: WireMessageType) -> bytes:
        """Wrap *payload* (a dataclass or dict) in a typed envelope and return UTF-8 JSON."""
        envelope = {
            "type": msg_type.value,
            "payload": WireCodec._to_jsonable(payload),
        }
        return json.dumps(envelope, separators=(",", ":")).encode()

    @staticmethod
    def encode_robot_state(state: RobotState) -> bytes:
        return WireCodec.encode(state, WireMessageType.ROBOT_STATE)

    @staticmethod
    def encode_ping() -> bytes:
        return WireCodec.encode({"t": time.time()}, WireMessageType.PING)

    @staticmethod
    def encode_pong() -> bytes:
        return WireCodec.encode({"t": time.time()}, WireMessageType.PONG)

    # ---- decode ----------------------------------------------------------

    @staticmethod
    def decode_envelope(raw: bytes | str) -> tuple[WireMessageType, dict]:
        """Return (type, payload_dict). Raises ValueError on bad input."""
        if isinstance(raw, bytes):
            raw = raw.decode()
        envelope = json.loads(raw)
        try:
            msg_type = WireMessageType(envelope["type"])
        except (KeyError, ValueError) as e:
            raise ValueError(f"Unknown message type: {envelope.get('type')}") from e
        return msg_type, envelope.get("payload", {})

    @staticmethod
    def decode_teleop_frame(payload: dict) -> TeleopFrame:
        """Parse a raw payload dict into a TeleopFrame."""
        return TeleopFrame(
            sequence=payload["sequence"],
            timestamp=payload["timestamp"],
            head=WireCodec._parse_head(payload["head"]),
            leftHand=WireCodec._parse_hand(payload["leftHand"]) if payload.get("leftHand") else None,
            rightHand=WireCodec._parse_hand(payload["rightHand"]) if payload.get("rightHand") else None,
            mode=TeleopMode(payload.get("mode", TeleopMode.MANIPULATION.value)),
            navJoystick=WireCodec._parse_joystick(payload.get("navJoystick")),
        )

    @staticmethod
    def _parse_joystick(d: dict | None) -> "NavJoystick | None":
        if not d:
            return None
        return NavJoystick(lx=float(d.get("lx", 0.0)), ly=float(d.get("ly", 0.0)),
                           rx=float(d.get("rx", 0.0)), ry=float(d.get("ry", 0.0)))

    # ---- private helpers --------------------------------------------------

    @staticmethod
    def _parse_transform(d: dict) -> Transform4x4:
        return Transform4x4(matrix=list(d["matrix"]))

    @staticmethod
    def _parse_head(d: dict) -> HeadPose:
        return HeadPose(
            transform=WireCodec._parse_transform(d["transform"]),
            trackingState=TrackingState(d.get("trackingState", "normal")),
        )

    @staticmethod
    def _parse_hand(d: dict) -> HandPose:
        joints = [
            HandJointPose(
                joint=j["joint"],
                localTransform=WireCodec._parse_transform(j["localTransform"]),
                isTracked=j["isTracked"],
            )
            for j in d.get("joints", [])
        ]
        return HandPose(
            chirality=Chirality(d["chirality"]),
            anchorTransform=WireCodec._parse_transform(d["anchorTransform"]),
            joints=joints,
            isTracked=d["isTracked"],
        )

    @staticmethod
    def _to_jsonable(obj: Any) -> Any:
        """Recursively convert dataclasses / Enums to plain JSON-serialisable types."""
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, dict):
            return {k: WireCodec._to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [WireCodec._to_jsonable(v) for v in obj]
        # dataclass
        if hasattr(obj, "__dataclass_fields__"):
            return {k: WireCodec._to_jsonable(getattr(obj, k))
                    for k in obj.__dataclass_fields__}
        return obj
