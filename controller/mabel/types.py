"""Frontend-agnostic data contracts for the MABEL teleop library.

Every frontend (Apple Vision Pro, iOS, web, other VR) only ever produces an
OperatorPose and consumes a RobotCommand -- it never touches MuJoCo, ZMQ, or the
retargeting internals. The documented operator frame is ARKit-style: +Y up, +X right,
-Z forward; the library maps it to the robot (Z up, -X front) internally.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class OperatorPose:
    """One frame of operator pose, device-agnostic. It is duck-compatible with the
    controller's AVPFrame (exposes .head / .right / .left / .right_joints /
    .left_joints / .timestamp), so it can be fed straight to the retargeters, while
    also carrying the extra signals the safety/clutch layers need (pinch, elbows,
    body yaw, head height)."""
    head: object = None              # 4x4 head pose (np.ndarray) or None
    right_hand: object = None        # list of 21 4x4 (MediaPipe), [0]=wrist, or None
    left_hand: object = None
    right_joints: object = None      # {name: 4x4} named skeleton (optional)
    left_joints: object = None
    right_pinch: object = None       # thumb-index gap [m] (drives the clutch)
    left_pinch: object = None
    right_elbow: object = None       # 4x4 (optional)
    left_elbow: object = None
    body_yaw: object = None          # operator torso heading [rad] (optional)
    head_height: object = None       # operator head height [m] (optional)
    timestamp: float = 0.0

    # --- AVPFrame-compatible aliases so the retargeters consume it directly ---
    @property
    def right(self):
        return self.right_hand

    @property
    def left(self):
        return self.left_hand

    @classmethod
    def from_avframe(cls, f):
        """Adapt a parsed AVPFrame (avp_receiver) into an OperatorPose."""
        return cls(head=f.head, right_hand=f.right, left_hand=f.left,
                   right_joints=f.right_joints, left_joints=f.left_joints,
                   timestamp=getattr(f, "timestamp", 0.0))

    @classmethod
    def from_teleop_frame(cls, payload):
        """Build from the apps' `teleop_frame` wire payload (head transform + per-hand
        anchor/named-joint skeletons + derived pinch). See `mabel.wire`. This is what
        the iOS / Vision Pro / web bridge feeds in -- the wire dialect lives here, not
        in the transport."""
        from .wire import operator_pose_from_teleop_frame
        return operator_pose_from_teleop_frame(payload)

    @classmethod
    def from_payload(cls, p):
        """Build directly from a wire payload dict (same schema avp_receiver parses,
        plus the optional right_pinch/left_pinch/body_yaw/head_height/elbow keys)."""
        from .avp_receiver import AVPFrame
        op = cls.from_avframe(AVPFrame(p))
        op.right_pinch = p.get("right_pinch")
        op.left_pinch = p.get("left_pinch")
        op.body_yaw = p.get("body_yaw")
        op.head_height = p.get("head_height")
        for s in ("right", "left"):
            e = p.get(f"{s}_elbow")
            if e is not None:
                setattr(op, f"{s}_elbow", np.asarray(e, float).reshape(4, 4))
        return op

    def signals(self):
        """Presence dict consumed by the EngagementSupervisor freshness check."""
        return {"head_mat": self.head, "right_hand_mats": self.right_hand,
                "left_hand_mats": self.left_hand,
                "right_elbow": self.right_elbow, "left_elbow": self.left_elbow}


@dataclass
class RobotCommand:
    """Robot actuator targets for one frame (frontend-agnostic). In the sim these are
    already in data.ctrl (the facade wrote them); a hardware bridge calls .named(model)
    to get {actuator_name: target} and forwards it to the robot.

    The mobile base is the one exception to "everything is in ctrl": the swerve law
    bakes the base motion into per-wheel steer/drive actuators, but the canonical
    hardware contract commands the base as a BODY TWIST (/mabel_cmd: vx, vy, omega).
    So the controller's already-smoothed, tip-safe twist is surfaced here directly,
    ready to publish -- no need to reverse the wheel IK."""
    ctrl: np.ndarray                 # full actuator target vector (copy of data.ctrl)
    engaged: float = 1.0             # fail-safe blend alpha in [0, 1] (1 = fully live)
    clutched: bool = False           # whole-body clutch frozen this frame
    mode: str = ""                   # human-readable state (engaged / HOMING / CLUTCH ...)
    base: tuple = (0.0, 0.0, 0.0)    # smoothed tip-safe base twist (vx, vy, omega) [m/s, rad/s]

    def named(self, model):
        import mujoco
        out = {}
        for i in range(len(self.ctrl)):
            nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if nm is not None:
                out[nm] = float(self.ctrl[i])
        return out
