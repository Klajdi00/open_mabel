"""Data types for the two-mode body retargeter — and, above all, the split of
the operator measurement stream into the two ADMITTED SETS of the paper's
eq. (12).

    S_L = {T_dh_L, T_dh_R, skeleton}      DeviceMeasurements
    S_G = S_L u {T_ad}                    DeviceMeasurements + WorldMeasurements

The split is the whole architecture, so it is enforced STRUCTURALLY rather than
by a branch: `LocalTorsoFit.update()` accepts a `DeviceMeasurements` and nothing
else, so there is no parameter through which the F_a-referenced device pose
could reach it. "Local mode ignores the world estimate" is then a property of
the type signatures, not a claim about the control flow — which is what makes
the vehicle-invariance test (tests/test_retarget_modes.py) meaningful instead of
tautological.

FRAMES
------
F_a   ARKit world ANCHOR frame. The headset's visual-inertial solution. X right,
      Y up, -Z forward. In a moving vehicle this frame is CORRUPTED: the cabin is
      visually static while the IMU senses the vehicle's specific force, so the
      anchor's heading integrates vehicle yaw and its gravity reference tilts
      under lateral acceleration.
F_d   DEVICE frame — the headset itself. Hand anchors resolved into F_d are pure
      optical measurements: T_dh = T_ad^-1 T_ah, in which the round trip through
      F_a cancels EXACTLY (same anchor, same instant). This is why local mode
      needs no app-side change — the cancellation happens server-side from what
      every client already streams.
F_b   Robot world frame. X forward, Y left, Z up.

BASIS CONVENTION (shared with whole_body_retargeting.calibration.body_frame and
BodyFrameEstimator.frame(), so the three interoperate): a body basis `B` is a
3x3 whose COLUMNS are (right, up, forward) expressed in the source frame; body
coordinates of a source-frame vector `v` are `B.T @ v`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class DeviceMeasurements:
    """S_L — everything the LOCAL mode is allowed to see.

    Every transform is expressed in the DEVICE frame F_d, so nothing here
    depends on the headset's world estimate.

    wrists    {side: (4,4)} wrist pose in F_d
    skeleton  {side: {joint_name: (4,4)}} full hand skeleton in F_d
    pinch     {side: float|None} thumb-index gap [m] (clutch / grasp intent)
    timestamp monotonic device clock [s]
    """
    wrists: dict = field(default_factory=dict)
    skeleton: dict = field(default_factory=dict)
    pinch: dict = field(default_factory=dict)
    timestamp: float = 0.0

    def sides(self):
        """Sides with a usable wrist this frame, in a deterministic order."""
        return [s for s in ("left", "right") if self.wrists.get(s) is not None]

    def wrist_pos(self, side):
        T = self.wrists.get(side)
        return None if T is None else np.asarray(T, float)[:3, 3]

    def elbow_pos(self, side):
        """Elbow position in F_d from ARKit's `forearm_arm` joint (the elbow end
        of the forearm), or None.

        Worth one caveat: ARKit does not TRACK the elbow — `forearm_arm` is
        extrapolated from the hand anchor, which is why the arm IK deliberately
        ignores it as a pose objective. It is used here for something much
        weaker and much more robust: the LENGTH of a rigid segment and a coarse
        direction, both of which survive an extrapolated joint far better than a
        pose objective does.
        """
        joints = self.skeleton.get(side) or {}
        T = joints.get("forearm_arm")
        return None if T is None else np.asarray(T, float)[:3, 3]


@dataclass(frozen=True)
class WorldMeasurements:
    """S_G \\ S_L — the F_a-referenced part of the stream. ONLY the global
    estimator (and the one-shot mode-handoff latch) may read this.

    T_ad   (4,4) device pose in the ARKit anchor frame.
    """
    T_ad: np.ndarray | None = None

    @property
    def R_ad(self):
        """Device ORIENTATION in F_a, or None."""
        return None if self.T_ad is None else np.asarray(self.T_ad, float)[:3, :3]

    @property
    def p_ad(self):
        """Device POSITION in F_a, or None."""
        return None if self.T_ad is None else np.asarray(self.T_ad, float)[:3, 3]


@dataclass
class BodyFrameEstimate:
    """One estimator's output: the operator's body basis, plus where the
    shoulders sit, all expressed in that estimator's OWN source frame.

    basis       (3,3) columns (right, up, forward) in the source frame
    shoulders   {side: (3,)} shoulder centre in the source frame
    root        (3,) mid-shoulder root in the source frame (mu_o of eq. 17)
    frame       "device" (F_d, local) or "anchor" (F_a, global) — which frame
                `basis`/`shoulders`/`root` and the wrist keypoints live in. The
                retargeter uses it to feed the map matching keypoints.
    confidence  >= 0; how strongly THIS FRAME's data constrained the estimate
                (smallest eigenvalue of the data Hessian for the local fit). 0
                means the estimate is carried entirely by temporal continuity.
    valid       False before the estimator has produced anything usable.
    diag        estimator-specific diagnostics (never read by control code).
    """
    basis: np.ndarray | None = None
    shoulders: dict = field(default_factory=dict)
    root: np.ndarray | None = None
    frame: str = "anchor"
    device_origin: np.ndarray | None = None
    confidence: float = 0.0
    valid: bool = False
    diag: dict = field(default_factory=dict)

    def rebuilt(self, model, basis):
        """The same estimate re-derived from a CORRECTED body basis.

        The shoulders and root are rigid consequences of the basis, so anything
        that corrects the basis — notably the mode-entry alignment latch — has
        to be pushed through them as well. Correcting only the map and leaving
        the shoulders where the raw estimator put them is a silent 30 cm target
        jump at a mode switch: the rotation is continuous, the anchors are not.
        """
        B = np.asarray(basis, float)
        origin = (np.zeros(3) if self.device_origin is None
                  else np.asarray(self.device_origin, float))
        root = origin - B @ model.device_body()
        return BodyFrameEstimate(
            basis=B, root=root, frame=self.frame, device_origin=origin,
            shoulders={s: root + B @ model.shoulder_body(s)
                       for s in ("left", "right")},
            confidence=self.confidence, valid=self.valid, diag=self.diag)


@dataclass
class RobotFrames:
    """The robot side of the map, as PLAIN ARRAYS — the package never imports
    MuJoCo, so every piece of it is unit-testable without a model.

    shoulders   {side: (3,)} arm-mount position in F_b
    root        (3,) mid arm-mount (mu_b of eq. 17)
    reach       {side: float} shoulder->wrist reach L_b [m]
    basis       (3,3) chassis body axes (right, up, forward) in F_b. Composing
                with it is what makes the map BODY-relative: the operator's
                heading and the robot's chassis heading both cancel, so the same
                body-relative reach lands the same body-relative target however
                either of them is turned.
    lift        current lift extension [m] (eq. 20 comfort region)
    """
    shoulders: dict = field(default_factory=dict)
    root: np.ndarray | None = None
    reach: dict = field(default_factory=dict)
    basis: np.ndarray | None = None
    lift: float = 0.0


@dataclass
class RetargetOutput:
    """What one retarget tick produces. `base_twist` is the RETARGET channel
    only — an explicit nav-joystick command is a separate, deliberate input and
    is never zeroed by local mode."""
    mode: str = "local"
    wrist_targets: dict = field(default_factory=dict)   # {side: (3,) in F_b}
    R_bo: np.ndarray | None = None                      # (3,3) source -> F_b
    base_twist: tuple = (0.0, 0.0, 0.0)                 # (v_fwd, v_left, omega)
    swivel: dict = field(default_factory=dict)          # {side: rad}
    lift_rate: float | None = None                      # eq. 20 (local only)
    torso: float | None = None                          # eq. 20 (local only)
    neck: np.ndarray | None = None                      # (3,3) head rel. torso
    confidence: float = 0.0
    valid: bool = False
    diag: dict = field(default_factory=dict)
