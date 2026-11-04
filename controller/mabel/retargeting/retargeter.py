"""BodyRetargeter — one tick of the two-mode pipeline.

    measurements (S_L [+ S_G])
        -> LocalTorsoFit (14)  ---.
        -> GlobalBodyFrame (13) --+-> ModeArbiter (15) -> R_bo, nu_b^cmd
                                                            |
                                       wrist targets (16-18)|
                                       swivel (19)          |
                                       lift/torso (20)      v
                                                       RetargetOutput

BOTH ESTIMATORS RUN EVERY FRAME, in either mode. That costs a few tens of
microseconds (the local fit is 4 Gauss-Newton steps on 3 unknowns; the global
estimator already runs today) and buys three things worth far more: a switch
takes effect on the very next frame with no warm-up, the alignment latch always
has both bases for the SAME instant so the handoff is exact rather than
approximate, and the inactive estimator's diagnostics stay available for
telemetry.

It does NOT weaken the admitted-set argument. The claim is about what drives the
OUTPUT, and in local mode the output is a function of the local estimate alone —
which by its type signature never saw T_ad. Running the global estimator beside
it is bookkeeping for the handoff, not an input. The vehicle-invariance test
pins exactly this: identical local-mode outputs under an arbitrary F_a drift.

The retargeter owns no robot state and imports no MuJoCo: the caller passes a
RobotFrames snapshot and receives targets. That is what makes the whole package
testable against synthetic operators, which is the only way the vehicle case can
be tested at all — there is no rig that shakes a headset like a car.
"""
from __future__ import annotations

import numpy as np

from .body_dofs import hand_centroid, lift_and_torso, palm_roll, swivel_target
from .global_frame import GlobalBodyFrame
from .local_frame import LocalTorsoFit
from .mode import ModeArbiter, RetargetMode
from .operator_model import OperatorModel
from .types import RetargetOutput
from .wrist_targets import wrist_targets, wrist_orientation


class BodyRetargeter:
    """The one object the engine holds. `update` is the whole contract."""

    def __init__(self, mode=RetargetMode.LOCAL, model=None, hand_clear=None):
        self.model = model if model is not None else OperatorModel()
        self.local = LocalTorsoFit(self.model)
        self.global_ = GlobalBodyFrame(self.model)
        self.arbiter = ModeArbiter(mode)
        self.hand_clear = hand_clear
        self._R_ad_prev = None
        self.last = RetargetOutput(mode=self.arbiter.mode)

    # ── configuration ─────────────────────────────────────────────────────────
    @property
    def mode(self):
        return self.arbiter.mode

    def set_mode(self, mode):
        """Request LOCAL or GLOBAL. Takes effect on the next `update`, where
        both estimates for that frame exist and the latch can be exact."""
        self.arbiter.request(mode)
        return self

    def apply_profile(self, profile):
        """Freeze a measured OperatorProfile into the anthropometry."""
        self.model.apply_profile(profile)
        return self

    def reset(self):
        """Drop every anchor: both estimators, the alignment latch and the
        ramps. Call on re-engage / clutch release, never mid-session."""
        self.local.reset()
        self.global_.reset()
        self._R_ad_prev = None
        self.arbiter = ModeArbiter(self.arbiter.mode, self.arbiter.tau)
        self.last = RetargetOutput(mode=self.arbiter.mode)
        return self

    # ── one tick ──────────────────────────────────────────────────────────────
    def update(self, dev, world, robot, dt):
        """dev: DeviceMeasurements (S_L). world: WorldMeasurements (S_G \\ S_L)
        or None. robot: RobotFrames snapshot. Returns a RetargetOutput."""
        dt = max(float(dt), 1e-4)
        R_ad = None if world is None else world.R_ad
        # OPT-IN dead-reckoning (off by default — see LocalTorsoFit's TRADE-OFF
        # note). This is the ONLY place an F_a-referenced quantity can reach the
        # local estimator, and only when explicitly enabled.
        if self.local.dead_reckon and R_ad is not None:
            if self._R_ad_prev is not None:
                self.local.propagate(self._R_ad_prev.T @ R_ad)
            self._R_ad_prev = np.asarray(R_ad, float).copy()
        est_l = self.local.update(dev, dt)
        est_g = self.global_.update(dev, world, dt)

        mode, align, twist, blend, switched = self.arbiter.update(
            est_g, est_l, R_ad, self.global_.base_twist(), dt)

        # The alignment latch corrects the operator's body BASIS, and the
        # shoulders and root are rigid consequences of it — so push it through
        # the whole estimate, not just through R_bo. B_corr = B align^T is the
        # basis whose transpose is the corrected map align B^T.
        est = est_l if mode == RetargetMode.LOCAL else est_g
        if est.valid and est.basis is not None:
            est = est.rebuilt(self.model, est.basis @ align.T)
        out = RetargetOutput(mode=mode, base_twist=twist,
                             confidence=est.confidence)
        out.diag = {"blend": blend, "switched": switched,
                    "switches": self.arbiter.switches,
                    "exact_latch": self.arbiter.exact_latch,
                    "local": dict(est_l.diag), "global": dict(est_g.diag),
                    "stature": self.model.stature}
        if not est.valid or robot is None:
            self.last = out
            return out

        # est.basis is ALREADY alignment-corrected, so the map is the plain
        # R_rob B_corr^T — the latch must not be applied twice.
        R_bo = (None if robot.basis is None
                else np.asarray(robot.basis, float) @ est.basis.T)
        out.R_bo = R_bo
        if R_bo is None:
            self.last = out
            return out

        # Operator keypoints in the ACTIVE estimate's own source frame. This is
        # the only place the two modes differ downstream of R_bo, and it is a
        # change of representation, not of law.
        points, orientations = self._source_keypoints(dev, world, est.frame)

        out.wrist_targets, tdiag = wrist_targets(
            points, est, robot, self.model, R_bo, hand_clear=self.hand_clear)
        out.diag["targets"] = tdiag

        # (19) swivel from palm roll, per side.
        for side, p in points.items():
            s_i = est.shoulders.get(side)
            if s_i is None or p is None:
                continue
            u = np.asarray(p, float) - np.asarray(s_i, float)
            n = float(np.linalg.norm(u))
            if n < 1e-6:
                continue
            theta = palm_roll(orientations.get(side), u / n)
            phi = swivel_target(theta)
            if phi is not None:
                out.swivel[side] = phi

        # (20) lift + torso from the hand centroid — LOCAL only. In global the
        # engine keeps its head-height law; `blend` in the diag is how the
        # caller cross-fades the two across a switch.
        if mode == RetargetMode.LOCAL:
            c = hand_centroid(out.wrist_targets)
            lift, torso, cdiag = lift_and_torso(c, robot)
            out.lift_rate, out.torso = lift, torso
            out.diag["comfort"] = cdiag
            # Gaze survives in local mode: the head's rotation RELATIVE TO THE
            # TORSO is exactly the local fit's inverse, and is drift-free.
            out.neck = self.local.neck_rotation()

        out.valid = True
        self.last = out
        return out

    # ── helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _source_keypoints(dev, world, frame):
        """Wrist positions + orientations in the estimate's source frame.

        `dev` always carries F_d poses; global mode needs them back in F_a, and
        T_ah = T_ad T_dh restores them exactly (it is the inverse of the
        reduction frames.device_measurements performed on ingest).
        """
        points, orientations = {}, {}
        T_ad = None if world is None else world.T_ad
        for side in dev.sides():
            T = np.asarray(dev.wrists[side], float)
            if frame == "anchor":
                if T_ad is None:
                    continue
                T = np.asarray(T_ad, float) @ T
            points[side] = T[:3, 3]
            orientations[side] = T[:3, :3]
        return points, orientations

    def wrist_orientation_targets(self, dev, world):
        """R* of eq. (18) per side, using the last tick's R_bo. Separate from
        `update` because the position targets and the orientation targets are
        consumed by different stages of the controller."""
        out = self.last
        if out.R_bo is None:
            return {}
        frame = "device" if out.mode == RetargetMode.LOCAL else "anchor"
        _, orientations = self._source_keypoints(dev, world, frame)
        return {side: wrist_orientation(R, out.R_bo)
                for side, R in orientations.items()}
