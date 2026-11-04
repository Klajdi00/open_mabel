"""Joint-space PD, with an OPTIONAL gravity feed-forward.

    tau = kp * (q_des - q)  -  kd * qd  [ + g(q) ]

`gravity=False` is the baseline: pure PD, nothing cancels the bias torque, so a
loaded joint settles exactly where `kp * e == tau_gravity`. That droop is a
RESULT we quantify, not a bug we paper over — MEASURED at kp=100: 3.8 deg worst
joint, which lands as 12-27 mm of palm error in task space even though the IK
solved to under 1 mm (exp_basic_controller.py E1-E3).

`gravity=True` adds `g(q)`, computed by `ArmGravity` from a-priori model
constants (link transforms + masses, baked from the MJCF into `arm_params.py`
the way a URDF is) and the MEASURED joint angles — numpy only, no simulator.
The PD is then left to correct only the residual and the steady-state droop
collapses toward zero without raising kp, which matters because raising kp to
hide droop buys stiffness you did not want and an oscillation you did not ask
for.

SIM AND REAL, ONE CODE PATH. The gravity term reads nothing privileged: no
`mj_forward`, no `qfrc_bias`, no simulator state. Everything it needs — the
model and the encoder angles — exists on the physical arm too, so the identical
controller runs in both places. (An earlier version read MuJoCo's `qfrc_bias`;
correct, one line, and unusable on hardware.) `ArmGravity` matches `qfrc_bias`
to machine precision over random arm poses — that agreement is what licenses
the swap; see `exp_gravity_model.py` and `test_gravity_model.py`.

BASE ORIENTATION: gravity enters through the shoulder's orientation, so a
pitched torso changes the load. Pass `base_R` (torso/lift encoders or an IMU on
the robot); it defaults to the rest-pose lean baked by the extractor, never to a
bare "upright" assumption.

ONE LAW AT A TIME
-----------------
The MJCF drives these joints with `position` actuators, which are themselves a
PD (kp=100, kd=20 on the arms). Leaving them enabled while also writing torque
means two controllers fighting over one joint, and every measurement afterwards
describes the sum rather than the thing you meant to test. `attach()` therefore
NEUTRALISES the position servo of every joint it owns and restores it on
`detach()`. If you measure a controller without doing this, you are measuring
the MJCF's gains as much as your own.
"""
from __future__ import annotations

import numpy as np
import mujoco

from .gravity import ArmGravity


class JointPD:
    """A pure PD over a named set of hinge joints.

    Parameters
    ----------
    model, data : the MuJoCo pair to act on.
    joints : joint names this controller OWNS (its servos are neutralised).
    kp, kd : scalar, or {joint: value}. Units are N·m/rad and N·m·s/rad.
    tau_max : per-joint torque clamp. None = take it from the actuator's
        `forcerange` when the model declares one, else uncapped.
    """

    def __init__(self, model, data, joints, kp=100.0, kd=20.0, tau_max=None,
                 gravity=False):
        self.model, self.data = model, data
        self.joints = list(joints)
        self._jid = {j: model.joint(j).id for j in self.joints}
        self._dof = {j: int(model.joint(j).dofadr[0]) for j in self.joints}
        self._qadr = {j: int(model.joint(j).qposadr[0]) for j in self.joints}
        self.kp = {j: float(kp[j] if isinstance(kp, dict) else kp) for j in self.joints}
        self.kd = {j: float(kd[j] if isinstance(kd, dict) else kd) for j in self.joints}
        self.tau_max = self._resolve_tau_max(tau_max)
        # `gravity` may be True/False, a side name ('left'|'right'), or an
        # ArmGravity instance. True picks the side from the joint names above.
        self.grav_model = self._make_grav_model(gravity)
        self.gravity = self.grav_model is not None
        self.base_R = None                # set per-tick from torso encoders/IMU
        self.q_des = {}
        self._saved = None            # actuator gains, restored on detach()
        self.last_tau = {}

    # ── model bookkeeping ────────────────────────────────────────────────────
    def _resolve_tau_max(self, tau_max):
        out = {}
        for j in self.joints:
            if isinstance(tau_max, dict):
                out[j] = float(tau_max.get(j, np.inf))
                continue
            if tau_max is not None:
                out[j] = float(tau_max)
                continue
            out[j] = np.inf
            for aid in range(self.model.nu):
                if (self.model.actuator_trntype[aid] == mujoco.mjtTrn.mjTRN_JOINT
                        and int(self.model.actuator_trnid[aid, 0]) == self._jid[j]):
                    lo, hi = self.model.actuator_forcerange[aid]
                    if hi > lo:                      # 0,0 means "unlimited"
                        out[j] = float(max(abs(lo), abs(hi)))
                    break
        return out

    def _owned_actuators(self):
        ids = set(self._jid[j] for j in self.joints)
        return [aid for aid in range(self.model.nu)
                if self.model.actuator_trntype[aid] == mujoco.mjtTrn.mjTRN_JOINT
                and int(self.model.actuator_trnid[aid, 0]) in ids]

    def attach(self):
        """Take ownership: silence the MJCF position servos on our joints."""
        if self._saved is not None:
            return self
        self._saved = (self.model.actuator_gainprm.copy(),
                       self.model.actuator_biasprm.copy())
        for aid in self._owned_actuators():
            self.model.actuator_gainprm[aid, :3] = 0.0
            self.model.actuator_biasprm[aid, :3] = 0.0
            self.data.ctrl[aid] = 0.0
        return self

    def detach(self):
        """Hand the joints back to the MJCF servos (and stop applying torque)."""
        if self._saved is None:
            return self
        self.model.actuator_gainprm[:] = self._saved[0]
        self.model.actuator_biasprm[:] = self._saved[1]
        self._saved = None
        for j in self.joints:
            self.data.qfrc_applied[self._dof[j]] = 0.0
        return self

    # ── gravity feed-forward (model-based, numpy only) ───────────────────────
    def _make_grav_model(self, gravity):
        if not gravity:
            return None
        if isinstance(gravity, ArmGravity):
            return gravity
        side = gravity if isinstance(gravity, str) else self._infer_side()
        return ArmGravity(side, joint_names=self.joints)

    def _infer_side(self):
        j0 = self.joints[0].lower()
        if "left" in j0:
            return "left"
        if "right" in j0:
            return "right"
        raise ValueError(
            f"cannot infer arm side from {self.joints[0]!r}; pass gravity='left'"
            " or gravity='right'")

    def set_base_orientation(self, base_R):
        """World<-base rotation for the gravity term (torso lean). None = the
        baked rest-pose value. On the robot: torso/lift encoders or an IMU."""
        self.base_R = None if base_R is None else np.asarray(base_R, float).reshape(3, 3)
        return self

    def gravity_torque(self):
        """{joint: g(q)} from the MODEL and the MEASURED angles. numpy only."""
        if self.grav_model is None:
            return {j: 0.0 for j in self.joints}
        q = [self.q(j) for j in self.joints]
        return self.grav_model.torque(q, base_R=self.base_R)

    def q(self, j):
        return float(self.data.qpos[self._qadr[j]])

    def qd(self, j):
        return float(self.data.qvel[self._dof[j]])

    def hold_here(self):
        """Set every target to the measured angle — a bumpless start."""
        self.q_des = {j: self.q(j) for j in self.joints}
        return self

    def set_targets(self, q_des):
        """Merge in desired angles. Unknown joints are ignored, not guessed."""
        for j, v in (q_des or {}).items():
            if j in self.kp and np.isfinite(v):
                self.q_des[j] = float(v)
        return self

    # ── the law ──────────────────────────────────────────────────────────────
    def step(self):
        """Compute and apply one tick of torque. Call before `mj_step`."""
        g = self.gravity_torque()
        for j in self.joints:
            tgt = self.q_des.get(j)
            if tgt is None:
                continue
            e = tgt - self.q(j)
            tau = self.kp[j] * e - self.kd[j] * self.qd(j) + g[j]
            lim = self.tau_max[j]
            if np.isfinite(lim):
                tau = float(np.clip(tau, -lim, lim))
            self.data.qfrc_applied[self._dof[j]] = tau
            self.last_tau[j] = tau
        return self

    def error(self):
        """{joint: q_des - q} for every joint currently commanded."""
        return {j: self.q_des[j] - self.q(j) for j in self.joints if j in self.q_des}
