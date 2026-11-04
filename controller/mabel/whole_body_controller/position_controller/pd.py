"""
Plain joint-space PD position controller — the Day-1 hardware controller.

No dynamics model, no gravity compensation, no system ID required: pure

    tau_j = Kp_j (q_des_j - q_j) - Kd_j qd_j  (+ optional feedforward hook)

per managed joint, over the swappable Plant seam (mabel/plant.py) so the same
code drives the MuJoCo twin and the real robot on day one. Expect a gravity
droop (steady-state error ~ g_j / Kp_j) until the gravity-compensation
feedforward is plugged in — pass `feedforward=lambda: region.gravity_ff()`
(a {joint: tau} dict) once the dynamic model is trusted, and the SAME
controller becomes the Day-2 PD+gravity controller with no restructuring.

For simulation-style position control (MuJoCo's built-in stiff position
servos / the real drivers' own position mode), no controller object is needed:
write actuator targets directly — that is what the kinematic teleop stack does.
This class is for the TORQUE-mode path where we own the loop.
"""
import math

__all__ = ["JointPositionController"]


class JointPositionController:
    """Per-joint PD in torque mode over a Plant.

    Parameters
    ----------
    plant     : mabel.plant.Plant — state reads + effort writes (sim or real).
    joints    : list of joint names this controller owns.
    kp, kd    : "servo" (default) inherits each joint's tuned position-servo
                gains from the MuJoCo model (kp = gainprm[0], kd = -biasprm[2]),
                which are per-joint, inertia-matched, and stable at the model
                timestep — one uniform gain across joints as different as a
                shoulder and the tiny ORCA wrist is NOT (the wrist limit-
                cycles). Or pass a scalar / {joint: value} explicitly (real
                hardware without a model must pass explicit gains).
    tau_limit : "servo" (default) inherits each joint's actuator
                |forcerange| from the model — the torque the HARDWARE can
                actually produce. A uniform scalar clamp is a trap: MABEL's
                distal arm joints are +-7 Nm actuators, and letting the PD
                drive one at 40 Nm both breaks sim-real fidelity and pumps a
                sustained limit cycle after a hard disturbance (observed: a
                20 N shove left the wrist oscillating at the 40 Nm clamp
                forever, which the wrench estimator then read as a standing
                ~7 N phantom contact). Or a scalar / {joint: value}; None =
                unclamped. Joints whose model forcerange is unset ([0,0])
                fall back to 40 Nm.
    feedforward : optional callable returning {joint: tau} added each tick
                (e.g. RegionDynamicModel.gravity_ff for Day-2 gravity comp).
    neutralize_servos : disable each joint's stiff position servo on entry so
                the PD owns the joint (matches the compliance/impedance engine).
    """

    def __init__(self, plant, joints, kp="servo", kd="servo", tau_limit="servo",
                 feedforward=None, neutralize_servos=True):
        self.plant = plant
        self.joints = list(joints)
        # Gains are read BEFORE the servos are neutralised (disable_servo zeroes
        # the very params "servo" inherits from).
        self.kp = self._per_joint(kp, servo_idx=0)
        self.kd = self._per_joint(kd, servo_idx=2)
        if tau_limit == "servo":
            tau_limit = {j: self._servo_force_limit(j) for j in self.joints}
        self.tau_limit = (self._per_joint(tau_limit)
                          if tau_limit is not None else None)
        self.feedforward = feedforward
        self.q_des = {j: plant.q(j) for j in self.joints}   # hold where it is
        if neutralize_servos:
            for j in self.joints:
                try:
                    self.plant.disable_servo(j)
                except KeyError:
                    pass    # joint has no same-named actuator (e.g. tendon lift)

    def _servo_gain(self, joint, idx):
        """Per-joint gain from the model's position servo: kp = gainprm[0],
        kd = -biasprm[2] (MuJoCo position actuator: biasprm = [0, -kp, -kv])."""
        model = getattr(self.plant, "model", None)
        if model is None:
            raise ValueError(
                "kp/kd='servo' needs a plant that exposes .model (SimPlant); "
                "pass explicit per-joint gains for a model-less hardware plant")
        aid = model.actuator(joint).id
        if idx == 0:
            return float(model.actuator_gainprm[aid, 0])
        return float(-model.actuator_biasprm[aid, 2])

    def _servo_force_limit(self, joint):
        """Per-joint effort clamp from the model actuator's |forcerange|
        (0 = unset in MuJoCo -> fall back to a 40 Nm safety clamp)."""
        model = getattr(self.plant, "model", None)
        if model is None:
            raise ValueError(
                "tau_limit='servo' needs a plant that exposes .model; pass an "
                "explicit clamp for a model-less hardware plant")
        fr = model.actuator_forcerange[model.actuator(joint).id]
        lim = float(max(abs(fr[0]), abs(fr[1])))
        return lim if lim > 0.0 else 40.0

    def _per_joint(self, v, servo_idx=None):
        if v == "servo":
            return {j: self._servo_gain(j, servo_idx) for j in self.joints}
        if isinstance(v, dict):
            return {j: float(v.get(j, 0.0)) for j in self.joints}
        return {j: float(v) for j in self.joints}

    # targets --------------------------------------------------------------------
    def set_targets(self, q_des):
        """Update desired positions for any subset of managed joints."""
        for j, q in q_des.items():
            if j in self.q_des:
                self.q_des[j] = float(q)

    def hold_current(self):
        """Re-seed every target to the measured pose (bumpless engage)."""
        for j in self.joints:
            self.q_des[j] = self.plant.q(j)

    # control law ----------------------------------------------------------------
    def update(self):
        """One tick: compute + apply PD efforts. Returns {joint: tau} applied."""
        ff = self.feedforward() if self.feedforward is not None else {}
        applied = {}
        self.plant.begin()
        for j in self.joints:
            q, qd = self.plant.q(j), self.plant.qd(j)
            tau = (self.kp[j] * (self.q_des[j] - q) - self.kd[j] * qd
                   + float(ff.get(j, 0.0)))
            if self.tau_limit is not None:
                lim = self.tau_limit[j]
                tau = math.copysign(min(abs(tau), lim), tau)
            self.plant.apply_effort(j, tau)
            applied[j] = tau
        self.plant.commit()
        return applied
