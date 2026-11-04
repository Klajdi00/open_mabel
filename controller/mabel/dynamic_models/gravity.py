"""
gravity.py — THE upper-body gravity-compensation layer.

Holding the robot up against gravity is a property of the ROBOT, not of which
control space the operator happens to have selected. So it is ONE component,
constructed once, and shared by every path that can move the upper body:

    mabel.teleop_engine.TeleopEngine   navigation (Drive) + retarget (Teleop)
    mabel.whole_body_controller.WBCEngine   joint / wrist (Joint + Task space)

Both call this object, so the compensated DoF set, the gain and the recompute
rate are the same in every mode BY CONSTRUCTION. That is the invariant; it is
asserted directly in controller/tests/test_mode_switch_no_sag.py.

WHY THIS EXISTS (operator report, 2026-07-24: "when I switch from drive to
joint or task space the robot arm will sag"). Gravity compensation used to be
implemented twice, over two different joint sets:

  * TeleopEngine wrote a feed-forward over all 22 actuated upper-body DoFs.
  * WBCEngine.step() called plant.begin() — which ZEROES the whole
    qfrc_applied vector — and then re-applied effort only to `self.managed`,
    the 14 arm joints. The ORCA wrists are deliberately outside the torque law
    (WBCEngine._managed_for), the torso is outside it in the "arm" region, and
    the lift is outside it unless freed.

So leaving Drive silently dropped the feed-forward on 8 upper-body DoFs and
they settled to their servos' tau_g/kp error. Measured before the fix, on the
switch Drive -> Task/Joint: wrists 1.6 deg, torso 0.18 deg, neck 0.28 deg.

TWO PROPERTIES THAT ARE LOAD-BEARING, not incidental:

1. A PRIVATE SCRATCH MjData. The torque is evaluated on this component's own
   MjData, never on the caller's. `UpperBodyDynamics.gravity()` computes g(q)
   by zeroing and restoring `data.qvel` around mj_rne — a mid-tick mutation of
   shared state that perturbs everything reading qvel in the same tick
   (measured: a 0.335 rad single-tick jump on left_arm_5 during the retarget
   re-acquire, tripping the real-robot step-free guard in
   controller/experiments/exp_real_retarget_chain.py). Scratch data has no such
   coupling.

2. qvel = 0, SO THE RESULT IS PURE g(q) — NO CORIOLIS, NO VELOCITY FEEDBACK.
   This is what makes it safe to feed the ORCA wrists, which the torque law
   cannot hold: their effective inertia is ~2.5e-3 kg m^2, ~100x lighter than
   an arm link, so a -D*qd term computed from a mirrored state one RPC hop old
   lands out of phase and injects energy (the wrist rang to -118 rad/s and
   tunnelled through its MJCF limit; see WBCEngine._managed_for). A static
   g(q) has no velocity term at all, so it cannot ring: d(tau)/dq for a wrist
   is bounded by m*g*r ~ 1.74*9.81*0.05 ~ 0.9 N.m/rad, orders below the servo
   gain it assists.
"""
import os

import numpy as np
import mujoco

#: The ACTUATED upper body — every joint the robot must hold up against gravity.
#: The floating base and the swerve wheels are excluded: their weight is carried
#: by the ground contacts, and feeding them a bias force would push the chassis.
UPPER_BODY_JOINTS = (
    ["Lift Lower", "Lift Upper", "Torso", "Neck Yaw", "Neck Pitch", "Neck Roll"]
    + [f"{S} Arm {i}" for S in ("Left", "Right") for i in range(1, 8)]
    + [f"{S} Wrist" for S in ("Left", "Right")]
)

#: Feed-forward scale (0 = pure PD, 1 = full compensation). Env-tunable so an
#: experiment can sweep it; `mabel.teleop_engine.hyperparams.GRAV_GAIN` re-exports.
GRAV_GAIN = float(os.environ.get("MABEL_GRAV_GAIN", "1.0"))

#: Recompute rate [Hz]. The sag this fights is slow, and the cost is one
#: mj_forward on the full model — one of several concurrent mj_forward callers
#: whose OpenMP threads oversubscribe the Jetson's cores. 30 Hz is ample.
GRAV_HZ = float(os.environ.get("MABEL_GRAV_HZ", "30"))


def gravity_dofs(model, joints=UPPER_BODY_JOINTS):
    """DoF addresses of `joints` that exist in `model` (np.int array)."""
    dofs = []
    for n in joints:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
        if jid != -1:
            dofs.append(int(model.jnt_dofadr[jid]))
    return np.array(dofs, dtype=int)


def gravity_joints(model, joints=UPPER_BODY_JOINTS):
    """The subset of `joints` present in `model`, in canonical order."""
    return [n for n in joints
            if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) != -1]


class UpperBodyGravityComp:
    """g(q) for the actuated upper body, cached at GRAV_HZ, computed on private
    scratch state.

    Usage — direct write (the kinematic path owns the whole qfrc vector):

        grav.apply(data, now, clear=True)

    Usage — through a Plant (the torque path, so the same call works on
    hardware), skipping joints whose own control law already carries the bias:

        tau = grav.tau(data, now)
        for j in grav.joints_except(managed):
            plant.apply_effort(j, grav.gain * tau[grav.dof(j)])
    """

    def __init__(self, model, gain=None, hz=None):
        self.model = model
        self.gain = GRAV_GAIN if gain is None else float(gain)
        self._gain_on = self.gain          # restored by set_enabled(True)
        self.hz = GRAV_HZ if hz is None else float(hz)
        self._scratch = mujoco.MjData(model)
        self.joints = gravity_joints(model)
        self.dofs = gravity_dofs(model)
        self._dof = {n: int(d) for n, d in zip(self.joints, self.dofs)}
        self._tau = np.zeros(model.nv)
        self._at = -1e18

    # ── enable / disable ─────────────────────────────────────────────────────
    def set_enabled(self, on):
        """Operator switch for the whole layer (iOS / Vision Pro "Gravity Comp").

        Implemented as gain -> 0 rather than by skipping `apply()`, and the
        difference matters: `apply(clear=True)` is the sole writer of
        `data.qfrc_applied` on the kinematic path, so NOT calling it would leave
        whatever a just-exited torque mode wrote in place — the arm would keep an
        arbitrary stale feed-forward instead of losing the compensation. Zero
        gain still clears, then writes zeros, which is what "off" should mean.

        This does NOT weaken the mode-invariance invariant (the root README): the
        layer is still the ONE shared object both engines hold, and it is still
        the same in every control mode. It is now globally switchable, which is
        an orthogonal axis to which mode is selected.
        """
        on = bool(on)
        if on:
            self.gain = self._gain_on
        else:
            if self.gain:
                self._gain_on = self.gain      # remember an operator-tuned gain
            self.gain = 0.0
        return on

    @property
    def enabled(self):
        return self.gain != 0.0

    # ── queries ──────────────────────────────────────────────────────────────
    def dof(self, joint):
        return self._dof[joint]

    def joints_except(self, exclude):
        """Upper-body joints this layer should feed, given the ones some other
        law already compensates (the WBC's torque-managed set)."""
        skip = set(exclude or ())
        return [j for j in self.joints if j not in skip]

    # ── the torque ───────────────────────────────────────────────────────────
    def tau(self, data, now, force=False):
        """g(q) at `data`'s pose, as a full-nv vector. Recomputed at most at
        `self.hz`; the cached vector is returned in between (gravity changes
        with POSE, and the pose barely moves in 33 ms at teleop speeds).

        `data` is only READ. The evaluation happens on private scratch state
        with qvel = 0, so the result is pure gravity and nothing the caller
        holds is mutated."""
        # `now < self._at` means the clock went backwards (an NTP step on the
        # wall clock the callers use): recompute rather than stall until it
        # catches up, the same guard the sim-time publishers use.
        if force or now < self._at or (now - self._at) >= (1.0 / self.hz):
            self._at = now
            s = self._scratch
            s.qpos[:] = data.qpos
            s.qvel[:] = 0.0                  # -> qfrc_bias == g(q), no Coriolis
            mujoco.mj_forward(self.model, s)
            self._tau[:] = s.qfrc_bias
        return self._tau

    def apply(self, data, now, clear=False, skip=(), force=False):
        """Write the feed-forward straight into `data.qfrc_applied`.

        `clear` zeroes the whole vector first (the kinematic path is the sole
        writer there, and this drops any residue a just-exited torque mode
        left). `skip` names joints to leave alone. Returns the DoFs written."""
        tau = self.tau(data, now, force=force)
        if clear:
            data.qfrc_applied[:] = 0.0
        dofs = (self.dofs if not skip
                else np.array([self._dof[j] for j in self.joints_except(skip)],
                              dtype=int))
        if dofs.size:
            data.qfrc_applied[dofs] = self.gain * tau[dofs]
        return dofs
