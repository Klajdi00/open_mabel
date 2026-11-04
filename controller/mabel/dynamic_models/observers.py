"""
Momentum-based external-torque observer (the estimation primitive of every
dynamic model region).

The simulator can hand us the true external wrench (data.xfrc_applied), but a
real robot cannot. The generalized-momentum disturbance observer (De Luca &
Mattone) recovers the external joint torque tau_ext WITHOUT joint acceleration,
as a first-order low-pass of bandwidth K:

    r = K ( p - p0 - INT( tau_cmd + tau_passive - b(q,qd) + Mdot*qd + r ) dt ),
    r -> tau_ext

using p = M(q) qd, bias b = C qd + g (MuJoCo qfrc_bias), and Mdot from a finite
difference of the mass matrix. Only q, qd, tau_cmd and the model are used --
all available on the real robot (torque from the DAMIAO drivers' current
sensing; the same MuJoCo model used as the dynamics model) -- so the estimate
transfers directly to hardware.

The Cartesian consumers of this residual (end-effector wrench, contact
localization, base wrench) live in
mabel.whole_body_controller.admittance.wrench_estimation.
"""
import numpy as np
import mujoco


class MomentumObserver:
    """First-order generalized-momentum observer: estimates the external joint
    torque tau_ext from (q, qd, tau_cmd) and the model, with no acceleration."""

    def __init__(self, model, data, gain=25.0, use_passive=True):
        self.m, self.d = model, data
        self.K = float(gain)            # observer bandwidth [1/s]
        self.use_passive = bool(use_passive)   # count qfrc_passive as modeled
        self.nv = model.nv
        self.r = np.zeros(self.nv)
        self.integ = np.zeros(self.nv)
        self._M_prev = None
        self._p0 = None

    def _full_M(self):
        M = np.zeros((self.nv, self.nv))
        mujoco.mj_fullM(self.m, M, self.d.qM)
        return M

    def update(self, tau_cmd, dt):
        """Advance the observer one tick with the commanded joint torque tau_cmd
        (full nv vector). Returns the external-torque estimate r ~ tau_ext."""
        qd = self.d.qvel
        M = self._full_M()
        p = M @ qd
        if self._M_prev is None:        # initialize so r starts at 0
            self._M_prev = M; self._p0 = p.copy()
            return self.r
        Mdot_qd = ((M - self._M_prev) / dt) @ qd
        bias = self.d.qfrc_bias         # C(q,qd) qd + g(q)
        # passive damping/springs are MODEL knowledge (mj_forward on the
        # mirrored state gives them on the real robot too): treating them as
        # modeled keeps them out of the "external" residual while moving
        tau_mod = tau_cmd + (self.d.qfrc_passive if self.use_passive else 0.0)
        self.integ += (tau_mod - bias + Mdot_qd + self.r) * dt
        self.r = self.K * (p - self._p0 - self.integ)
        self._M_prev = M
        return self.r
