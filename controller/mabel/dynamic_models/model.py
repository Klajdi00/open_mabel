"""
Core MuJoCo-backed dynamics model — the single physics oracle every dynamic
model region and torque controller queries.

Design goal: the *dynamics model* (what computes the torques) is separated from
the *plant* (what reads state and applies effort, see mabel/plant.py), so the
same controller runs in simulation and on hardware:

    MujocoDynamics  --- computes g(q), bias C qd + g, full M qacc + C qd + g,
                        and the joint-space mass matrix from a MuJoCo model
                        (also used on the real robot, fed with measured joint
                        states -> model-based control).

Adaptability: gravity / bias come from the model's body_mass and body_ipos (CoM),
which can be changed at runtime (set_link_mass / set_link_com / scale_total_mass)
so the controller tracks the real robot's (uncertain, changing) inertial params.

`UpperBodyDynamics` is the historical name of this class (it always computed
FULL-model terms; region selection happens at the joint-indexing layer, see
dynamic_models/regions.py) and is kept as an alias for compatibility.
"""
import numpy as np
import mujoco


def _dof(model, joint_name):
    return model.joint(joint_name).dofadr[0]


class MujocoDynamics:
    """Inverse-dynamics queries from a MuJoCo model. Works for sim and (fed with
    measured q, qd) for the real robot. Mass / CoM are adaptable at runtime."""

    def __init__(self, model, data):
        self.model = model
        self.data = data

    # mass / CoM adaptation -----------------------------------------------------
    def set_link_mass(self, body, mass):
        self.model.body(body).mass[0] = float(mass)
        self._refresh()

    def set_link_com(self, body, com_xyz):
        self.model.body(body).ipos[:] = np.asarray(com_xyz, dtype=float)
        self._refresh()

    def scale_total_mass(self, factor, bodies=None):
        names = bodies or [self.model.body(i).name for i in range(self.model.nbody)]
        for b in names:
            self.model.body(b).mass[0] *= factor
        self._refresh()

    def _refresh(self):
        # Recompute derived inertial quantities (subtree mass, etc.).
        mujoco.mj_setConst(self.model, self.data)

    # torque queries ------------------------------------------------------------
    def bias(self):
        """Generalized bias force b(q, qd) = C(q,qd) qd + g(q) for the current
        state (gravity + Coriolis/centrifugal), as computed by MuJoCo."""
        return self.data.qfrc_bias.copy()

    def gravity(self):
        """Gravity-only generalized force g(q): bias evaluated at qd = 0."""
        out = np.zeros(self.model.nv)
        qvel = self.data.qvel.copy()
        self.data.qvel[:] = 0.0
        mujoco.mj_rne(self.model, self.data, 0, out)   # flg_acc=0, qd=0 -> g(q)
        self.data.qvel[:] = qvel
        return out

    def inverse_dynamics(self, qacc):
        """Full inverse dynamics tau = M(q) qacc + C(q,qd) qd + g(q) (RNE)."""
        out = np.zeros(self.model.nv)
        save = self.data.qacc.copy()
        self.data.qacc[:] = qacc
        mujoco.mj_rne(self.model, self.data, 1, out)   # flg_acc=1 includes M qacc
        self.data.qacc[:] = save
        return out

    def mass_matrix(self):
        """Dense joint-space mass matrix M(q) (nv x nv)."""
        M = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, M, self.data.qM)
        return M


# Historical name (the class always computed full-model terms).
UpperBodyDynamics = MujocoDynamics
