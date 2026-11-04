"""
Cartesian admittance for MABEL's mobile base (whole-body force compliance).

With the arms gravity-compensated (see dynamics.py) the operator drags a hand;
the base then behaves like a virtual planar mass with viscous + Coulomb "virtual
friction", driven by the external wrench. So pushing the arm past its comfortable
workspace makes the whole robot glide along, and it coasts to a stop (does not
roll forever) when released. This is the force-driven analogue of Whole-Body
mode 3: there the lazy base followed palm *position*; here it admits the external
*wrench*.

Wrench sensing is abstracted behind `external_base_wrench`:
  - simulation: read data.xfrc_applied (the viewer's Ctrl-drag wrench);
  - hardware:   a wrist force/torque sensor, or the joint-torque residual
                tau_ext = b(q,qd) - tau_cmd mapped through J^T.
"""
import numpy as np
from ... import config as C


def external_base_wrench(model, data, bodies):
    """Net external wrench on `bodies` (from data.xfrc_applied), reduced to the
    chassis planar frame: returns (fx, fy) in the base frame and the moment Mz
    about the base z (so a couple between the two hands becomes a yaw intent)."""
    base = data.body(C.CHASSIS_BODY)
    pb = base.xpos
    Rb = base.xmat.reshape(3, 3)
    F = np.zeros(3); Mz = 0.0
    for b in bodies:
        bid = model.body(b).id
        f = data.xfrc_applied[bid, :3]
        m = data.xfrc_applied[bid, 3:]
        F += f
        r = data.xpos[bid] - pb
        Mz += np.cross(r, f)[2] + m[2]
    f_body = Rb.T @ F
    return float(f_body[0]), float(f_body[1]), float(Mz)


class BaseAdmittance:
    """Virtual planar mass-damper with Coulomb (static) and viscous (kinetic)
    'virtual friction'. Integrates an external wrench into a base body twist.

        m  v_dot  = F - b_lin v   (per linear axis)
        I  w_dot  = Mz - b_ang w
    plus a static deadband (ignore small wrench, no creep) and a kinetic stop
    (snap small velocity to zero, so it does not drift forever)."""

    def __init__(self, mass=None, inertia=None,
                 lin_damping=None, ang_damping=None,
                 f_static=None, m_static=None,
                 v_stop=None, w_stop=None):
        self.m  = C.ADMIT_MASS     if mass        is None else mass
        self.I  = C.ADMIT_INERTIA  if inertia     is None else inertia
        self.bl = C.ADMIT_LIN_DAMP if lin_damping is None else lin_damping
        self.ba = C.ADMIT_ANG_DAMP if ang_damping is None else ang_damping
        self.fs = C.ADMIT_F_STATIC if f_static    is None else f_static
        self.ms = C.ADMIT_M_STATIC if m_static    is None else m_static
        self.v_stop = C.ADMIT_V_STOP if v_stop is None else v_stop
        self.w_stop = C.ADMIT_W_STOP if w_stop is None else w_stop
        self.vx = self.vy = self.w = 0.0

    def step(self, fx, fy, mz, dt, gate=1.0):
        """Advance the virtual dynamics one tick. `gate` in [0,1] scales the
        driving wrench (0 = base inert, 1 = fully compliant); used to engage the
        base only once the hand leaves the comfort boundary."""
        # static (Coulomb) deadband: small wrench produces no motion
        fx = fx if abs(fx) > self.fs else 0.0
        fy = fy if abs(fy) > self.fs else 0.0
        mz = mz if abs(mz) > self.ms else 0.0
        fx *= gate; fy *= gate; mz *= gate

        # virtual mass-damper (viscous virtual friction)
        self.vx += (fx - self.bl * self.vx) / self.m * dt
        self.vy += (fy - self.bl * self.vy) / self.m * dt
        self.w  += (mz - self.ba * self.w)  / self.I * dt

        # kinetic stop: when unforced and slow, settle to rest (no endless roll)
        if fx == 0.0 and abs(self.vx) < self.v_stop: self.vx = 0.0
        if fy == 0.0 and abs(self.vy) < self.v_stop: self.vy = 0.0
        if mz == 0.0 and abs(self.w)  < self.w_stop: self.w = 0.0
        return self.vx, self.vy, self.w


def hand_centroid_base(data, p_l, p_r):
    """Hand centroid expressed in the chassis (base) frame -- invariant to base
    motion, so excursion is measured relative to where the hands rest."""
    base = data.body(C.CHASSIS_BODY)
    return base.xmat.reshape(3, 3).T @ (0.5 * (p_l + p_r) - base.xpos)


def boundary_gate(c_base, c0_base):
    """Smooth gate in [0,1] from the hand-centroid excursion (base frame) past a
    SMALL comfort box referenced to the neutral pose c0_base. 0 inside the box
    (arm compliant alone) -> ramps to 1 once dragged DRAG_BOX+RAMP out, so the
    base follows well before the arm reaches its workspace limit."""
    excursion = float(np.linalg.norm((c_base - c0_base)[:2]))
    return min(1.0, max(0.0, (excursion - C.DRAG_BOX) / C.DRAG_BOX_RAMP))
