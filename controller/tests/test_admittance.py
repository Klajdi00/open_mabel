"""Admittance stack: the virtual mass-damper base response (analytic) and the
comfort-box gate. The joint-torque -> wrench -> contact localization pipeline
is physically validated in test_dynamic_models.py."""
import numpy as np

import mabel.config as C
from mabel.whole_body_controller.admittance import BaseAdmittance, boundary_gate


def test_static_deadband_no_creep():
    adm = BaseAdmittance()
    for _ in range(500):
        vx, vy, w = adm.step(C.ADMIT_F_STATIC * 0.9, C.ADMIT_F_STATIC * 0.9,
                             C.ADMIT_M_STATIC * 0.9, 0.01)
    assert vx == 0.0 and vy == 0.0 and w == 0.0


def test_steady_state_velocity_is_force_over_damping():
    adm = BaseAdmittance()
    F = 20.0
    for _ in range(2000):                     # 20 s >> m/b time constant
        vx, _, _ = adm.step(F, 0.0, 0.0, 0.01)
    assert abs(vx - F / C.ADMIT_LIN_DAMP) < 0.05 * (F / C.ADMIT_LIN_DAMP)


def test_kinetic_stop_settles_to_rest():
    adm = BaseAdmittance()
    for _ in range(300):
        adm.step(25.0, 0.0, 0.0, 0.01)        # spin up
    for _ in range(3000):
        vx, vy, w = adm.step(0.0, 0.0, 0.0, 0.01)
    assert vx == 0.0 and vy == 0.0 and w == 0.0, "base rolled forever after release"


def test_gate_scales_the_drive():
    a_full = BaseAdmittance(); a_gated = BaseAdmittance()
    for _ in range(200):
        v_full, _, _ = a_full.step(20.0, 0, 0, 0.01, gate=1.0)
        v_half, _, _ = a_gated.step(20.0, 0, 0, 0.01, gate=0.5)
    assert 0.0 < v_half < v_full


def test_boundary_gate_ramp():
    c0 = np.zeros(3)
    inside = c0 + np.array([C.DRAG_BOX * 0.5, 0, 0])
    edge = c0 + np.array([C.DRAG_BOX + C.DRAG_BOX_RAMP * 0.5, 0, 0])
    beyond = c0 + np.array([C.DRAG_BOX + 2 * C.DRAG_BOX_RAMP, 0, 0])
    assert boundary_gate(inside, c0) == 0.0
    assert 0.0 < boundary_gate(edge, c0) < 1.0
    assert boundary_gate(beyond, c0) == 1.0
