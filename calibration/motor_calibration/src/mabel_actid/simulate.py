"""
simulate.py
===========

A ground-truth actuator simulator, so the whole identification pipeline can be
proven *without hardware* -- the same philosophy as ``mabel_sysid``: scramble a
known model, recover it, check the error is at the noise floor.

The plant is a 1-DOF rigid inertia driven by a velocity-loop FOC controller:

    J * dw/dt = tau_cmd - tau_f(w) - tau_g(theta)
    tau_cmd   = kd * (w_des - w) + kp * (q_des - q) + t_ff      (MIT-mode law)

The motor reports the electromagnetic torque it applied, ``tau_cmd``; at steady
velocity that equals ``tau_f + tau_g`` -- which is what the identifier reads.
Sensor reality is added on top: torque-estimate noise, velocity quantisation,
and a one-sample feedback delay.

``tau_g`` is optional.  With ``pendulum=None`` the shaft is bare (bench test,
tau_g = 0) -- the primary, cleanest case.  Pass a ``pendulum=(m, l, g)`` to put a
known gravity load on the output and exercise the +/- direction-averaging trick
that removes gravity from the friction estimate.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from .friction_model import FrictionParams, friction_torque
from .data_logging import CalibrationLog
from .trajectories import StaircaseProfile


@dataclasses.dataclass
class SimSensors:
    tau_noise: float = 0.02      # N*m   std of torque-estimate noise
    w_quant: float = 0.0         # rad/s velocity quantisation step (0 = off)
    tau_quant: float = 0.0       # N*m   torque quantisation step (0 = off)
    seed: int = 0


class SyntheticActuator:
    """Forward model of a Damiao joint with known friction + inertia."""

    def __init__(self, truth: FrictionParams, *,
                 kp: float = 0.0, kd: float = 2.0,
                 pendulum: tuple[float, float, float] | None = None,
                 sensors: SimSensors | None = None):
        self.truth = truth
        self.kp = kp
        self.kd = kd
        self.pendulum = pendulum            # (mass kg, length m, g m/s^2)
        self.sensors = sensors or SimSensors()

    def _tau_g(self, theta: float) -> float:
        if self.pendulum is None:
            return 0.0
        m, l, g = self.pendulum
        return m * g * l * np.sin(theta)

    # ------------------------------------------------------------------ run
    def run_velocity(self, prof: StaircaseProfile,
                     rate: float = 500.0) -> CalibrationLog:
        """Integrate the plant under the velocity-staircase command.

        Velocity is held with an integrated *position ramp* (q_des advances at
        the commanded speed) and the controller's ``kp`` term, which tracks low
        speeds accurately -- a pure velocity loop is limited by the Damiao
        ``kd_max`` and drifts badly near zero.  The interleaved +/- levels keep
        the ramp position bounded around the start.
        """
        dt = 1.0 / rate
        q_des = np.cumsum(prof.w_cmd) * dt           # ramp at commanded speed
        return self._integrate(prof.t, w_des=prof.w_cmd, q_des=q_des,
                               w_cmd_log=prof.w_cmd, steady=prof.steady,
                               rate=rate, note="velocity staircase (sim)")

    def run_position(self, t, q_des, w_des, rate: float = 500.0,
                     note: str = "position sweep (sim)") -> CalibrationLog:
        """Integrate the plant tracking a position+velocity reference."""
        steady = np.zeros(t.size, dtype=bool)
        return self._integrate(t, w_des=w_des, q_des=q_des,
                               w_cmd_log=w_des, steady=steady, rate=rate,
                               note=note)

    # ------------------------------------------------------------- internal
    def _integrate(self, t, w_des, q_des, w_cmd_log, steady, rate, note):
        rng = np.random.default_rng(self.sensors.seed)
        n = t.size
        dt = 1.0 / rate
        J = max(self.truth.J, 1e-6)

        q = np.zeros(n)
        w = np.zeros(n)
        a = np.zeros(n)
        tau = np.zeros(n)

        th, om = 0.0, 0.0
        w_fb = 0.0                       # one-sample-delayed velocity feedback
        for i in range(n):
            qd = 0.0 if q_des is None else q_des[i]
            tau_cmd = self.kd * (w_des[i] - w_fb)
            if self.kp > 0.0:
                tau_cmd += self.kp * (qd - th)

            tau_g = self._tau_g(th)
            tau_fr = float(friction_torque(self.truth, np.array([om]))[0])
            acc = (tau_cmd - tau_fr - tau_g) / J

            # log BEFORE integrating so (q,w,a,tau) are a consistent instant
            q[i], w[i], a[i] = th, om, acc
            # the motor reports the electromagnetic torque it applied
            tau[i] = tau_cmd + rng.normal(0.0, self.sensors.tau_noise)

            # semi-implicit Euler
            om += acc * dt
            th += om * dt
            w_fb = om

        if self.sensors.w_quant > 0:
            w = np.round(w / self.sensors.w_quant) * self.sensors.w_quant
        if self.sensors.tau_quant > 0:
            tau = np.round(tau / self.sensors.tau_quant) * self.sensors.tau_quant

        return CalibrationLog(t=t, q=q, w=w, a=a, tau=tau,
                              w_cmd=w_cmd_log, steady=steady,
                              motor="SIM", note=note, rate=rate)
