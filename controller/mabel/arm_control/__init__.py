"""arm_control — the real-robot arm joint control law, consolidated under controller/.

This is the single home for how the MABEL arms are driven on hardware: gravity
feed-forward in the DaMiao raw-motor frame, and the three control modes (pure PD,
PD + gravity comp, compliant drag-and-hold). It is a thin, frame-aware layer over the
shared dynamics engine (:mod:`mabel.dynamic_models`), so the sim, the hardware bridge,
and the experiments all run the SAME control code — the bridge only hosts the Teensy
link; the math lives here.

Layout
------
gravity.py    ArmGravityCompensator + HandsConfig — τ_g(q) → raw motor frame, with an
              EXPLICIT with/without-hands choice (the 1.12 kg phantom-hand shake guard).
impedance.py  ArmControlMode {PURE_PD, GRAVITY_PD, COMPLIANT} + ArmImpedance — turns a
              mode + base gains + τ_g into the per-joint (kp, kd, t_ff) MIT parameters.
admittance.py SoftArmAdmittance — FLOATING soft mode on the position contract:
              τ_ext = τ_g(q) − τ_meas observer → per-joint virtual mass–damper →
              streamed targets, with an auto-tare bias that absorbs gravity-model
              error at every rest pose (the imperfect-sysid defense).
friction.py   FrictionFF — Coulomb friction feed-forward on the COMMANDED velocity.
              Once gravity is compensated the residual error IS the friction deadband
              tau_c/kp (measured 0.62-1.09x of it on all 14 joints); this cancels it
              directly instead of paying for it in gain. OFF by default, hardware
              validation pending.
rate_limit.py JointRateLimiter — max joint speed. The firmware computes kp*(pos-q)
              itself, so bounding the commanded ERROR is the ONLY way the host can
              bound torque. The HAL carries no slew guard; this is it.

Related, already under controller/ (reused, not duplicated): MujocoDynamics /
RegionDynamicModel (gravity source), JointPositionController (sim torque-mode PD),
ComplianceController (sim gravity-comp float), SelfCollisionGuard (barrier force for
future collision avoidance — wire it into COMPLIANT mode's t_ff when ready).
"""
from .admittance import FLOAT_DEFAULTS, SoftArmAdmittance
from .drag_follow import DRAG_DEFAULTS, DragFollow
from .ff_schedule import GravityFFScheduler
from .friction import (FRIC_SCALE_DEFAULT, FRIC_VEPS_DEFAULT, FrictionFF,
                       friction_deadband)
from .gravity import (FF_CLAMP, FF_POSE_PAD, FF_RAMP_S, ArmGravityCompensator,
                      HandsConfig)
from .impedance import ArmControlMode, ArmImpedance, ComplianceTuning, JointCommand
from .rate_limit import (ARM_MOTOR_BY_INDEX, CONTINUOUS_FRAC,
                         DEFAULT_BLEND_SPEED, DEFAULT_ERR_MAX,
                         DEFAULT_MAX_SPEED, DEFAULT_TORQUE_BUDGET,
                         MOTOR_PEAK_NM, JointGlideLimiter, JointRateLimiter,
                         arm_error_ceilings, arm_velocity_ff_ceilings,
                         torque_error_ceiling)
from .wire_gate import (DEFAULT_VEL_FF_MAX, NECK_ACTUATORS, NECK_MAX_SPEED,
                        deployed_glide_limiter, err_ceilings_enabled,
                        torque_gate_enabled, velocity_ff_ceilings)

__all__ = [
    "ArmGravityCompensator", "HandsConfig",
    "FF_CLAMP", "FF_RAMP_S", "FF_POSE_PAD",
    "ArmControlMode", "ArmImpedance", "JointCommand", "ComplianceTuning",
    "SoftArmAdmittance", "FLOAT_DEFAULTS", "GravityFFScheduler",
    "FrictionFF", "friction_deadband", "FRIC_SCALE_DEFAULT", "FRIC_VEPS_DEFAULT",
    "DragFollow", "DRAG_DEFAULTS",
    "JointRateLimiter", "JointGlideLimiter", "DEFAULT_MAX_SPEED",
    "DEFAULT_BLEND_SPEED", "DEFAULT_ERR_MAX", "DEFAULT_TORQUE_BUDGET",
    "torque_error_ceiling", "arm_error_ceilings", "arm_velocity_ff_ceilings",
    "deployed_glide_limiter", "torque_gate_enabled", "err_ceilings_enabled",
    "velocity_ff_ceilings",
    "NECK_ACTUATORS", "NECK_MAX_SPEED", "DEFAULT_VEL_FF_MAX",
    "MOTOR_PEAK_NM", "CONTINUOUS_FRAC", "ARM_MOTOR_BY_INDEX",
]
