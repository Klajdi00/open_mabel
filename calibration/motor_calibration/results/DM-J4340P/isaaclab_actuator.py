# MABEL actuator calibration: DM-J4340P
# † MEASURED on the bench; others are datasheet/policy — see comments.
from isaaclab.actuators import DCMotorCfg

DM-J4340P_actuator = DCMotorCfg(
    joint_names_expr=[".*"],
    effort_limit=9.0,  # DATASHEET — FLAG: continuous/thermal limit, not bench-measurable
    saturation_effort=27.0,  # DATASHEET / peak t_max — FLAG: stall torque needs a load (dynamometer/brake); not measurable on a free shaft
    velocity_limit=5.8946,  # MEASURED (max-effort terminal speed ≈ no-load)
    stiffness={".*": 0.0},  # CONTROL POLICY — choose per task (not a physical parameter)
    damping=0.183973,  # viscous Fv MEASURED; but Isaac PD `damping` is a CONTROL gain — set per policy (this is the physical value for reference)
    armature=0.02590013,  # MEASURED (chirp inertia sweep)
    friction=0.339229,  # MEASURED (velocity staircase)
)
