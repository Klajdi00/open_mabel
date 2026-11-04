"""
Arms dynamic model — arm + hand dynamics only.

The 16 arm/wrist joints (7 arm + 1 ORCA wrist per side). The ORCA hands' masses
and inertias are part of the MuJoCo model, so every gravity / bias / mass-matrix
query on the arm joints already carries the hands; the finger joints themselves
stay position-controlled (hand control is decoupled by design — see
mabel.hand_retargeting). The torso, lift, and base are treated as environment:
their servos hold, and no feedforward is produced for them.

Use for: arms-only gravity compensation, arms-only compliance, external
force/torque estimation on the arms (push detection while the body holds).
"""
from ..regions import RegionDynamicModel

__all__ = ["ArmsDynamicModel"]


class ArmsDynamicModel(RegionDynamicModel):
    """Arms + hands region: gravity/bias feedforward, region mass matrix, and
    momentum-observer external-torque estimation on the arm joints only."""
    region = "arms"
