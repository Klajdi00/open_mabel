"""
Upper-body dynamic model — arms + hands + torso + lift.

Extends the arms region with the Torso hinge and the cascaded Lift. The lift is
tendon-driven: commands go to the "Lift" actuator, while state/feedforward are
read on its cascade joint (config.LIFT_JOINT = "Lift Lower") — this model
handles that mapping, so callers only ever see the actuator-level view. The
mobile base is treated as environment (held).

Use for: upper-body gravity compensation (the raised body neither sags nor
drops), upper-body compliance (drag the torso/lift around), external
force/torque estimation anywhere on the upper chain.
"""
from ..regions import RegionDynamicModel

__all__ = ["UpperBodyDynamicModel"]


class UpperBodyDynamicModel(RegionDynamicModel):
    """Arms + Torso + Lift: the whole upper chain above the chassis."""
    region = "upper_body"
