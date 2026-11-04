"""Back-compat body for the legacy `mabel.dynamics` path (aliased here by the finder
in `mabel/__init__.py`). The old dynamics.py was split by responsibility:

  * MujocoDynamics (a.k.a. UpperBodyDynamics) -> mabel.dynamic_models.model
  * Plant / SimPlant (the sim/real seam)      -> mabel.plant
  * ComplianceController                      -> mabel.whole_body_controller.compliance
  * COMPLIANT_ARM_JOINTS / LAZY_* joint sets  -> mabel.dynamic_models.regions

Kept so existing imports (`from mabel.dynamics import ...`) keep working. New code
should import from the canonical homes above.
"""
from ..dynamic_models.model import MujocoDynamics, UpperBodyDynamics, _dof  # noqa: F401
from ..dynamic_models.regions import (COMPLIANT_ARM_JOINTS, LAZY_JOINTS,    # noqa: F401
                                      LAZY_TENDON_ACTUATORS)
from ..plant import Plant, SimPlant                                         # noqa: F401
from ..whole_body_controller.compliance.compliance import ComplianceController  # noqa: F401
