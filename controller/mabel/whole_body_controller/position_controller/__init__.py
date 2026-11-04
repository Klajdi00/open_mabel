"""Day-1 position control: plain per-joint PD over the Plant seam (no model,
no system ID). See pd.py for the law and the Day-2 gravity-feedforward hook."""
from .pd import JointPositionController  # noqa: F401

__all__ = ["JointPositionController"]
