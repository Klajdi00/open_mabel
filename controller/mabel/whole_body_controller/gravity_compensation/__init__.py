"""Gravity-compensation feedforward (region-selectable): the model torque that
holds arms / upper body / whole body against gravity, leaving it back-drivable."""
from .gravity_comp import GravityCompController  # noqa: F401

__all__ = ["GravityCompController"]
