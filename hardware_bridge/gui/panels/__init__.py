"""panels - one control surface per hardware-bridge subsystem."""

from .overview import OverviewPanel
from .base import BasePanel
from .arms import ArmsPanel
from .body import BodyPanel
from .hands import HandsPanel
from .lift import LiftPanel
from .monitor import MonitorPanel

__all__ = ["OverviewPanel", "BasePanel", "ArmsPanel", "BodyPanel", "HandsPanel",
           "LiftPanel", "MonitorPanel"]
