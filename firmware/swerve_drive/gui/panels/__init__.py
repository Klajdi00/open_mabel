"""Studio panels.  Register new panels in app.py's PANELS list."""

from .base import Panel
from .drive import DrivePanel
from .can import CanPanel

__all__ = ["Panel", "DrivePanel", "CanPanel"]
