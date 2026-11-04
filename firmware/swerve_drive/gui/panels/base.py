"""Base class every studio panel inherits from.

A panel is just a tk.Frame that knows how to mount/unmount itself in the host
content area and (optionally) refresh from a telemetry snapshot.  Adding a new
panel later is a three-step job:

    1. subclass Panel, build your UI in ``__init__``
    2. (optional) implement ``on_telemetry(snapshot)`` for live updates
    3. register it in app.py's ``PANELS`` list

The host calls ``mount`` when the panel becomes active and ``unmount`` when the
user switches away, so panels never fight over screen space.
"""

import tkinter as tk

from .. import theme as T


class Panel(tk.Frame):
    #: short label shown in the toolbar segmented control
    title = "Panel"
    #: accent color used for this panel's segmented highlight
    accent = T.ORANGE

    def __init__(self, host, app):
        super().__init__(host, bg=T.BG)
        self.app = app
        self.bus = app.bus

    def mount(self):
        self.pack(fill="both", expand=True)

    def unmount(self):
        self.pack_forget()

    def on_telemetry(self, snapshot):
        """Override to update widgets from a ``{can_id: MotorState}`` snapshot."""
        pass
