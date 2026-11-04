"""hands.py — the Control Center's Hands tab: the full ORCA Hand Studio.

This used to be a thin DDS-only view (a few sliders over /<side>_hand/command). It is now
the complete firmware "ORCA Hand Studio" — Control · Setup · Calibration · Poses · Teleop,
with the live telemetry sidebar — embedded here as :class:`hand_studio.studio.HandStudio`.

The studio talks to the ROS-free **hand bridge** (``mabel_hw.rpc.hand_server``) instead of opening
the serial port directly, so it is a pure bridge client (matching the rest of the Control
Center) AND it works on macOS, where rclpy is unavailable and the right hand was previously
undetectable. On the robot it transparently borrows the serial port from the ROS bridge
while open and hands it back on close.
"""

import os
import sys
import tkinter as tk

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # gui/

from style import BG, RED, FONT_MONO_S


class HandsPanel(tk.Frame):
    """Container that hosts the embedded ORCA Hand Studio."""

    def __init__(self, parent, link):
        super().__init__(parent, bg=BG)
        self.link = link
        self._studio = None
        try:
            from hand_studio.studio import HandStudio
            self._studio = HandStudio(self, link)
            self._studio.pack(fill="both", expand=True)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            tk.Label(self, text=f"Hand Studio failed to load:\n{exc}",
                     bg=BG, fg=RED, justify="left", font=FONT_MONO_S).pack(
                         padx=40, pady=60, anchor="w")

    def refresh(self, snapshot):
        if self._studio is not None:
            try:
                self._studio.refresh(snapshot)
            except Exception:
                pass

    def teardown(self):
        if self._studio is not None:
            try:
                self._studio.teardown()
            except Exception:
                pass
