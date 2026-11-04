"""poses.py - Hands -> Poses: drive a hand to a predefined grip through the bridge.

The predefined poses (``hands/orcahand/poses.py``) are stored as percent-of-range; this
view converts them to radian targets for the selected side and sends them over
``/<side>_hand/command`` - so, like everything else in the control center, it goes
through the hardware bridge (no direct hardware access). Applying a pose engages torque
first (no-snap hold-then-enable on the bridge) unless you turn that off.
"""

import tkinter as tk

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import contract
from style import (BG, GREEN, LABEL, LABEL2, LABEL3, ORANGE, RED, FONT_MONO_S,
                   FONT_SMALL, FONT_SUB, AppleSegmentedControl, accent_btn, btn)

_PRETTY = {"thumbs_up": "Thumbs Up"}


def _title(name):
    return _PRETTY.get(name, name.replace("_", " ").title())


class HandPosesView(tk.Frame):
    def __init__(self, parent, link):
        super().__init__(parent, bg=BG)
        self.link = link
        self.side = "left"
        self._engage = tk.BooleanVar(value=True)

        head = tk.Frame(self, bg=BG)
        head.pack(fill="x", padx=20, pady=(12, 6))
        tk.Label(head, text="Poses", bg=BG, fg=LABEL, font=FONT_SUB).pack(side="left")
        self._side_seg = AppleSegmentedControl(head, contract.HANDS["sides"], "left",
                                               self._switch_side, width=160, height=34,
                                               parent_bg=BG)
        self._side_seg.pack(side="left", padx=16)
        self._status = tk.Label(head, text="pick a pose", bg=BG, fg=LABEL2, font=FONT_SMALL)
        self._status.pack(side="right")

        tk.Label(self, text="Sends the grip to the selected hand over the bridge "
                            "(percent-of-range -> radians). Torque engages on apply.",
                 bg=BG, fg=LABEL3, font=FONT_MONO_S, anchor="w",
                 wraplength=900, justify="left").pack(fill="x", padx=20, pady=(0, 8))

        grid = tk.Frame(self, bg=BG)
        grid.pack(fill="x", padx=16)
        names = contract.pose_names() or []
        cols = 3
        for i, name in enumerate(names):
            b = accent_btn(grid, _title(name), lambda n=name: self._apply(n),
                           color=ORANGE, parent_bg=BG, width=180, height=46)
            b.grid(row=i // cols, column=i % cols, padx=8, pady=8, sticky="w")
        if not names:
            tk.Label(grid, text="no poses available (vendored config not found)",
                     bg=BG, fg=LABEL3, font=FONT_SMALL).pack(anchor="w")

        bar = tk.Frame(self, bg=BG)
        bar.pack(fill="x", padx=20, pady=(14, 6))
        tk.Checkbutton(bar, text="Engage torque on apply", variable=self._engage,
                       bg=BG, fg=LABEL2, selectcolor="#2C2C2E", activebackground=BG,
                       activeforeground=LABEL, font=FONT_SMALL).pack(side="left")
        btn(bar, "Free-drive", self._free_drive, bg=RED, parent_bg=BG).pack(side="left", padx=12)

    def _switch_side(self, side):
        self.side = side
        self._status.config(text=f"{side} hand", fg=LABEL2)

    def _apply(self, name):
        targets = contract.pose_targets(self.side, name)
        if not targets:
            self._status.config(text="no joint map for this side", fg=RED)
            return
        if self._engage.get():
            self.link.hand_torque(self.side, True)
        self.link.hand_command(self.side, targets)
        self._status.config(text=f"applied {_title(name)}", fg=GREEN)

    def _free_drive(self):
        self.link.hand_torque(self.side, False)
        self._status.config(text="free-drive (torque off)", fg=ORANGE)

    def refresh(self, *_):
        pass
