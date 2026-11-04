#!/usr/bin/env python3
"""Tkinter slider panel -- one slider per joint, live readback.

Each slider commands a joint as 0-100 % of its calibrated range; the readback
column shows the measured percent polled from the hand. A lean rewrite of the
legacy ``joint_slider.py`` on top of the :mod:`orcahand` API.

    python3 scripts/slider.py --model left
    python3 scripts/slider.py --model left --mock      # no hardware
"""
from __future__ import annotations

import argparse

from _common import add_common_args, make_hand

try:
    import tkinter as tk
    from tkinter import ttk
except Exception as e:  # pragma: no cover - headless / no Tk
    raise SystemExit(f"tkinter is required for the slider GUI: {e}")

_FINGERS = ["thumb", "index", "middle", "ring", "pinky", "wrist"]


def _finger_of(joint: str) -> str:
    f = joint.split("_")[0]
    return f if f in _FINGERS else "other"


class SliderApp:
    def __init__(self, hand, poll_hz: float = 10.0):
        self.hand = hand
        self.period = int(1000 / max(1.0, poll_hz))
        self.scales: dict[str, tk.Scale] = {}
        self.readouts: dict[str, tk.Label] = {}

        self.root = tk.Tk()
        self.root.title(f"ORCA {hand.side} hand — joint sliders")
        self._build()
        self.root.protocol("WM_DELETE_WINDOW", self._quit)
        self._poll()

    # -- UI ---------------------------------------------------------------
    def _build(self) -> None:
        joints = sorted(self.hand.joint_names,
                        key=lambda j: (_FINGERS.index(_finger_of(j))
                                       if _finger_of(j) in _FINGERS else 99, j))
        row = 0
        last_finger = None
        for j in joints:
            fg = _finger_of(j)
            if fg != last_finger:
                ttk.Separator(self.root, orient="horizontal").grid(
                    row=row, column=0, columnspan=3, sticky="ew", pady=(8, 2))
                row += 1
                tk.Label(self.root, text=fg.upper(), font=("TkDefaultFont", 9, "bold"),
                         anchor="w").grid(row=row, column=0, sticky="w", padx=6)
                row += 1
                last_finger = fg

            tk.Label(self.root, text=j, width=12, anchor="w").grid(
                row=row, column=0, sticky="w", padx=6)
            s = tk.Scale(self.root, from_=0, to=100, orient="horizontal", length=280,
                         command=lambda v, jn=j: self._on_move(jn, v))
            s.set(50)
            s.grid(row=row, column=1, padx=4)
            self.scales[j] = s
            lbl = tk.Label(self.root, text="--", width=6, anchor="e", fg="#0a7")
            lbl.grid(row=row, column=2, padx=6)
            self.readouts[j] = lbl
            row += 1

        bar = tk.Frame(self.root)
        bar.grid(row=row, column=0, columnspan=3, pady=8)
        self.torque_on = True
        self.torque_btn = tk.Button(bar, text="Torque: ON", width=12, command=self._toggle_torque)
        self.torque_btn.pack(side="left", padx=4)
        tk.Button(bar, text="Center all", width=12, command=self._center).pack(side="left", padx=4)
        tk.Button(bar, text="Quit", width=8, command=self._quit).pack(side="left", padx=4)

    # -- actions ----------------------------------------------------------
    def _on_move(self, joint: str, value) -> None:
        try:
            self.hand.set_joint_percent({joint: float(value)})
        except Exception:
            pass

    def _toggle_torque(self) -> None:
        self.torque_on = not self.torque_on
        (self.hand.enable_torque if self.torque_on else self.hand.disable_torque)()
        self.torque_btn.config(text=f"Torque: {'ON' if self.torque_on else 'OFF'}")

    def _center(self) -> None:
        for j, s in self.scales.items():
            s.set(50)
        self.hand.set_joint_percent({j: 50.0 for j in self.scales})

    def _poll(self) -> None:
        try:
            state = self.hand.get_state()
            for j, lbl in self.readouts.items():
                jt = self.hand.config.joint(j)
                raw = state.raw_position.get(j)
                if raw is not None:
                    lbl.config(text=f"{jt.raw_to_percent(raw):5.0f}%")
        except Exception:
            pass
        self.root.after(self.period, self._poll)

    def _quit(self) -> None:
        try:
            self.hand.disconnect()
        finally:
            self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Tkinter joint-slider panel.")
    add_common_args(ap)
    args = ap.parse_args(argv)
    hand = make_hand(args).connect()
    SliderApp(hand).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
