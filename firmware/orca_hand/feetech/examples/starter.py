#!/usr/bin/env python3
"""Super-lean ORCA hand starter. Edit the marked block and run.

    python3 starter.py                 # real hand, right side, auto-detect port
    python3 starter.py --mock          # no hardware
    python3 starter.py --model left --port /dev/ttyUSB0
"""
import argparse
import time

from orcahand import OrcaHand

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="right", choices=["left", "right"])
ap.add_argument("--port", default=None)
ap.add_argument("--mock", action="store_true")
a = ap.parse_args()

hand = OrcaHand.mock(a.model) if a.mock else OrcaHand(a.model, port=a.port)

with hand:                                  # connects (torque on); frees on exit
    # ---- your code here ----------------------------------------------------
    hand.set_named_position("flat")         # named pose
    time.sleep(1)

    hand.set_joint_positions({"index_mcp": 1.0, "index_pip": 1.0})  # radians
    time.sleep(1)

    q = hand.get_joint_positions()          # {joint_name: radians}
    print(f"index_mcp is at {q['index_mcp']:.2f} rad")
    # ------------------------------------------------------------------------
