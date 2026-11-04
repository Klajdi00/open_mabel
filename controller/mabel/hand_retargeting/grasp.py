"""
Minimal pinch -> grasp mapping for the dexterous hands.

A grasp scalar in [0, 1] curls the four fingers' MCP/PIP joints (and the thumb)
from open toward their flexed limits. This is a deliberately simple stand-in for
full per-finger retargeting, which can be added later by mapping each Vision Pro
finger joint to its MABEL counterpart.
"""
import numpy as np

# Flexion actuators that close when driven toward their upper ctrl limit.
_FLEX_SUFFIXES = [
    "Thumb PIP", "Thumb DIP",
    "Index MCP", "Index PIP",
    "Middle MCP", "Middle PIP",
    "Ring MCP", "Ring PIP",
    "Pinky MCP", "Pinky PIP",
]


def apply_grasp(model, data, side, g):
    """side: 'Left' or 'Right'; g in [0,1] (0 = open, 1 = closed fist)."""
    g = float(np.clip(g, 0.0, 1.0))
    for suffix in _FLEX_SUFFIXES:
        name = f"{side} {suffix}"
        try:
            act_id = model.actuator(name).id
        except KeyError:
            continue
        lo, hi = model.actuator_ctrlrange[act_id]
        data.actuator(name).ctrl[0] = lo + g * (hi - lo)
