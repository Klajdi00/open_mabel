#!/usr/bin/env python3
"""Regenerate every figure in docs/figures/ from scratch (vector PDF).

    python3 docs/make_figures.py

Figures:
  architecture.pdf       the orcahand layering stack (API doc)
  angle_tick_map.pdf     joint angle <-> motor tick affine map, normal vs inverted
  calibration_ranges.pdf real per-joint calibrated tick spans for both hands

The last figure reads the shipped calibration JSON, so the numbers are the
measured hardware end-stops -- not hand-drawn.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from orcahand import Joint
from orcahand.config import _CONFIG_DIR

FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})


# ---------------------------------------------------------------- architecture
def architecture():
    layers = [
        ("Controllers / Policies / ROS node", "consumer code", "#d9e8fb"),
        ("OrcaHand  (hand.py)", "joint-space API: radians, names, safety", "#cfe8cf"),
        ("HandConfig / Joint / Calibration", "angle<->tick map, ROM, end-stops", "#f6e6c8"),
        ("FeetechBus  (bus.py)", "raw STS/SCS serial framing (pyserial)", "#f6d6d6"),
        ("Feetech HLS3915 / HLS3930 servos", "16 finger + 1 wrist, 1 Mbps TTL", "#e6e6e6"),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(layers) * 1.15)
    ax.axis("off")
    h = 0.9
    for i, (title, sub, col) in enumerate(reversed(layers)):
        y = i * 1.15 + 0.1
        ax.add_patch(FancyBboxPatch((0.4, y), 9.2, h,
                     boxstyle="round,pad=0.02,rounding_size=0.08",
                     fc=col, ec="#444", lw=1.0))
        ax.text(5.0, y + h * 0.62, title, ha="center", va="center",
                fontweight="bold", fontsize=10.5)
        ax.text(5.0, y + h * 0.24, sub, ha="center", va="center",
                fontsize=8.2, color="#333")
        if i:
            ax.add_patch(FancyArrowPatch((9.9, y + h + 0.12), (9.9, y - 0.02),
                         arrowstyle="-|>", mutation_scale=12, color="#777"))
    ax.text(9.95, len(layers) * 1.15 / 2, "depends on", rotation=90,
            ha="left", va="center", fontsize=7.5, color="#777")
    ax.set_title("orcahand layering: each layer depends only on the one below",
                 fontsize=10.5)
    fig.tight_layout()
    fig.savefig(FIG / "architecture.pdf")
    plt.close(fig)


# --------------------------------------------------------------- angle<->tick
def angle_tick_map():
    # a representative joint: index_mcp on the right hand (real calibration)
    j = Joint(name="index_mcp", motor_id=2, angle_min=-0.3491, angle_max=1.6581,
              invert=False, raw_min=1885, raw_max=2910, center=2397)
    ji = Joint(name="index_mcp (inverted)", motor_id=2, angle_min=-0.3491,
               angle_max=1.6581, invert=True, raw_min=1885, raw_max=2910, center=2397)

    angles = [j.angle_min + k * (j.angle_max - j.angle_min) / 200 for k in range(201)]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(angles, [j.angle_to_raw(a) for a in angles], lw=2, label="invert = false")
    ax.plot(angles, [ji.angle_to_raw(a) for a in angles], lw=2, ls="--",
            label="invert = true")
    ax.axhspan(j.raw_min, j.raw_max, color="#cfe8cf", alpha=0.35,
               label="calibrated tick range")
    ax.set_xlabel("joint angle  [rad]")
    ax.set_ylabel("motor goal position  [ticks]")
    ax.set_title("Affine joint-angle $\\to$ tick map (index\\_mcp, right hand)")
    ax.legend(loc="center right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIG / "angle_tick_map.pdf")
    plt.close(fig)


# --------------------------------------------------------- calibration ranges
def calibration_ranges():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.6), sharex=False)
    for ax, side in zip(axes, ("left", "right")):
        data = json.loads((_CONFIG_DIR / f"{side}_calibration.json").read_text())
        rows = sorted(data.values(), key=lambda r: r["joint"])
        names = [r["joint"] for r in rows]
        mins = [r["min"] for r in rows]
        spans = [r["max"] - r["min"] for r in rows]
        y = range(len(names))
        ax.barh(y, spans, left=mins, color="#5b8def", ec="#234", height=0.62)
        ax.set_yticks(list(y))
        ax.set_yticklabels(names, fontsize=7)
        ax.set_xlim(0, 4095)
        ax.set_xlabel("encoder ticks (0--4095)")
        ax.set_title(f"{side} hand", fontsize=10)
        ax.invert_yaxis()
    fig.suptitle("Measured calibrated travel per joint (from shipped calibration JSON)",
                 fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIG / "calibration_ranges.pdf")
    plt.close(fig)


if __name__ == "__main__":
    architecture()
    angle_tick_map()
    calibration_ranges()
    print("wrote:", *(p.name for p in sorted(FIG.glob("*.pdf"))))
