"""Shared figure style for the arm-calibration experiments.

Vector PDF, headless, one consistent palette so every figure in
docs/ARM_CALIBRATION.tex reads as one document.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Palette: neutral graphite for reference/truth, blue for the prior/CAD, orange
# for the identified result, red for danger/limits, green for "good".
INK = "#2b2b2b"
GRID = "#d8d8d8"
CAD = "#4c7fb8"       # prior / CAD
EST = "#e8833a"       # identified
TRUTH = "#2b2b2b"     # ground truth
DANGER = "#c0392b"    # friction floor, limits
GOOD = "#4a9d5b"
MUTED = "#9aa0a6"
PALETTE = [CAD, EST, GOOD, DANGER, MUTED, "#8e6fb8", "#c9a227"]


def apply():
    plt.rcParams.update({
        "figure.figsize": (7.0, 4.2),
        "figure.dpi": 110,
        "savefig.dpi": 110,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.8,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "lines.linewidth": 1.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save(fig, path):
    fig.savefig(path, format="pdf")
    plt.close(fig)
    print(f"  figure -> {path}")
