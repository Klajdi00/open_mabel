#!/usr/bin/env python3
"""Regenerate every MABEL BOM artefact from the CSVs in ``BOM/data/``.

One command reproduces the whole deliverable:

    python3 BOM/tools/build_bom.py

Model (mirrors the July 2026 Build Guide):

    core        48 line items that are identical in every build      -> $7,897.36
    choices     6 groups you pick from (compute, wrist, head, base
                depth, lidar, and the cabling those imply)
    builds      core + one option per group                          -> Essential /
                                                                        Recommended /
                                                                        Maximum
    open_items  corrections and uncosted hardware, held OUT of the totals

Reads   : data/{core,choices,machined,taobao,build_or_buy,open_items,
                peer_platforms}.csv
Writes  : figures/*.pdf                (vector, matplotlib Agg)
          generated/*.tex              (tables + \\newcommand numbers)
          generated/bom_summary.json   (the numbers the website renders)
          website/assets/bom/*         (published bundle: JS, CSVs, both PDFs)

The CSVs are the single source of truth: edit a price there, re-run this, and
the PDF, the figures and the website all move together.
"""
from __future__ import annotations

import csv
import json
import shutil
import sys
import textwrap
from collections import OrderedDict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]          # BOM/
REPO = ROOT.parent                                   # MABEL/
DATA, FIGS, GEN = ROOT / "data", ROOT / "figures", ROOT / "generated"

# ── figure style ────────────────────────────────────────────────────────────
sys.path.insert(0, str(REPO / "papers" / "ral2026" / "scripts"))
try:
    import figstyle                                   # type: ignore
except Exception:                                     # pragma: no cover
    figstyle = None

INK, GRID, GRIDTEXT = "#2B2B2B", "#DADCE0", "#888888"
BEST, HIGHLIGHT = "#FF9500", "#FFF5C8"
PALETTE = OrderedDict([
    ("blue", ("#3B6CA8", "#EAF2FB")), ("green", ("#4E8C5A", "#EAF6EC")),
    ("coral", ("#B45B52", "#FBECEC")), ("teal", ("#3A9A94", "#E1F0EF")),
    ("purple", ("#7C5AA6", "#EFE9F5")), ("amber", ("#B8893E", "#FBF1E0")),
    ("gray", ("#6B7280", "#EEF0F3")),
])
ORDER = list(PALETTE)


def apply_style():
    if figstyle is not None:
        figstyle.apply()
        return
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8.5,
        "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "axes.edgecolor": GRIDTEXT, "axes.linewidth": 0.6,
        "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK,
        "ytick.color": INK, "savefig.bbox": "tight", "pdf.fonttype": 42,
    })


def despine(ax, grid_axis="y"):
    if figstyle is not None:
        figstyle.despine(ax, grid_axis)
        return
    ax.spines[["top", "right"]].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRIDTEXT)
        ax.spines[s].set_linewidth(0.6)
    if grid_axis:
        ax.grid(True, axis=grid_axis, ls=":", lw=0.6, color=GRID, zorder=0)


def cat(i):
    return PALETTE[ORDER[i % len(ORDER)]]


# ── constants ───────────────────────────────────────────────────────────────
# The Build Guide's own rate, implied by its line reading CNY 599 = $85.91.
# Every CNY figure in this repo uses it, so rescaling is a one-line change.
CNY_TO_USD = 85.91 / 599.0          # = 0.1434224  (1 USD = 6.9724 CNY)
PRICE_DATE = "31 July 2026"

# NVIDIA raised every Jetson module and dev-kit price on 22 July 2026.
JETSON_RISE = {"date": "22 July 2026",
               "thor_before": 3499.00, "thor_after": 5499.00,
               "nano_before": 249.00, "nano_after": 399.00,
               # what the original sheet actually carried, not the MSRP
               "thor_as_quoted": 3500.00}

# The bill as first costed, for historical reference only: Thor at $3,500 and
# filament not yet costed.
AS_QUOTED_TOTAL = 12969.37

# Bambu Lab's standing bulk offer on PLA Basic.
BULK = {"vendor": "Bambu Lab", "discount": 0.25, "min_qty": 4,
        "applies_to": "3D printed material"}

FUNCTIONAL_DOF = 56
MOTOR_COUNT = 59

SECTION_ORDER = ["Mobile base", "Body / torso", "Arms - both", "Hands - both",
                 "Neck / head", "Structural hardware",
                 "Electronics, power & cabling", "3D printed material"]

FGROUPS = OrderedDict([
    ("actuation", "Actuation"), ("compute", "Compute"), ("perception", "Perception"),
    ("structure", "Structure"), ("data", "Data & connectivity"),
    ("power", "Power & wiring"), ("misc", "Control & misc"),
])

BUILDS = ["essential", "recommended", "maximum"]
BUILD_LABEL = {"essential": "Essential", "recommended": "Recommended",
               "maximum": "Maximum"}
BUILD_BLURB = {
    "essential": "Raspberry Pi 5, no base depth camera, LD19 lidar. A genuine "
                 "working robot rather than a stripped one - the mechanism is "
                 "identical and every difference upgrades later without touching "
                 "the structure.",
    "recommended": "16 GB of memory, 1200P global-shutter wrists, RPLIDAR C1. "
                   "Keeps what decides which policies run onboard and spends "
                   "nothing on sensing you are unlikely to use.",
    "maximum": "Jetson Thor, RealSense D405 wrists, ZED Mini head, base depth "
               "camera. Everything the original bill of materials specified.",
}


# ── helpers ─────────────────────────────────────────────────────────────────
def read_csv(name):
    with open(DATA / name, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def to_usd(price, currency):
    return price * CNY_TO_USD if currency.upper() == "CNY" else price


def tex(s):
    return (s.replace("\\", r"\textbackslash{}").replace("&", r"\&")
             .replace("%", r"\%").replace("$", r"\$").replace("#", r"\#")
             .replace("_", r"\_").replace("{", r"\{").replace("}", r"\}")
             .replace("~", r"\textasciitilde{}").replace("^", r"\textasciicircum{}"))


def money(x, cents=False):
    return f"{x:,.2f}" if cents else f"{x:,.0f}"


# ── load ────────────────────────────────────────────────────────────────────
def load_core():
    rows = []
    for r in read_csv("core.csv"):
        qty, unit = float(r["qty"]), float(r["unit_price"])
        unit_usd = to_usd(unit, r["currency"])
        rows.append({**r, "qty": qty, "unit_price": unit, "unit_usd": unit_usd,
                     "ext_usd": round(qty * unit_usd, 2)})
    return rows


def load_choices():
    rows = []
    for r in read_csv("choices.csv"):
        qty, unit = float(r["qty"]), float(r["unit_price"])
        unit_usd = to_usd(unit, r["currency"])
        rows.append({**r, "qty": qty, "unit_usd": unit_usd,
                     "ext_usd": round(qty * unit_usd, 2),
                     "tiers": r["tier"].split()})
    return rows


def core_sections(core):
    out = OrderedDict((s, 0.0) for s in SECTION_ORDER)
    for r in core:
        out[r["section"]] = round(out.get(r["section"], 0.0) + r["ext_usd"], 2)
    return out


def choice_groups(choices):
    groups = OrderedDict()
    for r in choices:
        groups.setdefault(r["choice_id"], {"id": r["choice_id"], "name": r["choice"],
                                           "qty": int(r["qty"]), "options": []})
        groups[r["choice_id"]]["options"].append(r)
    return groups


def build_totals(core_total, groups):
    out = OrderedDict()
    for b in BUILDS:
        picks, extra = [], 0.0
        for g in groups.values():
            opt = next((o for o in g["options"] if b in o["tiers"]), None)
            if opt is None:                       # fall back to the cheapest option
                opt = min(g["options"], key=lambda o: o["ext_usd"])
            picks.append({"choice_id": g["id"], "choice": g["name"],
                          "option": opt["option"], "usd": opt["ext_usd"]})
            extra += opt["ext_usd"]
        out[b] = {"id": b, "name": BUILD_LABEL[b], "blurb": BUILD_BLURB[b],
                  "picks": picks, "choices_usd": round(extra, 2),
                  "total": round(core_total + extra, 2)}
    return out


def functional_split(rows):
    """Cost by what a part does, not where it bolts on."""
    out = OrderedDict((k, 0.0) for k in FGROUPS)
    for r in rows:
        out[r["fgroup"]] = round(out.get(r["fgroup"], 0.0) + r["ext_usd"], 2)
    return OrderedDict((k, v) for k, v in out.items() if v > 0)


def _saving(sensing, key):
    """The span of one named sensing group -- what dropping to its floor saves."""
    return next(s["saving"] for s in sensing if key.lower() in s["name"].lower())


def picked_rows(groups, build):
    out = []
    for g in groups.values():
        opt = next((o for o in g["options"] if build in o["tiers"]), None)
        if opt and opt["ext_usd"] > 0:
            out.append(opt)
    return out


# ── figures ─────────────────────────────────────────────────────────────────
def fig_builds(builds, core_total):
    apply_style()
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    for i, b in enumerate(BUILDS):
        rec, hl = builds[b], (b == "recommended")
        line, fill = cat(i)
        ax.bar(i, core_total, color="#EEF0F3", edgecolor=PALETTE["gray"][0],
               linewidth=0.8, width=0.58, zorder=3)
        ax.bar(i, rec["choices_usd"], bottom=core_total,
               color=(HIGHLIGHT if hl else fill), edgecolor=(BEST if hl else line),
               linewidth=(1.3 if hl else 0.8), width=0.58, zorder=3)
        ax.text(i, rec["total"] + 260, f"\\${money(rec['total'])}", ha="center",
                fontsize=8.5, color=INK,
                fontweight=("bold" if hl else "normal"), zorder=4)
    ax.axhline(core_total, ls="--", lw=0.7, color=PALETTE["gray"][0], zorder=2)
    ax.text(-0.44, core_total * 0.5, f"fixed core\n\\${money(core_total)}",
            fontsize=6.6, color=GRIDTEXT, va="center", ha="left", zorder=5)
    ax.set_xticks(range(len(BUILDS)))
    ax.set_xticklabels([BUILD_LABEL[b] for b in BUILDS])
    ax.set_ylabel("build cost (USD)")
    ax.set_ylim(0, builds["maximum"]["total"] * 1.16)
    ax.set_title("Three ways to build MABEL — one core, five choices", loc="left")
    despine(ax)
    fig.savefig(FIGS / "bom_builds.pdf")
    plt.close(fig)


def fig_choices(groups, builds):
    """Each choice group's min-max span, with the three build picks marked."""
    apply_style()
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    gs = list(groups.values())
    for i, g in enumerate(gs):
        costs = [o["ext_usd"] for o in g["options"]]
        lo, hi = min(costs), max(costs)
        ax.plot([max(lo, 1), max(hi, 1)], [i, i], color=PALETTE["gray"][0], lw=1.0, zorder=2)
        for c in costs:
            ax.plot(max(c, 1), i, "o", ms=3.4, mfc="white",
                    mec=PALETTE["gray"][0], mew=0.9, zorder=3)
        for b, colr in (("essential", PALETTE["blue"][0]), ("recommended", BEST),
                        ("maximum", PALETTE["coral"][0])):
            pick = next(p for p in builds[b]["picks"] if p["choice_id"] == g["id"])
            ax.plot(max(pick["usd"], 1), i, "o", ms=5.6, mfc=colr, mec="white",
                    mew=0.8, zorder=4 + (b == "recommended"))
        ax.text(hi * 1.5 + 30, i, f"\\${money(lo)}–\\${money(hi)}", fontsize=6.4,
                va="center", color=GRIDTEXT)
    ax.set_yticks(range(len(gs)))
    ax.set_yticklabels([f"{g['id']}  {g['name']}" for g in gs], fontsize=7)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(0.8, 40000)
    ax.set_xticks([1, 10, 100, 1000, 10000])
    ax.set_xticklabels(["\\$0", "\\$10", "\\$100", "\\$1k", "\\$10k"])
    ax.set_xlabel("cost of the choice (log scale; \\$0 plotted at the axis floor)")
    ax.set_title("The five choices — where the spread actually lives", loc="left")
    handles = [plt.Line2D([], [], marker="o", ls="", mfc=c, mec="white", ms=5.6, label=l)
               for c, l in ((PALETTE["blue"][0], "Essential"), (BEST, "Recommended"),
                            (PALETTE["coral"][0], "Maximum"))]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=6.8)
    despine(ax, grid_axis="x")
    fig.savefig(FIGS / "bom_choices.pdf")
    plt.close(fig)


def fig_functional(split, total, label):
    apply_style()
    fig, ax = plt.subplots(figsize=(5.4, 2.9))
    items = sorted(split.items(), key=lambda kv: kv[1])
    for i, (k, v) in enumerate(items):
        line, fill = cat(i)
        ax.barh(i, v, color=fill, edgecolor=line, linewidth=0.8, height=0.68, zorder=3)
        ax.text(v + total * 0.012, i, f"\\${money(v)}  ({v / total * 100:.1f}%)",
                va="center", fontsize=6.8, color=INK, zorder=4)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([FGROUPS[k] for k, _ in items])
    ax.set_xlabel("cost (USD)")
    ax.set_xlim(0, max(v for _, v in items) * 1.42)
    ax.set_title(f"Where the money goes — {label} build, \\${money(total)}", loc="left")
    despine(ax, grid_axis="x")
    fig.savefig(FIGS / "bom_functional.pdf")
    plt.close(fig)


def fig_volume(machined):
    apply_style()
    fig, ax = plt.subplots(figsize=(5.4, 2.7))
    tiers = [("p1", 1), ("p2", 2), ("p10", 10), ("p50", 50), ("p100", 100), ("p1000", 1000)]
    totals = [sum(float(m[k]) * float(m["qty"]) for m in machined) for k, _ in tiers]
    xs = list(range(len(tiers)))
    ax.plot(xs, totals, "-o", color=PALETTE["blue"][0], ms=4.4, lw=1.2,
            mfc="white", mec=PALETTE["blue"][0], zorder=3)
    for x, v in zip(xs, totals):
        ax.annotate(f"\\${money(v)}", (x, v), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=6.8, color=INK)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"$\\times${n}" for _, n in tiers])
    ax.set_xlabel("sets ordered")
    ax.set_ylabel("machined set, cost per robot (USD)")
    ax.set_ylim(0, max(totals) * 1.24)
    ax.set_title("Setup dominates at quantity one — the case for a group buy", loc="left")
    despine(ax)
    fig.savefig(FIGS / "bom_volume.pdf")
    plt.close(fig)


def fig_buildbuy(rows):
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(5.8, 2.8))
    for ax, cat_name in zip(axes, ["Arms", "Hands"]):
        sel = sorted([r for r in rows if r["category"] == cat_name],
                     key=lambda r: float(r["both_usd"]))
        vals = [float(r["both_usd"]) for r in sel]
        for i, r in enumerate(sel):
            mine = r["is_mabel"] == "yes"
            line, fill = (PALETTE["coral"] if mine else PALETTE["gray"])
            ax.barh(i, float(r["both_usd"]),
                    color=(HIGHLIGHT if mine else fill),
                    edgecolor=(BEST if mine else line),
                    linewidth=(1.3 if mine else 0.8), height=0.62, zorder=3)
            ax.text(float(r["both_usd"]) + max(vals) * 0.03, i,
                    f"\\${money(float(r['both_usd']))}", va="center", fontsize=6.2,
                    color=INK, fontweight=("bold" if mine else "normal"))
        ax.set_yticks(range(len(sel)))
        ax.set_yticklabels([textwrap.fill(r["option"].split(",")[0], 20) for r in sel],
                           fontsize=6.2)
        ax.invert_yaxis()
        ax.set_xlim(0, max(vals) * 1.36)
        ax.set_title(f"{cat_name} — both sides", loc="left", fontsize=8)
        ax.set_xlabel("USD", fontsize=7)
        despine(ax, grid_axis="x")
    fig.tight_layout()
    fig.savefig(FIGS / "bom_buildbuy.pdf")
    plt.close(fig)


def fig_peers(peers, mabel_kusd):
    apply_style()
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    rows = sorted(peers, key=lambda r: float(r["cost_kusd"]))
    names = [r["platform"] for r in rows]
    vals = [float(r["cost_kusd"]) for r in rows]
    opens = [r["open"] == "yes" for r in rows]
    ins = next((i for i, v in enumerate(vals) if v > mabel_kusd), len(vals))
    names.insert(ins, "MABEL (Recommended)")
    vals.insert(ins, mabel_kusd)
    opens.insert(ins, True)
    for i, (v, is_open) in enumerate(zip(vals, opens)):
        mine = names[i].startswith("MABEL")
        line, fill = (PALETTE["blue"] if is_open else PALETTE["gray"])
        ax.barh(i, v, color=(HIGHLIGHT if mine else fill),
                edgecolor=(BEST if mine else line), linewidth=(1.3 if mine else 0.8),
                height=0.66, zorder=3)
        ax.text(v + 1.4, i, f"\\${v:.1f}k", va="center", fontsize=6.8, color=INK,
                fontweight=("bold" if mine else "normal"), zorder=4)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("platform cost (thousand USD)")
    ax.set_xlim(0, max(vals) * 1.14)
    ax.set_title("Mobile bimanual platforms — hardware cost", loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, fc=PALETTE["blue"][1], ec=PALETTE["blue"][0], lw=0.8),
               plt.Rectangle((0, 0), 1, 1, fc=PALETTE["gray"][1], ec=PALETTE["gray"][0], lw=0.8),
               plt.Rectangle((0, 0), 1, 1, fc=HIGHLIGHT, ec=BEST, lw=1.3)]
    ax.legend(handles, ["open-source", "commercial", "this work"],
              loc="lower right", frameon=False, fontsize=6.8)
    despine(ax, grid_axis="x")
    fig.savefig(FIGS / "bom_peers.pdf")
    plt.close(fig)


# ── LaTeX ───────────────────────────────────────────────────────────────────
FLAG = {"corrected": r"$\dagger$", "disputed": r"$\ast$", "sheet": "", "tbd": ""}


def emit_core_table(core, secs, core_total):
    head = (r"\textbf{Ref} & \textbf{Item} & \textbf{Qty} & \textbf{Unit (\$)} & "
            r"\textbf{Ext.\ (\$)} \\")
    L = [r"% AUTO-GENERATED by BOM/tools/build_bom.py -- do not edit by hand.",
         r"{\small", r"\begin{longtable}{@{}l p{0.42\linewidth} r r r@{}}",
         r"\caption{The fixed core: every line identical in the Essential, "
         r"Recommended and Maximum builds. $\dagger$ part number corrected against "
         r"the verified vendor listing; $\ast$ disputed against the repository "
         r"(\S\ref{sec:open}); \textyen\ quoted in CNY and converted by "
         r"Eq.~\ref{eq:fx}.}\label{tab:core}\\",
         r"\toprule", head, r"\midrule", r"\endfirsthead",
         r"\multicolumn{5}{@{}l}{\itshape Table \thetable\ -- continued}\\",
         r"\toprule", head, r"\midrule", r"\endhead"]
    for sec in SECTION_ORDER:
        rows = [r for r in core if r["section"] == sec]
        if not rows:
            continue
        L.append(r"\multicolumn{5}{@{}l}{\textbf{%s}}\\[1pt]" % tex(sec))
        for r in rows:
            cur = "" if r["currency"] == "USD" else r"\,\textsuperscript{\textyen}"
            tbd = (r["status"] == "tbd")
            L.append(r"%s & \bomitem{%s%s} & %d & %s%s & %s \\" % (
                tex(r["ref"]), tex(r["item"]), FLAG.get(r["status"], ""),
                int(r["qty"]),
                ("\\textit{TBD}" if tbd else money(r["unit_usd"], cents=True)),
                ("" if tbd else cur),
                ("\\textit{TBD}" if tbd else money(r["ext_usd"], cents=True))))
        L.append(r" & \bomitem{\textit{subtotal}} & & & \textbf{%s} \\[3pt]"
                 % money(secs[sec], cents=True))
    L += [r"\midrule",
          r" & \textbf{Core subtotal} & & & \textbf{%s} \\" % money(core_total, cents=True),
          r"\bottomrule", r"\end{longtable}", r"}"]
    (GEN / "bom_core.tex").write_text("\n".join(L) + "\n")


def emit_choices_table(groups):
    head = r"\textbf{Ref} & \textbf{Option} & \textbf{Cost (\$)} & \textbf{Build} \\"
    L = [r"{\small", r"\begin{longtable}{@{}l p{0.44\linewidth} r l@{}}",
         r"\caption{The five choices, plus the cabling they imply. Every option "
         r"publishes a standard interface, so any row swaps without touching the "
         r"mechanism.}\label{tab:choices}\\",
         r"\toprule", head, r"\midrule", r"\endfirsthead",
         r"\toprule", head, r"\midrule", r"\endhead"]
    for g in groups.values():
        costs = [o["ext_usd"] for o in g["options"]]
        L.append(r"\multicolumn{4}{@{}l}{\textbf{%s\ %s}\quad\footnotesize %d options, "
                 r"\$%s--\$%s}\\[1pt]" % (
                     tex(g["id"]), tex(g["name"]), len(g["options"]),
                     money(min(costs)), money(max(costs))))
        for o in g["options"]:
            tiers = ", ".join(BUILD_LABEL[t] for t in o["tiers"] if t in BUILD_LABEL)
            qty = r" ($\times$%d)" % int(o["qty"]) if o["qty"] > 1 else ""
            L.append(r" & \bomitem{%s%s} & %s & \footnotesize %s \\" % (
                tex(o["option"]), qty, money(o["ext_usd"], cents=True), tex(tiers)))
    L += [r"\bottomrule", r"\end{longtable}", r"}"]
    (GEN / "bom_choices.tex").write_text("\n".join(L) + "\n")


def emit_builds_table(builds, groups, core_total):
    L = [r"\begin{tabular}{@{}l r r r@{}}", r"\toprule",
         r" & \textbf{Essential} & \textbf{Recommended} & \textbf{Maximum} \\",
         r"\midrule",
         r"Fixed core (\S\ref{sec:core}) & %s & %s & %s \\" % tuple([money(core_total)] * 3)]
    for g in groups.values():
        vals = []
        for b in BUILDS:
            pick = next(p for p in builds[b]["picks"] if p["choice_id"] == g["id"])
            vals.append(money(pick["usd"], cents=True))
        L.append(r"%s\ %s & %s & %s & %s \\" % (tex(g["id"]), tex(g["name"]), *vals))
    L += [r"\midrule",
          r"\rowcolor{hl}\textbf{Total} & \textbf{%s} & \textbf{%s} & \textbf{%s} \\" % tuple(
              money(builds[b]["total"]) for b in BUILDS),
          r"\bottomrule", r"\end{tabular}"]
    (GEN / "bom_builds.tex").write_text("\n".join(L) + "\n")


def emit_functional_table(split, total):
    L = [r"\begin{tabular}{@{}l r r@{}}", r"\toprule",
         r"\textbf{What it does} & \textbf{Cost (\$)} & \textbf{Share} \\", r"\midrule"]
    for k, v in sorted(split.items(), key=lambda kv: -kv[1]):
        L.append(r"%s & %s & %.1f\%% \\" % (tex(FGROUPS[k]), money(v), v / total * 100))
    L += [r"\midrule",
          r"\textbf{Total} & \textbf{%s} & \textbf{100.0\%%} \\" % money(total),
          r"\bottomrule", r"\end{tabular}"]
    (GEN / "bom_functional.tex").write_text("\n".join(L) + "\n")


def emit_machined_table(machined):
    tiers = ["p1", "p2", "p10", "p50", "p100", "p1000"]
    L = [r"{\footnotesize", r"\begin{tabular}{@{}p{0.26\linewidth} c r r r r r r@{}}",
         r"\toprule",
         r"\textbf{Part} & \textbf{Qty} & $\times$\textbf{1} & $\times$\textbf{2} & "
         r"$\times$\textbf{10} & $\times$\textbf{50} & $\times$\textbf{100} & "
         r"$\times$\textbf{1000} \\", r"\midrule"]
    for m in machined:
        L.append(r"%s & %d & %s \\" % (
            tex(m["part"]), int(float(m["qty"])),
            " & ".join(money(float(m[t]), cents=True) for t in tiers)))
    tot = [sum(float(m[t]) * float(m["qty"]) for m in machined) for t in tiers]
    L += [r"\midrule",
          r"\textbf{Full set per robot} & & %s \\" % " & ".join(
              r"\textbf{%s}" % money(v, cents=True) for v in tot),
          r"\bottomrule", r"\end{tabular}", r"}"]
    (GEN / "bom_machined.tex").write_text("\n".join(L) + "\n")
    return tot


def emit_buildbuy_table(rows):
    L = [r"{\small", r"\begin{tabular}{@{}p{0.46\linewidth} r r@{}}", r"\toprule",
         r"\textbf{Option} & \textbf{Both sides (\$)} & \textbf{vs.\ MABEL} \\",
         r"\midrule"]
    for cat_name in ["Arms", "Hands"]:
        sel = [r for r in rows if r["category"] == cat_name]
        base = float(next(r for r in sel if r["is_mabel"] == "yes")["both_usd"])
        L.append(r"\multicolumn{3}{@{}l}{\textbf{%s --- both sides}}\\[1pt]" % cat_name)
        for r in sel:
            v = float(r["both_usd"])
            delta = "---" if r["is_mabel"] == "yes" else r"$+$%.0f\%%" % ((v / base - 1) * 100)
            L.append(r"\bomitem{%s\\[1pt]\footnotesize %s} & %s & %s \\" % (
                tex(r["option"]), tex(r["spec"]), money(v, cents=True), delta))
    L += [r"\bottomrule", r"\end{tabular}", r"}"]
    (GEN / "bom_buildbuy.tex").write_text("\n".join(L) + "\n")


def emit_open_table(open_items):
    L = [r"{\small", r"\begin{tabular}{@{}l p{0.50\linewidth} r@{}}", r"\toprule",
         r"\textbf{Kind} & \textbf{Item} & \textbf{Estimate (\$)} \\", r"\midrule"]
    lo = hi = 0.0
    for o in open_items:
        q = float(o["qty"])
        l, h = q * float(o["low_usd"]), q * float(o["high_usd"])
        if o["kind"] == "gap":
            lo += l
            hi += h
        est = "---" if h == 0 else (money(l) if l == h else f"{money(l)}--{money(h)}")
        L.append(r"\footnotesize %s & \bomitem{\textbf{%s}\\[2pt]\footnotesize %s} & %s \\" % (
            tex(o["kind"]), tex(o["item"]), tex(o["note"]), est))
    L += [r"\midrule",
          r" & \textbf{Uncosted hardware, estimated} & \textbf{%s--%s} \\" % (
              money(lo), money(hi)),
          r"\bottomrule", r"\end{tabular}", r"}"]
    (GEN / "bom_open.tex").write_text("\n".join(L) + "\n")
    return lo, hi


def emit_taobao_table(rows):
    L = [r"{\small", r"\begin{tabular}{@{}p{0.40\linewidth} r r r l@{}}", r"\toprule",
         r"\textbf{Model} & \textbf{List (\textyen)} & \textbf{Paid (\textyen)} & "
         r"\textbf{USD} & \textbf{Lines} \\", r"\midrule"]
    for r in rows:
        paid = float(r["paid_cny"])
        lst = r["list_cny"].strip()
        L.append(r"\bomitem{\textbf{%s}\\[1pt]\footnotesize %s} & %s & %s & %s & "
                 r"\footnotesize %s \\" % (
                     tex(r["model"]), tex(r["spec"]),
                     (money(float(lst), cents=True) if lst else "---"),
                     money(paid, cents=True), money(paid * CNY_TO_USD, cents=True),
                     tex(r["refs"])))
    L += [r"\bottomrule", r"\end{tabular}", r"}"]
    (GEN / "bom_taobao.tex").write_text("\n".join(L) + "\n")


def write_numbers(defs):
    L = [r"% AUTO-GENERATED by BOM/tools/build_bom.py"]
    for k, v in defs.items():
        L.append(r"\newcommand{\%s}{%s}" % (k, v))
    (GEN / "bom_numbers.tex").write_text("\n".join(L) + "\n")


# ── publish ─────────────────────────────────────────────────────────────────
def emit_website(summary):
    out = REPO / "web_gui" / "shared_assets" / "bom"
    if not out.parent.is_dir():
        print("! website/assets not found -- skipping website publish")
        return
    out.mkdir(parents=True, exist_ok=True)
    (out / "bom_data.js").write_text(
        "/* AUTO-GENERATED by BOM/tools/build_bom.py -- do not edit. */\n"
        "window.MABEL_BOM = " + json.dumps(summary, indent=1) + ";\n")
    for name in ("core.csv", "choices.csv", "machined.csv", "taobao.csv"):
        shutil.copyfile(DATA / name, out / name)
    for src, dst in ((ROOT / "mabel_bom.pdf", "mabel_bom.pdf"),
                     (ROOT / "MABEL_Build_Guide.pdf", "MABEL_Build_Guide.pdf")):
        if src.exists():
            shutil.copyfile(src, out / dst)
    print(f"website  -> {out}")


# ── main ────────────────────────────────────────────────────────────────────
def main():
    for d in (FIGS, GEN):
        d.mkdir(parents=True, exist_ok=True)

    core = load_core()
    choices = load_choices()
    secs = core_sections(core)
    core_total = round(sum(secs.values()), 2)
    groups = choice_groups(choices)
    builds = build_totals(core_total, groups)

    machined = read_csv("machined.csv")
    taobao = read_csv("taobao.csv")
    bb = read_csv("build_or_buy.csv")
    open_items = read_csv("open_items.csv")
    peers = read_csv("peer_platforms.csv")

    rec_total = builds["recommended"]["total"]
    max_total = builds["maximum"]["total"]
    split = functional_split(core + picked_rows(groups, "recommended"))

    max_rows = core + picked_rows(groups, "maximum")
    top10 = sorted(max_rows, key=lambda r: -r["ext_usd"])[:10]
    top10_sum = round(sum(r["ext_usd"] for r in top10), 2)

    # The headline span covers the sensing/compute choices; the cabling group is
    # a consequence of them, so it is reported separately. Match on what the
    # options ARE, so a renumbered guide edition cannot silently break this.
    def gkind(g):
        return g["options"][0]["fgroup"]
    five = [g for g in groups.values() if gkind(g) != "data"]
    perception_ids = [g["id"] for g in groups.values() if gkind(g) == "perception"]
    choices_lo = round(sum(min(o["ext_usd"] for o in g["options"]) for g in five), 2)
    choices_hi = round(sum(max(o["ext_usd"] for o in g["options"]) for g in five), 2)

    # Filament is carried at Bambu's bulk price (25% off four rolls or more; the
    # build takes seven). The list price is reported for the reader who buys one roll.
    fil_bulk = round(sum(r["ext_usd"] for r in core if r["section"] == BULK["applies_to"]), 2)
    fil = round(fil_bulk / (1 - BULK["discount"]), 2)
    core_bulk = round(core_total - fil_bulk + fil, 2)   # the core at LIST filament

    # Sensing is four independent decisions; each can be taken on its own.
    sensing = []
    for g in groups.values():
        if g["id"] not in perception_ids:
            continue
        costs = [o["ext_usd"] for o in g["options"]]
        sensing.append({"id": g["id"], "name": g["name"],
                        "saving": round(max(costs) - min(costs), 2)})

    fig_builds(builds, core_total)
    fig_choices(groups, builds)
    fig_functional(split, rec_total, "Recommended")
    fig_volume(machined)
    fig_buildbuy(bb)
    fig_peers(peers, rec_total / 1000.0)

    emit_core_table(core, secs, core_total)
    emit_choices_table(groups)
    emit_builds_table(builds, groups, core_total)
    emit_functional_table(split, rec_total)
    mach_tiers = emit_machined_table(machined)
    emit_buildbuy_table(bb)
    gaps_lo, gaps_hi = emit_open_table(open_items)
    emit_taobao_table(taobao)

    arms_mabel = float(next(r for r in bb if r["category"] == "Arms" and r["is_mabel"] == "yes")["both_usd"])
    arms_oa = float(next(r for r in bb if "OpenArm 2.0 assembled" in r["option"])["both_usd"])
    arms_piper = float(next(r for r in bb if "PiPER" in r["option"])["both_usd"])
    hands_mabel = float(next(r for r in bb if r["category"] == "Hands" and r["is_mabel"] == "yes")["both_usd"])
    arm_act = round(sum(r["ext_usd"] for r in core
                        if r["section"] == "Arms - both" and r["fgroup"] == "actuation"), 2)
    hand_act = round(sum(r["ext_usd"] for r in core
                         if r["section"] == "Hands - both" and r["fgroup"] == "actuation"), 2)
    ess_perc = sum(p["usd"] for p in builds["essential"]["picks"]
                   if p["choice_id"] in perception_ids)
    rec_perc = sum(p["usd"] for p in builds["recommended"]["picks"]
                   if p["choice_id"] in perception_ids)
    max_perc = sum(p["usd"] for p in builds["maximum"]["picks"]
                   if p["choice_id"] in perception_ids)

    write_numbers({
        "bomFX": f"{CNY_TO_USD:.4f}", "bomFXinv": f"{1 / CNY_TO_USD:.4f}",
        "bomDate": PRICE_DATE,
        "bomCore": money(core_total), "bomCoreCents": money(core_total, cents=True),
        "bomEssential": money(builds["essential"]["total"]),
        "bomRecommended": money(builds["recommended"]["total"]),
        "bomMaximum": money(builds["maximum"]["total"]),
        "bomCoreShare": f"{core_total / rec_total * 100:.0f}",
        "bomChoicesLo": money(choices_lo, cents=True),
        "bomChoicesHi": money(choices_hi, cents=True),
        "bomChoicesRatio": f"{choices_hi / choices_lo:.1f}",
        "bomEssSaves": money(max_total - builds["essential"]["total"]),
        "bomEssSavesPct": f"{(max_total - builds['essential']['total']) / max_total * 100:.0f}",
        "bomRecSaves": money(max_total - builds["recommended"]["total"]),
        "bomRecSavesPct": f"{(max_total - builds['recommended']['total']) / max_total * 100:.0f}",
        "bomLines": str(len(core)),
        "bomDof": str(FUNCTIONAL_DOF), "bomMotors": str(MOTOR_COUNT),
        "bomPerDof": money(rec_total / FUNCTIONAL_DOF),
        "bomThorBefore": money(JETSON_RISE["thor_before"]),
        "bomThorAfter": money(JETSON_RISE["thor_after"]),
        "bomNanoBefore": money(JETSON_RISE["nano_before"]),
        "bomNanoAfter": money(JETSON_RISE["nano_after"]),
        "bomRiseDate": JETSON_RISE["date"],
        "bomAsQuoted": money(AS_QUOTED_TOTAL),
        "bomAsQuotedToday": money(AS_QUOTED_TOTAL - JETSON_RISE["thor_as_quoted"]
                                  + JETSON_RISE["thor_after"]),
        "bomActuation": money(split.get("actuation", 0)),
        "bomActuationPct": f"{split.get('actuation', 0) / rec_total * 100:.1f}",
        "bomComputePct": f"{split.get('compute', 0) / rec_total * 100:.1f}",
        "bomTopTen": money(top10_sum),
        "bomTopTenPct": f"{top10_sum / max_total * 100:.1f}",
        "bomRest": money(max_total - top10_sum),
        "bomRestCount": str(len(max_rows) - 10),
        "bomMachinedOne": money(mach_tiers[0], cents=True),
        "bomMachinedTen": money(mach_tiers[2], cents=True),
        "bomMachinedDrop": f"{(1 - mach_tiers[2] / mach_tiers[0]) * 100:.0f}",
        "bomArmsMabel": money(arms_mabel, cents=True),
        "bomArmsOpenArm": money(arms_oa, cents=True),
        "bomArmsSavePct": f"{(1 - arms_mabel / arms_oa) * 100:.1f}",
        "bomArmsPiperPct": f"{(1 - arms_mabel / arms_piper) * 100:.1f}",
        "bomHandsMabel": money(hands_mabel, cents=True),
        "bomArmActEach": money(arm_act / 14, cents=True),
        "bomHandActEach": money(hand_act / 34, cents=True),
        "bomHandActPct": f"{hand_act / rec_total * 100:.1f}",
        "bomHandActPctMax": f"{hand_act / max_total * 100:.1f}",
        "bomArmHandAct": money(arm_act + hand_act, cents=True),
        "bomGapsLo": money(gaps_lo), "bomGapsHi": money(gaps_hi),
        "bomRecWithGapsLo": money(rec_total + gaps_lo),
        "bomRecWithGapsHi": money(rec_total + gaps_hi),
        "bomFilament": money(fil, cents=True),
        "bomFilamentBulk": money(fil_bulk, cents=True),
        "bomCoreBulk": money(core_bulk),
        "bomBulkPct": f"{BULK['discount']*100:.0f}",
        "bomBulkMin": str(BULK["min_qty"]),
        "bomPercCutRec": f"{(1 - rec_perc / max_perc) * 100:.0f}",
        "bomPercCutEss": f"{(1 - ess_perc / max_perc) * 100:.0f}",
        "bomSaveWrist": money(_saving(sensing, "Wrist"), cents=True),
        "bomSaveHead": money(_saving(sensing, "Head"), cents=True),
        "bomSaveBase": money(_saving(sensing, "Base depth"), cents=True),
        "bomSaveLidar": money(_saving(sensing, "lidar"), cents=True),
        "bomEssPerception": money(ess_perc, cents=True),
        "bomRecPerception": money(rec_perc, cents=True),
        "bomMaxPerception": money(max_perc, cents=True),
    })

    summary = {
        "generated_by": "BOM/tools/build_bom.py",
        "price_date": PRICE_DATE,
        "cny_to_usd": round(CNY_TO_USD, 7), "usd_to_cny": round(1 / CNY_TO_USD, 4),
        "jetson_rise": JETSON_RISE,
        "as_quoted_total": AS_QUOTED_TOTAL,
        "as_quoted_today": round(AS_QUOTED_TOTAL - JETSON_RISE["thor_as_quoted"]
                                 + JETSON_RISE["thor_after"], 2),
        "functional_dof": FUNCTIONAL_DOF, "motor_count": MOTOR_COUNT,
        "core_total": core_total, "core_line_count": len(core),
        "core_sections": [{"name": k, "usd": v, "share": round(v / core_total * 100, 2)}
                          for k, v in secs.items()],
        "core": [{"ref": r["ref"], "section": r["section"], "item": r["item"],
                  "spec": r["spec"], "qty": int(r["qty"]),
                  "unit_usd": round(r["unit_usd"], 2), "ext_usd": r["ext_usd"],
                  "currency": r["currency"], "unit_native": r["unit_price"],
                  "vendor": r["vendor"], "link": r["link"], "fgroup": r["fgroup"],
                  "status": r["status"], "note": r["note"]} for r in core],
        "choices": [{"id": g["id"], "name": g["name"], "qty": g["qty"],
                     "lo": min(o["ext_usd"] for o in g["options"]),
                     "hi": max(o["ext_usd"] for o in g["options"]),
                     "options": [{"option": o["option"], "spec": o["spec"],
                                  "unit_usd": round(o["unit_usd"], 2),
                                  "ext_usd": o["ext_usd"], "tiers": o["tiers"],
                                  "vendor": o["vendor"], "link": o["link"],
                                  "fgroup": o["fgroup"], "note": o["note"]}
                                 for o in g["options"]]} for g in groups.values()],
        "choices_range": {"lo": choices_lo, "hi": choices_hi,
                          "ratio": round(choices_hi / choices_lo, 1)},
        "builds": [{"id": b, "name": BUILD_LABEL[b], "blurb": BUILD_BLURB[b],
                    "core_usd": core_total, "choices_usd": builds[b]["choices_usd"],
                    "total": builds[b]["total"], "picks": builds[b]["picks"]}
                   for b in BUILDS],
        "functional": [{"key": k, "name": FGROUPS[k], "usd": v,
                        "share": round(v / rec_total * 100, 2)}
                       for k, v in sorted(split.items(), key=lambda kv: -kv[1])],
        "functional_of": "recommended",
        "top_ten": [{"item": r.get("item") or r.get("option"), "qty": int(r["qty"]),
                     "ext_usd": r["ext_usd"],
                     "share": round(r["ext_usd"] / max_total * 100, 1)} for r in top10],
        "top_ten_sum": top10_sum,
        "top_ten_share": round(top10_sum / max_total * 100, 1),
        "top_ten_rest": round(max_total - top10_sum, 2),
        "top_ten_rest_count": len(max_rows) - 10,
        "machined": [{"ref": m["ref"], "part": m["part"], "operations": m["operations"],
                      "qty": int(float(m["qty"])),
                      "tiers": {t: float(m[t]) for t in
                                ("p1", "p2", "p10", "p50", "p100", "p1000")}}
                     for m in machined],
        "machined_set": {t: round(v, 2) for t, v in
                         zip(("p1", "p2", "p10", "p50", "p100", "p1000"), mach_tiers)},
        "taobao": [{"model": r["model"], "search": r["search_string"], "spec": r["spec"],
                    "list_cny": (float(r["list_cny"]) if r["list_cny"].strip() else None),
                    "paid_cny": float(r["paid_cny"]),
                    "usd": round(float(r["paid_cny"]) * CNY_TO_USD, 2),
                    "refs": r["refs"]} for r in taobao],
        "build_or_buy": [{"category": r["category"], "option": r["option"],
                          "spec": r["spec"], "both_usd": float(r["both_usd"]),
                          "per_unit_usd": float(r["per_unit_usd"]),
                          "is_mabel": r["is_mabel"] == "yes", "source": r["source"]}
                         for r in bb],
        "open_items": [{"kind": o["kind"], "item": o["item"], "qty": int(float(o["qty"])),
                        "low_usd": float(o["low_usd"]), "high_usd": float(o["high_usd"]),
                        "impact": o["impact"], "note": o["note"]} for o in open_items],
        "uncosted_gap_usd": {"low": round(gaps_lo, 2), "high": round(gaps_hi, 2)},
        "hand_actuator_usd": hand_act,
        "hand_share_recommended": round(hand_act / rec_total * 100, 1),
        "hand_share_maximum": round(hand_act / max_total * 100, 1),
        "bulk_offer": {**BULK, "filament_list": fil, "filament_bulk": fil_bulk,
                       "core_at_bulk": core_bulk},
        "sensing_savings": sensing,
        "perception_cut": {"recommended": round((1 - rec_perc / max_perc) * 100),
                           "essential": round((1 - ess_perc / max_perc) * 100)},
        "perception_by_build": {"essential": round(ess_perc, 2),
                                "recommended": round(rec_perc, 2),
                                "maximum": round(max_perc, 2)},
        "peers": [{"platform": p["platform"], "cost_kusd": float(p["cost_kusd"]),
                   "open": p["open"] == "yes", "source": p["source"]} for p in peers],
    }
    (GEN / "bom_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    emit_website(summary)

    # ── stdout report ───────────────────────────────────────────────────────
    w = 68
    print("=" * w)
    print(f"MABEL BOM — {len(core)} core lines · {PRICE_DATE} · "
          f"1 USD = {1 / CNY_TO_USD:.4f} CNY")
    print("=" * w)
    for k, v in secs.items():
        print(f"  {k:<32s} ${v:>9,.2f}")
    print("-" * w)
    print(f"  {'FIXED CORE':<32s} ${core_total:>9,.2f}  "
          f"({core_total / rec_total * 100:.0f}% of Recommended)")
    print("-" * w)
    for b in BUILDS:
        print(f"  {BUILD_LABEL[b]:<32s} ${builds[b]['total']:>9,.2f}")
    print("-" * w)
    print(f"  choices span                     ${choices_lo:>9,.2f} – ${choices_hi:,.2f}"
          f"  ({choices_hi / choices_lo:.1f}x)")
    print(f"  uncosted gaps (est.)             ${gaps_lo:>9,.2f} – ${gaps_hi:,.2f}")
    print(f"  top 10 lines of Maximum          ${top10_sum:>9,.2f}  "
          f"({top10_sum / max_total * 100:.1f}% of ${max_total:,.2f})")
    print(f"  cost per functional DOF          ${rec_total / FUNCTIONAL_DOF:>9,.2f}  "
          f"(Recommended / {FUNCTIONAL_DOF} DOF)")
    print("=" * w)
    print(f"figures  -> {FIGS}")
    print(f"tables   -> {GEN}")


if __name__ == "__main__":
    main()
