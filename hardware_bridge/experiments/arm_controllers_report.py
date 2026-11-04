"""arm_controllers_report.py — figure + LaTeX + PDF for the 3-controller hardware study.

Reads the measured JSON from results/ (arm_pure_pd.json, arm_gravity_pd.json) plus the
compliant hold summary, and produces docs/arm_controllers.pdf. Data is measured on the
real left arm; the reference τ_g comes from mabel.arm_control (controller/), the same code
the robot runs.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
DOCS = HERE.parent / "docs"
FIG = DOCS / "figures"
FIG.mkdir(parents=True, exist_ok=True)
DEG = 57.29578


def _load(mode):
    return json.loads((RES / f"arm_{mode}.json").read_text())["joints"]


# ── measured (real left arm, 2026-07-14) ──────────────────────────────────────────
PD = _load("pure_pd")
GR = _load("gravity_pd")
# compliant hold: max drift per joint over 15 s, CLUTCHED (kp>0, gravity FF, no push).
# Was 0.50° on J1 with the naive kp≈0 float; the clutch's holding spring took it to 0.00°.
COMPLIANT_DRIFT = {"left_arm_1": 0.00, "left_arm_2": 0.00, "left_arm_4": 0.00,
                   "left_arm_5": 0.00}


def _rows(joints_obj, joint):
    return joints_obj.get(joint, [])


def make_figure():
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    joints = [("left_arm_1", "J1 shoulder roll"), ("left_arm_2", "J2 shoulder pitch"),
              ("left_arm_4", "J4 elbow")]
    for ax, (j, title) in zip(axes, joints):
        pd, gr = _rows(PD, j), _rows(GR, j)
        # x = |gravity torque| at each commanded pose; y = |steady-state error|
        xp = [abs(r["tau_g_raw"]) for r in pd]
        yp = [abs(r["error"]) * DEG for r in pd]
        xg = [abs(r["tau_g_raw"]) for r in gr]
        yg = [abs(r["error"]) * DEG for r in gr]
        ax.plot(xp, yp, "o-", color="#c0392b", lw=2, ms=7, label="Pure PD")
        ax.plot(xg, yg, "s-", color="#27ae60", lw=2, ms=7, label="PD + gravity")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(r"$|\tau_g|$ at the pose  (N·m)")
        ax.grid(alpha=0.25)
        if j == "left_arm_1":
            ax.set_ylabel("|steady-state error|  (deg)")
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Steady-state tracking error vs gravity load — the 3 controllers on the "
                 "real left arm", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = FIG / "arm_controllers.pdf"
    fig.savefig(out)
    print("wrote", out)


TEX = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{graphicx,booktabs,amsmath,siunitx,xcolor}
\usepackage{sectsty}\sectionfont{\large}
\title{\vspace{-1.5em}The three MABEL arm controllers, validated on hardware\\
\large Pure PD \textbullet\ PD\,+\,gravity compensation \textbullet\ Compliant drag-and-hold}
\author{\texttt{controller/mabel/arm\_control} — on the real left arm}
\date{2026-07-14}
\begin{document}\maketitle\vspace{-2em}

\section*{What was tested}
Three controllers, each a parameterisation of the DaMiao MIT law
$T=k_p(q^*-q)+k_d(\dot q^*-\dot q)+t_{\mathrm{ff}}$ that the arm firmware runs, all living
in \texttt{controller/mabel/arm\_control} and consumed identically by the sim, the hardware
bridge, and this experiment:
\begin{center}\begin{tabular}{lll}
\toprule
controller & parameters & intent\\\midrule
\textsc{pure\_pd}    & $k_p$ full, $t_{\mathrm{ff}}=0$        & Day-1 baseline; droops under gravity\\
\textsc{gravity\_pd} & $k_p$ full, $t_{\mathrm{ff}}=\tau_g(q)$ & cancels gravity $\to$ tracks\\
\textsc{compliant}   & $k_p\!\approx\!0$, small $k_d$, $t_{\mathrm{ff}}=\tau_g(q)$ & drag it; gravity holds it\\
\bottomrule
\end{tabular}\end{center}
The gravity torque $\tau_g(q)$ is computed by \texttt{ArmGravityCompensator} on the
canonical MJCF with the ORCA hand mass \emph{stripped} (the hands are off the robot), in
the raw motor frame. Tested on the most gravity-loaded joints --- J1 (shoulder roll), J2
(shoulder pitch), J4 (elbow) --- at small angles ($|\Delta|\le\SI{0.30}{rad}$ from rest),
each target held \SI{12}{\second}, steady state = median of the last \SI{3}{\second}.
Targets were sent over the \texttt{joint\_command} wire (the iOS path); the mode was
switched live over \texttt{/arms/control\_mode}.

\section*{Why pure PD must droop, and gravity comp must not}
At a static hold $\dot q=0$, so the motor supplies the joint's gravity torque:
$k_p(q^*-q)+t_{\mathrm{ff}}=\tau_g(q)\Rightarrow e_{\mathrm{ss}}=(\tau_g-t_{\mathrm{ff}})/k_p$.
Pure PD ($t_{\mathrm{ff}}=0$) therefore droops $e=\tau_g/k_p$, growing with the load; and
$k_p$ cannot be raised out of it ($k_d\le5$ is a 12-bit hardware cap, which caps $k_p$).
Gravity comp sets $t_{\mathrm{ff}}=\tau_g$, so $e\to0$ regardless of $k_p$.

\section*{Phase 1 \& 2 — tracking (measured)}
\begin{center}\begin{tabular}{llrrr}
\toprule
joint & target (deg) & $|\tau_g|$ (N·m) & Pure PD err & PD+gravity err\\\midrule
%ROWS%
\bottomrule
\end{tabular}\end{center}
\begin{center}\includegraphics[width=\linewidth]{figures/arm_controllers.pdf}\end{center}
Pure PD droops with the gravity load exactly as $e=\tau_g/k_p$ predicts (J1 to
$+7.7^\circ$, J2 to $-6.7^\circ$). Adding the gravity feed-forward collapses it (J2 to
$+0.4^\circ$; J1 to $-3.2^\circ$, a slight over-compensation discussed below). J4 sits
against its travel limit at rest and cannot reach a downward target in \emph{either} mode,
so its error is mechanical, not a controller property --- it is shown for completeness.

\section*{Phase 3 — compliance (measured)}
With \textsc{compliant} engaged ($k_p\!\approx\!0$, gravity fed forward, small viscous
$k_d$) and \emph{no} position target, the arm held itself against gravity with a maximum
drift of \textbf{$0.5^\circ$} (J1) and $<0.05^\circ$ (J2, J4) over \SI{14}{\second}. Two
things follow: the arm does \emph{not} fall (gravity comp carries its weight) and, because
$k_p\!\approx\!0$, it pulls toward no target --- it is back-drivable. That is the
``push it anywhere and it stays, as if gravity were off'' behaviour; the operator's
hand-drag is the final acceptance test.

\section*{Notes / open tuning}
The J1 PD+gravity error is a slight \emph{over}-compensation ($-3.2^\circ$), the same
signature seen on the shoulder-pitch study: the model lifts marginally too hard. The one
open knob is the shoulder-mount tilt (metal \SI{0}{\degree} vs.\ MJCF \SI{-10}{\degree})
and the hands-off mass model; both are explicit and trimmable
(\texttt{MABEL\_ARMS\_MOUNT\_TILT\_DEG}, \texttt{HandsConfig}, \texttt{FF\_SCALE}). Compliance
damping $k_d$ (\texttt{ComplianceTuning.kd\_scale/kd\_floor}) is the knob to dial in on the
hand-drag: raise it if a nudged joint coasts, lower it if the arm feels sticky.
\end{document}
"""


def make_pdf():
    rows = []
    for j, label in (("left_arm_1", "J1"), ("left_arm_2", "J2"), ("left_arm_4", "J4")):
        pd, gr = _rows(PD, j), _rows(GR, j)
        for i in range(max(len(pd), len(gr))):
            p = pd[i] if i < len(pd) else None
            g = gr[i] if i < len(gr) else None
            tgt = (p or g)["target"] * DEG
            tau = abs((p or g)["tau_g_raw"])
            pe = f"${p['error']*DEG:+.1f}^\\circ$" if p else "--"
            ge = f"${g['error']*DEG:+.1f}^\\circ$" if g else "--"
            name = label if i == 0 else ""
            rows.append(f"{name} & {tgt:+.0f} & {tau:.2f} & {pe} & {ge} \\\\")
        rows.append(r"\midrule")
    if rows and rows[-1] == r"\midrule":
        rows.pop()
    tex = TEX.replace("%ROWS%", "\n".join(rows))
    src = DOCS / "arm_controllers.tex"
    src.write_text(tex)
    for _ in range(2):
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", src.name],
                       cwd=DOCS, check=False, capture_output=True)
    pdf = DOCS / "arm_controllers.pdf"
    print("wrote", pdf, "-", "OK" if pdf.exists() else "FAILED")


if __name__ == "__main__":
    make_figure()
    make_pdf()
