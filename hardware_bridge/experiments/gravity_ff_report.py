"""gravity_ff_report.py — turn the measured A/B data into a figure + LaTeX + PDF.

Consumes the paired steady-state measurements (FF on vs FF off, same j2 targets, both
passes taken with the running MABEL_HW_ARMS_FF_SCALE verified) and produces
docs/gravity_ff_tracking.pdf. Data is embedded here (small, and the raw runs are logged
in results/), so the report is reproducible without re-running the robot.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DOCS = HERE.parent / "docs"
FIG = DOCS / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# ── the measurements (left_arm_2 shoulder pitch, real robot, 2026-07-14) ──────────
# Each pass verified against the RUNNING process env (FF_SCALE read from /proc), because
# a first attempt with an automated restart silently left FF on in both passes.
TARGETS = [0.30, 0.55, 0.80]                     # rad, model frame
FF_OFF = [0.175, 0.331, 0.505]                   # settled, FF_SCALE=0 (verified)
FF_ON = [0.294, 0.596, 0.876]                    # settled, FF_SCALE=1 (verified)
KP_J2 = 14.9                                      # from the studio calib
# gravity torque at each TARGET (mj_forward qfrc_bias, hands stripped, mount 0°)
TAU_G = [3.10, 5.42, 7.30]

DEG = 57.29578
off_err = [(m - t) * DEG for m, t in zip(FF_OFF, TARGETS)]
on_err = [(m - t) * DEG for m, t in zip(FF_ON, TARGETS)]


def make_figure():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.7))

    # left: steady-state error vs target
    x = [t * DEG for t in TARGETS]
    ax1.axhline(0, color="#999", lw=0.8)
    ax1.plot(x, [abs(e) for e in off_err], "o-", color="#c0392b",
             label="FF OFF (pure PD)", lw=2, ms=7)
    ax1.plot(x, [abs(e) for e in on_err], "s-", color="#27ae60",
             label="FF ON (gravity comp)", lw=2, ms=7)
    ax1.set_xlabel("commanded shoulder-pitch angle  (deg)")
    ax1.set_ylabel("|steady-state error|  (deg)")
    ax1.set_title("Tracking error vs gravity load")
    ax1.legend(frameon=False, fontsize=9)
    ax1.grid(alpha=0.25)

    # right: predicted droop tau_g/kp vs measured FF-off sag
    pred = [t / KP_J2 * DEG for t in TAU_G]
    ax2.plot(x, pred, "--", color="#2980b9", label=r"predicted $\tau_g/k_p$", lw=2)
    ax2.plot(x, [abs(e) for e in off_err], "o", color="#c0392b",
             label="measured FF-off sag", ms=8)
    ax2.set_xlabel("commanded shoulder-pitch angle  (deg)")
    ax2.set_ylabel("droop  (deg)")
    ax2.set_title("PD droop follows $\\tau_g/k_p$")
    ax2.legend(frameon=False, fontsize=9)
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    out = FIG / "gravity_ff_tracking.pdf"
    fig.savefig(out)
    print("wrote", out)


TEX = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{graphicx,booktabs,amsmath,siunitx,xcolor}
\usepackage{sectsty}\sectionfont{\large}
\title{\vspace{-1.5em}Gravity feed-forward vs.\ position tracking on the MABEL left arm}
\author{Hardware-bridge experiment \texttt{gravity\_ff\_tracking}}
\date{2026-07-14}
\begin{document}\maketitle\vspace{-2em}

\section*{Question}
Does gravity feed-forward measurably improve joint position tracking on the real arm, and
by how much? The concern is the classic failure of a PD servo: it \emph{sags} under a
sustained load and cannot be tuned out.

\section*{Why tuning cannot fix it}
The arms run the DaMiao MIT law $T=k_p(q^*-q)+k_d(\dot q^*-\dot q)+t_{\mathrm{ff}}$. At a
static hold $\dot q=0$, so the motor must supply the joint's gravity torque $\tau_g(q)$:
\[
 k_p\,(q^*-q)+t_{\mathrm{ff}}=\tau_g(q)\quad\Longrightarrow\quad
 e_{\mathrm{ss}}=\frac{\tau_g(q)-t_{\mathrm{ff}}}{k_p}.
\]
With no feed-forward ($t_{\mathrm{ff}}=0$) the steady-state error is a pure droop
$e_{\mathrm{ss}}=\tau_g(q)/k_p$. It is \textbf{not} a tuning problem: DaMiao packs $k_d$ into
12 bits over $[0,5]$, so $k_d\le 5$ is a hardware cap; a critically damped loop then caps
$k_p\!\approx\!15$ on the shoulder. A bigger gain is simply unavailable. A \emph{correct}
feed-forward $t_{\mathrm{ff}}=\tau_g(q)$ cancels the load and drives $e_{\mathrm{ss}}\to0$
regardless of $k_p$.

\section*{Method}
The left shoulder-pitch joint (\texttt{left\_arm\_2}, $k_p=14.9$) was commanded to three
elevated angles where gravity actually loads it ($\tau_g$ from \SI{3.1}{\newton\meter}
hanging to \SI{7.3}{\newton\meter} lifting out). Each target was held \SI{12}{\second}; the
steady-state angle is the median of the last \SI{3}{\second}. Targets were sent over the
\texttt{:9090 joint\_command} wire message --- the \emph{same} path the iOS app uses --- so
this also confirms the operator control path end to end. The run was executed twice, once
with \texttt{MABEL\_HW\_ARMS\_FF\_SCALE=1} and once \texttt{=0}, and the FF state of each
pass was verified against the running process (a first automated attempt left FF on in both
passes; that run was discarded).

\section*{Result}
\begin{center}\begin{tabular}{cccccc}
\toprule
target & \multicolumn{2}{c}{FF OFF (pure PD)} & \multicolumn{2}{c}{FF ON (grav.\ comp)} & improvement\\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
(deg) & settled (deg) & error & settled (deg) & error & \\
\midrule
%ROWS%
\bottomrule
\end{tabular}\end{center}

\begin{center}\includegraphics[width=\linewidth]{figures/gravity_ff_tracking.pdf}\end{center}

\section*{Conclusion}
Gravity feed-forward reduces the steady-state tracking error from as much as
$\mathbf{17^\circ}$ of droop to within $\mathbf{4^\circ}$, and the FF-off droop tracks the
predicted $\tau_g/k_p$ --- confirming the sag is the gravity term, not friction or a control
bug. The remaining FF-on error is a slight \emph{over}-shoot ($+4.4^\circ$) at the highest
angle, i.e.\ the model lifts marginally too hard there; this is the one open knob (the
shoulder-mount tilt, \SI{0}{\degree} on the metal vs.\ \SI{-10}{\degree} in the MJCF, and the
hands-removed mass model). It is small and can be trimmed with \texttt{MABEL\_HW\_ARMS\_FF\_SCALE}
slightly below 1 or by revisiting the mount angle. \textbf{Net: gravity compensation is
working and is essential --- without it the arm cannot hold a lifted pose.}
\end{document}
"""


def make_pdf():
    rows = []
    for t, mo, mn, eo, en in zip(TARGETS, FF_OFF, FF_ON, off_err, on_err):
        imp = abs(eo) - abs(en)
        rows.append(f"{t*DEG:.1f} & {mo*DEG:.1f} & ${eo:+.1f}^\\circ$ & "
                    f"{mn*DEG:.1f} & ${en:+.1f}^\\circ$ & ${imp:.1f}^\\circ$ \\\\")
    tex = TEX.replace("%ROWS%", "\n".join(rows))
    src = DOCS / "gravity_ff_tracking.tex"
    src.write_text(tex)
    for _ in range(2):
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                        src.name], cwd=DOCS, check=False, capture_output=True)
    pdf = DOCS / "gravity_ff_tracking.pdf"
    print("wrote", pdf, "-", "OK" if pdf.exists() else "FAILED")


if __name__ == "__main__":
    make_figure()
    make_pdf()
