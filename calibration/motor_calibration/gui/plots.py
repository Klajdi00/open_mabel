"""
plots.py
========

Dark-themed, Apple-styled matplotlib figures for the calibration GUI, embedded
live via ``FigureCanvasTkAgg``.  These render the same quantities as
``mabel_actid.metrics`` / ``hardware/plot_data.py`` but on the iOS-dark palette so
they sit cleanly inside the app.

The *saved* figures under ``results/<motor>/plots/`` stay on the default light
background — that is what the LaTeX docs / paper want — so this module only
affects what is shown on screen.

We build ``matplotlib.figure.Figure`` objects directly (not ``pyplot``) so each
canvas is independent of any global state and safe to create/destroy in Tk.
"""
from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure

from mabel_actid.friction_model import friction_torque, torque_speed_envelope
from mabel_actid.identify import differentiate, fit_friction_curve, fit_inertia

# ── palette (mirrors gui/style.py) ────────────────────────────────────────────
BG      = "#1C1C1E"
PANEL   = "#121214"
FG      = "#FFFFFF"
FG2     = "#8E8E93"
GRID    = "#38383A"
ORANGE  = "#FF9F0A"
BLUE    = "#0A84FF"
GREEN   = "#30D158"
RED     = "#FF453A"
PURPLE  = "#BF5AF2"


def _new_fig(nrows=1, **kw):
    fig = Figure(facecolor=PANEL, **kw)
    axes = fig.subplots(nrows, 1, squeeze=False, **{})[:, 0] if nrows > 1 else \
        [fig.add_subplot(111)]
    return fig, list(axes)


def _style(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors=FG2, labelsize=8)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.grid(True, color=GRID, alpha=0.5, lw=0.6)
    ax.title.set_color(FG)
    ax.xaxis.label.set_color(FG2)
    ax.yaxis.label.set_color(FG2)


def _legend(ax):
    leg = ax.legend(fontsize=7, framealpha=0.0)
    if leg:
        for t in leg.get_texts():
            t.set_color(FG2)


def placeholder(message="Run a calibration to see plots"):
    fig = Figure(facecolor=PANEL)
    ax = fig.add_subplot(111)
    ax.set_facecolor(BG)
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center",
            color=FG2, fontsize=12, transform=ax.transAxes)
    return fig


def friction_curve(w_lvl, tau_lvl, params, title="Friction curve"):
    fig, (ax,) = _new_fig()
    _style(ax)
    if len(w_lvl):
        order = np.argsort(w_lvl)
        w, tau = np.asarray(w_lvl)[order], np.asarray(tau_lvl)[order]
        grid = np.linspace(min(w.min(), -1e-3), max(w.max(), 1e-3), 400)
        ax.axhline(0, color=GRID, lw=0.8); ax.axvline(0, color=GRID, lw=0.8)
        ax.plot(w, tau, "o", ms=5, color=BLUE, label="measured (constant-v)")
        if params is not None:
            ax.plot(grid, friction_torque(params, grid), "-", color=ORANGE,
                    lw=2, label="identified model")
    ax.set_xlabel("velocity  ω  [rad/s]")
    ax.set_ylabel("friction τ_f  [N·m]")
    ax.set_title(title)
    _legend(ax)
    fig.tight_layout()
    return fig


def torque_fit(log, params, title="Dynamic torque: measured vs model"):
    fig = Figure(facecolor=PANEL)
    ax0 = fig.add_subplot(2, 1, 1)
    ax1 = fig.add_subplot(2, 1, 2, sharex=ax0)
    a = log.a
    if not np.all(np.isfinite(a)) or np.allclose(a, 0.0):
        a = differentiate(log.t, log.w, window=0.02)
    pred = params.J * a + friction_torque(params, log.w)
    resid = pred - log.tau
    rms = float(np.sqrt(np.mean(resid ** 2)))
    for ax in (ax0, ax1):
        _style(ax)
    ax0.plot(log.t, log.tau, color=BLUE, lw=1.0, label="measured")
    ax0.plot(log.t, pred, color=ORANGE, lw=1.0, alpha=0.85,
             label="model  J·a + τ_f(ω)")
    ax0.set_ylabel("torque [N·m]")
    ax0.set_title(f"{title}   (resid RMS {rms:.4f} N·m)")
    _legend(ax0)
    ax1.plot(log.t, resid, color=FG2, lw=0.8)
    ax1.set_ylabel("resid"); ax1.set_xlabel("time [s]")
    fig.tight_layout()
    return fig


def staircase(log, title="Velocity staircase"):
    fig = Figure(facecolor=PANEL)
    ax0 = fig.add_subplot(2, 1, 1)
    ax1 = fig.add_subplot(2, 1, 2)
    for ax in (ax0, ax1):
        _style(ax)
    # shade steady spans
    s = log.steady.astype(int)
    edges = np.flatnonzero(np.diff(s) != 0) + 1
    for span in np.split(np.arange(log.t.size), edges):
        if span.size and log.steady[span[0]]:
            ax0.axvspan(log.t[span[0]], log.t[span[-1]], color=GREEN, alpha=0.10)
    ax0.plot(log.t, log.w_cmd, color=ORANGE, lw=1.3, label="commanded ω_cmd")
    ax0.plot(log.t, log.w, color=BLUE, lw=0.8, label="readback ω")
    ax0.axhline(0, color=GRID, lw=0.8)
    ax0.set_ylabel("velocity [rad/s]"); ax0.set_title(title)
    _legend(ax0)

    w_lvl, tau_lvl = log.steady_points()
    ax1.axhline(0, color=GRID, lw=0.8); ax1.axvline(0, color=GRID, lw=0.8)
    if len(w_lvl):
        order = np.argsort(w_lvl)
        ax1.plot(np.asarray(w_lvl)[order], np.asarray(tau_lvl)[order], "o-",
                 ms=5, color=RED, label="steady points")
    else:
        ax1.text(0.5, 0.5, "no steady points\n(motor stalled?)", ha="center",
                 va="center", color=RED, transform=ax1.transAxes)
    ax1.set_xlabel("velocity ω [rad/s]"); ax1.set_ylabel("τ_f [N·m]")
    _legend(ax1)
    fig.tight_layout()
    return fig


def chirp(log, title="Frequency chirp"):
    fig = Figure(facecolor=PANEL)
    ax0 = fig.add_subplot(2, 1, 1)
    ax1 = fig.add_subplot(2, 1, 2, sharex=ax0)
    for ax in (ax0, ax1):
        _style(ax)
    a = log.a
    if not np.all(np.isfinite(a)):
        a = np.gradient(log.w, log.t)
    ax0.plot(log.t, log.q, color=BLUE, lw=1.0, label="q [rad]")
    ax0.plot(log.t, log.w, color=GREEN, lw=0.8, label="ω [rad/s]")
    ax0.plot(log.t, a, color=PURPLE, lw=0.6, alpha=0.7, label="a [rad/s²]")
    ax0.axhline(0, color=GRID, lw=0.8)
    ax0.set_ylabel("kinematics"); ax0.set_title(title)
    _legend(ax0)
    ax1.plot(log.t, log.tau, color=RED, lw=0.9, label="reported torque")
    ax1.axhline(0, color=GRID, lw=0.8)
    ax1.set_ylabel("torque [N·m]"); ax1.set_xlabel("time [s]")
    _legend(ax1)
    fig.tight_layout()
    return fig


def torque_speed(log, dc, title="Torque-speed envelope"):
    fig, (ax,) = _new_fig()
    _style(ax)
    ax.axhline(0, color=GRID, lw=0.8); ax.axvline(0, color=GRID, lw=0.8)
    ax.plot(log.w, log.tau, ".", ms=3, color=BLUE, alpha=0.5, label="measured sweep")
    if dc is not None and dc.w_noload > 0:
        grid = np.linspace(-dc.w_noload, dc.w_noload, 400)
        if dc.observed:
            ax.plot(grid, torque_speed_envelope(dc, grid), "-", color=ORANGE, lw=2,
                    label=f"fit: stall {dc.tau_stall:.1f}, no-load {dc.w_noload:.1f}")
        else:
            ax.axvline(dc.w_noload, color=GREEN, ls="--", lw=1.5,
                       label=f"no-load ≈ {dc.w_noload:.1f} rad/s (measured)")
            ax.axvline(-dc.w_noload, color=GREEN, ls="--", lw=1.5)
    ax.set_xlabel("velocity ω [rad/s]"); ax.set_ylabel("torque τ [N·m]")
    ax.set_title(title)
    _legend(ax)
    fig.tight_layout()
    return fig


RAD_TO_RPM = 60.0 / (2.0 * np.pi)


def _bin_curve(x, y, nbins=18):
    """Median ``y`` per ``x`` bin, sorted — a clean line through a scatter."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    if x.size < 4:
        return x, y
    edges = np.linspace(x.min(), x.max(), nbins + 1)
    bx, by = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi)
        if m.sum() >= 2:
            bx.append(0.5 * (lo + hi)); by.append(float(np.median(y[m])))
    return np.asarray(bx), np.asarray(by)


def performance(log, title="Motor performance curve"):
    """Foxtech-style performance curve from the motor's OWN feedback.

    Plots vs torque |τ|: **Speed (rpm)** and **Output power Pout = |τ·ω| (W)**.
    Vbus / Ibus / efficiency / Pin are intentionally absent — the DaMiao MIT
    feedback carries no bus voltage or current, so those traces would need a
    power meter / dynamometer (not a free-shaft run).
    """
    fig = Figure(facecolor=PANEL)
    ax1 = fig.add_subplot(111); _style(ax1)
    ax2 = ax1.twinx()
    ax2.tick_params(colors=FG2, labelsize=8)
    for sp in ax2.spines.values():
        sp.set_color(GRID)

    tau = np.abs(np.asarray(log.tau)); w = np.abs(np.asarray(log.w))
    rpm = w * RAD_TO_RPM
    pout = tau * w
    ax1.plot(tau, rpm, ".", ms=2, color=BLUE, alpha=0.3)
    ax2.plot(tau, pout, ".", ms=2, color=ORANGE, alpha=0.3)
    bx, by = _bin_curve(tau, rpm); ax1.plot(bx, by, "-", color=BLUE, lw=2, label="Speed (rpm)")
    px, py = _bin_curve(tau, pout); ax2.plot(px, py, "-", color=ORANGE, lw=2, label="Pout (W)")

    ax1.set_xlabel("torque  τ  [N·m]")
    ax1.set_ylabel("Speed [rpm]"); ax1.yaxis.label.set_color(BLUE)
    ax2.set_ylabel("Pout [W]");    ax2.yaxis.label.set_color(ORANGE)
    ax1.set_title(title)
    ax1.text(0.5, 0.04, "Vbus / Ibus / efficiency need current sensing (dyno) — "
             "not in DaMiao MIT feedback", transform=ax1.transAxes, ha="center",
             color=FG2, fontsize=7)
    fig.tight_layout()
    return fig


# ── live (streaming) fitting + plotting ───────────────────────────────────────

def steady_live(w, tau, wcmd, vel_tol=0.25, vel_abs=0.01):
    """Keep only samples whose actual speed has converged to the commanded one.

    Mirrors :meth:`CalibrationLog.steady_points` so the live view shows the same
    constant-velocity points the final fit uses — rejecting the position-ramp
    acceleration transients that otherwise inflate the live Coulomb estimate.
    """
    w = np.asarray(w, float); tau = np.asarray(tau, float)
    wc = np.asarray(wcmd, float)
    if wc.size != w.size or w.size == 0:
        return w, tau
    keep = (np.abs(wc) > vel_abs) & (np.abs(w - wc) <= vel_tol * np.abs(wc) + vel_abs)
    return w[keep], tau[keep]


def live_friction_fit(w, tau, wcmd=None):
    """Quick robust friction fit on the data gathered so far (or None).

    Uses the asymmetric Coulomb+viscous model — cheap and stable enough to
    refit every redraw while the staircase streams in.  Filters to steady
    (constant-velocity) samples first when ``wcmd`` is supplied.  Needs a few
    distinct speed levels before it means anything.
    """
    if wcmd is not None:
        w, tau = steady_live(w, tau, wcmd)
    w = np.asarray(w, float); tau = np.asarray(tau, float)
    if w.size < 8 or np.unique(np.round(w, 2)).size < 3:
        return None
    try:
        return fit_friction_curve(w, tau, model="asym").params
    except Exception:
        return None


def live_inertia(t, w, tau, fparams):
    """Live reflected-inertia estimate J from streaming chirp data (or None)."""
    t = np.asarray(t, float); w = np.asarray(w, float); tau = np.asarray(tau, float)
    if t.size < 30 or fparams is None:
        return None
    try:
        a = np.gradient(w, t)
        return fit_inertia(w, a, tau, fparams).J
    except Exception:
        return None


class LivePlot:
    """A persistent figure whose line artists are updated in place while a motor
    is moving — no figure rebuild per frame, so it stays smooth.

    ``build(tab)`` creates the artists for one plot tab and returns the Figure to
    mount; ``update(buf, params)`` pushes the latest streamed buffers (and an
    optionally live-fitted model) onto those artists.  ``params`` is whatever the
    live fit produced, so the regression line tracks the data as it accumulates.
    """

    def __init__(self):
        self.fig = None
        self.tab = None
        self.art: dict = {}

    def build(self, tab):
        self.tab = tab
        self.art = {}
        if tab in ("Friction",):
            self.fig = Figure(facecolor=PANEL)
            ax = self.fig.add_subplot(111); _style(ax)
            ax.axhline(0, color=GRID, lw=0.8); ax.axvline(0, color=GRID, lw=0.8)
            self.art["pts"], = ax.plot([], [], "o", ms=4, color=BLUE,
                                       label="measured (live)")
            self.art["fit"], = ax.plot([], [], "-", color=ORANGE, lw=2,
                                       label="live fit")
            ax.set_xlabel("velocity ω [rad/s]"); ax.set_ylabel("τ_f [N·m]")
            ax.set_title("Friction curve — live"); _legend(ax)
            self.art["ax"] = ax
        elif tab == "Staircase":
            self.fig = Figure(facecolor=PANEL)
            ax0 = self.fig.add_subplot(2, 1, 1); _style(ax0)
            ax1 = self.fig.add_subplot(2, 1, 2); _style(ax1)
            self.art["wcmd"], = ax0.plot([], [], color=ORANGE, lw=1.3, label="ω_cmd")
            self.art["w"], = ax0.plot([], [], color=BLUE, lw=0.9, label="ω readback")
            ax0.axhline(0, color=GRID, lw=0.8)
            ax0.set_ylabel("velocity [rad/s]"); ax0.set_title("Velocity staircase — live")
            _legend(ax0)
            ax1.axhline(0, color=GRID, lw=0.8); ax1.axvline(0, color=GRID, lw=0.8)
            self.art["pts"], = ax1.plot([], [], "o", ms=4, color=RED, label="measured")
            self.art["fit"], = ax1.plot([], [], "-", color=ORANGE, lw=1.8, label="live fit")
            ax1.set_xlabel("velocity ω [rad/s]"); ax1.set_ylabel("τ_f [N·m]")
            _legend(ax1)
            self.art["ax0"], self.art["ax1"] = ax0, ax1
        elif tab == "Torque–speed":
            self.fig = Figure(facecolor=PANEL)
            ax = self.fig.add_subplot(111); _style(ax)
            ax.axhline(0, color=GRID, lw=0.8); ax.axvline(0, color=GRID, lw=0.8)
            self.art["pts"], = ax.plot([], [], ".", ms=3, color=BLUE, alpha=0.6,
                                       label="sweep (max effort)")
            ax.set_xlabel("velocity ω [rad/s]"); ax.set_ylabel("torque τ [N·m]")
            ax.set_title("Torque–speed — live"); _legend(ax)
            self.art["ax"] = ax
        elif tab == "Performance":
            self.fig = Figure(facecolor=PANEL)
            ax1 = self.fig.add_subplot(111); _style(ax1)
            ax2 = ax1.twinx()
            ax2.tick_params(colors=FG2, labelsize=8)
            for sp in ax2.spines.values():
                sp.set_color(GRID)
            self.art["spd"], = ax1.plot([], [], ".", ms=3, color=BLUE, alpha=0.7,
                                        label="Speed (rpm)")
            self.art["pout"], = ax2.plot([], [], ".", ms=3, color=ORANGE, alpha=0.7,
                                         label="Pout (W)")
            ax1.set_xlabel("torque τ [N·m]")
            ax1.set_ylabel("Speed [rpm]"); ax1.yaxis.label.set_color(BLUE)
            ax2.set_ylabel("Pout [W]");    ax2.yaxis.label.set_color(ORANGE)
            ax1.set_title("Performance — live  (Vbus/Ibus/eff need a dyno)")
            self.art["ax1"], self.art["ax2"] = ax1, ax2
        elif tab == "Torque fit":
            self.fig = Figure(facecolor=PANEL)
            ax0 = self.fig.add_subplot(2, 1, 1); _style(ax0)
            ax1 = self.fig.add_subplot(2, 1, 2, sharex=ax0); _style(ax1)
            self.art["meas"], = ax0.plot([], [], color=BLUE, lw=1.0, label="measured")
            self.art["model"], = ax0.plot([], [], color=ORANGE, lw=1.0, alpha=0.85,
                                          label="live model J·a+τ_f")
            ax0.set_ylabel("torque [N·m]"); ax0.set_title("Dynamic torque — live")
            _legend(ax0)
            self.art["resid"], = ax1.plot([], [], color=FG2, lw=0.8)
            ax1.set_ylabel("resid"); ax1.set_xlabel("time [s]")
            self.art["ax0"], self.art["ax1"] = ax0, ax1
        else:  # Chirp
            self.fig = Figure(facecolor=PANEL)
            ax0 = self.fig.add_subplot(2, 1, 1); _style(ax0)
            ax1 = self.fig.add_subplot(2, 1, 2, sharex=ax0); _style(ax1)
            self.art["q"], = ax0.plot([], [], color=BLUE, lw=1.0, label="q [rad]")
            self.art["w"], = ax0.plot([], [], color=GREEN, lw=0.9, label="ω [rad/s]")
            ax0.set_ylabel("kinematics"); ax0.set_title("Frequency chirp — live")
            _legend(ax0)
            self.art["tau"], = ax1.plot([], [], color=RED, lw=0.9, label="torque")
            ax1.set_ylabel("torque [N·m]"); ax1.set_xlabel("time [s]")
            _legend(ax1)
            self.art["ax0"], self.art["ax1"] = ax0, ax1
        return self.fig

    @staticmethod
    def _relim(*axes):
        for ax in axes:
            ax.relim(); ax.autoscale_view()

    def update(self, buf, params):
        st = buf["staircase"]; ch = buf["chirp"]; tsw = buf.get("torque_speed", {})
        if self.tab == "Torque–speed":
            w = np.asarray(tsw.get("w", [])); tau = np.asarray(tsw.get("tau", []))
            self.art["pts"].set_data(w, tau)
            self._relim(self.art["ax"])
            return
        if self.tab == "Performance":
            # use whatever has torque/speed data — the sweep spans the most torque,
            # the staircase adds low-torque points
            w = np.abs(np.asarray(list(tsw.get("w", [])) + st["w"]))
            tau = np.abs(np.asarray(list(tsw.get("tau", [])) + st["tau"]))
            self.art["spd"].set_data(tau, w * RAD_TO_RPM)
            self.art["pout"].set_data(tau, tau * w)
            self._relim(self.art["ax1"], self.art["ax2"])
            return
        if self.tab == "Friction":
            ws, taus = steady_live(st["w"], st["tau"], st["wcmd"])
            self.art["pts"].set_data(ws, taus)
            if params is not None and ws.size:
                grid = np.linspace(min(ws.min(), -1e-3), max(ws.max(), 1e-3), 200)
                self.art["fit"].set_data(grid, friction_torque(params, grid))
            self._relim(self.art["ax"])
        elif self.tab == "Staircase":
            t = np.asarray(st["t"]); w = np.asarray(st["w"])
            self.art["wcmd"].set_data(t, np.asarray(st["wcmd"]))
            self.art["w"].set_data(t, w)
            ws, taus = steady_live(st["w"], st["tau"], st["wcmd"])
            self.art["pts"].set_data(ws, taus)
            if params is not None and ws.size:
                grid = np.linspace(min(ws.min(), -1e-3), max(ws.max(), 1e-3), 200)
                self.art["fit"].set_data(grid, friction_torque(params, grid))
            self._relim(self.art["ax0"], self.art["ax1"])
        elif self.tab == "Torque fit":
            t = np.asarray(ch["t"]); w = np.asarray(ch["w"]); tau = np.asarray(ch["tau"])
            self.art["meas"].set_data(t, tau)
            if params is not None and t.size > 5:
                a = np.gradient(w, t)
                pred = params.J * a + friction_torque(params, w)
                self.art["model"].set_data(t, pred)
                self.art["resid"].set_data(t, pred - tau)
            self._relim(self.art["ax0"], self.art["ax1"])
        else:  # Chirp
            t = np.asarray(ch["t"])
            self.art["q"].set_data(t, np.asarray(ch["q"]))
            self.art["w"].set_data(t, np.asarray(ch["w"]))
            self.art["tau"].set_data(t, np.asarray(ch["tau"]))
            self._relim(self.art["ax0"], self.art["ax1"])
