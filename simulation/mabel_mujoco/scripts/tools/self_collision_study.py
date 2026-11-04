"""
Self-collision avoidance study — regenerates the figures + metrics for
docs/self_collision.tex.

The avoidance layer is runtime-toggleable and OFF by default; when on it offers
two laws (config.SC_METHOD):
  "stop" (default) — uniformly slow the tracking solution so no monitored gap
                     closes faster than the barrier allows. Pure scaling keeps the
                     motion DIRECTION, so the hand stops straight along the
                     operator's path (a wall, not a force field). Evaluated at the
                     position-servo TARGET pose for speed-robustness.
  "cbf"            — legacy tracking-QP that re-optimises tracking under the
                     barrier; it slides the hand sideways around obstacles.

Figures (vector PDF -> docs/figures/):
  sc_intrusion.pdf    : drive-the-palm-into-the-chest, surface gap vs time —
                        OFF (penetrates) vs STOP vs CBF.
  sc_speed_robust.pdf : minimum standoff vs operator command speed — STOP holds a
                        flat margin; CBF tightens with speed; OFF penetrates.
  sc_envelope.pdf     : the STOP closing-speed barrier min(alpha*slack,
                        sqrt(2*decel*slack)) vs gap, with the realised approach.
  sc_latency.pdf      : per-solve wall-clock latency, bare DLS vs STOP vs CBF.
  sc_compliance.pdf   : compliance "soft wall" — drag a hand into the chest,
                        OFF (penetrates) vs ON (held at the margin).

    python scripts/tools/self_collision_study.py
"""
import os, sys, time, gc
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "..", "controller"))
from mabel import config as C
from mabel.wbc import WholeBodyIK
from mabel.wbc_runtime import WBCEngine
from mabel.self_collision import SelfCollisionGuard, _seg_seg

FIG = os.path.normpath(os.path.join(HERE, "..", "..", "docs", "figures"))
os.makedirs(FIG, exist_ok=True)
XML = os.path.join(HERE, "..", C.SCENE_XML)
plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "lines.linewidth": 1.3})
COL = (3.4, 2.5)

CLR = {None: "crimson", "stop": "seagreen", "cbf": "navy"}
LBL = {None: "OFF (no guard)", "stop": "STOP (default)", "cbf": "CBF-QP"}


def model():
    m = mujoco.MjModel.from_xml_path(XML)
    m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_SENSOR
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    return m, d


def min_gap(probe, data):
    """Tightest surface gap over every monitored capsule pair (m)."""
    g = np.inf
    for ca, cb in probe.pairs:
        p0, p1 = ca.world(data); q0, q1 = cb.world(data)
        wa, wb = _seg_seg(p0, p1, q0, q1)
        g = min(g, np.linalg.norm(wa - wb) - ca.radius - cb.radius)
    return g


# ── closed-loop teleop: drive the right palm straight at the chest ────────────
def run_drive_to_chest(method, speed=0.004, steps=2500):
    """Command the right palm straight at the torso centre under `method`
    (None | "stop" | "cbf"); log the tightest surface gap and its rate."""
    m, d = model()
    ik = WholeBodyIK(m, d)
    ik.set_self_collision(method is not None, method=(method or "stop"))
    ik.sync_to_current()
    probe = SelfCollisionGuard(m, d)
    chest = d.body("torso").xpos.copy()
    dt = m.opt.timestep
    t, gap, ddot = [], [], []
    prev = None
    for k in range(steps):
        ik.target_r_pos += (chest - ik.target_r_pos) * speed
        ik.apply(ik.solve(arms_only=True), dt, arms_only=True)
        mujoco.mj_step(m, d)
        g = min_gap(probe, d)
        t.append(k * dt); gap.append(g)
        ddot.append((g - prev) / dt if prev is not None else 0.0)
        prev = g
    return np.array(t), np.array(gap), np.array(ddot)


def fig_intrusion():
    """OFF vs STOP vs CBF, surface gap vs time at a moderate command speed."""
    fig, ax = plt.subplots(figsize=COL)
    res = {}
    for meth in (None, "stop", "cbf"):
        t, g, _ = run_drive_to_chest(meth, speed=0.004)
        res[meth] = g.min()
        ax.plot(t, g * 100, color=CLR[meth], label=LBL[meth])
    ax.axhline(C.SC_MARGIN * 100, ls="--", color="gray", lw=0.8, label="safety margin $d_s$")
    ax.axhline(0.0, ls=":", color="black", lw=0.8, label="contact ($d{=}0$)")
    ax.fill_between([0, max(t)], -12, 0, color="crimson", alpha=0.06)
    ax.set_xlabel("time [s]"); ax.set_ylabel("min. surface gap [cm]")
    ax.set_ylim(min(res[None] * 100 - 1, -10), 12)
    ax.legend(fontsize=5.5, loc="upper right", ncol=1)
    fig.tight_layout(); fig.savefig(f"{FIG}/sc_intrusion.pdf"); plt.close(fig)
    return res


def fig_speed_robust():
    """Minimum standoff vs operator command speed for each method."""
    speeds = [0.002, 0.004, 0.006, 0.010, 0.015, 0.020]
    out = {meth: [] for meth in (None, "stop", "cbf")}
    for meth in out:
        for s in speeds:
            _, g, _ = run_drive_to_chest(meth, speed=s)
            out[meth].append(g.min() * 100)
    fig, ax = plt.subplots(figsize=COL)
    for meth in (None, "stop", "cbf"):
        ax.plot(speeds, out[meth], "o-", ms=3, color=CLR[meth], label=LBL[meth])
    ax.axhline(C.SC_MARGIN * 100, ls="--", color="gray", lw=0.8, label="margin $d_s$")
    ax.axhline(0.0, ls=":", color="black", lw=0.8, label="contact")
    ax.set_xlabel("command speed (target lerp / tick)")
    ax.set_ylabel("min. standoff [cm]")
    ax.legend(fontsize=5.5, loc="center right")
    fig.tight_layout(); fig.savefig(f"{FIG}/sc_speed_robust.pdf"); plt.close(fig)
    return speeds, out


def fig_envelope():
    """STOP closing-speed barrier vs gap, with the realised approach riding it."""
    t, gap, ddot = run_drive_to_chest("stop", speed=0.004)
    mask = gap < C.SC_INFLUENCE
    dd = np.linspace(C.SC_MARGIN, C.SC_INFLUENCE, 200)
    slack = dd - C.SC_MARGIN
    cap = np.minimum(C.SC_ALPHA * slack, np.sqrt(2.0 * C.SC_STOP_DECEL * slack))
    fig, ax = plt.subplots(figsize=COL)
    ax.fill_between(dd * 100, cap * 100, 60, color="seagreen", alpha=0.10,
                    label="admissible $-\\dot d \\leq v_{\\max}(d)$")
    ax.plot(dd * 100, C.SC_ALPHA * slack * 100, ls="--", color="gray", lw=1.0,
            label="linear $\\alpha(d-d_s)$")
    ax.plot(dd * 100, np.sqrt(2.0 * C.SC_STOP_DECEL * slack) * 100, ls=":",
            color="darkorange", lw=1.0, label="braking $\\sqrt{2a(d-d_s)}$")
    ax.plot(dd * 100, cap * 100, color="seagreen", lw=1.4, label="cap $v_{\\max}(d)$")
    ax.scatter(gap[mask] * 100, -ddot[mask] * 100, s=4, color="navy", alpha=0.5,
               label="realised approach")
    ax.axvline(C.SC_MARGIN * 100, ls="--", color="gray", lw=0.8)
    ax.set_xlabel("surface gap $d$ [cm]"); ax.set_ylabel("approach speed $-\\dot d$ [cm/s]")
    ax.set_xlim(0, C.SC_INFLUENCE * 100); ax.set_ylim(-5, 40)
    ax.legend(fontsize=5.0, loc="upper left")
    fig.tight_layout(); fig.savefig(f"{FIG}/sc_envelope.pdf"); plt.close(fig)


# ── solver latency ───────────────────────────────────────────────────────────
def latency_samples(method, n=6000):
    m, d = model()
    ik = WholeBodyIK(m, d)
    ik.set_self_collision(method is not None, method=(method or "stop"))
    ik.sync_to_current()
    chest = d.body("torso").xpos.copy()
    for _ in range(300):                            # bias into the active-constraint region
        ik.target_r_pos += (chest - ik.target_r_pos) * 0.01
        ik.apply(ik.solve(arms_only=True), m.opt.timestep, arms_only=True)
        mujoco.mj_step(m, d)
    ts = np.empty(n)
    gc.collect(); gc.disable()                      # measure the algorithm, not GC pauses
    try:
        for i in range(n):
            t0 = time.perf_counter(); ik.solve(arms_only=True); ts[i] = time.perf_counter() - t0
    finally:
        gc.enable()
    return ts * 1e3


def fig_latency():
    samp = {meth: latency_samples(meth) for meth in (None, "stop", "cbf")}
    hi = max(np.percentile(samp[m], 99.5) for m in samp)
    bins = np.linspace(0, hi, 70)
    fig, ax = plt.subplots(figsize=COL)
    for meth in (None, "stop", "cbf"):
        ax.hist(samp[meth], bins=bins, color=CLR[meth], alpha=0.55, label=LBL[meth])
    ax.axvline(np.percentile(samp["stop"], 50), color=CLR["stop"], ls="--", lw=0.9,
               label=f"STOP p50 {np.percentile(samp['stop'],50):.2f} ms")
    ax.set_xlabel("solve latency [ms]"); ax.set_ylabel("count")
    ax.legend(fontsize=5.5); fig.tight_layout()
    fig.savefig(f"{FIG}/sc_latency.pdf"); plt.close(fig)
    return samp


# ── compliance "soft wall" (torque domain) ───────────────────────────────────
def run_compliance_drag(sc_on, force=40.0, steps=4000):
    """Drag the right palm toward the chest with a constant pull under the
    gravity-comp compliance law; log the tightest surface gap. With the soft wall
    on, the dragged hand stops at the margin instead of pushing through."""
    m, d = model(); mujoco.mj_step(m, d)
    eng = WBCEngine(m, d)
    eng.set_self_collision(sc_on)
    eng.enter("compliance", "arm")
    probe = SelfCollisionGuard(m, d)
    chest = d.body("torso").xpos.copy()
    dt = m.opt.timestep
    t, gap = [], []
    for k in range(steps):
        dirn = chest - d.body("right_palm").xpos
        nrm = np.linalg.norm(dirn)
        if nrm > 1e-6:
            eng.apply_external({"right_palm": force * dirn / nrm})
        eng.step(dt)
        mujoco.mj_step(m, d)
        t.append(k * dt); gap.append(min_gap(probe, d))
    return np.array(t), np.array(gap)


def fig_compliance():
    fig, ax = plt.subplots(figsize=COL)
    res = {}
    for on in (False, True):
        t, g = run_compliance_drag(on)
        res[on] = g.min()
        ax.plot(t, g * 100, color=("seagreen" if on else "crimson"),
                label=("soft wall ON" if on else "OFF"))
    ax.axhline(C.SC_MARGIN * 100, ls="--", color="gray", lw=0.8, label="safety margin $d_s$")
    ax.axhline(0.0, ls=":", color="black", lw=0.8, label="contact ($d{=}0$)")
    ax.fill_between([0, max(t)], -8, 0, color="crimson", alpha=0.06)
    ax.set_xlabel("time [s]"); ax.set_ylabel("min. surface gap [cm]")
    ax.set_ylim(min(res[False] * 100 - 1, -6), 10)
    ax.legend(fontsize=5.5, loc="upper right")
    fig.tight_layout(); fig.savefig(f"{FIG}/sc_compliance.pdf"); plt.close(fig)
    return res


def pct(ts):
    return ts.mean(), np.percentile(ts, 50), np.percentile(ts, 99), ts.max()


def main():
    print("rendering self-collision figures ->", FIG)
    intr = fig_intrusion()
    speeds, robust = fig_speed_robust()
    fig_envelope()
    lat = fig_latency()
    comp = fig_compliance()

    print("\n=== metrics (for self_collision.tex) ===")
    print(f"margin d_s = {C.SC_MARGIN*100:.1f} cm, influence d_i = {C.SC_INFLUENCE*100:.1f} cm, "
          f"alpha = {C.SC_ALPHA:.0f}/s, decel = {C.SC_STOP_DECEL:.1f} m/s^2")
    print("\n-- intrusion (drive palm into chest, command lerp 0.004) min gap --")
    for meth in (None, "stop", "cbf"):
        print(f"  {LBL[meth]:16s}: {intr[meth]*100:+.2f} cm")
    print("\n-- speed robustness: min standoff [cm] vs command speed --")
    print("  speed   " + "  ".join(f"{s:6.3f}" for s in speeds))
    for meth in (None, "stop", "cbf"):
        print(f"  {LBL[meth]:8s}" + "  ".join(f"{v:+6.2f}" for v in robust[meth]))
    print("\n-- solver latency (6000 samples, GC off) --")
    for meth in (None, "stop", "cbf"):
        mean, p50, p99, mx = pct(lat[meth])
        print(f"  {LBL[meth]:16s}: mean {mean:.3f}  p50 {p50:.3f}  p99 {p99:.3f}  max {mx:.3f} ms")
    p99s = np.percentile(lat["stop"], 99)
    print(f"  physics tick = 2.0 ms; STOP p99 leaves {2.0 - p99s:.2f} ms headroom")
    print("\n-- compliance soft wall (40 N drag into chest) min gap --")
    print(f"  OFF        : {comp[False]*100:+.2f} cm")
    print(f"  soft wall  : {comp[True]*100:+.2f} cm")


if __name__ == "__main__":
    main()
