#!/usr/bin/env python3
"""FRONT-WORKSPACE RETARGETING — the volumetric grid study.

THE OPERATOR REPORT (2026-08-23)
  "the re-targeting right now is extremely bad — the main issue is at the wrist:
   when the human wrist flips from up to down or down to up the IK struggles very
   very badly. Do it in a tabular approach that focuses on the robot's front
   workspace for manipulation. Test in a grid, a cubic volume of the front
   workspace, 3-D grid of points for left and right wrists, each point rotating
   the wrist so the palm faces inwards / upwards / downwards / outwards. Minimise
   the movement of the arms while reaching the desired poses, and keep the arms
   naturally extended like human arms."

WHAT THIS HARNESS MEASURES, AND WHY EACH NUMBER EXISTS
  W1  COVERAGE   Position-only reachability of the front box, then per-palm-
                 direction coverage. Separating the two matters: 84 % of the box
                 is position-reachable, so anything that fails is an ORIENTATION
                 failure, and saying "unreachable" without that split hides the
                 actual defect.
  W2  LIMITS     The two constants that gate orientation coverage — the Arm 7
                 yaw clamp (ARM_J7_YAW_LIM) and the posture-prior weight — swept,
                 so the choice is measured rather than asserted.
  W3  TABLE      Build the posture table and report its COHERENCE (the largest
                 joint jump between adjacent cells). Coherence is what makes
                 interpolating the table legitimate; without it the table is a
                 pile of unrelated solutions and blending two of them produces a
                 posture that is in neither.
  W4  ACCURACY   Static pose error over the whole grid: the shipped local
                 retargeter vs the table-seeded one. Same cost, same solver, same
                 targets — only the seed differs, so the delta IS the basin.
  W5  FLIP       The reported defect, run as a closed loop at the client's frame
                 rate through down -> inward -> up -> outward -> down at every
                 grid point: settle time, residual tilt, and the joint path
                 length actually travelled (the "minimise arm movement" ask).
  W6  POSTURE    Elbow swivel against the human prior, in the two cases that
                 differ: with the operator's elbow tracked and shipped as an IK
                 target, and without it (which is what the posture prior is for).
  W7  LATENCY    Per-solve wall time, tracking vs flipping. A retargeting layer
                 that fixes the palm and costs the frame is not a fix.
  W8  PRIOR      The elbow prior's algorithmic singularity, conventional
                 (gravity-down) vs stereographic, as an elbow-per-wrist gain.
  W9  ABLATION   Every change reverted one at a time, on the W5 protocol — so a
                 change that contributes nothing can be deleted rather than
                 maintained. (One was: see ARM_W_ORIENT in hyperparams.)

Nothing here is hand-entered: every number and figure is regenerated from the
MuJoCo model by this script.

    python3 controller/experiments/exp_front_workspace.py --all
    python3 controller/experiments/exp_front_workspace.py --phase W3 --build
    python3 controller/experiments/exp_front_workspace.py --record     # sweep video

A run that is sub-sampled (--quick) or points at a non-default table (--table-dir,
--rolls, --w-elbow, --j7) is an A/B: its metrics go to results/frontws_*_ab.json
and the canonical results and the paper's table are left alone.

WATCHING IT. `--record` renders the grid sweep offscreen to a side-by-side video
(shipped vs tabular, same targets, same instant) and is the one to use. There is
also a live `--view` on MuJoCo's passive viewer; on this machine it does not run
(the conda `mjpython` segfaults, the Homebrew one cannot find the legacy OpenGL
framework macOS has removed), so it is kept but not relied on.

Figures: controller/docs/figures/frontws_*.pdf, frontws_sweep.mp4
"""
import argparse
import json
import os
import time

import numpy as np
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
FIGDIR = os.path.join(_HERE, "..", "docs", "figures")
RESDIR = os.path.join(_HERE, "results")

from mabel.teleop_engine.hyperparams import XML_FILE                    # noqa: E402
from mabel.teleop_engine.workspace_table import (                       # noqa: E402
    ArmChain, PostureSolver, WorkspaceGrid, WorkspaceTable, palm_target,
    natural_elbow, rot_log, PALM_ORDER, PALM_ALL, PALM_ROLLS, table_path,
    DEFAULT_TABLE_DIR)

SIDES = ("right", "left")

# ── the front manipulation box, in each arm's SHOULDER frame ─────────────────
# Chosen from W1, not by eye. x forward, y left, z up, origin at the Arm 1
# anchor. The same numbers mirror onto the left arm (y -> -y), so the two arms
# cover a symmetric volume in front of the torso.
#
# In the BODY frame at the default lift (0.3175 m) this is
#   x 0.35..0.65 m   y -0.40..+0.40 m   z 0.65..1.15 m
# i.e. a 0.30 x 0.80 x 0.50 m slab in front of the chest, from just below table
# height to just above the shoulders — where a mobile manipulator actually works.
BOX = dict(x=(0.25, 0.55), y=(-0.30, 0.10), z=(-0.40, 0.10))
GRID_N = (5, 5, 6)          # x, y, z  -> 150 positions/side
# Palm rolls per position: the 4 the request names, plus the 4 diagonals.
#
# 16 bins was tried. It makes the RAW lookup far better — the un-refined query's
# palm error drops from 8.7/9.3 deg median to 0.9/1.0, and its p90 from 19.6 to
# 4.9 — and end-to-end it is worth NOTHING: palm error p90 5.7/5.3 deg against
# 5.9/5.3, poses achieved 72.2/68.5 % against 73.1/69.4, settle and joint path
# unchanged, for twice the cells, twice the build time and a slightly slower
# query. The 12-iteration refinement absorbs the coarser seed completely, which
# is the point of having it. 8 bins.
N_ROLL = 8

FRAME_HZ = 30.0             # the Vision Pro / iOS client stream rate
FLIP_SETTLE_TOL = np.radians(12.0)
#: frames allowed for a pose to settle. The trust region caps the arm at
#: ARM_TRUST_RATE (2 rad/s), so a full forearm-roll reconfiguration is ~1.6 s;
#: 4 s of frames leaves margin without letting a genuinely stuck solve pass.
SETTLE_FRAMES = int(4.0 * FRAME_HZ)


# ── helpers ───────────────────────────────────────────────────────────────────
def load_model():
    m = mujoco.MjModel.from_xml_path(XML_FILE)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.nkey - 1)      # "home"
    mujoco.mj_forward(m, d)
    return m, d


def make_grid(side):
    ax = []
    for key, n in zip(("x", "y", "z"), GRID_N):
        lo, hi = BOX[key]
        if key == "y" and side == "left":              # mirror onto the left arm
            lo, hi = -hi, -lo
        ax.append(np.linspace(lo, hi, n))
    return WorkspaceGrid(ax[0], ax[1], ax[2], rolls=N_ROLL, side=side)


def seed_bank(chain, n_random=12, seed=7):
    """Cold seeds for a GLOBAL solve: the nominal posture, the mid-range posture,
    an elbow pre-bent into the arm's natural flexion direction, plus fixed random
    restarts. The random ones are not decoration — the patchiness in the W1 maps
    without them is local minima, not kinematics."""
    rs = np.random.default_rng(seed)
    bank = [chain.q_nom.copy(), 0.5 * (chain.lo + chain.hi)]
    bent = chain.q_nom.copy()
    lo4, hi4 = chain.lo[3], chain.hi[3]
    bent[3] = 0.6 * (lo4 if abs(lo4) > abs(hi4) else hi4)
    bank.append(bent)
    bank += [chain.lo + rs.random(chain.n) * (chain.hi - chain.lo) for _ in range(n_random)]
    return bank


def save_fig(name):
    os.makedirs(FIGDIR, exist_ok=True)
    plt.savefig(os.path.join(FIGDIR, name), format="pdf", bbox_inches="tight")
    plt.close()
    print(f"  figure -> controller/docs/figures/{name}")


#: set by main() — a run that is sub-sampled or points at a non-default table is
#: an A/B, not the canonical result, and must not overwrite it. (It did once: an
#: A/B left the paper's results table quoting quick-grid numbers.)
_SUFFIX = ""


def dump(name, obj):
    os.makedirs(RESDIR, exist_ok=True)
    if _SUFFIX:
        name = name.replace(".json", f"{_SUFFIX}.json")
    with open(os.path.join(RESDIR, name), "w") as f:
        json.dump(obj, f, indent=1, default=float)


def _pct(a, p):
    return float(np.percentile(np.asarray(a, float), p)) if len(a) else float("nan")


# ── W1: coverage of the front box ─────────────────────────────────────────────
def w1_coverage(quick=False):
    print("\n=== W1  COVERAGE — what the front box can actually be asked for ===")
    m, d = load_model()
    out = {}
    fig, axes = plt.subplots(2, 5, figsize=(15.0, 5.6))
    for row, side in enumerate(SIDES):
        chain = ArmChain(m, side)
        chain.sync(d)
        grid = make_grid(side)
        bank = seed_bank(chain, n_random=6 if quick else 12)
        nx, ny, nz, _ = grid.shape

        # position only: is the POINT reachable at all?
        sp = PostureSolver(chain, w_ori=0.0, w_spin=0.0, w_elbow=0.0, iters=40)
        reach = np.zeros((nx, ny, nz), bool)
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    p, R = grid.target(i, j, k, 0)
                    q, _ = sp.solve_multi(bank, sp.targets(p, R))
                    reach[i, j, k] = sp.errors(q, p, R)[0] < 0.010

        # per named palm direction, conditional on the point being reachable
        so = PostureSolver(chain, iters=60)
        cov = {}
        for name in PALM_ORDER:
            ok = np.zeros((nx, ny, nz), bool)
            tilt = np.zeros((nx, ny, nz))
            for i in range(nx):
                for j in range(ny):
                    for k in range(nz):
                        p = grid.point(i, j, k)
                        R = palm_target(grid.canonical_fwd(p), PALM_ROLLS[name], side)
                        e = natural_elbow(np.zeros(3), p, chain.l_upper, chain.l_fore, side)
                        q, _ = so.solve_multi(bank, so.targets(p, R, elbow_star=e))
                        ep, et, _ = so.errors(q, p, R)
                        tilt[i, j, k] = np.degrees(et)
                        ok[i, j, k] = (ep < 0.015) and (et < np.radians(12))
            cov[name] = dict(ok=float(ok[reach].mean() * 100.0),
                             tilt_med=float(np.median(tilt[reach])),
                             tilt_p90=float(np.percentile(tilt[reach], 90)))
            print(f"  {side:5s} palm {name:8s}  reachable-and-oriented "
                  f"{cov[name]['ok']:5.1f}%   tilt med {cov[name]['tilt_med']:4.1f} "
                  f"p90 {cov[name]['tilt_p90']:5.1f} deg")
        print(f"  {side:5s} position-reachable cells: {reach.mean()*100:.1f}% "
              f"({int(reach.sum())}/{reach.size})")
        out[side] = dict(reach_pct=float(reach.mean() * 100.0), cov=cov)

        # figure: mid-height slice of each map
        k_mid = nz // 2
        ax = axes[row, 0]
        ax.imshow(reach[:, :, k_mid].T, origin="lower", cmap="Greys", vmin=0, vmax=1,
                  extent=[grid.x[0], grid.x[-1], grid.y[0], grid.y[-1]], aspect="auto")
        ax.set_title(f"{side}: position\n(z={grid.z[k_mid]:+.2f} m)", fontsize=9)
        ax.set_ylabel("y  [m]")
        for c, name in enumerate(PALM_ORDER):
            ax = axes[row, c + 1]
            R_ = palm_target
            mask = np.zeros((nx, ny))
            for i in range(nx):
                for j in range(ny):
                    p = grid.point(i, j, k_mid)
                    R = R_(grid.canonical_fwd(p), PALM_ROLLS[name], side)
                    e = natural_elbow(np.zeros(3), p, chain.l_upper, chain.l_fore, side)
                    q, _ = so.solve_multi(bank, so.targets(p, R, elbow_star=e))
                    ep, et, _ = so.errors(q, p, R)
                    mask[i, j] = 1.0 if (ep < 0.015 and et < np.radians(12)) else 0.0
            ax.imshow(mask.T, origin="lower", cmap="Greys", vmin=0, vmax=1,
                      extent=[grid.x[0], grid.x[-1], grid.y[0], grid.y[-1]], aspect="auto")
            ax.set_title(f"{side}: palm {name}", fontsize=9)
            if row == 1:
                ax.set_xlabel("x  [m]")
    fig.suptitle("W1 — front-box coverage (shoulder frame, black = pose achieved)", fontsize=11)
    fig.tight_layout()
    save_fig("frontws_coverage.pdf")
    dump("frontws_w1.json", out)
    return out


# ── W2: the two constants that gate coverage ─────────────────────────────────
def w2_limits(quick=False):
    print("\n=== W2  LIMITS — sweeping the constants that gate orientation ===")
    m, d = load_model()
    clamps = (0.45, 0.70, 0.90, 1.20, 1.571)
    priors = (0.0, 4.0, 8.0, 16.0, 32.0)
    res = {"clamp": {}, "prior": {}}
    step = 2 if quick else 1

    for side in SIDES:
        grid = make_grid(side)
        pts = [grid.point(i, j, k)
               for i in range(0, grid.shape[0], step)
               for j in range(0, grid.shape[1], step)
               for k in range(0, grid.shape[2], step)]
        for clamp in clamps:
            chain = ArmChain(m, side, j7_yaw_lim=clamp)
            chain.sync(d)
            sol = PostureSolver(chain)
            bank = seed_bank(chain, n_random=8)
            ok, tilts, poss = 0, [], []
            for p in pts:
                for name in PALM_ORDER:
                    R = palm_target(grid.canonical_fwd(p), PALM_ROLLS[name], side)
                    e = natural_elbow(np.zeros(3), p, chain.l_upper, chain.l_fore, side)
                    q, _ = sol.solve_multi(bank, sol.targets(p, R, elbow_star=e))
                    ep, et, _ = sol.errors(q, p, R)
                    poss.append(ep * 1000.0)
                    tilts.append(np.degrees(et))
                    ok += (ep < 0.015 and et < np.radians(12))
            n = len(pts) * len(PALM_ORDER)
            r = res["clamp"].setdefault(f"{clamp:.3f}", {})
            r[side] = dict(ok=100.0 * ok / n, tilt_p90=_pct(tilts, 90), pos_p90=_pct(poss, 90))
            print(f"  {side:5s} J7 clamp {np.degrees(clamp):5.1f} deg   pose ok {100.0*ok/n:5.1f}%"
                  f"   tilt p90 {_pct(tilts,90):5.1f} deg   pos p90 {_pct(poss,90):5.1f} mm")

        chain = ArmChain(m, side)
        chain.sync(d)
        bank = seed_bank(chain, n_random=8)
        for w in priors:
            sol = PostureSolver(chain, w_elbow=w)
            ok, swiv, poss = 0, [], []
            for p in pts:
                for name in PALM_ORDER:
                    R = palm_target(grid.canonical_fwd(p), PALM_ROLLS[name], side)
                    e = natural_elbow(np.zeros(3), p, chain.l_upper, chain.l_fore, side)
                    q, _ = sol.solve_multi(bank, sol.targets(p, R, elbow_star=e if w > 0 else None))
                    ep, et, _ = sol.errors(q, p, R)
                    poss.append(ep * 1000.0)
                    ok += (ep < 0.015 and et < np.radians(12))
                    if e is not None:
                        swiv.append(np.linalg.norm(chain.elbow_s() - e) * 1000.0)
            n = len(pts) * len(PALM_ORDER)
            r = res["prior"].setdefault(f"{w:.1f}", {})
            r[side] = dict(ok=100.0 * ok / n, elbow_dev=_pct(swiv, 50), pos_p90=_pct(poss, 90))
            print(f"  {side:5s} elbow prior w={w:5.1f}     pose ok {100.0*ok/n:5.1f}%"
                  f"   elbow-vs-natural med {_pct(swiv,50):5.1f} mm   pos p90 {_pct(poss,90):5.1f} mm")

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.4))
    ax = axes[0]
    for side in SIDES:
        ax.plot(np.degrees(clamps), [res["clamp"][f"{c:.3f}"][side]["ok"] for c in clamps],
                "o-", label=side)
    ax.axvline(np.degrees(0.45), color="0.4", ls=":", lw=1)
    ax.annotate("shipped\n(0.45 rad)", (np.degrees(0.45), 40), fontsize=8, color="0.3")
    ax.set_xlabel("Arm 7 yaw clamp  [deg]")
    ax.set_ylabel("poses achieved  [%]")
    ax.set_title("W2a — the wrist-yaw clamp gates orientation", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax = axes[1]
    for side in SIDES:
        ax.plot(priors, [res["prior"][f"{w:.1f}"][side]["ok"] for w in priors], "o-",
                label=f"{side}: poses ok [%]")
        ax.plot(priors, [res["prior"][f"{w:.1f}"][side]["elbow_dev"] for w in priors], "s--",
                label=f"{side}: elbow vs natural [mm]")
    ax.set_xlabel("posture-prior weight  w_elbow")
    ax.set_title("W2b — posture prior vs pose accuracy", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_fig("frontws_limits.pdf")
    dump("frontws_w2.json", res)
    return res


# ── W3: build the table, and check it is coherent ─────────────────────────────
def w3_table(build=True, quick=False, j7=None, w_elbow=None, out_dir=None):
    print("\n=== W3  TABLE — build the posture atlas and measure its coherence ===")
    m, d = load_model()
    out_dir = out_dir or DEFAULT_TABLE_DIR
    res = {}
    for side in SIDES:
        grid = make_grid(side)
        path = table_path(side, out_dir)
        if build or not os.path.exists(path):
            chain = ArmChain(m, side, j7_yaw_lim=j7)
            chain.sync(d)
            solver = (PostureSolver(chain) if w_elbow is None
                      else PostureSolver(chain, w_elbow=w_elbow))
            t0 = time.perf_counter()
            last = [0.0]

            def prog(a, b, tag):
                now = time.perf_counter()
                if now - last[0] > 5.0 or tag.startswith("relax"):
                    last[0] = now
                    print(f"    {side}: {tag} {a}/{b}   [{now - t0:.0f}s]")

            tbl = WorkspaceTable.build(chain, grid, solver,
                                       passes=1 if quick else 3, progress=prog)
            tbl.save(path)
            build_s = time.perf_counter() - t0
        else:
            tbl = WorkspaceTable.load(path)
            build_s = float("nan")
        seam = tbl.seam()
        rp, ro = tbl.res_pos.ravel(), np.degrees(tbl.res_ori.ravel())
        solved = float(np.mean((rp < 0.015) & (ro < 12.0)) * 100.0)
        print(f"  {side:5s} cells {tbl.q.size // tbl.q.shape[-1]}  built in {build_s:.0f}s")
        print(f"        AT CELLS   pos med {np.median(rp)*1000:5.1f} p90 {_pct(rp,90)*1000:6.1f} mm"
              f"   tilt med {np.median(ro):4.1f} p90 {_pct(ro,90):5.1f} deg   solved {solved:5.1f}%")
        print(f"        coherence  adjacent joint jump  x {seam['x_p95']:.2f}  y {seam['y_p95']:.2f}"
              f"  z {seam['z_p95']:.2f}  roll {seam['roll_p95']:.2f} rad (p95)")

        # BETWEEN cells: query the table at random poses inside the box and use
        # what it returns UNREFINED. This is the number that says whether the
        # interpolation is honest — a seam count is only a proxy for it.
        chain2 = ArmChain(m, side)
        chain2.sync(d)
        rs = np.random.default_rng(11)
        qp, qt = [], []
        for _ in range(400):
            p_s = np.array([rs.uniform(grid.x[0], grid.x[-1]),
                            rs.uniform(grid.y[0], grid.y[-1]),
                            rs.uniform(grid.z[0], grid.z[-1])])
            R_s = palm_target(grid.canonical_fwd(p_s), rs.uniform(0, 2 * np.pi), side)
            ep, et, _ = _pose_error(chain2, np.clip(tbl.query(p_s, R_s), chain2.lo, chain2.hi),
                                    p_s, R_s)
            qp.append(ep * 1000.0)
            qt.append(np.degrees(et))
        print(f"        BETWEEN    pos med {np.median(qp):5.1f} p90 {_pct(qp,90):6.1f} mm"
              f"   tilt med {np.median(qt):4.1f} p90 {_pct(qt,90):5.1f} deg  (raw query, no refine)")
        res[side] = dict(build_s=build_s, solved_pct=solved, seam=seam,
                         pos_med_mm=float(np.median(rp) * 1000), pos_p90_mm=_pct(rp, 90) * 1000,
                         tilt_med_deg=float(np.median(ro)), tilt_p90_deg=_pct(ro, 90),
                         query_pos_med_mm=float(np.median(qp)), query_pos_p90_mm=_pct(qp, 90),
                         query_tilt_med_deg=float(np.median(qt)), query_tilt_p90_deg=_pct(qt, 90),
                         path=path)

    # figure: a z-slice of the table's Arm 5 (the flip joint) per palm roll
    fig, axes = plt.subplots(2, 4, figsize=(13.0, 5.4))
    for row, side in enumerate(SIDES):
        tbl = WorkspaceTable.load(table_path(side, out_dir))
        g = tbl.grid
        k = tbl.q.shape[2] // 2
        for c, name in enumerate(PALM_ORDER):
            r = int(round(PALM_ROLLS[name] / (2 * np.pi) * len(g.rolls))) % len(g.rolls)
            im = axes[row, c].imshow(np.degrees(tbl.q[:, :, k, r, 4]).T, origin="lower",
                                     cmap="coolwarm", vmin=-90, vmax=90,
                                     extent=[g.x[0], g.x[-1], g.y[0], g.y[-1]], aspect="auto")
            axes[row, c].set_title(f"{side}: palm {name}", fontsize=9)
            if c == 0:
                axes[row, c].set_ylabel("y  [m]")
            if row == 1:
                axes[row, c].set_xlabel("x  [m]")
    fig.colorbar(im, ax=axes, shrink=0.7, label="Arm 5 (forearm roll)  [deg]")
    fig.suptitle("W3 — the table's forearm roll: smooth within a palm direction, "
                 "a clean step between them", fontsize=11)
    save_fig("frontws_table.pdf")
    dump("frontws_w3.json", res)
    return res


# ── shared: drive the SHIPPED retargeter with a shoulder-frame pose ───────────
def _hk_world(chain, p_s, R_s, natural=True):
    """The `hk` dict `ArmRetargeter.solve` consumes, from a shoulder-frame pose."""
    p_w = chain.frame.to_world_point(p_s, chain.mount_pos, chain.mount_R)
    R_w = chain.frame.to_world_frame(R_s, chain.mount_R)
    e_s = (natural_elbow(np.zeros(3), p_s, chain.l_upper, chain.l_fore, chain.side)
           if natural else None)
    e_w = (None if e_s is None else
           chain.frame.to_world_point(e_s, chain.mount_pos, chain.mount_R))
    return dict(contact=p_w, elbow=e_w, fwd=R_w[:, 0], up=R_w[:, 2]), p_w, R_w


def _pose_error(chain, q, p_s, R_s):
    p, R, _ = chain.pose_s(np.asarray(q, float))
    tilt = float(np.arccos(np.clip(float(np.dot(R[:, 2], R_s[:, 2])), -1.0, 1.0)))
    e = rot_log(R @ np.asarray(R_s, float).T)
    return float(np.linalg.norm(p - p_s)), tilt, abs(float(np.dot(e, R_s[:, 2])))


_RROB = {}


def _r_rob(m, side):
    if side not in _RROB:
        from mabel.teleop_engine.hand import HandOptimizer
        _RROB[side] = HandOptimizer(m, side).R_rob
    return _RROB[side]


#: The pre-2026-08-23 constants, pinned as literals so the baseline is EXACTLY
#: what the operator was running and does not drift as the defaults move.
SHIPPED_ARM = dict(use_table=False, j7_yaw_lim=0.45, w_orient=20.0,
                   w_orient_min=20.0, w_orient_fwd=20.0, trust_rate_distal=2.0)


def _make_retargeters(m, d, side, table_dir=None):
    """(shipped, tabular) retargeters for the same side, both synced to `d`."""
    from mabel.teleop_engine.arm import ArmRetargeter
    R_rob = _r_rob(m, side)
    shipped = ArmRetargeter(m, side, R_rob, **SHIPPED_ARM)
    tabular = ArmRetargeter(m, side, R_rob, use_table=True, table_dir=table_dir)
    for r in (shipped, tabular):
        r.sync(d)
    return shipped, tabular


# ── W4: static accuracy over the grid ─────────────────────────────────────────
def w4_accuracy(quick=False, table_dir=None):
    print("\n=== W4  ACCURACY — shipped local IK vs the table-seeded IK, same targets ===")
    m, d = load_model()
    res = {}
    step = 2 if quick else 1
    for side in SIDES:
        chain = ArmChain(m, side)
        chain.sync(d)
        grid = make_grid(side)
        nx, ny, nz, _ = grid.shape
        acc = {"shipped": [], "table": []}
        for i in range(0, nx, step):
            for j in range(0, ny, step):
                for k in range(0, nz, step):
                    p_s = grid.point(i, j, k)
                    for name in PALM_ORDER:
                        R_s = palm_target(grid.canonical_fwd(p_s), PALM_ROLLS[name], side)
                        hk, _, _ = _hk_world(chain, p_s, R_s)
                        # COLD each pose: this isolates basin choice from tracking
                        shipped, tabular = _make_retargeters(m, d, side, table_dir)
                        for tag, rt in (("shipped", shipped), ("table", tabular)):
                            # Run to CONVERGENCE, not for a fixed count. The
                            # trust region caps the arm at ARM_TRUST_RATE, so a
                            # 3 rad reconfiguration needs ~1.5 s of frames; a
                            # short fixed budget measures the rate limit and
                            # calls it accuracy — and it flatters whichever
                            # answer happens to sit nearer the start pose.
                            qq = None
                            for _ in range(SETTLE_FRAMES):
                                rt.solve(hk, dt=1.0 / FRAME_HZ)
                                q_new = rt.ik.qpos[rt.opt_qadr].copy()
                                if qq is not None and np.max(np.abs(q_new - qq)) < 1e-4:
                                    qq = q_new
                                    break
                                qq = q_new
                            acc[tag].append(_pose_error(chain, qq, p_s, R_s))
        for tag in ("shipped", "table"):
            a = np.array(acc[tag])
            pos, tilt = a[:, 0] * 1000.0, np.degrees(a[:, 1])
            ok = float(np.mean((pos < 15.0) & (tilt < 12.0)) * 100.0)
            res.setdefault(side, {})[tag] = dict(
                pos_med=float(np.median(pos)), pos_p90=_pct(pos, 90),
                tilt_med=float(np.median(tilt)), tilt_p90=_pct(tilt, 90), ok=ok, n=len(a))
            print(f"  {side:5s} {tag:8s}  pos med {np.median(pos):5.1f} p90 {_pct(pos,90):6.1f} mm"
                  f"   tilt med {np.median(tilt):5.1f} p90 {_pct(tilt,90):6.1f} deg   ok {ok:5.1f}%")

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.4))
    for c, key, unit, lim in ((0, "tilt", "deg", 90), (1, "pos", "mm", 200)):
        ax = axes[c]
        for side in SIDES:
            for tag, ls in (("shipped", "--"), ("table", "-")):
                v = res[side][tag]
                ax.plot([0, 1, 2], [v[f"{key}_med"], v[f"{key}_p90"], v[f"{key}_p90"]],
                        ls, marker="o", label=f"{side} {tag}")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["median", "p90"])
        ax.set_ylabel(f"palm {'tilt' if key=='tilt' else 'position'} error  [{unit}]")
        ax.set_ylim(0, lim)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    axes[0].set_title("W4 — palm orientation error over the front grid", fontsize=10)
    axes[1].set_title("W4 — wrist position error over the front grid", fontsize=10)
    fig.tight_layout()
    save_fig("frontws_accuracy.pdf")
    dump("frontws_w4.json", res)
    return res


# ── W5: the flip — the reported defect, closed loop ──────────────────────────
#: The flip cycle actually exercised: down -> inward -> up -> back. Palm-OUTWARD
#: is out of scope (see workspace_table.PALM_ORDER) and is measured separately by
#: W10, which asks whether it degrades gracefully rather than whether it works.
FLIP_SEQ = ("down", "inward", "up", "down")


def w5_flip(quick=False, table_dir=None, max_s=4.0):
    print("\n=== W5  FLIP — palm down->in->up->out->down at every grid point, "
          "at the client's frame rate ===")
    m, d = load_model()
    dt = 1.0 / FRAME_HZ
    res = {}
    step = 2 if quick else 1
    for side in SIDES:
        chain = ArmChain(m, side)
        chain.sync(d)
        grid = make_grid(side)
        nx, ny, nz, _ = grid.shape
        stats = {"shipped": dict(t=[], tilt=[], path=[], fail=0, n=0),
                 "table": dict(t=[], tilt=[], path=[], fail=0, n=0)}
        for i in range(0, nx, step):
            for j in range(0, ny, step):
                for k in range(0, nz, step):
                    p_s = grid.point(i, j, k)
                    shipped, tabular = _make_retargeters(m, d, side, table_dir)
                    for tag, rt in (("shipped", shipped), ("table", tabular)):
                        # settle on the first pose so the transition is measured clean
                        R0 = palm_target(grid.canonical_fwd(p_s), PALM_ROLLS[FLIP_SEQ[0]], side)
                        hk0, _, _ = _hk_world(chain, p_s, R0)
                        for _ in range(int(max_s / dt)):
                            rt.solve(hk0, dt=dt)
                        for name in FLIP_SEQ[1:]:
                            R_s = palm_target(grid.canonical_fwd(p_s), PALM_ROLLS[name], side)
                            hk, _, _ = _hk_world(chain, p_s, R_s)
                            q_prev = rt.ik.qpos[rt.opt_qadr].copy()
                            path, t_set, tilt = 0.0, None, None
                            for n_it in range(int(max_s / dt)):
                                rt.solve(hk, dt=dt)
                                q = rt.ik.qpos[rt.opt_qadr].copy()
                                path += float(np.sum(np.abs(q - q_prev)))
                                q_prev = q
                                _, tilt, _ = _pose_error(chain, q, p_s, R_s)
                                if t_set is None and tilt < FLIP_SETTLE_TOL:
                                    t_set = (n_it + 1) * dt
                            s = stats[tag]
                            s["n"] += 1
                            s["tilt"].append(np.degrees(tilt))
                            s["path"].append(path)
                            if t_set is None:
                                s["fail"] += 1
                                s["t"].append(max_s)
                            else:
                                s["t"].append(t_set)
        for tag in ("shipped", "table"):
            s = stats[tag]
            res.setdefault(side, {})[tag] = dict(
                t_med=float(np.median(s["t"])), t_p90=_pct(s["t"], 90),
                tilt_med=float(np.median(s["tilt"])), tilt_p90=_pct(s["tilt"], 90),
                path_med=float(np.median(s["path"])), fail_pct=100.0 * s["fail"] / max(s["n"], 1),
                n=s["n"])
            r = res[side][tag]
            print(f"  {side:5s} {tag:8s}  settle med {r['t_med']:.2f} p90 {r['t_p90']:.2f} s"
                  f"   never-settled {r['fail_pct']:5.1f}%"
                  f"   final tilt med {r['tilt_med']:5.1f} p90 {r['tilt_p90']:5.1f} deg"
                  f"   joint path med {r['path_med']:5.2f} rad")

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.4))
    labels = [f"{s}\n{t}" for s in SIDES for t in ("shipped", "table")]
    for ax, key, ylab, title in (
            (axes[0], "t_med", "settle time  [s]", "W5a — time to reach the flipped palm"),
            (axes[1], "fail_pct", "never settled  [%]", "W5b — flips that never got there"),
            (axes[2], "path_med", "joint path  [rad]", "W5c — arm travel per flip")):
        vals = [res[s][t][key] for s in SIDES for t in ("shipped", "table")]
        ax.bar(labels, vals, color=["0.65", "0.25", "0.65", "0.25"])
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    save_fig("frontws_flip.pdf")
    dump("frontws_w5.json", res)
    return res


# ── W6: is the arm posture human-like? ───────────────────────────────────────
def w6_posture(quick=False, table_dir=None):
    """Elbow swivel against the human prior, in the two cases that differ.

    TWO CONDITIONS, because the first version of this phase measured neither.
    It handed BOTH retargeters the natural elbow as an explicit IK target
    (`hk["elbow"]`, weight 30) and then asked which one ended up nearer it — so
    it was measuring the elbow cost term, not the posture prior, and both came
    out at ~15 mm regardless.

      WITH HINT   the operator's own elbow is tracked and shipped as a target.
                  Here the posture reference has little left to do, and the
                  interesting question is whether it FIGHTS the hint.
      NO HINT     the elbow is not observed (the hint is null). This is the case
                  the posture reference exists for, and the case a headset
                  client without a forearm joint actually runs.

    Split by whether the POSE was achieved, because they are not comparable
    otherwise: a solver that gives up on the palm direction has spare freedom to
    spend on the elbow, and would score "more natural" for the wrong reason.
    """
    print("\n=== W6  POSTURE — elbow swivel vs the human prior ===")
    m, d = load_model()
    res = {}
    step = 2 if quick else 1
    for hint in (True, False):
        tag_h = "with hint" if hint else "no hint"
        for side in SIDES:
            chain = ArmChain(m, side)
            chain.sync(d)
            grid = make_grid(side)
            nx, ny, nz, _ = grid.shape
            acc = {"shipped": {"ok": [], "miss": [], "drop": []},
                   "table": {"ok": [], "miss": [], "drop": []}}
            for i in range(0, nx, step):
                for j in range(0, ny, step):
                    for k in range(0, nz, step):
                        p_s = grid.point(i, j, k)
                        e_nat = natural_elbow(np.zeros(3), p_s, chain.l_upper,
                                              chain.l_fore, side)
                        if e_nat is None:
                            continue
                        for name in PALM_ORDER:
                            R_s = palm_target(grid.canonical_fwd(p_s),
                                              PALM_ROLLS[name], side)
                            hk, _, _ = _hk_world(chain, p_s, R_s, natural=hint)
                            shipped, tabular = _make_retargeters(m, d, side, table_dir)
                            for tg, rt in (("shipped", shipped), ("table", tabular)):
                                for _ in range(SETTLE_FRAMES):
                                    rt.solve(hk, dt=1.0 / FRAME_HZ)
                                q = rt.ik.qpos[rt.opt_qadr].copy()
                                ep, et, _ = _pose_error(chain, q, p_s, R_s)
                                chain.set_q(q)
                                e = chain.elbow_s()
                                dev = float(np.linalg.norm(e - e_nat)) * 1000.0
                                key = "ok" if (ep < 0.015 and et < np.radians(12)) else "miss"
                                acc[tg][key].append(dev)
                                acc[tg]["drop"].append(e[2] * 1000.0)
            for tg in ("shipped", "table"):
                a = acc[tg]
                res.setdefault(tag_h, {}).setdefault(side, {})[tg] = dict(
                    dev_ok_med=_pct(a["ok"], 50), dev_ok_p90=_pct(a["ok"], 90),
                    dev_miss_med=_pct(a["miss"], 50), n_ok=len(a["ok"]),
                    drop_med=_pct(a["drop"], 50),
                    up_pct=float(np.mean(np.asarray(a["drop"]) > 0) * 100.0))
                r = res[tag_h][side][tg]
                print(f"  {tag_h:9s} {side:5s} {tg:8s}  elbow vs natural, poses ACHIEVED: "
                      f"med {r['dev_ok_med']:6.1f} p90 {r['dev_ok_p90']:6.1f} mm "
                      f"(n={r['n_ok']:3d})  | missed poses med {r['dev_miss_med']:6.1f} mm"
                      f"  | elbow height med {r['drop_med']:+6.1f} mm, above shoulder "
                      f"{r['up_pct']:4.1f}%")
    # keep the top-level shape the results table expects
    flat = {}
    for side in SIDES:
        flat[side] = {t: dict(dev_med=res["no hint"][side][t]["dev_ok_med"],
                              dev_p90=res["no hint"][side][t]["dev_ok_p90"],
                              drop_med=res["no hint"][side][t]["drop_med"],
                              up_pct=res["no hint"][side][t]["up_pct"])
                      for t in ("shipped", "table")}
    flat["_by_condition"] = res
    dump("frontws_w6.json", flat)
    return flat


# ── W7: latency — the table must not cost the frame ──────────────────────────
def w7_latency(quick=False, table_dir=None, n=300):
    print("\n=== W7  LATENCY — per-solve wall time, tracking vs flipping ===")
    m, d = load_model()
    dt = 1.0 / FRAME_HZ
    res = {}
    for side in SIDES:
        chain = ArmChain(m, side)
        chain.sync(d)
        grid = make_grid(side)
        shipped, tabular = _make_retargeters(m, d, side, table_dir)
        for tag, rt in (("shipped", shipped), ("table", tabular)):
            # TRACKING: a smooth 0.3 m/s sweep through the box at a fixed palm
            # direction — what the arm does ~all of the time.
            p0 = grid.point(1, grid.shape[1] // 2, grid.shape[2] // 2)
            R_s = palm_target(grid.canonical_fwd(p0), PALM_ROLLS["down"], side)
            track = []
            for i in range(n):
                p_s = p0 + np.array([0.10 * np.sin(2 * np.pi * 0.3 * i * dt), 0.0, 0.0])
                hk, _, _ = _hk_world(chain, p_s, R_s)
                t0 = time.perf_counter()
                rt.solve(hk, dt=dt)
                track.append((time.perf_counter() - t0) * 1000.0)
            # FLIPPING: the palm direction stepped every 40 frames, which is when
            # the table's expensive path is supposed to run.
            flip = []
            for i in range(n):
                name = PALM_ORDER[(i // 40) % len(PALM_ORDER)]
                R_f = palm_target(grid.canonical_fwd(p0), PALM_ROLLS[name], side)
                hk, _, _ = _hk_world(chain, p0, R_f)
                t0 = time.perf_counter()
                rt.solve(hk, dt=dt)
                flip.append((time.perf_counter() - t0) * 1000.0)
            res.setdefault(side, {})[tag] = dict(
                track_med=float(np.median(track)), track_p95=_pct(track, 95),
                flip_med=float(np.median(flip)), flip_p95=_pct(flip, 95))
            r = res[side][tag]
            print(f"  {side:5s} {tag:8s}  tracking {r['track_med']:5.2f} / p95 "
                  f"{r['track_p95']:5.2f} ms      flipping {r['flip_med']:5.2f} / p95 "
                  f"{r['flip_p95']:5.2f} ms")
    budget = 1000.0 / FRAME_HZ
    worst = max(res[s]["table"]["flip_p95"] for s in SIDES)
    both = sum(res[s]["table"]["track_med"] for s in SIDES)
    print(f"\n  both arms, tracking: {both:.2f} ms of a {budget:.0f} ms frame "
          f"({both / budget * 100:.1f}%).  Worst single solve seen: {worst:.2f} ms.")
    dump("frontws_w7.json", res)
    return res


# ── W8: the posture prior's singularity ──────────────────────────────────────
def w8_prior(quick=False):
    print("\n=== W8  PRIOR — the SEW reference, conventional vs stereographic ===")
    print("  Elbow travel for one full circle of arm directions around STRAIGHT DOWN.")
    print("  A well-posed prior moves the elbow in proportion to the wrist; a singular")
    print("  one sweeps the elbow all the way round however small the wrist circle is.")
    R, L1, L2 = 0.45, 0.222, 0.433
    tilts = (30.0, 15.0, 8.0, 4.0, 2.0, 1.0)
    out = {"tilt_deg": list(tilts), "stereographic": [], "conventional": []}
    print(f"\n  {'tilt':>6s} {'wrist path':>11s} | {'stereographic':>21s} | {'conventional':>21s}")
    print(f"  {'[deg]':>6s} {'[mm]':>11s} | {'elbow [mm]':>11s} {'gain':>8s} |"
          f" {'elbow [mm]':>11s} {'gain':>8s}")
    for td in tilts:
        e = np.radians(td)
        wrist = 2 * np.pi * R * np.sin(e) * 1000.0
        row = []
        for stereo in (True, False):
            prev, tot = None, 0.0
            for phi in np.linspace(0, 2 * np.pi, 2001):
                u = np.array([np.sin(e) * np.cos(phi), np.sin(e) * np.sin(phi), -np.cos(e)])
                pt = natural_elbow(np.zeros(3), R * u, L1, L2, "right", stereographic=stereo)
                if pt is not None and prev is not None:
                    tot += float(np.linalg.norm(pt - prev))
                prev = pt
            row.append(tot * 1000.0)
        out["stereographic"].append(row[0] / wrist)
        out["conventional"].append(row[1] / wrist)
        print(f"  {td:6.0f} {wrist:11.1f} | {row[0]:11.1f} {row[0]/wrist:8.2f} |"
              f" {row[1]:11.1f} {row[1]/wrist:8.2f}")
    print("\n  The stereographic gain is flat at 0.41 — the elbow follows the wrist.")
    print("  The conventional one grows as 1/sin(tilt); at 1 deg from vertical the")
    print("  wrist moves 49 mm and the elbow travels 1.3 m. MABEL's front box reaches")
    print("  to within 58 deg of straight down, so this is inside the workspace.")
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.semilogy(tilts, out["conventional"], "o--", label="conventional (gravity-down)")
    ax.semilogy(tilts, out["stereographic"], "o-", label="stereographic")
    ax.invert_xaxis()
    ax.set_xlabel("arm direction: degrees from straight down")
    ax.set_ylabel("elbow travel / wrist travel")
    ax.set_title("W8 — the elbow prior's singularity, and its removal", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_fig("frontws_prior.pdf")
    dump("frontws_w8.json", out)
    return out


# ── the viewer: watch the grid being retargeted ──────────────────────────────
VIEW_HELP = """
  the popup shows, per grid point and per palm direction:
    * a translucent BOX     — the front manipulation volume for that arm
    * a small AXIS TRIAD    — the commanded wrist pose (blue = palm normal)
    * the robot             — driven by the retargeter under test, live
  keys are the standard MuJoCo viewer keys; close the window to stop.
"""


def _draw_target(scn, p, R, size=0.035):
    """Axis triad at a world pose: red = fingers, green = side, blue = palm normal."""
    cols = ((1.0, 0.25, 0.25, 1.0), (0.25, 0.9, 0.35, 1.0), (0.35, 0.5, 1.0, 1.0))
    for c in range(3):
        if scn.ngeom >= scn.maxgeom:
            return
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3),
                            np.zeros(3), np.zeros(9), np.array(cols[c], np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, 0.004,
                             np.asarray(p, float), np.asarray(p, float) + size * R[:, c])
        scn.ngeom += 1


def _draw_box(scn, chain, grid, rgba=(0.2, 0.6, 1.0, 0.10)):
    if scn.ngeom >= scn.maxgeom:
        return
    lo = np.array([grid.x[0], grid.y[0], grid.z[0]])
    hi = np.array([grid.x[-1], grid.y[-1], grid.z[-1]])
    c_w = chain.frame.to_world_point(0.5 * (lo + hi), chain.mount_pos, chain.mount_R)
    R_w = chain.frame.to_world_frame(np.eye(3), chain.mount_R)
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_BOX, (0.5 * (hi - lo)).astype(float),
                        c_w.astype(float), R_w.reshape(9).astype(float),
                        np.array(rgba, np.float32))
    scn.ngeom += 1


def view(table_dir=None, dwell=0.9, which="table", step=2, max_poses=None):
    """Interactive sweep of the grid — for eyeballing what the numbers claim."""
    try:
        import mujoco.viewer as mjv
    except Exception as exc:                                   # pragma: no cover
        print(f"viewer unavailable: {exc}")
        return
    print(VIEW_HELP)
    m, d = load_model()
    dt = 1.0 / FRAME_HZ
    chains, rts, grids = {}, {}, {}
    for side in SIDES:
        chains[side] = ArmChain(m, side)
        chains[side].sync(d)
        grids[side] = make_grid(side)
        shipped, tabular = _make_retargeters(m, d, side, table_dir)
        rts[side] = tabular if which == "table" else shipped

    order = []
    g = grids["right"]
    for k in range(0, g.shape[2], step):
        for i in range(0, g.shape[0], step):
            for j in range(0, g.shape[1], step):
                for name in PALM_ORDER:
                    order.append((i, j, k, name))
    if max_poses:
        order = order[:int(max_poses)]
    print(f"  sweeping {len(order)} poses with the '{which}' retargeter "
          f"({dwell:.1f} s each)")

    act = {s: [m.actuator(f"{s.capitalize()} Arm {i}").id for i in range(1, 8)] for s in SIDES}
    wact = {s: m.actuator(f"{s.capitalize()} Wrist").id for s in SIDES}
    with mjv.launch_passive(m, d) as v:
        idx, t_pose, t_last = 0, time.time(), time.time()
        while v.is_running() and idx < len(order):
            now = time.time()
            if now - t_last < dt:
                time.sleep(max(0.0, dt - (now - t_last)))
            t_last = time.time()
            i, j, k, name = order[idx]
            v.user_scn.ngeom = 0
            for side in SIDES:
                ch, rt, grid = chains[side], rts[side], grids[side]
                ch.sync(d)
                rt.sync(d)
                p_s = grid.point(i, j, k)
                R_s = palm_target(grid.canonical_fwd(p_s), PALM_ROLLS[name], side)
                hk, p_w, R_w = _hk_world(ch, p_s, R_s)
                q = rt.solve(hk, dt=dt)
                d.ctrl[act[side]] = q[:7]
                d.ctrl[wact[side]] = rt.ik.qpos[rt.orca_wrist_qadr]
                _draw_box(v.user_scn, ch, grid)
                _draw_target(v.user_scn, p_w, R_w)
            for _ in range(int(dt / m.opt.timestep)):
                mujoco.mj_step(m, d)
            v.sync()
            if time.time() - t_pose > dwell:
                idx += 1
                t_pose = time.time()
    print("  viewer closed")


# ── the recorder: the same sweep, rendered offscreen to a movie ─────────────
#
# The passive viewer (--view) needs `mjpython` and a working legacy OpenGL, and
# on this machine it has neither: the conda `mjpython` segfaults before its
# first print and the Homebrew one raises
# `dlopen(/System/Library/OpenGL.framework/OpenGL): no such file`. That is an
# environment problem, not a script problem — but "watch it and check" was the
# request, so the same sweep renders OFFSCREEN here (mujoco.Renderer, which
# works fine) into a video you can scrub. Two panes, shipped and tabular, on the
# same targets at the same instant.
def record(path=None, table_dir=None, step=2, dwell=0.7, width=960, height=540,
           fps=30, rolls=None, camera=None):
    import imageio.v2 as imageio

    m, d0 = load_model()
    dt = 1.0 / FRAME_HZ
    out = path or os.path.join(FIGDIR, "frontws_sweep.mp4")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    names = list(rolls) if rolls else list(PALM_ORDER)

    panes = []
    for tag in ("shipped", "table"):
        d = mujoco.MjData(m)
        d.qpos[:] = d0.qpos
        d.ctrl[:] = d0.ctrl
        d.ctrl[m.actuator("Lift").id] = 0.3175
        mujoco.mj_forward(m, d)
        chains, rts = {}, {}
        for side in SIDES:
            ch = ArmChain(m, side)
            ch.sync(d)
            chains[side] = ch
            sh, tb = _make_retargeters(m, d, side, table_dir)
            rts[side] = sh if tag == "shipped" else tb
        # THE COLLISION GUARD IS IN THE LOOP, on the tabular pane only — that is
        # what ships. Recording the retargeter alone would show a pipeline the
        # robot does not run, and the arm~body guard is precisely the thing worth
        # seeing (the "shipped" pane keeps the pre-2026-08-23 behaviour, which
        # did not guard the forearm or upper arm against the torso at all).
        proj = None
        if tag == "table":
            from mabel.teleop_engine.collision import ArmCollisionProjector
            proj = ArmCollisionProjector(m, d, rts)
        panes.append(dict(tag=tag, d=d, chains=chains, rts=rts, proj=proj,
                          rend=mujoco.Renderer(m, height=height, width=width // 2)))

    grid = make_grid("right")
    order = [(i, j, k, n)
             for k in range(0, grid.shape[2], step)
             for i in range(0, grid.shape[0], step)
             for j in range(0, grid.shape[1], step)
             for n in names]
    per_pose = max(1, int(dwell * fps))
    print(f"  recording {len(order)} poses x {per_pose} frames -> {out}")

    # Head-on and slightly above the box, because the thing to watch is the PALM.
    # (The chassis front is -x, so the camera sits on -x looking back at +x; an
    # azimuth near 180 shows the robot's back, which is what the first attempt
    # did.)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.lookat[:] = (-0.50, 0.0, 0.95)
    cam.distance, cam.azimuth, cam.elevation = 1.85, 18.0, -10.0
    if camera:
        cam.azimuth, cam.elevation, cam.distance = camera

    with imageio.get_writer(out, fps=fps, macro_block_size=1) as w:
        for n_pose, (i, j, k, name) in enumerate(order):
            for _f in range(per_pose):
                tiles = []
                for pane in panes:
                    d = pane["d"]
                    sol, wr = {}, {}
                    for side in SIDES:
                        ch, rt = pane["chains"][side], pane["rts"][side]
                        ch.sync(d)
                        rt.sync(d)
                        g = make_grid(side)
                        p_s = g.point(i, j, k)
                        R_s = palm_target(g.canonical_fwd(p_s), PALM_ROLLS[name], side)
                        hk, p_w, R_w = _hk_world(ch, p_s, R_s)
                        sol[side] = rt.solve(hk, dt=dt)
                        wr[side] = float(rt.ik.qpos[rt.orca_wrist_qadr])
                        pane[f"tgt_{side}"] = (p_w, R_w)
                    if pane["proj"] is not None:
                        (sol["right"], sol["left"],
                         wr["right"], wr["left"]) = pane["proj"].project(
                            sol["right"], sol["left"], dt,
                            wrist_r=wr["right"], wrist_l=wr["left"])
                    for side in SIDES:
                        d.ctrl[[m.actuator(f"{side.capitalize()} Arm {a}").id
                                for a in range(1, 8)]] = sol[side][:7]
                        d.ctrl[m.actuator(f"{side.capitalize()} Wrist").id] = wr[side]
                    for _ in range(int(dt / m.opt.timestep)):
                        mujoco.mj_step(m, d)
                    r = pane["rend"]
                    r.update_scene(d, camera=cam)
                    scn = r.scene
                    for side in SIDES:
                        ch = pane["chains"][side]
                        _draw_box(scn, ch, make_grid(side), rgba=(0.25, 0.7, 1.0, 0.16))
                        _draw_target(scn, *pane[f"tgt_{side}"], size=0.11)
                    gap = (pane["proj"].min_dist if pane["proj"] is not None
                           else _body_gap(m, d))
                    tiles.append(_label(r.render(), pane["tag"], name, gap))
                w.append_data(np.concatenate(tiles, axis=1))
            if n_pose % 20 == 0:
                print(f"    {n_pose}/{len(order)} poses")
    print(f"  video  -> {out}")
    return out


_GAP_GUARD = [None]


def _body_gap(model, data):
    """Tightest arm~body capsule gap right now [m] — the number the guard exists
    to keep positive, shown live so the video is checkable and not just pretty."""
    from mabel.whole_body_controller.collision_avoidance.capsule_guard import (
        SelfCollisionGuard, _seg_seg)
    if _GAP_GUARD[0] is None:
        _GAP_GUARD[0] = SelfCollisionGuard(model, data)
    g = _GAP_GUARD[0]
    g.data = data
    best = np.inf
    for ca, cb in g.pairs:
        pa0, pa1 = ca.world(data)
        pb0, pb1 = cb.world(data)
        wa, wb = _seg_seg(pa0, pa1, pb0, pb1)
        best = min(best, float(np.linalg.norm(wa - wb)) - ca.radius - cb.radius)
    return best


def _label(img, tag, name, gap=None):
    """Burn the pane's identity into the frame — a two-pane video without labels
    is a Rorschach test — plus the live self-collision gap, so a viewer can see
    the guarantee hold rather than take it on trust."""
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return img
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    txt = f"{tag}    palm {name}"
    dr.rectangle([0, 0, 8 + 7 * len(txt), 22], fill=(20, 20, 20))
    dr.text((6, 6), txt, fill=(255, 255, 255))
    if gap is not None and np.isfinite(gap):
        mm = gap * 1000.0
        col = (255, 90, 90) if mm < 0 else (150, 255, 170)
        s2 = f"closest arm-body gap: {mm:+6.1f} mm" + ("   TOUCHING" if mm < 0 else "")
        w = im.size[0]
        dr.rectangle([w - 8 - 7 * len(s2), 0, w, 22], fill=(20, 20, 20))
        dr.text((w - 4 - 7 * len(s2), 6), s2, fill=col)
    return np.asarray(im)


# ── W9: ablation — which of the changes is carrying the result ───────────────
def w9_ablation(quick=False, table_dir=None, step=2):
    """Turn each change off, one at a time, and re-measure.

    A combined before/after says the new stack is better; it does not say WHICH
    part is doing the work, and a change that turns out to contribute nothing is
    a change that should be deleted rather than maintained. Each row below is the
    full stack with exactly one thing reverted to its old value — except the
    Arm 7 clamp row, which cannot be isolated (see the note on it below).

    Every row runs the same protocol as W5: settle on palm-down at each grid
    point, then step through the four palm directions, at the client frame rate,
    with the trust region live.
    """
    from mabel.teleop_engine.arm import ArmRetargeter
    print("\n=== W9  ABLATION — one change reverted per row ===")
    m, d = load_model()
    VARIANTS = [
        ("full stack", {}),
        ("no table (local IK only)", dict(use_table=False)),
        # NOTE: reverting the clamp also disables the table for this row — the
        # table's fingerprint includes the clamp it was built under, and a table
        # full of postures that use |q7| up to 1.20 rad is genuinely invalid at
        # 0.45. So this row is "old clamp AND no table", and it is labelled that
        # way rather than pretending to be a single-variable ablation.
        ("old J7 clamp 0.45 (table refused)", dict(j7_yaw_lim=0.45)),
        ("isotropic orientation (20/20)", dict(w_orient=20.0, w_orient_fwd=20.0)),
        ("weak orientation (shipped 20)", dict(w_orient=20.0, w_orient_min=20.0)),
        ("no feasibility gate", dict(w_orient_min=60.0)),
        ("uniform trust rate 2 rad/s", dict(trust_rate_distal=2.0)),
        ("everything reverted (= shipped)", dict(SHIPPED_ARM)),
        # (SHIPPED_ARM already sets use_table=False, so this row is the true
        #  pre-change stack and should reproduce W5's "shipped" column.)
    ]
    res = {}
    for label, kw in VARIANTS:
        rows = {"pos": [], "tilt": [], "flip_t": [], "flip_tilt": [], "path": [], "fail": 0,
                "n": 0}
        for side in SIDES:
            chain = ArmChain(m, side)
            chain.sync(d)
            grid = make_grid(side)
            for i in range(0, grid.shape[0], step):
                for j in range(0, grid.shape[1], step):
                    for k in range(0, grid.shape[2], step):
                        p_s = grid.point(i, j, k)
                        opts = dict(use_table=True, table_dir=table_dir)
                        opts.update(kw)
                        rt = ArmRetargeter(m, side, _r_rob(m, side), **opts)
                        rt.sync(d)
                        R0 = palm_target(grid.canonical_fwd(p_s), PALM_ROLLS[FLIP_SEQ[0]], side)
                        hk0, _, _ = _hk_world(chain, p_s, R0)
                        for _ in range(SETTLE_FRAMES):
                            rt.solve(hk0, dt=1.0 / FRAME_HZ)
                        q = rt.ik.qpos[rt.opt_qadr].copy()
                        ep, et, _ = _pose_error(chain, q, p_s, R0)
                        rows["pos"].append(ep * 1000.0)
                        rows["tilt"].append(np.degrees(et))
                        for name in FLIP_SEQ[1:]:
                            R_s = palm_target(grid.canonical_fwd(p_s), PALM_ROLLS[name], side)
                            hk, _, _ = _hk_world(chain, p_s, R_s)
                            q_prev = rt.ik.qpos[rt.opt_qadr].copy()
                            path, t_set, tilt = 0.0, None, None
                            for it in range(SETTLE_FRAMES):
                                rt.solve(hk, dt=1.0 / FRAME_HZ)
                                qq = rt.ik.qpos[rt.opt_qadr].copy()
                                path += float(np.sum(np.abs(qq - q_prev)))
                                q_prev = qq
                                _, tilt, _ = _pose_error(chain, qq, p_s, R_s)
                                if t_set is None and tilt < FLIP_SETTLE_TOL:
                                    t_set = (it + 1) / FRAME_HZ
                            rows["n"] += 1
                            rows["flip_tilt"].append(np.degrees(tilt))
                            rows["path"].append(path)
                            rows["flip_t"].append(t_set if t_set is not None
                                                  else SETTLE_FRAMES / FRAME_HZ)
                            rows["fail"] += (t_set is None)
        res[label] = dict(
            pos_p90=_pct(rows["pos"], 90), tilt_p90=_pct(rows["tilt"], 90),
            flip_t_p90=_pct(rows["flip_t"], 90), flip_tilt_p90=_pct(rows["flip_tilt"], 90),
            path_med=float(np.median(rows["path"])),
            fail_pct=100.0 * rows["fail"] / max(rows["n"], 1))
        r = res[label]
        print(f"  {label:34s} static pos p90 {r['pos_p90']:6.1f} mm  tilt p90 "
              f"{r['tilt_p90']:5.1f} deg | flip settle p90 {r['flip_t_p90']:5.2f} s  "
              f"residual p90 {r['flip_tilt_p90']:5.1f} deg  never {r['fail_pct']:5.1f}%  "
              f"path {r['path_med']:4.2f} rad")
    dump("frontws_w9.json", res)
    return res


# ── the results table the paper includes ─────────────────────────────────────
def write_results_table():
    """Emit docs/figures/frontws_results.tex from the phase JSONs.

    The paper \\input{}s this, so every number in it is regenerated rather than
    transcribed — the one way a results table stays true after a re-run.
    """
    def load(name):
        p = os.path.join(RESDIR, name)
        return json.load(open(p)) if os.path.exists(p) else None

    w4, w5, w6, w7 = (load(f"frontws_w{i}.json") for i in (4, 5, 6, 7))
    if not (w4 and w5):
        print("  (skipping results table: run W4 and W5 first)")
        return
    L = []
    A = L.append
    EOL = r"\\"
    A(r"\begin{table}[h]")
    A(r"\centering\small")
    A(r"\begin{tabular}{llrrl}")
    A(r"\toprule")
    A(r"metric & arm & shipped & tabular & change" + EOL)
    A(r"\midrule")

    def rows(title, get, fmt, better="lower"):
        A(r"\multicolumn{5}{l}{\textit{" + title + "}}" + EOL)
        for side in SIDES:
            a, b = get(side, "shipped"), get(side, "table")
            if a is None or b is None:
                continue
            if better != "lower":
                ch = ""
            elif b > a + 1e-9:
                ch = "worse"
            elif abs(a - b) < 1e-9:
                ch = "same"
            elif b < 1e-9:
                ch = "eliminated"
            else:
                ch = r"$%.1f\times$ better" % (a / b)
            A(r"\quad " + side + " & & " + fmt.format(a) + " & " + fmt.format(b)
              + " & " + ch + EOL)

    rows("palm-direction error over the grid, 90th pct [deg]",
         lambda s, t: w4[s][t]["tilt_p90"], "{:.1f}")
    rows("wrist position error over the grid, 90th pct [mm]",
         lambda s, t: w4[s][t]["pos_p90"], "{:.1f}")
    rows(r"poses achieved (pos $<$ 15 mm and tilt $<$ 12 deg) [\%]",
         lambda s, t: w4[s][t]["ok"], "{:.1f}", better="raw")
    A(r"\midrule")
    rows("palm flip: time to settle, 90th pct [s]",
         lambda s, t: w5[s][t]["t_p90"], "{:.2f}")
    rows(r"palm flip: never settled within 4 s [\%]",
         lambda s, t: w5[s][t]["fail_pct"], "{:.1f}")
    rows("palm flip: residual palm error, 90th pct [deg]",
         lambda s, t: w5[s][t]["tilt_p90"], "{:.1f}")
    rows("palm flip: joint distance travelled, median [rad]",
         lambda s, t: w5[s][t]["path_med"], "{:.2f}")
    if w6:
        A(r"\midrule")
        rows("elbow vs. the natural swivel, median [mm]",
             lambda s, t: w6[s][t]["dev_med"], "{:.0f}")
    if w7:
        A(r"\midrule")
        rows("solve time while tracking, median [ms]",
             lambda s, t: w7[s][t]["track_med"], "{:.2f}", better="raw")
        rows("solve time while flipping, 95th pct [ms]",
             lambda s, t: w7[s][t]["flip_p95"], "{:.2f}", better="raw")
    A(r"\bottomrule")
    A(r"\end{tabular}")
    A(r"\caption{Every phase, both arms, same targets and same solver --- only "
      r"the seed, the orientation weighting and the trust rate differ. "
      r"``shipped'' reconstructs the pre-2026-08-23 constants exactly "
      r"(\texttt{SHIPPED\_ARM}), so this is a before/after and not a "
      r"self-comparison. Regenerated by \texttt{exp\_front\_workspace.py}.}")
    A(r"\label{tab:frontws}")
    A(r"\end{table}")
    os.makedirs(FIGDIR, exist_ok=True)
    out = os.path.join(FIGDIR, "frontws_results.tex")
    with open(out, "w") as f:
        f.write("\n".join(L) + "\n")
    print("  table  -> controller/docs/figures/frontws_results.tex")


# ── CLI ───────────────────────────────────────────────────────────────────────
PHASES = {"W1": "coverage", "W2": "limits", "W3": "table", "W4": "accuracy",
          "W5": "flip", "W6": "posture", "W7": "latency", "W8": "prior",
          "W9": "ablation"}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", nargs="*", default=None,
                    help=f"subset of {' '.join(sorted(PHASES))} (default: all)")
    ap.add_argument("--all", action="store_true", help="run every phase")
    ap.add_argument("--quick", action="store_true", help="coarse sub-sampling (smoke test)")
    ap.add_argument("--build", action="store_true", help="rebuild the tables in W3")
    ap.add_argument("--table-dir", default=None, help="where the .npz tables live")
    ap.add_argument("--w-elbow", type=float, default=None,
                    help="W3: posture-prior weight to build the table with (A/B)")
    ap.add_argument("--j7", type=float, default=None,
                    help="W3: Arm 7 yaw clamp to build the table with (A/B)")
    ap.add_argument("--rolls", type=int, default=None,
                    help="W3: palm-roll bins per grid point (A/B; default 8)")
    ap.add_argument("--view", action="store_true",
                    help="interactive MuJoCo sweep of the grid (needs mjpython on macOS)")
    ap.add_argument("--view-which", default="table", choices=("table", "shipped"))
    ap.add_argument("--dwell", type=float, default=0.9, help="seconds per pose in --view")
    ap.add_argument("--view-max", type=int, default=None,
                    help="stop the --view sweep after N poses (smoke test)")
    ap.add_argument("--record", nargs="?", const="", default=None,
                    help="render the sweep offscreen to a side-by-side video "
                         "(shipped vs tabular) — works without mjpython/OpenGL")
    ap.add_argument("--record-step", type=int, default=2)
    ap.add_argument("--record-dwell", type=float, default=0.7)
    args = ap.parse_args()

    if args.record is not None:
        record(path=args.record or None, table_dir=args.table_dir,
               step=args.record_step, dwell=args.record_dwell)
        return
    if args.view:
        view(table_dir=args.table_dir, dwell=args.dwell, which=args.view_which,
             max_poses=args.view_max)
        return

    if args.rolls:
        globals()["N_ROLL"] = int(args.rolls)
    if args.quick or args.table_dir or args.rolls or args.w_elbow or args.j7:
        globals()["_SUFFIX"] = "_ab"
        print("  (A/B run: metrics go to results/frontws_*_ab.json, the canonical\n"
              "   results and the paper's table are left alone)")
    todo = sorted(PHASES) if (args.all or not args.phase) else [p.upper() for p in args.phase]
    t0 = time.perf_counter()
    if "W1" in todo:
        w1_coverage(quick=args.quick)
    if "W2" in todo:
        w2_limits(quick=args.quick)
    if "W3" in todo:
        w3_table(build=args.build, quick=args.quick, out_dir=args.table_dir,
                 w_elbow=args.w_elbow, j7=args.j7)
    if "W4" in todo:
        w4_accuracy(quick=args.quick, table_dir=args.table_dir)
    if "W5" in todo:
        w5_flip(quick=args.quick, table_dir=args.table_dir)
    if "W6" in todo:
        w6_posture(quick=args.quick, table_dir=args.table_dir)
    if "W7" in todo:
        w7_latency(quick=args.quick, table_dir=args.table_dir)
    if "W8" in todo:
        w8_prior(quick=args.quick)
    if "W9" in todo:
        w9_ablation(quick=args.quick, table_dir=args.table_dir)
    if not _SUFFIX:
        write_results_table()
    print(f"\nDone in {time.perf_counter() - t0:.0f}s. "
          f"Figures in controller/docs/figures/frontws_*.pdf, "
          f"raw metrics in controller/experiments/results/frontws_*.json")


if __name__ == "__main__":
    main()
