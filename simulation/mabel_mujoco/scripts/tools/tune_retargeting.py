"""
tune_retargeting.py -- rigorous, headless tuning of the arm-IK / whole-body objective
weights. It stresses the retargeter on a battery of EDGE-CASE targets (reach across the
body, flip up, full extension / near-singular, fast trajectory) and scores each weight
setting on the objectives the operator cares about:

    fast        -> solve latency (ms)
    no singularity -> manipulability  sqrt(det(J Jᵀ))  (higher = farther from singular)
    human-like  -> contact tracking error (mm) + palm-orientation error
    no weird twist -> wrist-yaw (Arm7) + shoulder-roll (Arm3) deviation from rest, and
                      elbow-below-shoulder fraction
    smooth      -> per-frame joint jitter on a moving trajectory (mm of contact)

It prints the BASELINE (current hyperparameters), a one-at-a-time SENSITIVITY sweep of
each objective weight (so you can see the trade-offs), and a RECOMMENDED setting found
by a small scalarized search. Nothing is mutated on disk -- it edits the in-memory
weights of the teleop.arm module only.

Run:  python scripts/tools/tune_retargeting.py [--quick]
"""
import os
import sys
import time
import argparse
import numpy as np
import mujoco

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.join(_HERE, "..")
sys.path.insert(0, _SCRIPTS)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "..", "..", "controller"))
import teleop as tc
from teleop import arm as armmod

C = tc._A   # ANSI palette for pretty tables


# ----------------------------------------------------------------------------
#  Edge-case target battery (all in the robot world frame, derived from the
#  robot's own FK so they are reachable; we know the natural ground-truth config).
# ----------------------------------------------------------------------------
def make_targets(model, seed=0):
    rng = np.random.default_rng(seed)
    a = tc.ArmRetargeter(model, "right", tc.HandOptimizer(model, "right").R_rob, False)
    a.sync(mujoco.MjData(model))
    lo, hi = a.lo.copy(), a.hi.copy()

    def hk_from_q(q):
        e, c, sh, fwd, up = a._kps(q)
        return {"contact": c, "elbow": e, "fwd": fwd, "up": up, "q": q.copy()}

    targets = {"reachable": [], "extended": [], "cross_up": []}
    # (1) general reachable interior
    for _ in range(14):
        q = lo + (hi - lo) * rng.uniform(0.25, 0.75, size=len(lo))
        targets["reachable"].append(hk_from_q(q))
    # (2) near-singular: elbow nearly straight (Arm4 -> its straight end) + extended
    for _ in range(8):
        q = lo + (hi - lo) * rng.uniform(0.2, 0.8, size=len(lo))
        q[3] = hi[3] - 0.05 * (hi[3] - lo[3])      # elbow near full extension -> low manip
        targets["extended"].append(hk_from_q(q))
    # (3) cross-body / flip-up: shoulder + forearm toward their flip ends
    for _ in range(8):
        q = lo + (hi - lo) * rng.uniform(0.2, 0.8, size=len(lo))
        q[0] = lo[0] + 0.8 * (hi[0] - lo[0])       # shoulder across
        q[4] = lo[4] + 0.9 * (hi[4] - lo[4])       # forearm flip
        targets["cross_up"].append(hk_from_q(q))
    # (4) a smooth trajectory for jitter (interpolate between two reachable configs)
    qa = lo + (hi - lo) * rng.uniform(0.3, 0.5, size=len(lo))
    qb = lo + (hi - lo) * rng.uniform(0.5, 0.7, size=len(lo))
    traj = [hk_from_q(qa + (qb - qa) * t) for t in np.linspace(0, 1, 30)]
    return targets, traj, a


# ----------------------------------------------------------------------------
#  Metrics for one solved target
# ----------------------------------------------------------------------------
def _manip(a, q):
    """Translational manipulability sqrt(det(J Jᵀ)) at the contact over the arm DoFs."""
    a.ik.qpos[a.qadr] = q
    mujoco.mj_kinematics(a.model, a.ik)
    mujoco.mj_comPos(a.model, a.ik)
    jp = np.zeros((3, a.model.nv))
    pt = a.ik.xanchor[a.orca_wrist_jid]
    mujoco.mj_jac(a.model, a.ik, jp, None, pt, a.palm_bid)
    dofs = [a.model.jnt_dofadr[j] for j in a.jids]
    J = jp[:, dofs]
    return float(np.sqrt(max(np.linalg.det(J @ J.T), 0.0)))


def solve_metrics(a, hk, warm=False):
    if not warm:
        a.last_q = a.q_nom.copy()
    t0 = time.perf_counter()
    sol = a.solve(hk)
    ms = (time.perf_counter() - t0) * 1e3
    re, rc, rsh, rfwd, rup = a._kps(sol)
    contact_mm = np.linalg.norm(rc - hk["contact"]) * 1e3
    orient = (1 - float(np.dot(rfwd, hk["fwd"]))) + (1 - float(np.dot(rup, hk["up"])))
    manip = _manip(a, sol)
    j7 = abs(sol[6] - a.q_nom[6]) if len(sol) > 6 else 0.0     # wrist-yaw twist
    j3 = abs(sol[2] - a.q_nom[2])                              # shoulder-roll twist
    elbow_ok = 1.0 if re[2] < rsh[2] + 0.02 else 0.0          # elbow at/below shoulder
    return dict(contact_mm=contact_mm, orient=orient, manip=manip,
                j7=j7, j3=j3, elbow_ok=elbow_ok, ms=ms, sol=sol)


def evaluate(model, a, targets, traj):
    """Aggregate metrics over the whole battery for the CURRENT teleop.arm weights."""
    flat = sum(targets.values(), [])
    M = [solve_metrics(a, hk, warm=False) for hk in flat]
    agg = lambda k: float(np.mean([m[k] for m in M]))
    p95 = lambda k: float(np.percentile([m[k] for m in M], 95))
    # jitter: warm-started trajectory -> mean contact step that is NOT explained by the
    # target step (i.e. solver wobble). We approximate with the std of contact error.
    a.last_q = a.q_nom.copy()
    jit = []
    prev = None
    for hk in traj:
        m = solve_metrics(a, hk, warm=True)
        _, rc, *_ = a._kps(m["sol"])
        if prev is not None:
            jit.append(np.linalg.norm(rc - prev) - np.linalg.norm(hk["contact"] - prevt))
        prev = rc; prevt = hk["contact"]
    jitter_mm = float(np.mean(np.abs(jit))) * 1e3 if jit else 0.0
    return dict(contact_mm=agg("contact_mm"), contact_p95=p95("contact_mm"),
                orient=agg("orient"), manip=agg("manip"),
                j7=np.degrees(agg("j7")), j3=np.degrees(agg("j3")),
                elbow_ok=agg("elbow_ok") * 100, ms=agg("ms"), jitter_mm=jitter_mm)


# weights we tune, and how to set them on the live arm module / instance
WEIGHTS = ["ARM_W_WRIST", "ARM_W_ELBOW", "ARM_W_ORIENT", "ARM_W_SMOOTH", "ARM_MAXITER"]


def set_weights(a, w):
    for k, v in w.items():
        if k == "ARM_REG_J7":
            a.reg_w[6] = v
        elif k == "ARM_REG_J3":
            a.reg_w[2] = v
        else:
            setattr(armmod, k, v)


def current_weights():
    return {k: getattr(armmod, k) for k in WEIGHTS}


# ----------------------------------------------------------------------------
#  Scoring + reporting
# ----------------------------------------------------------------------------
def score(r):
    """Scalarized cost (LOWER = better): balances the objectives. Normalisers chosen so
    each term is ~O(1) at typical values."""
    return (r["contact_mm"] / 30.0          # tracking (target ~30 mm)
            + r["orient"] / 0.15            # orientation
            + 0.5 * (0.02 / max(r["manip"], 1e-4))   # singularity (penalise low manip)
            + r["j7"] / 20.0 + r["j3"] / 25.0        # twist (deg)
            + (100 - r["elbow_ok"]) / 50.0           # natural elbow
            + r["ms"] / 5.0                          # speed (ms)
            + r["jitter_mm"] / 3.0)                  # smoothness


def row(label, r):
    return (f"  {C['gry']}{label:<22}{C['rst']} "
            f"trk {r['contact_mm']:5.1f}/{r['contact_p95']:5.1f}  ori {r['orient']:.3f}  "
            f"manip {r['manip']*1e3:5.2f}  twistJ7/J3 {r['j7']:4.1f}/{r['j3']:4.1f}°  "
            f"elbow {r['elbow_ok']:3.0f}%  {r['ms']:4.1f}ms  jit {r['jitter_mm']:4.1f}  "
            f"{C['vio']}score {score(r):.2f}{C['rst']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    model = mujoco.MjModel.from_xml_path(tc.XML_FILE)
    targets, traj, a = make_targets(model)

    print(f"\n{C['b']}  Arm-IK objective-weight tuning  (legend: trk=contact mm mean/p95, "
          f"ori=orient err, manip x1e3, twist=deg dev, elbow=natural %, jit=jitter mm){C['rst']}")
    print(C['cyn'] + "─" * 100 + C['rst'])

    base_w = current_weights()
    base = evaluate(model, a, targets, traj)
    print(f"{C['b']}  BASELINE (current hyperparameters){C['rst']}")
    print("   ", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in base_w.items()},
          "REG J3/J7", round(a.reg_w[2], 3), "/", round(a.reg_w[6], 3))
    print(row("baseline", base))

    # one-at-a-time sensitivity sweep
    sweeps = {
        "ARM_W_WRIST":  [300, 600, 1000],
        "ARM_W_ELBOW":  [10, 30, 60, 100],
        "ARM_W_ORIENT": [10, 20, 40],
        "ARM_W_SMOOTH": [1.0, 2.0, 4.0, 8.0],
        "ARM_MAXITER":  [20, 30, 45] if not args.quick else [30, 45],
        "ARM_REG_J7":   [0.3, 0.6, 1.2],
        "ARM_REG_J3":   [0.2, 0.4, 0.8],
    }
    print(f"\n{C['b']}  SENSITIVITY (one weight varied, others at baseline){C['rst']}")
    best = {"score": score(base), "w": dict(base_w), "regJ3": a.reg_w[2], "regJ7": a.reg_w[6]}
    for key, vals in sweeps.items():
        print(f"  {C['cyn']}{key}{C['rst']}")
        for v in vals:
            set_weights(a, dict(base_w))                  # reset all weights to baseline
            a.reg_w[2] = 0.40; a.reg_w[6] = 0.60          # reset J3/J7 reg to baseline
            set_weights(a, {key: v})                      # vary the one under test
            r = evaluate(model, a, targets, traj)
            print(row(f"  {key}={v}", r))
            if score(r) < best["score"]:
                best = {"score": score(r), "w": current_weights(),
                        "regJ3": a.reg_w[2], "regJ7": a.reg_w[6], "key": f"{key}={v}"}
        set_weights(a, dict(base_w)); a.reg_w[2] = 0.40; a.reg_w[6] = 0.60   # restore

    # small greedy combined search around the best single moves
    print(f"\n{C['b']}  RECOMMENDED (best of sweep){C['rst']}")
    set_weights(a, best["w"]); a.reg_w[2] = best["regJ3"]; a.reg_w[6] = best["regJ7"]
    rec = evaluate(model, a, targets, traj)
    print("    weights:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in best["w"].items()},
          "REG J3/J7", round(best["regJ3"], 3), "/", round(best["regJ7"], 3),
          f"({best.get('key','baseline')})")
    print(row("recommended", rec))
    print(f"\n  {C['dim']}baseline score {score(base):.2f} -> recommended {score(rec):.2f} "
          f"({'improved' if score(rec) < score(base) else 'no improvement'}){C['rst']}\n")


if __name__ == "__main__":
    main()
