#!/usr/bin/env python3
"""Derive MABEL's published platform specifications from the canonical MJCF.

    python3 simulation/tools/platform_specs.py

Every platform specification that is a property of the *model* (mass, standing height, footprint, vertical reach,
static payload) is computed here rather than quoted from a datasheet, so it
re-derives whenever `simulation/mabel_mujoco/models/mabel_full.xml` changes.
Numbers that are properties of the *built robot* (scale mass, battery runtime,
repeatability) are NOT here -- they must be measured on hardware.

Method
------
mass        sum of body masses in the MJCF.
height      max world z over every visual mesh vertex, at lift retracted and at
            lift fully extended, with all other joints at `qpos0`.
footprint   x/y extent of the base subtree at `qpos0` (chassis, not wheelbase;
            the wheel-contact support polygon lives in
            controller/mabel/motion_model/tip_over.py:SUPPORT_PTS).
reach       min/max world z of the palm over a uniform sample of arm + torso
            configurations inside the joint limits, at both ends of the lift.
            Horizontal reach is reported to BOTH endpoints, because they differ
            by 0.36 m and the manuscript needs each in a different place:
            `..._fingertip_m` (0.87) is the workspace number and is what the
            specification table quotes, `..._wrist_m` (0.51) is the payload
            moment arm.  `max_horizontal_reach_m` aliases the fingertip value.
payload     largest tip mass m for which the static gravity torque stays inside
            every arm actuator's forcerange.  Reported at full horizontal
            extension (the worst case) and at half extension, as the minimum
            over every sampled configuration in that shell.  No safety factor,
            no friction, no gearbox derating: an upper bound on sustained load.

Writes simulation/tools/out/platform_specs.json.
"""
import json
import os

import mujoco
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
MJCF = os.path.join(REPO, "simulation", "mabel_mujoco", "models", "mabel_full.xml")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
G = 9.81
SEED = 0
N_SAMPLE = 20000

ARM_JOINTS = [f"{s} Arm {i}" for s in ("Left", "Right") for i in range(1, 8)]
LIFT_JOINTS = ["Lift Lower", "Lift Upper"]


def _jid(m, name):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)


def _qadr(m, name):
    return m.jnt_qposadr[_jid(m, name)]


def _mesh_extremes(m, d):
    """World-frame bounding box over every mesh vertex of every geom."""
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for g in range(m.ngeom):
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = m.geom_dataid[g]
        v0, nv = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        V = m.mesh_vert[v0:v0 + nv]
        R = d.geom_xmat[g].reshape(3, 3)
        W = V @ R.T + d.geom_xpos[g]
        lo = np.minimum(lo, W.min(0))
        hi = np.maximum(hi, W.max(0))
    return lo, hi


def _subtree_geoms(m, root_name):
    """Geom ids whose body lies in the subtree rooted at `root_name`."""
    root = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, root_name)
    if root < 0:
        return []
    keep = {root}
    for b in range(m.nbody):
        p = b
        while p > 0:
            if p in keep:
                keep.add(b)
                break
            p = m.body_parentid[p]
    return [g for g in range(m.ngeom) if m.geom_bodyid[g] in keep]


def _set_lift(m, d, frac):
    for j in LIFT_JOINTS:
        a = _qadr(m, j)
        d.qpos[a] = m.jnt_range[_jid(m, j)][1] * frac
    mujoco.mj_forward(m, d)


def height_and_footprint(m, d):
    res = {}
    for name, frac in (("retracted", 0.0), ("extended", 1.0)):
        d.qpos[:] = m.qpos0
        _set_lift(m, d, frac)
        lo, hi = _mesh_extremes(m, d)
        res[f"height_{name}_m"] = round(float(hi[2] - lo[2]), 3)

    d.qpos[:] = m.qpos0
    _set_lift(m, d, 0.0)
    gids = _subtree_geoms(m, "base_link") or list(range(m.ngeom))
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for g in gids:
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        if d.geom_xpos[g][2] > 0.45:            # chassis only, not the torso above it
            continue
        mid = m.geom_dataid[g]
        v0, nv = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        W = m.mesh_vert[v0:v0 + nv] @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
        lo = np.minimum(lo, W.min(0))
        hi = np.maximum(hi, W.max(0))
    res["footprint_x_m"] = round(float(hi[0] - lo[0]), 3)
    res["footprint_y_m"] = round(float(hi[1] - lo[1]), 3)
    return res


def _palm_body(m, side="Left"):
    """The WRIST body -- the last body of the arm chain, before the hand.

    Named "palm" historically, but the match below is on ``wrist``, so this stops
    at the hand ROOT and excludes the whole ORCA hand.  Horizontal reach measured
    to this body is therefore a shoulder-to-wrist distance, roughly 0.36 m short
    of the fingertip.  Kept because it is the right moment arm for the payload
    bound (the tip load hangs off the wrist interface); anything describing the
    workspace should use `_fingertip_body` instead.
    """
    best = None
    for b in range(m.nbody):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
        if n and n.lower().startswith(side.lower()) and "wrist" in n.lower():
            best = b
    if best is None:                            # fall back to the arm-7 child
        j = _jid(m, f"{side} Arm 7")
        best = m.jnt_bodyid[j]
    return best


def _fingertip_body(m, side="Left"):
    """(body id, tip offset in that body's frame) for the index fingertip.

    The same point `papers/ral2026/scripts/make_workspace_coverage.py` plots, so
    the table's reach rows and the workspace figure quote one envelope.  The
    offset is derived, not assumed: every mesh vertex of the distal phalanx is
    mapped through its geom pose into the body frame and the farthest one from
    the body origin is the tip.  Returns (id, offset[3]) or (None, None).
    """
    bname = f"{side.lower()}_index_ip"
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, bname)
    if bid < 0:
        return None, None
    pts = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != bid:
            continue
        mid = int(m.geom_dataid[g])
        if mid < 0:
            continue
        va, vn = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
        V = np.asarray(m.mesh_vert[va:va + vn], float)
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, m.geom_quat[g])
        pts.append(V @ R.reshape(3, 3).T + m.geom_pos[g])
    if not pts:
        return bid, np.zeros(3)
    P = np.vstack(pts)
    return bid, P[int(np.argmax(np.linalg.norm(P, axis=1)))].copy()


def _arm_sample(m, d, side, n, seed, lift_frac):
    """Uniform sample of (arm 1-7 + torso) inside the joint limits.

    Returns (wrist world z, horizontal shoulder-to-wrist reach, shoulder world z,
    qpos rows, base qpos, qpos addresses, horizontal shoulder-to-fingertip reach).
    """
    rng = np.random.default_rng(seed)
    palm = _palm_body(m, side)
    tip_bid, tip_off = _fingertip_body(m, side)
    names = [f"{side} Arm {i}" for i in range(1, 8)] + ["Torso"]
    jid = [_jid(m, nm) for nm in names]
    adr = [m.jnt_qposadr[j] for j in jid]
    lim = np.array([m.jnt_range[j] for j in jid])
    shoulder = m.jnt_bodyid[jid[0]]

    d.qpos[:] = m.qpos0
    _set_lift(m, d, lift_frac)
    base = d.qpos.copy()
    Q = rng.uniform(lim[:, 0], lim[:, 1], size=(n, len(names)))
    z = np.empty(n)
    r = np.empty(n)
    zs = np.empty(n)
    rt = np.full(n, np.nan)
    for k in range(n):
        d.qpos[:] = base
        d.qpos[adr] = Q[k]
        mujoco.mj_kinematics(m, d)
        z[k] = d.xpos[palm][2]
        zs[k] = d.xpos[shoulder][2]
        r[k] = np.linalg.norm(d.xpos[palm][:2] - d.xpos[shoulder][:2])
        if tip_bid is not None:
            tip = d.xpos[tip_bid] + d.xmat[tip_bid].reshape(3, 3) @ tip_off
            rt[k] = np.linalg.norm(tip[:2] - d.xpos[shoulder][:2])
    return z, r, zs, Q, base, adr, rt


def vertical_reach(m, d, n=N_SAMPLE, seed=SEED, side="Left"):
    out = {}
    for name, frac in (("retracted", 0.0), ("extended", 1.0)):
        z, r, _, _, _, _, rt = _arm_sample(m, d, side, n, seed, frac)
        out[f"palm_z_min_{name}_m"] = round(float(z.min()), 3)
        out[f"palm_z_max_{name}_m"] = round(float(z.max()), 3)
        if frac == 1.0:
            # Two endpoints, two numbers.  The manuscript's specification table
            # quotes the FINGERTIP value so it agrees with the vertical-reach row
            # and the workspace figure; the wrist value is the payload moment arm.
            out["max_horizontal_reach_wrist_m"] = round(float(r.max()), 3)
            if np.isfinite(rt).any():
                out["max_horizontal_reach_fingertip_m"] = round(float(np.nanmax(rt)), 3)
            out["max_horizontal_reach_m"] = out.get(
                "max_horizontal_reach_fingertip_m", out["max_horizontal_reach_wrist_m"])
    out["vertical_reach_m"] = [out["palm_z_min_retracted_m"],
                               out["palm_z_max_extended_m"]]
    return out


def _arm_subtree(m, side):
    """Bodies carried by the arm chain: everything below shoulder joint 1."""
    root = m.jnt_bodyid[_jid(m, f"{side} Arm 1")]
    keep = {root}
    for b in range(m.nbody):
        p = b
        while p > 0:
            if p in keep:
                keep.add(b)
                break
            p = m.body_parentid[p]
    return sorted(keep)


def static_payload(m, d, side="Left", n=N_SAMPLE, seed=SEED):
    """Static tip payload from the arm actuator torque limits.

    Gravity torque is assembled directly, tau_g,j = sum_b m_b g Jz(com_b)_j over
    the arm subtree, rather than through mj_inverse -- the base is a free joint,
    so the inverse solution charges the arm DOFs with the constraint forces that
    hold the whole robot up.  The payload adds m g |Jz(tip)|, so the admissible
    mass is min_j (tau_max,j - |tau_g,j|) / (g |Jz(tip)_j|).

    Reported at two reach shells: the outer shell (>=95% of the sampled maximum
    horizontal reach) is the number a spec sheet should quote; the mid shell
    (45-55%) is the load close to the body.  Both are medians over the shell,
    with the 5th percentile as the pessimistic bound.
    """
    palm = _palm_body(m, side)
    jid = [_jid(m, f"{side} Arm {i}") for i in range(1, 8)]
    dof = [m.jnt_dofadr[j] for j in jid]
    tau_max = np.array([m.actuator_forcerange[
        mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{side} Arm {i}")][1]
        for i in range(1, 8)])
    carried = _arm_subtree(m, side)

    z, r, zs, Q, base, adr, _ = _arm_sample(m, d, side, n, seed, 0.0)
    rmax = float(r.max())
    horiz = (np.abs(z - zs) <= 0.05) & (r >= 0.90 * rmax)
    J = np.zeros((3, m.nv))
    cap = np.full(n, np.nan)
    keep = np.where((r >= 0.95 * rmax) | horiz
                    | ((r >= 0.45 * rmax) & (r <= 0.55 * rmax)))[0]
    for k in keep:
        d.qpos[:] = base
        d.qpos[adr] = Q[k]
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        tau = np.zeros(7)
        for b in carried:
            if m.body_mass[b] <= 0:
                continue
            mujoco.mj_jacBodyCom(m, d, J, None, b)
            tau += m.body_mass[b] * G * np.array([J[2, i] for i in dof])
        mujoco.mj_jacBody(m, d, J, None, palm)
        jg = np.abs([J[2, i] for i in dof])
        head = tau_max - np.abs(tau)
        with np.errstate(divide="ignore", invalid="ignore"):
            c = np.where(jg > 1e-9, head / (jg * G), np.inf)
        cap[k] = max(0.0, float(np.min(c)))

    out = {}
    for tag, sel in (("horiz", horiz),
                     ("outer", r >= 0.95 * rmax),
                     ("mid", (r >= 0.45 * rmax) & (r <= 0.55 * rmax))):
        v = cap[sel & ~np.isnan(cap)]
        out[f"payload_{tag}_min_kg"] = round(float(v.min()), 2)
        out[f"payload_{tag}_median_kg"] = round(float(np.median(v)), 2)
        out[f"payload_{tag}_n"] = int(v.size)
    out["arm_torque_limit_Nm"] = [float(t) for t in tau_max]
    out["arm_carried_mass_kg"] = round(float(sum(m.body_mass[b] for b in carried)), 2)
    return out


def main():
    m = mujoco.MjModel.from_xml_path(MJCF)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    spec = {
        "mjcf": os.path.relpath(MJCF, REPO),
        "nq": int(m.nq),
        "nu": int(m.nu),
        "ncam": int(m.ncam),
        "nsensor": int(m.nsensor),
        "nsensordata": int(m.nsensordata),
        "mass_kg": round(float(m.body_mass.sum()), 2),
        "lift_stroke_m": round(float(sum(m.jnt_range[_jid(m, j)][1] for j in LIFT_JOINTS)), 4),
    }
    spec.update(height_and_footprint(m, d))
    spec.update(vertical_reach(m, d))
    spec.update(static_payload(m, d))

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "platform_specs.json"), "w") as f:
        json.dump(spec, f, indent=2)

    w = max(len(k) for k in spec)
    print("=" * (w + 34))
    print("MABEL platform specifications (model-derived)")
    print("=" * (w + 34))
    for k, v in spec.items():
        print(f"  {k:<{w}}  {v}")
    print("=" * (w + 34))
    print(f"-> {os.path.join(OUT, 'platform_specs.json')}")


if __name__ == "__main__":
    main()
