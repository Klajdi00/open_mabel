#!/usr/bin/env python3
"""
verify_scenes.py -- acceptance checks for every scene in scenes/scenes.json.

Per scene:
  1. LOAD      MjModel.from_xml_path succeeds.
  2. SPAWN     after mj_forward, no penetrating contact (dist < -1e-6) between a
               robot geom (mabel_chassis subtree) and any environment geom, except
               wheels/casters on the geom named "floor".
  3. SETTLE    mj_step x 200: no NaN in qpos/qacc, max |qvel| < 20 rad/s
               (clutter settling is fine, explosions are not).

  4. DYNAMICS  if a mabel_<scene>.dynamics.json sidecar exists, every actor /
               conveyor spec must bind to real joints (simulation_bridge/dynamics.py)
               and 500 driven steps must stay NaN-free.

Prints a summary table and exits nonzero on any failure. Run from the repo root:
    python3 scripts/tools/verify_scenes.py
"""
import json
import os
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
SCENES_DIR = os.path.join(ROOT, "scenes")
REGISTRY = os.path.join(SCENES_DIR, "scenes.json")

N_STEPS = 200
QVEL_LIMIT = 20.0          # rad/s
PEN_TOL = -1e-6            # m
WHEEL_TOKENS = ("wheel", "caster")


def robot_geom_mask(model):
    root = model.body("mabel_chassis").id
    mask = np.zeros(model.ngeom, dtype=bool)
    for g in range(model.ngeom):
        b = model.geom_bodyid[g]
        while b != 0 and b != root:
            b = model.body_parentid[b]
        mask[g] = (b == root)
    return mask


def check_dynamics(xml_path, model):
    """('-'|'ok'|'FAIL', detail) for the scene's optional dynamics sidecar."""
    side = os.path.splitext(xml_path)[0] + ".dynamics.json"
    if not os.path.isfile(side):
        return "-", ""
    import json as _json
    import logging
    repo = os.path.normpath(os.path.join(ROOT, "..", ".."))
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from simulation_bridge import dynamics as SD
    logging.getLogger("simbridge.dynamics").setLevel(logging.ERROR)
    with open(side) as f:
        spec = _json.load(f)
    want = len(spec.get("actors", [])) + len(spec.get("conveyors", []))
    dyn = SD.SceneDynamics(spec, model)
    got = len(dyn.actors) + len(dyn.belts)
    if got != want:
        return "FAIL", f"sidecar binds {got}/{want} entries"
    data = mujoco.MjData(model)
    for _ in range(500):
        dyn.step(model, data)
        mujoco.mj_step(model, data)
    if np.isnan(data.qpos).any() or np.isnan(data.qacc).any():
        return "FAIL", "NaN while driven"
    return "ok", f"{len(dyn.actors)}a/{len(dyn.belts)}c"


def check_scene(xml_path):
    res = {"load": "-", "spawn": "-", "settle": "-", "dyn": "-", "detail": ""}

    # 1. load
    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
    except Exception as e:
        res["load"] = "FAIL"
        res["detail"] = str(e).splitlines()[0][:70]
        return res
    res["load"] = "ok"

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    is_robot = robot_geom_mask(model)

    # 2. spawn penetration check
    bad = []
    for i in range(data.ncon):
        con = data.contact[i]
        g1, g2 = con.geom1, con.geom2
        if is_robot[g1] == is_robot[g2]:
            continue                      # robot-robot or env-env
        if con.dist >= PEN_TOL:
            continue
        rg, eg = (g1, g2) if is_robot[g1] else (g2, g1)
        rname = model.geom(rg).name or f"geom{rg}"
        ename = model.geom(eg).name or f"geom{eg}"
        if ename == "floor" and any(t in rname.lower() for t in WHEEL_TOKENS):
            continue                      # wheels/casters resting on the floor
        bad.append(f"{rname}~{ename}:{con.dist:.1e}")
    if bad:
        res["spawn"] = "FAIL"
        res["detail"] = "; ".join(bad[:3])
        return res
    res["spawn"] = "ok"

    # 3. settling stability
    for _ in range(N_STEPS):
        mujoco.mj_step(model, data)
    nan = bool(np.isnan(data.qpos).any() or np.isnan(data.qacc).any())
    vmax = float(np.abs(data.qvel).max()) if model.nv else 0.0
    if nan or vmax >= QVEL_LIMIT:
        res["settle"] = "FAIL"
        res["detail"] = f"nan={nan} max|qvel|={vmax:.2f}"
        return res
    res["settle"] = "ok"
    res["detail"] = f"max|qvel|={vmax:.2f}"

    # 4. dynamics sidecar (scripted actors), if present
    try:
        res["dyn"], dyn_detail = check_dynamics(xml_path, model)
    except Exception as e:
        res["dyn"], dyn_detail = "FAIL", str(e).splitlines()[0][:60]
    if res["dyn"] == "FAIL":
        res["detail"] = dyn_detail
    elif dyn_detail:
        res["detail"] += f" dyn[{dyn_detail}]"
    return res


def main():
    with open(REGISTRY) as f:
        registry = json.load(f)

    rows, failed = [], False
    for entry in registry:
        xml = os.path.join(SCENES_DIR, entry["xml"])
        r = check_scene(xml)
        ok = all(r[k] in ("ok", "-") for k in ("load", "spawn", "settle", "dyn")) \
            and r["load"] == "ok"
        failed |= not ok
        rows.append((entry["name"], entry["xml"], r, "PASS" if ok else "FAIL"))

    w = max(len(r[0]) for r in rows) + 2
    print(f"\n{'scene':<{w}}{'load':<7}{'spawn':<7}{'settle':<8}{'dyn':<7}{'result':<8}detail")
    print("-" * (w + 40 + 30))
    for name, xml, r, verdict in rows:
        print(f"{name:<{w}}{r['load']:<7}{r['spawn']:<7}{r['settle']:<8}{r['dyn']:<7}{verdict:<8}{r['detail']}")
    print()
    n_pass = sum(1 for r in rows if r[3] == "PASS")
    print(f"{n_pass}/{len(rows)} scenes pass "
          f"(checks: load, spawn penetration<{-PEN_TOL:g}, "
          f"{N_STEPS} steps no-NaN & max|qvel|<{QVEL_LIMIT:g} rad/s)")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
