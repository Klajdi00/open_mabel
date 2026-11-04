#!/usr/bin/env python3
"""The Python side of tools/webcam_parity.mjs: NOVA's runtime on a list of
`teleop_frame` payloads.

Runs ../NOVA/controller/lab_retarget.py LabRetargeter — the class NOVA's
arm_retarget.py --serve steps on the robot — offline (frame clock), seeded at
the home keyframe, upper-body scope, the lift held at its home height (NOVA
mirrors its measured column into the engine). The neck is the engine's own
(free) neck within NOVA's windows — JOINTS carries it, as on the robot.

    python3 web_gui/retarget_lab/tools/webcam_parity_ref.py FRAMES.json OUT.json [--no-self-collision]

Writes per frame: the commanded joints (arms, wrists, torso clamped to NOVA's
range, neck), the wrist targets, and whether the self-collision QP had any
active barrier row that frame.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))     # MABEL/
NOVA = os.environ.get("NOVA_ROOT", os.path.join(os.path.dirname(REPO), "NOVA"))
os.environ.setdefault("NOVA_MABEL_ROOT", REPO)
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.path.insert(0, os.path.join(NOVA, "controller"))
sys.path.insert(0, NOVA)


def main():
    src, dst = sys.argv[1], sys.argv[2]
    no_sc = "--no-self-collision" in sys.argv
    if not os.path.exists(os.path.join(NOVA, "controller", "lab_retarget.py")):
        print(json.dumps({"error": "NOVA checkout not found next to this repository"}))
        sys.exit(2)
    import numpy as np
    from lab_retarget import LabRetargeter, JOINTS, TORSO_CMD_RANGE

    frames = json.load(open(src))
    rt = LabRetargeter(log=lambda s: None)
    rt.offline_clock = True
    rt.set_mode({"method": "retarget", "scope": "local"})
    kr = rt.kr
    home = {u: float(kr.data.qpos[kr.qnames[u]]) for u in JOINTS}
    meas = dict(home, _lift=(kr.lift_home, 0.0))
    rt.feed(frames[0]["p"])
    rt._seed(meas)
    kr = rt.kr
    eng = kr.eng
    if no_sc:
        eng.set_self_collision(False)
    # count the barrier rows the projector sees each frame (0 = its fast path)
    rows = {"n": 0}
    if eng.projector is not None:
        g = eng.projector.guard
        orig = g.rows

        def spy(*a, **k):
            G, h = orig(*a, **k)
            rows["n"] += len(G)
            return G, h
        g.rows = spy
    wrists = ("right_wrist", "left_wrist")
    out = []
    for f in frames:
        p, t = f["p"], float(f["t"])
        p = dict(p, timestamp=t)
        rows["n"] = 0
        rt.step_frame(p, meas)
        d = rt.d
        q = {u: float(d.qpos[rt._qa[u]]) for u in JOINTS}
        q["torso"] = min(TORSO_CMD_RANGE[1], max(TORSO_CMD_RANGE[0], q["torso"]))
        for w in wrists:
            q[w] = float(d.qpos[kr.qnames[w]])
        out.append({"q": q, "targets": {s: v.tolist() for s, v in kr._tgt.items()},
                    "qp_rows": rows["n"], "engaged": eng._world_ref is not None})
    json.dump({"frames": out, "selfcol": bool(eng.self_collision)}, open(dst, "w"))


if __name__ == "__main__":
    main()
