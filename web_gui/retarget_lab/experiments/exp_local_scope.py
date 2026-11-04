"""Upper body (LOCAL) vs whole body (GLOBAL) on a synthetic operator: walk forward vs a hip lean.

Upper body must track the operator LOCALLY: walking (head + hands translate together, no head drop)
must not bend the torso, move the lift or move the base; a real hip-pivot lean must still bend
the torso, and the lift holds even when the operator rises. Whole body tracks globally: the walk moves the base (and the lift follows head height).

    cd web_gui && MABEL_RTL_ROBOT=nova python -m retarget_lab.experiments.exp_local_scope
"""
import gzip
import json

import numpy as np

from retarget_lab.server import config as CFG
from retarget_lab.server.worker import KinematicRetargeter, pose_from_frame
from retarget_lab.tests.test_wholebody import CLIPS, _backward, _shift, _walk_payload

L = 0.70                                                   # hip -> eyes [m], hyperparams TORSO_LEAN_PIVOT


def _lean_payload(p, th, fwd):
    """Hip-pivot bend by th about a hip L below the head: head + hands move fwd L sin th, down L(1-cos th)."""
    return _walk_payload(p, fwd * L * np.sin(th) + np.array([0.0, -L * (1 - np.cos(th)), 0.0]))


def run(scope, motion, n=360):
    p0 = json.load(gzip.open(CLIPS[0]))["payloads"][0]
    kr = KinematicRetargeter("ours"); kr.apply_settings({"scope": scope, "robot": "nova"})
    fwd = -_backward(p0)
    rec0 = None
    for i in range(n):
        u = min(1.0, max(0.0, (i - 60) / 180))             # 1 s still, 3 s ramp, 1 s hold
        if motion == "walk":
            bob = np.array([0.0, 0.02 * np.sin(2 * np.pi * 1.8 * i / 60) * (0 < u < 1), 0.0])   # 2 cm gait bob
            p = _walk_payload(p0, 0.6 * u * fwd + bob)
        elif motion == "rise":
            p = _walk_payload(p0, np.array([0.0, 0.15 * u, 0.0]))
        else:
            p = _lean_payload(p0, np.radians(30) * u, fwd)
        rec = kr.step(pose_from_frame({"kind": "avp", "p": p}), i / 60.0)
        if i == 59:
            rec0 = rec
    q0, q = rec0["q"], rec["q"]
    arms = max(abs(q[k] - q0[k]) for k in q if "_arm_" in k)
    lift = sum(q.get(k, 0) - q0.get(k, 0) for k in ("lift_lower", "lift_upper"))
    return dict(torso=q["torso"] - q0["torso"], lift=lift, base=float(np.hypot(rec["base"]["x"], rec["base"]["y"])), arms=arms)


if __name__ == "__main__":
    print(f"{'scope':7s} {'motion':12s} {'torso [rad]':>11s} {'lift [m]':>9s} {'base [m]':>9s} {'max arm dq':>10s}")
    for scope in ("local", "global"):
        for motion in ("walk", "lean", "rise"):
            r = run(scope, motion)
            name = {"walk": "walk 0.6 m", "lean": "lean 30 deg", "rise": "rise 0.15 m"}[motion]
            print(f"{scope:7s} {name:12s} {r['torso']:+11.3f} {r['lift']:+9.3f} {r['base']:9.3f} {r['arms']:10.3f}")
