"""End-effector toggle on a REAL Vision Pro take: ORCA hand v1 vs the OpenArm v1 gripper (teleop_engine/end_effector.py).

Per end effector, the whole take through the lab engine (robot NOVA, upper body):
  track   |arm target - tracked point| [cm] (ORCA wrist anchor / wrist centre J5xJ6xJ7), mean / p95 / max
  step    largest per-frame arm command step [rad] (max over joints), p99 / max
  nan     frames with a non-finite joint
  solve   ms per frame, p50 / p95
and, with the gripper, the hand-closure estimate against the engine's thumb-index pinch distance: the opening
while pinched (distance < 2.5 cm) vs not, how often a pinch leaves the gripper more than 1 cm open, and the
correlation of the opening with that distance.

    cd web_gui && MABEL_RTL_ROBOT=nova python -m retarget_lab.experiments.exp_gripper
"""
import os

import numpy as np

from retarget_lab.server import episodes
from retarget_lab.server.worker import KinematicRetargeter, pose_from_frame

TAKES = [t for t in os.environ.get("EXP_TAKES", "20261015_103700_Oct_15th_test1").split(",") if t]
ARMS = [f"{s}_arm_{i}" for s in ("left", "right") for i in range(1, 8)]


def run(take, ee):
    _, rows = episodes.load(take)
    fr = [r for r in rows if r.get("kind") == "wire"]
    kr = KinematicRetargeter("ours")
    st = kr.apply_settings({"scope": "local", "robot": "nova", "ee": ee})
    assert st["ee"] == ee, st
    eng, d = kr.eng, kr.data
    track, steps, solve, nan = [], [], [], 0
    op_pinch, op_free, miss_pinch, pairs = [], [], 0, []
    prev = None
    for f in fr:
        rec = kr.step(pose_from_frame(f), f["t"], frame=f)
        q = np.array([rec["q"][k] for k in ARMS])
        if not np.all(np.isfinite(q)):
            nan += 1
            continue
        if prev is not None:
            steps.append(float(np.abs(q - prev).max()))
        prev = q
        solve.append(rec["m"].get("solve_ms", np.nan) if isinstance(rec["m"].get("solve_ms"), float) else np.nan)
        for side in ("right", "left"):
            tg = kr._tgt.get(side)
            if tg is not None:
                track.append(float(np.linalg.norm(tg - eng.arm[side]._ee_point(d)[0])) * 100)
            if ee == "gripper":
                pl = eng._payload(pose_from_frame(f)) or {}
                pd = pl.get(f"{side}_pinch")                    # thumb-index distance [m], the engine's
                if pl.get(f"{side}_joints") and pd is not None:
                    pinch = float(pd) < 0.025
                    g = eng.gripper_aperture[side].open * 1000
                    (op_pinch if pinch else op_free).append(g)
                    miss_pinch += pinch and g > 10.0
                    pairs.append((float(pd) * 1000, g))
    t = np.array(track); s = np.array(steps); sv = np.array([x for x in solve if np.isfinite(x)])
    out = (f"{take[:24]:24s} {ee:8s} track {t.mean():5.1f}/{np.percentile(t, 95):5.1f}/{t.max():5.1f} cm | "
           f"step p99 {np.percentile(s, 99):.3f} max {s.max():.3f} rad | nan {nan} | "
           f"solve {np.median(sv):.1f}/{np.percentile(sv, 95):.1f} ms")
    if ee == "gripper":
        out += (f"\n{'':33s} opening while pinching {np.mean(op_pinch) if op_pinch else float('nan'):5.1f} mm "
                f"(n {len(op_pinch)}, >10 mm open {miss_pinch}), not pinching {np.mean(op_free):5.1f} mm "
                f"(p10 {np.percentile(op_free, 10):.1f}, p90 {np.percentile(op_free, 90):.1f}, n {len(op_free)}); "
                f"corr(opening, pinch distance) {np.corrcoef(np.array(pairs).T)[0, 1]:.3f}")
    print(out, flush=True)


if __name__ == "__main__":
    for take in TAKES:
        for ee in ("orca", "gripper"):
            run(take, ee)
