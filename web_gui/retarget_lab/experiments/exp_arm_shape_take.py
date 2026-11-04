"""The shape gate on a REAL Vision Pro take: robot arm direction vs the operator's (shoulder -> wrist, through
the engine's body map M) per frame, binned by the operator's extension rho, and the palm's miss, with the gate
on and off.

    cd web_gui && MABEL_RTL_ROBOT=nova python -m retarget_lab.experiments.exp_arm_shape_take
"""
import os
import subprocess
import sys

import numpy as np

TAKE = os.environ.get("NOVA_LAB_TAKE", "20261015_103700_Oct_15th_test1")
BINS = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 0.95), (0.95, 1.01)]


def run():
    from retarget_lab.server import episodes
    from retarget_lab.server.worker import KinematicRetargeter, pose_from_frame
    _, rows = episodes.load(TAKE)
    fr = [r for r in rows if r.get("kind") == "wire"]
    kr = KinematicRetargeter("ours"); kr.apply_settings({"scope": "local", "robot": "nova"})
    eng, d = kr.eng, kr.data
    out = {b: [] for b in BINS}
    miss = []
    for f in fr:
        kr.step(pose_from_frame(f), f["t"], collide=False, frame=f)
        M = getattr(eng, "_frame_M", None)
        for side in ("right", "left"):
            if M is None or not eng.shoulder.ready(side) or side not in kr._tgt:
                continue
            mats = (eng._payload(pose_from_frame(f)) or {}).get(f"{side}_hand_mats")
            if not mats:
                continue
            w = np.asarray(mats[0], float).reshape(4, 4)[:3, 3]
            s_h = eng.shoulder.shoulder_world(side)
            L = max(eng.shoulder.arm_length(side), 1e-6)
            rho = float(np.linalg.norm(w - s_h)) / L
            uh = M @ ((w - s_h) / np.linalg.norm(w - s_h))
            arm = eng.arm[side]
            s, pm = d.xpos[arm.shoulder_bid], d.xpos[arm.palm_bid]
            ur = (pm - s) / np.linalg.norm(pm - s)
            ang = float(np.degrees(np.arccos(np.clip(ur @ (uh / np.linalg.norm(uh)), -1, 1))))
            for b in BINS:
                if b[0] <= rho < b[1]:
                    out[b].append(ang)
            miss.append(float(np.linalg.norm(kr._tgt[side] - pm)) * 100)
    print(" | ".join(f"rho {a:.2f}-{b:.2f}: {np.mean(v):5.1f} deg (n {len(v)})" if v else f"rho {a:.2f}-{b:.2f}: -"
                     for (a, b), v in out.items()), f"| palm miss mean {np.mean(miss):.1f} cm, p95 {np.percentile(miss, 95):.1f} cm")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--one":
        run()
    else:
        for g in ("0", "1"):
            print(f"shape gate {g}:", flush=True)
            subprocess.run([sys.executable, "-m", "retarget_lab.experiments.exp_arm_shape_take", "--one"],
                           env=dict(os.environ, MABEL_ARM_SHAPE_GATE=g), check=True)
