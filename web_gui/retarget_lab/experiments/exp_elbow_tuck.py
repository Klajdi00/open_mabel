"""Elbow tuck on a REAL Vision Pro take: how far the robot's upper arm is held out from the body.

Per frame and side: robot upper-arm ABDUCTION = the angle of shoulder->elbow out of the body's sagittal plane
(outward +, deg), the operator's the same from the engine's reconstructed elbow (_elbow_ema, through the body
map M), and the robot elbow's horizontal distance from the torso midline [cm]. Binned by operator extension
rho. Run with the shape gate on and off (and any env knobs) to compare.

    cd web_gui && MABEL_RTL_ROBOT=nova python -m retarget_lab.experiments.exp_elbow_tuck
"""
import os
import subprocess
import sys

import numpy as np

TAKE = os.environ.get("NOVA_LAB_TAKE", "20261015_103700_Oct_15th_test1")
BINS = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 1.01)]


def run():
    from retarget_lab.server import episodes
    from retarget_lab.server.worker import KinematicRetargeter, pose_from_frame
    _, rows = episodes.load(TAKE)
    fr = [r for r in rows if r.get("kind") == "wire"]
    kr = KinematicRetargeter("ours"); kr.apply_settings({"scope": "local", "robot": "nova"})
    eng, d = kr.eng, kr.data
    acc = {b: {"rob": [], "hum": [], "mid": []} for b in BINS}
    for f in fr:
        kr.step(pose_from_frame(f), f["t"], collide=False, frame=f)
        M = getattr(eng, "_frame_M", None)
        pl = eng._payload(pose_from_frame(f)) or {}
        if M is None:
            continue
        sR, sL = d.xpos[eng.arm["right"].shoulder_bid], d.xpos[eng.arm["left"].shoulder_bid]
        mid = 0.5 * (sR + sL)
        for side in ("right", "left"):
            ema = getattr(eng, "_elbow_ema", {}).get(side)
            mats = pl.get(f"{side}_hand_mats")
            if ema is None or not mats or not eng.shoulder.ready(side):
                continue
            s_h = eng.shoulder.shoulder_world(side)
            w = np.asarray(mats[0], float).reshape(4, 4)[:3, 3]
            rho = float(np.linalg.norm(w - s_h)) / max(eng.shoulder.arm_length(side), 1e-6)
            s = d.xpos[eng.arm[side].shoulder_bid]
            out = s - (sL if side == "right" else sR); out[2] = 0.0; out /= np.linalg.norm(out)
            e = d.xpos[eng.arm[side].elbow_bid]
            ab = lambda v: float(np.degrees(np.arcsin(np.clip(v @ out / np.linalg.norm(v), -1, 1))))
            hv = M @ (ema - s_h)
            for b in BINS:
                if b[0] <= rho < b[1]:
                    acc[b]["rob"].append(ab(e - s)); acc[b]["hum"].append(ab(hv))
                    acc[b]["mid"].append(float(np.linalg.norm((e - mid)[:2])) * 100)
    for (a, b), v in acc.items():
        if v["rob"]:
            print(f"  rho {a:.2f}-{b:.2f} (n {len(v['rob']):5d}): upper-arm abduction robot {np.mean(v['rob']):5.1f} deg, "
                  f"operator {np.mean(v['hum']):5.1f} deg | elbow from midline {np.mean(v['mid']):4.1f} cm "
                  f"(p5 {np.percentile(v['mid'], 5):4.1f})")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--one":
        run()
    else:
        for g in ("0", "1"):
            print(f"shape gate {g}:", flush=True)
            subprocess.run([sys.executable, "-m", "retarget_lab.experiments.exp_elbow_tuck", "--one"],
                           env=dict(os.environ, MABEL_ARM_SHAPE_GATE=g), check=True)
