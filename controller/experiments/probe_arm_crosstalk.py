#!/usr/bin/env python3
"""Which stage couples the two arms? (bimanual cross-talk bisection)

E1 measured the ROBOT's right arm tracking the OPERATOR's LEFT hand at
r = -0.85 while tracking its own at +0.58, with the two operator hands
themselves uncorrelated (+0.11) — i.e. real cross-coupling, not a metric
artifact. This bisects the candidate stages by toggling one at a time:

  body-frame map   ARM_BODY_FRAME_MAP — the arm target is decomposed in the
                   operator BODY frame, whose axes come from the shoulder
                   estimator, which is fed BOTH wrists. If one hand's motion
                   rotates the estimated body frame, it rotates the OTHER arm's
                   target: coupling by construction.
  body yaw         the same question for the base-yaw estimator.
  bimanual blend   the near-contact 1:1 correction + separation floor.
  self-collision   the projector solves ONE QP over both arms' 14 joints.

    python3 controller/experiments/probe_arm_crosstalk.py
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)


def run_case(env, label):
    for k, v in env.items():
        os.environ[k] = v
    for m in [m for m in list(sys.modules) if m.startswith("mabel")]:
        del sys.modules[m]
    import importlib
    h = importlib.import_module("exp_vp_bimanual_latency")
    importlib.reload(h)

    op = h.SyntheticVPOperator(lambda s, t: h.traj_bimanual(s, t))
    rig = h.Rig(frame_hz=30.0, self_collision=(env.get("SC", "1") == "1"))
    rec = rig.run(op, 8.0)
    own_r = h.track_corr(rec["op_r"][:, 0], rec["wrist_r"][:, 0])
    own_l = h.track_corr(rec["op_l"][:, 0], rec["wrist_l"][:, 0])
    x_rl = h.track_corr(rec["op_l"][:, 0], rec["wrist_r"][:, 0])
    x_lr = h.track_corr(rec["op_r"][:, 0], rec["wrist_l"][:, 0])
    ctrl = h.track_corr(rec["op_r"][:, 0], rec["op_l"][:, 0])
    print(f"{label:24s} own R {own_r:+.3f} L {own_l:+.3f} | "
          f"CROSS R<-L {x_rl:+.3f} L<-R {x_lr:+.3f} | operator-corr {ctrl:+.3f}")
    return abs(x_rl)


def main():
    base = {"MABEL_ARM_BIMANUAL": "1", "MABEL_ARM_HAND_CLEAR": "0.18",
            "MABEL_ARM_BODY_FRAME": "1", "SC": "1"}
    cases = [
        (dict(base), "baseline"),
        (dict(base, MABEL_ARM_BODY_FRAME="0"), "body-frame map OFF"),
        (dict(base, MABEL_ARM_BIMANUAL="0", MABEL_ARM_HAND_CLEAR="0"), "bimanual+floor OFF"),
        (dict(base, SC="0"), "self-collision OFF"),
        (dict(base, MABEL_ARM_BODY_FRAME="0", MABEL_ARM_BIMANUAL="0",
              MABEL_ARM_HAND_CLEAR="0", SC="0"), "ALL OFF"),
    ]
    print("\nBIMANUAL CROSS-TALK BISECTION  (|CROSS| near 0 = arms independent)\n")
    for env, label in cases:
        run_case(env, label)
    print()


if __name__ == "__main__":
    main()
