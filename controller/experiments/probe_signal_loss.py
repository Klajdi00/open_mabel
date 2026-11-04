#!/usr/bin/env python3
"""What happens when the Vision Pro stream DIES mid-teleop?

THE OPERATOR'S QUESTION (2026-08-01)
  "is there any safety feature where if the Vision Pro pose streaming is lost,
   the robot won't freak out — it moves back to the home position over ~5
   seconds, slowly but at a reasonable speed?"

There is (`EngagementSupervisor`), but a safety claim read off constant names is
not a safety claim. This measures the real behaviour on the real engine:

  detect    how long after the last frame before the robot reacts at all
            (SAFE_STREAM_TIMEOUT / SAFE_SIGNAL_TIMEOUT)
  retreat   how long the ease to HOME actually takes (SAFE_RETREAT_TIME)
  speed     peak commanded joint speed during the retreat — "slow but
            reasonable" has to be a number, and a lurch would show up here
  base      the base must be frozen, not coasting on its last command
  re-engage that recovery is also eased, not a snap back to the live pose

    python3 controller/experiments/probe_signal_loss.py
"""
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
import exp_vp_bimanual_latency as h                              # noqa: E402
from mabel.teleop_engine import hyperparams as HP                # noqa: E402

ARM = [f"right_arm_{i}" for i in range(1, 8)]


def run(live_s=4.0, dead_s=7.0, back_s=4.0, frame_hz=30.0):
    """REAL-TIME paced. The supervisor's ease clock is the WALL clock
    (`time.time()`), not the pose timestamps, so a harness that replays sim
    seconds as fast as it can compute them measures a retreat in units the
    supervisor never used — the first version reported a 12.96 s ease and a
    968 deg/s peak, both artefacts of sim time and wall time being decoupled.
    Pacing the loop makes the two agree, so seconds mean seconds."""
    op = h.SyntheticVPOperator(lambda s, t: h.traj_bimanual(s, t))
    rig = h.Rig(frame_hz=frame_hz)
    rig.teleop.set_control_spec(method="retarget", control_type="impedance",
                                region="whole_body", scope="global")
    acts = [rig.model.actuator(f"Right Arm {i}").id for i in range(1, 8)]
    rec = {"t": [], "q": [], "base": [], "state": []}
    t, nxt = 0.0, 0.0
    t0 = time.time()
    total = live_s + dead_s + back_s
    while True:
        # PHASES ON THE WALL CLOCK, not on sim time. The engine solve costs more
        # than one control tick, so sim time falls behind wall time and a cut
        # scheduled in sim seconds lands at a different wall instant than the
        # supervisor's ease clock uses — which is how the first version reported
        # "detect 0.02 s" against a declared 0.4 s timeout. Everything the
        # supervisor does is wall-clocked, so the experiment must be too.
        w = time.time() - t0
        if w >= total:
            break
        alive = (w < live_s) or (w >= live_s + dead_s)     # the stream DIES in between
        pose = None
        if alive and w >= nxt:
            pose = op.frame(w)
            nxt += 1.0 / frame_hz
        if not alive:
            nxt = w                                        # resume immediately on return
        rig.teleop.update(pose, None, h.DT)
        rig._plant()
        rec["t"].append(w)
        rec["q"].append([float(rig.data.ctrl[a]) for a in acts])
        rec["base"].append(max(abs(v) for v in rig.eng.wbc.base_tgt))
        rec["state"].append(str(rig.eng.mode))
    return {k: (np.asarray(v, float) if k != "state" else v) for k, v in rec.items()}, live_s


def main():
    print("\nVISION PRO SIGNAL LOSS — measured on the real engine")
    print(f"declared: stream timeout {HP.SAFE_STREAM_TIMEOUT}s · signal timeout "
          f"{HP.SAFE_SIGNAL_TIMEOUT}s · retreat {HP.SAFE_RETREAT_TIME}s · "
          f"re-engage {HP.SAFE_ENGAGE_TIME}s\n")
    rec, cut = run()
    t, q, base = rec["t"], rec["q"], rec["base"]
    # TRUE local derivative. The loop does not tick evenly (a frame that solves
    # the arm IK costs ~15 ms, a dark tick ~0.5 ms), so dividing every joint
    # step by a single MEDIAN dt turns each slow sample into a fake spike — that
    # is where the earlier "1048 deg/s" came from, not from the robot.
    dts = np.diff(t)
    dts[dts <= 0] = np.nan
    speed = np.abs(np.diff(q, axis=0)).max(axis=1) / dts        # rad/s, worst joint

    # motion AFTER the stream dies
    # The retreat window must END when the stream comes back, or the re-engage
    # transient lands inside it and both phases report the same peak — which is
    # exactly what the first version did, blaming the retreat for a spike that
    # belongs to recovery.
    DEAD = 7.0
    post = (t >= cut) & (t < cut + DEAD)
    post_d = (t[:-1] >= cut) & (t[:-1] < cut + DEAD)
    q_at_cut = q[int(np.searchsorted(t, cut))]
    moved = np.abs(q - q_at_cut).max(axis=1)

    start = next((tt for tt, m in zip(t, moved) if tt >= cut and m > 0.01), None)
    # "arrived" = the command has stopped changing meaningfully
    tail = [tt for tt, s in zip(t[:-1], speed) if cut <= tt < cut + DEAD and np.isfinite(s) and s > 0.02]
    done = (tail[-1] if tail else None)

    print(f"  detect   robot first moves {start - cut:.2f} s after the last frame"
          if start else "  detect   NO retreat motion at all")
    if start and done:
        print(f"  retreat  motion ends {done - cut:.2f} s after the last frame "
              f"(ease itself {done - start:.2f} s)")
    print(f"  speed    peak joint speed during retreat "
          f"{float(np.nanmax(speed[post_d])):.3f} rad/s "
          f"({np.degrees(float(np.nanmax(speed[post_d]))):.0f} deg/s)")
    print(f"  travel   worst joint moved {np.degrees(float(np.max(moved[post]))):.1f} deg "
          f"toward HOME")
    print(f"  base     max |command| while dark {float(np.max(base[post])):.4f} m/s "
          f"({'frozen' if float(np.max(base[post])) < 1e-9 else 'STILL MOVING'})")

    # re-engage
    ret = t[:-1] >= (cut + 7.0)
    if ret.any():
        sp_back = speed[ret]
        print(f"  recover  peak joint speed on re-acquire "
              f"{float(np.nanmax(sp_back)):.3f} rad/s "
              f"({np.degrees(float(np.nanmax(sp_back))):.0f} deg/s)")
    states = []
    for s in rec["state"]:
        if not states or states[-1] != s:
            states.append(s)
    print(f"  states   {' -> '.join(states[:6])}")
    print()


if __name__ == "__main__":
    main()
