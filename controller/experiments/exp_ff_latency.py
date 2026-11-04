#!/usr/bin/env python3
"""exp_ff_latency.py — verify the gravity-FF real-timeness fix ("step-like when
pushed in stiff", 2026-07-30).

Reproduces the gateway's exact FF pipeline at the 50 Hz control tick against a
hand-push pose trajectory (0.5 rad/s ramps + direction reversals on the gravity-
loaded joints), and measures the streamed feedforward against the TRUE τ_g(q(t)):

  OLD  clock-cached eval every 0.25 s + 10 N·m/s output slew
  NEW  pose-triggered eval (GravityFFScheduler: min 0.04 s / eps 0.01 rad /
       max 0.25 s) + 40 N·m/s slew

Also reports each pipeline's eval rate (the CPU story) and the worst single-tick
FF jump (the "step" the operator feels through stiff gains).
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
REPO = os.path.dirname(os.path.dirname(HERE))
MJCF = os.path.join(REPO, "simulation", "mabel_mujoco", "models", "mabel_full.xml")

from mabel.arm_control import (ArmGravityCompensator, GravityFFScheduler,
                               HandsConfig)

DT = 0.02                      # 50 Hz gateway tick
JOINTS = [f"l_j{k}" for k in range(1, 8)] + [f"r_j{k}" for k in range(1, 8)]
PUSHED = ["l_j2", "l_j4"]      # the gravity-loaded joints a chest-high push moves


def push_pose(t):
    """A hand push: 0.5 rad/s ramp out for 1.2 s, hold 0.5 s, ramp back, with a
    small 2 Hz tremor — the pose trajectory the FF must track in real time."""
    pose = {j: 0.0 for j in JOINTS}
    for j in PUSHED:
        if t < 1.2:
            base = 0.5 * t
        elif t < 1.7:
            base = 0.6
        elif t < 2.9:
            base = 0.6 - 0.5 * (t - 1.7)
        else:
            base = 0.0
        pose[j] = base + 0.01 * np.sin(2 * np.pi * 2.0 * t)
    return pose


class Pipeline:
    """The gateway's cache+slew, parameterized OLD/NEW."""

    def __init__(self, comp, sched, slew):
        self.comp, self.sched, self.slew = comp, sched, slew
        self.cache = None
        self.smooth = {}
        self.evals = 0

    def tick(self, now, pose):
        if self.cache is None or self.sched.due(now, pose):
            self.cache = self.comp.tff(pose, frame="model")
            self.sched.mark(now, pose)
            self.evals += 1
        step = self.slew * DT
        for j, t in self.cache.items():
            cur = self.smooth.get(j, t)
            d = t - cur
            self.smooth[j] = t if abs(d) <= step else cur + (step if d > 0 else -step)
        return dict(self.smooth)


def main():
    comp = ArmGravityCompensator(MJCF, {}, {}, hands=HandsConfig.NONE,
                                 log=lambda *a, **k: None)
    assert comp.available
    # OLD = clock-only scheduling: pose_eps=inf disables the motion trigger
    old = Pipeline(comp, GravityFFScheduler(0.25, 0.25, float("inf")), 10.0)
    new = Pipeline(comp, GravityFFScheduler(0.04, 0.25, 0.01), 40.0)

    T = 3.2
    errs = {"old": [], "new": []}
    jumps = {"old": 0.0, "new": 0.0}
    prev = {"old": {}, "new": {}}
    for k in range(int(T / DT)):
        t = k * DT
        pose = push_pose(t)
        truth = comp.tff(pose, frame="model")
        for name, pl in (("old", old), ("new", new)):
            ff = pl.tick(t, pose)
            e = max(abs(ff.get(j, 0.0) - truth.get(j, 0.0)) for j in PUSHED)
            errs[name].append(e)
            for j in PUSHED:
                d = abs(ff.get(j, 0.0) - prev[name].get(j, ff.get(j, 0.0)))
                jumps[name] = max(jumps[name], d)
            prev[name] = ff

    print(f"{'pipeline':>8} | {'peak FF err':>11} | {'rms FF err':>10} | "
          f"{'worst tick jump':>15} | {'evals':>5} ({T:.1f} s)")
    ok = True
    for name in ("old", "new"):
        e = np.array(errs[name])
        print(f"{name:>8} | {e.max():9.2f} Nm | {np.sqrt((e**2).mean()):8.2f} Nm | "
              f"{jumps[name]:13.2f} Nm | {(('old', 'new').index(name) and new or old).evals:5d}")
    e_old, e_new = np.array(errs["old"]), np.array(errs["new"])
    ok &= e_new.max() < 0.35
    ok &= e_new.max() < 0.4 * e_old.max()
    ok &= jumps["new"] <= 40.0 * DT + 1e-9           # slew-bounded, no eval spikes
    # idle economy: at rest (constant pose) the NEW pipeline must not evaluate
    # faster than the old drift-safety clock
    idle = Pipeline(comp, GravityFFScheduler(0.04, 0.25, 0.01), 40.0)
    pose0 = push_pose(0.0)
    for k in range(int(2.0 / DT)):
        idle.tick(k * DT, pose0)
    print(f"idle evals over 2 s: {idle.evals} (≤ {int(2.0/0.25)+1} = the 4 Hz clock)")
    ok &= idle.evals <= int(2.0 / 0.25) + 1
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
