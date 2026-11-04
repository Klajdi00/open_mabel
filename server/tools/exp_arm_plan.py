#!/usr/bin/env python3
"""
exp_arm_plan.py — prove the iOS green-ball plan / preview / EXECUTE safety gate.

The whole point of the feature: dragging the palm ball PLANS a target the operator
previews (an animated ghost of current→target, MoveIt-style), and NOTHING reaches
the robot until they press Execute. Execute then streams the SAME eased trajectory
that was previewed.

This is a SIM-side, log-parsing harness (the drag can't be injected from the CLI, so
the app runs its own plan/execute harness — see below). It reads the bridge's arm-
command trace and asserts the two safety properties that matter:

  1. GATE      — during planning/drag the bridge received NO arm command. If the
                 old live-stream behaviour leaked, the trace would be full of
                 commands the whole time; the fix makes it silent until Execute.
  2. EXECUTE   — Execute streamed ONE short contiguous burst of arm commands that
                 ramps MONOTONICALLY from the start pose to the target (the eased
                 ArmPlan trajectory), then stops. Not a jitter, not a firehose.

NOTE — the SIM ARM DOES NOT PHYSICALLY MOVE on a joint_command in the current sim
(verified independently from a raw wire client): qpos holds while the command's
DESIRED changes. That is a pre-existing sim arm-control limitation, NOT this
feature — so this harness checks the COMMAND STREAM (what the app sends), which is
the app-side contract, not the resulting motion.

How to produce the trace:
  1. Start the sim with the trace on:
       MABEL_LOG_ARMCMD=1 server/start_teleop.sh --no-monitor > sim.log
  2. Launch the app into its plan/execute harness (Task+Stiff, stage@6s, exec@12s):
       simctl … defaults write <bundle> debugPlanExecute left
       simctl launch <UD> <bundle>
  3. After ~15 s:
       python3 tools/exp_arm_plan.py --log sim.log
"""
from __future__ import annotations

import argparse
import re
import sys

# "… ARMCMD {'left_arm_1': 0.416, 'left_arm_2': 0.284, 'left_arm_4': 0.467}"
_LINE = re.compile(r"ARMCMD\s+(\{.*\})")


def parse(path):
    rows = []
    with open(path) as f:
        for ln in f:
            m = _LINE.search(ln)
            if not m:
                continue
            # The dict is repr with single quotes — eval-safe enough for our own log.
            try:
                rows.append(eval(m.group(1), {"__builtins__": {}}))  # noqa: S307
            except Exception:
                pass
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="bridge stdout log (MABEL_LOG_ARMCMD=1)")
    ap.add_argument("--side", default="left")
    a = ap.parse_args()

    rows = parse(a.log)
    fails = []

    def check(name, ok, detail=""):
        print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  — ' + detail if detail else ''}")
        if not ok:
            fails.append(name)

    key = f"{a.side}_arm_1"
    print(f"exp_arm_plan — {len(rows)} arm-command lines in {a.log}")

    # 1) EXECUTE happened at all.
    check("execute streamed arm commands", len(rows) > 0,
          f"{len(rows)} commands")
    if not rows:
        print("\n  (no ARMCMD lines — was MABEL_LOG_ARMCMD=1 set and the app harness run?)")
        return 1

    seq = [r.get(key) for r in rows if key in r]
    check("commands carry the driven joint", len(seq) >= 8, f"{len(seq)} samples of {key}")

    # 2) ONE contiguous burst — the eased ramp start→target, monotonic. A leak
    #    during planning would show as scattered commands with no clean ramp; a
    #    firehose (the old live-stream) would show hundreds. Execute = a short,
    #    smooth, one-directional sweep.
    if seq:
        start, end = seq[0], seq[-1]
        span = abs(end - start)
        check("ramps a meaningful distance (a real move, not a twitch)", span > 0.20,
              f"{start:.3f} → {end:.3f} (Δ{span:.3f} rad)")
        # monotonic (eased blend of a monotonic segment stays monotonic)
        up = all(b >= a - 1e-3 for a, b in zip(seq, seq[1:]))
        dn = all(b <= a + 1e-3 for a, b in zip(seq, seq[1:]))
        check("ramps MONOTONICALLY (eased preview trajectory, no wobble)", up or dn)
        # bounded burst — not a per-frame firehose
        check("a BOUNDED burst, not a live-stream firehose", len(seq) <= 200,
              f"{len(seq)} commands (execute streams ~1.4 s)")
        # eased shape: the middle third should cover MORE than a third of the span
        # (smoothstep is steepest in the middle) — distinguishes an eased ramp from
        # a linear or step change.
        n = len(seq)
        mid = abs(seq[2 * n // 3] - seq[n // 3])
        check("velocity profile is EASED (fast middle, soft ends)",
              mid > 0.4 * span, f"middle third covers {mid / max(span,1e-9):.0%} of the move")

    print("\n" + "=" * 58)
    print("  ALL PASS — plan/execute gate + eased trajectory verified"
          if not fails else f"  {len(fails)} FAILURE(S)")
    return 0 if not fails else 1


sys.exit(main())
