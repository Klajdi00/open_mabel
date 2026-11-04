#!/usr/bin/env python3
"""MABEL gesture demo — cycle named poses on the ORCA hand.

Rewritten on the lean :mod:`orcahand` API (``set_joint_percent`` /
``set_named_position``); poses come from ``orcahand.poses.POSES``.

    python control/demo_gestures.py                      # cycle the demo sequence
    python control/demo_gestures.py --mode random
    python control/demo_gestures.py --mode wave          # wave / ripple flourishes
    python control/demo_gestures.py --pose fist pinch point
    python control/demo_gestures.py --loops 0            # run forever
    python control/demo_gestures.py --list
    python control/demo_gestures.py --mock               # no hardware
"""
from __future__ import annotations

import argparse
import random
import sys
import time

from utils.hand_config import (BOLD, CYAN, GREEN, RED, RESET, YELLOW, ALL_FINGERS,
                               FINGERS, POSES, connect_hand)

# name, hold_time (s), transition_time (s)
DEMO_SEQUENCE = [
    ("flat", 1.5, 1.2), ("fist", 1.5, 1.2), ("flat", 0.8, 0.8),
    ("point", 1.5, 1.0), ("peace", 1.5, 1.0), ("ok", 1.5, 1.0),
    ("pinch", 1.5, 1.0), ("thumbs_up", 1.5, 1.0), ("rock", 1.5, 1.0),
    ("spider", 1.5, 1.0), ("flat", 1.0, 1.2),
]


def _ease(t: float) -> float:
    return 3 * t * t - 2 * t * t * t  # smoothstep


class GesturePlayer:
    """Plays poses through an :class:`orcahand.OrcaHand` with smooth ramps."""

    def __init__(self, hand):
        self.hand = hand
        self.names = hand.joint_names
        self._cur = {j: 50.0 for j in self.names}  # last commanded percent

    def _percent_now(self) -> dict:
        return dict(self._cur)

    def go(self, pose_name: str) -> None:
        pose = POSES.get(pose_name, {})
        for j, p in pose.items():
            if j in self._cur:
                self._cur[j] = float(p)
        self.hand.set_joint_percent({j: self._cur[j] for j in pose if j in self._cur})

    def transition(self, pose_name: str, duration: float = 1.0, steps: int = 30) -> None:
        target = {j: float(p) for j, p in POSES.get(pose_name, {}).items() if j in self._cur}
        start = self._percent_now()
        for i in range(1, steps + 1):
            a = _ease(i / steps)
            cmd = {j: start[j] + (tp - start[j]) * a for j, tp in target.items()}
            self.hand.set_joint_percent(cmd)
            time.sleep(max(0.0, duration / steps))
        self._cur.update(target)

    def wave(self, cycles: int = 2, period: float = 1.2) -> None:
        fingers = [f for f in ALL_FINGERS if f != "thumb"]
        t0 = time.time()
        while time.time() - t0 < cycles * period:
            ph = (time.time() - t0) / period
            cmd = {}
            for k, fg in enumerate(fingers):
                amp = 35 * (0.5 + 0.5 * _wave(ph - k * 0.25))
                for j in FINGERS[fg]:
                    if j.endswith(("mcp", "pip")):
                        cmd[j] = 30 + amp
            self.hand.set_joint_percent(cmd)
            self._cur.update(cmd)
            time.sleep(0.02)

    def ripple(self, cycles: int = 2, period: float = 0.8) -> None:
        order = ["index", "middle", "ring", "pinky"]
        for _ in range(cycles):
            for fg in order:
                self.hand.set_joint_percent({j: 80 for j in FINGERS[fg] if not j.endswith("abd")})
                time.sleep(period / len(order))
            for fg in order:
                self.hand.set_joint_percent({j: 15 for j in FINGERS[fg] if not j.endswith("abd")})
                time.sleep(period / len(order))
        self._cur.update({j: 15 for fg in order for j in FINGERS[fg]})


def _wave(x: float) -> float:
    import math
    return math.sin(2 * math.pi * x)


def run_sequence(player: GesturePlayer, sequence, speed_mult: float) -> None:
    for name, hold, trans in sequence:
        print(f"  → {BOLD}{name:<14}{RESET}", end=" ", flush=True)
        player.transition(name, duration=trans / speed_mult)
        time.sleep(hold / speed_mult)
        print(f"{GREEN}✓{RESET}")


def main() -> int:
    ap = argparse.ArgumentParser(description="MABEL gesture demo")
    ap.add_argument("--port", type=str, default=None)
    ap.add_argument("--hand", choices=["left", "right"], default="left")
    ap.add_argument("--mode", default="sequence",
                    choices=["sequence", "random", "wave", "ripple"])
    ap.add_argument("--pose", nargs="+", metavar="POSE", help="custom pose sequence")
    ap.add_argument("--loops", type=int, default=1, help="repeats; 0 = forever")
    ap.add_argument("--speed", type=float, default=1.0, help="speed multiplier")
    ap.add_argument("--list", action="store_true", help="list poses and exit")
    ap.add_argument("--mock", action="store_true", help="no hardware")
    args = ap.parse_args()

    if args.list:
        print(f"\n{BOLD}Available poses:{RESET}")
        for name, pose in POSES.items():
            print(f"  {name:<14}  ({len(pose)} joints)")
        print(f"\n{BOLD}Built-in flourishes:{RESET}  wave, ripple  (use --mode)\n")
        return 0

    print(f"\n{BOLD}{'='*50}{RESET}\n{BOLD}  MABEL — GESTURE DEMO{RESET}\n{BOLD}{'='*50}{RESET}")
    hand = connect_hand(args.hand, port=args.port, mock=args.mock)
    print(f"{CYAN}  {hand.bus.port}   speed {args.speed}×   mode {args.mode}{RESET}\n")
    player = GesturePlayer(hand)

    if args.pose:
        seq = [(p, 1.2, 1.0) for p in args.pose]
    elif args.mode == "random":
        seq = [(random.choice(list(POSES)), 1.5, 1.0) for _ in range(12)]
    else:
        seq = DEMO_SEQUENCE

    print("  Starting from flat …")
    player.go("flat")
    time.sleep(1.0)

    loop = 0
    try:
        while True:
            loop += 1
            print(f"\n{BOLD}  Loop {loop}{RESET}")
            if args.mode == "wave" and not args.pose:
                player.wave()
            elif args.mode == "ripple" and not args.pose:
                player.ripple()
            else:
                run_sequence(player, seq, speed_mult=args.speed)
            if args.loops != 0 and loop >= args.loops:
                break
    except KeyboardInterrupt:
        print(f"\n{YELLOW}  Demo stopped.{RESET}")

    print("\n  Returning to flat …")
    player.transition("flat", duration=1.5)
    hand.disconnect()
    print(f"\n{GREEN}Done.{RESET}\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print(f"\n{RED}Aborted.{RESET}")
        sys.exit(1)
