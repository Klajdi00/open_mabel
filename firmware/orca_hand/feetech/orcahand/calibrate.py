"""Robust, fast end-stop calibration for the MABEL ORCA hand (Feetech bus).

This is the MABEL port of the ``orca_core`` auto-calibration routine
(https://github.com/orcahand/orca_core). The *algorithm* is the same idea --
drive each motor gently toward a mechanical hard-stop and detect the stop by
watching the encoder stop advancing -- but it is re-implemented against this
repo's :class:`orcahand.bus.FeetechBus` (a position bus with a torque ceiling)
instead of orca_core's Dynamixel *current-based-position* client.

Why the re-implementation
-------------------------
orca_core puts each motor into ``current_based_position`` mode, caps the current,
and nudges the goal relative to the present position until the encoder stalls.
The Feetech STS/SCS motors here have no current-based-position mode, but they do
have a **torque ceiling** (``ADDR_TORQUE_LIMIT``). So we cap the torque low,
then walk the goal ``present + dir * step`` each tick: while the joint is free
the encoder follows; once it jams against the end-stop the limited torque can't
push further, the encoder freezes, and the rolling-window stability test fires --
exactly orca_core's ``np.allclose(buffer, buffer[0])`` end-stop test.

The motor<->joint map is read straight from the hand's ``*_hand.yaml``, so this
works for *this* hand's ID assignment (which differs from orca_core's) with no
edits -- e.g. on the MABEL left hand ``index_mcp`` is motor 7, ``middle_pip`` is
motor 1, ``thumb_mp`` is motor 9.

Output
------
Writes the same ``<side>_calibration.json`` the rest of the stack already reads::

    { "7": {"joint": "index_mcp", "min": 1885, "max": 2910,
            "center": 2397, "range": 1025}, ... }

The previous file is backed up to ``<file>.bak`` first. Every motor is restored
to its measured centre and torque is released on exit (incl. Ctrl-C).

Usage
-----
::

    # real hardware, left hand, auto-detect port -- writes the packaged config
    python3 -m orcahand.calibrate --model left

    # only re-do a few joints (keeps the rest of the file intact)
    python3 -m orcahand.calibrate --model left --joints index_mcp index_pip

    # hardware-free self-test: synthetic end-stops seeded from the current file,
    # then reports how well the routine recovers them (no robot needed)
    python3 -m orcahand.calibrate --model left --sim
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from . import constants as C
from .config import HandConfig
from .hand import OrcaHand


# ============================================================================
# Pretty terminal output -- dependency-free ANSI (no rich/colorama needed).
# ============================================================================
class Term:
    """Minimal ANSI styler. Colour auto-disables off a TTY or with NO_COLOR."""

    def __init__(self, color: Optional[bool] = None):
        if color is None:
            color = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
        self.color = bool(color)

    def _w(self, code: str, s: str) -> str:
        return f"\033[{code}m{s}\033[0m" if self.color else s

    def bold(self, s):    return self._w("1", s)
    def dim(self, s):     return self._w("2", s)
    def red(self, s):     return self._w("31", s)
    def green(self, s):   return self._w("32", s)
    def yellow(self, s):  return self._w("33", s)
    def blue(self, s):    return self._w("34", s)
    def magenta(self, s): return self._w("35", s)
    def cyan(self, s):    return self._w("36", s)
    def grey(self, s):    return self._w("90", s)

    # status glyphs ---------------------------------------------------------
    @property
    def ok(self):   return self.green("✓")
    @property
    def warn(self): return self.yellow("▲")
    @property
    def bad(self):  return self.red("✗")

    def rule(self, char: str = "─", width: int = 64) -> str:
        return self.grey(char * width)

    def banner(self, title: str, subtitle: str = "") -> None:
        w = 64
        top = "╭" + "─" * (w - 2) + "╮"
        bot = "╰" + "─" * (w - 2) + "╯"
        mid = "│" + title.center(w - 2) + "│"
        print(self.cyan(self.bold(top)))
        print(self.cyan(self.bold(mid)))
        if subtitle:
            print(self.cyan("│" + subtitle.center(w - 2) + "│"))
        print(self.cyan(self.bold(bot)))

    def kv(self, key: str, val: str, pad: int = 22) -> None:
        print(f"  {self.grey(key.ljust(pad))} {val}")

    def step(self, idx: int, total: int, label: str) -> None:
        tag = self.magenta(f"[{idx:>2}/{total}]")
        print(f"\n{tag} {self.bold(label)}")

    def info(self, s: str) -> None:
        print(f"      {self.grey('·')} {s}")

    def bar(self, frac: float, width: int = 24) -> str:
        frac = max(0.0, min(1.0, frac))
        n = int(round(frac * width))
        return self.cyan("▕" + "█" * n + self.grey("░" * (width - n)) + "▕"
                         if self.color else "[" + "#" * n + "-" * (width - n) + "]")

    def live(self, s: str) -> None:
        """Overwrite the current line (no newline)."""
        if self.color:
            sys.stdout.write("\r\033[K" + s)
            sys.stdout.flush()
        # On a non-tty we stay quiet to avoid spamming logs with partial lines.

    def endlive(self) -> None:
        if self.color:
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()


# ============================================================================
# Tuning parameters (the Feetech analogues of orca_core's calibration_* knobs)
# ============================================================================
@dataclass
class CalibrationParams:
    calib_torque: int = 220          # torque ceiling during sweep (0-1000). Low = gentle stops.
    goal_speed: int = 600            # ticks/s cap during sweep (slow & controlled)
    goal_acc: int = 20               # acceleration units
    step_size: int = 18              # ticks added toward the stop each iteration (~0.028 rad)
    step_period: float = 0.02        # s between goal nudges / encoder reads
    num_stable: int = 8              # consecutive samples within threshold => stalled
    stable_threshold: int = 6        # max tick spread of the stability window
    warmup_iters: int = 6            # iterations before stall detection may trigger
    settle_time: float = 0.12        # s to let a motor settle before recording a stop
    max_sweep_s: float = 8.0         # hard wall-clock cap per direction (anti-hang)
    min_span: int = 120              # ticks; below this a joint is flagged "barely moved"
    max_span: int = 3600             # ticks; above this likely an encoder wrap / bad read
    recenter: bool = True            # drive each joint to its measured centre when done


# ============================================================================
# Per-joint result
# ============================================================================
@dataclass
class JointResult:
    name: str
    motor_id: int
    raw_min: int
    raw_max: int
    center: int
    status: str = "ok"               # ok | warn | fail
    notes: List[str] = field(default_factory=list)
    prev: Optional[Tuple[int, int]] = None   # previous (min, max) if any

    @property
    def span(self) -> int:
        return self.raw_max - self.raw_min

    def to_json(self) -> dict:
        return {
            "joint": self.name,
            "min": int(self.raw_min),
            "max": int(self.raw_max),
            "center": int(self.center),
            # The terminal sweep clamps commands at 0/4095 so it never crosses the
            # single-turn seam -- its ranges are always raw (offset 0). Seam
            # straddlers are caught by the max_span warning and handled in the GUI
            # (continuous sweep -> software offset). See orcahand.joints.Joint.
            "offset": 0,
            "range": int(self.span),
        }


# ============================================================================
# Calibrator
# ============================================================================
class Calibrator:
    """Drives the stall-detection sweep for every requested joint."""

    def __init__(self, hand: OrcaHand, params: CalibrationParams, term: Term):
        self.hand = hand
        self.bus = hand.bus
        self.p = params
        self.t = term

    # -- low-level helpers ---------------------------------------------------
    def _present(self, motor_id: int) -> Optional[int]:
        fb = self.bus.read_feedback(motor_id)
        return None if fb is None else int(fb["position"])

    def _feedback(self, motor_id: int) -> dict:
        return self.bus.read_feedback(motor_id) or {"position": 0, "load": 0, "temperature": 0}

    def _sweep(self, motor_id: int, name: str, direction: int, label: str) -> Optional[int]:
        """Walk ``motor_id`` toward one hard-stop; return the stall tick.

        ``direction`` is +1 (toward higher ticks) or -1 (toward lower ticks).
        Mirrors orca_core: command ``present + dir*step`` each tick and declare
        the stop once the encoder window is flat (``allclose``) for
        ``num_stable`` samples past a short warm-up.
        """
        t = self.t
        start = self._present(motor_id)
        if start is None:
            self.t.info(t.bad + f" motor {motor_id} not responding during sweep")
            return None

        window: List[int] = []
        last = start
        seen_lo, seen_hi = start, start
        arrow = "→ flex " if direction > 0 else "← extend"
        t0 = time.time()
        it = 0

        while time.time() - t0 < self.p.max_sweep_s:
            it += 1
            present = self._present(motor_id)
            if present is None:
                continue
            goal = present + direction * self.p.step_size
            goal = max(C.RAW_MIN, min(C.RAW_MAX, goal))
            self.bus.sync_write_positions({motor_id: goal})
            time.sleep(self.p.step_period)

            present = self._present(motor_id)
            if present is None:
                continue
            last = present
            seen_lo, seen_hi = min(seen_lo, present), max(seen_hi, present)

            window.append(present)
            if len(window) > self.p.num_stable:
                window.pop(0)

            # live readout -------------------------------------------------
            fb = self._feedback(motor_id)
            frac = present / float(C.RAW_MAX)
            t.live(
                f"      {t.cyan(name.ljust(11))} id {motor_id:<2} {t.dim(arrow)} "
                f"tick {t.bold(str(present).rjust(4))}  "
                f"load {str(int(fb.get('load', 0))).rjust(4)}  {t.bar(frac)}"
            )

            stalled = (
                it > self.p.warmup_iters
                and len(window) == self.p.num_stable
                and (max(window) - min(window)) <= self.p.stable_threshold
            )
            if stalled:
                break

        t.endlive()
        time.sleep(self.p.settle_time)
        final = self._present(motor_id)
        stop = final if final is not None else last
        travel = abs(stop - start)
        t.info(
            f"{t.dim(arrow)} stop at tick {t.bold(str(stop))} "
            f"{t.grey(f'(travelled {travel} ticks in {time.time()-t0:.1f}s)')}"
        )
        return stop

    # -- per-joint -----------------------------------------------------------
    def calibrate_joint(self, name: str, idx: int, total: int,
                        prev: Optional[Tuple[int, int]]) -> Optional[JointResult]:
        t = self.t
        joint = self.hand.config.joint(name)
        mid = joint.motor_id
        t.step(idx, total, f"{name}  (motor id {mid})")

        if mid not in self.hand._active_ids:
            t.info(t.bad + " motor did not answer connect scan -- skipped")
            return None

        # isolate this motor: only it holds torque during its own sweep
        self.bus.set_torque_enable(False)
        self.bus.set_torque_limit(self.p.calib_torque, mid)
        self.bus.set_torque_enable(True, mid)
        time.sleep(0.03)

        # sweep both physical stops (+ then -); order is irrelevant to min/max
        hi = self._sweep(mid, name, +1, "flex")
        lo = self._sweep(mid, name, -1, "extend")
        self.bus.set_torque_enable(False, mid)

        if hi is None or lo is None:
            t.info(t.bad + " sweep failed (no encoder data)")
            return None

        raw_min, raw_max = sorted((lo, hi))
        center = (raw_min + raw_max) // 2
        res = JointResult(name=name, motor_id=mid, raw_min=raw_min,
                          raw_max=raw_max, center=center, prev=prev)

        # sanity checks ----------------------------------------------------
        if res.span < self.p.min_span:
            res.status = "warn"
            res.notes.append(f"span {res.span} < {self.p.min_span} ticks (barely moved?)")
        if res.span > self.p.max_span:
            res.status = "fail"
            res.notes.append(f"span {res.span} > {self.p.max_span} ticks (encoder wrap / bad read)")

        glyph = {"ok": t.ok, "warn": t.warn, "fail": t.bad}[res.status]
        delta = ""
        if prev is not None:
            d = res.span - (prev[1] - prev[0])
            sign = "+" if d >= 0 else ""
            delta = t.grey(f"  Δspan {sign}{d}")
        t.info(
            f"{glyph} min {t.bold(str(raw_min))}  max {t.bold(str(raw_max))}  "
            f"center {center}  span {t.bold(str(res.span))}{delta}"
        )
        for n in res.notes:
            t.info(t.yellow("  " + n))

        # park at the measured centre under gentle torque
        if self.p.recenter:
            self.bus.set_torque_enable(True, mid)
            self.bus.sync_write_positions({mid: center})
            time.sleep(self.p.settle_time)
            self.bus.set_torque_enable(False, mid)
        return res

    # -- full run ------------------------------------------------------------
    def run(self, joints: List[str], prev_map: Dict[int, Tuple[int, int]]) -> List[JointResult]:
        self.bus.set_goal_acc(self.p.goal_acc)
        self.bus.set_goal_speed(self.p.goal_speed)
        self.bus.set_torque_limit(self.p.calib_torque)
        self.bus.set_torque_enable(False)

        results: List[JointResult] = []
        total = len(joints)
        for i, name in enumerate(joints, 1):
            mid = self.hand.config.joint(name).motor_id
            res = self.calibrate_joint(name, i, total, prev_map.get(mid))
            if res is not None:
                results.append(res)
        return results


# ============================================================================
# Calibration file I/O
# ============================================================================
def load_prev(path: Path) -> Dict[int, Tuple[int, int]]:
    """Previous (min, max) per motor id, for delta reporting / merge."""
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text())
    except Exception:
        return {}
    out: Dict[int, Tuple[int, int]] = {}
    for k, v in raw.items():
        try:
            out[int(k)] = (int(v["min"]), int(v["max"]))
        except Exception:
            continue
    return out


def write_calibration(path: Path, results: List[JointResult],
                      hand: OrcaHand, merge: bool, term: Term) -> None:
    """Write the ``<side>_calibration.json``, backing up any existing file.

    ``merge=True`` keeps entries for joints that were *not* recalibrated this
    run (so ``--joints`` is non-destructive); ``False`` writes only this run.
    """
    data: Dict[str, dict] = {}
    if merge and path.exists():
        try:
            data = json.loads(path.read_text())
        except Exception:
            data = {}

    for r in results:
        data[str(r.motor_id)] = r.to_json()

    # order keys by the hand's canonical joint order for a readable file
    order = {j.motor_id: i for i, j in enumerate(hand.config.joints)}
    ordered = dict(sorted(data.items(), key=lambda kv: order.get(int(kv[0]), 999)))

    if path.exists():
        bak = path.with_suffix(path.suffix + ".bak")
        bak.write_text(path.read_text())
        term.info(term.grey(f"backed up previous calibration -> {bak.name}"))

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(ordered, indent=4) + "\n")
    tmp.replace(path)
    term.info(term.green(f"wrote {len(results)} joint(s) -> {path}"))


# ============================================================================
# Summary table
# ============================================================================
def print_summary(term: Term, results: List[JointResult], requested: int,
                  elapsed: float, truth: Optional[Dict[int, Tuple[int, int]]] = None) -> None:
    t = term
    print("\n" + t.rule("═"))
    print(t.bold("  Calibration summary"))
    print(t.rule("═"))
    hdr = f"  {'joint':<12}{'id':>3}  {'min':>5}{'max':>6}{'center':>8}{'span':>6}  status"
    print(t.grey(hdr))
    n_ok = n_warn = n_fail = 0
    for r in sorted(results, key=lambda x: x.motor_id):
        glyph = {"ok": t.ok, "warn": t.warn, "fail": t.bad}[r.status]
        n_ok += r.status == "ok"
        n_warn += r.status == "warn"
        n_fail += r.status == "fail"
        line = (f"  {r.name:<12}{r.motor_id:>3}  {r.raw_min:>5}{r.raw_max:>6}"
                f"{r.center:>8}{r.span:>6}  {glyph}")
        if truth is not None and r.motor_id in truth:
            tlo, thi = truth[r.motor_id]
            err = max(abs(r.raw_min - tlo), abs(r.raw_max - thi))
            line += t.grey(f"   (sim err {err} ticks)")
        print(line)

    print(t.rule("─"))
    tally = (f"  {t.green(str(n_ok)+' ok')}   "
             f"{t.yellow(str(n_warn)+' warn')}   "
             f"{t.red(str(n_fail)+' fail')}   "
             f"{t.grey(f'/ {requested} requested')}")
    print(tally)
    print(f"  {t.grey('elapsed')} {elapsed:.1f}s")
    if truth is not None and results:
        errs = [max(abs(r.raw_min - truth[r.motor_id][0]),
                    abs(r.raw_max - truth[r.motor_id][1]))
                for r in results if r.motor_id in truth]
        if errs:
            print(f"  {t.grey('sim recovery')} max {max(errs)} ticks, "
                  f"mean {sum(errs)/len(errs):.1f} ticks "
                  f"({t.ok + ' algorithm verified' if max(errs) <= 25 else t.warn + ' large error'})")
    print(t.rule("═") + "\n")


# ============================================================================
# Simulated end-stop bus (hardware-free self-test of the routine)
# ============================================================================
class _SimEndstopBus:
    """A MockFeetechBus that clamps motion to per-motor synthetic end-stops.

    Seeded with ground-truth ``(min, max)`` per motor; the present position
    glides toward the last goal at a fixed sim speed and clamps at the stops,
    so the *real* stall-detection logic gets exercised end-to-end with no robot.
    """

    def __init__(self, stops: Dict[int, Tuple[int, int]], sim_speed: int = 80):
        from .mock import MockFeetechBus
        self._mock = MockFeetechBus(list(stops))
        self.port = "sim"
        self.baudrate = C.DEFAULT_BAUDRATE
        self.stops = stops
        self.sim_speed = sim_speed
        self._goal: Dict[int, int] = {m: s[0] + (s[1] - s[0]) // 2 for m, s in stops.items()}
        for m, g in self._goal.items():
            self._mock._pos[m] = g

    # passthrough lifecycle / config
    def open(self):  return self._mock.open()
    def close(self): self._mock.close()
    @property
    def is_open(self): return self._mock.is_open
    def __enter__(self): return self.open()
    def __exit__(self, *e): self.close()
    def set_torque_enable(self, *a, **k): pass
    def set_torque_limit(self, *a, **k): pass
    def set_goal_speed(self, *a, **k): pass
    def set_goal_acc(self, *a, **k): pass
    def ping(self, mid, timeout=0.02): return mid in self.stops
    def scan(self, id_range=range(1, 21)): return [i for i in id_range if i in self.stops]

    def sync_write_positions(self, id_to_raw):
        for mid, raw in id_to_raw.items():
            if mid in self._goal:
                self._goal[mid] = int(raw)

    def write_register(self, *a, **k): pass

    def _advance(self, mid: int) -> int:
        lo, hi = self.stops[mid]
        cur = self._mock._pos[mid]
        goal = max(lo, min(hi, self._goal[mid]))   # goal itself can't exceed the stop
        d = goal - cur
        step = max(-self.sim_speed, min(self.sim_speed, d))
        cur = max(lo, min(hi, cur + step))
        self._mock._pos[mid] = cur
        return cur

    def read_feedback(self, mid, timeout=0.01):
        if mid not in self.stops:
            return None
        pos = self._advance(mid)
        lo, hi = self.stops[mid]
        at_stop = pos <= lo or pos >= hi
        return {"position": pos, "speed": 0,
                "load": 380 if at_stop else 40, "voltage": 12.0, "temperature": 26.0}

    def read_positions(self, motor_ids, timeout=0.01):
        return {m: self._advance(m) for m in motor_ids if m in self.stops}


# ============================================================================
# CLI
# ============================================================================
# Nice per-finger ordering so the log reads thumb -> pinky -> wrist.
_FINGER_ORDER = ["thumb", "index", "middle", "ring", "pinky", "wrist"]
_JOINT_ORDER = ["abd", "cmc", "mcp", "mp", "pip", "ip", "dip"]


def _ordered_joints(hand: OrcaHand) -> List[str]:
    def key(name: str):
        parts = name.split("_")
        finger = parts[0]
        jt = parts[1] if len(parts) > 1 else ""
        fi = _FINGER_ORDER.index(finger) if finger in _FINGER_ORDER else 99
        ji = _JOINT_ORDER.index(jt) if jt in _JOINT_ORDER else 99
        return (fi, ji, name)
    return sorted(hand.joint_names, key=key)


def calibrate_main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Stall-detection end-stop calibration for the ORCA hand.")
    ap.add_argument("--model", default="left", choices=HandConfig.available_models(),
                    help="which hand config to calibrate (default: left)")
    ap.add_argument("--port", default=None, help="serial device ('' = auto-detect)")
    ap.add_argument("--out", default=None,
                    help="output calibration JSON (default: the model's calibration_file)")
    ap.add_argument("--joints", nargs="+", default=None,
                    help="only calibrate these joints (default: all). Implies merge.")
    ap.add_argument("--finger", default=None,
                    help="only calibrate one finger, e.g. 'index' or 'thumb'")
    ap.add_argument("--sim", action="store_true",
                    help="hardware-free self-test: synthetic stops from the current file")
    ap.add_argument("--no-color", action="store_true", help="disable ANSI colour")
    ap.add_argument("--yes", "-y", action="store_true", help="skip the confirm prompt")
    # tuning overrides
    ap.add_argument("--torque", type=int, default=None, help="calib torque ceiling 0-1000")
    ap.add_argument("--step", type=int, default=None, help="tick step per iteration")
    args = ap.parse_args(argv)

    term = Term(color=False if args.no_color else None)
    params = CalibrationParams()
    if args.torque is not None:
        params.calib_torque = args.torque
    if args.step is not None:
        params.step_size = args.step

    # resolve config + output path -----------------------------------------
    cfg = HandConfig.load(model=args.model)
    out_path = Path(args.out) if args.out else cfg.calibration_path
    if out_path is None:
        # config had no calibration_file yet -> default next to the yaml
        out_path = cfg.config_path.parent / f"{args.model}_calibration.json"

    term.banner("ORCA HAND  ·  end-stop calibration",
                f"{args.model.upper()} hand   ·   stall-detection sweep")

    # which joints ----------------------------------------------------------
    prev_map = load_prev(out_path)
    truth: Optional[Dict[int, Tuple[int, int]]] = None
    sim_bus = None
    if args.sim:
        if not prev_map:
            term.info(term.bad + " --sim needs an existing calibration file to seed stops")
            return 2
        truth = dict(prev_map)
        sim_bus = _SimEndstopBus(truth)

    hand = OrcaHand(args.model, port=args.port, bus=sim_bus,
                    torque_limit=params.calib_torque)

    if args.joints:
        joints = [j for j in _ordered_joints(hand) if j in set(args.joints)]
        unknown = set(args.joints) - set(hand.joint_names)
        if unknown:
            term.info(term.yellow(f"ignoring unknown joints: {sorted(unknown)}"))
    elif args.finger:
        joints = [j for j in _ordered_joints(hand) if j.split("_")[0] == args.finger]
    else:
        joints = _ordered_joints(hand)
    if not joints:
        term.info(term.bad + " no joints selected")
        return 2
    merge = bool(args.joints or args.finger)

    # report plan -----------------------------------------------------------
    term.kv("port", str(args.port or ("sim" if args.sim else "auto-detect")))
    term.kv("output file", str(out_path))
    term.kv("joints to calibrate", f"{len(joints)}  ({', '.join(joints)})")
    term.kv("merge with existing", "yes" if merge else "no (full rewrite)")
    term.kv("torque ceiling", f"{params.calib_torque} / 1000")
    term.kv("step / period", f"{params.step_size} ticks / {params.step_period*1000:.0f} ms")
    term.kv("stop test", f"{params.num_stable} samples within {params.stable_threshold} ticks")
    if args.sim:
        term.info(term.magenta("SIM MODE — no hardware will move"))

    if not args.yes and not args.sim:
        print()
        term.info(term.yellow("The hand will move each tendon to its hard-stops."))
        term.info(term.yellow("Keep fingers clear. "
                              "Ensure tendons are tensioned first."))
        try:
            ans = input(f"  {term.bold('Proceed?')} [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            ans = ""
        if ans not in ("y", "yes"):
            term.info("aborted by user")
            return 1

    # connect ---------------------------------------------------------------
    t_start = time.time()
    try:
        hand.connect(enable_torque=False, scan=True)
    except Exception as e:
        term.info(term.bad + f" could not connect: {e}")
        return 2

    found = sorted(hand._active_ids)
    print()
    term.kv("bus port", hand.bus.port)
    term.kv("motors responding", f"{len(found)} / {len(hand.motor_ids)}  {found}")
    if hand.missing_joints:
        term.info(term.yellow(f"missing (will skip): {hand.missing_joints}"))

    cal = Calibrator(hand, params, term)
    results: List[JointResult] = []
    rc = 0
    try:
        results = cal.run(joints, prev_map)
    except KeyboardInterrupt:
        term.endlive()
        print()
        term.info(term.yellow("interrupted — saving partial progress, releasing torque"))
        rc = 130
    except Exception as e:
        term.endlive()
        term.info(term.bad + f" calibration error: {e}")
        rc = 2
    finally:
        try:
            hand.disable_torque()
            hand.disconnect()
        except Exception:
            pass

    # write + summarise -----------------------------------------------------
    if results and not args.sim:
        write_calibration(out_path, results, hand, merge=merge or bool(args.joints), term=term)
    elif args.sim:
        term.info(term.grey("sim mode: results NOT written to disk"))

    print_summary(term, results, requested=len(joints),
                  elapsed=time.time() - t_start, truth=truth)
    return rc


if __name__ == "__main__":
    raise SystemExit(calibrate_main())
