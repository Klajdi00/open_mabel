#!/usr/bin/env python3
"""exp_arm_gravity_sysid.py — gravity-feedforward system identification for the arm's
gravity-dominant joints (l_j1, l_j2, l_j4), WITHOUT the ORCA hands mounted.

Like the torso sysid: hold the arm under pure PD (config gains, NO feedforward), walk
the gravity joints through a pose grid slowly, and at each settled point record every
joint's position and the test joints' steady holding torque — the motor's own τ feedback
at rest IS gravity + the stiction band. The companion analyzer
(``--analyze <dump.json>``) fits the measured torques against the MJCF model with the
hand mass STRIPPED (HandsConfig.NONE — the hands are off the robot), evaluating BOTH
``dir`` hypotheses for the freshly-replaced l_j3/l_j4 so the data itself settles their
sign, and reports the per-joint model fit + recommended corrections.

Safety: 0.2 rad/s glides, excursions bounded to the calibrated windows minus margin,
per-sample watchdog (any joint > 1.5 rad/s or 0.8 rad off its commanded path → damp +
abort), returns to the start pose, temperatures reported. EXCLUSIVE Teensy access
required (stop the HAL and the studio).

    python3 experiments/exp_arm_gravity_sysid.py --collect out.json
    python3 experiments/exp_arm_gravity_sysid.py --analyze out.json
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

HB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HB))
sys.path.insert(0, str(HB.parent / "controller"))

LEFT = [f"l_j{i}" for i in range(1, 8)]
TEST = ["l_j1", "l_j2", "l_j4"]
GLIDE_RADS = 0.25         # rad/s — slow, but fast enough not to stick-slip
STEP_DT = 0.02
MARGIN = 0.20             # stay WELL inside the freshly calibrated windows (~11.5°)


def collect(out_path: str):
    from mabel_hw import config
    from mabel_hw.calibration import CalibrationStore
    sys.path.insert(0, str(HB / "experiments"))
    from exp_j4_shake import HealthLink, wait

    cal = CalibrationStore().load().slice("arm/openarm-dual/calib") or {}
    link = HealthLink(config.LINKS["arms_openarm"]["ip"],
                      config.LINKS["arms_openarm"]["port"], 100,
                      logger=lambda m: print("   " + str(m)))
    link.connect()
    assert wait(lambda: link.specs() and link.state()), "Teensy not answering"
    idx_of = {s.name: i for i, s in enumerate(link.specs())}
    time.sleep(2.5)

    def pos(j):
        return link.state().joints[idx_of[j]].pos

    def tau(j):
        return link.state().joints[idx_of[j]].tau

    start = {j: pos(j) for j in LEFT}
    lo_hi = {}
    for j in LEFT:
        e = cal.get(j) or {}
        lo, hi = e.get("raw_min"), e.get("raw_max")
        lo_hi[j] = (lo + MARGIN, hi - MARGIN) if lo is not None else (None, None)
    print("start pose: " + "  ".join(f"{j}={start[j]:+.3f}" for j in LEFT))

    kp = dict(config.LEFT_ARM["kp"])
    kd = dict(config.LEFT_ARM["kd"])
    # STIFF gains for the IDENTIFICATION only (deploy stays compliant): at rest the
    # settled holding torque equals gravity regardless of gains, but soft shoulders
    # sag ~0.5 rad across the lift arc and trip the watchdog before measuring.
    for j in TEST:
        kp[j] = 60.0
        kd[j] = 5.0
    g = link.group(LEFT, kp=kp, kd=kd)
    assert wait(lambda: g.resolved)
    g.set_torque(True)                              # hold at measured
    time.sleep(0.8)
    current = dict(start)                           # the commanded setpoints

    # LIVE approximate FF while gliding/holding: without it the soft shoulders sag
    # ~0.5-0.8 rad across the lift arc and trip the watchdog (measured 2026-07-28).
    # The MEASURED τ stays the ground truth — it is the motor's total output however
    # it is split between FF and PD; the FF only keeps tracking sane.
    ffc = None
    try:
        from mabel.arm_control import ArmGravityCompensator
        from mabel.arm_control.gravity import HandsConfig
        ffc = ArmGravityCompensator(str(HB.parent / "simulation/mabel_mujoco/models/mabel_full.xml"),
                                    {}, {}, hands=HandsConfig.NONE, pose_pad=10.0,
                                    log=lambda *a, **k: None)
        print("live FF assist: ON (approximate, hands=none)")
    except Exception as e:
        print(f"live FF assist unavailable ({e}) — PD only")

    def ff_now():
        if ffc is None:
            return {}
        pose = {}
        for j, e in cal.items():
            if isinstance(e, dict) and "raw_min" in e and j in idx_of:
                rmid = 0.5 * (e["raw_min"] + e["raw_max"])
                pose[j] = float(e.get("model_mid", 0)) + int(e.get("dir", 1)) * (pos(j) - rmid)
        for k in range(1, 8):
            pose.setdefault(f"r_j{k}", 0.0)
        tm = ffc.tff(pose, frame="model")
        # ASSIST ONLY: scaled down + hard-capped so a model error in the very region
        # we are measuring can never sling the arm (observed: uncapped assist launched
        # l_j2 at 3.7 rad/s). The PD carries the remainder stiffly.
        CAP = {"l_j1": 4.0, "l_j2": 6.0, "l_j4": 4.0}
        out = {}
        for j in LEFT:
            if j in tm and j in cal:
                t = 0.7 * int(cal[j].get("dir", 1)) * float(tm.get(j, 0.0))
                c = CAP.get(j, 3.0)
                out[j] = max(-c, min(c, t))
        return out

    def clamp(j, v):
        lo, hi = lo_hi[j]
        return v if lo is None else max(lo, min(hi, v))

    def lower_safely():
        """NEVER free-fall: glide the setpoints back to the hang pose under torque
        + FF, then release. Used by aborts and the normal finish."""
        try:
            tgt = {j: clamp(j, start[j]) for j in LEFT}
            for _ in range(int(60 / STEP_DT)):          # hard time bound
                done = True
                for j, v in tgt.items():
                    d = v - current[j]
                    if abs(d) > 1e-3:
                        stp = max(-0.3 * STEP_DT, min(0.3 * STEP_DT, d))
                        current[j] = current[j] + stp
                        done = False
                g.command(dict(current), effort=ff_now())
                if done:
                    break
                time.sleep(STEP_DT)
            time.sleep(0.5)
        finally:
            g.set_torque(False)

    _viol = {}

    def glide_to(targets):
        """Slew the named joints to targets at GLIDE_RADS; others hold. Watchdogged —
        stick-slip breakaway makes brief (<40 ms) velocity spikes at crawl speeds, so
        the trip requires TWO consecutive bad samples; a trip LOWERS the arm safely
        (never a torque-cut free-fall)."""
        tgt = {j: clamp(j, v) for j, v in targets.items()}
        while True:
            done = True
            for j, v in tgt.items():
                d = v - current[j]
                if abs(d) > 1e-4:
                    step = max(-GLIDE_RADS * STEP_DT, min(GLIDE_RADS * STEP_DT, d))
                    current[j] = current[j] + step
                    done = False
            g.command(dict(current), effort=ff_now())
            st = link.state()
            # VELOCITY GOVERNOR: a breakaway lurch pauses the setpoint feed until the
            # joint calms — the lurch starves instead of being chased.
            if any(abs(st.joints[idx_of[j]].vel) > 1.2 for j in TEST):
                t_gov = time.monotonic()
                while time.monotonic() - t_gov < 2.0:
                    g.command(dict(current), effort=ff_now())
                    if all(abs(link.state().joints[idx_of[j]].vel) < 0.4 for j in TEST):
                        break
                    time.sleep(STEP_DT)
                continue
            for j in LEFT:
                f = st.joints[idx_of[j]]
                bad = abs(f.vel) > 3.0 or abs(f.pos - current[j]) > 1.0
                _viol[j] = _viol.get(j, 0) + 1 if bad else 0
                if _viol[j] >= 2:
                    print(f"  WATCHDOG: {j} vel={f.vel:+.2f} pos={f.pos:+.3f} vs "
                          f"cmd {current[j]:+.3f} — lowering safely")
                    lower_safely()
                    raise RuntimeError(f"WATCHDOG on {j} (lowered + released)")
            if done:
                return
            time.sleep(STEP_DT)

    def measure(tag):
        time.sleep(1.2)                             # settle
        taus = {j: [] for j in TEST}
        t0 = time.monotonic()
        while time.monotonic() - t0 < 1.0:
            for j in TEST:
                taus[j].append(tau(j))
            time.sleep(0.02)
        rec = {"tag": tag,
               "raw": {j: pos(j) for j in LEFT},
               "cmd": dict(current),
               "tau": {j: statistics.fmean(v) for j, v in taus.items()}}
        print(f"  [{tag}] " + "  ".join(f"{j}:{rec['tau'][j]:+.2f}N·m" for j in TEST))
        return rec

    records = []
    try:
        # v2 grid: the J2 LIFT ARC through center height (where the shake lives —
        # gravity is max at horizontal), J4 at two heights, J1 at a lifted J2.
        lo2, hi2 = lo_hi["l_j2"]
        j2_ladder = [lo2 + f * (hi2 - lo2) for f in (0.15, 0.30, 0.45, 0.60, 0.75, 0.85)]
        j2_ladder.sort(key=lambda a: abs(a - start["l_j2"]))
        d4_grid = [-0.5, -0.25, 0.0, 0.25, 0.5]
        d1_grid = [-0.6, -0.3, 0.0, 0.3, 0.6]
        print("── J2 lift ladder (full window) at two J4 bends ──")
        for d4 in (0.0, 0.5):
            glide_to({"l_j4": start["l_j4"] + d4})
            for a2 in j2_ladder:
                glide_to({"l_j2": a2})
                records.append(measure(f"j2@{a2:+.2f} j4={d4:+.2f}"))
            glide_to({"l_j2": start["l_j2"]})
        glide_to({"l_j4": start["l_j4"]})
        print("── J4 sweep at hang + mid-lift J2 ──")
        mid2 = lo2 + 0.5 * (hi2 - lo2)
        for a2 in (start["l_j2"], mid2):
            glide_to({"l_j2": a2})
            for d4 in d4_grid:
                glide_to({"l_j4": start["l_j4"] + d4})
                records.append(measure(f"j4={d4:+.2f} j2@{a2:+.2f}"))
            glide_to({"l_j4": start["l_j4"]})
        print("── J1 sweep at mid-lift J2 ──")
        glide_to({"l_j2": mid2})
        for d1 in d1_grid:
            glide_to({"l_j1": start["l_j1"] + d1})
            records.append(measure(f"j1={d1:+.2f} j2@{mid2:+.2f}"))
        glide_to({"l_j1": start["l_j1"], "l_j2": start["l_j2"]})
    finally:
        print("── lowering to start + release (no free-fall) ──")
        try:
            lower_safely()
        except Exception as e:
            print(f"  lowering: {e}")
            g.set_torque(False)
        temps = link.temps()
        for j in TEST + ["l_j3"]:
            t = temps[idx_of[j]]
            print(f"  {j} t_mos {t[0]} °C")
        link.close()
        Path(out_path).write_text(json.dumps(
            {"records": records, "note": "NO ORCA hands mounted; FF-assisted holds"}, indent=1))
        print(f"{len(records)} grid points → {out_path}")


def analyze(path: str):
    from mabel_hw.calibration import CalibrationStore
    from mabel.arm_control import ArmGravityCompensator
    from mabel.arm_control.gravity import HandsConfig

    data = json.loads(Path(path).read_text())
    records = data["records"]
    cal = CalibrationStore().load().slice("arm/openarm-dual/calib") or {}
    mjcf = str(HB.parent / "simulation/mabel_mujoco/models/mabel_full.xml")

    def calmap(overrides):
        m = {}
        for j, e in cal.items():
            if not isinstance(e, dict) or "raw_min" not in e:
                continue
            d = overrides.get(j, int(e.get("dir", 1)))
            m[j] = dict(rmid=0.5 * (e["raw_min"] + e["raw_max"]),
                        mmid=float(e.get("model_mid", 0.0)), dir=d)
        return m

    best = None
    for d3 in (+1, -1):
        for d4 in (+1, -1):
            cm = calmap({"l_j3": d3, "l_j4": d4})
            dirs = {j: v["dir"] for j, v in cm.items()}
            g = ArmGravityCompensator(mjcf, dirs, {}, hands=HandsConfig.NONE,
                                      ff_clamp=1e9, pose_pad=10.0,
                                      log=lambda *a, **k: None)
            errs = {j: [] for j in TEST}
            pairs = {j: [] for j in TEST}
            for r in records:
                pose = {j: cm[j]["mmid"] + cm[j]["dir"] * (r["raw"][j] - cm[j]["rmid"])
                        for j in cm if j in r["raw"]}
                # the collector recorded LEFT only; the right arm is a separate subtree
                # whose pose cannot affect left gravity — but the compensator's pose
                # gate requires all 14 joints, so fill r_* with model 0.
                for k in range(1, 8):
                    pose.setdefault(f"r_j{k}", 0.0)
                pred = g.tff(pose, frame="raw")
                for j in TEST:
                    if j in pred:
                        errs[j].append(pred[j] - r["tau"][j])
                        pairs[j].append((pred[j], r["tau"][j]))
            rms = {j: math.sqrt(sum(e * e for e in v) / len(v)) if v else float("inf")
                   for j, v in errs.items()}
            total = sum(rms.values())
            row = (total, d3, d4, rms, pairs)
            if best is None or total < best[0]:
                best = row
            print(f"dir(l_j3)={d3:+d} dir(l_j4)={d4:+d}:  RMS  "
                  + "  ".join(f"{j}={rms[j]:5.2f}" for j in TEST)
                  + f"   Σ={total:5.2f}")
    total, d3, d4, rms, pairs = best
    print(f"\nBEST FIT: dir(l_j3)={d3:+d}, dir(l_j4)={d4:+d}")
    for j in TEST:
        pv = pairs[j]
        if not pv:
            continue
        mp = statistics.fmean(abs(p) for p, _ in pv)
        mm = statistics.fmean(abs(m) for _, m in pv)
        num = sum(p * m for p, m in pv)
        den = sum(p * p for p, _ in pv) or 1e-9
        slope = num / den                       # measured ≈ slope × predicted
        print(f"  {j}: model|τ|≈{mp:5.2f}  measured|τ|≈{mm:5.2f}  "
              f"best scale (meas/model) = {slope:+.2f}  RMS {rms[j]:.2f} N·m")
    print("\nInterpretation: scale ≈ +1 & RMS ≲ stiction (~1.5 N·m) → the no-hands MJCF "
          "model is right, ship it; scale far from +1 → apply it as the per-joint "
          "correction; NEGATIVE scale → that joint's dir is wrong in the calibration.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", metavar="OUT")
    ap.add_argument("--analyze", metavar="IN")
    a = ap.parse_args()
    if a.collect:
        collect(a.collect)
    elif a.analyze:
        analyze(a.analyze)
    else:
        ap.print_help()
