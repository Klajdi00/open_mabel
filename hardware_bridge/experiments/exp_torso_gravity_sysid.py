#!/usr/bin/env python3
"""exp_torso_gravity_sysid.py — POSTURE-AWARE torso gravity feedforward sysid.

Like the arm sysid (exp_arm_gravity_sysid.py): hold the torso at a ladder of angles
under PD (stiff ID gains, assist-FF capped low), and at each settled point record the
motor's own steady τ feedback — at rest that IS gravity ± the friction band — TOGETHER
WITH the measured pose of all 14 arm joints. The torso subtree carries both arms, so
the hold torque depends on the arm pose; the analyzer fits the measured torques against
the ARM-POSE-AWARE MJCF prediction (arm_control.ArmGravityCompensator with "torso" in
the pose, hands per HandsConfig) and prints the trim for control_param.json
``torso.gravity_feedforward`` (tau_ff = scale·tau_mjcf + offset_nm).

Run it at MORE THAN ONE arm pose (e.g. arms parked, then arms half-extended via the
studio) and analyze the dumps together — the scale then pins the arm-dependent term
against reality instead of extrapolating the MJCF.

The ladder walks down then back up, so static friction cancels in the per-angle mean.

Safety: 0.15 rad/s glides, excursions bounded to the calibrated window minus margin,
two-strike watchdog (vel/tracking), abort = glide back upright then release (the torso
BALANCES at ~+0.6°, operator-confirmed 2026-07-27). The arms are NEVER commanded.
EXCLUSIVE Teensy access required (stop the HAL and the studio).

    python3 experiments/exp_torso_gravity_sysid.py --collect out.json
    python3 experiments/exp_torso_gravity_sysid.py --analyze out.json [more.json ...]
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path

HB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HB))
sys.path.insert(0, str(HB.parent / "controller"))

TORSO = "torso"
ARMS = [f"{s}_j{k}" for s in ("l", "r") for k in range(1, 8)]
# Hold ladder (MODEL rad, 0 = upright; window is [-1.035, +0.215] minus margin).
DOWN = [0.10, 0.0, -0.15, -0.30, -0.45, -0.60, -0.75]
LADDER = DOWN + DOWN[::-1][1:]          # down then back up: friction cancels
GLIDE_RADS = 0.15
STEP_DT = 0.02
MARGIN = 0.06                            # stay inside the calibrated stops
ID_KP, ID_KD = 150.0, 3.0                # STIFF for the ID only; deploy is compliant
ASSIST_SCALE, ASSIST_CAP = 0.8, 16.0     # capped sinusoid assist — measured τ is
                                         # still ground truth (motor TOTAL output)
MJCF = HB.parent / "simulation" / "mabel_mujoco" / "models" / "mabel_full.xml"


def _assert_sole_commander():
    out = subprocess.run(["pgrep", "-af", "hardware_bridge.py|sim_teleop_bridge"],
                         capture_output=True, text=True).stdout.strip()
    if out:
        sys.exit(f"REFUSING: the HAL/gateway is running — stop it first:\n{out}")


def _model_map(cal, joint):
    """(raw->model, model->raw, dir) closures from a CalibrationStore entry
    (center-anchored law: model = model_mid + dir·(raw − raw_mid))."""
    e = cal.get(joint)
    if not (isinstance(e, dict) and "raw_min" in e):
        return (lambda r: r), (lambda m: m), 1
    rmid = 0.5 * (e["raw_min"] + e["raw_max"])
    mmid = float(e.get("model_mid", 0.0))
    d = int(e.get("dir", 1))
    return ((lambda r, rm=rmid, mm=mmid, dd=d: mm + dd * (r - rm)),
            (lambda m, rm=rmid, mm=mmid, dd=d: rm + (m - mm) / dd),
            d)


def collect(out_path: str):
    _assert_sole_commander()
    from mabel_hw import config
    from mabel_hw.calibration import CalibrationStore
    sys.path.insert(0, str(HB / "experiments"))
    from exp_j4_shake import HealthLink, wait

    store = CalibrationStore().load()
    tcal = store.slice("torso/damiao/calib") or {}
    acal = store.slice("arm/openarm-dual/calib") or {}
    to_model, to_raw, tdir = _model_map(tcal, TORSO)
    e = tcal.get(TORSO) or {}
    lo = (e.get("raw_min", -1.065)) + MARGIN
    hi = (e.get("raw_max", 0.245)) - MARGIN

    link = HealthLink(config.LINKS["arms_openarm"]["ip"],
                      config.LINKS["arms_openarm"]["port"], 100,
                      logger=lambda m: print("   " + str(m)))
    link.connect()
    assert wait(lambda: link.specs() and link.state()), "Teensy not answering"
    idx_of = {s.name: i for i, s in enumerate(link.specs())}
    if TORSO not in idx_of:
        sys.exit(f"no '{TORSO}' on the manifest — it has: {sorted(idx_of)}")
    time.sleep(2.0)

    def raw(j):
        return link.state().joints[idx_of[j]].pos

    def tau(j):
        return link.state().joints[idx_of[j]].tau

    def vel(j):
        return link.state().joints[idx_of[j]].vel

    start_raw = raw(TORSO)
    if not (lo - 0.35 <= start_raw <= hi + 0.35):
        link.close()
        sys.exit(f"torso raw {start_raw:+.3f} is OUTSIDE the calibrated window "
                 f"[{lo:+.3f}, {hi:+.3f}] — a multi-turn fold / stale calibration; "
                 "re-anchor in the setup studio before sweeping (a clamped command "
                 "here would be a full-revolution jump).")
    start_model = to_model(start_raw)
    print(f"torso at raw {start_raw:+.3f} (model {math.degrees(start_model):+.1f}°); "
          f"window model [{math.degrees(to_model(lo)):+.1f}°, "
          f"{math.degrees(to_model(hi)):+.1f}°]")
    arm_here = {j: raw(j) for j in ARMS if j in idx_of}
    print(f"arm pose captured for {len(arm_here)} joints (arms are NOT commanded)")

    # Sinusoid assist (validated law) so the stiff PD never stalls against the
    # gravity-loaded forward travel; capped LOW so a law error cannot sling.
    try:
        from mabel.torso_control import hold_tau
    except Exception:
        def hold_tau(_):
            return 0.0

    def assist(model_theta):
        t = ASSIST_SCALE * hold_tau(model_theta)
        return {TORSO: tdir * max(-ASSIST_CAP, min(ASSIST_CAP, t))}

    g = link.group([TORSO], kp={TORSO: ID_KP}, kd={TORSO: ID_KD},
                   wire_t_max={TORSO: config.TORSO["wire_t_max"]})
    assert wait(lambda: g.resolved), "torso group never resolved on the manifest"
    g.set_torque(True)                          # holds at measured — no lurch
    time.sleep(0.8)
    errs = link.errs()
    if errs and any(x not in (0, 1) for x in errs):
        print(f"⚠ DM error nibbles on the bus: {errs} (0x9 = undervoltage — 24V rail?)")

    current = start_raw                          # commanded RAW setpoint
    _viol = 0

    def send():
        g.command({TORSO: current}, effort=assist(to_model(current)))

    def glide_to(target_model):
        nonlocal current, _viol
        tgt = max(lo, min(hi, to_raw(target_model)))     # clamp in RAW (window frame)
        while abs(tgt - current) > 1e-4:
            current += max(-GLIDE_RADS * STEP_DT, min(GLIDE_RADS * STEP_DT, tgt - current))
            send()
            f_vel, f_pos = vel(TORSO), raw(TORSO)
            bad = abs(f_vel) > 1.5 or abs(f_pos - current) > 0.5
            _viol = _viol + 1 if bad else 0
            if _viol >= 2:
                print(f"  WATCHDOG: vel={f_vel:+.2f} pos={f_pos:+.3f} vs cmd "
                      f"{current:+.3f} — returning upright")
                go_home_and_release()
                raise RuntimeError("WATCHDOG (returned upright + released)")
            time.sleep(STEP_DT)

    def go_home_and_release():
        nonlocal current
        try:
            tgt = max(lo, min(hi, to_raw(0.0)))          # model upright, RAW frame
            while abs(tgt - current) > 1e-3:
                current += max(-0.25 * STEP_DT, min(0.25 * STEP_DT, tgt - current))
                send()
                time.sleep(STEP_DT)
            time.sleep(0.5)
        finally:
            g.set_torque(False)

    rows = []
    try:
        for a in LADDER:
            glide_to(a)
            time.sleep(1.2)                     # settle
            ts, ps = [], []
            t0 = time.monotonic()
            while time.monotonic() - t0 < 1.5:
                send()                          # keep the target latched
                ts.append(tau(TORSO))
                ps.append(raw(TORSO))
                time.sleep(0.02)
            row = {
                "target_model": a,
                "torso_raw": statistics.fmean(ps),
                "torso_model": to_model(statistics.fmean(ps)),
                "tau_raw": statistics.fmean(ts),
                "tau_model": tdir * statistics.fmean(ts),
                "arm_raw": {j: raw(j) for j in arm_here},
            }
            rows.append(row)
            print(f"  target {math.degrees(a):+6.1f}° → held "
                  f"{math.degrees(row['torso_model']):+6.1f}°, "
                  f"τ {row['tau_model']:+6.2f} N·m")
        go_home_and_release()
        print("upright again, torque OFF — sweep done")
    except Exception:
        raise
    finally:
        link.close()

    # Arm raw -> model for the analyzer (bake it in so analysis needs no store).
    arm_model = {}
    for j in arm_here:
        f, _, _ = _model_map(acal, j)
        arm_model[j] = {"per_row": [f(r["arm_raw"][j]) for r in rows]}
    dump = {"when": time.strftime("%Y-%m-%d %H:%M:%S"),
            "id_gains": {"kp": ID_KP, "kd": ID_KD},
            "torso_calib": e, "rows": rows,
            "arm_model_per_row": {j: arm_model[j]["per_row"] for j in arm_model}}
    Path(out_path).write_text(json.dumps(dump, indent=1))
    print(f"wrote {out_path} ({len(rows)} holds)")


def analyze(paths):
    from mabel.arm_control import ArmGravityCompensator
    from mabel.arm_control.gravity import HandsConfig
    from mabel.torso_control import hold_tau

    g = ArmGravityCompensator(str(MJCF), {}, {}, hands=HandsConfig.NONE,
                              log=lambda *a, **k: None)
    assert g.available, "MJCF compensator failed to load"

    meas, pred, sinu = [], [], []
    for p in paths:
        d = json.loads(Path(p).read_text())
        per_row = d.get("arm_model_per_row", {})
        for i, r in enumerate(d["rows"]):
            pose = {j: per_row[j][i] for j in per_row}
            for j in ARMS:                       # absent arm joints: MJCF rest pose
                pose.setdefault(j, 0.0)
            pose["torso"] = float(r["torso_model"])
            tm = g.tff(pose, frame="model").get("torso")
            if tm is None:
                print(f"  ⚠ row {i} of {p}: pose rejected by the model — skipped")
                continue
            meas.append(float(r["tau_model"]))
            pred.append(float(tm))
            sinu.append(hold_tau(float(r["torso_model"])))

    n = len(meas)
    assert n >= 4, f"only {n} usable rows"
    mx, my = sum(pred) / n, sum(meas) / n
    sxx = sum((x - mx) ** 2 for x in pred)
    sxy = sum((x - mx) * (y - my) for x, y in zip(pred, meas))
    scale = sxy / sxx if sxx > 1e-9 else 1.0
    offset = my - scale * mx

    def rms(res):
        return math.sqrt(sum(e * e for e in res) / n)

    r_raw = rms([y - x for x, y in zip(pred, meas)])
    r_fit = rms([y - (scale * x + offset) for x, y in zip(pred, meas)])
    r_sin = rms([y - s for s, y in zip(sinu, meas)])
    print(f"\n{n} holds from {len(paths)} dump(s)")
    print(f"  measured vs MJCF (identity trim):  rms {r_raw:6.2f} N·m")
    print(f"  measured vs sinusoid fallback:     rms {r_sin:6.2f} N·m")
    print(f"  measured vs MJCF (fitted trim):    rms {r_fit:6.2f} N·m")
    print(f"\n  fitted trim:  scale {scale:+.4f}   offset {offset:+.3f} N·m")
    if scale < 0:
        print("  ⚠ NEGATIVE scale — a sign convention is wrong (dir?); do NOT deploy.")
        return
    if r_fit > max(2.5, r_sin):
        print("  ⚠ fit no better than the sinusoid — inspect before deploying.")
    print("\n  → paste into controller/control_param.json torso.gravity_feedforward:")
    print(f'       "scale": {scale:.4f},')
    print(f'       "offset_nm": {offset:.3f}')
    print("  (restart the gateway to pick it up; ff_trim() reads it live per call)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    mx = ap.add_mutually_exclusive_group(required=True)
    mx.add_argument("--collect", metavar="OUT_JSON")
    mx.add_argument("--analyze", nargs="+", metavar="JSON")
    args = ap.parse_args()
    if args.collect:
        collect(args.collect)
    else:
        analyze(args.analyze)
