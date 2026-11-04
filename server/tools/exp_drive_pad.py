#!/usr/bin/env python3
"""exp_drive_pad.py — does the split Drive cockpit move the robot the right way?

The 2026-10-16 Drive cockpit (iOS v2 + VisionOS V2) replaced the ROTATE·LIFT
stick with three controls: TRANSLATE 2-D stick · LIFT vertical slider · ROTATE
1-D stick (`DriveControls.swift`). Unit tests pin the drag→deflection signs;
this pins the REST of the chain — app callbacks → RobotController → wire →
bridge → engine → MuJoCo — by reading the robot's response off robot_state.

One SCRIPT, three runs, same measurements:

  reference   this tool drives the wire itself (`navJoystick` lx/ly/rx/ry with
              the stick conventions the apps use) — the ground truth.
  ios / vp    the app drives, from its DEBUG `debugDrivePad` default (the
              Simulator cannot synthesise a drag); this tool only OBSERVES and
              takes the step boundaries from the app's `[MABEL-DRIVEPAD]` log.

Per step it reports the body-frame forward/lateral displacement (projected on
the heading at step start), the yaw change and the lift change, and checks the
app's SIGN on every axis against the reference. Magnitudes differ only by the
app's speed governor (`speedScale`), so signs are the pass/fail criterion and
magnitudes are reported.

    # bridge:  server/start_teleop.sh   (sim, :9090)
    python3 exp_drive_pad.py run --mode reference --out /tmp/dp_ref.json
    python3 exp_drive_pad.py run --mode app --device <UDID> --out /tmp/dp_ios.json
    python3 exp_drive_pad.py report /tmp/dp_ref.json /tmp/dp_ios.json /tmp/dp_vp.json \
        --fig docs/figures/drive_pad_e2e.pdf
"""

import argparse
import asyncio
import json
import math
import re
import subprocess
import threading
import time

SCRIPT = "ly=-0.8:3;rot=0.8:3;lift=1:3;lift=-1:3"
GAP = 1.0                      # release between steps (DebugDriveScript)
START_DELAY = 4.0              # reference waits like the app does


def parse_script(spec):
    steps = []
    for part in spec.split(";"):
        k, v = part.split("=")
        val, secs = v.split(":")
        steps.append({"axis": k, "value": float(val), "seconds": float(secs)})
    return steps


# The apps' TwistSigns for the sim (NetworkConfig.swift): fwd +1, strafe −1, yaw −1.
SIGNS = {"fwd": +1.0, "strafe": -1.0, "yaw": -1.0}


def stick_frame(axis, value):
    """App pad axis → navJoystick, replicating the apps' chain exactly:
    TRANSLATE (dx, dy) with dy DOWN+ → twist (fwd = −dy, strafe = dx) →
    WebSocketLink nav.ly = SIGNS.fwd·fwd, nav.lx = SIGNS.strafe·strafe;
    ROTATE r → yaw = r → nav.rx = SIGNS.yaw·r; LIFT l → nav.ry = l.
    (Full-scale stick = full-scale nav: the maxLinear/maxAngular factors
    cancel; the app's speed governor only scales magnitude.)"""
    nav = {"lx": 0.0, "ly": 0.0, "rx": 0.0, "ry": 0.0}
    if axis == "ly":
        nav["ly"] = SIGNS["fwd"] * -value
    elif axis == "lx":
        nav["lx"] = SIGNS["strafe"] * value
    elif axis == "rot":
        nav["rx"] = SIGNS["yaw"] * value
    else:
        nav["ry"] = value
    return nav


async def record(args, steps):
    import websockets
    url = f"ws://{args.host}:{args.port}/teleop?client=exp_drive_pad"
    samples, events = [], []
    t_end = time.time() + args.duration

    log_proc = None
    if args.mode == "app":
        # Step boundaries come from the app itself, on the same host clock.
        log_proc = subprocess.Popen(
            ["xcrun", "simctl", "spawn", args.device, "log", "stream", "--style", "compact",
             "--predicate", 'eventMessage CONTAINS "[MABEL-DRIVEPAD]"'],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)

        def tail():
            pat = re.compile(r"\[MABEL-DRIVEPAD\] (?:step (\w+)=(-?[\d.]+) for ([\d.]+)s|(done))")
            for line in log_proc.stdout:
                m = pat.search(line)
                if not m:
                    continue
                now = time.time()
                if m.group(4):
                    events.append({"t": now, "kind": "done"})
                else:
                    events.append({"t": now, "kind": "step", "axis": m.group(1),
                                   "value": float(m.group(2)), "seconds": float(m.group(3))})
                    print(f"[app] {m.group(1)}={m.group(2)} for {m.group(3)}s")
        threading.Thread(target=tail, daemon=True).start()

    async with websockets.connect(url, open_timeout=8, max_size=2 ** 23) as ws:
        if args.reset:
            # Every run must start from the SAME pose: a run that begins where
            # the last one parked the robot can drive it into furniture, which
            # reads as a weak/short response while the command is held.
            await ws.send(json.dumps({"type": "reset", "payload": {"random": False}}))
            print("[exp] sim reset to home; settling 3 s")
            await asyncio.sleep(3.0)

        async def reader():
            while time.time() < t_end:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                env = json.loads(msg)
                if env.get("type") != "robot_state":
                    continue
                p = env.get("payload", {})
                b = p.get("base") or {}
                samples.append({"t": time.time(), "x": b.get("x"), "y": b.get("y"),
                                "yaw": b.get("yaw"),
                                "lift": (p.get("jointPositions") or {}).get("lift_lower")})

        async def driver():
            seq = 0

            async def send(nav):
                nonlocal seq
                seq += 1
                await ws.send(json.dumps({"type": "teleop_frame", "payload": {
                    "sequence": seq, "timestamp": time.time(),
                    "head": {"transform": {"matrix": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]},
                             "trackingState": "notAvailable"},
                    "mode": "Base Driving", "navJoystick": nav}}))

            async def hold(nav, secs):
                end = time.time() + secs
                while time.time() < end:
                    await send(nav)
                    await asyncio.sleep(1 / 30)

            # Same preconditions the apps establish: motors on, navigation spec.
            await ws.send(json.dumps({"type": "estop", "payload": {"active": False}}))
            await ws.send(json.dumps({"type": "control_mode", "payload": {
                "method": "navigation", "type": "impedance", "region": "whole_body"}}))
            zero = stick_frame("lx", 0.0)
            await hold(zero, START_DELAY)
            for st in steps:
                events.append({"t": time.time(), "kind": "step", **st})
                print(f"[ref] {st['axis']}={st['value']} for {st['seconds']}s")
                await hold(stick_frame(st["axis"], st["value"]), st["seconds"])
                await hold(zero, GAP)
            events.append({"t": time.time(), "kind": "done"})
            while time.time() < t_end:
                await hold(zero, 0.5)

        tasks = [reader()] + ([driver()] if args.mode == "reference" else [])
        await asyncio.gather(*tasks)

    if log_proc:
        log_proc.terminate()
    return samples, events


def analyse(samples, events):
    """Per step: body-frame Δforward/Δlateral, Δyaw (deg), Δlift (m) over
    [start, start + seconds + GAP] (the release gap lets the ramp settle)."""
    out = []
    for ev in [e for e in events if e["kind"] == "step"]:
        t0, t1 = ev["t"], ev["t"] + ev["seconds"] + GAP
        win = [s for s in samples if t0 <= s["t"] <= t1 and s["x"] is not None]
        if len(win) < 2:
            out.append({**ev, "n": len(win)})
            continue
        a, b = win[0], win[-1]
        dx, dy = b["x"] - a["x"], b["y"] - a["y"]
        h = a["yaw"]
        fwd = dx * math.cos(h) + dy * math.sin(h)
        lat = -dx * math.sin(h) + dy * math.cos(h)
        dyaw = sum(math.remainder(w2["yaw"] - w1["yaw"], 2 * math.pi)
                   for w1, w2 in zip(win, win[1:]))
        dlift = ((b["lift"] or 0) - (a["lift"] or 0)) if a["lift"] is not None else None
        out.append({**ev, "n": len(win), "fwd": fwd, "lat": lat,
                    "dyaw_deg": math.degrees(dyaw), "dlift": dlift})
    return out


# Which measured quantity each pad axis is supposed to move.
PRIMARY = {"ly": "fwd", "lx": "lat", "rot": "dyaw_deg", "lift": "dlift"}
# PHYSICAL expectation, independent of any sign table: the ROTATE stick's right
# end wears a turn-RIGHT glyph, so +deflection must turn the robot CLOCKWISE
# seen from above (MuJoCo yaw DEcreasing); the LIFT slider pushed up must
# raise the column. (Forward has no such check: which body axis is the nose is
# the sign table's business, so ly is compared against the reference only.)
EXPECT = {"rot": -1.0, "lift": +1.0}


def cmd_run(args):
    steps = parse_script(args.script)
    samples, events = asyncio.run(record(args, steps))
    res = analyse(samples, events)
    print(f"\n{len(samples)} robot_state samples, {len(res)} steps ({args.mode})")
    for r in res:
        print(f"  {r['axis']:>4}={r['value']:+.1f}  n={r['n']:4d}  "
              f"fwd={r.get('fwd', float('nan')):+.4f} m  lat={r.get('lat', float('nan')):+.4f} m  "
              f"yaw={r.get('dyaw_deg', float('nan')):+7.2f} deg  lift={r.get('dlift') or float('nan'):+.4f} m")
    json.dump({"mode": args.mode, "label": args.label or args.mode, "script": args.script,
               "samples": samples, "events": events, "steps": res}, open(args.out, "w"))
    print(f"wrote {args.out}")


def cmd_report(args):
    runs = [json.load(open(p)) for p in args.runs]
    ref = runs[0]
    print(f"{'axis':>10} " + " ".join(f"{r['label']:>22}" for r in runs) + "   sign")
    ok_all = True
    for i, st in enumerate(ref["steps"]):
        key = PRIMARY[st["axis"]]
        vals = [r["steps"][i].get(key) if i < len(r["steps"]) else None for r in runs]
        unit = "deg" if key == "dyaw_deg" else "m"
        ref_v = vals[0]
        signs = [v is not None and ref_v is not None and abs(v) > 1e-4 and
                 math.copysign(1, v) == math.copysign(1, ref_v) for v in vals[1:]]
        phys = ""
        if st["axis"] in EXPECT:
            want = EXPECT[st["axis"]] * math.copysign(1, st["value"])
            good = [v is not None and abs(v) > 1e-4 and math.copysign(1, v) == want for v in vals]
            ok_all &= all(good)
            phys = "  physical " + ("OK" if all(good) else "WRONG")
        ok_all &= all(signs)
        print(f"{st['axis']+'='+format(st['value'], '+.1f'):>10} "
              + " ".join(f"{(v if v is not None else float('nan')):>18.4f} {unit:>3}" for v in vals)
              + "   " + ("MATCH" if all(signs) else "MISMATCH") + phys)
    # Cross-axis leak: on a lift step the base must not turn, and vice versa.
    for r in runs:
        for st in r["steps"]:
            if st["axis"] == "lift":
                print(f"  [{r['label']}] lift={st['value']:+.0f}: yaw leak {st.get('dyaw_deg', 0):+.3f} deg, "
                      f"fwd leak {st.get('fwd', 0):+.4f} m")
            if st["axis"] == "rot":
                print(f"  [{r['label']}] rot={st['value']:+.1f}: lift leak {st.get('dlift') or 0:+.5f} m")
    print("\nVERDICT:", "every app axis moves the robot the same way as the reference"
          if ok_all else "SIGN MISMATCH — see table")
    if args.fig:
        plot(runs, args.fig)


def plot(runs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(7, 6.2), sharex=True)
    for r in runs:
        ev = [e for e in r["events"] if e["kind"] == "step"]
        if not ev:
            continue
        t0 = ev[0]["t"]
        S = [s for s in r["samples"] if s["x"] is not None and t0 - 0.5 <= s["t"]]
        if not S:
            continue
        x0, y0, h0 = S[0]["x"], S[0]["y"], S[0]["yaw"]
        t = [s["t"] - t0 for s in S]
        fwd = [(s["x"] - x0) * math.cos(h0) + (s["y"] - y0) * math.sin(h0) for s in S]
        yaw, acc = [], 0.0
        for a, b in zip([S[0]] + S, S):
            acc += math.remainder(b["yaw"] - a["yaw"], 2 * math.pi)
            yaw.append(math.degrees(acc))
        lift = [s["lift"] if s["lift"] is not None else float("nan") for s in S]
        axes[0].plot(t, fwd, label=r["label"], lw=1.4)
        axes[1].plot(t, yaw, lw=1.4)
        axes[2].plot(t, lift, lw=1.4)
    ref_ev = [e for e in runs[0]["events"] if e["kind"] == "step"]
    t0 = ref_ev[0]["t"]
    for e in ref_ev:
        for ax in axes:
            ax.axvspan(e["t"] - t0, e["t"] - t0 + e["seconds"], color="0.88", lw=0)
        axes[0].text(e["t"] - t0 + 0.1, 0, f"{e['axis']}={e['value']:+.1f}", fontsize=7,
                     va="bottom", color="0.35")
    axes[0].set_ylabel("forward disp. (m)")
    axes[1].set_ylabel("yaw (deg)")
    axes[2].set_ylabel("lift_lower (m)")
    axes[2].set_xlabel("time from first step (s)  —  shaded = pad held")
    axes[0].legend(fontsize=8, loc="upper right")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--mode", choices=["reference", "app"], required=True)
    r.add_argument("--host", default="127.0.0.1")
    r.add_argument("--port", type=int, default=9090)
    r.add_argument("--device", default="booted", help="simulator UDID (app mode)")
    r.add_argument("--script", default=SCRIPT)
    r.add_argument("--duration", type=float, default=30.0)
    r.add_argument("--label", default=None)
    r.add_argument("--reset", action="store_true", help="home the sim before recording")
    r.add_argument("--out", required=True)
    p = sub.add_parser("report")
    p.add_argument("runs", nargs="+", help="reference JSON first")
    p.add_argument("--fig", default=None)
    args = ap.parse_args()
    cmd_run(args) if args.cmd == "run" else cmd_report(args)


if __name__ == "__main__":
    main()
