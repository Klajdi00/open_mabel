#!/usr/bin/env python3
"""
test_streaming.py — rigorous stability test for the DaMiao teleop telemetry path.

Drives the GUI's own DamiaoBus exactly like the GUI's 50 Hz refresh loop, but
headless, and measures whether a found motor *stays* streaming on the bus without
being enabled:

  • feedback rate (Hz)            — how fast pos/vel/tau actually update
  • stale events                  — times feedback gapped > 0.5 s
  • found→lost transitions        — times the motor dropped off entirely
  • largest feedback gap (ms)     — worst-case silence
  • optional enable/disable cycle — confirms torque is gated by enable only

Usage:
  python scripts/tools/test_streaming.py --secs 20
  python scripts/tools/test_streaming.py --secs 20 --interface gs_usb
  python scripts/tools/test_streaming.py --secs 20 --enable-cycle   # also toggle torque
"""
import argparse, importlib.util, sys, time
from pathlib import Path

GUI = Path(__file__).resolve().parents[1] / "teleop_gui.py"


def load_gui():
    spec = importlib.util.spec_from_file_location("teleop_gui", GUI)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--secs", type=float, default=20.0)
    ap.add_argument("--hz", type=float, default=50.0, help="control loop rate")
    ap.add_argument("--interface", default=None)
    ap.add_argument("--port", default=None)
    ap.add_argument("--enable-cycle", action="store_true",
                    help="after streaming, enable then disable each found motor")
    args = ap.parse_args()

    tg = load_gui()
    iface = args.interface or ("serial" if (args.port or tg._find_serial_port())
                               else ("gs_usb" if tg._DARWIN else "socketcan"))
    print(f"opening bus: interface={iface}")
    bus = tg.DamiaoBus(interface=iface, channel="canable2", port=args.port)
    bus.scan()
    time.sleep(0.5)

    dt = 1.0 / args.hz
    period = max(1, int(args.hz // 2))   # discovery sweep every ~0.5 s
    # Per-motor telemetry trackers
    last_tfb = {}      # mid -> last seen t_fb
    last_rx  = {}      # mid -> wall time of last new feedback
    n_fb     = {}      # mid -> feedback count
    n_stale  = {}      # mid -> stale-event count
    n_lost   = {}      # mid -> found→lost transitions
    max_gap  = {}      # mid -> largest gap between feedbacks (s)
    was_found= {}
    was_stale= {}
    pos_min  = {}
    pos_max  = {}

    print(f"streaming for {args.secs:.0f}s at {args.hz:.0f} Hz (no enable)…")
    t_end = time.perf_counter() + args.secs
    tick = 0
    while time.perf_counter() < t_end:
        tick += 1
        do_disc = (tick % period == 0)
        with bus._lock:
            motors = list(bus.motors.items())
        for mid, m in motors:
            if m.found:
                bus.refresh(mid)
            elif do_disc:
                bus.refresh(mid)
        # sample telemetry
        nowt = time.perf_counter()
        with bus._lock:
            for mid, m in bus.motors.items():
                found = m.found
                if m.t_fb > 0 and last_tfb.get(mid) != m.t_fb:
                    last_tfb[mid] = m.t_fb
                    n_fb[mid] = n_fb.get(mid, 0) + 1
                    if mid in last_rx:
                        gap = nowt - last_rx[mid]
                        max_gap[mid] = max(max_gap.get(mid, 0.0), gap)
                    last_rx[mid] = nowt
                    pos_min[mid] = min(pos_min.get(mid,  1e9), m.pos)
                    pos_max[mid] = max(pos_max.get(mid, -1e9), m.pos)
                if found and m.stale and not was_stale.get(mid, False):
                    n_stale[mid] = n_stale.get(mid, 0) + 1
                was_stale[mid] = m.stale and found
                if was_found.get(mid, False) and not found:
                    n_lost[mid] = n_lost.get(mid, 0) + 1
                was_found[mid] = found
        time.sleep(dt)

    print("\n── streaming results (no enable) ──")
    live = [mid for mid in n_fb if n_fb[mid] > 0]
    if not live:
        print("  NO motor produced any feedback — nothing on the bus.")
    for mid in sorted(live):
        hz = n_fb[mid] / args.secs
        rng = (pos_max[mid] - pos_min[mid]) if mid in pos_min else 0.0
        print(f"  motor {mid}: {hz:5.1f} Hz   feedbacks={n_fb[mid]:4d}   "
              f"stale_events={n_stale.get(mid,0)}   lost={n_lost.get(mid,0)}   "
              f"max_gap={1000*max_gap.get(mid,0):.0f}ms   pos_range={rng:.4f}rad")
    verdict = all(n_stale.get(mid,0) == 0 and n_lost.get(mid,0) == 0 for mid in live)
    print(f"\n  VERDICT: {'STABLE — no stale/lost events' if (live and verdict) else 'see numbers above'}")

    if args.enable_cycle and live:
        print("\n── enable/disable cycle (torque gating) ──")
        for mid in sorted(live):
            print(f"  motor {mid}: enable (torque ON, hold pos)…")
            bus.enable(mid)
            t = time.perf_counter() + 2.0
            while time.perf_counter() < t:
                bus.send_mit(bus.motors[mid]); time.sleep(dt)
            with bus._lock: m = bus.motors[mid]
            print(f"    after enable: enabled={m.enabled} pos={m.pos:+.3f} tau={m.tau:+.2f} found={m.found}")
            print(f"  motor {mid}: disable (torque OFF, keep streaming)…")
            bus.disable(mid)
            t = time.perf_counter() + 2.0
            cnt0 = n_fb.get(mid,0)
            while time.perf_counter() < t:
                bus.refresh(mid); time.sleep(dt)
            with bus._lock: m = bus.motors[mid]
            print(f"    after disable: enabled={m.enabled} found={m.found} still_streaming={m.found and not m.stale}")

    bus.shutdown()
    print("\nbus closed.")


if __name__ == "__main__":
    main()
