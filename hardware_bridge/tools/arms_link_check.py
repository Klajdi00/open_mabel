#!/usr/bin/env python3
"""arms_link_check.py — READ-ONLY arms Teensy connection probe.

Walks the whole chain and reports each hop:

    lane carrier → host IP → ping → openarm HELLO → MANIFEST + STATE telemetry (per-motor)

Sends ONLY HELLO keepalives (the telemetry-request handshake) — NEVER a COMMAND
datagram, never a torque flag: probing cannot move or energise the arms. Safe to
run any time, including while the bridge owns the link (both bind ephemeral
ports; the Teensy streams to whichever endpoint spoke last, so a probe may
briefly steal the bridge's stream — it recovers via its 2 s service watchdog).

Usage:  python3 tools/arms_link_check.py  [--ip 192.168.124.11] [--port 5800]
        (run from hardware_bridge/, any python with stdlib — no ROS needed)
"""

import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=5).stdout.strip()
    except Exception:
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default=os.environ.get("MABEL_HW_ARMS_TEENSY_IP",
                                                   "192.168.124.11"))
    ap.add_argument("--port", type=int,
                    default=int(os.environ.get("MABEL_HW_ARMS_PORT", "5800")))
    ap.add_argument("--wait", type=float, default=3.0,
                    help="seconds to listen for STATE telemetry")
    args = ap.parse_args()

    print(f"── arms link check → {args.ip}:{args.port}  (READ-ONLY: hello+listen)\n")

    # 1) physical lanes
    lanes = sh("ip -br link | grep -E 'mgbe|enP'")
    up = [l.split()[0] for l in lanes.splitlines() if "NO-CARRIER" not in l]
    print(lanes or "  (no ethernet lanes found)")
    if up:
        print(f"  ✓ carrier on: {', '.join(up)}")
    else:
        print("  ✗ NO-CARRIER on every lane — the Teensy end is not bringing the "
              "link up.\n    A powered Teensy 4.1 only negotiates link when its "
              "firmware initialises\n    the PHY: flash/power the openarm-dual "
              "firmware (cd firmware/openarm && pio run -t upload).\n"
              "    (Cable/port problems also land here — but firmware is the "
              "usual cause on a fresh Teensy.)")

    # 2) host IP on the arms lane
    addr = sh("ip -br addr | grep 192.168.124.")
    print(f"\n  host lane IP: {addr or '✗ none — nmcli connection up teensy-base'}")

    # 3) ping (ICMP — not an arms command)
    alive = subprocess.run(["ping", "-c1", "-W1", args.ip],
                           capture_output=True).returncode == 0
    print(f"  ping {args.ip}: {'✓ reply' if alive else '✗ no reply'}")

    # 4) HELLO → MANIFEST + STATE (telemetry only: OpenArmLink sends a COMMAND
    #    only from send()/send_command(), which this probe never calls, and the
    #    stream thread stays idle because no group ever writes a slice). Loaded
    #    by file path so the stdlib-only promise holds.
    import importlib.util
    _p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "mabel_hw", "drivers", "openarm_link.py")
    _spec = importlib.util.spec_from_file_location("openarm_link", _p)
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules["openarm_link"] = _mod       # dataclasses needs it registered
    _spec.loader.exec_module(_mod)
    lk = _mod.OpenArmLink(teensy_ip=args.ip, port=args.port)
    got = specs = None
    try:
        lk.connect()
        deadline = time.monotonic() + args.wait
        while time.monotonic() < deadline:
            got, specs = lk.state(), lk.specs()
            if got is not None and specs:
                break
            time.sleep(0.1)
    except OSError as e:
        print(f"  openarm socket: ✗ {e}")
    finally:
        try:
            lk.close()
        except Exception:
            pass

    if got is None or not specs:
        print(f"  openarm STATE: ✗ no telemetry within {args.wait:.0f}s")
        print("\n── VERDICT: NOT CONNECTED." if not alive else
              "\n── VERDICT: host reaches the IP but no openarm firmware answers.")
        print("   Next: power-cycle the arms Teensy and re-run this probe.")
        return 1

    print(f"  openarm STATE: ✓ streaming ({len(specs)} actuators in the manifest)")
    for i, sp in enumerate(specs):
        if i >= len(got.joints):
            break
        m = got.joints[i]
        t = lk.temps()[i] if i < len(lk.temps()) else (None, None)
        print(f"    {sp.name:<10} {sp.motor:<7} bus{sp.bus} id{sp.mid}: "
              f"pos {m.pos:+7.3f} rad  vel {m.vel:+6.2f}  tau {m.tau:+6.2f} Nm "
              f"(t_max {sp.t_max:.0f})  T {t[0]}°C/{t[1]}°C")
    print("\n── VERDICT: CONNECTED. The bridge will publish /arms/joint_states + "
          "/arms/motor_health;\n   RViz merges them via joint_state_publisher; "
          "apps get them on :9090 robot_state + :9600 proprio.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
