#!/usr/bin/env python3
"""
collect.py
=================

Drive a REAL Damiao actuator and log the data the identifier needs.  Runs two
excitations back-to-back:

  * velocity staircase  -> friction curve   (saved results/<motor>/logs/staircase.csv)
  * frequency chirp      -> reflected inertia (saved results/<motor>/logs/chirp.csv)

The actual drive loop lives in :mod:`mabel_actid.session` so the GUI and this CLI
share one implementation.

SAFETY -- read before running:
  * Bench-mount the motor with the OUTPUT SHAFT FREE (no link, or a known,
    balanced load).  The default velocity is a conservative fraction of the
    motor's rated speed; the chirp amplitude is small.
  * Keep clear of the shaft; keep a hand on the e-stop / power.
  * Start with a low ``--vel-frac`` and short ``--dwell`` and increase once you
    trust the setup.
  * The script enables on start and ALWAYS disables on exit (incl. Ctrl-C).

Transport (auto-selected per platform):
  * macOS / Windows  -> ``--backend serial`` (DM-USB2FDCAN, ``--port`` auto-detected)
  * Linux            -> ``--backend socketcan`` (``--channel can0`` @ 1 Mbit)
    (sudo ip link set can0 up type can bitrate 1000000)

This is the only script that needs hardware; everything else (identify / plot)
runs offline from the CSVs it writes.
"""
import argparse

import _bootstrap                                   # noqa: F401
from mabel_actid.motors import get as get_motor
from mabel_actid import session


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motor", required=True, help="e.g. DM-J4340")
    ap.add_argument("--id", type=int, default=1, help="CAN id")
    ap.add_argument("--backend", choices=["serial", "gs_usb", "socketcan", "sim"],
                    default=None,
                    help="transport; macOS auto-picks serial-adapter-or-gs_usb, "
                         "Linux socketcan; sim runs a synthetic motor (no hardware)")
    ap.add_argument("--port", default=None,
                    help="serial device for --backend serial (auto-detect if omitted)")
    ap.add_argument("--channel", default="can0", help="SocketCAN interface")
    ap.add_argument("--rate", type=float, default=500.0, help="loop Hz")
    ap.add_argument("--vel-frac", type=float, default=0.25,
                    help="max staircase speed as fraction of rated v_max")
    ap.add_argument("--levels", type=int, default=12)
    ap.add_argument("--dwell", type=float, default=1.5)
    ap.add_argument("--kd", type=float, default=2.0,
                    help="velocity-loop gain for the staircase")
    ap.add_argument("--stair-kp", type=float, default=20.0,
                    help="position-ramp gain for the staircase; the velocity "
                         "staircase is realised as a position ramp so the motor "
                         "is forced through stiction even at low speed (set 0 "
                         "for legacy pure velocity-hold)")
    ap.add_argument("--chirp-amp", type=float, default=0.5, help="rad")
    ap.add_argument("--chirp-f0", type=float, default=0.2)
    ap.add_argument("--chirp-f1", type=float, default=3.0)
    ap.add_argument("--chirp-dur", type=float, default=20.0)
    ap.add_argument("--chirp-kp", type=float, default=40.0)
    ap.add_argument("--chirp-kd", type=float, default=2.0)
    ap.add_argument("--no-chirp", action="store_true",
                    help="skip the inertia sweep (friction only)")
    args = ap.parse_args()

    prof = get_motor(args.motor)
    cfg = session.CollectConfig(
        rate=args.rate, vel_frac=args.vel_frac, levels=args.levels,
        dwell=args.dwell, kd=args.kd, stair_kp=args.stair_kp,
        chirp=not args.no_chirp, chirp_amp=args.chirp_amp, chirp_f0=args.chirp_f0,
        chirp_f1=args.chirp_f1, chirp_dur=args.chirp_dur, chirp_kp=args.chirp_kp,
        chirp_kd=args.chirp_kd)

    backend = args.backend or session.default_backend()
    hooks = session.Hooks(log=print)
    print(f"backend={backend}" + (f"  port={args.port or 'auto'}" if backend == "serial"
                                  else f"  channel={args.channel}"))

    with session.make_bus(backend, channel=args.channel, port=args.port) as bus:
        bus.add_motor(args.id, prof)
        session.run_collection(bus, prof, args.id, cfg, hooks)

    rel = f"results/{args.motor}/logs"
    print(f"done. now run:  python identify.py "
          f"--staircase ../{rel}/staircase.csv "
          f"--dynamic ../{rel}/chirp.csv --motor {args.motor}")


if __name__ == "__main__":
    main()
