"""Entry point:  python -m gui  [--iface can0] [--mock]"""

import argparse
import tkinter as tk

from . import config as C
from .app import SwerveStudio
from .bus import SwerveBus


def main() -> int:
    ap = argparse.ArgumentParser(description="MABEL Swerve Studio")
    ap.add_argument("--iface", default=C.CAN_IFACE,
                    help=f"CAN interface (default: {C.CAN_IFACE})")
    ap.add_argument("--mock", action="store_true",
                    help="force simulation mode (no CAN hardware needed)")
    args = ap.parse_args()

    bus = SwerveBus(iface=args.iface, mock=args.mock)
    if bus.error:
        print(bus.error)

    root = tk.Tk()
    app = SwerveStudio(root, bus)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
