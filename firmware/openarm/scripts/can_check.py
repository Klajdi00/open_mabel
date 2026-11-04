#!/usr/bin/env python3
"""
can_check.py — gs_usb (CANable / candleLight) bus diagnostic for macOS.

Answers one question definitively: *is the adapter healthy, and is anything
responding on the CAN bus?*  It does this in three stages, each with a clear
PASS/FAIL, so a silent bus can be attributed to the right layer:

  1. ADAPTER   — find + open the gs_usb device over USB (libusb/pyusb).
  2. LOOPBACK  — internal TX→RX self-test. Proves the adapter's CAN engine and
                 the USB read path work *without any motor on the bus*.
                 If this fails, the adapter is wedged → unplug/replug it.
  3. BUS SCAN  — DaMiao broadcast query (0x33) to IDs 1..15 in normal mode.
                 Genuine motor feedback frames (echoes filtered out) are
                 counted. Zero replies while LOOPBACK passed ⇒ the adapter is
                 fine but nothing on the wire answered → physical bus issue
                 (motor power, CANH/CANL wiring, or 120 Ω termination).

Usage:
    python scripts/can_check.py            # full diagnostic
    python scripts/can_check.py --no-reset # skip the USB reset on open
"""
import argparse, sys, time

VID, PID = 0x1D50, 0x606F
ERR_FLAG = 0x20000000


def _p(tag, ok, msg=""):
    mark = "\033[92m PASS\033[0m" if ok else "\033[91m FAIL\033[0m"
    print(f"[{mark} ] {tag}" + (f"  — {msg}" if msg else ""))


def main():
    ap = argparse.ArgumentParser(description="gs_usb CAN bus diagnostic")
    ap.add_argument("--bitrate", type=int, default=1_000_000)
    ap.add_argument("--no-reset", action="store_true",
                    help="don't USB-reset the adapter before opening")
    args = ap.parse_args()

    try:
        import usb.core
        from gs_usb.gs_usb import GsUsb
        from gs_usb.gs_usb_frame import GsUsbFrame
    except ImportError as e:
        print(f"Missing deps ({e}). Install:\n"
              "  brew install libusb && pip install python-can gs_usb pyusb")
        sys.exit(2)

    print(f"\n── gs_usb CAN diagnostic @ {args.bitrate} bps ──\n")

    # 1) ADAPTER — optionally reset, then open and program timing.
    if not args.no_reset:
        dev = usb.core.find(idVendor=VID, idProduct=PID)
        if dev is not None:
            try:
                dev.reset()
                time.sleep(2.0)   # macOS re-enumeration takes a beat
            except Exception:
                pass

    devs = GsUsb.scan()
    if not devs:
        _p("ADAPTER", False, "no CANable/candleLight found on USB "
                             "(is it plugged in? try a different port/cable)")
        sys.exit(1)
    d = devs[0]
    cap = d.device_capability
    bt = _timing(cap.fclk_can, args.bitrate)
    d.set_timing(**bt)
    _p("ADAPTER", True, f"canable found, fclk={cap.fclk_can/1e6:.0f} MHz, "
                        f"timing brp={bt['brp']} tseg1={bt['phase_seg1']+bt['prop_seg']} "
                        f"tseg2={bt['phase_seg2']}")

    # 2) LOOPBACK self-test (internal, no bus needed).
    GS_MODE_LOOP_BACK = 1 << 1
    lb_ok = False
    try:
        d.start(GS_MODE_LOOP_BACK)
        d.send(GsUsbFrame(can_id=0x7AA, data=[0xC0, 0xFF, 0xEE, 0x11]))
        end = time.time() + 0.6
        while time.time() < end:
            rf = GsUsbFrame()
            if d.read(rf, 2) and (rf.can_id & 0x7FF) == 0x7AA:
                lb_ok = True
                break
    finally:
        try: d.stop()
        except Exception: pass
    _p("LOOPBACK", lb_ok,
       "adapter CAN engine + USB read path OK" if lb_ok
       else "adapter wedged — UNPLUG the USB adapter, wait 3 s, replug, retry")
    if not lb_ok:
        sys.exit(1)
    time.sleep(0.3)

    # 3) BUS SCAN in normal mode.
    d.start(16)  # GS_CAN_MODE_HW_TIMESTAMP (normal operation)
    replies, errs = {}, 0
    end = time.time() + 2.0
    last = 0.0
    while time.time() < end:
        if time.time() - last > 0.04:           # ~25 Hz broadcast query
            for mid in range(1, 16):
                d.send(GsUsbFrame(can_id=0x7FF, data=[mid, 0, 0x33, 0]))
            last = time.time()
        rf = GsUsbFrame()
        if not d.read(rf, 2):
            continue
        if rf.can_id & ERR_FLAG:
            errs += 1
        elif rf.echo_id != 0xFFFFFFFF:
            continue                              # our own TX echo — ignore
        else:
            data = bytes(rf.data[:rf.can_dlc])
            if len(data) >= 3 and data[2] == 0x33:        # scan response
                replies[data[0] | (data[1] << 8)] = data.hex()
            else:                                          # MIT feedback
                replies[rf.can_id & 0x0F] = data.hex()
    try: d.stop()
    except Exception: pass

    if replies:
        _p("BUS SCAN", True, f"{len(replies)} motor(s) responding")
        for mid in sorted(replies):
            print(f"         • motor ID {mid}: {replies[mid]}")
    else:
        _p("BUS SCAN", False,
           f"no motor replies (bus error frames: {errs})")
        print("\n  Adapter is healthy (loopback passed) but nothing answered on")
        print("  the CAN bus. Check, in order of likelihood:")
        print("    1. Motor power — DaMiao needs its own DC supply; USB only")
        print("       powers the adapter, not the motor.")
        print("    2. CANH/CANL wiring — adapter CANH↔motor CANH, CANL↔CANL.")
        print("    3. 120 Ω termination — enable the CANable termination")
        print("       jumper and/or add a terminator at the motor end.")
        sys.exit(1)
    print()


def _timing(fclk, bitrate):
    """python-can-style sample-point timing → gs_usb set_timing kwargs."""
    import can
    bt = can.BitTiming.from_sample_point(f_clock=fclk, bitrate=bitrate,
                                         sample_point=87.5)
    prop = 1
    return dict(prop_seg=prop, phase_seg1=bt.tseg1 - prop,
                phase_seg2=bt.tseg2, sjw=bt.sjw, brp=bt.brp)


if __name__ == "__main__":
    main()
