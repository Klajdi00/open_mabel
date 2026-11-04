"""rs_util.py — ONE long-lived librealsense context shared by all RealSense drivers.

Never create a throwaway rs.context() per poll. With the RSUSB backend every new
context OPENS each USB device to read its descriptors; a device that is busy
streaming in this same process makes that open throw ('failed to set power state')
and the file handle LEAKS. Measured on thor (2026-07-08): the front-D435 driver's
2 s idle poll accumulated 16,340 open fds on the streaming D405's /dev/bus/usb node
in ~27 h, hit the 16384 fd rlimit (EMFILE 'Too many open files'), and blinded the
bridge to the hot-plugged D435.

A single persistent context fixes the leak, but NOT the enumeration itself: with
RSUSB, query_devices() throws WHOLESALE ('failed to set power state') while ANY
RealSense is busy streaming — even in this same process — it does not skip the busy
device (measured on thor, 2026-07-08). What DOES work while a device is busy is
pipeline.start() with config.enable_device(serial) or even a serial-less start:
pipeline RESOLUTION skips busy devices (both verified live). Hence the discovery
strategy the drivers use:

  1. scan()      — live query_devices(); updates the registry. Succeeds whenever
                   no RealSense is mid-stream (e.g. bridge startup).
  2. registry()  — serials+names cached from every past successful scan; the
                   fallback while streaming blocks live scans.
  3. sysfs_products() — product names straight from /sys (never opens the device,
                   always readable). Gates the last resort for a camera that was
                   HOT-PLUGGED while another streams (its librealsense serial is
                   unknowable until a scan succeeds): the driver does a serial-less
                   pipeline.start and verifies the claimed device's name, releasing
                   it if it grabbed the wrong kind. NOTE the sysfs 'serial' file is
                   the USB iSerial, which does NOT match the librealsense
                   serial_number — use sysfs only for presence-by-name.
"""

try:
    import pyrealsense2 as rs
    _RS_OK = True
except Exception:                                     # pragma: no cover
    rs = None
    _RS_OK = False

_ctx = None
_registry = {}          # librealsense serial -> device name (grows; never forgets)


def context():
    """The process-wide rs.context (created lazily on first use)."""
    global _ctx
    if _ctx is None:
        _ctx = rs.context()
    return _ctx


def scan():
    """Live query_devices() on the shared context; updates + returns the registry.
    Raises while any RealSense is busy streaming (see module docstring)."""
    for d in context().query_devices():
        try:
            _registry[d.get_info(rs.camera_info.serial_number)] = \
                d.get_info(rs.camera_info.name)
        except Exception:
            continue
    return dict(_registry)


def registry():
    """Serials+names from past successful scans (stale-tolerant fallback)."""
    return dict(_registry)


def remember(serial, name):
    """Record a device learned outside scan() (e.g. from a blind pipeline start)."""
    _registry[serial] = name


def sysfs_products():
    """Product strings of Intel (8086) USB devices from /sys — presence-by-name
    that stays readable while devices stream (reading sysfs never opens them)."""
    import glob
    out = []
    for d in glob.glob("/sys/bus/usb/devices/*"):
        try:
            with open(d + "/idVendor") as f:
                if f.read().strip() != "8086":
                    continue
            with open(d + "/product") as f:
                out.append(f.read().strip())
        except OSError:
            continue
    return out
