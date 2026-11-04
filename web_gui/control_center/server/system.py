"""
Host + robot-server status for the networking panel: interface IPs, service
port probes (teleop/map/video planes), CPU/mem/disk, thermal zones, uptime.
Pure stdlib except psutil (optional — degrades to /proc parsing on Linux).
"""
import glob
import os
import socket
import time

from . import config as CFG

try:
    import psutil
except Exception:                                          # pragma: no cover
    psutil = None

_BOOT = time.time()


def interfaces():
    """[{name, ip}] for every non-loopback IPv4 interface."""
    out = []
    if psutil:
        for name, addrs in psutil.net_if_addrs().items():
            for a in addrs:
                if a.family == socket.AF_INET and not a.address.startswith("127."):
                    out.append({"name": name, "ip": a.address})
        return out
    # Fallback: one best-effort IP via a UDP connect (no traffic sent).
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        out.append({"name": "default", "ip": s.getsockname()[0]})
        s.close()
    except Exception:
        pass
    return out


def probe(port, host="127.0.0.1", timeout=0.4):
    """True if something is listening on host:port."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


def services():
    """Liveness of the robot-server planes this GUI consumes."""
    return {
        "teleop": {"port": CFG.TELEOP_PORT, "up": probe(CFG.TELEOP_PORT),
                   "what": "control + robot_state (ws /teleop, /monitor)"},
        "map": {"port": CFG.MAP_PORT, "up": probe(CFG.MAP_PORT),
                "what": "SLAM mesh / grid / path (ws /map)"},
        "video": {"port": CFG.VIDEO_PORT, "up": probe(CFG.VIDEO_PORT),
                  "what": "MJPEG / H.264 cameras"},
        "studio": {"port": CFG.PORT, "up": True, "what": "this control center"},
    }


def thermal():
    """Jetson/Linux thermal zones -> {label: degC}."""
    out = {}
    for zone in sorted(glob.glob("/sys/class/thermal/thermal_zone*")):
        try:
            with open(os.path.join(zone, "type")) as f:
                label = f.read().strip()
            with open(os.path.join(zone, "temp")) as f:
                out[label] = round(int(f.read().strip()) / 1000.0, 1)
        except Exception:
            continue
    return out


def _meminfo():
    info = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":", 1)
                info[k] = int(v.strip().split()[0]) * 1024
    except Exception:
        pass
    return info


def stats():
    if psutil:
        vm = psutil.virtual_memory()
        du = psutil.disk_usage("/")
        cpu = psutil.cpu_percent(interval=None)
        mem = {"total": vm.total, "used": vm.used, "pct": vm.percent}
        disk = {"total": du.total, "used": du.used, "pct": du.percent}
    else:
        mi = _meminfo()
        total = mi.get("MemTotal", 0)
        avail = mi.get("MemAvailable", 0)
        mem = {"total": total, "used": total - avail,
               "pct": round(100 * (1 - avail / total), 1) if total else 0}
        disk = {"total": 0, "used": 0, "pct": 0}
        cpu = None
    try:
        load = os.getloadavg()
    except OSError:
        load = (0, 0, 0)
    return {"cpu_pct": cpu, "load": load, "mem": mem, "disk": disk}


def snapshot():
    return {
        "hostname": socket.gethostname(),
        "interfaces": interfaces(),
        "services": services(),
        "thermal": thermal(),
        "uptime_s": round(time.time() - _BOOT, 1),
        **stats(),
    }
