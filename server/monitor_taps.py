"""monitor_taps.py — producer-side helpers shared by every MABEL bridge so each
one feeds the Command Monitor identically (device identity, network/transport
classification, Bonjour advertisement). Stdlib-only, so it imports on the Linux
robot host too.

Both `sim_teleop_bridge.py` (sim) and `mabel_swerve_bridge.py` (real robot) use
these, so a physical robot shows up in the monitor — with its connected clients —
exactly like a simulation. Pair with `monitor_hub.MONITOR` (the fan-out hub).
"""
from __future__ import annotations

import atexit
import ipaddress
import shutil
import socket
import subprocess
import sys
from urllib.parse import parse_qs

# Explicit ?client= tag → (device, app, platform). Clients self-identify.
CLIENT_TAGS = {
    "visionpro": ("Vision Pro", "MabelTeleop", "visionOS"),
    "avp":       ("Vision Pro", "MabelTeleop", "visionOS"),
    "ios":       ("iPhone",     "MABEL Mobile App", "iOS"),
    "iphone":    ("iPhone",     "MABEL Mobile App", "iOS"),
    "ipad":      ("iPad",       "MABEL Mobile App", "iPadOS"),
    "web":       ("Website",    "teleop-web", "Browser"),
    "browser":   ("Website",    "teleop-web", "Browser"),
    "keyboard":  ("Keyboard",   "ROS node",   "ROS"),
    "gamepad":   ("Gamepad",    "—",          "—"),
    "linux":     ("Linux",      "client",     "Linux"),
    "desktop":   ("Linux",      "client",     "Linux"),
}


def peer_str(ws) -> str:
    addr = getattr(ws, "remote_address", None)
    if isinstance(addr, (tuple, list)) and len(addr) >= 2:
        return f"{addr[0]}:{addr[1]}"
    return str(addr) if addr else "?"


def ws_headers(ws):
    """Handshake request headers across websockets versions: ≤11 exposes
    `ws.request_headers`; ≥12 moves them to `ws.request.headers`."""
    return (getattr(ws, "request_headers", None)
            or getattr(getattr(ws, "request", None), "headers", None))


def parse_web_ua(ua: str) -> tuple[str, str]:
    """(browser, os) from a browser User-Agent, by token sniffing."""
    low = ua.lower()
    if "windows nt" in low:       os_name = "Windows"
    elif "iphone" in low:         os_name = "iPhone"
    elif "ipad" in low:           os_name = "iPad"
    elif "android" in low:        os_name = "Android"
    elif "cros" in low:           os_name = "ChromeOS"
    elif "mac os x" in low or "macintosh" in low: os_name = "macOS"
    elif "linux" in low:          os_name = "Linux"
    else:                         os_name = "—"
    if "edg" in low:              br = "Edge"
    elif "opr/" in low or "opera" in low: br = "Opera"
    elif "crios/" in low:         br = "Chrome"
    elif "fxios/" in low or "firefox/" in low: br = "Firefox"
    elif "samsungbrowser" in low: br = "Samsung"
    elif "chrome/" in low or "chromium/" in low: br = "Chrome"
    elif "safari/" in low:        br = "Safari"
    else:                         br = "Browser"
    return br, os_name


def identify_device(query: str, headers) -> dict:
    """Best-effort client identity: explicit ?client= tag, else User-Agent/Origin
    sniff, else None (hub falls back to a hands heuristic). Resolves browser+OS
    and a per-tab session id for web clients."""
    q = parse_qs(query or "")
    tag = (q.get("client", [""])[0] or "").strip().lower()
    session = (q.get("sid", [""])[0] or "").strip()[:8]

    ua = origin = ""
    if headers is not None:
        try:
            ua = headers.get("User-Agent", "") or ""
            origin = headers.get("Origin", "") or ""
        except Exception:
            pass
    low = ua.lower()

    if tag in CLIENT_TAGS:
        d = CLIENT_TAGS[tag]
        info = {"device": d[0], "app": d[1], "platform": d[2], "src": "tag"}
    elif "mabelteleop" in low:
        info = {"device": "Vision Pro", "app": "MabelTeleop", "platform": "visionOS", "src": "ua"}
    elif "mabel" in low and "mobile" in low:
        info = {"device": "iPhone", "app": "MABEL Mobile App", "platform": "iOS", "src": "ua"}
    elif origin or "mozilla" in low:
        info = {"device": "Website", "app": "browser", "platform": "Browser", "src": "ua"}
    elif "linux" in low:
        info = {"device": "Linux", "app": "client", "platform": "Linux", "src": "ua"}
    else:
        info = {"device": None, "app": None, "platform": None, "src": "none"}

    info["session"] = session
    if info["device"] == "Website" and (ua or origin):
        br, os_name = parse_web_ua(ua)
        info["browser"], info["os"] = br, os_name
        info["app"] = br
        info["platform"] = f"{br} · {os_name}" if os_name != "—" else br
    return info


def classify_network(sock_ip: str) -> str:
    try:
        ip = ipaddress.ip_address(sock_ip)
    except ValueError:
        return "Unknown"
    if ip.is_loopback:
        return "Localhost"
    if ip.version == 4 and ip in ipaddress.ip_network("100.64.0.0/10"):
        return "Tailscale"
    if ip.is_private or ip.is_link_local:
        return "LAN"
    return "Internet"


def network_info(ws, path: str) -> dict:
    """Per-connection networking detail: client IP, transport, secure flag,
    addressed host / origin."""
    peer = peer_str(ws)
    sock_ip, _, port = peer.rpartition(":")
    sock_ip = sock_ip or peer
    headers = ws_headers(ws)

    def h(name: str) -> str:
        try:
            return (headers.get(name, "") or "") if headers is not None else ""
        except Exception:
            return ""

    transport = classify_network(sock_ip)
    secure = transport == "Tailscale"
    return {
        "ip": sock_ip, "port": port,
        "transport": transport, "secure": secure,
        "host": h("Host"), "origin": h("Origin"), "path": path,
    }


def local_ipv4s() -> list[str]:
    """This host's non-loopback IPv4 addresses (LAN + Tailscale)."""
    addrs: set[str] = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            addrs.add(info[4][0])
    except Exception:
        pass
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            addrs.add(s.getsockname()[0])
    except OSError:
        pass
    try:
        out = subprocess.run(["tailscale", "ip", "-4"], capture_output=True,
                             text=True, timeout=2)
        for ln in out.stdout.split():
            if ln.strip():
                addrs.add(ln.strip())
    except Exception:
        pass
    return sorted(a for a in addrs if a and not a.startswith("127."))


def server_info(ws_port: int, *, http_port: int = 0, udp_port: int = 0,
                version: str = "", target: str = "sim:mujoco", name: str = "",
                ws_path: str = "/teleop", monitor_path: str = "/monitor",
                extra: dict | None = None) -> dict:
    """The bridge's own connection info for the monitor's Server card."""
    import platform
    import time
    addrs = local_ipv4s()
    lan = tail = None
    for a in addrs:
        try:
            ip = ipaddress.ip_address(a)
        except ValueError:
            continue
        if ip.version == 4 and ip in ipaddress.ip_network("100.64.0.0/10"):
            tail = a
        elif ip.is_private and lan is None:
            lan = a
    sysname = platform.system().replace("Darwin", "macOS")
    info = {
        "host": socket.gethostname(), "name": name, "os": sysname,
        "platform": f"{sysname} {platform.release()}".strip(),
        "python": platform.python_version(),
        "lanIp": lan or (addrs[0] if addrs else ""), "tailscaleIp": tail,
        "addresses": addrs, "wsPort": ws_port, "wsPath": ws_path,
        "monitorPath": monitor_path, "httpPort": http_port, "udpPort": udp_port,
        "version": version, "target": target, "startedAt": time.time(),
    }
    if extra:
        info.update(extra)
    return info


def advertise_bonjour(name: str, ws_port: int, log=print):
    """Advertise _mabel-teleop._tcp via dns-sd (macOS) or avahi (Linux) so the
    bridge is auto-discovered by clients AND the command monitor. Returns the
    process (kept alive) or None."""
    if sys.platform == "darwin" and shutil.which("dns-sd"):
        cmd = ["dns-sd", "-R", name, "_mabel-teleop._tcp", "local", str(ws_port)]
    elif shutil.which("avahi-publish-service"):
        cmd = ["avahi-publish-service", name, "_mabel-teleop._tcp", str(ws_port)]
    else:
        log(f"[bonjour] no dns-sd/avahi — '{name}' not discoverable "
            f"(install avahi-utils, or add its IP in the monitor)")
        return None
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
        atexit.register(proc.terminate)
        log(f"[bonjour] advertising '{name}' _mabel-teleop._tcp on :{ws_port}")
        return proc
    except Exception as e:
        log(f"[bonjour] advertisement failed: {e}")
        return None
