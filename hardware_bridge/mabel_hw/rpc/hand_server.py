#!/usr/bin/env python3
"""hand_bridge.py — the standalone, ROS-free ORCA-hand bridge (Mac-capable).

A tiny localhost TCP server that owns the Feetech serial bus through
:class:`hand_rpc.HandBridgeService` and answers newline-delimited JSON requests. It is
"the hardware bridge" for hands wherever ROS/rclpy is unavailable (notably macOS): the
GUI is a pure client of it, exactly as it is a DDS client of the ROS bridge on the robot.

Run it standalone::

    python3 -m mabel_hw.rpc.hand_server          # 127.0.0.1:8779 by default
    python3 -m mabel_hw.rpc.hand_server --port 8780

…or let the GUI start one on demand (see ``ensure_daemon``). Only one daemon binds the
port; a second serial owner is anyway refused by the bus's exclusive lock, so the bridge
stays the single owner of the hardware.

Protocol (one JSON object per line, UTF-8):
    request   {"id": <int>, "op": "<name>", "a": {<args>}}
    response  {"id": <int>, "ok": true,  "r": <result>}
              {"id": <int>, "ok": false, "e": "<message>"}
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
from typing import Optional

# Runnable from a bare checkout: this file lives at
# <bridge>/mabel_hw/rpc/hand_server.py, so the bridge dir is two up. ensure_daemon
# spawns the daemon with cwd=_BRIDGE_DIR so `-m mabel_hw.rpc.hand_server` resolves
# without a pip install.
_HERE = os.path.dirname(os.path.abspath(__file__))
_BRIDGE_DIR = os.path.dirname(os.path.dirname(_HERE))
if _BRIDGE_DIR not in sys.path:
    sys.path.insert(0, _BRIDGE_DIR)

from mabel_hw.rpc.hand_service import HandBridgeService, HandBridgeError  # noqa: E402

DEFAULT_HOST = os.environ.get("MABEL_HAND_BRIDGE_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.environ.get("MABEL_HAND_BRIDGE_PORT", "8779"))


# ── server ────────────────────────────────────────────────────────────────────
class HandBridgeServer:
    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, logger=None):
        self.host, self.port = host, port
        self._log = logger or (lambda *a, **k: None)
        self.service = HandBridgeService(logger=self._log)
        self._sock: Optional[socket.socket] = None
        self._running = False

    def serve_forever(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((self.host, self.port))
        s.listen(8)
        self._sock = s
        self._running = True
        self._log(f"hand-bridge listening on {self.host}:{self.port}")
        try:
            while self._running:
                try:
                    conn, _addr = s.accept()
                except OSError:
                    break
                threading.Thread(target=self._handle, args=(conn,), daemon=True).start()
        finally:
            self.stop()

    def stop(self):
        self._running = False
        try:
            self.service.shutdown()         # disables torque, releases the bus
        except Exception:
            pass
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None

    def _handle(self, conn: socket.socket):
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        rf = conn.makefile("r", encoding="utf-8")
        wf = conn.makefile("w", encoding="utf-8")
        try:
            for line in rf:
                line = line.strip()
                if not line:
                    continue
                try:
                    req = json.loads(line)
                except Exception:
                    continue
                rid = req.get("id")
                try:
                    r = self.service.op(req.get("op", ""), req.get("a") or {})
                    resp = {"id": rid, "ok": True, "r": r}
                except HandBridgeError as e:
                    resp = {"id": rid, "ok": False, "e": str(e)}
                except Exception as e:
                    resp = {"id": rid, "ok": False, "e": f"{type(e).__name__}: {e}"}
                wf.write(json.dumps(resp) + "\n")
                wf.flush()
        except Exception:
            pass
        finally:
            try:
                conn.close()
            except Exception:
                pass


# ── client ──────────────────────────────────────────────────────────────────--
class HandBridgeClient:
    """Synchronous client: one in-flight request at a time (serialised by a lock)."""

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT,
                 timeout: float = 4.0):
        self.host, self.port, self.timeout = host, port, timeout
        self._sock: Optional[socket.socket] = None
        self._rf = None
        self._wf = None
        self._lock = threading.Lock()
        self._id = 0

    @property
    def connected(self) -> bool:
        return self._sock is not None

    def connect(self) -> None:
        s = socket.create_connection((self.host, self.port), timeout=self.timeout)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._sock = s
        self._rf = s.makefile("r", encoding="utf-8")
        self._wf = s.makefile("w", encoding="utf-8")

    def close(self) -> None:
        for f in (self._rf, self._wf):
            try:
                if f is not None:
                    f.close()
            except Exception:
                pass
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
        self._sock = self._rf = self._wf = None

    def call(self, op: str, **args):
        """Send one op and return its result; raises HandBridgeError on failure."""
        with self._lock:
            if self._sock is None:
                raise HandBridgeError("not connected to hand bridge")
            self._id += 1
            rid = self._id
            try:
                self._wf.write(json.dumps({"id": rid, "op": op, "a": args}) + "\n")
                self._wf.flush()
                line = self._rf.readline()
            except Exception as e:
                self.close()
                raise HandBridgeError(f"hand bridge link lost: {e}")
            if not line:
                self.close()
                raise HandBridgeError("hand bridge closed the connection")
            resp = json.loads(line)
            if not resp.get("ok"):
                raise HandBridgeError(resp.get("e", "unknown error"))
            return resp.get("r")


# ── auto-start helpers (used by the GUI) ───────────────────────────────────────
def is_running(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT,
               timeout: float = 0.4) -> bool:
    """True if a hand bridge is already answering on host:port."""
    try:
        c = HandBridgeClient(host, port, timeout=timeout)
        c.connect()
        c.call("hello")
        c.close()
        return True
    except Exception:
        return False


def ensure_daemon(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT,
                  logger=None) -> tuple:
    """Make sure a hand bridge is up, starting one if needed.

    Returns ``(client, how)`` where how is 'existing' or 'started'. Raises if a daemon
    can't be reached/started. The started daemon is a detached subprocess that survives
    this call (so closing one GUI client doesn't kill it).
    """
    log = logger or (lambda *a, **k: None)
    if is_running(host, port):
        c = HandBridgeClient(host, port)
        c.connect()
        return c, "existing"
    # Start a detached daemon and wait for it to accept.
    env = dict(os.environ)
    env["MABEL_HAND_BRIDGE_HOST"] = host
    env["MABEL_HAND_BRIDGE_PORT"] = str(port)
    log(f"starting hand bridge on {host}:{port} …")
    try:
        logf = open("/tmp/mabel_hand_bridge.log", "a")
    except Exception:
        logf = subprocess.DEVNULL
    subprocess.Popen([sys.executable, "-m", "mabel_hw.rpc.hand_server"],
                     cwd=_BRIDGE_DIR, env=env,
                     stdout=logf, stderr=subprocess.STDOUT,
                     start_new_session=True)
    for _ in range(60):                  # up to ~6 s for the port to come up
        if is_running(host, port):
            c = HandBridgeClient(host, port)
            c.connect()
            return c, "started"
        time.sleep(0.1)
    raise HandBridgeError("hand bridge did not come up (see /tmp/mabel_hand_bridge.log)")


# ── main ───────────────────────────────────────────────────────────────────────
def main(argv=None):
    import argparse
    import signal

    ap = argparse.ArgumentParser(description="MABEL ROS-free ORCA hand bridge")
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = ap.parse_args(argv)

    if is_running(args.host, args.port):
        sys.stderr.write(f"[hand_bridge] one already running on {args.host}:{args.port}; "
                         f"refusing to start a second.\n")
        return 3

    def _log(m):
        sys.stderr.write(f"[hand_bridge] {m}\n")
        sys.stderr.flush()

    server = HandBridgeServer(args.host, args.port, logger=_log)

    def _sig(_signum, _frame):
        _log("shutting down (disabling torque, releasing bus)…")
        server.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
