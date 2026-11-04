"""
rpc.py — the ROS-free command/state channel of the simulation bridge, wire-
compatible in style with hardware_bridge/hands/hand_bridge.py (the macOS
pattern): newline-delimited JSON over localhost TCP.

  request   {"id": <int>, "op": "<name>", "a": {<args>}}
  response  {"id": <int>, "ok": true,  "r": <result>}
            {"id": <int>, "ok": false, "e": "<message>"}

The gateway (SimBridgePlant), CLIs, and any non-ROS client command the sim
through this; bulk state rides the memmap, not this socket.
"""
from __future__ import annotations

import json
import logging
import socket
import threading

from . import config as CFG

log = logging.getLogger("simbridge.rpc")


class SimBridgeError(RuntimeError):
    pass


class SimBridgeService:
    """Op dispatch over one SimWorld. One RLock — many clients, safe."""

    def __init__(self, world):
        self.world = world
        self._lock = threading.RLock()

    def op(self, name, a):
        fn = getattr(self, f"_op_{name}", None)
        if fn is None:
            raise SimBridgeError(f"unknown op: {name}")
        with self._lock:
            return fn(**(a or {}))

    # ── ops ──────────────────────────────────────────────────────────────────
    def _op_hello(self):
        return {"server": "mabel_simulation_bridge", "kind": "sim",
                "instance": self.world.instance,
                "scene": self.world.scene,
                "statePath": self.world.state_path}

    def _op_info(self):
        return self.world.info()

    def _op_state(self):
        return self.world.snapshot()

    def _op_set_twist(self, vx=0.0, vy=0.0, wz=0.0):
        self.world.set_twist(vx, vy, wz)
        return True

    def _op_set_estop(self, active=True):
        self.world.set_estop(active)
        return True

    def _op_set_targets(self, targets=None):
        self.world.set_targets(targets or {})
        return True

    def _op_set_effort(self, efforts=None):
        self.world.set_effort(efforts or {})
        return True

    def _op_set_wrench(self, forces=None):
        self.world.set_wrench(forces or {})
        return True

    def _op_set_group_torque(self, group="", enabled=True):
        self.world.set_group_torque(group, enabled)
        return True

    def _op_set_speed_caps(self, vmax=None, wmax=None):
        self.world.set_speed_caps(vmax, wmax)
        return True

    def _op_reset(self, pose=None):
        self.world.reset(pose)
        return True

    def _op_set_state(self, qpos=None):
        return self.world.set_state(qpos)


class SimBridgeServer:
    def __init__(self, service, host=None, port=None):
        self.service = service
        self.host = host or CFG.RPC_HOST
        self.port = port or CFG.rpc_port()
        self._sock = None
        self._stop = threading.Event()

    def start(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((self.host, self.port))
        s.listen(8)
        self._sock = s
        threading.Thread(target=self._accept, daemon=True,
                         name="simbridge-rpc").start()
        log.info("RPC up on %s:%d", self.host, self.port)
        return self

    def stop(self):
        self._stop.set()
        try:
            self._sock and self._sock.close()
        except Exception:
            pass

    def _accept(self):
        while not self._stop.is_set():
            try:
                conn, addr = self._sock.accept()
            except OSError:
                return
            threading.Thread(target=self._serve, args=(conn,), daemon=True,
                             name=f"simbridge-rpc-{addr[1]}").start()

    def _serve(self, conn):
        try:
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            rf = conn.makefile("r", encoding="utf-8", newline="\n")
            wf = conn.makefile("w", encoding="utf-8", newline="\n")
            for line in rf:
                line = line.strip()
                if not line:
                    continue
                try:
                    req = json.loads(line)
                    rid = req.get("id")
                    r = self.service.op(req.get("op", ""), req.get("a"))
                    resp = {"id": rid, "ok": True, "r": r}
                except SimBridgeError as e:
                    resp = {"id": rid, "ok": False, "e": str(e)}
                except Exception as e:              # noqa: BLE001
                    resp = {"id": rid, "ok": False,
                            "e": f"{type(e).__name__}: {e}"}
                wf.write(json.dumps(resp, separators=(",", ":")) + "\n")
                wf.flush()
        except Exception:
            pass
        finally:
            try:
                conn.close()
            except Exception:
                pass


class SimBridgeClient:
    """Synchronous client; one in-flight request, auto-reconnect on error."""

    def __init__(self, host=None, port=None, timeout=2.0):
        self.host = host or CFG.RPC_HOST
        self.port = port or CFG.rpc_port()
        self.timeout = timeout
        self._lock = threading.Lock()
        self._sock = None
        self._rf = self._wf = None
        self._id = 0

    def _connect(self):
        s = socket.create_connection((self.host, self.port),
                                     timeout=self.timeout)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._sock = s
        self._rf = s.makefile("r", encoding="utf-8", newline="\n")
        self._wf = s.makefile("w", encoding="utf-8", newline="\n")

    def close(self):
        with self._lock:
            for h in (self._rf, self._wf, self._sock):
                try:
                    h and h.close()
                except Exception:
                    pass
            self._sock = self._rf = self._wf = None

    def call(self, op, **args):
        with self._lock:
            if self._sock is None:
                self._connect()
            self._id += 1
            try:
                self._wf.write(json.dumps(
                    {"id": self._id, "op": op, "a": args},
                    separators=(",", ":")) + "\n")
                self._wf.flush()
                line = self._rf.readline()
            except Exception as e:
                self.close()
                raise SimBridgeError(f"bridge link lost: {e}") from e
            if not line:
                self.close()
                raise SimBridgeError("bridge closed the connection")
            resp = json.loads(line)
            if not resp.get("ok"):
                raise SimBridgeError(resp.get("e", "bridge error"))
            return resp.get("r")

    def try_call(self, op, **args):
        """call() that swallows link errors (deadman semantics on the bridge
        side make dropped commands safe) — returns None on failure."""
        try:
            return self.call(op, **args)
        except Exception:
            return None


def is_running(host=None, port=None):
    try:
        c = SimBridgeClient(host, port, timeout=0.6)
        r = c.call("hello")
        c.close()
        return r
    except Exception:
        return None
