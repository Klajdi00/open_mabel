"""
Sim e-stop: there is no hardware bring-up to kill in simulation — the bridge's
wire-level ESTOP is the whole story (zero base drive + hand-torque gate, and
the control loop holds). The studio's E-STOP button still works identically:
`engage`/`release` forward an `estop` envelope up the backend's own bridge
socket (the SimSource), and `status` reports when telemetry is flowing again.

Kept API-compatible with control_center/server/hardware.py so app.py and the
frontend polling loop are unchanged.
"""
import json
import time

from . import config as CFG


def _send_wire(source, msg_type, payload):
    """Deliver EXACTLY ONCE. When a live SimSource exists, its socket is the
    only path — never fall through to a one-shot on a slow/failed send, or a
    duplicate command (reset!) can land seconds later and silently re-home
    the world after the user has moved on. The one-shot client exists only
    for source-less setups (mock telemetry in tests)."""
    if source is not None and hasattr(source, "send"):
        return bool(source.send(msg_type, payload))
    try:
        import asyncio
        import websockets

        async def _oneshot():
            url = f"ws://{CFG.UPSTREAM_HOST}:{CFG.TELEOP_PORT}/teleop?client=scc-estop"
            if CFG.APP_TOKEN:
                url += f"&key={CFG.APP_TOKEN}"
            async with websockets.connect(url, open_timeout=2,
                                          close_timeout=1) as ws:
                await ws.send(json.dumps({"type": msg_type,
                                          "payload": payload}))
                await asyncio.sleep(0.1)       # let the frame flush
        asyncio.run(_oneshot())
        return True
    except Exception:
        return False


def engage(source=None):
    """E-STOP: the bridge zeroes the base and gates hand torque instantly."""
    ok = _send_wire(source, "estop", {"active": True})
    return {"found": ok, "sim": True,
            "note": "sim e-stop held by the bridge" if ok
            else "bridge unreachable — nothing to stop"}


def release(telemetry_state=None, source=None):
    """Release: clear the bridge e-stop; sim 'bring-up' is instant."""
    ok = _send_wire(source, "estop", {"active": False})
    if telemetry_state is not None and ok:
        telemetry_state.estop = False
    return {"started": ok, "sim": True,
            "note": "" if ok else "bridge unreachable"}


def status(telemetry_state, source=None):
    fresh = (time.time() - telemetry_state.stamp) < 2.0 \
        if telemetry_state.stamp else False
    connected = bool(getattr(source, "connected", False))
    return {"launch_running": connected, "telemetry_fresh": fresh,
            "up": fresh and not telemetry_state.estop, "sim": True}
