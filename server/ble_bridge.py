#!/usr/bin/env python3
"""
ble_bridge.py — MABEL as a Bluetooth LE peripheral, so an iPhone or Vision Pro can teleop the
whole body with NO network at all.

    /usr/bin/python3 server/ble_bridge.py --selftest   # protocol + D-Bus objects, no radio
    /usr/bin/python3 server/ble_bridge.py              # advertise and serve

RUN IT WITH /usr/bin/python3, not anaconda: python3-dbus is a system package and is not
installed in the conda env the gateway uses. That is also why this is a SEPARATE PROCESS
from teleop_bridge.py rather than a thread inside it.

WHY BLE AND NOT CLASSIC SPP: iOS refuses Bluetooth Classic (RFCOMM/SPP) to any accessory
without an Apple MFi authentication coprocessor. BLE/GATT carries no such restriction, so
Linux is the PERIPHERAL (GATT server) and the phone/headset is the CENTRAL — the opposite of
the usual "the robot connects out" instinct.

NO VIDEO over this link, by design. iOS negotiates ~15-30 ms connection intervals at
ATT_MTU ~185, which is roughly 5-20 kB/s; the head camera alone wants ~5 Mbit/s. Control and
telemetry fit with room to spare — a whole-body 20-joint pose packs to 102 bytes.

THIS IS A TRANSPORT, NOT A SECOND CONTROLLER. Every GATT write is decoded and re-sent to the
gateway as the SAME wire message the websocket carries, over a loopback websocket, so this
bridge is simply another client in the roster. Whole-body arms, base twist, lift, torso and
neck therefore work here for free (they are already wire messages), e-stop keeps the
ungated path it has on every other transport, and there is exactly one place where a command
becomes motor torque. A parallel control path would mean two things commanding the robot
under different safety rules, which is how robots get hurt.

  service  6d696c6f-0001-4a55-9e00-000000000000    advertised as "MABEL" (the UUIDs are shared
                                                    with NOVA: one BLEProtocol.swift in both apps)
    0002  command  write, write-without-response   JOINT_COMMAND / TELEOP_FRAME / MODE
    0003  estop    write                           highest priority, never gated
    0004  state    notify, read                    joint positions + estop/torque flags
    0005  info     read                            JSON identity + how to reach us on WiFi

The advertisement also carries manufacturer data (company 0xFFFF, the SIG's "internal use"
id) holding the robot kind and the live e-stop/torque flags, so the app's proximity card can
show real status from a passive scan, BEFORE anyone connects.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import socket
import struct
import sys
import threading
import time
from hashlib import sha1

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ble_protocol as P                                              # noqa: E402
from nav_axes import nav_from_twist                                   # noqa: E402  (the ONE motion model)

BLUEZ = "org.bluez"
IF_ADAPTER = "org.bluez.Adapter1"
IF_SERVICE = "org.bluez.GattService1"
IF_CHRC = "org.bluez.GattCharacteristic1"
IF_ADVERT = "org.bluez.LEAdvertisement1"
IF_ADV_MGR = "org.bluez.LEAdvertisingManager1"
IF_GATT_MGR = "org.bluez.GattManager1"
IF_PROPS = "org.freedesktop.DBus.Properties"
IF_OM = "org.freedesktop.DBus.ObjectManager"

BASE = "6d696c6f-{:04x}-4a55-9e00-000000000000"
SVC_UUID = BASE.format(1)
CMD_UUID, ESTOP_UUID, STATE_UUID, INFO_UUID = (BASE.format(n) for n in (2, 3, 4, 5))

COMPANY_ID = 0xFFFF            # Bluetooth SIG "internal / testing" id — we are not a member
ADV_VERSION = 1
NOTIFY_HZ = 20.0               # state pushes; the phone renders at 60 but needs no more data


# ══════════════════════════════════════════════════════════════════════════════
#  Minimal RFC6455 websocket client (stdlib only)
# ══════════════════════════════════════════════════════════════════════════════
class WSClient:
    """Just enough websocket to be a gateway client. `websockets` is not installed under
    /usr/bin/python3 and this needs ~80 lines, so importing a dependency to reach a socket
    on 127.0.0.1 is not worth it."""

    GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

    def __init__(self, host="127.0.0.1", port=9090, path="/teleop", on_msg=None, log=print):
        self.host, self.port, self.path = host, port, path
        self.on_msg, self.log = on_msg, log
        self.sock = None
        self.connected = False
        self._tx = threading.Lock()
        self._stop = threading.Event()

    # ── framing ──────────────────────────────────────────────────────────────
    def _send_frame(self, opcode: int, payload: bytes) -> None:
        n = len(payload)
        head = bytearray([0x80 | opcode])
        if n < 126:
            head.append(0x80 | n)
        elif n < 65536:
            head.append(0x80 | 126); head += struct.pack(">H", n)
        else:
            head.append(0x80 | 127); head += struct.pack(">Q", n)
        mask = os.urandom(4)
        head += mask
        masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        with self._tx:
            if self.sock:
                self.sock.sendall(bytes(head) + masked)

    def _read_exact(self, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("gateway closed the socket")
            buf += chunk
        return buf

    def _read_frame(self):
        b0, b1 = self._read_exact(2)
        opcode, masked, n = b0 & 0x0F, b1 & 0x80, b1 & 0x7F
        if n == 126:
            n = struct.unpack(">H", self._read_exact(2))[0]
        elif n == 127:
            n = struct.unpack(">Q", self._read_exact(8))[0]
        mask = self._read_exact(4) if masked else b""
        data = self._read_exact(n)
        if masked:
            data = bytes(b ^ mask[i % 4] for i, b in enumerate(data))
        return opcode, data

    # ── lifecycle ────────────────────────────────────────────────────────────
    def _handshake(self) -> None:
        key = base64.b64encode(os.urandom(16)).decode()
        req = (f"GET {self.path} HTTP/1.1\r\nHost: {self.host}:{self.port}\r\n"
               f"Upgrade: websocket\r\nConnection: Upgrade\r\n"
               f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n"
               f"User-Agent: mabel-ble-bridge\r\n\r\n")
        self.sock.sendall(req.encode())
        head = b""
        while b"\r\n\r\n" not in head:
            chunk = self.sock.recv(1024)
            if not chunk:
                raise ConnectionError("gateway closed during handshake")
            head += chunk
        if b"101" not in head.split(b"\r\n", 1)[0]:
            raise ConnectionError(f"handshake refused: {head.split(chr(13).encode(), 1)[0]!r}")
        want = base64.b64encode(sha1((key + self.GUID).encode()).digest()).decode().lower()
        if want.encode() not in head.lower():
            raise ConnectionError("handshake accept key mismatch")

    def start(self) -> None:
        threading.Thread(target=self._run, daemon=True, name="ble-ws").start()

    def _run(self) -> None:
        backoff = 0.5
        while not self._stop.is_set():
            try:
                self.sock = socket.create_connection((self.host, self.port), timeout=5)
                self.sock.settimeout(None)
                self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self._handshake()
                self.connected = True
                backoff = 0.5
                self.log(f"[ble] gateway link up  ws://{self.host}:{self.port}{self.path}")
                while not self._stop.is_set():
                    opcode, data = self._read_frame()
                    if opcode == 0x8:                                  # close
                        break
                    if opcode == 0x9:                                  # ping -> pong
                        self._send_frame(0xA, data); continue
                    if opcode in (0x1, 0x2) and self.on_msg:
                        try:
                            self.on_msg(json.loads(data.decode("utf-8", "replace")))
                        except (ValueError, TypeError):
                            pass
            except Exception as e:                                     # noqa: BLE001
                if not self._stop.is_set():
                    self.log(f"[ble] gateway link down ({e}); retrying in {backoff:.1f}s")
            finally:
                self.connected = False
                try:
                    if self.sock:
                        self.sock.close()
                except OSError:
                    pass
                self.sock = None
            self._stop.wait(backoff)
            backoff = min(5.0, backoff * 1.6)

    def send(self, mtype: str, payload: dict) -> bool:
        """Fire-and-forget. Returns False when the gateway link is down, so the caller can
        say so on the BLE side rather than pretending the command landed."""
        if not self.connected:
            return False
        try:
            self._send_frame(0x1, json.dumps({"type": mtype, "payload": payload}).encode())
            return True
        except OSError as e:
            self.log(f"[ble] send failed: {e}")
            return False

    def stop(self) -> None:
        self._stop.set()
        try:
            if self.sock:
                self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass


# ══════════════════════════════════════════════════════════════════════════════
#  BLE  ->  gateway
# ══════════════════════════════════════════════════════════════════════════════
def _actuator_order(mid: str, names: list) -> list:
    """The per-module actuator ORDER the apps label by index (BLEProtocol.swift
    moduleLabels): arms J1..J7 by number, the neck yaw/pitch/roll, the rest sorted."""
    if mid in ("leftArm", "rightArm"):
        return sorted(names, key=lambda n: int(n.rsplit("_", 1)[-1]) if n.rsplit("_", 1)[-1].isdigit() else 99)
    if mid == "neck":
        want = ("neck_yaw", "neck_pitch", "neck_roll")
        return [n for n in want if n in names] + sorted(n for n in names if n not in want)
    return sorted(names)


# status.torque key per module, when no motor reports its own `enabled` flag
_TORQUE_KEY = {"leftArm": "arms", "rightArm": "arms", "torso": "neck", "neck": "neck",
               "base": "base", "lift": "lift", "leftHand": "left_hand", "rightHand": "right_hand"}


def _health_rows(roster: list, health: dict, positions: dict, torque: dict) -> list:
    """MABEL's robot_state — module_status roster rows + per-motor motorHealth + the joint
    stream — folded into the rows ble_protocol.encode_health packs (one bit per motor).
    A motor the firmware has never heard carries no `live` flag, only stale_ms = uptime,
    so age decides when the flag is absent; a joint with no health at all counts as up
    while it is in the stream (hands, base wheels)."""
    try:
        from module_status import module_of
    except Exception:                                              # noqa: BLE001
        return []
    by_mod: dict = {}
    for n in set(positions) | set(health):
        mid = module_of(n)
        if mid:
            by_mod.setdefault(mid, []).append(n)
    rows = []
    for r in roster:
        mid = r.get("id")
        if mid not in P.MODULE_INDEX:
            continue
        acts, enabled = [], []
        for n in _actuator_order(mid, by_mod.get(mid, [])):
            h = health.get(n) or {}
            if "live" in h:
                ok = bool(h["live"])
            elif h.get("stale_ms") is not None:
                ok = float(h["stale_ms"]) < 1000.0
            else:
                ok = n in positions
            acts.append({"id": n, "ok": ok})
            if h.get("enabled") is not None:
                enabled.append(bool(h["enabled"]))
        if mid == "lift" and acts:                 # one column, two URDF stages: ONE actuator
            acts = [{"id": "lift", "ok": any(a["ok"] for a in acts)}]
        tq = all(enabled) if enabled else bool(torque.get(_TORQUE_KEY.get(mid, ""), False))
        rows.append({"id": mid, "ok": bool(r.get("ok")), "n": len(acts),
                     "up": sum(1 for a in acts if a["ok"]), "torque": tq, "actuators": acts})
    return rows


class Bridge:
    """Decodes BLE writes into gateway wire messages. Holds no control state of its own."""

    def __init__(self, ws: WSClient, log=print):
        self.ws, self.log = ws, log
        self.state = {"positions": {}, "estop": True, "torque": False}
        # The per-module / per-actuator roster, latched from robot_state.modules.
        self.modules: list = []
        self._roster: list = []         # MABEL: module_status rows (every ~2 s)
        self._health: dict = {}         # MABEL: per-motor motorHealth (~4 Hz)
        self._tick = 0
        self.health_every = max(1, int(NOTIFY_HZ))               # one health frame a second
        self.n_cmd = 0
        self.n_bad = 0
        self.last_cmd = 0.0

    # inbound from the phone ---------------------------------------------------
    def on_write(self, data: bytes) -> None:
        try:
            kind, payload = P.decode(bytes(data))
        except P.BleProtocolError as e:
            self.n_bad += 1
            self.log(f"[ble] dropped malformed frame: {e}")
            return                                    # a bad packet is dropped, never guessed
        self.n_cmd += 1
        self.last_cmd = time.monotonic()
        if kind == P.JOINT_COMMAND:
            self.ws.send("joint_command", {"joints": payload})
        elif kind == P.TELEOP_FRAME:
            # The gateway's TELEOP_FRAME takes normalised stick axes and applies its own
            # speed caps; ble_protocol carries a twist. Invert the gateway's mapping
            # (vx = ly*vmax, vy = -lx*vmax, wz = rx*wmax) with unit caps, so the SERVER's
            # caps stay the single authority on how fast the base may actually move.
            # ry is the LIFT STICK, and it goes to the gateway as a rate — the same
            # channel Wi-Fi uses, so the column is driven by open-loop duty on both
            # transports instead of the app integrating a position on one of them.
            # ONE motion model: nav_from_twist is the exact inverse of what the gateway applies,
            # so this transport cannot drift away from the others' idea of which way is forward.
            nav = nav_from_twist(payload["vx"], payload["vy"], payload["wz"])
            nav.pop("ry", None)                        # absent unless the frame carries the lift stick
            if "lift" in payload:
                nav["ry"] = _unit(payload["lift"])
            self.ws.send("teleop_frame", {"navJoystick": nav})
        elif kind == P.HAND_POSE:
            # the headset's hand over Bluetooth: the same HandPose a Wi-Fi teleop_frame carries,
            # so the gateway retargets it on the same path (controller/hand_retarget.py)
            key = "leftHand" if payload["side"] == "left" else "rightHand"
            self.ws.send("teleop_frame", {key: P.hand_pose_payload(payload["side"], payload["points"], payload["tracked"])})
        elif kind == P.HEAD_POSE:
            self.ws.send("teleop_frame", {"head": P.head_pose_payload(payload["R"], payload["tracked"])})
        elif kind == P.ESTOP:
            self.ws.send("estop", payload)
        elif kind == P.CONTROL_MODE:
            self.ws.send("control_mode", payload)

    # inbound from the gateway -------------------------------------------------
    def on_gateway(self, msg: dict) -> None:
        if msg.get("type") != "robot_state":
            return
        p = msg.get("payload") or {}
        tq = p.get("torque") or {}
        # NOVA's gateway summarises as torque.any; MABEL's sends one flag per group
        # (arms / neck / base / left_hand / right_hand) — energised if ANY group is.
        any_on = bool(tq.get("any")) if "any" in tq else any(v is True for v in tq.values())
        self.state = {"positions": p.get("jointPositions") or {},
                      "estop": bool(p.get("estop")),
                      "torque": any_on}
        # LATCHED, because the gateway sends the roster only when it changes (see
        # Gateway._modules). Overwriting it with "absent" every frame would leave the
        # phone's status panel dark between updates.
        mods = p.get("modules")
        if mods:
            if any(isinstance(r, dict) and "actuators" in r for r in mods):
                self.modules = mods             # already per-actuator rows (NOVA's gateway)
            else:
                self._roster = mods             # MABEL: module_status rows, folded below
        if isinstance(p.get("motorHealth"), dict):
            self._health = p["motorHealth"]
        if self._roster:
            self.modules = _health_rows(self._roster, self._health, self.state["positions"], tq)

    def state_frame(self) -> bytes:
        s = self.state
        return P.encode_state(s["positions"], estop=s["estop"], torque=s["torque"])

    def next_frame(self) -> bytes:
        """What to notify this tick. The POSE is the hot path — it is what you are
        driving — so the module roster rides one frame a second in its place. A motor
        dropping off the bus is not a 20 Hz event, and at 35 bytes the health frame costs
        one skipped pose update a second, which is invisible while it is the difference
        between a status panel that works over Bluetooth and one that shows nothing."""
        self._tick += 1
        if self.modules and self._tick % self.health_every == 0:
            try:
                return P.encode_health(self.modules)
            except Exception:                                    # noqa: BLE001
                pass                                             # never lose the pose to it
        return self.state_frame()

    def adv_data(self) -> list:
        """The passive-scan payload: version, kind, flags. Lets the phone's proximity card
        show live status without connecting."""
        flags = (0x01 if self.state["estop"] else 0) | (0x02 if self.state["torque"] else 0) \
                | (0x04 if self.ws.connected else 0)
        return [ADV_VERSION, 0x00, flags]             # 0x00 = MABEL (0x01 = NOVA; BLEProtocol.swift isNova)


def _unit(v: float) -> float:
    return max(-1.0, min(1.0, float(v)))


# ══════════════════════════════════════════════════════════════════════════════
#  BlueZ GATT peripheral
# ══════════════════════════════════════════════════════════════════════════════
def run_peripheral(bridge: Bridge, adapter_name: str, local_name: str, info: dict, log=print):
    import dbus
    import dbus.mainloop.glib
    import dbus.service
    from gi.repository import GLib

    dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
    bus = dbus.SystemBus()

    # find the adapter -------------------------------------------------------
    om = dbus.Interface(bus.get_object(BLUEZ, "/"), IF_OM)
    objects = om.GetManagedObjects()
    path = next((p for p, ifs in objects.items()
                 if IF_ADAPTER in ifs and (not adapter_name or p.endswith(adapter_name))), None)
    if path is None:
        raise RuntimeError("no Bluetooth adapter — is bluetoothd running?")
    props = dbus.Interface(bus.get_object(BLUEZ, path), IF_PROPS)
    props.Set(IF_ADAPTER, "Powered", dbus.Boolean(True))
    addr = str(props.Get(IF_ADAPTER, "Address"))
    log(f"[ble] adapter {path.rsplit('/', 1)[-1]}  {addr}")

    class Chrc(dbus.service.Object):
        def __init__(self, index, uuid, flags, on_write=None, reader=None):
            self.path = f"/mabel/ble/service0/char{index}"
            self.uuid, self.flags = uuid, flags
            self.on_write, self.reader = on_write, reader
            self.notifying = False
            self.value = b""
            super().__init__(bus, self.path)

        def props(self):
            return {"Service": dbus.ObjectPath("/mabel/ble/service0"),
                    "UUID": dbus.String(self.uuid),
                    "Flags": dbus.Array(self.flags, signature="s"),
                    "Notifying": dbus.Boolean(self.notifying),
                    "Value": dbus.Array(self.value, signature="y")}

        @dbus.service.method(IF_PROPS, in_signature="s", out_signature="a{sv}")
        def GetAll(self, interface):
            if interface != IF_CHRC:
                raise dbus.exceptions.DBusException("org.bluez.Error.InvalidArguments")
            return self.props()

        @dbus.service.method(IF_PROPS, in_signature="ss", out_signature="v")
        def Get(self, interface, name):
            return self.props()[name]

        @dbus.service.method(IF_CHRC, in_signature="a{sv}", out_signature="ay")
        def ReadValue(self, options):
            if self.reader:
                self.value = self.reader()
            return dbus.Array(self.value, signature="y")

        @dbus.service.method(IF_CHRC, in_signature="aya{sv}", out_signature="")
        def WriteValue(self, value, options):
            if self.on_write:
                self.on_write(bytes(bytearray(value)))

        @dbus.service.method(IF_CHRC, in_signature="", out_signature="")
        def StartNotify(self):
            self.notifying = True
            log("[ble] central subscribed to state")

        @dbus.service.method(IF_CHRC, in_signature="", out_signature="")
        def StopNotify(self):
            self.notifying = False

        @dbus.service.signal(IF_PROPS, signature="sa{sv}as")
        def PropertiesChanged(self, interface, changed, invalidated):
            pass

        def notify(self, payload: bytes):
            """Push state. Silently skipped when nobody subscribed — the robot must not care
            whether a phone happens to be listening."""
            if not self.notifying:
                return
            self.value = payload
            self.PropertiesChanged(IF_CHRC, {"Value": dbus.Array(payload, signature="y")}, [])

    class Service(dbus.service.Object):
        PATH = "/mabel/ble/service0"

        def __init__(self):
            super().__init__(bus, self.PATH)

        def props(self):
            return {"UUID": dbus.String(SVC_UUID), "Primary": dbus.Boolean(True)}

        @dbus.service.method(IF_PROPS, in_signature="s", out_signature="a{sv}")
        def GetAll(self, interface):
            return self.props()

        @dbus.service.method(IF_PROPS, in_signature="ss", out_signature="v")
        def Get(self, interface, name):
            return self.props()[name]

    class Application(dbus.service.Object):
        PATH = "/mabel/ble"

        def __init__(self, service, chrcs):
            self.service, self.chrcs = service, chrcs
            super().__init__(bus, self.PATH)

        @dbus.service.method(IF_OM, out_signature="a{oa{sa{sv}}}")
        def GetManagedObjects(self):
            out = {dbus.ObjectPath(Service.PATH): {IF_SERVICE: self.service.props()}}
            for c in self.chrcs:
                out[dbus.ObjectPath(c.path)] = {IF_CHRC: c.props()}
            return out

    class Advert(dbus.service.Object):
        PATH = "/mabel/ble/advert0"

        def __init__(self):
            super().__init__(bus, self.PATH)

        def props(self):
            return {"Type": dbus.String("peripheral"),
                    "ServiceUUIDs": dbus.Array([SVC_UUID], signature="s"),
                    "LocalName": dbus.String(local_name),
                    "ManufacturerData": dbus.Dictionary(
                        {dbus.UInt16(COMPANY_ID): dbus.Array(bridge.adv_data(), signature="y")},
                        signature="qv"),
                    "Includes": dbus.Array(["tx-power"], signature="s"),
                    "Discoverable": dbus.Boolean(True)}

        @dbus.service.method(IF_PROPS, in_signature="s", out_signature="a{sv}")
        def GetAll(self, interface):
            return self.props()

        @dbus.service.method(IF_PROPS, in_signature="ss", out_signature="v")
        def Get(self, interface, name):
            return self.props()[name]

        @dbus.service.method(IF_ADVERT, in_signature="", out_signature="")
        def Release(self):
            log("[ble] advertisement released by bluetoothd")

    # build ------------------------------------------------------------------
    svc = Service()
    state_chrc = Chrc(4, STATE_UUID, ["notify", "read"], reader=bridge.state_frame)
    chrcs = [
        Chrc(2, CMD_UUID, ["write", "write-without-response"], on_write=bridge.on_write),
        Chrc(3, ESTOP_UUID, ["write", "write-without-response"], on_write=bridge.on_write),
        state_chrc,
        Chrc(5, INFO_UUID, ["read"], reader=lambda: json.dumps(info).encode()),
    ]
    app = Application(svc, chrcs)
    advert = Advert()

    loop = GLib.MainLoop()

    def ok(what):
        return lambda: log(f"[ble] {what} registered")

    def fail(what):
        def _f(e):
            log(f"[ble] {what} FAILED: {e}")
            loop.quit()
        return _f

    dbus.Interface(bus.get_object(BLUEZ, path), IF_GATT_MGR).RegisterApplication(
        app.PATH, {}, reply_handler=ok("gatt app"), error_handler=fail("gatt app"))
    dbus.Interface(bus.get_object(BLUEZ, path), IF_ADV_MGR).RegisterAdvertisement(
        advert.PATH, {}, reply_handler=ok("advertisement"), error_handler=fail("advertisement"))

    def tick():
        state_chrc.notify(bridge.next_frame())
        return True

    GLib.timeout_add(int(1000 / NOTIFY_HZ), tick)
    log(f"[ble] advertising as '{local_name}'  service {SVC_UUID}")
    log("[ble] on the phone: open the app — the MABEL card appears when you are in range")
    try:
        loop.run()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            dbus.Interface(bus.get_object(BLUEZ, path), IF_ADV_MGR
                           ).UnregisterAdvertisement(advert.PATH)
        except Exception:                                              # noqa: BLE001
            pass
        log("[ble] stopped")



# ══════════════════════════════════════════════════════════════════════════════
#  CoreBluetooth GATT peripheral (macOS)
# ══════════════════════════════════════════════════════════════════════════════
def run_peripheral_corebluetooth(bridge: Bridge, local_name: str, info: dict, log=print):
    """The same service, the same UUIDs, the same frames — on the Mac's own radio.

    BlueZ is Linux-only, so bringing the robot up on the Mac meant no BLE at all and the
    phone could only reach it over Wi-Fi. CoreBluetooth does the peripheral role natively
    and PyObjC is already in the gateway's environment, so this needs no new dependency.

    Only `run_peripheral` above is BlueZ-specific: WSClient and Bridge are transport-neutral,
    so this backend is just the radio half and every frame, UUID and safety rule is shared.

    TWO honest differences from the BlueZ path:
      · NO MANUFACTURER DATA. CBPeripheralManager advertises a local name and service UUIDs
        and nothing else — Apple does not expose the manufacturer field to peripherals. The
        app's proximity card therefore cannot read e-stop/torque from a passive scan here;
        it gets them the moment it connects and reads/subscribes to `state` (0004).
      · macOS gates Bluetooth per APPLICATION. The first run prompts, and the permission is
        granted to the terminal (or IDE) that launched python, not to python itself. If no
        prompt appears and the state never reaches PoweredOn, that is the block — grant
        Bluetooth to the terminal in System Settings ▸ Privacy & Security ▸ Bluetooth.
    """
    import objc
    from CoreBluetooth import (CBAdvertisementDataLocalNameKey,
                               CBAdvertisementDataServiceUUIDsKey,
                               CBATTErrorSuccess,
                               CBAttributePermissionsReadable,
                               CBAttributePermissionsWriteable,
                               CBCharacteristicPropertyNotify,
                               CBCharacteristicPropertyRead,
                               CBCharacteristicPropertyWrite,
                               CBCharacteristicPropertyWriteWithoutResponse,
                               CBManagerStatePoweredOn,
                               CBMutableCharacteristic, CBMutableService,
                               CBPeripheralManager, CBUUID)
    from Foundation import NSData, NSDate, NSObject, NSRunLoop

    uuid = lambda u: CBUUID.UUIDWithString_(u)
    data = lambda b: NSData.dataWithBytes_length_(bytes(b), len(bytes(b)))

    # `write` (with response) AND `write-without-response` on both command paths, matching
    # the BlueZ flags: the app streams poses without responses and sends e-stop with one.
    W = CBCharacteristicPropertyWrite | CBCharacteristicPropertyWriteWithoutResponse
    cmd_c = CBMutableCharacteristic.alloc().initWithType_properties_value_permissions_(
        uuid(CMD_UUID), W, None, CBAttributePermissionsWriteable)
    estop_c = CBMutableCharacteristic.alloc().initWithType_properties_value_permissions_(
        uuid(ESTOP_UUID), W, None, CBAttributePermissionsWriteable)
    state_c = CBMutableCharacteristic.alloc().initWithType_properties_value_permissions_(
        uuid(STATE_UUID), CBCharacteristicPropertyNotify | CBCharacteristicPropertyRead,
        None, CBAttributePermissionsReadable)
    # A read-only characteristic with a CONSTANT value is handed to CoreBluetooth up front;
    # it then answers reads itself and never calls the delegate for it.
    info_c = CBMutableCharacteristic.alloc().initWithType_properties_value_permissions_(
        uuid(INFO_UUID), CBCharacteristicPropertyRead,
        data(json.dumps(info).encode()), CBAttributePermissionsReadable)

    service = CBMutableService.alloc().initWithType_primary_(uuid(SVC_UUID), True)
    service.setCharacteristics_([cmd_c, estop_c, state_c, info_c])

    state = {"subscribers": 0, "ready": True, "advertising": False}

    class Delegate(NSObject):
        def peripheralManagerDidUpdateState_(self, mgr):
            if mgr.state() != CBManagerStatePoweredOn:
                log(f"[ble] bluetooth not powered on (state {mgr.state()}) — "
                    "check System Settings > Privacy & Security > Bluetooth")
                return
            mgr.addService_(service)

        def peripheralManager_didAddService_error_(self, mgr, svc, err):
            if err is not None:
                log(f"[ble] gatt service FAILED: {err}")
                return
            log("[ble] gatt app registered")
            mgr.startAdvertising_({CBAdvertisementDataLocalNameKey: local_name,
                                   CBAdvertisementDataServiceUUIDsKey: [uuid(SVC_UUID)]})

        def peripheralManagerDidStartAdvertising_error_(self, mgr, err):
            if err is not None:
                log(f"[ble] advertisement FAILED: {err}")
                return
            state["advertising"] = True
            log("[ble] advertisement registered")

        def peripheralManager_didReceiveWriteRequests_(self, mgr, requests):
            # EVERY request is executed; only the first is answered, which is what
            # CoreBluetooth requires for a batch.
            for req in requests:
                val = req.value()
                if val is not None:
                    bridge.on_write(bytes(val))
            if len(requests):
                mgr.respondToRequest_withResult_(requests[0], CBATTErrorSuccess)

        def peripheralManager_didReceiveReadRequest_(self, mgr, request):
            if request.characteristic().UUID().UUIDString().lower() == STATE_UUID:
                request.setValue_(data(bridge.state_frame()))
            mgr.respondToRequest_withResult_(request, CBATTErrorSuccess)

        def peripheralManager_central_didSubscribeToCharacteristic_(self, mgr, central, chrc):
            state["subscribers"] += 1
            log(f"[ble] central subscribed ({state['subscribers']} listening)")

        def peripheralManager_central_didUnsubscribeFromCharacteristic_(self, mgr, central, chrc):
            state["subscribers"] = max(0, state["subscribers"] - 1)

        def peripheralManagerIsReadyToUpdateSubscribers_(self, mgr):
            state["ready"] = True                      # the send queue drained

    delegate = Delegate.alloc().init()
    mgr = CBPeripheralManager.alloc().initWithDelegate_queue_options_(delegate, None, None)
    log(f"[ble] advertising as '{local_name}'  service {SVC_UUID}")
    log("[ble] on the phone: open the app — the MABEL card appears when you are in range")
    period = 1.0 / NOTIFY_HZ
    try:
        while True:
            # Drive CoreBluetooth's callbacks, then push one state frame. updateValue
            # returns False when the queue is full; the delegate above re-arms us, and the
            # frame is simply skipped — state is a SNAPSHOT, so the newest one is the only
            # one worth sending.
            NSRunLoop.currentRunLoop().runMode_beforeDate_(
                "kCFRunLoopDefaultMode", NSDate.dateWithTimeIntervalSinceNow_(period))
            if state["subscribers"] and state["ready"]:
                state["ready"] = bool(mgr.updateValue_forCharacteristic_onSubscribedCentrals_(
                    data(bridge.next_frame()), state_c, None))
    except KeyboardInterrupt:
        pass
    finally:
        try:
            mgr.stopAdvertising()
            mgr.removeAllServices()
        except Exception:                                              # noqa: BLE001
            pass
        log("[ble] stopped")


# ══════════════════════════════════════════════════════════════════════════════
def _lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80)); return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def selftest() -> int:
    ws = WSClient()
    bridge = Bridge(ws, log=lambda *a: None)
    sent = []
    ws.send = lambda t, p: (sent.append((t, p)), True)[1]              # type: ignore[assignment]
    ws.connected = True

    frames = [("whole-body pose", P.encode_joint_command(
                  {n: 0.1 * i for i, n in enumerate(P.JOINTS)})),
              ("two joints", P.encode_joint_command({"left_arm_2": -0.3, "torso": -1.0})),
              ("base twist", bytes([P.TELEOP_FRAME]) + struct.pack("<fff", 0.4, 0.0, 0.2)),
              ("estop engage", bytes([P.ESTOP, 1])),
              ("soft + gravity", bytes([P.CONTROL_MODE, 0x03])),
              ("malformed", b"\x99bad")]
    for label, f in frames:
        n0 = len(sent)
        bridge.on_write(f)
        got = sent[-1] if len(sent) > n0 else None
        print(f"  {label:<16} {len(f):>4} B -> {got[0] if got else 'DROPPED':<14}"
              f" {str(got[1])[:58] if got else ''}")
    assert len(sent) == 5, f"expected 5 dispatches, got {len(sent)}"
    assert bridge.n_bad == 1, "malformed frame was not rejected"

    bridge.on_gateway({"type": "robot_state", "payload": {
        "jointPositions": {"left_arm_1": 0.25, "torso": -1.047},
        "estop": False, "torque": {"any": True}}})
    st = bridge.state_frame()
    pos, es, tq = P.decode_state(st)
    print(f"  state frame      {len(st):>4} B -> {pos} estop={es} torque={tq}")
    assert abs(pos["torso"] + 1.047) < 1e-6 and not es and tq   # f32 round-trip
    print(f"  adv payload      {len(bridge.adv_data()):>4} B -> {bridge.adv_data()}")

    # THE LIFT STICK rides the teleop frame, and an OLDER app that cannot send it must not
    # be read as asking for a stop.
    n0 = len(sent)
    bridge.on_write(P.encode_teleop_frame(0.0, 0.0, 0.0, -0.4))
    nav = sent[-1][1]["navJoystick"]
    print(f"  lift stick       {len(P.encode_teleop_frame(0,0,0,-0.4)):>4} B -> ry={nav.get('ry')}")
    assert abs(nav["ry"] + 0.4) < 1e-6, "the lift stick did not reach navJoystick.ry"
    bridge.on_write(P.encode_teleop_frame(0.2, 0.0, 0.0, 0.9))
    assert abs(sent[-1][1]["navJoystick"]["ry"] - 0.9) < 1e-6
    import struct as _s
    bridge.on_write(bytes([P.TELEOP_FRAME]) + _s.pack("<fff", 0.2, 0.0, 0.0))   # old app
    assert "ry" not in sent[-1][1]["navJoystick"], \
        "a 12-byte frame has no lift stick; it must not be turned into ry=0"
    print(f"  old 12 B frame       -> no ry (not read as a stop)")
    assert len(sent) == n0 + 3

    # HEALTH: the roster the status panel renders, packed small enough for one write.
    # A motor that has dropped off the bus must survive the round trip as ONE BIT in the
    # right place — that bit is the whole point of the frame.
    bridge.on_gateway({"type": "robot_state", "payload": {"jointPositions": {}, "estop": False,
        "torque": {"any": False}, "modules": [
            {"id": "rightArm", "label": "Right Arm", "ok": False, "up": 5, "n": 7, "torque": True,
             "actuators": [{"id": f"right_arm_{i}", "label": f"J{i}", "ok": i not in (3, 6),
                            "detail": ""} for i in range(1, 8)]},
            {"id": "lift", "label": "Lift", "ok": True, "up": 1, "n": 1,
             "actuators": [{"id": "lift", "label": "Column", "ok": True, "detail": "675 mm"}]},
            {"id": "leftHand", "label": "Left Hand", "ok": False, "up": 0, "n": 0, "actuators": []}]}})
    bridge._tick = bridge.health_every - 1                       # land exactly on a health tick
    hf = bridge.next_frame()
    rows = {r["id"]: r for r in P.decode_health(hf)}
    print(f"  health frame     {len(hf):>4} B -> " +
          " ".join(f"{k} {v['up']}/{v['n']}" for k, v in rows.items()))
    assert len(hf) <= P.MAX_PAYLOAD, "health frame must fit ONE write"
    assert rows["rightArm"]["live"] == [True, True, False, True, True, False, True], "wrong motor flagged"
    assert rows["rightArm"]["up"] == 5 and rows["rightArm"]["torque"]
    assert rows["lift"]["ok"] and rows["lift"]["up"] == 1
    assert not rows["leftHand"]["hasDriver"], "a module with no driver must not read as broken"
    assert P.decode_state(bridge.next_frame()), "the pose must resume after a health frame"

    # MABEL's gateway shape: module_status roster rows + per-motor motorHealth (no `any`
    # torque key, no per-actuator rows). The right shoulder2 and the two neck servos that
    # never answered the Teensy (2026-09-30) must come out as exactly those bits.
    bridge.on_gateway({"type": "robot_state", "payload": {
        "jointPositions": {**{f"{sd}_arm_{i}": 0.0 for sd in ("left", "right") for i in range(1, 8)},
                           "torso": 0.0, "neck_yaw": 0.0, "neck_pitch": 0.0, "neck_roll": 0.0,
                           "lift_lower": 0.1, "lift_upper": 0.1},
        "estop": False, "torque": {"arms": True, "neck": False, "left_hand": False},
        "modules": [{"id": "rightArm", "label": "Right Arm", "group": None, "ok": True, "detail": "7 joints"},
                    {"id": "neck", "label": "Neck", "group": None, "ok": True, "detail": "3 joints"},
                    {"id": "lift", "label": "Lift", "group": None, "ok": False, "detail": "not connected"},
                    {"id": "leftHand", "label": "Left Hand", "group": None, "ok": False, "detail": "not connected"}],
        "motorHealth": {**{f"right_arm_{i}": {"live": True, "enabled": True, "stale_ms": 3} for i in (1, 3, 4, 5, 6, 7)},
                        "right_arm_2": {"stale_ms": 5020544},
                        "neck_yaw": {"live": True, "enabled": False, "stale_ms": 4},
                        "neck_pitch": {"stale_ms": 5020544}, "neck_roll": {"stale_ms": 5020544}}}})
    bridge._tick = bridge.health_every - 1
    hf = bridge.next_frame()
    rows = {r["id"]: r for r in P.decode_health(hf)}
    print(f"  MABEL health     {len(hf):>4} B -> " +
          " ".join(f"{k} {v['up']}/{v['n']}" for k, v in rows.items()) + f"  torque(any)={bridge.state['torque']}")
    assert rows["rightArm"]["up"] == 6 and rows["rightArm"]["n"] == 7 and rows["rightArm"]["torque"]
    assert rows["rightArm"]["live"] == [True, False, True, True, True, True, True], "shoulder2 must be the dead bit"
    assert rows["neck"]["live"] == [True, False, False], "yaw/pitch/roll order: yaw up, servos dead"
    assert rows["lift"]["n"] == 1 and not rows["lift"]["ok"], "the column is ONE actuator, offline per the roster"
    assert not rows["leftHand"]["hasDriver"]
    assert bridge.state["torque"], "MABEL's per-group torque dict must fold to 'any'"

    if sys.platform == "darwin":
        try:
            import CoreBluetooth  # noqa: F401
            import objc  # noqa: F401
            print("  CoreBluetooth    OK (peripheral can start)")
        except Exception as e:                                         # noqa: BLE001
            print(f"  CoreBluetooth    MISSING ({e}) — pip install pyobjc-framework-CoreBluetooth")
            return 1
    else:
        try:
            import dbus  # noqa: F401
            from gi.repository import GLib  # noqa: F401
            print("  dbus + GLib      OK (peripheral can start)")
        except Exception as e:                                         # noqa: BLE001
            print(f"  dbus + GLib      MISSING ({e}) — run with /usr/bin/python3")
            return 1
    print("  selftest PASSED")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="MABEL BLE teleop peripheral")
    ap.add_argument("--selftest", action="store_true", help="protocol check, no radio")
    ap.add_argument("--adapter", default="", help="e.g. hci0 (default: first adapter)")
    ap.add_argument("--name", default=os.getenv("MABEL_BLE_NAME", "MABEL"),
                    help="advertised name, shown on the phone's connect card")
    ap.add_argument("--ws-host", default="127.0.0.1")
    ap.add_argument("--ws-port", type=int, default=int(os.getenv("MABEL_WS_PORT", "9090")))
    a = ap.parse_args()

    if a.selftest:
        return selftest()

    ws = WSClient(a.ws_host, a.ws_port, path="/teleop?client=ble")   # the gateway names clients by tag
    bridge = Bridge(ws)
    ws.on_msg = bridge.on_gateway
    ws.start()
    info = {"name": a.name, "kind": "real", "robot": "MABEL", "lanIp": _lan_ip(),
            "wsPort": a.ws_port, "httpPort": int(os.getenv("MABEL_HTTP_PORT", "8080")),
            "wsPath": "/teleop", "transport": "ble", "version": ADV_VERSION}
    try:
        if sys.platform == "darwin":
            run_peripheral_corebluetooth(bridge, a.name, info)
        else:
            run_peripheral(bridge, a.adapter, a.name, info)
    except Exception as e:                                             # noqa: BLE001
        print(f"[ble] peripheral failed: {e}", file=sys.stderr)
        return 1
    finally:
        ws.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
