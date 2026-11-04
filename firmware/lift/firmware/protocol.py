# protocol.py - Serial command parsing and telemetry formatting.
#
# Line-oriented ASCII commands in, human-readable or JSON telemetry out.
# The CommandConsole holds the small amount of UI state (telemetry stream
# on/off and rate) and delegates all motion to the Lift instance.

import json
import utime

import config
import lift as lift_mod
from lift import emit_log

HELP = """Commands:
  up [effort]        open-loop drive up   (effort 0-1, default 1.0)
  down [effort]      open-loop drive down
  vel <mm/s>         closed-loop velocity hold (signed: + up / - down)
  pos <mm> [vmax]    trapezoidal move to absolute height
  stop               halt the motor
  torque on|off      enable/disable the motor driver (free-wheel when off)
  calibrate          stall-seek floor then ceiling, set soft limits
  home               stall-seek floor and define it as 0 mm
  park | setpark <mm>  go to / set the resting height
  zero | set <mm>    redefine current height
  home | homeheight <mm>  seek floor / set the height assigned there
  sethome            treat the current position as the home/bottom datum
  count              print raw encoder count
  status             print one telemetry line
  json on|off        stream JSON telemetry
  rate <hz>          telemetry stream rate (1-50)
  vmax <mm/s>        cruise ceiling for position moves
  accel <mm/s2>      trapezoidal ramp rate
  limits             print vmax / accel / tolerance
  contact on|off|clear  external-contact detection (speed deficit at known duty)
  limit on|off       enable/disable soft limits & stall stops
  grace <ms>         stall-detection grace period
  gpio up|down 0|1   raw open-loop drive (full speed) / off
  test               brief up/down motion test
  help"""


class CommandConsole:
    def __init__(self, lift):
        self.lift = lift
        # Telemetry streams ON by default so the Pico emits status/telemetry
        # consistently from boot — no host "json on" handshake required. This
        # closes the failure where a lost/starved enable left the stream off and
        # /lift/joint_states silent forever ("connected but no telemetry"). The
        # host can still toggle it with "json off"/"json on".
        self.json_stream = True
        self.telemetry_hz = config.TELEMETRY_HZ

    # ----------------------------------------------------------- telemetry
    def telemetry(self):
        l = self.lift
        return {
            "status": l.status,
            "mode": l.mode,
            "height": round(l.position_mm, 2),
            # The height the lift is driving to. Idle, it reads back the current
            # height, so a host can adopt it as its own setpoint on connect and
            # hold station instead of inheriting a stale target.
            "target": round(l.target_mm, 2),
            "vel": round(l.velocity_mm_s, 2),
            "vset": round(l.vloop.setpoint, 2),
            "effort": round(l.motor.effort, 3),
            "count": l.encoder.count,
            "calibrated": l.is_calibrated,
            "min_height": round(l.min_height_mm, 2),
            "max_height": round(l.max_height_mm, 2),
            "mm_per_count": round(l.encoder.mm_per_count, 6),
            "torque": l.torque_on,
            # Height at which external contact was last detected, or null. Status
            # reads CONTACT (4) at the moment it trips.
            "contact": None if l.contact_mm is None else round(l.contact_mm, 2),
            # BTS7960 R_IS on GP26. Valid while driving UP (the right half-bridge);
            # reads ~0 going down, where L_IS carries the current.
            "adc": l.motor.adc_raw,
            "adc0": l.motor.adc_zero,
            "amps": None if l.motor.amps is None else round(l.motor.amps, 3),
            # EXTRA load beyond the unloaded baseline at this duty — the number that
            # says how much external force is on the column and how it is changing.
            "lc": None if l.motor.adc_load_counts is None else int(l.motor.adc_load_counts),
            "load_a": None if l.motor.load_amps is None else round(l.motor.load_amps, 3),
            "load_n": None if l.motor.load_newtons is None else round(l.motor.load_newtons, 1),
        }

    def telemetry_json(self):
        return json.dumps(self.telemetry())

    def telemetry_line(self):
        t = self.telemetry()
        return ("[{}/{}] h={:.2f}mm tgt={:.2f}mm v={:.2f}mm/s vset={:.2f} "
                "eff={:.2f} cnt={} {}").format(
            lift_mod.STATUS_NAMES.get(t["status"], "?"),
            lift_mod.MODE_NAMES.get(t["mode"], "?"),
            t["height"], t["target"], t["vel"], t["vset"], t["effort"],
            t["count"], "CAL" if t["calibrated"] else "UNCAL")

    # ------------------------------------------------------------- dispatch
    def handle(self, line):
        parts = line.strip().split()
        if not parts:
            return
        cmd = parts[0].lower()
        l = self.lift

        try:
            if cmd == "up":
                eff = float(parts[1]) if len(parts) > 1 else 1.0
                l.set_manual(abs(eff))
            elif cmd == "down":
                eff = float(parts[1]) if len(parts) > 1 else 1.0
                l.set_manual(-abs(eff))
            elif cmd == "vel":
                l.set_velocity(float(parts[1]))
            elif cmd == "pos":
                vmax = float(parts[2]) if len(parts) > 2 else None
                l.set_position(float(parts[1]), vmax)
                # Deliberately SILENT. A policy or animation streams 'pos' at 50 Hz,
                # and acknowledging each one put 50 lines/s on the wire competing with
                # telemetry — enough to block the control loop on a full USB buffer.
                # The commanded height is already in every telemetry frame as 'target'.
            elif cmd == "stop":
                l.stop()
            elif cmd == "torque":
                on = len(parts) > 1 and parts[1] == "on"
                l.set_torque(on)
                print("Torque", "ON" if on else "OFF")
            elif cmd == "calibrate":
                l.calibrate()
                print("Calibrating: seeking floor...")
            elif cmd == "home":
                l.home()
                print("Homing: seeking floor...")
            elif cmd == "sethome":
                l.set_home()
                print("Home set at current position "
                      "({:.1f} mm)".format(l.min_height_mm))
            elif cmd == "park":
                if not l.park():
                    print("ERR: calibrate first")
            elif cmd == "setpark":
                l.park_mm = float(parts[1])
                print("Park height = {:.2f} mm".format(l.park_mm))
            elif cmd == "zero":
                l.encoder.zero()
                print(self.telemetry_line())
            elif cmd == "set":
                l.encoder.set_position_mm(float(parts[1]))
                print(self.telemetry_line())
            elif cmd == "homeheight":
                l.home_height_mm = float(parts[1])
                print("Home height = {:.2f} mm".format(l.home_height_mm))
            elif cmd == "count":
                print("Count:", l.encoder.count)
            elif cmd == "status":
                print(self.telemetry_line())
            elif cmd == "json":
                self.json_stream = (len(parts) > 1 and parts[1] == "on")
                print("JSON stream:", "ON" if self.json_stream else "OFF")
            elif cmd == "rate":
                hz = int(float(parts[1]))
                self.telemetry_hz = max(1, min(50, hz))
                print("Telemetry rate = {} Hz".format(self.telemetry_hz))
            elif cmd == "limit":
                config.LIMIT_ENABLED = (len(parts) > 1 and parts[1] == "on")
                print("Soft limits:", "ON" if config.LIMIT_ENABLED else "OFF")
            elif cmd == "accel":
                # Live tuning of the trapezoidal ramp rate. This is the binding
                # constraint for SMALL FAST moves: following +/-A mm at period T needs
                # (2*pi/T)^2 * A mm/s^2, which passes 200 well before the velocity
                # ceiling is anywhere near reached.
                config.MAX_ACCEL_MM_S2 = max(10.0, float(parts[1]))
                print("Accel = {:.0f} mm/s^2".format(config.MAX_ACCEL_MM_S2))
            elif cmd == "vmax":
                config.MAX_VELOCITY_MM_S = max(1.0, float(parts[1]))
                l.profile.v_max = config.MAX_VELOCITY_MM_S
                print("Vmax = {:.1f} mm/s".format(config.MAX_VELOCITY_MM_S))
            elif cmd == "contact":
                if len(parts) > 1 and parts[1] in ("on", "off"):
                    config.CONTACT_DETECT = (parts[1] == "on")
                if len(parts) > 1 and parts[1] == "clear":
                    l.contact_mm = None
                print("contact detect: {} | last at {}".format(
                    "ON" if config.CONTACT_DETECT else "OFF", l.contact_mm))
            elif cmd == "limits":
                print("vmax={:.1f} mm/s accel={:.0f} mm/s^2 tol={:.2f} mm".format(
                    config.MAX_VELOCITY_MM_S, config.MAX_ACCEL_MM_S2, config.POSITION_TOL_MM))
            elif cmd == "grace":
                config.GRACE_MS = max(0, int(float(parts[1])))
                print("Grace = {} ms".format(config.GRACE_MS))
            elif cmd == "gpio":
                # Legacy raw drive: 'gpio up 1' / 'gpio down 1' / '... 0'.
                on = len(parts) > 2 and parts[2] not in ("0", "off")
                if not on:
                    l.stop()
                elif parts[1] == "up":
                    l.set_manual(1.0)
                elif parts[1] == "down":
                    l.set_manual(-1.0)
                print("gpio {} {}".format(parts[1], "1" if on else "0"))
            elif cmd == "test":
                print("Test pulse up/down...")
                l.set_manual(1.0); utime.sleep_ms(150)
                l.stop();          utime.sleep_ms(150)
                l.set_manual(-1.0); utime.sleep_ms(150)
                l.stop()
                print("Test done.")
            elif cmd == "active":
                # Legacy relay-polarity command; no-op on the BTS7960 build.
                print("active: no-op (BTS7960 H-bridge build)")
            elif cmd in ("help", "h", "?"):
                print(HELP)
            else:
                emit_log("Unknown cmd '{}'".format(cmd))
        except (IndexError, ValueError):
            emit_log("ERR: bad arguments for '{}'".format(cmd))
