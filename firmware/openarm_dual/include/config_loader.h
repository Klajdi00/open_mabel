#pragma once
#include <Arduino.h>
#include <SD.h>
#include <ArduinoJson.h>
#include "RobotConfig.h"

// config_loader.h — parse /robot.json off the Teensy 4.1 built-in SD card into a
// RobotConfig. Pulls in SD + ArduinoJson, so include it ONLY from the application /
// test sketches (never from the Robot class or the diagnostic envs).
//
// Every failure mode returns ConfigResult{ok=false, error=<reason>}; nothing here
// touches motors. The caller is responsible for the fail-safe (refuse to enable on
// !ok) — see main.cpp / fail-safe phase.

namespace cfg {

// Parse "a.b.c.d" into out[4]. Returns false on malformed input.
inline bool parseIp(const char *s, uint8_t out[4]) {
  if (!s) return false;
  int a, b, c, d;
  if (sscanf(s, "%d.%d.%d.%d", &a, &b, &c, &d) != 4) return false;
  if ((a | b | c | d) & ~0xFF) return false;   // each octet must be 0..255
  out[0] = a; out[1] = b; out[2] = c; out[3] = d;
  return true;
}

inline void copyStr(char *dst, size_t cap, const char *src) {
  if (!src) { dst[0] = 0; return; }
  strncpy(dst, src, cap - 1);
  dst[cap - 1] = 0;
}

inline ConfigResult fail(ConfigResult &r, const char *fmt, ...) {
  r.ok = false;
  va_list ap; va_start(ap, fmt);
  vsnprintf(r.error, sizeof(r.error), fmt, ap);
  va_end(ap);
  return r;
}

// Load + validate /robot.json into `out`. The default path matches the file the
// user writes to the card.
inline ConfigResult loadConfig(RobotConfig &out, const char *path = "/robot.json") {
  ConfigResult r;
  out = RobotConfig{};   // reset to defaults

  // Retry SD init: the first SD.begin() after power-up can fail intermittently on
  // SDXC cards before the card is ready.
  bool sd_ok = false;
  for (int attempt = 0; attempt < 10 && !sd_ok; ++attempt) {
    if (SD.begin(BUILTIN_SDCARD)) { sd_ok = true; break; }
    delay(100);
  }
  if (!sd_ok)
    return fail(r, "SD card not found (BUILTIN_SDCARD) after retries");

  File f = SD.open(path);
  if (!f)
    return fail(r, "%s not found on SD card", path);

  JsonDocument doc;                          // ArduinoJson v7: elastic capacity
  DeserializationError jerr = deserializeJson(doc, f);
  f.close();
  if (jerr)
    return fail(r, "JSON parse error: %s", jerr.c_str());

  // ---- top-level + network -------------------------------------------------
  copyStr(out.name, sizeof(out.name), doc["name"] | "robot");

  JsonObjectConst net = doc["net"];
  if (net.isNull())
    return fail(r, "missing 'net' object");
  if (!parseIp(net["ip"] | (const char *)nullptr, out.net.ip))
    return fail(r, "net.ip missing or malformed (need \"a.b.c.d\")");
  parseIp(net["netmask"] | "255.255.255.0", out.net.netmask);
  parseIp(net["gateway"] | "0.0.0.0", out.net.gateway);
  out.net.port = net["port"] | 5800;

  // ---- stream + watchdog (all optional, sane defaults) ---------------------
  JsonObjectConst stream = doc["stream"];
  out.stream_default_hz = stream["default_hz"] | 1000;
  out.stream_max_hz     = stream["max_hz"] | out.stream_default_hz;
  if (out.stream_max_hz < out.stream_default_hz)
    return fail(r, "stream.max_hz (%u) < default_hz (%u)",
                out.stream_max_hz, out.stream_default_hz);

  JsonObjectConst wd = doc["watchdog"];
  out.conn_watchdog_ms = wd["conn_ms"] | 250;
  out.cmd_watchdog_ms  = wd["cmd_ms"] | 150;

  // ---- dynamixel ports (optional) ------------------------------------------
  // "dxl_ports": [ { "port": 1, "baud": 1000000 } ]
  // Omit the block, or omit/zero a baud, to auto-detect that port at bring-up.
  JsonArrayConst dports = doc["dxl_ports"];
  if (!dports.isNull()) {
    for (JsonObjectConst dp : dports) {
      uint8_t port = dp["port"] | 0;
      if (port < 1 || port > MAX_DXL_PORTS)
        return fail(r, "dxl_ports port=%u out of range 1..%u", port, MAX_DXL_PORTS);
      out.dxl_baud[port] = dp["baud"] | 0u;
    }
  }

  // ---- actuators -----------------------------------------------------------
  JsonArrayConst acts = doc["actuators"];
  if (acts.isNull() || acts.size() == 0)
    return fail(r, "'actuators' array missing or empty");
  if (acts.size() > MAX_ACTUATORS)
    return fail(r, "%u actuators exceeds MAX_ACTUATORS (%u)",
                (unsigned)acts.size(), (unsigned)MAX_ACTUATORS);

  size_t i = 0;
  for (JsonObjectConst a : acts) {
    ActuatorConfig &ac = out.actuators[i];

    const char *name = a["name"] | (const char *)nullptr;
    if (!name || !name[0])
      return fail(r, "actuators[%u].name missing", (unsigned)i);
    copyStr(ac.name, sizeof(ac.name), name);

    // ---- transport ---------------------------------------------------------
    // Absent => "dm_can", so every robot.json written before the DYNAMIXEL
    // extension loads byte-for-byte unchanged. An UNRECOGNISED value is fatal
    // rather than silently defaulted: a typo'd transport that quietly fell back
    // to CAN would try to speak MIT to a servo that is not there, and the only
    // symptom would be a permanently offline joint.
    const char *tr = a["transport"] | "dm_can";
    if (strcmp(tr, "dm_can") == 0) {
      ac.transport = Transport::DM_CAN;
    } else if (strcmp(tr, "dynamixel") == 0) {
      ac.transport = Transport::DYNAMIXEL;
    } else {
      return fail(r, "actuators[%u] '%s' unknown transport '%s' "
                     "(expected 'dm_can' or 'dynamixel')", (unsigned)i, ac.name, tr);
    }

    if (ac.transport == Transport::DYNAMIXEL) {
      ac.port = a["port"] | 0;
      if (ac.port < 1 || ac.port > MAX_DXL_PORTS)
        return fail(r, "actuators[%u] '%s' port=%u out of range 1..%u",
                    (unsigned)i, ac.name, ac.port, MAX_DXL_PORTS);
      // 0..252 are legal IDs; 253/254(broadcast)/255 are not addressable units.
      uint32_t did = a["dxl_id"] | 0xFFFFu;
      if (did > 252)
        return fail(r, "actuators[%u] '%s' dxl_id missing or >252", (unsigned)i, ac.name);
      ac.dxl_id    = (uint8_t)did;
      ac.raw_gains = a["raw_gains"] | false;
    } else {
      ac.bus = a["bus"] | 0;
      if (ac.bus < 1 || ac.bus > MAX_BUSES)
        return fail(r, "actuators[%u] '%s' bus=%u out of range 1..%u",
                    (unsigned)i, ac.name, ac.bus, MAX_BUSES);

      ac.can_id = a["can_id"] | 0;
      ac.mst_id = a["mst_id"] | 0;
      if (ac.can_id == 0)
        return fail(r, "actuators[%u] '%s' can_id missing/zero", (unsigned)i, ac.name);
      if (ac.mst_id == 0)
        return fail(r, "actuators[%u] '%s' mst_id missing/zero", (unsigned)i, ac.name);
    }

    copyStr(ac.model, sizeof(ac.model), a["model"] | "");


    ac.default_kp = a["default_kp"] | 0.0f;
    ac.default_kd = a["default_kd"] | 1.0f;

    // Uniqueness within the same bus — a duplicate mst_id collides on feedback
    // arbitration; a duplicate name breaks by-name host addressing.
    for (size_t j = 0; j < i; ++j) {
      const ActuatorConfig &p = out.actuators[j];
      if (strcmp(p.name, ac.name) == 0)
        return fail(r, "duplicate actuator name '%s'", ac.name);
      if (p.transport != ac.transport) continue;   // different wires cannot collide
      if (ac.transport == Transport::DYNAMIXEL) {
        if (p.port == ac.port && p.dxl_id == ac.dxl_id)
          return fail(r, "duplicate dxl_id %u on port %u ('%s' & '%s')",
                      (unsigned)ac.dxl_id, ac.port, p.name, ac.name);
        continue;
      }
      if (p.bus == ac.bus && p.can_id == ac.can_id)
        return fail(r, "duplicate can_id 0x%X on bus %u ('%s' & '%s')",
                    (unsigned)ac.can_id, ac.bus, p.name, ac.name);
      if (p.bus == ac.bus && p.mst_id == ac.mst_id)
        return fail(r, "duplicate mst_id 0x%X on bus %u ('%s' & '%s')",
                    (unsigned)ac.mst_id, ac.bus, p.name, ac.name);
    }
    ++i;
  }
  out.count = i;

  r.ok = true;
  return r;
}

// Pretty-print a loaded config to Serial (Phase-1 verification + boot diagnostics).
inline void printConfig(const RobotConfig &c) {
  Serial.printf("Robot '%s': %u actuators\n", c.name, (unsigned)c.count);
  Serial.printf("  net  %u.%u.%u.%u:%u  mask %u.%u.%u.%u  gw %u.%u.%u.%u\n",
                c.net.ip[0], c.net.ip[1], c.net.ip[2], c.net.ip[3], c.net.port,
                c.net.netmask[0], c.net.netmask[1], c.net.netmask[2], c.net.netmask[3],
                c.net.gateway[0], c.net.gateway[1], c.net.gateway[2], c.net.gateway[3]);
  for (uint8_t k = 1; k <= MAX_DXL_PORTS; ++k)
    if (c.dxl_baud[k])
      Serial.printf("  dxl port%u baud %lu\n", k, (unsigned long)c.dxl_baud[k]);
  Serial.printf("  stream %u Hz (max %u)  watchdog conn %lu ms / cmd %lu ms\n",
                c.stream_default_hz, c.stream_max_hz,
                (unsigned long)c.conn_watchdog_ms, (unsigned long)c.cmd_watchdog_ms);
  for (size_t i = 0; i < c.count; ++i) {
    const ActuatorConfig &a = c.actuators[i];
    // Ranges are absent here by design — they are read from the motor at
    // Robot::begin(), so there is nothing to print until the robot is up.
    if (a.transport == Transport::DYNAMIXEL)
      Serial.printf("  [%u] %-10s DXL port%u id=%-3u        %-8s  gains:%s\n",
                    (unsigned)i, a.name, a.port, (unsigned)a.dxl_id, a.model,
                    a.raw_gains ? "raw-register" : "servo-stored (wire kp/kd ignored)");
    else
      Serial.printf("  [%u] %-10s bus%u id=0x%02X mst=0x%02X %-8s  kp%.1f kd%.1f\n",
                    (unsigned)i, a.name, a.bus, (unsigned)a.can_id, (unsigned)a.mst_id,
                    a.model, a.default_kp, a.default_kd);
  }
}

}  // namespace cfg
