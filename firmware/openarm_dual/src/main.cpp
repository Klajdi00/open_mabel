// Generic robot DAQ/control firmware for Teensy 4.1 — top-level application.
//
// One Teensy == one robot, described entirely by /robot.json on the SD card: a set of
// DAMIAO actuators on CAN buses, driven in MIT mode, aggregated onto a UDP+protobuf
// link (RobotHostLink, see host_link.h) to a Python client on the PC. No robot-specific
// code lives here — flash once, swap the SD card to change robots.
//
// Host interface (mirrors the base module): the host STREAMS in State at a fixed rate
// and the firmware applies Commands as MIT setpoints at the control rate. Output is
// gated on host liveness, so a disconnected/crashed host fails safe — RobotHostLink
// disableAll()s the motors (they go LIMP, since the DM comm-loss watchdog is off).
// There is no motion until a host connects and commands.
//
// Bad/missing config fails safe: if /robot.json is absent or invalid, NO motor is ever
// brought up; the firmware idles and reports the error (LED fault code in Phase 5).
//
// For single-actuator CAN bring-up, see src/test_j1.cpp (env teensy41_test_j1).
// For config-loader verification, see src/test_config.cpp (env teensy41_test_config).

#include <FlexCAN_T4.h>
#include <QNEthernet.h>

#include "CanBus.h"
#include "DxlBus.h"
#include "Robot.h"
#include "config_loader.h"
#include "host_link.h"

using namespace qindesign::network;

// CAN peripherals are hardware-fixed (FlexCAN template params can't be runtime), so the
// application instantiates all three and exposes them as a 1-based table the config's
// bus index resolves against: buses[1]=CAN1, buses[2]=CAN2, buses[3]=CAN3 (buses[0]
// unused). Which actuators land on which bus is pure config.
FlexCAN_T4<CAN1, RX_SIZE_256, TX_SIZE_16> Can1;
CanBus                                    bus1;
FlexCAN_T4<CAN2, RX_SIZE_256, TX_SIZE_16> Can2;
CanBus                                    bus2;
FlexCAN_T4<CAN3, RX_SIZE_256, TX_SIZE_16> Can3;
CanBus                                    bus3;
static CanBus *const kBuses[] = { nullptr, &bus1, &bus2, &bus3 };
static constexpr size_t kNumBuses = sizeof(kBuses) / sizeof(kBuses[0]);

// DYNAMIXEL half-duplex serial buses, exposed as a 1-based table exactly like the CAN
// one so `port` in /robot.json resolves the same way `bus` does. Serial3 is the port
// the MABEL motherboard actually brings out: Teensy pins 14 (TX3) / 15 (RX3) land on
// pins 4 and 5 of the 6-pin JST SH connector silkscreened "I2C" (nets GPIO4/GPIO5;
// nothing else on the board uses them). In SERIAL_8N1_HALF_DUPLEX the LPUART runs in
// single-wire mode and the RX pin is unused, so the bus is pin 14 alone.
//   ⚠ ELECTRICAL: the X-series TTL bus is 5V logic and Teensy 4.1 pins are NOT 5V
//   tolerant. A level shifter between pin 14 and the servo data line is REQUIRED —
//   this is a board question that no firmware setting can substitute for.
DxlBus dxl1;
static DxlBus *const kDxlBuses[] = { nullptr, &dxl1 };
static constexpr size_t kNumDxlBuses = sizeof(kDxlBuses) / sizeof(kDxlBuses[0]);

// The line rate is NOT compiled in — it comes from /robot.json's `dxl_ports`, and is
// auto-detected when absent or wrong (see setup()). Rationale: the baud is a property
// of the servo, stored in ITS EEPROM, so hardcoding a constant here would mean every
// servo swap required reprogramming hardware to match the firmware, which is backwards.
// The read/write cadence adapts to whatever rate is found (DxlBus::roundTripMs) — at
// 57600 a 21-byte state read costs ~8 ms against ~0.5 ms at 1 Mbps, and a fixed cadence
// oversubscribes the slow case badly enough to starve one of two axes outright.

RobotConfig    g_config;
Robot          robot;
RobotHostLink  host(robot);

// DYNAMIXEL port-1 actuator IDs, kept for the idle-time baud re-sweep in loop().
static uint8_t  g_dxl_ids[MAX_ACTUATORS];
static size_t   g_dxl_n = 0;
static uint32_t g_dxl_last_sweep_ms = 0;
static constexpr uint32_t kDxlResweepMs = 5000;

static bool    g_ready = false;     // true once config loaded + robot brought up
static char    g_error[96] = {0};   // last fatal error for the idle report

static void fatal(const char *msg) {
  g_ready = false;
  snprintf(g_error, sizeof(g_error), "%s", msg);
  Serial.printf("FATAL: %s — robot stays disabled (no output).\n", g_error);
}

// Onboard-LED status code (headless fault indication, no display on the bench):
//   fast blink  (~5 Hz)  = NOT ready: bad/missing config, motors will never enable
//   slow heartbeat (1 Hz)= ready, idle (no host) — safe, waiting for a connection
//   solid on             = host connected + motors enabled (driving)
static void updateLed() {
  uint32_t now = millis();
  bool led;
  if (!g_ready)              led = (now % 200) < 100;   // fast blink: fault
  else if (host.enabled())   led = true;                // solid: enabled/driving
  else                       led = (now % 1000) < 100;  // heartbeat: ready, idle
  digitalWrite(LED_BUILTIN, led);
}

void setup() {
  Serial.begin(115200);
  pinMode(LED_BUILTIN, OUTPUT);
  delay(400);

  bus1.begin(Can1);
  bus2.begin(Can2);
  bus3.begin(Can3);
  delay(100);

  // 1) Load the robot description from SD. Any failure is fatal (no motors).
  ConfigResult cr = cfg::loadConfig(g_config);
  if (!cr.ok) { fatal(cr.error); return; }
  Serial.println("Config loaded:");
  cfg::printConfig(g_config);

  // 1b) Open each DYNAMIXEL port at the configured rate — or find it. The servo's
  //     baud belongs to the servo; the firmware adapts to it rather than requiring an
  //     EEPROM write on hardware to match a constant compiled in here. A configured
  //     baud is tried first (instant when correct); anything else falls back to a
  //     sweep of the standard rates against this port's own actuator IDs.
  {
    uint8_t ids[MAX_ACTUATORS];
    size_t n = 0;
    for (size_t i = 0; i < g_config.count; ++i) {
      const ActuatorConfig &a = g_config.actuators[i];
      if (a.transport == Transport::DYNAMIXEL && a.port == 1) ids[n++] = a.dxl_id;
    }
    if (n) {
      uint32_t want = g_config.dxl_baud[1];
      bool ok = false;
      if (want) {
        dxl1.begin(Serial3, want);
        delay(20);
        ok = true;
        for (size_t k = 0; k < n && ok; ++k) {
          uint16_t model = 0;
          ok = dxl1.ping(ids[k], model, 20);
        }
        if (ok) Serial.printf("dxl port1: %lu baud (from config)\n", (unsigned long)want);
        else Serial.printf("dxl port1: no answer at the configured %lu baud — sweeping\n",
                           (unsigned long)want);
      }
      if (!ok) {
        uint32_t found = dxl1.detectBaud(Serial3, ids, n);
        if (found) {
          Serial.printf("dxl port1: auto-detected %lu baud\n", (unsigned long)found);
        } else {
          // Nothing answered (unpowered / unplugged / no level shifter at boot). A
          // failed sweep used to leave the port at the LAST rate tried (4.5 Mbaud), so
          // serviceOffline()'s retries pinged a rate no servo will ever speak and a
          // neck powered a minute later stayed OFFLINE until a power-cycle
          // (2026-09-30, Mac bring-up). Park at the configured rate instead, and let
          // loop() re-sweep while the robot is idle (see g_dxl_* below).
          uint32_t park = want ? want : 1000000UL;
          dxl1.begin(Serial3, park);
          Serial.printf("dxl port1: NO servo answered at any standard baud — parked at "
                        "%lu baud; re-sweeping every %lu s while no host is connected\n",
                        (unsigned long)park, (unsigned long)(kDxlResweepMs / 1000));
        }
      }
      memcpy(g_dxl_ids, ids, n);
      g_dxl_n = n;
    }
  }

  // 2) Bring up every actuator (disable -> MIT -> enable), then leave the robot
  //    DISABLED until a host connects (fail-safe: no host, no output).
  if (!robot.begin(g_config, kBuses, kNumBuses, kDxlBuses, kNumDxlBuses)) {
    fatal("robot.begin failed (bad bus/port index in config)");
    return;
  }
  robot.disableAll();

  // 3) Network identity comes from the config.
  IPAddress ip(g_config.net.ip[0], g_config.net.ip[1], g_config.net.ip[2], g_config.net.ip[3]);
  IPAddress mask(g_config.net.netmask[0], g_config.net.netmask[1], g_config.net.netmask[2], g_config.net.netmask[3]);
  IPAddress gw(g_config.net.gateway[0], g_config.net.gateway[1], g_config.net.gateway[2], g_config.net.gateway[3]);
  if (Ethernet.begin(ip, mask, gw)) {
    Serial.printf("Ethernet: static IP %u.%u.%u.%u up (waiting for link/cable)\n",
                  ip[0], ip[1], ip[2], ip[3]);
  } else {
    Serial.println("Ethernet: begin() FAILED (bad config)");
  }

  host.begin(g_config.net.port);
  Serial.printf("Robot '%s' up — host link UDP port %u; no host, no output.\n",
                g_config.name, g_config.net.port);
  g_ready = true;
}

void loop() {
  uint32_t now = millis();
  updateLed();

  // Not ready (bad config): idle and re-report the error at low rate. No CAN, no output.
  if (!g_ready) {
    static uint32_t lastErrMs = 0;
    if (now - lastErrMs >= 2000) {
      lastErrMs = now;
      Serial.printf("idle (config error: %s)\n", g_error);
    }
    return;
  }

  // CAN dispatch (feedback handlers run here, in thread context).
  bus1.poll();
  bus2.poll();
  bus3.poll();

  // DYNAMIXEL: hand each serial actuator its slot (cache a setpoint, keep a read
  // armed), then advance the single-wire bus by at most one transaction. Both are
  // non-blocking; a half-duplex bus cannot overlap requests, so the scheduling that
  // FlexCAN gets from hardware has to happen here in software.
  robot.service();
  dxl1.poll();

  // DYNAMIXEL baud re-sweep: while NO host is connected (the robot is limp and the
  // 150 ms command watchdog has nothing to protect), re-run the blocking baud sweep
  // (~0.4 s) every kDxlResweepMs as long as a serial actuator is still OFFLINE. With a
  // host connected we never block the loop; serviceOffline() keeps pinging at the
  // parked rate instead, which finds a servo that speaks the configured baud.
  if (g_dxl_n && !host.connected() && (now - g_dxl_last_sweep_ms) >= kDxlResweepMs) {
    g_dxl_last_sweep_ms = now;
    bool any_offline = false;
    for (size_t i = 0; i < g_config.count && !any_offline; ++i)
      any_offline = robot.transport(i) == Transport::DYNAMIXEL && !robot.online(i);
    if (any_offline) {
      uint32_t before = dxl1.baud();
      uint32_t found = dxl1.detectBaud(Serial3, g_dxl_ids, g_dxl_n);
      if (found) Serial.printf("dxl port1: servo(s) answered at %lu baud — bringing them up\n",
                               (unsigned long)found);
      else dxl1.begin(Serial3, before);            // still nobody: stay parked
    }
  }

  // Recover actuators that were absent at bring-up (unpowered, unplugged, or on a
  // bus that was down). Costs nothing once everything is online.
  int recovered = robot.serviceOffline();
  if (recovered >= 0) {
    const ActuatorLimits &l = robot.limits(recovered);
    if (robot.transport(recovered) == Transport::DYNAMIXEL)
      Serial.printf("actuator '%s' came ONLINE (dynamixel; no MIT ranges)\n",
                    robot.name(recovered));
    else
      Serial.printf("actuator '%s' came ONLINE (P%.2f V%.2f T%.2f)\n",
                    robot.name(recovered), l.p_max, l.v_max, l.t_max);
    if (host.enabled()) robot.enable(recovered);   // match the current output state
  }

  // Host link: drain inbound (cache commands) + stream setpoints/State.
  host.poll();
  host.service();

  // Ethernet link-state edge report: print only on change.
  static int lastLink = -1;
  int link = Ethernet.linkState() ? 1 : 0;
  if (link != lastLink) {
    lastLink = link;
    Serial.printf("Ethernet link %s\n", link ? "UP" : "DOWN");
  }

  // 20 Hz diagnostic print.
  static uint32_t lastPrintMs = 0;
  if (now - lastPrintMs >= 50) {
    lastPrintMs = now;
    Serial.printf("[host %s%s seq=%lu rx=%lu tx=%lu @%uHz] ",
                  host.connected() ? "CONNECTED" : "idle",
                  host.enabled() ? "/EN" : "/dis",
                  (unsigned long)host.lastSeq(), (unsigned long)host.rxCount(),
                  (unsigned long)host.txCount(), host.streamHz());
    for (size_t i = 0; i < robot.count(); ++i) {
      Robot::ActuatorState s = robot.state(i);
      Serial.printf("%s %c%X p%.2f ", robot.name(i),
                    robot.enabled(i) ? 'E' : 'e', s.err, s.pos);
    }
    Serial.println();
  }
}
