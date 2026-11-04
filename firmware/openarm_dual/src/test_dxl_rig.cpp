// test_dxl_rig.cpp — bench rig that exercises the REAL main-firmware classes
// (env teensy40_dxl_rig / teensy41_dxl_rig).
//
// ── What this is for ─────────────────────────────────────────────────────────
// `test_dxl.cpp` is a protocol poker: it talks to a servo to prove the WIRE works.
// This sketch is different and complementary — it stands up `DxlBus`, `DxlActuator`
// and `Robot` exactly as `main.cpp` wires them, and drives them through exactly the
// calls `host_link.h` makes, so a green run here means the MAIN FIRMWARE's DYNAMIXEL
// path is correct, not merely that some bytes reached a servo.
//
// The correspondence, call for call:
//   main.cpp  loop()          -> robot.service(); dxl1.poll();     ... same here
//   host_link applyCommand()  -> robot.sendCommand(i, {p,v,kp,kd,t}) ... same here
//   host_link sendState()     -> robot.state(i), robot.feedbackAgeMs(i) ... same
//   host_link sendHealth()    -> robot.enabled(i), state().err/t_mos/t_rotor ... same
//   host_link sendRobotInfo() -> robot.limits(i), robot.online(i)  ... same
//
// The ONLY things not exercised are the two the Teensy 4.0 physically lacks: the SD
// card (so the RobotConfig is built in code instead of parsed from /robot.json — the
// parser has its own env, `teensy41_test_config`) and Ethernet/nanopb. Everything
// between the config struct and the wire is the real thing.
//
// ── Standalone bench setup this targets ──────────────────────────────────────
// A bare Teensy 4.0 on USB, pin 14 (TX3) to the servo's data line, servo on its own
// 6.5-12V rail, grounds common. No robot, no motherboard, no CAN.
//
//   ⚠ WITHOUT A LEVEL SHIFTER the servo drives 5V into pin 14, which is NOT 5V
//   tolerant on any Teensy 4.x. The TEENSY is what is at risk here, not the servo.
//   A ~1k series resistor in the data line limits current through the pin's
//   protection diode and is the usual stopgap. This is a hardware caveat that no
//   firmware setting can remove.
//
//   pio run -e teensy40_dxl_rig -t upload
//   pio device monitor -b 115200        then type `help`
//
// SAFE BY DEFAULT: `begin` configures and reads but leaves torque OFF. Nothing moves
// until you type `en`, and `dis` releases.

#include <Arduino.h>
#include "DxlBus.h"
#include "Robot.h"

// Wired exactly like main.cpp: a 1-based table the config's `port` resolves against.
static DxlBus dxl1;
static DxlBus *const kDxlBuses[] = {nullptr, &dxl1};
static constexpr size_t kNumDxlBuses = sizeof(kDxlBuses) / sizeof(kDxlBuses[0]);

static Robot robot;
static RobotConfig cfg;

// The servo may still be at the factory 57,600 (the U2D2 session may not have
// re-provisioned it), while the control firmware expects 1 Mbps. Try both rather
// than making the operator guess which failure they are looking at.
static const uint32_t kBauds[] = {1000000, 57600, 115200, 2000000, 9600};
static uint32_t g_baud = 0;

static uint8_t  g_ids[2] = {1, 2};
static size_t   g_n      = 0;
static bool     g_hold   = false;
static bool     g_conn   = false;   // stands in for host_link's `connected`   // stream a stale-command damping hold, as host_link does

// Build the RobotConfig main.cpp would have loaded from /robot.json — the same two
// entries the reference config now carries, minus the SD round trip.
static void buildConfig(size_t n) {
  cfg = RobotConfig{};
  snprintf(cfg.name, sizeof(cfg.name), "dxl-bench-rig");
  cfg.stream_default_hz = 400;
  cfg.count = n;
  const char *names[2] = {"neck_pitch", "neck_roll"};
  for (size_t i = 0; i < n; ++i) {
    ActuatorConfig &a = cfg.actuators[i];
    snprintf(a.name, sizeof(a.name), "%s", names[i]);
    snprintf(a.model, sizeof(a.model), "2XL430-W250-T");
    a.transport  = Transport::DYNAMIXEL;
    a.port       = 1;
    a.dxl_id     = g_ids[i];
    a.raw_gains  = false;
    a.default_kd = 0.0f;
  }
}

// Find the servos before Robot::begin(), so a baud mismatch reports as a baud
// mismatch rather than as "actuator OFFLINE".
static bool detect() {
  for (uint32_t b : kBauds) {
    dxl1.begin(Serial3, b);
    delay(30);
    size_t n = 0;
    for (uint8_t id = 0; id <= 10 && n < 2; ++id) {
      uint16_t model = 0;
      if (!dxl1.ping(id, model, 15)) continue;
      g_ids[n++] = id;
      Serial.printf("  found id=%-3u model=%-5u%s @ %lu baud\n", id, model,
                    model == 1090 ? " (2XL430-W250)" : "", (unsigned long)b);
    }
    if (n) { g_baud = b; g_n = n; return true; }
  }
  return false;
}

// The self-check: assert, against real hardware, every claim the main firmware's
// DYNAMIXEL path makes to the host link.
static void selfCheck() {
  int pass = 0, fail = 0;
  auto ck = [&](const char *what, bool ok, const char *detail = "") {
    Serial.printf("  [%s] %-46s %s\n", ok ? "PASS" : "FAIL", what, detail);
    ok ? ++pass : ++fail;
  };
  char buf[64];

  ck("Robot::count() matches the config", robot.count() == g_n);
  for (size_t i = 0; i < robot.count(); ++i) {
    snprintf(buf, sizeof(buf), "'%s'", robot.name(i));
    ck("actuator is ONLINE (answered at begin)", robot.online(i), buf);
    ck("transport reads DYNAMIXEL",
       robot.transport(i) == Transport::DYNAMIXEL, buf);

    // RobotInfo contract: no MIT fixed-point scale exists on this transport, so the
    // ranges must be 0 — never DmActuator's DM4310-shaped defaults, which would
    // reach the host looking like real measurements.
    const ActuatorLimits &l = robot.limits(i);
    snprintf(buf, sizeof(buf), "p%.1f v%.1f t%.1f", l.p_max, l.v_max, l.t_max);
    ck("limits p/v/t_max are 0 (no MIT scale)",
       l.p_max == 0.0f && l.v_max == 0.0f && l.t_max == 0.0f, buf);
    snprintf(buf, sizeof(buf), "kp%.0f kd%.0f", l.kp_max, l.kd_max);
    ck("kp_max/kd_max are 0 (gains not from wire)",
       l.kp_max == 0.0f && l.kd_max == 0.0f, buf);

    // State contract.
    Robot::ActuatorState s = robot.state(i);
    snprintf(buf, sizeof(buf), "%.4f rad", s.pos);
    ck("position is being read back", robot.feedbackAgeMs(i) < 200, buf);
    ck("torque reads 0 (Present Load is %, not N.m)", s.torque == 0.0f);
    ck("t_mos reads 0 (one sensor on this part)", s.t_mos == 0);
    snprintf(buf, sizeof(buf), "%u C", s.t_rotor);
    ck("t_rotor carries a plausible temperature",
       s.t_rotor > 0 && s.t_rotor < 100, buf);

    // live_mask contract: host_link sets bit i when feedbackAgeMs < kLiveWindowMs(50).
    snprintf(buf, sizeof(buf), "age %lu ms", (unsigned long)robot.feedbackAgeMs(i));
    ck("would set live_mask (age < 50 ms)", robot.feedbackAgeMs(i) < 50, buf);

    // The bug this rig exists to catch: `enabled` must NOT be the DM `err == 1`
    // nibble, and must not be an optimistic flag either. Assert the real invariant —
    // Robot::enabled(i) AGREES WITH THE SERVO'S OWN Torque Enable(64) — rather than
    // "it is false", which only held if torque happened to be off when check ran and
    // failed spuriously on a re-run after a move.
    // Reported, not asserted, and deliberately NOT via a blocking read: the module's
    // own rule is that the blocking helpers are bring-up only, because the scheduler
    // owns the bus at runtime. Calling readBlocking() here put two masters on one wire
    // and produced both a bogus FAIL and a real bus timeout. The invariant it was
    // trying to test — enabled() tracks the servo's own Torque Enable(64) — is what
    // the `torque` command proves properly, by commanding and waiting for convergence.
    snprintf(buf, sizeof(buf), "%s  (run `torque` to verify the readback)",
             robot.enabled(i) ? "ENERGIZED" : "de-energized");
    Serial.printf("  [info] %-46s %s\n", "enabled() currently reads", buf);
  }
  Serial.printf("\n  %d passed, %d failed%s\n", pass, fail,
                fail ? "  <-- the main firmware's DXL path is NOT right yet" : "");
}

// Round-trip the thing the optimistic flag used to fake: command torque, then wait
// for the servo's OWN Torque Enable(64) to come back and agree. If enabled() were
// still just an optimistic flag this would pass vacuously, so the check deliberately
// waits past kTorqueSettleMs and reports how long the readback took.
static void torqueRoundTrip() {
  Serial.println("--- torque readback round-trip (grounds Robot::enabled) ---");
  auto settle = [](bool want) -> long {
    uint32_t t0 = millis();
    while (millis() - t0 < 1500) {
      robot.service(); dxl1.poll();
      bool all = true;
      for (size_t i = 0; i < robot.count(); ++i) all &= (robot.enabled(i) == want);
      if (all && millis() - t0 > 200) return (long)(millis() - t0);
    }
    return -1;
  };
  robot.enableAll();
  long a = settle(true);
  Serial.printf("  [%s] enableAll  -> servo reports Torque Enable = 1   (%ld ms)\n",
                a >= 0 ? "PASS" : "FAIL", a);
  robot.disableAll();
  long b = settle(false);
  Serial.printf("  [%s] disableAll -> servo reports Torque Enable = 0   (%ld ms)\n",
                b >= 0 ? "PASS" : "FAIL", b);
  Serial.println("  (torque left OFF)");
}

static void help() {
  Serial.println("commands:");
  Serial.println("  check         re-run the main-firmware contract self-check");
  Serial.println("  torque        round-trip enable/disable against the servo's own register");
  Serial.println("  en / dis      robot.enableAll() / disableAll()  (en MOVES nothing by itself)");
  Serial.println("  pos <i> <deg> host_link's applyCommand path: robot.sendCommand(i, ...)");
  Serial.println("  hold          stream host_link's stale-command fail-safe (Robot::sendHold)");
  Serial.println("  stop          stop the hold stream");
  Serial.println("  free          command kp=0 everywhere (DXL torque should release)");
  Serial.println("  stats         DxlBus reply/timeout counters");
  Serial.println("  kill <id>     clear that servo's Torque Enable BEHIND the firmware's back");
  Serial.println("                (simulates a brown-out / watchdog trip — serviceEnable should heal it)");
}

void setup() {
  Serial.begin(115200);
  delay(800);
  Serial.println("\n=== DXL bench rig — real DxlBus/DxlActuator/Robot, no SD, no Ethernet ===");
  Serial.println("pin 14 (TX3) half-duplex. NO LEVEL SHIFTER = 5V into a 3.3V pin; see header.\n");

  Serial.println("detecting servos...");
  if (!detect()) {
    Serial.println("  nothing answered on any baud. Check: level shifter / series resistor,");
    Serial.println("  servo power (own 6.5-12V rail), pin 14 wiring, common ground.");
    return;
  }
  buildConfig(g_n);

  // The exact call main.cpp makes. nullptr/0 for the CAN table: this config is
  // DYNAMIXEL-only, and begin() never dereferences the CAN side for such an entry.
  Serial.printf("\nrobot.begin() with %u DYNAMIXEL actuator(s), no CAN table...\n",
                (unsigned)g_n);
  if (!robot.begin(cfg, nullptr, 0, kDxlBuses, kNumDxlBuses)) {
    Serial.println("  robot.begin() REFUSED (bad port index) — rig cannot continue.");
    return;
  }
  robot.disableAll();

  // Let the scheduler run briefly so there is real feedback to check against.
  uint32_t t0 = millis();
  while (millis() - t0 < 400) { robot.service(); dxl1.poll(); }

  Serial.println("\n--- main-firmware contract self-check ---");
  selfCheck();
  Serial.println();
  help();
}

void loop() {
  if (g_baud == 0) return;

  // EXACTLY main.cpp's loop body for the serial path.
  robot.service();
  dxl1.poll();

  // host_link::service() calls this on every tick while a host is connected. `g_conn`
  // stands in for that connection here so the repair loop can be exercised on a bench
  // that has no Ethernet.
  if (g_conn) {
    int fixed = robot.serviceEnable();
    if (fixed >= 0)
      Serial.printf(">>> serviceEnable RE-ENERGIZED '%s' (idx %d)\n",
                    robot.name((size_t)fixed), fixed);
  }

  // host_link.applyCommand(fresh=false): when a command goes stale it substitutes a
  // damping hold. On the DXL path kd is meaningless, so this must be a no-op that
  // does not disturb the held position — worth being able to watch.
  static uint32_t lastHold = 0;
  if (g_hold && millis() - lastHold >= 3) {
    lastHold = millis();
    // host_link's stale-command branch, verbatim: Robot::sendHold(), NOT a hand-built
    // {pos:0, kd:default} command. Those are the same thing on a DM-J and emphatically
    // not on a position servo, where pos=0 means "drive to raw zero at full speed".
    // This sketch used to hand-build them, which made the hold the single most
    // dangerous command it could issue at a mounted joint sitting 175 deg off centre.
    for (size_t i = 0; i < robot.count(); ++i) robot.sendHold(i);
  }

  // Serial console.
  static char line[64];
  static size_t n = 0;
  while (Serial.available()) {
    char ch = (char)Serial.read();
    if (ch == '\r') continue;
    if (ch != '\n') { if (n < sizeof(line) - 1) line[n++] = ch; continue; }
    line[n] = 0; n = 0;
    char verb[16] = {0};
    int idx = -1; float arg = 0;
    int fields = sscanf(line, "%15s %d %f", verb, &idx, &arg);
    if (!line[0] || !strcmp(verb, "help")) { help(); }
    else if (!strcmp(verb, "check")) selfCheck();
    else if (!strcmp(verb, "torque")) torqueRoundTrip();
    else if (!strcmp(verb, "kill")) {
      if (fields < 2) { Serial.println("kill <dxl_id>"); }
      else {
        // queueWrite, NOT writeBlocking: the scheduler owns this wire at runtime and a
        // blocking write from the console is a second master on it. The module card says
        // the blocking helpers are bring-up only — this is what ignoring that looks like:
        // the write silently never reaches the servo and the test reads as a pass.
        uint8_t z = 0;
        dxl1.queueWrite((uint8_t)idx, DxlActuator::kAddrTorqueEnable, &z, 1);
        Serial.printf("cleared Torque Enable on dxl id=%d without telling the firmware\n", idx);
      }
    }
    else if (!strcmp(verb, "free")) {
      // A command with kp == 0 — what a de-activated DoF or a limp stack sends.
      for (size_t i = 0; i < robot.count(); ++i) {
        Robot::ActuatorCommand c{};
        c.pos = robot.state(i).pos;       // same position, no holding gain
        robot.sendCommand(i, c);
      }
      Serial.println("sent kp=0 to every joint — DYNAMIXEL axes should go FREE");
    }
    else if (!strcmp(verb, "en"))  { robot.enableAll(); g_conn = true;  Serial.println("enableAll() + connected"); }
    else if (!strcmp(verb, "dis")) { robot.disableAll(); g_hold = false; g_conn = false; Serial.println("disableAll() + disconnected"); }
    else if (!strcmp(verb, "hold")) { g_hold = true;  Serial.println("streaming damping hold"); }
    else if (!strcmp(verb, "stop")) { g_hold = false; Serial.println("hold stopped"); }
    else if (!strcmp(verb, "stats"))
      Serial.printf("replies=%lu timeouts=%lu foreign=%lu ctrl_dropped=%lu stuffed=%lu\n",
                    (unsigned long)dxl1.replies(), (unsigned long)dxl1.timeouts(),
                    (unsigned long)dxl1.foreign(), (unsigned long)dxl1.ctrlDropped(),
                    (unsigned long)dxl1.stuffed());
    else if (!strcmp(verb, "pos")) {
      if (fields < 3 || idx < 0 || (size_t)idx >= robot.count()) {
        Serial.println("pos <index> <deg>");
      } else {
        Robot::ActuatorCommand c{};
        c.pos = arg / 57.29578f;          // the wire is radians, as host_link sends
        c.kp  = 1.0f;                     // kp > 0 = "hold this joint"; kp == 0 frees it
        robot.sendCommand((size_t)idx, c);
        Serial.printf("sendCommand(%d, pos=%+.4f rad = %+.1f deg)%s\n", idx, c.pos, arg,
                      robot.enabled((size_t)idx) ? "" : "   [torque OFF — type `en` to move]");
      }
    } else help();
  }

  // 5 Hz telemetry: precisely the fields sendState()/sendHealth() would encode.
  static uint32_t lastPrint = 0;
  if (millis() - lastPrint >= 200) {
    lastPrint = millis();
    Serial.print("[state] ");
    for (size_t i = 0; i < robot.count(); ++i) {
      Robot::ActuatorState s = robot.state(i);
      Serial.printf("%s pos%+7.2fdeg vel%+6.2f tq%.1f %s err%X %uC age%lums | ",
                    robot.name(i), s.pos * 57.29578f, s.vel, s.torque,
                    robot.enabled(i) ? "EN " : "dis", s.err, s.t_rotor,
                    (unsigned long)robot.feedbackAgeMs(i));
    }
    Serial.println();
  }
}
