// test_dxl.cpp — standalone DYNAMIXEL bring-up diagnostic (env teensy41_test_dxl).
//
// The serial-bus counterpart of test_j1/test_regs, and it exists for the same reason:
// when a new actuator will not answer, you want a sketch that touches ONLY that wire,
// with no Ethernet, no SD card, no CAN and no control loop to confuse the picture.
//
// SAFE BY DEFAULT: everything except `move` is read-only. Nothing energizes a servo
// until you type `move`, and `off` de-energizes.
//
// ── The first thing that will bite ───────────────────────────────────────────
// A factory DYNAMIXEL runs at 57,600 baud. The control firmware runs the bus at
// 1 Mbps (a 57.6 kbps Goal Position write costs ~2.8 ms, which alone exceeds the
// 2.5 ms budget of a 400 Hz tick). So a brand-new servo is INVISIBLE to the control
// firmware until it has been re-provisioned. `scan` therefore sweeps every standard
// baud rate rather than assuming one, and `baud` does the re-provisioning.
//
// ── The second thing that will bite ──────────────────────────────────────────
// The X-series TTL bus is 5V logic; Teensy 4.1 pins are 3.3V and NOT 5V tolerant.
// If there is no level shifter between pin 14 and the servo's data line, `scan` will
// find nothing at any baud — and may damage the pin. Check the wiring before
// concluding the firmware is wrong.
//
// Wiring on the MABEL motherboard: Serial3 = Teensy pin 14 (TX3), which is pin 4 of
// the 6-pin JST SH connector silkscreened "I2C" (net GPIO4). In half-duplex
// single-wire mode the RX pin is unused — the single wire IS pin 14.
//
//   pio run -e teensy41_test_dxl -t upload
//   pio device monitor -b 115200      then type `help`

#include <Arduino.h>
#include "DxlBus.h"
#include "DxlActuator.h"

static DxlBus bus;

// Standard DYNAMIXEL baud table (Baud Rate register 8 -> bps).
struct BaudEntry { uint8_t code; uint32_t bps; };
static const BaudEntry kBauds[] = {
  {0, 9600}, {1, 57600}, {2, 115200}, {3, 1000000},
  {4, 2000000}, {5, 3000000}, {6, 4000000}, {7, 4500000},
};
static constexpr size_t kNumBauds = sizeof(kBauds) / sizeof(kBauds[0]);
static constexpr uint8_t kScanMaxId = 20;      // IDs 0..20 covers any sane setup

static uint32_t g_baud = 1000000;              // whatever we last opened the port at

static void openAt(uint32_t bps) {
  g_baud = bps;
  bus.begin(Serial3, bps);
  delay(20);
}

static void cmdScan() {
  Serial.println("scanning — every standard baud x IDs 0..20 (a factory servo is at 57600)");
  int found = 0;
  uint32_t found_baud = g_baud;
  for (size_t b = 0; b < kNumBauds; ++b) {
    openAt(kBauds[b].bps);
    for (uint8_t id = 0; id <= kScanMaxId; ++id) {
      uint16_t model = 0;
      if (!bus.ping(id, model, 12)) continue;
      ++found;
      found_baud = kBauds[b].bps;
      const char *what = (model == 1090) ? "2XL430-W250 (dual axis)"
                       : (model == 1060) ? "XL430-W250"
                       : "unknown model";
      Serial.printf("  FOUND id=%-3u model=%-5u %-24s @ %lu baud (code %u)\n",
                    id, model, what, (unsigned long)kBauds[b].bps, kBauds[b].code);
    }
  }
  if (found) {
    // Stay at the baud where something answered. Resetting to 1 Mbps here made the
    // very next `info`/`move` fail against a servo scan had just located.
    openAt(found_baud);
    Serial.printf("  (port left at %lu baud)\n", (unsigned long)found_baud);
    return;
  }
  {
    Serial.println("  nothing answered on ANY baud. In order of likelihood:");
    Serial.println("   1. no level shifter — the bus is 5V, Teensy pins are 3.3V and not 5V tolerant");
    Serial.println("   2. servo unpowered — it needs 6.5-12V on its own rail; the connector's");
    Serial.println("      +5V cannot run a 2XL430 (5V is below its 6.5V minimum)");
    Serial.println("   3. data line not on Teensy pin 14, or GND not common with the servo");
  }
}

static void dumpByte(uint8_t id, uint16_t addr, const char *label) {
  uint8_t v = 0;
  if (bus.readBlocking(id, addr, 1, &v, 1)) Serial.printf("  %-22s %u\n", label, v);
  else                                      Serial.printf("  %-22s <no reply>\n", label);
}

static void cmdInfo(uint8_t id) {
  uint16_t model = 0;
  if (!bus.ping(id, model, 30)) {
    Serial.printf("id %u did not answer at %lu baud — try `scan`\n",
                  id, (unsigned long)g_baud);
    return;
  }
  Serial.printf("id %u  model=%u  @%lu baud\n", id, model, (unsigned long)g_baud);
  dumpByte(id, 6,  "firmware version");
  dumpByte(id, 8,  "baud code");
  dumpByte(id, 9,  "return delay (x2us)");
  dumpByte(id, 11, "operating mode");
  dumpByte(id, 64, "torque enable");
  dumpByte(id, 68, "status return level");
  dumpByte(id, 70, "hardware error");
  dumpByte(id, 146,"temperature C");
  uint8_t g[4];
  if (bus.readBlocking(id, DxlActuator::kAddrGoalPosition, 4, g, 4)) {
    int32_t goal = (int32_t)((uint32_t)g[0] | ((uint32_t)g[1] << 8) |
                             ((uint32_t)g[2] << 16) | ((uint32_t)g[3] << 24));
    Serial.printf("  %-22s %ld pulses = %+.1f deg\n", "GOAL position", (long)goal,
                  (float)(goal - DxlActuator::kCenterPulse) * 360.0f / DxlActuator::kPulsesPerRev);
  }

  uint8_t buf[21];
  if (bus.readBlocking(id, DxlActuator::kAddrPresentBlock,
                       DxlActuator::kPresentBlockLen, buf, sizeof(buf))) {
    int32_t pos = (int32_t)((uint32_t)buf[6] | ((uint32_t)buf[7] << 8) |
                            ((uint32_t)buf[8] << 16) | ((uint32_t)buf[9] << 24));
    int32_t vel = (int32_t)((uint32_t)buf[2] | ((uint32_t)buf[3] << 8) |
                            ((uint32_t)buf[4] << 16) | ((uint32_t)buf[5] << 24));
    float rad = (float)(pos - DxlActuator::kCenterPulse) *
                (2.0f * 3.14159265f / DxlActuator::kPulsesPerRev);
    Serial.printf("  present position       %ld pulses = %+.4f rad (%+.1f deg)\n",
                  (long)pos, rad, rad * 57.29578f);
    Serial.printf("  present velocity       %ld raw = %+.3f rad/s\n",
                  (long)vel, (float)vel * DxlActuator::kVelUnitRadS);
  }
}

// Re-provision one servo to 1 Mbps and persist it. EEPROM writes need torque off.
// NOTE both axes of a 2XL430 share one comm module, so this changes the baud of the
// WHOLE module — do it once per module, not once per axis.
static void cmdBaud(uint8_t id) {
  uint8_t off = 0, code = 3;   // 3 = 1 Mbps
  Serial.printf("id %u: torque off (ALL axes), Baud Rate(8) <- 3 (1 Mbps)\n", id);
  // EEPROM is writable only while Torque Enable(64) is 0 on EVERY axis of the module
  // — a 2XL430 gates its whole control table on both. Releasing only `id` leaves a
  // sibling axis holding the gate shut and the write is silently REFUSED, which looks
  // exactly like a servo that ignored you.
  bus.writeBlocking(DxlBus::kBroadcastId, 64, &off, 1);
  delay(20);
  bus.writeBlocking(id, 8, &code, 1);
  delay(50);
  openAt(1000000);
  uint16_t model = 0;
  if (bus.ping(id, model, 30)) Serial.printf("  OK — answers at 1 Mbps (model %u)\n", model);
  else Serial.println("  did NOT answer at 1 Mbps — rescan to find where it landed");
}

// Energize and drive to an angle. The only command here that moves anything.
static void cmdMove(uint8_t id, float deg) {
  uint8_t mode = DxlActuator::kModePosition, on = 1, lvl = 1, d0 = 0;
  bus.writeBlocking(id, 64, &d0, 1);          // torque off to allow EEPROM writes
  bus.writeBlocking(id, 68, &lvl, 1);
  bus.writeBlocking(id, 9,  &d0, 1);
  bus.writeBlocking(id, 11, &mode, 1);
  bus.writeBlocking(id, 64, &on, 1);
  delay(10);

  float rad = deg / 57.29578f;
  int32_t pulse = (int32_t)(DxlActuator::kCenterPulse +
                            rad * (DxlActuator::kPulsesPerRev / (2.0f * 3.14159265f)));
  if (pulse < 0) pulse = 0;
  if (pulse > 4095) pulse = 4095;
  uint8_t b[4] = {(uint8_t)(pulse & 0xFF), (uint8_t)((pulse >> 8) & 0xFF),
                  (uint8_t)((pulse >> 16) & 0xFF), (uint8_t)((pulse >> 24) & 0xFF)};
  Serial.printf("id %u -> %+.1f deg (pulse %ld); watching for 1.5 s\n",
                id, deg, (long)pulse);
  bus.writeBlocking(id, DxlActuator::kAddrGoalPosition, b, 4);

  uint32_t t0 = millis();
  while (millis() - t0 < 1500) {
    uint8_t buf[21];
    if (bus.readBlocking(id, DxlActuator::kAddrPresentBlock,
                         DxlActuator::kPresentBlockLen, buf, sizeof(buf))) {
      int32_t pos = (int32_t)((uint32_t)buf[6] | ((uint32_t)buf[7] << 8) |
                              ((uint32_t)buf[8] << 16) | ((uint32_t)buf[9] << 24));
      float r = (float)(pos - DxlActuator::kCenterPulse) *
                (2.0f * 3.14159265f / DxlActuator::kPulsesPerRev);
      Serial.printf("   pos %+7.2f deg   temp %u C\n", r * 57.29578f, buf[20]);
    }
    delay(150);
  }
  Serial.println("  (still energized — type `off <id>` to release)");
}

static void help() {
  Serial.println("commands:");
  Serial.println("  scan             sweep every baud x IDs 0..20 and list what answers");
  Serial.println("  info <id>        dump identity/mode/health/position for one servo");
  Serial.println("  baud <id>        re-provision that MODULE to 1 Mbps and verify");
  Serial.println("  move <id> <deg>  energize and drive to an angle (THIS MOVES)");
  Serial.println("  off <id>         torque off");
  Serial.println("  stats            bus reply/timeout counters");
}

void setup() {
  Serial.begin(115200);
  delay(600);
  Serial.println("\n=== DYNAMIXEL bring-up diagnostic (Serial3 / Teensy pin 14, half duplex) ===");
  Serial.println("read-only unless you type `move`. Level shifter required: bus is 5V, pin is 3.3V.");
  openAt(1000000);
  help();
}

void loop() {
  static char line[64];
  static size_t n = 0;
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\r') continue;
    if (c != '\n') { if (n < sizeof(line) - 1) line[n++] = c; continue; }
    line[n] = 0; n = 0;
    if (!line[0]) { help(); continue; }

    char verb[16] = {0};
    int  id = -1;
    float arg = 0;
    int fields = sscanf(line, "%15s %d %f", verb, &id, &arg);

    if      (!strcmp(verb, "scan"))  cmdScan();
    else if (!strcmp(verb, "stats")) Serial.printf("replies=%lu timeouts=%lu\n",
                                                   (unsigned long)bus.replies(),
                                                   (unsigned long)bus.timeouts());
    else if (!strcmp(verb, "help"))  help();
    else if (fields < 2 || id < 0 || id > 252) Serial.println("need a valid <id>");
    else if (!strcmp(verb, "info"))  cmdInfo((uint8_t)id);
    else if (!strcmp(verb, "baud"))  cmdBaud((uint8_t)id);
    else if (!strcmp(verb, "off"))   { uint8_t z = 0;
                                       bus.writeBlocking((uint8_t)id, 64, &z, 1);
                                       Serial.printf("id %d torque off\n", id); }
    else if (!strcmp(verb, "move"))  { if (fields < 3) Serial.println("move <id> <deg>");
                                       else cmdMove((uint8_t)id, arg); }
    else help();
  }
}
