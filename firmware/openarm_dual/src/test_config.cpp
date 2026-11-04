// test_config.cpp — Phase-1 standalone verification of the SD/JSON config layer.
//
//   pio run -e teensy41_test_config -t upload
//   pio device monitor -b 115200
//
// Loads /robot.json off the built-in SD card and prints the parsed RobotConfig (or
// the validation error). Touches NO CAN bus and NO motors — pure config round-trip.

#include <Arduino.h>
#include "config_loader.h"

static RobotConfig g_config;
static ConfigResult g_res;

// Low-level SD probe to separate "card not detected" (hardware/seating) from
// "card detected but filesystem won't mount" (formatting). SD.sdfs is the Teensy
// SD library's underlying SdFs object.
static void probeSd() {
  bool ok = SD.begin(BUILTIN_SDCARD);
  Serial.printf("  SD.begin -> %s\n", ok ? "OK" : "FAIL");
  uint32_t sectors = SD.sdfs.card() ? SD.sdfs.card()->sectorCount() : 0;
  Serial.printf("  card sectors=%lu (%lu MB)  fatType=%d\n",
                (unsigned long)sectors, (unsigned long)(sectors / 2048), SD.sdfs.fatType());
  Serial.printf("  sdErrorCode=0x%02X sdErrorData=0x%02X\n",
                SD.sdfs.sdErrorCode(), SD.sdfs.sdErrorData());
  if (sectors == 0)
    Serial.println("  => card NOT detected: seating / wiring / card-dead");
  else
    Serial.println("  => card detected; if FAIL it's a filesystem/format problem");
}

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 3000) {}
  delay(200);
  g_res = cfg::loadConfig(g_config);
}

// Reprint the result every 2 s so a monitor attached at any time sees it.
void loop() {
  Serial.println("\n=== config loader test ===");
  if (!g_res.ok) {
    Serial.printf("CONFIG ERROR: %s\n", g_res.error);
    probeSd();
    Serial.println("(fail-safe: a real boot would refuse to enable any motor)");
  } else {
    Serial.println("CONFIG OK:");
    cfg::printConfig(g_config);
  }
  delay(2000);
}
