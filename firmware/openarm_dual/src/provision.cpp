// Standalone one-shot motor provisioning sketch.
//
// Build & flash with:
//   pio run -e teensy41_provision -t upload
//
// Workflow:
//   1. Edit the constants below to match the motor's CURRENT id and the
//      desired NEW (CAN_ID, MST_ID).
//   2. Put a SINGLE motor on the bus (multiple motors with the same factory
//      default 0x01 will collide).
//   3. Flash this firmware. The Teensy disables the motor, writes the new
//      IDs, saves to flash, prints a confirmation, then halts.
//   4. Power-cycle the motor. Re-flash with `pio run -t upload` to return to
//      the normal control firmware.

#include <FlexCAN_T4.h>
#include "CanBus.h"
#include "DmActuator.h"

constexpr uint32_t PROVISION_FROM_ID = 0x01;  // motor's current ID
constexpr uint32_t PROVISION_NEW_ID  = 0x02;  // desired new CAN_ID
constexpr uint32_t PROVISION_NEW_MST = 0x12;  // desired new MST_ID

FlexCAN_T4<CAN2, RX_SIZE_256, TX_SIZE_16> Can2;
CanBus bus2;

void setup() {
  Serial.begin(115200); delay(400);
  bus2.begin(Can2);
  delay(100);

  DmActuator provisioner(bus2, PROVISION_FROM_ID);
  provisioner.disable();
  delay(5);
  provisioner.setCanId(PROVISION_NEW_ID);     // motor now listens on new ID
  delay(20);
  provisioner.setMasterId(PROVISION_NEW_MST);
  delay(20);
  provisioner.saveParams();                   // persist to flash

  Serial.print("Provisioned motor 0x"); Serial.print(PROVISION_FROM_ID, HEX);
  Serial.print(" -> CAN_ID=0x"); Serial.print(PROVISION_NEW_ID, HEX);
  Serial.print(" MST_ID=0x");    Serial.println(PROVISION_NEW_MST, HEX);
  Serial.println("Power-cycle the motor and reflash with the main firmware.");
}

void loop() {
  bus2.poll();
  delay(1000);
}
