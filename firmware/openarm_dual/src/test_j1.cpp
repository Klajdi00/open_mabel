// Focused single-actuator diagnostic for J1.
//
// Deliberately bypasses the CanBus / OpenArmActuator abstraction and talks to
// the CAN2 peripheral directly, so a communication failure can be isolated to
// the wire / transceiver / motor rather than the dispatch layer.
//
// J1 wiring:  bus2 = CAN2 (pins 0/1), CAN_ID 0x01, MST_ID 0x11.
// (J1 was reprovisioned to the OpenArms convention joint_1 -> recv id 0x11; the
//  old 0x00 in OpenArm.cpp / earlier notes is stale.)
//
// Build & run:
//   ~/.platformio/penv/bin/pio run -e teensy41_test_j1 -t upload
//   ~/.platformio/penv/bin/pio device monitor -b 115200
//
// SAFE BY DESIGN: after enable it streams MIT frames with kp=kd=t_ff=0, i.e. a
// pure feedback request with zero commanded torque — the motor should NOT move.

#include <Arduino.h>
#include <FlexCAN_T4.h>

FlexCAN_T4<CAN2, RX_SIZE_256, TX_SIZE_16> Can2;

static constexpr uint8_t J1_CAN_ID = 0x01;  // frame ID we send commands to
static constexpr uint8_t J1_MST_ID = 0x11;  // feedback ID we expect back

// DM8009-class fixed-point limits (T_MAX read live from J1 = 54 Nm, not the
// J8009P stock 40 Nm; mismatched T_MAX silently mis-scales torque feedback).
static constexpr float P_MAX = 12.5f, V_MAX = 45.0f, T_MAX = 54.0f,
                       KP_MAX = 500.0f, KD_MAX = 5.0f;

static uint16_t f2u(float x, float lo, float hi, uint8_t bits) {
  if (x < lo) x = lo;
  if (x > hi) x = hi;
  return (uint16_t)((x - lo) * (float)((1 << bits) - 1) / (hi - lo));
}
static float u2f(uint16_t v, float lo, float hi, uint8_t bits) {
  return (float)v * (hi - lo) / (float)((1 << bits) - 1) + lo;
}

static uint32_t tx_count = 0, tx_fail = 0, rx_count = 0;
static uint32_t rx_on_expected = 0, rx_on_other = 0;
static uint8_t  last_err = 0xFF;
static CAN_message_t last_fb{};
static bool     have_fb = false;

// Read the FlexCAN2 fault-confinement state straight from the peripheral so we
// can tell "transmitting but nobody ACKs" (-> bus-off, TX err counter pinned)
// apart from any other failure. ECR bits[7:0]=TXERR, [15:8]=RXERR; ESR1[5:4]=
// FLT_CONF (0=ErrorActive,1=ErrorPassive,2/3=BusOff).
static void printCanState() {
  uint32_t ecr  = FLEXCAN2_ECR;
  uint32_t esr1 = FLEXCAN2_ESR1;
  uint8_t txerr = ecr & 0xFF;
  uint8_t rxerr = (ecr >> 8) & 0xFF;
  uint8_t flt   = (esr1 >> 4) & 0x3;
  const char *state = (flt == 0) ? "ERR_ACTIVE" : (flt == 1) ? "ERR_PASSIVE" : "BUS_OFF";
  Serial.print("  [CAN2 ESR1=0x");
  Serial.print(esr1, HEX);
  Serial.print(" txerr=");
  Serial.print(txerr);
  Serial.print(" rxerr=");
  Serial.print(rxerr);
  Serial.print(" state=");
  Serial.print(state);
  Serial.print("]");
}

static void printFrame(const char *tag, const CAN_message_t &m) {
  Serial.print(tag);
  Serial.print(" id=0x");
  Serial.print(m.id, HEX);
  Serial.print(" len=");
  Serial.print(m.len);
  Serial.print(" data=");
  for (uint8_t i = 0; i < m.len; i++) {
    if (m.buf[i] < 0x10) Serial.print('0');
    Serial.print(m.buf[i], HEX);
    Serial.print(' ');
  }
  Serial.println();
}

static void txSpecial(uint8_t last) {
  CAN_message_t m;
  m.id = J1_CAN_ID;
  m.len = 8;
  for (uint8_t i = 0; i < 7; i++) m.buf[i] = 0xFF;
  m.buf[7] = last;
  int r = Can2.write(m);
  tx_count++;
  if (r <= 0) tx_fail++;
  Serial.print("[TX special 0x");
  Serial.print(last, HEX);
  Serial.print("] write()=");
  Serial.println(r);
}

static void txMIT(float p, float v, float kp, float kd, float tff) {
  uint16_t P = f2u(p, -P_MAX, P_MAX, 16);
  uint16_t V = f2u(v, -V_MAX, V_MAX, 12);
  uint16_t KP = f2u(kp, 0.0f, KP_MAX, 12);
  uint16_t KD = f2u(kd, 0.0f, KD_MAX, 12);
  uint16_t T = f2u(tff, -T_MAX, T_MAX, 12);

  CAN_message_t m;
  m.id = J1_CAN_ID;
  m.len = 8;
  m.buf[0] = P >> 8;
  m.buf[1] = P & 0xFF;
  m.buf[2] = V >> 4;
  m.buf[3] = ((V & 0xF) << 4) | (KP >> 8);
  m.buf[4] = KP & 0xFF;
  m.buf[5] = KD >> 4;
  m.buf[6] = ((KD & 0xF) << 4) | (T >> 8);
  m.buf[7] = T & 0xFF;
  int r = Can2.write(m);
  tx_count++;
  if (r <= 0) tx_fail++;
}

void setup() {
  Serial.begin(115200);
  delay(600);
  Serial.println();
  Serial.println("=== J1 focused CAN diagnostic (CAN2 @ 1 Mbps) ===");
  Serial.print("CAN_ID=0x");
  Serial.print(J1_CAN_ID, HEX);
  Serial.print("  expected MST_ID=0x");
  Serial.println(J1_MST_ID, HEX);

  Can2.begin();
  Can2.setBaudRate(1000000);
  Can2.setMaxMB(16);
  Can2.enableFIFO();
  // No FIFO interrupt: we poll Can2.read() in loop() (thread context only).

  delay(100);
  Serial.println("-- enabling motor (0xFC) --");
  txSpecial(0xFC);
  delay(20);
}

void loop() {
  // 1) Drain every received frame, print it raw, classify by ID.
  CAN_message_t rx;
  while (Can2.read(rx)) {
    rx_count++;
    if (rx.id == J1_MST_ID) {
      rx_on_expected++;
      last_err = (rx.buf[0] >> 4) & 0x0F;
      last_fb = rx;
      have_fb = true;
    } else {
      rx_on_other++;
    }
    // Print the first 20 frames verbatim, then go quiet (summary takes over).
    if (rx_count <= 20) printFrame("[RX]", rx);
  }

  // 2) Stream a safe zero-torque MIT request at 100 Hz (feedback request only).
  static elapsedMicros tick;
  if (tick >= 10000) {
    tick -= 10000;
    txMIT(0, 0, 0, 0, 0);  // kp=kd=tff=0 -> no motion, still elicits feedback
  }

  // 3) 1 Hz summary.
  static elapsedMillis report;
  if (report >= 1000) {
    report -= 1000;
    Serial.print("[SUMMARY] tx=");
    Serial.print(tx_count);
    Serial.print(" tx_fail=");
    Serial.print(tx_fail);
    Serial.print(" rx=");
    Serial.print(rx_count);
    Serial.print(" (expected_id=");
    Serial.print(rx_on_expected);
    Serial.print(", other_id=");
    Serial.print(rx_on_other);
    Serial.print(")");
    if (have_fb) {
      uint16_t p = ((uint16_t)last_fb.buf[1] << 8) | last_fb.buf[2];
      uint16_t v = ((uint16_t)last_fb.buf[3] << 4) | (last_fb.buf[4] >> 4);
      uint16_t t = (((uint16_t)last_fb.buf[4] & 0x0F) << 8) | last_fb.buf[5];
      Serial.print("  err=0x");
      Serial.print(last_err, HEX);
      Serial.print(" pos=");
      Serial.print(u2f(p, -P_MAX, P_MAX, 16), 3);
      Serial.print(" vel=");
      Serial.print(u2f(v, -V_MAX, V_MAX, 12), 3);
      Serial.print(" tau=");
      Serial.print(u2f(t, -T_MAX, T_MAX, 12), 3);
      Serial.print(" Tmos=");
      Serial.print(last_fb.buf[6]);
      Serial.print(" Trot=");
      Serial.print(last_fb.buf[7]);
    } else {
      Serial.print("  <no feedback yet>");
    }
    printCanState();
    Serial.println();
  }
}
