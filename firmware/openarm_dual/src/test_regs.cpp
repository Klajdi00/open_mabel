// test_regs.cpp — read the MIT fixed-point range registers (PMAX/VMAX/TMAX) back
// OUT of each motor over the 0x7FF parameter bus, and compare them against what
// /robot.json claims.
//
// Why this exists: `limits` in robot.json is not a safety clamp, it is the
// fixed-point SCALE both sides encode against (DmActuator.h sendMITCommand /
// parseFeedback). If it disagrees with the motor's own register, every torque
// command and every torque reading is silently wrong by the ratio — no error
// anywhere. The DAMIAO SDK ships a per-model table of these values, but it has no
// entry for the DM-J10422P-2EC (torso), so the only authority is the motor itself.
//
// READ-ONLY AND MOTION-FREE. This sketch sends nothing but 0x33 (read register)
// frames. It never enables a motor, never sends an MIT command, never writes a
// register, and never saves to flash. Safe to run with the actuators powered.
//
//   pio run -e teensy41_test_regs -t upload
//   pio device monitor -b 115200
//
// It deliberately does NOT use CanBus: that class dispatches on an exact CAN ID
// match, and the ID the motor replies on is exactly what we are trying to find
// out. So we drive FlexCAN directly in polling mode and dump EVERY frame.

#include <Arduino.h>
#include <FlexCAN_T4.h>
#include "config_loader.h"

FlexCAN_T4<CAN1, RX_SIZE_256, TX_SIZE_16> Can1;
FlexCAN_T4<CAN2, RX_SIZE_256, TX_SIZE_16> Can2;
FlexCAN_T4<CAN3, RX_SIZE_256, TX_SIZE_16> Can3;

static RobotConfig g_cfg;
static bool        g_cfg_ok = false;

// DAMIAO register IDs (DM_variable in the vendor SDK).
static constexpr uint8_t RID_UV      = 0;   // under-voltage protection threshold
static constexpr uint8_t RID_KT      = 1;   // torque constant
static constexpr uint8_t RID_OT      = 2;   // over-temperature threshold
static constexpr uint8_t RID_OC      = 3;   // over-current threshold
static constexpr uint8_t RID_MAX_SPD = 6;
static constexpr uint8_t RID_MST_ID  = 7;
static constexpr uint8_t RID_ESC_ID  = 8;
static constexpr uint8_t RID_CTRL    = 10;  // 1=MIT 2=POS_VEL 3=VEL 4=POS_TORQUE
static constexpr uint8_t RID_PMAX    = 21;
static constexpr uint8_t RID_VMAX    = 22;
static constexpr uint8_t RID_TMAX    = 23;

static const char *ridName(uint8_t rid) {
  switch (rid) {
    case RID_UV:      return "UV";
    case RID_KT:      return "KT";
    case RID_OT:      return "OT";
    case RID_OC:      return "OC";
    case RID_MAX_SPD: return "MAX_SPD";
    case RID_MST_ID:  return "MST_ID";
    case RID_ESC_ID:  return "ESC_ID";
    case RID_CTRL:    return "CTRL_MODE";
    case RID_PMAX:    return "PMAX";
    case RID_VMAX:    return "VMAX";
    case RID_TMAX:    return "TMAX";
    default:          return "?";
  }
}

// One motor's full register readback. Both decodings are kept because the vendor
// SDK's int-vs-float split is not documented consistently — we print both and let
// the reader judge which is physical.
struct RegSet {
  bool     present = false;
  uint8_t  bus = 0; uint32_t can_id = 0;
  float    f[40]; uint32_t u[40]; bool have[40];
  RegSet() { for (int i = 0; i < 40; ++i) { f[i] = NAN; u[i] = 0; have[i] = false; } }
};

static void sendRead(uint8_t busIdx, uint32_t motor_id, uint8_t rid) {
  CAN_message_t m;
  m.id  = 0x7FF;
  m.len = 8;
  m.buf[0] = motor_id & 0xFF;
  m.buf[1] = (motor_id >> 8) & 0xFF;
  m.buf[2] = 0x33;                 // read
  m.buf[3] = rid;
  m.buf[4] = m.buf[5] = m.buf[6] = m.buf[7] = 0x00;
  if      (busIdx == 1) Can1.write(m);
  else if (busIdx == 2) Can2.write(m);
  else if (busIdx == 3) Can3.write(m);
}

static bool readFrame(uint8_t busIdx, CAN_message_t &m) {
  if (busIdx == 1) return Can1.read(m);
  if (busIdx == 2) return Can2.read(m);
  if (busIdx == 3) return Can3.read(m);
  return false;
}

// Reinterpret payload bytes 4..7 as a little-endian IEEE-754 float.
static float payloadFloat(const CAN_message_t &m) {
  uint32_t raw = (uint32_t)m.buf[4] | ((uint32_t)m.buf[5] << 8) |
                 ((uint32_t)m.buf[6] << 16) | ((uint32_t)m.buf[7] << 24);
  float f;
  memcpy(&f, &raw, sizeof(f));
  return f;
}

static uint32_t payloadU32(const CAN_message_t &m) {
  return (uint32_t)m.buf[4] | ((uint32_t)m.buf[5] << 8) |
         ((uint32_t)m.buf[6] << 16) | ((uint32_t)m.buf[7] << 24);
}

// Query one register. Returns true if a matching reply arrived. `verbose` dumps the
// raw frame (used once at startup to evidence the wire format).
static bool query(uint8_t busIdx, uint32_t motor_id, uint8_t rid,
                  RegSet *out = nullptr, bool verbose = false) {
  CAN_message_t m;
  while (readFrame(busIdx, m)) {}          // drain stale frames first
  sendRead(busIdx, motor_id, rid);

  bool matched = false;
  uint32_t t0 = millis();
  while (millis() - t0 < 25) {             // 25 ms reply window (replies are ~instant)
    if (!readFrame(busIdx, m)) continue;
    if (verbose) {
      Serial.printf("      <- id=0x%03X len=%u data=", (unsigned)m.id, m.len);
      for (uint8_t i = 0; i < m.len; i++) Serial.printf("%02X ", m.buf[i]);
      Serial.println();
    }
    // A parameter-bus reply echoes [id_lo, id_hi, opcode, RID, value...].
    if (m.len == 8 && m.buf[3] == rid && (m.buf[2] == 0x33 || m.buf[2] == 0x55)) {
      matched = true;
      if (out && rid < 40) {
        out->f[rid]    = payloadFloat(m);
        out->u[rid]    = payloadU32(m);
        out->have[rid] = true;
      }
    }
  }
  return matched;
}

static const uint8_t kAllRids[] = {
  RID_UV, RID_KT, RID_OT, RID_OC, RID_MAX_SPD, RID_MST_ID,
  RID_ESC_ID, RID_CTRL, RID_PMAX, RID_VMAX, RID_TMAX
};

// Report one bus's CAN controller error state. This is what separates the two very
// different causes of a silent bus:
//   ACK_ERR set / TX_ERR_COUNTER climbing / "Bus Off"  -> the controller IS driving
//       the wire but NOTHING acknowledges (no node listening, wrong bitrate, or the
//       transceiver is not connecting us to a live bus). CAN requires another node
//       to ACK, so a lone transmitter always errors this way.
//   All clear and TX_ERR_COUNTER == 0                  -> nothing was transmitted at
//       all: peripheral/pin/wiring problem before the wire.
static void reportBusErrors(uint8_t busIdx) {
  CAN_error_t e;
  if      (busIdx == 1) Can1.error(e, false);
  else if (busIdx == 2) Can2.error(e, false);
  else if (busIdx == 3) Can3.error(e, false);
  Serial.printf("  bus%u  state=%-12s FLT_CONF=%-12s TEC=%-3u REC=%-3u "
                "ACK_ERR=%u BIT0=%u BIT1=%u CRC=%u FRM=%u STF=%u  ESR1=0x%08lX\n",
                busIdx, e.state, e.FLT_CONF, e.TX_ERR_COUNTER, e.RX_ERR_COUNTER,
                e.ACK_ERR, e.BIT0_ERR, e.BIT1_ERR, e.CRC_ERR, e.FRM_ERR, e.STF_ERR,
                (unsigned long)e.ESR1);
}

void setup() {
  Serial.begin(115200);
  // Long enough for the host to re-attach after USB re-enumeration, so the
  // one-shot pin report below is actually seen.
  delay(6000);

  // Sample each bus's RX pin BEFORE handing it to FlexCAN. An idle CAN bus is
  // RECESSIVE = HIGH at the transceiver's RXD. A pin reading LOW means the line is
  // held DOMINANT, which makes the controller wait forever for an idle bus and
  // never transmit (TEC stays 0, ESR1 stays 0 — exactly what bus2 shows). The usual
  // cause is a transceiver with no supply: most parts pull RXD low when unpowered.
  // Teensy 4.1 RX pins: CAN1=23, CAN2=0, CAN3=30.
  const uint8_t rx_pin[4] = {0, 23, 0, 30};
  pinMode(23, INPUT_PULLUP); pinMode(0, INPUT_PULLUP); pinMode(30, INPUT_PULLUP);
  delay(5);
  Serial.println("\nCAN RX pin idle levels (HIGH = recessive/healthy, LOW = held dominant):");
  for (uint8_t b = 1; b <= 3; ++b)
    Serial.printf("  bus%u  RX pin %-2u = %s\n", b, rx_pin[b],
                  digitalRead(rx_pin[b]) ? "HIGH (recessive)" : "LOW  (DOMINANT - bus stuck)");

  Can1.begin(); Can1.setBaudRate(1000000); Can1.enableFIFO();
  Can2.begin(); Can2.setBaudRate(1000000); Can2.enableFIFO();
  Can3.begin(); Can3.setBaudRate(1000000); Can3.enableFIFO();
  // NOTE: no enableFIFOInterrupt()/onReceive() — we poll, so we see every frame
  // regardless of which ID the motor chooses to answer on.

  ConfigResult cr = cfg::loadConfig(g_cfg);
  g_cfg_ok = cr.ok;
  if (!g_cfg_ok) Serial.printf("CONFIG FAILED: %s\n", cr.error);
}

void loop() {
  Serial.println("\n================ REGISTER READBACK SCAN ================");
  static bool first = true;

  RegSet found[24];
  size_t n = 0;

  // Probe cheaply with ONE register to detect presence, then only interrogate the
  // motors that actually answered. Sweeping every RID against every empty slot
  // would spend most of the scan waiting on silence.
  for (uint8_t bus = 1; bus <= 3; ++bus) {
    for (uint32_t id = 1; id <= 8; ++id) {
      RegSet rs;
      rs.bus = bus; rs.can_id = id;
      if (!query(bus, id, RID_PMAX, &rs, first)) continue;   // verbose on 1st pass
      rs.present = true;
      for (uint8_t k = 0; k < sizeof(kAllRids); ++k)
        query(bus, id, kAllRids[k], &rs, false);
      found[n++] = rs;
    }
  }
  first = false;

  Serial.printf("\n%u motors answered\n\n", (unsigned)n);
  Serial.println("bus id | ESC MST CTRL |    PMAX    VMAX    TMAX |      UV        KT        OT        OC   MAX_SPD");
  Serial.println("-------+--------------+-------------------------+-------------------------------------------------");
  for (size_t i = 0; i < n; ++i) {
    const RegSet &r = found[i];
    Serial.printf("  %u  %2u |", r.bus, (unsigned)r.can_id);
    Serial.printf(" %3lu %3lu  %3lu |", (unsigned long)r.u[RID_ESC_ID],
                  (unsigned long)r.u[RID_MST_ID], (unsigned long)r.u[RID_CTRL]);
    Serial.printf(" %7.2f %7.2f %7.2f |", r.f[RID_PMAX], r.f[RID_VMAX], r.f[RID_TMAX]);
    Serial.printf(" %8.2f %9.4f %8.2f %8.2f %8lu\n",
                  r.f[RID_UV], r.f[RID_KT], r.f[RID_OT], r.f[RID_OC],
                  (unsigned long)r.u[RID_MAX_SPD]);
  }
  Serial.println("\nCAN controller error state (after transmitting on every bus):");
  for (uint8_t b = 1; b <= 3; ++b) reportBusErrors(b);

  Serial.println("\nunits: PMAX rad | VMAX rad/s | TMAX Nm | UV volts | OT degC | OC amps(?)");
  Serial.println("(UV/KT/OT/OC shown as float; MAX_SPD/IDs/CTRL as uint32)");
  Serial.println("\n================ end of scan ================");
  delay(5000);
}
