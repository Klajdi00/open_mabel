#pragma once
#include "ActuatorLimits.h"
#include "CanBus.h"

// Driver for one DAMIAO DM-J series actuator (DM8009 / DM4340 / DM4310 / …).
// Holds a CanBus reference, so the same DmActuator type works for actuators
// on any physical bus; the motor's own ranges are read back from it at bring-up.
//
// On construction the actuator self-registers a feedback handler on its CanBus,
// so application code never writes a dispatch switch for incoming frames.
class DmActuator {
public:
  enum Mode : uint8_t {
    MODE_MIT        = 1,
    MODE_POS_VEL    = 2,
    MODE_VEL        = 3,
    MODE_POS_TORQUE = 4,
  };

  struct Feedback {
    uint8_t id;       // motor ID (low nibble of D[0])
    uint8_t err;      // 1 = enabled OK; 8..E = fault, see manual
    float position;   // rad
    float velocity;   // rad/s
    float torque;     // Nm
    uint8_t t_mos;    // °C, driver MOSFET
    uint8_t t_rotor;  // °C, motor coil
  };

  // Float ↔ fixed-point scale for the MIT frame. p/v/t are read from the motor's
  // own PMAX/VMAX/TMAX registers at bring-up (groundLimitsFromMotor) — the struct's
  // initialisers are only what an actuator holds before that succeeds, and an
  // actuator that never grounds is never driven. kp_max/kd_max are MIT protocol
  // constants, not motor registers, so they stay fixed.
  //
  // The struct itself now lives in ActuatorLimits.h so the generic Robot/host-link
  // layer can name an actuator's ranges without naming a transport. This alias keeps
  // every existing `DmActuator::Limits` reference compiling unchanged.
  using Limits = ActuatorLimits;

  DmActuator(CanBus &bus, uint32_t id, uint32_t feedback_id = 0)
    : bus_(bus), id_(id), feedback_id_(feedback_id) {
    bus.registerHandler(feedback_id_, &DmActuator::onRxStatic, this);
  }

  uint32_t id()         const { return id_; }
  uint32_t feedbackId() const { return feedback_id_; }

  // millis() timestamp of the last successfully parsed feedback frame (0 = never).
  // Feedback is request/response, so this advances every command the host streams;
  // a stale value means the actuator stopped answering (unpowered / off the bus).
  uint32_t lastRxMs() const { return last_rx_ms_; }

  // MIT control: T_ref = kp*(p_des - q) + kd*(v_des - dq) + t_ff.
  // kd MUST be > 0 when commanding position, or the motor oscillates.
  void sendMITCommand(float p_des, float v_des, float kp, float kd, float t_ff) {
    uint16_t p   = floatToUint(p_des, -limits_.p_max,  limits_.p_max, 16);
    uint16_t v   = floatToUint(v_des, -limits_.v_max,  limits_.v_max, 12);
    uint16_t kpi = floatToUint(kp,     0.0f,           limits_.kp_max, 12);
    uint16_t kdi = floatToUint(kd,     0.0f,           limits_.kd_max, 12);
    uint16_t t   = floatToUint(t_ff,  -limits_.t_max,  limits_.t_max, 12);

    CAN_message_t msg;
    msg.id  = id_;
    msg.len = 8;
    msg.buf[0] = p >> 8;
    msg.buf[1] = p & 0xFF;
    msg.buf[2] = v >> 4;
    msg.buf[3] = ((v & 0xF) << 4) | (kpi >> 8);
    msg.buf[4] = kpi & 0xFF;
    msg.buf[5] = kdi >> 4;
    msg.buf[6] = ((kdi & 0xF) << 4) | (t >> 8);
    msg.buf[7] = t & 0xFF;
    bus_.send(msg);
  }

  // Write the control-mode register over the parameter bus (CAN ID 0x7FF).
  // Call with the motor disabled. Effect is volatile unless saveParams() runs.
  void switchMode(Mode mode) { writeRegister(10, static_cast<uint8_t>(mode)); }

  // Change the motor's slave ID (RID 8). The motor immediately starts listening
  // on `new_id` — this object's stored id_ is updated to match so subsequent
  // calls (setMasterId, saveParams, …) keep working. Volatile until saveParams().
  void setCanId(uint8_t new_id) {
    writeRegister(8, new_id);
    id_ = new_id;
  }

  // Change the feedback / master ID (RID 7). Volatile until saveParams().
  // Note: the existing CanBus dispatch entry still uses the old feedback_id_;
  // this is fine for one-shot provisioning (the actuator is short-lived) but
  // means you should not change MST_ID on a long-lived actuator without also
  // re-registering the handler.
  void setMasterId(uint8_t new_mst_id) {
    writeRegister(7, new_mst_id);
    feedback_id_ = new_mst_id;
  }

  // Commit all pending parameter writes to flash. ~100 ms blocking — the motor
  // is unresponsive while it programs its flash.
  void saveParams() {
    CAN_message_t msg;
    msg.id  = 0x7FF;
    msg.len = 8;
    msg.buf[0] = id_ & 0xFF;
    msg.buf[1] = (id_ >> 8) & 0xFF;
    msg.buf[2] = 0xAA;          // save
    msg.buf[3] = 0x01;
    msg.buf[4] = msg.buf[5] = msg.buf[6] = msg.buf[7] = 0x00;
    bus_.send(msg);
    delay(100);
  }

  // Send a sentinel frame to the motor ID. last = FC enable / FD disable / FE zero.
  void sendSpecial(uint8_t last) {
    CAN_message_t msg;
    msg.id  = id_;
    msg.len = 8;
    for (uint8_t i = 0; i < 7; i++) msg.buf[i] = 0xFF;
    msg.buf[7] = last;
    bus_.send(msg);
  }

  // The stale-command fail-safe, MIT flavour: the caller supplies the per-actuator
  // damping gain and we command kp=0 / pos=0 / vel=0 / t_ff=0 with it, i.e. resist
  // motion without commanding a position. Kept as a named operation so the host link
  // no longer has to know which numbers mean "safe" on which transport.
  void safeHold(float default_kd) { sendMITCommand(0.0f, 0.0f, 0.0f, default_kd, 0.0f); }

  void enable()  { sendSpecial(0xFC); }
  void disable() { sendSpecial(0xFD); }
  void setZero() { sendSpecial(0xFE); }

  // Latest cached feedback. Zero-initialized until first frame arrives.
  const Feedback &lastFeedback() const { return last_fb_; }

  // Parse `msg` into the internal feedback cache. Returns false on length mismatch.
  // Stamps last_rx_ms_ on success (handler runs in bus.poll() thread context, so
  // millis() is safe here — never call this from an ISR).
  bool updateFromMessage(const CAN_message_t &msg) {
    // Parameter-bus REPLIES arrive on MST_ID — the same CAN ID as feedback frames,
    // so they land in this handler and would otherwise be decoded as garbage
    // position/velocity/torque. While a read is outstanding, intercept the reply
    // instead. The signature [id_lo, id_hi, 0x33, RID, <value>] is only accepted
    // for the exact RID we asked for, so normal feedback is never swallowed.
    if (pending_rid_ != kNoPendingRid && msg.len == 8 &&
        msg.buf[2] == 0x33 && msg.buf[3] == pending_rid_) {
      uint32_t raw = (uint32_t)msg.buf[4] | ((uint32_t)msg.buf[5] << 8) |
                     ((uint32_t)msg.buf[6] << 16) | ((uint32_t)msg.buf[7] << 24);
      memcpy(&reg_value_, &raw, sizeof(reg_value_));
      reg_ok_ = true;
      return true;
    }
    if (!parseFeedback(msg, last_fb_)) return false;
    last_rx_ms_ = millis();
    return true;
  }

  // ---- limit grounding (read the motor's own fixed-point ranges) --------------
  // DAMIAO register IDs for the MIT range registers. These are the SINGLE SOURCE
  // OF TRUTH for the fixed-point scale: the motor encodes/decodes against its own
  // stored values, so any number we hold that differs silently mis-scales every
  // command and every reading by the ratio.
  static constexpr uint8_t RID_PMAX = 21;
  static constexpr uint8_t RID_VMAX = 22;
  static constexpr uint8_t RID_TMAX = 23;

  // Blocking read of one float register over the 0x7FF parameter bus. Pumps
  // bus_.poll() so the reply is dispatched into this object. Returns false on
  // timeout (motor absent / unpowered / wrong CAN ID).
  bool readRegisterFloat(uint8_t rid, float &out, uint32_t timeout_ms = 50) {
    reg_ok_      = false;
    pending_rid_ = rid;

    CAN_message_t msg;
    msg.id  = 0x7FF;
    msg.len = 8;
    msg.buf[0] = id_ & 0xFF;
    msg.buf[1] = (id_ >> 8) & 0xFF;
    msg.buf[2] = 0x33;          // read
    msg.buf[3] = rid;
    msg.buf[4] = msg.buf[5] = msg.buf[6] = msg.buf[7] = 0x00;
    bus_.send(msg);

    uint32_t t0 = millis();
    while (!reg_ok_ && (millis() - t0) < timeout_ms) bus_.poll();

    pending_rid_ = kNoPendingRid;
    if (reg_ok_) out = reg_value_;
    return reg_ok_;
  }

  // Ask the motor for PMAX/VMAX/TMAX and adopt them as this actuator's scale.
  // The encoding is then correct BY CONSTRUCTION, which also makes floatToUint()'s
  // existing clamp a true cap at the motor's real range. Returns false if the motor
  // did not answer, in which case its scale stays unknown and it must not be driven
  // — but only IT is affected; the rest of the robot runs normally.
  // `attempts` exists because ONE 50 ms miss is not evidence that a motor is absent.
  // The 48 V DM4340s lose this race at power-on — they are still coming up when the
  // Teensy asks — and a single miss condemned them to OFFLINE for the whole session:
  // never enabled, never commanded, and only visible as a joint that felt dead
  // (`late_join_enable`, found on l_j3/l_j4 by feel on 2026-08-23).
  //
  // Retry only where it is free: at boot. `serviceOffline()` passes attempts = 1
  // because it already re-probes once a second, and it runs INSIDE the control loop
  // where a multiplied timeout is a multiplied stall.
  bool groundLimitsFromMotor(uint32_t timeout_ms = 50, uint8_t attempts = 3) {
    for (uint8_t a = 0; a < attempts; ++a) {
      float p = 0, v = 0, t = 0;
      if (readRegisterFloat(RID_PMAX, p, timeout_ms) &&
          readRegisterFloat(RID_VMAX, v, timeout_ms) &&
          readRegisterFloat(RID_TMAX, t, timeout_ms) &&
          p > 0 && v > 0 && t > 0) {              // the comparisons also reject NaN
        limits_.p_max = p;
        limits_.v_max = v;
        limits_.t_max = t;
        online_ = true;
        return true;
      }
    }
    return false;
  }

  // True once the motor answered and its real ranges are loaded. An offline
  // actuator is never enabled or commanded; it needs no separate wire field
  // because it returns no feedback and so already reads clear in State.live_mask.
  bool          online() const { return online_; }
  const Limits &limits() const { return limits_; }

  // Decode an MST_ID feedback frame into a caller-supplied struct.
  bool parseFeedback(const CAN_message_t &msg, Feedback &out) const {
    if (msg.len != 8) return false;
    out.id  = msg.buf[0] & 0x0F;
    out.err = (msg.buf[0] >> 4) & 0x0F;
    uint16_t p = ((uint16_t)msg.buf[1] << 8) | msg.buf[2];
    uint16_t v = ((uint16_t)msg.buf[3] << 4) | (msg.buf[4] >> 4);
    uint16_t t = (((uint16_t)msg.buf[4] & 0x0F) << 8) | msg.buf[5];
    out.position = uintToFloat(p, -limits_.p_max, limits_.p_max, 16);
    out.velocity = uintToFloat(v, -limits_.v_max, limits_.v_max, 12);
    out.torque   = uintToFloat(t, -limits_.t_max, limits_.t_max, 12);
    out.t_mos    = msg.buf[6];
    out.t_rotor  = msg.buf[7];
    return true;
  }

private:
  static void onRxStatic(const CAN_message_t &msg, void *ctx) {
    static_cast<DmActuator*>(ctx)->updateFromMessage(msg);
  }

  // Generic single-byte register write on the 0x7FF parameter bus.
  void writeRegister(uint8_t rid, uint8_t value) {
    CAN_message_t msg;
    msg.id  = 0x7FF;
    msg.len = 8;
    msg.buf[0] = id_ & 0xFF;
    msg.buf[1] = (id_ >> 8) & 0xFF;
    msg.buf[2] = 0x55;          // write
    msg.buf[3] = rid;
    msg.buf[4] = value;
    msg.buf[5] = msg.buf[6] = msg.buf[7] = 0x00;
    bus_.send(msg);
  }

  static uint16_t floatToUint(float x, float xmin, float xmax, uint8_t bits) {
    if (x < xmin) x = xmin;
    if (x > xmax) x = xmax;
    return (uint16_t)((x - xmin) * (float)((1 << bits) - 1) / (xmax - xmin));
  }
  static float uintToFloat(uint16_t v, float xmin, float xmax, uint8_t bits) {
    return (float)v * (xmax - xmin) / (float)((1 << bits) - 1) + xmin;
  }

  static constexpr uint8_t kNoPendingRid = 0xFF;

  CanBus  &bus_;
  uint32_t id_;
  uint32_t feedback_id_;
  Limits   limits_;
  Feedback last_fb_{};
  uint32_t last_rx_ms_ = 0;

  // Parameter-read rendezvous. pending_rid_ is set only for the duration of a
  // blocking readRegisterFloat(), so the feedback path is untouched at runtime.
  uint8_t  pending_rid_ = kNoPendingRid;
  volatile bool reg_ok_ = false;
  float    reg_value_   = 0.0f;
  bool     online_      = false;
};
