#pragma once
#include "ActuatorLimits.h"
#include "DxlBus.h"

// DxlActuator — ONE axis of a ROBOTIS DYNAMIXEL (Protocol 2.0, X-series) presented
// with the same surface as DmActuator, so `Robot` and the UDP host link cannot tell
// which kind of actuator sits behind an index.
//
// "One axis" is the right unit even for a 2XL430-W250-T, which is physically one
// module with TWO output axes: each axis carries its own complete control table and
// answers on its own DXL ID, so ROBOTIS's own model is two independent servos that
// happen to share a connector. Two config entries, two DxlActuators, one DxlBus.
// (The one thing genuinely shared is the comm module: Baud Rate(8) and Protocol
// Type(13) are synced across both axes, which is why baud lives on the bus, not here.)
//
// ── What is real and what is 0 ────────────────────────────────────────────────
// The wire API is MIT-shaped (pos/vel/kp/kd/t_ff -> position/velocity/torque). A
// DYNAMIXEL in Position Control Mode implements a strict subset, so the honest
// mapping publishes 0 for everything it cannot actually do, per the convention this
// firmware already used for an offline actuator (RobotInfo publishes 0.0 ranges,
// "unknown can never be mistaken for a measurement"):
//
//   cmd.pos    -> Goal Position(116)          REAL
//   cmd.vel    -> ignored                     0   (Profile Velocity(112) is a speed
//                                                  CAP, not a velocity setpoint —
//                                                  a different quantity, not a
//                                                  coarser version of the same one)
//   cmd.kp/kd  -> ignored by default          0   (Position P/D Gain(84/80) are
//                                                  unitless 0..16383 register values,
//                                                  NOT the MIT gains that share the
//                                                  name. Passing a MIT kp=20 through
//                                                  as a P gain of 20 would leave the
//                                                  joint nearly limp. Opt in with
//                                                  `raw_gains` in robot.json if you want
//                                                  register units on the wire.)
//   cmd.t_ff   -> ignored                     0   (no torque feedforward in position
//                                                  mode at all)
//   st.position<- Present Position(132)       REAL
//   st.velocity<- Present Velocity(128)       REAL
//   st.torque  -> 0                               (Present Load(126) is 0.1%-of-max,
//                                                  not N.m; publishing it as N.m
//                                                  would be a unit lie)
//   hl.err     <- Hardware Error Status(70)       (NOT the DM error nibble — see
//                                                  `enabled()` below)
//   hl.enabled <- Torque Enable(64)           REAL
//   hl.t_rotor <- Present Temperature(146)    REAL
//   hl.t_mos   -> 0                               (one sensor, not two)
//   limits p/v/t_max -> 0                         (no MIT fixed-point scale exists;
//                                                  joint travel limits belong to
//                                                  the calibration store `pos_limits`, as for the
//                                                  arms — see ActuatorLimits.h)
//
// ⚠ `enabled` is why this class must exist rather than a shim: host_link computed it
// as `err == 1`, the DAMIAO "enabled OK" nibble. A DYNAMIXEL reports 0 for healthy,
// so that formula would report a perfectly good neck joint as de-energized forever —
// the exact signature of the `late_join_enable` bug already on the board. `enabled`
// is therefore supplied BY the actuator now, not derived by the host link.
class DxlActuator {
public:
  using Limits = ActuatorLimits;

  // Control table addresses (X-series / 2XL430-W250-T).
  static constexpr uint16_t kAddrReturnDelay   = 9;
  static constexpr uint16_t kAddrOperatingMode = 11;
  static constexpr uint16_t kAddrTorqueEnable  = 64;
  static constexpr uint16_t kAddrStatusReturn  = 68;
  static constexpr uint16_t kAddrHwError       = 70;
  static constexpr uint16_t kAddrPosDGain      = 80;
  static constexpr uint16_t kAddrPosPGain      = 84;
  static constexpr uint16_t kAddrGoalPosition  = 116;
  static constexpr uint16_t kAddrPresentBlock  = 126;  // Load..Temperature, 21 bytes
  static constexpr uint16_t kPresentBlockLen   = 21;
  static constexpr uint8_t  kModePosition      = 3;

  // 12-bit encoder over one turn; 2048 is mid-scale and is taken as joint zero.
  // The firmware reports RAW servo radians on purpose: joint zero/direction are
  // owned by the calibration store for the arms and must not acquire a second home here.
  static constexpr float kPulsesPerRev = 4096.0f;
  static constexpr int32_t kCenterPulse = 2048;
  static constexpr float kVelUnitRadS = 0.229f * 2.0f * 3.14159265f / 60.0f;  // 0.229 rev/min

  // Same field names as DmActuator::Feedback so Robot's accessor is uniform.
  struct Feedback {
    uint8_t id      = 0;
    uint8_t err     = 0;   // Hardware Error Status(70), NOT the DM nibble
    float   position = 0;  // rad
    float   velocity = 0;  // rad/s
    float   torque   = 0;  // always 0 — see the class comment
    uint8_t t_mos    = 0;  // always 0 — the servo has one temperature sensor
    uint8_t t_rotor  = 0;  // °C, Present Temperature(146)
  };

  DxlActuator(DxlBus &bus, uint8_t dxl_id, bool raw_gains)
      : bus_(bus), id_(dxl_id), raw_gains_(raw_gains) {
    bus_.registerHandler(id_, &DxlActuator::onStatusStatic, this);
    // No MIT fixed-point scale exists on this transport. Publish 0 rather than
    // DmActuator's DM4310-shaped defaults, which would read as real measurements.
    limits_.p_max = 0.0f;
    limits_.v_max = 0.0f;
    limits_.t_max = 0.0f;
    // 0..16383 if the operator opted into raw register gains; otherwise 0 =
    // "this actuator does not take gains from the wire".
    limits_.kp_max = raw_gains_ ? 16383.0f : 0.0f;
    limits_.kd_max = raw_gains_ ? 16383.0f : 0.0f;
  }

  uint32_t id()         const { return id_; }
  uint32_t feedbackId() const { return id_; }
  uint32_t lastRxMs()   const { return last_rx_ms_; }
  bool     online()     const { return online_; }
  const Limits   &limits()       const { return limits_; }
  const Feedback &lastFeedback() const { return last_fb_; }

  // Whether the output stage is actually energized.
  //
  // GROUNDED IN THE SERVO: the low-rate health poll reads Torque Enable(64) back and
  // corrects this. enable()/disable() also set it optimistically so the host sees the
  // intent immediately rather than a read-cycle late — but an optimistic flag ALONE
  // would report intent, not reality, and a servo that refused to energize (latched
  // hardware error, brown-out, torque cleared by a Bus Watchdog trip) would read
  // enabled forever. That is exactly the `late_join_enable` failure this field exists
  // to make visible, so it must not be re-introduced here.
  bool enabled() const { return torque_enabled_; }

  // ---- bring-up (BLOCKING — setup() only, mirrors groundLimitsFromMotor) ------
  // Presence check + put the servo in a known control regime. Named to match the
  // DM path so Robot::begin() and serviceOffline() read identically for both.
  bool groundLimitsFromMotor(uint32_t timeout_ms = 50) {
    if (!bus_.ready()) return false;
    uint16_t model = 0;
    if (!bus_.ping(id_, model, timeout_ms)) return false;
    model_number_ = model;

    // Torque must be off to write EEPROM (Operating Mode / Return Delay Time /
    // Status Return Level all live there).
    uint8_t off = 0;
    bus_.writeBlocking(id_, kAddrTorqueEnable, &off, 1);

    // Reply only to READ instructions. This is the single biggest half-duplex win —
    // every position write then costs its TX time alone, with no line turnaround —
    // and the runtime scheduler DEPENDS on it: it does not await anything after a
    // write, so an unexpected status reply would land mid-read and corrupt feedback.
    //
    // Therefore SET AND VERIFY, with retries. Assuming this write landed is what made
    // the first hardware run fail: the write was lost to a collision, the level stayed
    // 2, every subsequent write drew a reply, and the symptom was feedback starving
    // and goal positions never arriving — nowhere near the actual cause. If it will
    // not stick, the actuator is reported OFFLINE rather than driven on a false
    // assumption; serviceOffline() will retry it like any other absent actuator.
    uint8_t lvl_now = 0xFF;
    bool level_ok = bus_.readBlocking(id_, kAddrStatusReturn, 1, &lvl_now, 1, timeout_ms)
                    && lvl_now == 1;
    for (int attempt = 0; attempt < 2 && !level_ok; ++attempt) {
      uint8_t lvl = 1;
      bus_.writeBlocking(id_, kAddrStatusReturn, &lvl, 1);
      uint8_t rb = 0xFF;
      if (bus_.readBlocking(id_, kAddrStatusReturn, 1, &rb, 1, timeout_ms) && rb == 1)
        level_ok = true;
    }
    if (!level_ok) {
      Serial.printf("  dxl id=%u: Status Return Level would not take (still != 1) — "
                    "refusing to drive it; the scheduler requires reply-free writes\n",
                    (unsigned)id_);
      return false;
    }

    // Return Delay Time(9) and Operating Mode(11) live in EEPROM. Write them ONLY if
    // they are wrong, for two reasons:
    //   * EEPROM has a finite write budget, and bring-up runs on every boot AND on
    //     every serviceOffline() recovery — a blind write burns cells for nothing;
    //   * EEPROM is gated on Torque Enable being 0 across EVERY axis of the module.
    //     On a 2XL430 that gate is shared, so a blind write during recovery of one
    //     axis is silently REFUSED whenever its sibling is energized — the same class
    //     of invisible failure as `late_join_enable`, and it would leave the actuator
    //     apparently up but in the wrong control mode.
    // The default 250 (=500 us) answer delay is pure latency for a host that is
    // already listening, so the target is 0.
    uint8_t cur = 0xFF;
    if (!bus_.readBlocking(id_, kAddrReturnDelay, 1, &cur, 1, timeout_ms)) return false;
    if (cur != 0) {
      uint8_t delay0 = 0;
      bus_.writeBlocking(id_, kAddrReturnDelay, &delay0, 1);
    }
    cur = 0xFF;
    if (!bus_.readBlocking(id_, kAddrOperatingMode, 1, &cur, 1, timeout_ms)) return false;
    if (cur != kModePosition) {
      uint8_t mode = kModePosition;
      bus_.writeBlocking(id_, kAddrOperatingMode, &mode, 1);
      uint8_t rb = 0xFF;
      // If the module-wide EEPROM gate refused it, say so rather than driving an
      // actuator that is not in position mode.
      if (!bus_.readBlocking(id_, kAddrOperatingMode, 1, &rb, 1, timeout_ms) ||
          rb != kModePosition) {
        Serial.printf("  dxl id=%u: Operating Mode would not take (reads %u, wanted %u) "
                      "— EEPROM is gated on torque being OFF across every axis of the "
                      "module; refusing to drive it\n",
                      (unsigned)id_, rb, (unsigned)kModePosition);
        return false;
      }
    }

    online_ = true;
    armRead(kAddrPresentBlock, kPresentBlockLen);
    return true;
  }

  // MODE_* is a DM concept; accepted and ignored so Robot's bring-up is uniform.
  void switchMode(uint8_t) {}

  void enable() {
    host_enabled_ = true;
    gains_dirty_  = raw_gains_;   // re-assert gains after a re-energize
    applyTorque();
  }

  void disable() {
    host_enabled_ = false;
    applyTorque();
  }

  void setZero() {}   // DM-only (sentinel frame); calib owns zeroing on this path.

  // The stale-command fail-safe, DYNAMIXEL flavour: DO NOTHING, deliberately.
  //
  // host_link's fallback for a lapsed command is a damping hold — kp=0, kd=default,
  // pos=0 — which for a DM-J in MIT mode means "resist motion, command no position".
  // Pushed through a POSITION servo the same numbers mean "drive to raw zero", so the
  // safe hold would become the most dangerous command the joint can receive. Holding
  // the last goal is the correct analogue: the servo already resists disturbance.
  void safeHold() {}

  // True when the servo's OWN Torque Enable disagrees with what we last asked for,
  // past the settle window. That is a torque state we did not choose — a brown-out, a
  // Bus Watchdog trip, or a write that never landed — and it is the DYNAMIXEL form of
  // the `late_join_enable` fault: online, streaming, and silently not holding.
  bool torqueMismatch() const {
    return torque_applied_ != torque_enabled_ &&
           (millis() - last_torque_cmd_ms_) > kTorqueSettleMs;
  }

  // Re-assert whatever we last decided. Note it re-asserts the INTENT, so an axis
  // deliberately released by a kp == 0 command is pushed back to released, never
  // energized behind the operator's back.
  void reapplyTorque() {
    uint8_t v = torque_applied_ ? 1 : 0;
    bus_.queueWrite(id_, kAddrTorqueEnable, &v, 1);
    last_torque_cmd_ms_ = millis();
  }

  // Effective torque = the host has energized the robot AND the last command asked
  // this joint to be held. Written only on a CHANGE, so a 400 Hz command stream does
  // not put a torque write on the wire every tick.
  void applyTorque() {
    const bool want = host_enabled_ && cmd_torque_;
    if (want == torque_applied_) return;
    uint8_t v = want ? 1 : 0;
    bus_.queueWrite(id_, kAddrTorqueEnable, &v, 1);
    torque_applied_     = want;
    torque_enabled_     = want;   // optimistic; corrected by the Torque Enable readback
    last_torque_cmd_ms_ = millis();
  }

  // The MIT-shaped command entry point. Only `p_des` is honoured; see the class
  // comment for why each of the others is dropped rather than approximated.
  void sendMITCommand(float p_des, float /*v_des*/, float kp, float kd, float /*t_ff*/) {
    goal_rad_ = p_des;
    have_cmd_ = true;
    // kp == 0 means "do not hold this joint" EVERYWHERE ELSE IN THIS ROBOT: under the
    // MIT law kp=kd=t_ff=0 develops no torque and the joint is free, which is how a
    // de-activated DoF and a limp stack are expressed. A position servo has no such
    // continuum — it either holds its goal or it is de-energized — so the honest
    // translation of kp == 0 is to RELEASE TORQUE. Without it the neck stays rigid
    // under the very command that limps every DAMIAO joint beside it, which is both
    // surprising to the operator and wrong: "limp" has to mean limp for all 20.
    cmd_torque_ = (kp > 0.0f);
    applyTorque();
    if (raw_gains_) {
      uint16_t p = clampGain(kp), d = clampGain(kd);
      if (p != cmd_p_gain_ || d != cmd_d_gain_) {
        cmd_p_gain_ = p; cmd_d_gain_ = d; gains_dirty_ = true;
      }
    }
  }

  // Called every loop() by Robot. Hands the bus this actuator's next write and
  // keeps a read armed. Never blocks; the bus decides when either goes out.
  void service() {
    if (!online_) return;

    // A gain change displaces one position write every kGainSlot services. Position
    // is the hot path, so gains ride along rather than pre-empting it outright.
    // Gains are transitions, so they ride the control FIFO and both can go at once —
    // no alternation dance, and neither can be overwritten by a setpoint.
    if (gains_dirty_) {
      uint8_t g[2] = {(uint8_t)(cmd_p_gain_ & 0xFF), (uint8_t)(cmd_p_gain_ >> 8)};
      uint8_t d[2] = {(uint8_t)(cmd_d_gain_ & 0xFF), (uint8_t)(cmd_d_gain_ >> 8)};
      if (bus_.queueWrite(id_, kAddrPosPGain, g, 2) &&
          bus_.queueWrite(id_, kAddrPosDGain, d, 2))
        gains_dirty_ = false;      // retried next service() if the queue was full
    }

    // Queue a position write when the setpoint MOVED, or periodically to refresh it.
    // service() runs at loop rate (kHz); re-arming the write every call would push the
    // same unchanged pulse count onto a single-wire bus thousands of times a second,
    // crowding out telemetry for no benefit. The host still commands at the stream
    // rate — this only collapses duplicate setpoints.
    if (torque_applied_) {
      uint32_t now_w = millis();
      int32_t pulse = radToPulse(goal_rad_);
      if (pulse != last_sent_pulse_ || now_w - last_write_ms_ >= kWriteRefreshMs) {
        last_sent_pulse_ = pulse;
        last_write_ms_   = now_w;
        uint8_t b[4] = {(uint8_t)(pulse & 0xFF), (uint8_t)((pulse >> 8) & 0xFF),
                        (uint8_t)((pulse >> 16) & 0xFF), (uint8_t)((pulse >> 24) & 0xFF)};
        bus_.setWrite(id_, kAddrGoalPosition, b, 4);
      }
    }

    // Keep exactly one read armed. This doubles as timeout recovery: if a reply is
    // lost the bus simply never dispatches, and the next expiry re-arms it.
    // Cadence follows what the wire can actually carry: fast at 1 Mbps, self-limiting
    // at 57600, rather than a constant that oversubscribes a slow bus.
    uint32_t now = millis();
    uint32_t period = bus_.roundTripMs(kPresentBlockLen);
    if (period < kReadPeriodMs) period = kReadPeriodMs;
    if (now - last_read_req_ms_ >= period && !bus_.readPending(id_)) {
      last_read_req_ms_ = now;
      // Mostly the present block; every kHealthEvery-th read fetches the health byte
      // instead, which is low-rate telemetry and must not crowd out position.
      if (++read_tick_ % kHealthEvery == 0) {
        // Health telemetry alternates so BOTH bytes stay fresh without either
        // crowding out position: hardware error one slot, torque enable the next.
        health_phase_ = !health_phase_;
        armRead(health_phase_ ? kAddrHwError : kAddrTorqueEnable, 1);
      } else {
        armRead(kAddrPresentBlock, kPresentBlockLen);
      }
    }
  }

private:
  static constexpr uint32_t kReadPeriodMs = 10;   // 100 Hz state refresh
  static constexpr uint32_t kHealthEvery  = 20;   // ~5 Hz health
  static constexpr uint32_t kGainSlot     = 8;
  static constexpr uint32_t kWriteRefreshMs = 20;  // re-assert an unchanged setpoint at 50 Hz
  // A torque write and the reply to a read issued just before it can cross on a
  // single wire; ignore readbacks this soon after commanding torque.
  static constexpr uint32_t kTorqueSettleMs = 150;

  void armRead(uint16_t addr, uint16_t len) {
    bus_.setRead(id_, addr, len);
  }

  static uint16_t clampGain(float g) {
    if (!(g > 0.0f)) return 0;
    if (g > 16383.0f) return 16383;
    return (uint16_t)g;
  }

  static int32_t radToPulse(float rad) {
    float p = (float)kCenterPulse + rad * (kPulsesPerRev / (2.0f * 3.14159265f));
    if (p < 0.0f) p = 0.0f;
    if (p > kPulsesPerRev - 1.0f) p = kPulsesPerRev - 1.0f;
    return (int32_t)p;
  }
  static float pulseToRad(int32_t pulse) {
    return (float)(pulse - kCenterPulse) * (2.0f * 3.14159265f / kPulsesPerRev);
  }

  static void onStatusStatic(uint8_t err, uint16_t addr, const uint8_t *p,
                             uint8_t len, void *ctx) {
    static_cast<DxlActuator *>(ctx)->onStatus(err, addr, p, len);
  }

  // `addr` is the address the BUS actually had in flight — never this object's
  // last-armed address, which may already have moved on. See DxlBus::Handler.
  void onStatus(uint8_t /*pkt_err*/, uint16_t addr, const uint8_t *p, uint8_t len) {
    if (addr == kAddrPresentBlock && len >= kPresentBlockLen) {
      // 126 Present Load(2) | 128 Present Velocity(4) | 132 Present Position(4)
      // | 136,140 trajectories(8) | 144 Voltage(2) | 146 Temperature(1)
      int32_t vel   = (int32_t)((uint32_t)p[2] | ((uint32_t)p[3] << 8) |
                                ((uint32_t)p[4] << 16) | ((uint32_t)p[5] << 24));
      int32_t pos   = (int32_t)((uint32_t)p[6] | ((uint32_t)p[7] << 8) |
                                ((uint32_t)p[8] << 16) | ((uint32_t)p[9] << 24));
      last_fb_.id       = id_;
      last_fb_.position = pulseToRad(pos);
      last_fb_.velocity = (float)vel * kVelUnitRadS;
      last_fb_.torque   = 0.0f;                 // Present Load is %, not N.m
      last_fb_.t_mos    = 0;                    // one sensor on this part
      last_fb_.t_rotor  = p[20];                // Present Temperature(146), °C
      last_rx_ms_       = millis();
      // Until a host has actually commanded this joint, the goal FOLLOWS the measured
      // position. Otherwise goal_rad_ sits at its 0.0 initialiser and the first
      // enable() drives the joint to RAW ZERO at full speed — which on a servo whose
      // mechanical neutral is nowhere near its encoder centre is a full-travel slam.
      // Measured on the robot: an axis resting at pulse 4042 (+175.3 deg) would have
      // swung ~175 deg the instant torque came on.
      if (!have_cmd_) goal_rad_ = last_fb_.position;
    } else if (addr == kAddrHwError && len >= 1) {
      last_fb_.err = p[0];
      last_rx_ms_  = millis();
    } else if (addr == kAddrTorqueEnable && len >= 1) {
      // Correct the optimistic flag from the servo itself — but not within the
      // settle window after our own enable()/disable(), where a reply already in
      // flight still carries the OLD value and would flip the flag straight back.
      if (millis() - last_torque_cmd_ms_ > kTorqueSettleMs)
        torque_enabled_ = (p[0] != 0);
      last_rx_ms_ = millis();
    }
  }

  DxlBus  &bus_;
  uint8_t  id_;
  bool     raw_gains_ = false;

  Limits   limits_{};
  Feedback last_fb_{};
  uint32_t last_rx_ms_ = 0;
  bool     online_ = false;
  uint16_t model_number_ = 0;

  float    goal_rad_ = 0.0f;
  bool     have_cmd_ = false;      // a host has commanded this joint at least once
  bool     host_enabled_ = false;  // Robot::enableAll/disableAll (host connection state)
  bool     cmd_torque_   = true;   // last command's kp > 0 (i.e. "hold this joint")
  bool     torque_applied_ = false;
  bool     torque_enabled_ = false;

  uint16_t cmd_p_gain_ = 0, cmd_d_gain_ = 0;
  bool     gains_dirty_ = false, gain_phase_ = false;
  uint32_t svc_tick_ = 0, read_tick_ = 0;
  bool     health_phase_ = false;
  uint32_t last_torque_cmd_ms_ = 0;

  uint32_t last_read_req_ms_ = 0;
  int32_t  last_sent_pulse_ = INT32_MIN;
  uint32_t last_write_ms_ = 0;
};
