#include <Arduino.h>
#include "Robot.h"

// An unpopulated slot has no ranges at all. All-zero, not the ActuatorLimits
// defaults, for the same reason RobotInfo publishes 0 for an offline actuator:
// a fallback initialiser is some other motor's numbers and would read as real.
const ActuatorLimits Robot::kNoLimits = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

// Bring one DM-J actuator into MIT closed-loop control: clear any latched state,
// switch the control mode, then enable. Short delays let each parameter-bus write
// land. (The DYNAMIXEL path does its own equivalent inside
// DxlActuator::groundLimitsFromMotor: torque off -> EEPROM writes -> position mode.)
static void bringUp(DmActuator &a) {
  a.disable();
  delay(5);
  a.switchMode(DmActuator::MODE_MIT);
  delay(5);
  a.enable();
}

// Each DM actuator self-registers one feedback handler on its bus, and a config MAY put
// every actuator on a single bus. If the handler table were the smaller of the two, the
// surplus actuators would register nothing and go permanently silent with no error --
// the constructor cannot report the failure. Keep the capacities ordered instead.
static_assert(MAX_ACTUATORS <= CanBus::kMaxHandlers,
              "CanBus::kMaxHandlers must be >= MAX_ACTUATORS, else actuators beyond the "
              "table size silently lose CAN feedback when packed onto one bus");

bool Robot::begin(const RobotConfig &cfg, CanBus *const *buses, size_t numBuses,
                  DxlBus *const *dxlBuses, size_t numDxlBuses) {
  // Validate every bus/port reference up front — bring up nothing if any is
  // unresolvable, so a bad config can't leave a half-initialized robot with some
  // motors live. A DYNAMIXEL entry naming a port the application did not provide is
  // exactly as fatal as a bad CAN bus index, and for the same reason.
  for (size_t i = 0; i < cfg.count; ++i) {
    const ActuatorConfig &ac = cfg.actuators[i];
    if (ac.transport == Transport::DYNAMIXEL) {
      if (ac.port == 0 || ac.port >= numDxlBuses || dxlBuses == nullptr ||
          dxlBuses[ac.port] == nullptr)
        return false;
    } else {
      if (ac.bus == 0 || ac.bus >= numBuses || buses[ac.bus] == nullptr) return false;
    }
  }

  cfg_   = cfg;
  count_ = cfg.count;
  for (size_t i = 0; i < count_; ++i) {
    const ActuatorConfig &ac = cfg_.actuators[i];

    if (ac.transport == Transport::DYNAMIXEL) {
      actuators_[i].emplace<DxlActuator>(*dxlBuses[ac.port], ac.dxl_id, ac.raw_gains);
      // Same contract as the DM path: an actuator that does not answer is reported
      // OFFLINE and skipped, keeps its index, and costs nobody else control.
      if (!std::get<DxlActuator>(actuators_[i]).groundLimitsFromMotor()) {
        Serial.printf("  [%u] %-10s OFFLINE (dxl port%u id=%u did not answer) — "
                      "skipped, rest of the robot unaffected\n",
                      (unsigned)i, ac.name, ac.port, (unsigned)ac.dxl_id);
        continue;
      }
      // Configured and in position mode, but NOT energized: whether to enable
      // depends on host connection state, which the Robot does not own.
      continue;
    }

    actuators_[i].emplace<DmActuator>(*buses[ac.bus], ac.can_id, ac.mst_id);

    // The motor is the authority on its own fixed-point ranges: ask it. A number
    // typed into robot.json that disagreed would silently rescale every command
    // and every reading, with no error anywhere.
    //
    // A motor that does not answer is reported OFFLINE and skipped — it is NOT a
    // reason to refuse the whole robot. One dead actuator must never cost control
    // of the other seventeen. It keeps its index (the host layout is index-
    // addressed, so nothing may shift underneath it) and reads clear in
    // State.live_mask because it returns no feedback.
    DmActuator &dm = std::get<DmActuator>(actuators_[i]);
    if (!dm.groundLimitsFromMotor()) {
      Serial.printf("  [%u] %-10s OFFLINE (bus%u can_id=0x%02X did not answer) — "
                    "skipped, rest of the robot unaffected\n",
                    (unsigned)i, ac.name, ac.bus, (unsigned)ac.can_id);
      continue;
    }
    bringUp(dm);
  }
  return true;
}

// Re-query one offline actuator per interval. See the header for the cost argument.
int Robot::serviceOffline(uint32_t interval_ms, uint32_t timeout_ms) {
  if (count_ == 0) return -1;
  uint32_t now = millis();
  if (now - last_retry_ms_ < interval_ms) return -1;
  last_retry_ms_ = now;

  for (size_t k = 0; k < count_; ++k) {
    size_t i = (retry_idx_ + k) % count_;
    if (online(i)) continue;
    retry_idx_ = (i + 1) % count_;          // advance regardless, so one dead
                                            // actuator cannot starve the others

    if (auto *x = std::get_if<DxlActuator>(&actuators_[i])) {
      // The DYNAMIXEL probe re-runs its own bring-up (ping + control-regime writes)
      // and leaves torque OFF, matching the DM branch below.
      if (!x->groundLimitsFromMotor(timeout_ms)) return -1;
      return (int)i;
    }

    auto *d = std::get_if<DmActuator>(&actuators_[i]);
    if (!d) return -1;
    if (!d->groundLimitsFromMotor(timeout_ms, /*attempts=*/1)) return -1;

    // It answered. Put it in MIT mode but leave it DISABLED: whether to energize
    // depends on host connection state, which the Robot does not own.
    d->disable();
    delay(5);
    d->switchMode(DmActuator::MODE_MIT);
    delay(5);
    return (int)i;
  }
  return -1;                                 // nothing offline
}

int Robot::indexOf(const char *name) const {
  for (size_t i = 0; i < count_; ++i)
    if (strcmp(cfg_.actuators[i].name, name) == 0) return (int)i;
  return -1;
}
