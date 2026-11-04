#pragma once
#include <array>
#include <variant>
#include "DmActuator.h"
#include "DxlActuator.h"
#include "RobotConfig.h"

// Robot — a generic collection of actuators, driven from a uniform index-addressed
// surface and aggregated onto the UDP host link. NOT arm-specific and, since the
// DYNAMIXEL extension, NOT transport-specific either: the actuator count, names,
// IDs, bus/port assignment, per-motor limits and default gains all come from a
// RobotConfig (loaded from /robot.json — see config_loader.h). One firmware binary,
// any robot.
//
// There is no "gripper" special case and no "spine" special case: every actuator is
// uniform and addressed by index [0, count()) or by name. Bus instances live in the
// application (main.cpp); begin() resolves each actuator's 1-based bus/port index
// against the supplied tables.
//
// ── How the two transports coexist ───────────────────────────────────────────
// Storage is a std::variant per slot rather than a pointer to an abstract base:
// everything stays statically allocated (no heap on a Teensy), and the two actuator
// classes already expose identical method names, so dispatch is a two-line get_if
// rather than a vtable. The host link above this class cannot tell the difference —
// which is the whole point: the UDP API is per-JOINT, not per-transport.
//
// An actuator whose transport cannot do something reports 0 for that field rather
// than a plausible-looking approximation (see DxlActuator.h for the full mapping).
//
// Provisioning (CAN_ID / MST_ID writes, flash save, set-zero) is intentionally NOT
// exposed here; use the `teensy41_provision` sketch.

class Robot {
public:
  // Per-actuator command/state, addressed by index. The host link iterates [0,count).
  struct ActuatorCommand { float pos = 0, vel = 0, kp = 0, kd = 0, t_ff = 0; };
  struct ActuatorState   { float pos = 0, vel = 0, torque = 0;
                           uint8_t err = 0, t_mos = 0, t_rotor = 0; };

  // Construct + bring up every actuator from `cfg`.
  //   `buses`    — 1-based table of CanBus for Transport::DM_CAN actuators.
  //   `dxlBuses` — 1-based table of DxlBus for Transport::DYNAMIXEL actuators.
  // Both default to empty so an all-CAN caller (and the existing test sketches) can
  // keep using the original 3-argument form unchanged. Call AFTER the buses are up.
  // Returns false (and brings up nothing) if any bus/port index is unresolvable.
  bool begin(const RobotConfig &cfg, CanBus *const *buses, size_t numBuses,
             DxlBus *const *dxlBuses = nullptr, size_t numDxlBuses = 0);

  size_t                count()  const { return count_; }
  const RobotConfig    &config() const { return cfg_; }
  const ActuatorConfig &actuatorConfig(size_t i) const { return cfg_.actuators[i]; }
  const char           *name(size_t i) const { return cfg_.actuators[i].name; }

  // Index of the actuator named `name`, or -1 if none. Linear scan (count is small).
  int indexOf(const char *name) const;

  // Actuator i's declared ranges. For a DM-J these are the motor's own fixed-point
  // MIT scale, read from it at begin(); for a DYNAMIXEL they are 0 because no such
  // scale exists on that transport (ActuatorLimits.h explains why 0 and not a guess).
  const ActuatorLimits &limits(size_t i) const {
    if (auto *d = std::get_if<DmActuator>(&actuators_[i]))  return d->limits();
    if (auto *x = std::get_if<DxlActuator>(&actuators_[i])) return x->limits();
    return kNoLimits;
  }

  // False when the actuator never answered at begin(). Such an actuator is inert:
  // never enabled, never commanded. It still occupies its index and reports
  // offline through the existing live_mask / Health.live path.
  bool online(size_t i) const {
    return apply(i, [](const auto &a) { return a.online(); }, false);
  }

  // Whether actuator i's output stage is energized.
  //
  // This USED to be computed in host_link as `err == 1` — the DAMIAO "enabled OK"
  // nibble. That is a DM-specific encoding: a DYNAMIXEL reports Hardware Error
  // Status 0 when healthy, so the old formula would have reported a working neck
  // joint as de-energized forever — indistinguishable from the `late_join_enable`
  // fault already on the board. Each actuator therefore answers for itself.
  bool enabled(size_t i) const {
    if (auto *d = std::get_if<DmActuator>(&actuators_[i]))
      return d->lastFeedback().err == 1;
    if (auto *x = std::get_if<DxlActuator>(&actuators_[i])) return x->enabled();
    return false;
  }

  Transport transport(size_t i) const { return cfg_.actuators[i].transport; }

  // Re-query ONE offline actuator per `interval_ms` (round-robin) so an actuator that
  // was absent at boot — unpowered, unplugged, or behind a dead bus — recovers
  // without a reboot. Returns the index that just came online, else -1.
  //
  // Cost: ZERO while everything is online. While something IS offline it costs one
  // recovery probe per interval, and the two transports are NOT alike:
  //   * DM-J  — three register reads, <= ~3*timeout_ms (~15 ms at the defaults).
  //   * DYNAMIXEL — a ping, then a control-regime check. The usual absent case is one
  //     FAILED ping (~timeout_ms) and returns immediately. The worst case is a servo
  //     that answers the ping and then goes flaky mid-bring-up: ~7 blocking exchanges,
  //     order 40 ms. That is ~16 missed ticks of a 400 Hz loop, once per second, and
  //     it is borne by the ARMS as well as the neck. Reads come before writes and the
  //     retry count is deliberately small to keep that bound down; if a flaky serial
  //     actuator ever becomes a real operating condition, this recovery wants moving
  //     off the control loop rather than tuning further. That is safe here because the DM comm-loss watchdog is
  // off (motors hold their last MIT command, so a brief gap moves nothing) and the
  // host command watchdog is 150 ms.
  int  serviceOffline(uint32_t interval_ms = 1000, uint32_t timeout_ms = 5);

  // Re-energize ONE actuator per interval that is online but is not actually holding.
  //
  // This replaces host_link's robot-wide `enabled_` one-shot latch, which fired on the
  // first connect and never again — so an actuator that joined the bus later (powered
  // late, unplugged and returned, or rescued by serviceOffline) came online, streamed
  // feedback, sat in MIT mode, and was NEVER ENERGIZED. It reads as a joint that is
  // present and healthy in every telemetry field while being limp to the touch, which
  // is exactly how it was found: by feel, on l_j3/l_j4 (`late_join_enable`).
  //
  // Rate-limited and round-robin so a persistently faulted actuator cannot monopolise
  // the pass. Returns the index just repaired, else -1. Call only while a host is
  // connected — "should be energized" is the host's state to know, not the Robot's.
  int serviceEnable(uint32_t interval_ms = 250) {
    if (count_ == 0) return -1;
    uint32_t now = millis();
    if (now - last_enable_ms_ < interval_ms) return -1;
    last_enable_ms_ = now;
    for (size_t k = 0; k < count_; ++k) {
      size_t i = (enable_idx_ + k) % count_;
      if (!online(i)) continue;
      if (auto *d = std::get_if<DmActuator>(&actuators_[i])) {
        if (d->lastFeedback().err == 1) continue;       // 1 = enabled OK
        enable_idx_ = (i + 1) % count_;
        d->enable();
        return (int)i;
      }
      if (auto *x = std::get_if<DxlActuator>(&actuators_[i])) {
        if (!x->torqueMismatch()) continue;
        enable_idx_ = (i + 1) % count_;
        x->reapplyTorque();
        return (int)i;
      }
    }
    return -1;
  }

  // Give every actuator its per-loop slice. Only the DYNAMIXEL path needs one: its
  // bus is a single wire that must serialize one transaction at a time, so writes
  // and reads are SCHEDULED here rather than issued inline from sendCommand(). DM
  // actuators are fire-and-forget on a full-duplex FIFO and do nothing here.
  void service() {
    for (size_t i = 0; i < count_; ++i)
      if (auto *x = std::get_if<DxlActuator>(&actuators_[i])) x->service();
  }

  // Energize a single actuator (used when one comes online while a host is already
  // connected; otherwise it would wait for the next connect edge).
  void enable(size_t i) {
    apply(i, [](auto &a) { if (a.online()) a.enable(); return 0; }, 0);
  }

  // Hand one setpoint to actuator i.
  //
  // NOTE the deliberate semantic widening: for a DM-J this puts an MIT frame on CAN
  // immediately (which also elicits its feedback). For a DYNAMIXEL it caches the
  // setpoint, and Robot::service() -> DxlBus::poll() puts it on the wire on that
  // device's next scheduled slot. The host link calls this at the stream rate for
  // both; a single-wire bus simply cannot honour a synchronous send-now contract
  // without blocking the control loop, and staleness is already expressed honestly
  // through live_mask / Health.stale_ms.
  //
  // Dropped for an offline actuator: its scale is unknown, so any encoding would
  // be a guess.
  void sendCommand(size_t i, const ActuatorCommand &c) {
    apply(i, [&](auto &a) {
      if (a.online()) a.sendMITCommand(c.pos, c.vel, c.kp, c.kd, c.t_ff);
      return 0;
    }, 0);
  }

  // Apply actuator i's stale-command fail-safe. NOT the same command on both
  // transports, which is exactly why it is a named operation rather than a magic
  // tuple of zeros assembled in the host link: kp=0/kd=default/pos=0 damps a DM-J in
  // MIT mode, but commands a DYNAMIXEL position servo to drive to raw zero.
  void sendHold(size_t i) {
    if (auto *d = std::get_if<DmActuator>(&actuators_[i])) {
      if (d->online()) d->safeHold(cfg_.actuators[i].default_kd);
    } else if (auto *x = std::get_if<DxlActuator>(&actuators_[i])) {
      if (x->online()) x->safeHold();
    }
  }

  ActuatorState state(size_t i) const {
    return apply(i, [](const auto &a) {
      const auto &fb = a.lastFeedback();
      return ActuatorState{fb.position, fb.velocity, fb.torque,
                           fb.err, fb.t_mos, fb.t_rotor};
    }, ActuatorState{});
  }

  // Output gating for the host link: enable/disable every actuator at once.
  // disableAll() is the disconnect fail-safe — the DM comm-loss watchdog is off
  // (timeout=0), so the motors would otherwise hold their last MIT command. On the
  // DYNAMIXEL path it clears Torque Enable(64), which is the equivalent fail-safe.
  void enableAll()  { for (size_t i = 0; i < count_; ++i) enable(i); }
  void disableAll() { for (size_t i = 0; i < count_; ++i)
                        apply(i, [](auto &a) { a.disable(); return 0; }, 0); }

  // Milliseconds since actuator i last returned feedback. Drives State.live_mask.
  uint32_t feedbackAgeMs(size_t i) const {
    return millis() - apply(i, [](const auto &a) { return a.lastRxMs(); }, (uint32_t)0);
  }

private:
  using Slot = std::variant<std::monostate, DmActuator, DxlActuator>;

  // Both actuator classes expose the same method names, so one generic lambda
  // covers both alternatives. `fallback` is what an unpopulated slot returns.
  template <typename F, typename R>
  R apply(size_t i, F &&f, R fallback) {
    if (auto *d = std::get_if<DmActuator>(&actuators_[i]))  return f(*d);
    if (auto *x = std::get_if<DxlActuator>(&actuators_[i])) return f(*x);
    return fallback;
  }
  template <typename F, typename R>
  R apply(size_t i, F &&f, R fallback) const {
    if (auto *d = std::get_if<DmActuator>(&actuators_[i]))  return f(*d);
    if (auto *x = std::get_if<DxlActuator>(&actuators_[i])) return f(*x);
    return fallback;
  }

  static const ActuatorLimits kNoLimits;

  RobotConfig cfg_;
  size_t      count_ = 0;
  size_t      retry_idx_ = 0;        // round-robin cursor for serviceOffline()
  uint32_t    last_retry_ms_ = 0;
  size_t      enable_idx_ = 0;       // round-robin cursor for serviceEnable()
  uint32_t    last_enable_ms_ = 0;
  Slot        actuators_[MAX_ACTUATORS];
};
