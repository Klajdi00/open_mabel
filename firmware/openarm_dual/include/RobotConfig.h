#pragma once
#include <Arduino.h>
#include "ActuatorLimits.h"   // the transport-neutral ranges block

// RobotConfig — the runtime description of a robot, loaded from /robot.json on the
// SD card (see config_loader.h). This is what makes one firmware binary adapt to any
// robot: a collection of actuators on CAN buses plus the DAQ/network settings that
// aggregate them onto the UDP host link.
//
// Pure data, no SD/JSON dependency — the generic Robot class includes this for the
// struct definitions; only main.cpp (via config_loader.h) actually parses the file.
//
// Capacity is compile-time (MAX_ACTUATORS); the populated count is runtime. Bus is
// stored as a 1-based INDEX, not a pointer: which FlexCAN peripheral a bus maps to is
// hardware-fixed (FlexCAN is a template parameter, see main.cpp), so the application
// owns a small CanBus* table that the Robot resolves the index against at begin().

// Compile-time capacities. These are DELIBERATELY generous: the whole point of the
// config-driven design is that adding an actuator is a /robot.json edit, and a cap that
// needs a reflash to move defeats it. Sizing rationale — the fully-populated upper body
// is 22 (16 arm + torso + 3 neck + 2 lift), so 32 leaves room to grow without another
// firmware change. Costs are trivial on a Teensy 4.1 (~1 MB RAM).
//
// THESE FOUR MOVE TOGETHER — a mismatch is a silent bug, so each pairing is enforced by
// a static_assert rather than a comment (see host_link.h and Robot.cpp):
//   MAX_ACTUATORS  <->  proto/robot.options  max_count
//   MAX_ACTUATORS  <=   CanBus::kMaxHandlers        (one bus may host every actuator)
//   robot_RobotInfo_size  <  RobotHostLink::tx_     (the largest packet must fit)
//   MAX_NAME/MODEL_LEN <-> robot.options max_size   (ActuatorInfo.name / .model)
static constexpr size_t   MAX_ACTUATORS = 32;   // static footprint cap
static constexpr uint8_t  MAX_BUSES     = 3;    // HARDWARE-FIXED: Teensy 4.1 has exactly
                                                // CAN1/CAN2/CAN3. Not a soft cap.
static constexpr uint8_t  MAX_DXL_PORTS = 2;   // half-duplex UARTs wired for DYNAMIXEL.
                                                // Soft cap, unlike MAX_BUSES: the Teensy
                                                // 4.1 has 8 LPUARTs, so raising this is a
                                                // matter of adding a port to the table in
                                                // main.cpp, not of new silicon.
static constexpr size_t   MAX_NAME_LEN  = 24;   // actuator name buffer (incl. NUL)
static constexpr size_t   MAX_MODEL_LEN = 24;   // DAMIAO model string buffer

// Which wire an actuator lives on. Absent from robot.json => DM_CAN, so every
// pre-existing config file keeps working byte-for-byte: this extension is additive
// on the SD card as well as on the UDP API.
enum class Transport : uint8_t {
  DM_CAN    = 0,   // DAMIAO DM-J, MIT mode, FlexCAN  (bus = 1-based CanBus index)
  DYNAMIXEL = 1,   // ROBOTIS Protocol 2.0, half-duplex UART (port = 1-based DxlBus index)
};

struct ActuatorConfig {
  char     name[MAX_NAME_LEN]  = {0};  // e.g. "j1", "gripper" — must be unique
  uint8_t  bus                 = 0;    // 1-based bus index into the app's CanBus table
  uint32_t can_id              = 0;    // motor command ID
  uint32_t mst_id              = 0;    // motor feedback / master ID (must be unique per bus)
  char     model[MAX_MODEL_LEN] = {0}; // documentation only

  // ---- transport ------------------------------------------------------------
  // `bus`/`can_id`/`mst_id` above are the DM_CAN wiring. A DYNAMIXEL actuator uses
  // `port`/`dxl_id` instead — deliberately SEPARATE keys rather than reusing `bus`,
  // because MAX_BUSES is documented as hardware-fixed to the Teensy's three FlexCAN
  // peripherals and config_loader validates `bus` against it. Overloading that field
  // for a UART index would make both the constant and its validation a lie.
  Transport transport = Transport::DM_CAN;
  uint8_t   port      = 0;    // 1-based index into the app's DxlBus table
  uint8_t   dxl_id    = 0;    // DYNAMIXEL ID (0..252); each axis of a 2XL430 is its own
  // Pass commanded kp/kd through as RAW Position P/D Gain register units (0..16383)
  // instead of ignoring them. Off by default: MIT gains and DYNAMIXEL gain registers
  // are different quantities that share a name, so a MIT kp=20 arriving as P gain 20
  // would leave the joint nearly limp. See DxlActuator.h.
  bool      raw_gains = false;
  // NOTE: no limits here. The MIT fixed-point ranges live in the motor (PMAX/VMAX/
  // TMAX registers) and are read at Robot::begin(); kp_max/kd_max are protocol
  // constants and live in DmActuator::Limits' defaults.
  float    default_kp          = 0.0f; // damping-hold gains used when no fresh host command
  float    default_kd          = 1.0f;
};

struct NetConfig {
  uint8_t  ip[4]      = {0, 0, 0, 0};
  uint8_t  netmask[4] = {255, 255, 255, 0};
  uint8_t  gateway[4] = {0, 0, 0, 0};
  uint16_t port       = 5800;
};

struct RobotConfig {
  char       name[32]          = {0};
  // Per-DYNAMIXEL-port line rate, 1-based to match `port` (index 0 unused).
  // 0 = AUTO-DETECT: the firmware sweeps the standard rates at bring-up until the
  // configured IDs answer. That is deliberate — the servo's baud is a property of
  // the servo, and a number compiled into the firmware would make swapping one in
  // an EEPROM-write operation on hardware rather than a config edit.
  uint32_t   dxl_baud[MAX_DXL_PORTS + 1] = {0};
  NetConfig  net;
  uint16_t   stream_default_hz = 1000;
  uint16_t   stream_max_hz     = 1000;
  uint32_t   conn_watchdog_ms  = 250;
  uint32_t   cmd_watchdog_ms   = 150;
  ActuatorConfig actuators[MAX_ACTUATORS];
  size_t     count             = 0;    // number of populated actuators

  bool empty() const { return count == 0; }
};

// Outcome of a config load. On failure `error` carries a human-readable reason for
// the serial log; the caller must treat this as fatal (refuse to enable any motor).
struct ConfigResult {
  bool ok = false;
  char error[96] = {0};
};
