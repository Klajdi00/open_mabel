#ifndef ARMLINK_PROTOCOL_H
#define ARMLINK_PROTOCOL_H

// ArmLink — UDP wire contract between the host bridge and the arms Teensy 4.1.
// The host client was removed with the arms refactor (this firmware is unflashed);
// the full contract (framing, watchdog, safety) is hardware_bridge/docs/ARMLINK.md.
//
// One UDP datagram per message: byte 0 = tag, then a little-endian packed struct.
// Teensy = static server 192.168.125.10:5800 on the arms QSFP28 lane; it learns
// the host endpoint from inbound packets (host binds an ephemeral port).

#include <stdint.h>

namespace armlink {

constexpr uint8_t TAG_COMMAND = 0x01;   // host -> teensy
constexpr uint8_t TAG_HELLO   = 0x03;   // host -> teensy (keepalive / stream cfg)
constexpr uint8_t TAG_STATE   = 0x10;   // teensy -> host (streamed)

constexpr uint8_t FLAG_TORQUE = 0x01;   // Command.flags / State.flags bit0

constexpr uint16_t SERVER_PORT       = 5800;
constexpr uint16_t DEFAULT_STREAM_HZ = 100;

// Watchdogs (see ARMLINK.md): a stale COMMAND while torque is on means HOLD the
// measured position (never limp — arms must not fall on a dead commander); a
// fully silent host stops the STATE stream but keeps holding.
constexpr uint32_t CMD_WATCHDOG_MS  = 150;
constexpr uint32_t CONN_WATCHDOG_MS = 1000;

#pragma pack(push, 1)

// HELLO body
struct Hello {
    uint16_t stream_hz;          // 0 = keep current / default
};

// COMMAND body: header then count x CommandMotor
struct CommandHead {
    uint32_t seq;                // monotonically increasing; stale seq is dropped
    uint8_t  flags;              // FLAG_TORQUE: output enable (0 = free-drive/limp)
    uint8_t  count;
};
struct CommandMotor {
    uint8_t bus;                 // 1 = CAN1 (left arm), 2 = CAN2 (right arm)
    uint8_t id;                  // DaMiao motor id on that bus
    float   pos;                 // MIT position target, MOTOR frame [rad]
    float   kp;                  // MIT impedance gains
    float   kd;
};

// STATE body: header then count x StateMotor (every motor in the firmware table)
struct StateHead {
    uint32_t seq;                // echo of the last APPLIED Command.seq
    uint32_t t_us;               // micros() at send
    uint8_t  flags;              // FLAG_TORQUE: current latched output state
    uint8_t  count;
};
struct StateMotor {
    uint8_t bus;
    uint8_t id;
    float   pos;                 // MIT feedback, MOTOR frame [rad]
    float   vel;                 // [rad/s]
    float   tau;                 // [N·m]
    uint8_t t_mos;               // MOSFET temp [degC]   (DaMiao feedback d[6])
    uint8_t t_rotor;             // rotor temp [degC]    (DaMiao feedback d[7])
    uint8_t err;                 // DaMiao error/status nibble (0 = ok)
};

#pragma pack(pop)

}  // namespace armlink

#endif  // ARMLINK_PROTOCOL_H
