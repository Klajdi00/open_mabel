#ifndef DAMIAO_PROTOCOL_H
#define DAMIAO_PROTOCOL_H

#include <stdint.h>

// Represents the unpacked physical state of the motor
struct MotorState {
    float position;  // rad
    float velocity;  // rad/s
    float torque;    // Nm
    float t_mos;     // Celsius
    float t_rotor;   // Celsius
    uint8_t error_code; 
};

#endif // DAMIAO_PROTOCOL_H