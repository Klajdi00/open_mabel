#ifndef DAMIAO_MOTOR_H
#define DAMIAO_MOTOR_H

#include <Arduino.h>
#include <FlexCAN_T4.h>
#include "damiao_protocol.h"

// Define a template for the CAN bus type to pass it safely
typedef FlexCAN_T4<CAN1, RX_SIZE_256, TX_SIZE_16> CAN_BUS_TYPE;

class DamiaoMotor {
public:
    DamiaoMotor(uint32_t id, float p_max, float v_max, float t_max);

    // Current state of the motor
    MotorState state;

    // Control Commands
    void enable(CAN_BUS_TYPE& can_bus);
    void disable(CAN_BUS_TYPE& can_bus);
    void setZeroPosition(CAN_BUS_TYPE& can_bus);
    void clearFaults(CAN_BUS_TYPE& can_bus);
    
    // Core MIT Mode Command
    void sendMITCommand(CAN_BUS_TYPE& can_bus, float p_des, float v_des, float kp, float kd, float t_ff);
    
    // Feedback parser
    void parseFeedback(const CAN_message_t& msg);

    // Getters
    uint32_t getID() const { return can_id; }

private:
    uint32_t can_id;
    float p_max, v_max, t_max;

    // Damiao specific float/uint packing math
    int float_to_uint(float x, float x_min, float x_max, int bits);
    float uint_to_float(int x_int, float x_min, float x_max, int bits);
};

#endif // DAMIAO_MOTOR_H