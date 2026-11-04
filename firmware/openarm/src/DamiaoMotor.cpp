#include "DamiaoMotor.h"

DamiaoMotor::DamiaoMotor(uint32_t id, float p_max, float v_max, float t_max) 
    : can_id(id), p_max(p_max), v_max(v_max), t_max(t_max) {
    
    // Initialize state
    state.position = 0.0f;
    state.velocity = 0.0f;
    state.torque = 0.0f;
    state.t_mos = 0.0f;
    state.t_rotor = 0.0f;
    state.error_code = 0;
}

int DamiaoMotor::float_to_uint(float x, float x_min, float x_max, int bits) {
    float span = x_max - x_min;
    float offset = x_min;
    if (x > x_max) x = x_max;
    if (x < x_min) x = x_min;
    return (int) ((x - offset) * ((float)((1 << bits) - 1)) / span);
}

float DamiaoMotor::uint_to_float(int x_int, float x_min, float x_max, int bits) {
    float span = x_max - x_min;
    float offset = x_min;
    return ((float)x_int) * span / ((float)((1 << bits) - 1)) + offset;
}

void DamiaoMotor::enable(CAN_BUS_TYPE& can_bus) {
    CAN_message_t msg;
    msg.id = can_id;
    msg.len = 8;
    for(int i=0; i<7; i++) msg.buf[i] = 0xFF;
    msg.buf[7] = 0xFC;
    can_bus.write(msg);
}

void DamiaoMotor::disable(CAN_BUS_TYPE& can_bus) {
    CAN_message_t msg;
    msg.id = can_id;
    msg.len = 8;
    for(int i=0; i<7; i++) msg.buf[i] = 0xFF;
    msg.buf[7] = 0xFD;
    can_bus.write(msg);
}

void DamiaoMotor::setZeroPosition(CAN_BUS_TYPE& can_bus) {
    CAN_message_t msg;
    msg.id = can_id;
    msg.len = 8;
    for(int i=0; i<7; i++) msg.buf[i] = 0xFF;
    msg.buf[7] = 0xFE;
    can_bus.write(msg);
}

void DamiaoMotor::clearFaults(CAN_BUS_TYPE& can_bus) {
    CAN_message_t msg;
    msg.id = can_id;
    msg.len = 8;
    for(int i=0; i<7; i++) msg.buf[i] = 0xFF;
    msg.buf[7] = 0xFB;
    can_bus.write(msg);
}

void DamiaoMotor::sendMITCommand(CAN_BUS_TYPE& can_bus, float p_des, float v_des, float kp, float kd, float t_ff) {
    int p_int = float_to_uint(p_des, -p_max, p_max, 16);
    int v_int = float_to_uint(v_des, -v_max, v_max, 12);
    int kp_int = float_to_uint(kp, 0.0f, 500.0f, 12);
    int kd_int = float_to_uint(kd, 0.0f, 5.0f, 12);
    int t_int = float_to_uint(t_ff, -t_max, t_max, 12);

    CAN_message_t msg;
    msg.id = can_id;
    msg.len = 8;
    msg.buf[0] = p_int >> 8;
    msg.buf[1] = p_int & 0xFF;
    msg.buf[2] = v_int >> 4;
    msg.buf[3] = ((v_int & 0xF) << 4) | (kp_int >> 8);
    msg.buf[4] = kp_int & 0xFF;
    msg.buf[5] = kd_int >> 4;
    msg.buf[6] = ((kd_int & 0xF) << 4) | (t_int >> 8);
    msg.buf[7] = t_int & 0xFF;

    can_bus.write(msg);
}

void DamiaoMotor::parseFeedback(const CAN_message_t& msg) {
    // Extract Error Code (Upper 4 bits of Byte 1)
    state.error_code = msg.buf[1] >> 4;

    int p_int = (msg.buf[1] << 8) | msg.buf[2];
    int v_int = (msg.buf[3] << 4) | (msg.buf[4] >> 4);
    int t_int = ((msg.buf[4] & 0xF) << 8) | msg.buf[5];

    state.position = uint_to_float(p_int, -p_max, p_max, 16);
    state.velocity = uint_to_float(v_int, -v_max, v_max, 12);
    state.torque = uint_to_float(t_int, -t_max, t_max, 12);
    state.t_mos = (float)msg.buf[6];
    state.t_rotor = (float)msg.buf[7];
}