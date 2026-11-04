#include "RevProtocol.h"

namespace RevProtocol {
    uint32_t generateID(uint8_t api_class, uint8_t api_index, uint8_t can_id) {
        uint32_t device_type = 2;  // FRC Brushless Motor
        uint32_t manufacturer = 5; // REV Robotics
        return (device_type << 24) | (manufacturer << 16) | (api_class << 10) | (api_index << 6) | can_id;
    }

    void sendFloatCommand(uint8_t can_id, uint8_t api_class, uint8_t api_index, float value) {
        CAN_message_t msg;
        msg.flags.extended = 1;
        msg.len = 8;
        msg.id = generateID(api_class, api_index, can_id);
        
        memset(msg.buf, 0, 8);
        memcpy(msg.buf, &value, sizeof(float));
        can1.write(msg);
    }

    void sendHeartbeat(uint8_t can_id) {
        CAN_message_t msg;
        msg.flags.extended = 1;
        msg.len = 8;
        msg.id = generateID(0, 2, can_id); 
        
        memset(msg.buf, 0, 8);
        msg.buf[0] = 0x01; // Enable Flag
        can1.write(msg);
    }
}