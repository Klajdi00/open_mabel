#pragma once
#include <Arduino.h>
#include <FlexCAN_T4.h>

extern FlexCAN_T4<CAN1, RX_SIZE_256, TX_SIZE_16> can1;

namespace RevProtocol {
    uint32_t generateID(uint8_t api_class, uint8_t api_index, uint8_t can_id);
    void sendFloatCommand(uint8_t can_id, uint8_t api_class, uint8_t api_index, float value);
    void sendHeartbeat(uint8_t can_id);
}