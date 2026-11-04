#pragma once
#include <Arduino.h>

namespace Config {
    // 1. Physical Robot Parameters
    constexpr float CHASSIS_RADIUS_M = 0.3f;
    constexpr float MAX_SPEED_MPS = 3.0f;

    // 2. Network Settings
    constexpr uint8_t MAC_ADDRESS[] = { 0xDE, 0xAD, 0xBE, 0xEF, 0xFE, 0xED };
    constexpr uint8_t IP_ADDRESS[] = { 192, 168, 1, 177 };
    constexpr unsigned int UDP_PORT = 8888;

    // 3. CAN IDs [Drive, Steer]
    constexpr uint8_t ID_FR_DRIVE = 1;
    constexpr uint8_t ID_FR_STEER = 2;
    constexpr uint8_t ID_FL_DRIVE = 3;
    constexpr uint8_t ID_FL_STEER = 4;
    constexpr uint8_t ID_BC_DRIVE = 5;
    constexpr uint8_t ID_BC_STEER = 6;

    // 4. Absolute Encoder Zero Offsets (Fractions 0.0 to 1.0)
    constexpr float OFFSET_FR = 0.000f; // Calibrate these on your physical robot
    constexpr float OFFSET_FL = 0.000f;
    constexpr float OFFSET_BC = 0.000f;

    // 5. Safety Timers
    constexpr unsigned long HEARTBEAT_INTERVAL_MS = 20;
    constexpr unsigned long UDP_TIMEOUT_MS = 100;
}