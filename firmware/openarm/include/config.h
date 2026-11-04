#ifndef CONFIG_H
#define CONFIG_H

#include <Arduino.h>

namespace MABEL_CONFIG {
    // CAN Bus Baud Rate
    constexpr uint32_t CAN_BAUD_RATE = 1000000; // 1 Mbps

    // Motor Limit Profiles (Verify these in the Damiao Desktop App!)
    // DM-J10422P (Torso)
    constexpr float J10422_PMAX = 12.566f;
    constexpr float J10422_VMAX = 12.5f;   // ~120 rpm
    constexpr float J10422_TMAX = 400.0f;  // Peak torque

    // DM-J8009P (Shoulders)
    constexpr float J8009_PMAX = 12.566f;
    constexpr float J8009_VMAX = 33.5f;    // ~320 rpm
    constexpr float J8009_TMAX = 40.0f;    // Peak torque

    // DM-J4340 (Elbows)
    constexpr float J4340_PMAX = 12.566f;
    constexpr float J4340_VMAX = 10.4f;    // ~100 rpm
    constexpr float J4340_TMAX = 27.0f;    // Peak torque

    // DM-J3507 (Wrists/Neck)
    constexpr float J3507_PMAX = 12.566f;
    constexpr float J3507_VMAX = 48.0f;    // ~460 rpm
    constexpr float J3507_TMAX = 3.0f;     // Peak torque
}

#endif // CONFIG_H