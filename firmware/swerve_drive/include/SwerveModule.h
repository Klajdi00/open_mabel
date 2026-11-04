#pragma once
#include "RevMotor.h"

class SwerveModule {
private:
    RevMotor driveMotor;
    RevMotor steerMotor;
    float posX;
    float posY;
    float steerOffset;

public:
    SwerveModule(uint8_t drive_id, uint8_t steer_id, float x, float y, float offset);

    void calculateAndCommand(float vx, float vy, float omega);
    void emergencyStop();
    void sendHeartbeats();
    void routeCANFrame(uint8_t target_id, uint8_t api_index, const uint8_t* payload);
};