#pragma once
#include <Arduino.h>

class RevMotor {
private:
    uint8_t motorID;
    float currentVelocity;
    float currentPosition;

public:
    RevMotor(uint8_t can_id);

    uint8_t getID() const;

    // Commands
    void setVelocity(float speed);
    void setPositionFraction(float fraction);
    void setCurrent(float current_amps);
    void pingHeartbeat();

    // Telemetry
    float getVelocity() const;
    float getPosition() const;
    void parseStatusFrame(uint8_t api_index, const uint8_t* payload);
};