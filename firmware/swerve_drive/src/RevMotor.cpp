#include "RevMotor.h"
#include "RevProtocol.h"

RevMotor::RevMotor(uint8_t can_id) : motorID(can_id), currentVelocity(0.0f), currentPosition(0.0f) {}

uint8_t RevMotor::getID() const { return motorID; }

void RevMotor::setVelocity(float speed) {
    RevProtocol::sendFloatCommand(motorID, 1, 1, speed);
}

void RevMotor::setPositionFraction(float fraction) {
    if (fraction > 1.0f) fraction -= 1.0f;
    if (fraction < 0.0f) fraction += 1.0f;
    RevProtocol::sendFloatCommand(motorID, 1, 3, fraction);
}

void RevMotor::setCurrent(float current_amps) {
    RevProtocol::sendFloatCommand(motorID, 1, 4, current_amps);
}

void RevMotor::pingHeartbeat() {
    RevProtocol::sendHeartbeat(motorID);
}

float RevMotor::getVelocity() const { return currentVelocity; }
float RevMotor::getPosition() const { return currentPosition; }

void RevMotor::parseStatusFrame(uint8_t api_index, const uint8_t* payload) {
    if (api_index == 2) {
        memcpy(&currentPosition, &payload[0], sizeof(float));
        memcpy(&currentVelocity, &payload[4], sizeof(float));
    }
}