#pragma once
#include "SwerveModule.h"
#include <FlexCAN_T4.h>

class SwerveChassis {
private:
    SwerveModule frontRight;
    SwerveModule frontLeft;
    SwerveModule backCenter;
    
    unsigned long lastHeartbeatTime;
    unsigned long lastCommandTime;
    bool isSafetyLocked;

public:
    SwerveChassis();

    void updateKinematics(float vx, float vy, float omega);
    void processIncomingCAN(const CAN_message_t& msg);
    void runWatchdogs();
};