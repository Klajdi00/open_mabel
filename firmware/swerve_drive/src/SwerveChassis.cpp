#include "SwerveChassis.h"
#include "Config.h"
#include <Arduino.h>

SwerveChassis::SwerveChassis() : 
    frontRight(Config::ID_FR_DRIVE, Config::ID_FR_STEER,  0.866f * Config::CHASSIS_RADIUS_M,  0.5f * Config::CHASSIS_RADIUS_M, Config::OFFSET_FR),
    frontLeft (Config::ID_FL_DRIVE, Config::ID_FL_STEER, -0.866f * Config::CHASSIS_RADIUS_M,  0.5f * Config::CHASSIS_RADIUS_M, Config::OFFSET_FL),
    backCenter(Config::ID_BC_DRIVE, Config::ID_BC_STEER,  0.0f,                              -1.0f * Config::CHASSIS_RADIUS_M, Config::OFFSET_BC),
    lastHeartbeatTime(0), lastCommandTime(0), isSafetyLocked(true) 
{}

void SwerveChassis::updateKinematics(float vx, float vy, float omega) {
    if (vx > Config::MAX_SPEED_MPS) vx = Config::MAX_SPEED_MPS;
    if (vx < -Config::MAX_SPEED_MPS) vx = -Config::MAX_SPEED_MPS;
    if (vy > Config::MAX_SPEED_MPS) vy = Config::MAX_SPEED_MPS;
    if (vy < -Config::MAX_SPEED_MPS) vy = -Config::MAX_SPEED_MPS;

    frontRight.calculateAndCommand(vx, vy, omega);
    frontLeft.calculateAndCommand(vx, vy, omega);
    backCenter.calculateAndCommand(vx, vy, omega);
    
    lastCommandTime = millis();
    isSafetyLocked = false;
}

void SwerveChassis::processIncomingCAN(const CAN_message_t& msg) {
    if (!msg.flags.extended) return;

    uint8_t device_id = msg.id & 0x3F;             
    uint8_t api_index = (msg.id >> 6) & 0x0F;      
    uint8_t api_class = (msg.id >> 10) & 0x3F;     
    uint8_t manufacturer = (msg.id >> 16) & 0xFF;  

    // API Class 8 is Periodic Status
    if (manufacturer != 5 || api_class != 8) return;

    if (device_id == Config::ID_FR_DRIVE || device_id == Config::ID_FR_STEER) {
        frontRight.routeCANFrame(device_id, api_index, msg.buf);
    } else if (device_id == Config::ID_FL_DRIVE || device_id == Config::ID_FL_STEER) {
        frontLeft.routeCANFrame(device_id, api_index, msg.buf);
    } else if (device_id == Config::ID_BC_DRIVE || device_id == Config::ID_BC_STEER) {
        backCenter.routeCANFrame(device_id, api_index, msg.buf);
    }
}

void SwerveChassis::runWatchdogs() {
    unsigned long current_time = millis();

    if (current_time - lastHeartbeatTime >= Config::HEARTBEAT_INTERVAL_MS) {
        frontRight.sendHeartbeats();
        frontLeft.sendHeartbeats();
        backCenter.sendHeartbeats();
        lastHeartbeatTime = current_time;
    }

    if (!isSafetyLocked && (current_time - lastCommandTime > Config::UDP_TIMEOUT_MS)) {
        frontRight.emergencyStop();
        frontLeft.emergencyStop();
        backCenter.emergencyStop();
        isSafetyLocked = true;
        Serial.println("SAFETY LOCKOUT: Jetson Network Timeout.");
    }
}