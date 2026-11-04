#include "SwerveModule.h"
#include <math.h>

SwerveModule::SwerveModule(uint8_t drive_id, uint8_t steer_id, float x, float y, float offset) 
    : driveMotor(drive_id), steerMotor(steer_id), posX(x), posY(y), steerOffset(offset) {}

void SwerveModule::calculateAndCommand(float vx, float vy, float omega) {
    float v_rot_x = -omega * posY;
    float v_rot_y =  omega * posX;
    
    float v_total_x = vx + v_rot_x;
    float v_total_y = vy + v_rot_y;
    
    float target_speed = hypot(v_total_x, v_total_y);
    float target_angle_rad = atan2(v_total_y, v_total_x);
    
    float target_fraction = (target_angle_rad / (2.0 * M_PI));
    if (target_fraction < 0) target_fraction += 1.0f; 

    float final_steer_cmd = fmod((target_fraction + steerOffset), 1.0f);

    steerMotor.setPositionFraction(final_steer_cmd);
    driveMotor.setVelocity(target_speed);
}

void SwerveModule::emergencyStop() {
    driveMotor.setVelocity(0.0f);
}

void SwerveModule::sendHeartbeats() {
    driveMotor.pingHeartbeat();
    steerMotor.pingHeartbeat();
}

void SwerveModule::routeCANFrame(uint8_t target_id, uint8_t api_index, const uint8_t* payload) {
    if (target_id == driveMotor.getID()) {
        driveMotor.parseStatusFrame(api_index, payload);
    } else if (target_id == steerMotor.getID()) {
        steerMotor.parseStatusFrame(api_index, payload);
    }
}