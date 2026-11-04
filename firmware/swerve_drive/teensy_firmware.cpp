#include <Arduino.h>
#include <FlexCAN_T4.h>
#include <NativeEthernet.h>
#include <NativeEthernetUdp.h>
#include <math.h>

// ============================================================================
// 1. HARDWARE & NETWORK CONFIGURATION
// ============================================================================
FlexCAN_T4<CAN1, RX_SIZE_256, TX_SIZE_16> can1;

// Ethernet Setup (Must match the Python script on Jetson Thor)
byte mac[] = { 0xDE, 0xAD, 0xBE, 0xEF, 0xFE, 0xED };
IPAddress ip(192, 168, 1, 177); 
unsigned int localPort = 8888;
EthernetUDP Udp;

// Swerve Geometry (Meters)
const float CHASSIS_RADIUS = 0.3; 
const float MAX_SPEED_MPS = 3.0; // Hard clamp for safety

// Safety Timers
const unsigned long HEARTBEAT_INTERVAL_MS = 20; 
const unsigned long UDP_TIMEOUT_MS = 100;       // Deadman switch timeout

// ============================================================================
// 2. REV CAN PROTOCOL ENGINE
// ============================================================================
namespace RevProtocol {
    // Generates the FRC 29-bit Extended ID
    uint32_t generateID(uint8_t api_class, uint8_t api_index, uint8_t can_id) {
        uint32_t device_type = 2;  // Brushless Motor
        uint32_t manufacturer = 5; // REV Robotics
        return (device_type << 24) | (manufacturer << 16) | (api_class << 10) | (api_index << 6) | can_id;
    }

    void sendFloatCommand(uint8_t can_id, uint8_t api_class, uint8_t api_index, float value) {
        CAN_message_t msg;
        msg.flags.extended = 1;
        msg.len = 8;
        msg.id = generateID(api_class, api_index, can_id);
        
        // Zero out the buffer, then copy the 4-byte float into it
        memset(msg.buf, 0, 8);
        memcpy(msg.buf, &value, sizeof(float));
        can1.write(msg);
    }

    void sendHeartbeat(uint8_t can_id) {
        CAN_message_t msg;
        msg.flags.extended = 1;
        msg.len = 8;
        msg.id = generateID(0, 2, can_id); // Class 0, Index 2 is Heartbeat
        
        memset(msg.buf, 0, 8);
        msg.buf[0] = 0x01; // 0x01 means System Enabled
        can1.write(msg);
    }
}

// ============================================================================
// 3. HARDWARE CLASSES (Encapsulation)
// ============================================================================

// Low-level representation of a single REV Motor
class RevMotor {
private:
    uint8_t id;
public:
    RevMotor(uint8_t can_id) : id(can_id) {}

    void setVelocity(float speed) {
        // Class 1 (Setpoint), Index 1 (Velocity)
        RevProtocol::sendFloatCommand(id, 1, 1, speed);
    }

    void setPositionFraction(float fraction) {
        // Class 1 (Setpoint), Index 3 (Position)
        RevProtocol::sendFloatCommand(id, 1, 3, fraction);
    }

    void pingHeartbeat() {
        RevProtocol::sendHeartbeat(id);
    }
};

// Mid-level representation of a Swerve Corner
class SwerveModule {
private:
    RevMotor driveMotor;
    RevMotor steerMotor;
    float posX, posY;
    float steerOffset; // Encoder zero offset fraction

public:
    SwerveModule(uint8_t drive_id, uint8_t steer_id, float x, float y, float offset) 
        : driveMotor(drive_id), steerMotor(steer_id), posX(x), posY(y), steerOffset(offset) {}

    // Computes specific wheel speed and angle based on global chassis velocities
    void calculateAndCommand(float vx, float vy, float omega) {
        // Kinematic cross-product for rotation
        float v_rot_x = -omega * posY;
        float v_rot_y =  omega * posX;
        
        // Combine translation and rotation vectors
        float v_total_x = vx + v_rot_x;
        float v_total_y = vy + v_rot_y;
        
        // Target speed (Hypotenuse)
        float target_speed = hypot(v_total_x, v_total_y);
        
        // Target angle in radians (-PI to PI)
        float target_angle_rad = atan2(v_total_y, v_total_x);
        
        // Convert radians to REV position fraction (0.0 to 1.0)
        float target_fraction = (target_angle_rad / (2.0 * M_PI));
        if (target_fraction < 0) target_fraction += 1.0; 

        // Apply physical hardware offset
        float final_steer_cmd = fmod((target_fraction + steerOffset), 1.0);

        // Hardware command dispatch
        steerMotor.setPositionFraction(final_steer_cmd);
        driveMotor.setVelocity(target_speed);
    }

    void emergencyStop() {
        driveMotor.setVelocity(0.0);
    }

    void sendHeartbeats() {
        driveMotor.pingHeartbeat();
        steerMotor.pingHeartbeat();
    }
};

// High-level representation of the entire robot base
class SwerveChassis {
private:
    SwerveModule frontRight;
    SwerveModule frontLeft;
    SwerveModule backCenter;
    
    unsigned long lastHeartbeatTime;
    unsigned long lastCommandTime;
    bool isSafetyLocked;

public:
    SwerveChassis() : 
        // Init Modules: Drive ID, Steer ID, X pos, Y pos, Encoder Offset
        frontRight(1, 2,  0.866 * CHASSIS_RADIUS,  0.5 * CHASSIS_RADIUS, 0.0),
        frontLeft (3, 4, -0.866 * CHASSIS_RADIUS,  0.5 * CHASSIS_RADIUS, 0.0),
        backCenter(5, 6,  0.0,                    -1.0 * CHASSIS_RADIUS, 0.0),
        lastHeartbeatTime(0), lastCommandTime(0), isSafetyLocked(true) {}

    void updateKinematics(float vx, float vy, float omega) {
        // Clamp velocities for absolute safety
        if (vx > MAX_SPEED_MPS) vx = MAX_SPEED_MPS;
        if (vx < -MAX_SPEED_MPS) vx = -MAX_SPEED_MPS;
        if (vy > MAX_SPEED_MPS) vy = MAX_SPEED_MPS;
        if (vy < -MAX_SPEED_MPS) vy = -MAX_SPEED_MPS;

        frontRight.calculateAndCommand(vx, vy, omega);
        frontLeft.calculateAndCommand(vx, vy, omega);
        backCenter.calculateAndCommand(vx, vy, omega);
        
        lastCommandTime = millis();
        isSafetyLocked = false;
    }

    // Must be called in the main loop every cycle
    void runWatchdogs() {
        unsigned long current_time = millis();

        // 1. Hardware Watchdog: Send 20ms heartbeat to REV controllers
        if (current_time - lastHeartbeatTime >= HEARTBEAT_INTERVAL_MS) {
            frontRight.sendHeartbeats();
            frontLeft.sendHeartbeats();
            backCenter.sendHeartbeats();
            lastHeartbeatTime = current_time;
        }

        // 2. Network Watchdog: Stop robot if Jetson UDP goes silent
        if (!isSafetyLocked && (current_time - lastCommandTime > UDP_TIMEOUT_MS)) {
            frontRight.emergencyStop();
            frontLeft.emergencyStop();
            backCenter.emergencyStop();
            isSafetyLocked = true;
            Serial.println("SAFETY LOCKOUT: Jetson Network Timeout.");
        }
    }
};

// ============================================================================
// 4. MAIN EXECUTION
// ============================================================================

SwerveChassis mabelBase;

void setup() {
    Serial.begin(115200);

    // Initialize CAN Bus
    can1.begin();
    can1.setBaudRate(1000000);

    // Initialize Ethernet
    Ethernet.begin(mac, ip);
    Udp.begin(localPort);

    Serial.println("Mabel Swerve Node Online. Waiting for Jetson Thor...");
}

void loop() {
    // 1. Process Network Commands
    int packetSize = Udp.parsePacket();
    if (packetSize == 12) { // Expecting three 32-bit floats (12 bytes)
        byte packetBuffer[12];
        Udp.read(packetBuffer, 12);
        
        float vx, vy, omega;
        memcpy(&vx, &packetBuffer[0], 4);
        memcpy(&vy, &packetBuffer[4], 4);
        memcpy(&omega, &packetBuffer[8], 4);
        
        // Pass data to chassis object
        mabelBase.updateKinematics(vx, vy, omega);
    }

    // 2. Service the Safety Timers (Non-blocking)
    mabelBase.runWatchdogs();
}