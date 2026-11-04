#include <Arduino.h>
#include <FlexCAN_T4.h>
#include "DamiaoMotor.h"

FlexCAN_T4<CAN1, RX_SIZE_256, TX_SIZE_16> Can1;
DamiaoMotor testMotor(1, 12.566f, 30.0f, 40.0f); // Adjust limits to match your motor!

// Sine wave parameters
const float AMPLITUDE = 1.5f; // Radians
const float FREQUENCY = 0.5f; // Hz
const float KP = 30.0f;
const float KD = 1.5f;

void setup() {
    Serial.begin(115200);
    Can1.begin();
    Can1.setBaudRate(1000000);
    
    delay(1000);
    testMotor.enable(Can1);
    Serial.println("Motor Enabled. Starting Sine Sweep...");
}

void loop() {
    // 1. Calculate trajectory
    float time_sec = millis() / 1000.0f;
    float target_pos = AMPLITUDE * sin(2 * PI * FREQUENCY * time_sec);
    float target_vel = AMPLITUDE * 2 * PI * FREQUENCY * cos(2 * PI * FREQUENCY * time_sec);

    // 2. Send Command
    testMotor.sendMITCommand(Can1, target_pos, target_vel, KP, KD, 0.0f);
    
    // 3. Read Feedback (Non-blocking)
    CAN_message_t inMsg;
    if (Can1.read(inMsg) && inMsg.id == 0) {
        testMotor.parseFeedback(inMsg);
        // Print for Serial Plotter visualization
        Serial.printf("Target:%.2f,Actual:%.2f\n", target_pos, testMotor.state.position);
    }
    
    delay(2); // ~500Hz control loop
}