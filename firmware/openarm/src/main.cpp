#include <Arduino.h>
#include <FlexCAN_T4.h>
#include "config.h"
#include "DamiaoMotor.h"

// Instantiate CAN Buses for Left Arm, Right Arm, and Torso/Neck
FlexCAN_T4<CAN1, RX_SIZE_256, TX_SIZE_16> LeftArmBus;
FlexCAN_T4<CAN2, RX_SIZE_256, TX_SIZE_16> RightArmBus;
FlexCAN_T4<CAN3, RX_SIZE_256, TX_SIZE_16> TorsoBus;

// Instantiate Motors (Example: Left Arm Setup)
// Passing ID and the limits defined in config.h
DamiaoMotor leftShoulder1(1, MABEL_CONFIG::J8009_PMAX, MABEL_CONFIG::J8009_VMAX, MABEL_CONFIG::J8009_TMAX);
DamiaoMotor leftShoulder2(2, MABEL_CONFIG::J8009_PMAX, MABEL_CONFIG::J8009_VMAX, MABEL_CONFIG::J8009_TMAX);
DamiaoMotor leftElbow(3, MABEL_CONFIG::J4340_PMAX, MABEL_CONFIG::J4340_VMAX, MABEL_CONFIG::J4340_TMAX);
DamiaoMotor torso(1, MABEL_CONFIG::J10422_PMAX, MABEL_CONFIG::J10422_VMAX, MABEL_CONFIG::J10422_TMAX);

// Timing variables for the control loop
elapsedMillis printTimer;
elapsedMicros loopTimer;
const unsigned int LOOP_TIME_US = 2000; // 500Hz control loop

void setup() {
    Serial.begin(115200);
    while (!Serial && millis() < 3000); // Wait for Serial to initialize

    Serial.println("MABEL Mechatronics Initializing...");

    // 1. Initialize Buses
    LeftArmBus.begin();
    LeftArmBus.setBaudRate(MABEL_CONFIG::CAN_BAUD_RATE);
    
    RightArmBus.begin();
    RightArmBus.setBaudRate(MABEL_CONFIG::CAN_BAUD_RATE);
    
    TorsoBus.begin();
    TorsoBus.setBaudRate(MABEL_CONFIG::CAN_BAUD_RATE);

    delay(1000); // Let buses settle

    // 2. Enable Motors
    leftShoulder1.enable(LeftArmBus);
    torso.enable(TorsoBus);
    
    Serial.println("System Active.");
}

void processIncomingCAN() {
    CAN_message_t msg;
    
    // Read Left Arm Bus
    while (LeftArmBus.read(msg)) {
        // Assuming Master ID is 0 for feedback
        // The actual motor ID is returned in the lower 4 bits of buf[1] and buf[0] based on the manual, 
        // but typically you map it via the frame. Here we assume you track who sent what.
        if (msg.id == 0) { 
            // In a real system, you'd match the ID to the object. 
            // For now, we update the object directly.
            leftShoulder1.parseFeedback(msg);
        }
    }

    // Read Torso Bus
    while (TorsoBus.read(msg)) {
        if (msg.id == 0) {
            torso.parseFeedback(msg);
        }
    }
}

void loop() {
    // 1. Unpack all incoming CAN hardware frames
    processIncomingCAN();

    // 2. High-Frequency Control Loop (Executes exactly every 2ms)
    if (loopTimer >= LOOP_TIME_US) {
        loopTimer -= LOOP_TIME_US;

        // Apply Impedance Control (Spring-like behavior holding zero position)
        // Send: Target Pos, Target Vel, Kp, Kd, Feedforward Torque
        leftShoulder1.sendMITCommand(LeftArmBus, 0.0f, 0.0f, 40.0f, 1.5f, 0.0f);
        torso.sendMITCommand(TorsoBus, 0.0f, 0.0f, 100.0f, 5.0f, 0.0f); // Higher gains for massive torso
    }

    // 3. Low-Frequency Serial Debugging (10Hz)
    if (printTimer >= 100) {
        printTimer = 0;
        Serial.printf("Torso  -> Pos: %5.2f | Vel: %5.2f | Torq: %5.2f | Temp: %2.0fC | Err: %X\n",
                      torso.state.position, torso.state.velocity, torso.state.torque, torso.state.t_mos, torso.state.error_code);
    }
}