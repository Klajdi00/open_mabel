#include <Arduino.h>
#include <FlexCAN_T4.h>

FlexCAN_T4<CAN1, RX_SIZE_256, TX_SIZE_16> Can1;

// Modes: 1=MIT, 2=Pos-Vel, 3=Vel, 4=Force-Pos
void setControlMode(uint8_t motor_id, uint32_t mode) {
    CAN_message_t msg;
    msg.id = 0x7FF;
    msg.len = 8;
    msg.buf[0] = motor_id & 0xFF;
    msg.buf[1] = (motor_id >> 8) & 0xFF;
    msg.buf[2] = 0x55; // Write Command
    msg.buf[3] = 0x0A; // CTRL_MODE Register
    
    // Data is a 32-bit uint, LSB first (Bytes 4-7)
    msg.buf[4] = mode & 0xFF;
    msg.buf[5] = (mode >> 8) & 0xFF;
    msg.buf[6] = (mode >> 16) & 0xFF;
    msg.buf[7] = (mode >> 24) & 0xFF;
    
    Can1.write(msg);
    Serial.printf("Commanded Motor %d to Mode %d\n", motor_id, mode);
    
    // Note: The motor will internally reset its command values to 0 when switched.
}

void setup() {
    Serial.begin(115200);
    Can1.begin();
    Can1.setBaudRate(1000000);
    
    delay(2000);
    setControlMode(1, 1); // Set motor 1 to MIT mode
}

void loop() {
    // Empty
}