#include <Arduino.h>
#include <FlexCAN_T4.h>

FlexCAN_T4<CAN1, RX_SIZE_256, TX_SIZE_16> Can1;

void pingMotor(uint8_t target_id) {
    CAN_message_t msg;
    msg.id = 0x7FF; // Standard Read/Write ID [cite: 1156]
    msg.len = 4;
    msg.buf[0] = target_id & 0xFF;        // CANID_L [cite: 1156]
    msg.buf[1] = (target_id >> 8) & 0xFF; // CANID_H [cite: 1156]
    msg.buf[2] = 0x33;                    // Read Command [cite: 1156]
    msg.buf[3] = 0x00;                    // Register 0x00 (UV_Value) 
    Can1.write(msg);
}

void setup() {
    Serial.begin(115200);
    Can1.begin();
    Can1.setBaudRate(1000000); // 1 Mbps [cite: 1195]
    
    Serial.println("Starting CAN Sweep on Bus 1...");
    for (uint8_t id = 1; id <= 15; id++) {
        pingMotor(id);
        delay(10); // Wait for response
    }
}

void loop() {
    CAN_message_t inMsg;
    if (Can1.read(inMsg)) {
        if (inMsg.id != 0 && inMsg.buf[2] == 0x33) { // Response to Read [cite: 1159]
            uint16_t responding_id = inMsg.buf[0] | (inMsg.buf[1] << 8);
            Serial.printf("SUCCESS: Motor ID %d detected on CAN1\n", responding_id);
        }
    }
}