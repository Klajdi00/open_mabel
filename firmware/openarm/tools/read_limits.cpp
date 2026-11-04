#include <Arduino.h>
#include <FlexCAN_T4.h>

FlexCAN_T4<CAN1, RX_SIZE_256, TX_SIZE_16> Can1;

void requestRegister(uint8_t motor_id, uint8_t reg) {
    CAN_message_t msg;
    msg.id = 0x7FF;
    msg.len = 4;
    msg.buf[0] = motor_id & 0xFF;
    msg.buf[1] = (motor_id >> 8) & 0xFF;
    msg.buf[2] = 0x33; // Read Command
    msg.buf[3] = reg;
    Can1.write(msg);
}

void setup() {
    Serial.begin(115200);
    Can1.begin();
    Can1.setBaudRate(1000000);
    
    uint8_t target_motor = 1;
    
    Serial.println("Requesting Mapping Limits...");
    requestRegister(target_motor, 0x15); // PMAX
    delay(10);
    requestRegister(target_motor, 0x16); // VMAX
    delay(10);
    requestRegister(target_motor, 0x17); // TMAX
}

void loop() {
    CAN_message_t inMsg;
    if (Can1.read(inMsg)) {
        if (inMsg.id == 0x7FF && inMsg.buf[2] == 0x33) {
            uint8_t reg = inMsg.buf[3];
            
            // Reconstruct the 32-bit float from bytes 4-7
            uint32_t raw_bytes = (inMsg.buf[7] << 24) | (inMsg.buf[6] << 16) | 
                                 (inMsg.buf[5] << 8)  | inMsg.buf[4];
            float value;
            memcpy(&value, &raw_bytes, sizeof(value));
            
            if (reg == 0x15) Serial.printf("PMAX: %.2f rad\n", value);
            if (reg == 0x16) Serial.printf("VMAX: %.2f rad/s\n", value);
            if (reg == 0x17) Serial.printf("TMAX: %.2f Nm\n", value);
        }
    }
}