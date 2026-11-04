void changeMotorID(uint8_t current_id, uint8_t new_id) {
    CAN_message_t msg;
    
    // 1. Write new ID to register 0x08 [cite: 1210, 1165]
    msg.id = 0x7FF;
    msg.len = 8;
    msg.buf[0] = current_id & 0xFF;
    msg.buf[1] = (current_id >> 8) & 0xFF;
    msg.buf[2] = 0x55; // Write Command [cite: 1165]
    msg.buf[3] = 0x08; // ESC_ID Register 
    msg.buf[4] = new_id & 0xFF;        // Data LSB
    msg.buf[5] = (new_id >> 8) & 0xFF; // Data ...
    msg.buf[6] = 0x00;
    msg.buf[7] = 0x00;
    Can1.write(msg);
    delay(50);
    
    // 2. Save Parameters to Flash [cite: 1172]
    msg.len = 4;
    msg.buf[0] = current_id & 0xFF;
    msg.buf[1] = (current_id >> 8) & 0xFF;
    msg.buf[2] = 0xAA; // Save Command [cite: 1172]
    msg.buf[3] = 0x01; // Save Arg [cite: 1172]
    Can1.write(msg);
    
    Serial.printf("Motor ID changed from %d to %d and saved to flash.\n", current_id, new_id);
}