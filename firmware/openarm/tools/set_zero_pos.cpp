void setZeroPosition(uint8_t target_id) {
    CAN_message_t msg;
    msg.id = target_id; // Frame ID is the motor's CAN ID 
    msg.len = 8;
    for(int i=0; i<7; i++) msg.buf[i] = 0xFF; // Bytes 0-6 are 0xFF [cite: 1149]
    msg.buf[7] = 0xFE; // Set Zero Command [cite: 1149]
    Can1.write(msg);
    Serial.printf("Zero position set for Motor %d\n", target_id);
}