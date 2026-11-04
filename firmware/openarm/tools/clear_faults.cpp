void clearMotorFaults(uint8_t target_id) {
    CAN_message_t msg;
    msg.id = target_id; 
    msg.len = 8;
    for(int i=0; i<7; i++) msg.buf[i] = 0xFF; // Bytes 0-6 are 0xFF [cite: 1154]
    msg.buf[7] = 0xFB; // Clear Fault Command [cite: 1154]
    Can1.write(msg);
    Serial.printf("Faults cleared for Motor %d\n", target_id);
}