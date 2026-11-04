// Pseudocode structure for your CLI
void processSerialCommands() {
    if (Serial.available()) {
        String cmd = Serial.readStringUntil('\n');
        // Example syntax expected: "M 1 0.5" -> Move Motor 1 to 0.5 rad
        
        if (cmd.startsWith("M")) {
            int motor_id = cmd.substring(2, 3).toInt();
            float target_pos = cmd.substring(4).toFloat();
            
            // Assume DamiaoMotor class is instantiated
            motors[motor_id].sendMITCommand(Can1, target_pos, 0.0, 50.0, 2.0, 0.0);
            Serial.printf("Commanded Motor %d to %.2f rad\n", motor_id, target_pos);
        }
        else if (cmd.startsWith("E")) {
            int motor_id = cmd.substring(2, 3).toInt();
            motors[motor_id].enable(Can1);
            Serial.printf("Enabled Motor %d\n", motor_id);
        }
    }
}