#include <Arduino.h>
#include <FlexCAN_T4.h>
#include <NativeEthernet.h>
#include <NativeEthernetUdp.h>
#include "Config.h"
#include "SwerveChassis.h"

// Instantiate Global Hardware Interfaces
FlexCAN_T4<CAN1, RX_SIZE_256, TX_SIZE_16> can1; 
EthernetUDP Udp;

// Instantiate the Robot Base
SwerveChassis mabelBase;

void setup() {
    Serial.begin(115200);

    // Initialize CAN Bus
    can1.begin();
    can1.setBaudRate(1000000);

    // Initialize Ethernet
    IPAddress ip(Config::IP_ADDRESS[0], Config::IP_ADDRESS[1], Config::IP_ADDRESS[2], Config::IP_ADDRESS[3]);
    Ethernet.begin(Config::MAC_ADDRESS, ip);
    Udp.begin(Config::UDP_PORT);

    Serial.println("Mabel Swerve Node Online. Waiting for Jetson Thor...");
}

void loop() {
    // 1. Process Incoming CAN Telemetry (Non-Blocking)
    CAN_message_t incomingMsg;
    while (can1.read(incomingMsg)) {
        mabelBase.processIncomingCAN(incomingMsg);
    }

    // 2. Process Network Commands from Jetson Thor
    int packetSize = Udp.parsePacket();
    if (packetSize == 12) {
        byte buffer[12];
        Udp.read(buffer, 12);
        
        float vx, vy, omega;
        memcpy(&vx, &buffer[0], 4);
        memcpy(&vy, &buffer[4], 4);
        memcpy(&omega, &buffer[8], 4);
        
        mabelBase.updateKinematics(vx, vy, omega);
    }

    // 3. Process Safety Timers
    mabelBase.runWatchdogs();
}