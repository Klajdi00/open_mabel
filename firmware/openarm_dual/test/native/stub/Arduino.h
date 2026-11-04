// Minimal Arduino shim so DxlBus.h can be compiled and exercised natively.
#pragma once
#include <cstdint>
#include <cstring>
#include <cstddef>
#include <vector>
static const int SERIAL_8N1_HALF_DUPLEX = 0x200;
inline uint32_t millis() { return 0; }
inline uint32_t micros() { return 0; }
inline void delayMicroseconds(uint32_t) {}
inline void delay(uint32_t) {}
struct HardwareSerial {
  std::vector<uint8_t> sent;
  std::vector<uint8_t> inbox;
  void begin(uint32_t, int) {}
  int  availableForWrite() { return 1024; }
  int  available() { return (int)inbox.size(); }
  int  read() { if (inbox.empty()) return -1; int c = inbox.front(); inbox.erase(inbox.begin()); return c; }
  void write(const uint8_t* b, size_t n) { sent.insert(sent.end(), b, b + n); }
  void flush() {}
};
