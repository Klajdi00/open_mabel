#include "Arduino.h"
#include "DxlBus.h"
#include <cstdio>
int fails = 0;
static void expect(const char* what, const std::vector<uint8_t>& got,
                   const std::vector<uint8_t>& want) {
  bool ok = got == want;
  printf("%-46s %s\n  got  ", what, ok ? "PASS" : "*** FAIL ***");
  for (auto b : got) printf("%02X ", b);
  if (!ok) { printf("\n  want "); for (auto b : want) printf("%02X ", b); fails++; }
  printf("\n");
}
int main() {
  { // PING id=1 — the packet the ROBOTIS e-manual documents verbatim.
    HardwareSerial p; DxlBus bus; bus.begin(p, 1000000);
    uint16_t model = 0; bus.ping(1, model, 0);   // times out; we only want the TX bytes
    expect("PING id=1 (e-manual reference packet)", p.sent,
           {0xFF,0xFF,0xFD,0x00,0x01,0x03,0x00,0x01,0x19,0x4E});
  }
  { // WRITE Goal Position(116) = 1000 to id 1.
    HardwareSerial p; DxlBus bus; bus.begin(p, 1000000);
    bus.registerHandler(1, nullptr, nullptr);
    uint8_t v[4] = {0xE8,0x03,0x00,0x00};
    bus.setWrite(1, 116, v, 4); bus.poll();
    expect("WRITE id=1 GoalPosition(116)=1000", p.sent,
           {0xFF,0xFF,0xFD,0x00,0x01,0x09,0x00,0x03,0x74,0x00,0xE8,0x03,0x00,0x00,0xF0,0xA9});
  }
  { // READ 21 bytes from 126 — the present-state block the actuator polls.
    HardwareSerial p; DxlBus bus; bus.begin(p, 1000000);
    bus.registerHandler(2, nullptr, nullptr);
    bus.setRead(2, 126, 21); bus.poll();
    printf("%-46s len=%zu\n  ", "READ id=2 addr=126 len=21", p.sent.size());
    for (auto b : p.sent) { printf("%02X ", b); }
    printf("\n");
    if (p.sent.size() != 14) { printf("  *** FAIL expected 14 bytes\n"); fails++; }
  }
  { // Byte stuffing: a payload containing FF FF FD must gain an extra FD, and the
    // LENGTH field must count it.
    HardwareSerial p; DxlBus bus; bus.begin(p, 1000000);
    bus.registerHandler(3, nullptr, nullptr);
    uint8_t v[4] = {0xFF,0xFF,0xFD,0x00};
    bus.setWrite(3, 116, v, 4); bus.poll();
    bool stuffed = p.sent.size() == 17 && p.sent[10]==0xFF && p.sent[11]==0xFF
                   && p.sent[12]==0xFD && p.sent[13]==0xFD;
    uint16_t len = p.sent[5] | (p.sent[6] << 8);
    printf("%-46s %s (len field=%u, total=%zu)\n  ",
           "byte-stuffing FF FF FD -> FF FF FD FD",
           (stuffed && len == 10) ? "PASS" : "*** FAIL ***", len, p.sent.size());
    for (auto b : p.sent) { printf("%02X ", b); }
    printf("\n");
    if (!(stuffed && len == 10)) fails++;
  }
  { // Round trip: feed a synthetic status packet back and check it is parsed
    // and dispatched with the right parameters.
    static int hits = 0; static uint8_t seen[4] = {0};
    HardwareSerial p; DxlBus bus; bus.begin(p, 1000000);
    bus.registerHandler(5, [](uint8_t err, uint16_t addr, const uint8_t* pr, uint8_t n, void*) {
      hits++; if (n >= 4) memcpy(seen, pr, 4); (void)err; (void)addr;
    }, nullptr);
    bus.setRead(5, 132, 4); bus.poll();          // arms the read
    // Status: FF FF FD 00 05 08 00 55 00 <4 params> CRC — CRC computed by our own
    // routine, which the PING case above already proved against the e-manual.
    std::vector<uint8_t> st = {0xFF,0xFF,0xFD,0x00,0x05,0x08,0x00,0x55,0x00,
                               0x11,0x22,0x33,0x44};
    uint16_t c = 0;
    for (size_t i = 0; i < st.size(); ++i) {
      uint16_t acc = (uint16_t)((((c >> 8) ^ st[i]) & 0xFF) << 8);
      for (int b = 0; b < 8; ++b) acc = (acc & 0x8000) ? (uint16_t)((acc<<1)^0x8005) : (uint16_t)(acc<<1);
      c = (uint16_t)((c << 8) ^ acc);
    }
    st.push_back(c & 0xFF); st.push_back(c >> 8);
    p.inbox = st;
    bus.poll();
    bool ok = hits == 1 && seen[0]==0x11 && seen[1]==0x22 && seen[2]==0x33 && seen[3]==0x44;
    printf("%-46s %s (dispatched %d, params %02X %02X %02X %02X, replies=%u)\n",
           "status packet parse + dispatch", ok ? "PASS" : "*** FAIL ***",
           hits, seen[0], seen[1], seen[2], seen[3], (unsigned)bus.replies());
    if (!ok) fails++;
  }
  { // A REAL status packet whose payload happens to contain FF FF FD.
    // Present Load = -1 (0xFFFF) and Present Velocity = -3 (0xFFFFFFFD) put FF FF FD at
    // the very front of the 21-byte present block -- both are ordinary values for a
    // lightly-loaded servo creeping backwards. The servo MUST stuff an extra FD there,
    // LENGTH counts it, and the CRC covers the stuffed form, so the frame is perfectly
    // valid. A receiver that does not de-stuff reads every later field one byte late.
    static int hits = 0; static uint8_t seen[24] = {0}; static uint8_t seen_len = 0;
    HardwareSerial p; DxlBus bus; bus.begin(p, 1000000);
    bus.registerHandler(7, [](uint8_t, uint16_t, const uint8_t* pr, uint8_t n, void*) {
      hits++; seen_len = n; if (n <= sizeof(seen)) memcpy(seen, pr, n);
    }, nullptr);
    bus.setRead(7, 126, 21); bus.poll();

    std::vector<uint8_t> payload = {
      0xFF,0xFF,             // Present Load  = -1
      0xFD,0xFF,0xFF,0xFF,   // Present Velocity = -3
      0x00,0x08,0x00,0x00,   // Present Position = 2048  (= 0 rad)
      0,0,0,0, 0,0,0,0, 0,0, 0x1B };            // traj, voltage, temp 27C
    // what the servo actually puts on the wire:
    std::vector<uint8_t> stuffed;
    { int run = 0;
      for (auto b : payload) { stuffed.push_back(b);
        if (b==0xFF) run = (run>=1)?2:1; else if (b==0xFD && run==2) { stuffed.push_back(0xFD); run=0; } else run=0; } }
    std::vector<uint8_t> st = {0xFF,0xFF,0xFD,0x00,0x07,0,0,0x55,0x00};
    st.insert(st.end(), stuffed.begin(), stuffed.end());
    uint16_t length = (uint16_t)(st.size() - 7 + 2);
    st[5] = length & 0xFF; st[6] = length >> 8;
    uint16_t c = 0;
    for (size_t i = 0; i < st.size(); ++i) {
      uint16_t acc = (uint16_t)((((c >> 8) ^ st[i]) & 0xFF) << 8);
      for (int b = 0; b < 8; ++b) acc = (acc & 0x8000) ? (uint16_t)((acc<<1)^0x8005) : (uint16_t)(acc<<1);
      c = (uint16_t)((c << 8) ^ acc); }
    st.push_back(c & 0xFF); st.push_back(c >> 8);
    printf("\nwire payload is %zu bytes (21 + %zu stuffed)\n", stuffed.size(), stuffed.size()-21);
    p.inbox = st; bus.poll();

    int32_t pos = (int32_t)((uint32_t)seen[6] | ((uint32_t)seen[7]<<8) |
                            ((uint32_t)seen[8]<<16) | ((uint32_t)seen[9]<<24));
    double deg = (pos - 2048) * 360.0 / 4096.0;
    bool ok = (hits==1) && (seen_len==21) && (pos==2048);
    printf("%-46s %s\n  dispatched=%d params=%u  decoded position = %ld pulses = %.1f deg\n",
           "stuffed status packet de-stuffs correctly", ok ? "PASS" : "*** FAIL ***",
           hits, seen_len, (long)pos, deg);
    if (!ok) { printf("  ^ a spike: the servo is at 0 deg and the parse says %.0f deg\n", deg); fails++; }
  }

  printf("\n%s\n", fails ? "SOME CHECKS FAILED" : "all framing checks passed");
  return fails ? 1 : 0;
}
