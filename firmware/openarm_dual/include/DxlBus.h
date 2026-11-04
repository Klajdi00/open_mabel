#pragma once
#include <Arduino.h>

// DxlBus — one half-duplex DYNAMIXEL Protocol 2.0 bus on a Teensy hardware UART.
//
// The serial-side counterpart of CanBus, and deliberately the same shape: the
// application owns the bus instances, actuators self-register a handler keyed by
// their address (DXL ID here, MST_ID there), and dispatch happens from poll() in
// thread context so a handler may safely start the next transaction.
//
// ── Why this layer exists at all ──────────────────────────────────────────────
// A DM-J actuator writes one CAN frame and is done: FlexCAN is full duplex with a
// hardware FIFO, so DmActuator::sendMITCommand() is fire-and-forget and its reply
// lands asynchronously in a handler. A DYNAMIXEL bus is ONE WIRE. Only one device
// may drive it at a time, a reply cannot overlap a request, and the reply is not
// dispatched by any hardware — it must be matched to the request that is still
// outstanding. That serialization is this class's whole job, and it is why the
// actuator cannot simply own its own UART calls the way DmActuator owns its CAN
// sends.
//
// ── Half duplex on Teensy 4.x ────────────────────────────────────────────────
// begin() opens the port with SERIAL_8N1_HALF_DUPLEX, which the Teensy 4 core
// implements as the i.MX RT1062 LPUART's single-wire mode (CTRL.LOOPS | CTRL.RSRC):
// the receiver takes its input from the TXD pin, and the core flips CTRL.TXDIR on
// each write and clears it on transmit-complete. So the SINGLE WIRE IS THE TX PIN
// (Serial3 -> pin 14) and no external tri-state buffer is needed for direction.
//   ⚠ ELECTRICAL: a DYNAMIXEL X-series TTL bus is 5V logic; Teensy 4.x pins are
//   3.3V and NOT 5V tolerant ("do not drive any digital pin higher than 3.3V").
//   A level shifter is REQUIRED between this pin and the servo. Nothing in
//   firmware can substitute for it.
//
// ── The scheduler ────────────────────────────────────────────────────────────
// poll() advances AT MOST ONE transaction and never blocks:
//   * a reply that is outstanding is drained/parsed/timed out first;
//   * otherwise the next registered actuator in round-robin gets one slot, and
//     each actuator alternates WRITE (its cached setpoint) then READ (its state).
// Writes elicit no reply at all because setup() sets Status Return Level (68) = 1
// ("reply to READ instructions only"), which removes a line turnaround per write —
// the single biggest cost on a half-duplex bus.
//
// Blocking helpers (pingRead / writeBlocking) exist for BRING-UP ONLY, where the
// DM path blocks too (groundLimitsFromMotor). They must never be called from the
// control loop.
class DxlBus {
public:
  // Dispatched when a status packet arrives for a registered ID.
  // `err` is the status packet's Error byte; `params`/`len` are its parameters.
  //
  // `addr` is the control-table address of the read THIS reply answers — supplied by
  // the bus, which is the only party that knows it. A device must not infer it from
  // whatever it most recently asked for: it re-arms reads on a timer, so with more
  // than one device sharing the wire the next request is routinely armed while the
  // previous reply is still in flight. Parsing by the last-armed address then decodes
  // a reply under the wrong schema, or matches nothing at all and silently stops
  // refreshing the feedback timestamp — which on hardware looked like `age` climbing
  // into the seconds while the servo was in fact answering every request.
  using Handler = void (*)(uint8_t err, uint16_t addr, const uint8_t *params,
                           uint8_t len, void *ctx);

  // A pending read one actuator wants serviced. The actuator fills this in; the
  // bus decides when it goes on the wire.
  struct ReadRequest {
    uint16_t addr = 0;
    uint16_t len  = 0;
    bool     want = false;
  };

  static constexpr size_t  kMaxDevices  = 8;    // devices per serial bus
  static constexpr uint8_t kBroadcastId = 0xFE;
  static constexpr size_t  kMaxPacket   = 64;   // our largest tx/rx frame

  // Instruction bytes (Protocol 2.0).
  static constexpr uint8_t kPing      = 0x01;
  static constexpr uint8_t kRead      = 0x02;
  static constexpr uint8_t kWrite     = 0x03;
  static constexpr uint8_t kStatus    = 0x55;

  // Every standard DYNAMIXEL rate, in the order worth trying (Baud Rate reg 8 codes
   // 3,1,2,4,0,5,6,7). 1 Mbps first because that is what a provisioned robot runs;
  // 57600 second because that is what a factory servo runs.
  static constexpr uint32_t kStdBauds[8] = {1000000, 57600, 115200, 2000000,
                                            9600, 3000000, 4000000, 4500000};

  void begin(HardwareSerial &port, uint32_t baud = 1000000) {
    port_ = &port;
    baud_ = baud;
    port.begin(baud, SERIAL_8N1_HALF_DUPLEX);
    tx_cap_ = (size_t)port.availableForWrite();   // idle => full TX capacity
    // One byte-time at this baud, in microseconds (10 bits per 8N1 frame), used
    // to size the reply timeout against the actual line rate rather than a
    // hard-coded millisecond guess that would be wrong at 57.6 kbps.
    byte_us_ = (10UL * 1000000UL) / baud_;
    if (byte_us_ == 0) byte_us_ = 1;
  }

  bool ready() const { return port_ != nullptr; }
  uint32_t baud() const { return baud_; }
  uint32_t byteUs() const { return byte_us_; }
  size_t   deviceCount() const { return n_dev_; }

  // Re-open the port at each standard rate and stay at the one where the MOST of
  // `ids` answer (all of them ends the sweep early). This is what lets the FIRMWARE
  // adapt to whatever a servo is set to, instead of the servo having to be
  // re-provisioned to match a number compiled into the firmware.
  //
  // "Most", not "all" (2026-09-30): the two axes of a 2XL430 share one baud, but one
  // axis with a wrong ID — or one axis simply absent — used to fail EVERY rate and
  // leave the port at the last rate tried, so the axis that WAS answering was lost
  // with it. Returns the baud that worked, or 0 when nothing answered anywhere; the
  // caller decides where to leave the port in that case.
  uint32_t detectBaud(HardwareSerial &port, const uint8_t *ids, size_t n) {
    uint32_t best_baud = 0;
    size_t   best_hits = 0;
    for (uint32_t b : kStdBauds) {
      begin(port, b);
      delay(20);
      size_t hits = 0;
      for (size_t i = 0; i < n; ++i) {
        uint16_t model = 0;
        if (ping(ids[i], model, 15)) ++hits;
      }
      if (hits == n && n > 0) return b;            // everyone home: done
      if (hits > best_hits) { best_hits = hits; best_baud = b; }
    }
    if (best_hits > 0) {
      begin(port, best_baud);                        // the sweep ended elsewhere
      delay(20);
      return best_baud;
    }
    return 0;
  }

  // Time one full round of present-state reads costs on this wire, in ms. The read
  // cadence must not be a constant: 21 payload bytes plus framing is ~0.5 ms at
  // 1 Mbps but ~8 ms at 57600, so a fixed 10 ms period silently oversubscribes a slow
  // bus by 2.5x once there is more than one device — which starves whichever device
  // loses the race, permanently. Measured on hardware before this existed.
  uint32_t roundTripMs(uint16_t payload) const {
    uint32_t per = (byte_us_ * (uint32_t)(payload + 11 + 14)) / 1000u + 1u;
    uint32_t n = n_dev_ ? (uint32_t)n_dev_ : 1u;
    return per * n;
  }

  bool registerHandler(uint8_t id, Handler h, void *ctx) {
    if (n_dev_ >= kMaxDevices) return false;
    devices_[n_dev_++] = Device{id, h, ctx, {}, false, false, 0, 0, {}, {}, 0, 0};
    return true;
  }

  // ── Two write channels, because they need opposite delivery guarantees ──────
  //
  // setWrite()   — the SETPOINT channel: one coalescing slot, latest-wins. A goal
  //                position superseded before it reached the wire was never worth
  //                sending, and collapsing duplicates is what keeps a single-wire
  //                bus from drowning in identical position writes.
  //
  // queueWrite() — the CONTROL channel: a small FIFO, every entry delivered. Torque
  //                enable, gains and mode are EVENTS, not setpoints; dropping one
  //                loses a state transition rather than a stale sample.
  //
  // They were one slot to begin with, and that silently ate torque-enable: enable()
  // cached the write, service() overwrote it with a goal position microseconds later,
  // and whether torque ever came on depended on if the bus happened to be idle in
  // between. It passed intermittently, which is worse than failing.
  void setWrite(uint8_t id, uint16_t addr, const uint8_t *data, uint8_t len) {
    Device *d = find(id);
    if (!d || len > sizeof(d->wr_data)) return;
    d->wr_addr = addr;
    d->wr_len  = len;
    memcpy(d->wr_data, data, len);
    d->wr_want = true;
  }

  // Enqueue a control-register write. Returns false if the queue is full (which would
  // mean the caller is producing transitions faster than a serial bus can carry them).
  bool queueWrite(uint8_t id, uint16_t addr, const uint8_t *data, uint8_t len) {
    Device *d = find(id);
    if (!d || len > sizeof(d->ctrl_q[0].data)) return false;
    uint8_t next = (uint8_t)((d->ctrl_head + 1) % kCtrlQueue);
    if (next == d->ctrl_tail) { ++ctrl_dropped_; return false; }
    PendingWrite &w = d->ctrl_q[d->ctrl_head];
    w.addr = addr;
    w.len  = len;
    memcpy(w.data, data, len);
    d->ctrl_head = next;
    return true;
  }

  // Ask for `len` bytes from `addr` on the next scheduled slot for `id`.
  void setRead(uint8_t id, uint16_t addr, uint16_t len) {
    Device *d = find(id);
    if (!d) return;
    d->rd = {addr, len, true};
  }

  // Advance the bus by at most one transaction. Never blocks. Call from loop().
  void poll() {
    if (!port_) return;

    if (awaiting_) {
      drainReply();
      // SIGNED compare: tx_done_us_ is the moment transmission will FINISH, which is
      // in the future when bytes are still queued. An unsigned compare against a
      // future stamp reads as a huge elapsed time and fires the timeout instantly.
      if (awaiting_ &&
          (int32_t)(micros() - tx_done_us_) > (int32_t)reply_timeout_us_) {
        awaiting_ = false;
        ++timeouts_;
        rx_len_ = 0;
      }
      return;                       // one thing at a time on a single wire
    }

    if (n_dev_ == 0) return;

    // Round-robin over devices; WITHIN a device, strictly alternate write and read.
    //
    // The alternation is not politeness, it is required. A caller re-arms its write
    // every service() call — i.e. at loop rate, far faster than a single-wire bus can
    // drain it — so "send the write if one is pending" means a write is ALWAYS
    // pending and the read never gets a slot. That starves feedback completely the
    // moment a position stream starts, which on hardware looked like the servo going
    // deaf the instant torque was enabled (feedback age climbing monotonically while
    // commands were still being accepted). `prefer_read` makes each device give up
    // its slot to the read after every write it sends.
    for (size_t k = 0; k < n_dev_; ++k) {
      size_t i = (cursor_ + k) % n_dev_;
      Device &d = devices_[i];
      const bool c = (d.ctrl_head != d.ctrl_tail);   // control write pending
      const bool w = d.wr_want, r = d.rd.want;
      if (!c && !w && !r) continue;
      cursor_ = (i + 1) % n_dev_;
      if (r && (d.prefer_read || (!c && !w))) {
        d.rd.want = false;
        d.prefer_read = false;
        sendRead(d.id, d.rd.addr, d.rd.len);
      } else if (c) {
        // Control writes go first and are never dropped — they are transitions.
        PendingWrite &pw = d.ctrl_q[d.ctrl_tail];
        d.ctrl_tail = (uint8_t)((d.ctrl_tail + 1) % kCtrlQueue);
        d.prefer_read = true;
        sendWrite(d.id, pw.addr, pw.data, pw.len);
      } else {
        d.wr_want = false;
        d.prefer_read = true;      // the next slot for this device is the read's
        sendWrite(d.id, d.wr_addr, d.wr_data, d.wr_len);
      }
      return;
    }
  }

  // ---- bring-up helpers (BLOCKING — setup() only, never the control loop) ----

  // Ping `id` and, on success, return its Model Number. Presence check.
  bool ping(uint8_t id, uint16_t &model_out, uint32_t timeout_ms = 20) {
    uint8_t params[8];
    uint8_t n = 0, err = 0;
    if (!txrxBlocking(id, kPing, nullptr, 0, params, sizeof(params), n, err, timeout_ms))
      return false;
    if (n < 3) return false;
    model_out = (uint16_t)params[0] | ((uint16_t)params[1] << 8);
    return true;
  }

  // Blocking register write with no status expected (Status Return Level 1) —
  // used before the servo is configured, when its return level is still unknown,
  // so we simply let the line settle instead of waiting for a reply.
  void writeBlocking(uint8_t id, uint16_t addr, const uint8_t *data, uint8_t len) {
    sendWrite(id, addr, data, len);
    port_->flush();
    // Wait out — and discard — a status packet the servo MAY send. Whether it does
    // depends on Status Return Level, which during bring-up is exactly what we are
    // still trying to set, so we cannot assume either way. An 11-byte status frame
    // plus turnaround; cheap, and only ever used off the control loop.
    uint32_t t0 = micros();
    const uint32_t wait_us = byte_us_ * 14 + 1500;
    while ((uint32_t)(micros() - t0) < wait_us)
      while (port_->available()) port_->read();
  }

  // Blocking register read. Returns false on timeout.
  bool readBlocking(uint8_t id, uint16_t addr, uint16_t len,
                    uint8_t *out, uint8_t out_cap, uint32_t timeout_ms = 20) {
    uint8_t p[2] = {(uint8_t)(addr & 0xFF), (uint8_t)(addr >> 8)};
    uint8_t req[4] = {p[0], p[1], (uint8_t)(len & 0xFF), (uint8_t)(len >> 8)};
    uint8_t n = 0, err = 0;
    if (!txrxBlocking(id, kRead, req, 4, out, out_cap, n, err, timeout_ms)) return false;
    return n >= len;
  }

  // True while `id` already has a read queued or on the wire. A device must not
  // stack another request on top: on a serial bus the backlog does not make the
  // answers arrive sooner, it just spends the next request's timeout budget.
  bool readPending(uint8_t id) {
    Device *d = find(id);
    if (!d) return false;
    return d->rd.want || (awaiting_ && awaiting_id_ == id);
  }

  uint32_t timeouts() const { return timeouts_; }
  uint32_t replies()  const { return replies_;  }
  uint32_t ctrlDropped() const { return ctrl_dropped_; }
  uint32_t foreign()     const { return foreign_; }
  // How many stuffed bytes have been removed from inbound payloads. Non-zero proves
  // the stuffing is REAL traffic on this bus, not a spec footnote — each one is a
  // frame that would previously have decoded every later field one byte late.
  uint32_t stuffed()     const { return stuffed_; }

private:
  static constexpr size_t kCtrlQueue = 6;   // pending control writes per device

  struct PendingWrite {
    uint16_t addr = 0;
    uint8_t  len  = 0;
    uint8_t  data[8] = {0};
  };

  struct Device {
    uint8_t     id      = 0;
    Handler     h       = nullptr;
    void       *ctx     = nullptr;
    ReadRequest rd      = {};
    bool        wr_want = false;
    bool        prefer_read = false;   // this device's next slot belongs to its read
    uint16_t    wr_addr = 0;
    uint8_t     wr_len  = 0;
    uint8_t     wr_data[8] = {0};
    PendingWrite ctrl_q[kCtrlQueue]{};
    uint8_t     ctrl_head = 0, ctrl_tail = 0;
  };

  Device *find(uint8_t id) {
    for (size_t i = 0; i < n_dev_; ++i)
      if (devices_[i].id == id) return &devices_[i];
    return nullptr;
  }

  // ---- Protocol 2.0 framing -------------------------------------------------

  // CRC-16 (IBM/ANSI), polynomial 0x8005, init 0, MSB-first, no reflection, no
  // final XOR — over every byte from the header through the last parameter.
  // Verified against the e-manual's documented PING packet, whose CRC is 0x4E19.
  static uint16_t crc16(const uint8_t *data, size_t len, uint16_t crc = 0) {
    for (size_t i = 0; i < len; ++i) {
      uint16_t acc = (uint16_t)((((crc >> 8) ^ data[i]) & 0xFF) << 8);
      for (uint8_t b = 0; b < 8; ++b)
        acc = (acc & 0x8000) ? (uint16_t)((acc << 1) ^ 0x8005) : (uint16_t)(acc << 1);
      crc = (uint16_t)((crc << 8) ^ acc);
    }
    return crc;
  }

  // Build an instruction packet into tx_, applying the FF FF FD byte-stuffing
  // rule to the parameter region. Returns the total length.
  size_t buildPacket(uint8_t id, uint8_t instr, const uint8_t *params, uint8_t nparams) {
    size_t i = 0;
    tx_[i++] = 0xFF; tx_[i++] = 0xFF; tx_[i++] = 0xFD; tx_[i++] = 0x00;
    tx_[i++] = id;
    size_t len_pos = i; i += 2;          // LENGTH filled in after stuffing
    tx_[i++] = instr;

    // Byte stuffing: an FF FF FD occurring in the payload gets an extra FD so it
    // can never be mistaken for a header. It changes LENGTH, hence the fill-in.
    uint8_t run = 0;
    for (uint8_t k = 0; k < nparams && i < kMaxPacket - 2; ++k) {
      uint8_t b = params[k];
      tx_[i++] = b;
      if (run == 0 && b == 0xFF)      run = 1;
      else if (run == 1 && b == 0xFF) run = 2;
      else if (run == 2 && b == 0xFD) { tx_[i++] = 0xFD; run = 0; }
      else                            run = (b == 0xFF) ? 1 : 0;
    }

    uint16_t length = (uint16_t)(i - len_pos - 2) + 2;   // instr+params+2 CRC bytes
    tx_[len_pos]     = (uint8_t)(length & 0xFF);
    tx_[len_pos + 1] = (uint8_t)(length >> 8);

    uint16_t crc = crc16(tx_, i);
    tx_[i++] = (uint8_t)(crc & 0xFF);
    tx_[i++] = (uint8_t)(crc >> 8);
    return i;
  }

  // Remove the byte stuffing a TRANSMITTER inserted, in place, returning the true
  // length. Protocol 2.0 stuffs in BOTH directions: any FF FF FD occurring in a
  // payload is sent as FF FF FD FD so it can never be mistaken for a header.
  //
  // Omitting this does not corrupt the frame in any way a checksum can see — LENGTH
  // counts the inserted byte and the CRC is computed over the stuffed form, so the
  // packet validates perfectly and every field after the insertion is read ONE BYTE
  // LATE. On the 21-byte present block that silently turns Present Position into
  // garbage: Present Load = -1 with Present Velocity = -3 puts FF FF FD at the front
  // of the block — both unremarkable values for a lightly loaded servo creeping
  // backwards — and a joint sitting at 0 deg decodes as 45922 deg. Intermittent by
  // nature, because it depends on the measured values rather than on anything
  // structural, which is what makes it read as "the servo glitches sometimes".
  uint8_t destuff(uint8_t *p, uint8_t len) {
    uint8_t w = 0, run = 0;
    for (uint8_t r = 0; r < len; ++r) {
      uint8_t b = p[r];
      if (run == 3) { run = 0; if (b == 0xFD) { ++stuffed_; continue; } }  // drop it
      p[w++] = b;
      if (b == 0xFF)                 run = (run >= 1) ? 2 : 1;
      else if (b == 0xFD && run == 2) run = 3;
      else                            run = 0;
    }
    return w;
  }

  void sendRaw(size_t n) {
    // Never block on a full TX FIFO: at 1 Mbps a 64-byte buffer drains in ~640 us,
    // which would stall the whole control loop. Our frames are <= ~24 B and the
    // scheduler issues one at a time, so this only ever declines a pathological case.
    if ((size_t)port_->availableForWrite() < n) { ++dropped_; return; }
    while (port_->available()) port_->read();      // discard stale RX before TX
    port_->write(tx_, n);
    // Stamp when the wire will actually be free, not when write() returned. write()
    // only queues; anything already in the TX FIFO (typically another device's write)
    // goes out first. Timing the reply from the queue moment charges this request for
    // the previous one's transmission and produces timeouts that look like a servo
    // that stopped answering.
    size_t free_now = (size_t)port_->availableForWrite();
    size_t queued = (tx_cap_ > free_now) ? (tx_cap_ - free_now) : n;
    tx_done_us_ = micros() + (uint32_t)(queued * byte_us_);
  }

  void sendWrite(uint8_t id, uint16_t addr, const uint8_t *data, uint8_t len) {
    uint8_t p[10];
    p[0] = (uint8_t)(addr & 0xFF);
    p[1] = (uint8_t)(addr >> 8);
    for (uint8_t k = 0; k < len && k < 8; ++k) p[2 + k] = data[k];
    sendRaw(buildPacket(id, kWrite, p, (uint8_t)(2 + len)));
    // Status Return Level 1 => a write draws no reply. Nothing to await.
  }

  void sendRead(uint8_t id, uint16_t addr, uint16_t len) {
    uint8_t p[4] = {(uint8_t)(addr & 0xFF), (uint8_t)(addr >> 8),
                    (uint8_t)(len & 0xFF),  (uint8_t)(len >> 8)};
    sendRaw(buildPacket(id, kRead, p, 4));
    awaiting_id_   = id;
    awaiting_addr_ = addr;
    awaiting_      = true;
    rx_len_       = 0;
    // Expected reply: 11 framing bytes + len parameters, plus the servo's Return
    // Delay Time and our own TX drain. Generous x4 margin — a timeout here costs
    // one stale sample, but a too-tight one would drop every good reply.
    reply_timeout_us_ = byte_us_ * (uint32_t)(len + 11) * 4 + 2000;
  }

  // Accumulate bytes and dispatch a complete, CRC-valid status packet.
  void drainReply() {
    while (port_->available()) {
      uint8_t b = (uint8_t)port_->read();
      if (rx_len_ < sizeof(rx_)) rx_[rx_len_++] = b;
      else { rx_len_ = 0; continue; }

      // Resync on the header. Slide the window left on a mismatch instead of
      // dropping everything: discarding the whole partial match loses the header's
      // own bytes. `FF FF FF FD 00` — one stray FF ahead of a valid header — would
      // clear the buffer on the third FF and then never match again, silently eating
      // an entire good reply. Sliding re-tests the shorter suffix, which is what a
      // rolling header search actually requires.
      if (rx_len_ <= 4) {
        static const uint8_t hdr[4] = {0xFF, 0xFF, 0xFD, 0x00};
        while (rx_len_ > 0 && rx_[rx_len_ - 1] != hdr[rx_len_ - 1]) {
          for (size_t k = 1; k < rx_len_; ++k) rx_[k - 1] = rx_[k];
          --rx_len_;
        }
        continue;
      }
      if (rx_len_ < 7) continue;

      uint16_t length = (uint16_t)rx_[5] | ((uint16_t)rx_[6] << 8);
      size_t   total  = 7 + (size_t)length;
      if (total > sizeof(rx_)) { rx_len_ = 0; continue; }
      if (rx_len_ < total) continue;

      uint16_t want = crc16(rx_, total - 2);
      uint16_t got  = (uint16_t)rx_[total - 2] | ((uint16_t)rx_[total - 1] << 8);
      rx_len_ = 0;
      // A frame that is not a good status packet must NOT end the wait. On a
      // single-wire bus the receiver's input is the TX pin, so our own instruction
      // packet can echo back into the RX path; treating that as "the reply came and
      // was wrong" would cancel the wait microseconds before the servo actually
      // answers. Only a dispatch or a timeout ends a wait.
      if (want != got) { ++crc_errors_; continue; }
      if (rx_[7] != kStatus) continue;          // our own echo, or a foreign frame

      uint8_t id  = rx_[4];
      uint8_t err = rx_[8];
      // params run from index 9 to total-3 (before the 2 CRC bytes), then de-stuff.
      uint8_t *params = &rx_[9];
      uint8_t plen = destuff(params, (uint8_t)(total - 2 - 9));

      // The reply MUST be from the device we are waiting on. A late or duplicated
      // frame from another device is not an answer to this request: crediting it
      // would decode it under the wrong address AND end this wait, so the device we
      // actually asked would time out. With two devices that degenerates into a
      // lockstep where one is serviced forever and the other never is — observed on
      // hardware as one axis at age 2 ms and the other climbing past 13 s with 371
      // timeouts, while both servos were provably healthy and answering.
      if (id != awaiting_id_) { ++foreign_; continue; }

      ++replies_;
      awaiting_ = false;
      Device *d = find(id);
      if (d && d->h) d->h(err, awaiting_addr_, params, plen, d->ctx);
      return;
    }
  }

  // One blocking request/response, for bring-up only.
  bool txrxBlocking(uint8_t id, uint8_t instr, const uint8_t *params, uint8_t nparams,
                    uint8_t *out, uint8_t out_cap, uint8_t &out_len, uint8_t &err,
                    uint32_t timeout_ms) {
    if (!port_) return false;
    while (port_->available()) port_->read();
    size_t n = buildPacket(id, instr, params, nparams);
    if ((size_t)port_->availableForWrite() < n) return false;
    port_->write(tx_, n);
    port_->flush();

    size_t got = 0;
    uint32_t t0 = millis();
    while ((millis() - t0) < timeout_ms) {
      if (!port_->available()) continue;
      uint8_t b = (uint8_t)port_->read();
      if (got < sizeof(rx_)) rx_[got++] = b; else { got = 0; continue; }
      if (got <= 4) {
        static const uint8_t hdr[4] = {0xFF, 0xFF, 0xFD, 0x00};
        if (rx_[got - 1] != hdr[got - 1]) got = 0;
        continue;
      }
      if (got < 7) continue;
      uint16_t length = (uint16_t)rx_[5] | ((uint16_t)rx_[6] << 8);
      size_t   total  = 7 + (size_t)length;
      if (total > sizeof(rx_)) { got = 0; continue; }
      if (got < total) continue;
      uint16_t want = crc16(rx_, total - 2);
      uint16_t gotc = (uint16_t)rx_[total - 2] | ((uint16_t)rx_[total - 1] << 8);
      // Same echo rule as drainReply(): keep looking until the timeout rather than
      // declaring failure on the first frame that is not the answer.
      if (want != gotc) { got = 0; continue; }
      if (rx_[4] != id || rx_[7] != kStatus) { got = 0; continue; }
      err = rx_[8];
      uint8_t plen = destuff(&rx_[9], (uint8_t)(total - 2 - 9));
      out_len = plen < out_cap ? plen : out_cap;
      memcpy(out, &rx_[9], out_len);
      return true;
    }
    return false;
  }

  HardwareSerial *port_ = nullptr;
  uint32_t baud_    = 1000000;
  uint32_t byte_us_ = 10;
  size_t   tx_cap_  = 64;   // TX FIFO capacity, sampled while idle in begin()

  Device devices_[kMaxDevices]{};
  size_t n_dev_  = 0;
  size_t cursor_ = 0;

  bool     awaiting_    = false;
  uint8_t  awaiting_id_ = 0;
  uint16_t awaiting_addr_ = 0;   // address of the read currently in flight
  uint32_t tx_done_us_  = 0;
  uint32_t reply_timeout_us_ = 3000;

  uint8_t tx_[kMaxPacket]{};
  uint8_t rx_[kMaxPacket]{};
  size_t  rx_len_ = 0;

  uint32_t timeouts_   = 0;
  uint32_t replies_    = 0;
  uint32_t crc_errors_ = 0;
  uint32_t dropped_      = 0;
  uint32_t ctrl_dropped_ = 0;
  uint32_t foreign_      = 0;   // well-formed replies from a device we were not awaiting
  uint32_t stuffed_      = 0;   // stuffed bytes removed from inbound payloads
};
