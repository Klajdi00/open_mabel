#pragma once
#include <FlexCAN_T4.h>

// Thin wrapper around one FlexCAN_T4 peripheral that adds:
//   - one-line init at 1 Mbps standard frames + FIFO + interrupt
//   - a per-bus dispatch table mapping (CAN ID -> handler)
//   - a bus-agnostic send() and poll()
//
// One CanBus instance per physical bus; instantiate as many as you have buses.
// Up to MAX_BUSES (4) concurrent instances are supported.
//
// Handlers fire from poll() in thread context, NOT in ISR context, so calling
// send() from inside a handler is safe.
class CanBus {
public:
  using Handler = void(*)(const CAN_message_t &msg, void *context);

  // Bind to a FlexCAN_T4<CANx, ...> instance and configure it. Templated only
  // so we can call the derived-class init methods (begin/setMaxMB/...); the
  // stored reference is the abstract base, so the rest of the API is bus-agnostic.
  template <typename BusT>
  void begin(BusT &bus, uint32_t baud = 1000000) {
    bus_ = &bus;
    uint8_t slot = next_slot_++;
    if (slot >= MAX_BUSES) return;             // silently cap; should never happen
    instances_[slot] = this;
    bus.begin();
    bus.setBaudRate(baud);
    bus.setMaxMB(16);
    bus.enableFIFO();
    bus.enableFIFOInterrupt();
    bus.onReceive(trampolines_[slot]);
  }

  // Drain the FlexCAN RX FIFO; call from loop().
  void poll() { if (bus_) bus_->events(); }

  // Direct write to the underlying bus.
  void send(const CAN_message_t &msg) { bus_->write(msg); }

  // Register a handler for frames whose ID matches `can_id`. Returns false if
  // the table is full (MAX_HANDLERS). Multiple handlers per ID are allowed.
  bool registerHandler(uint32_t can_id, Handler h, void *context) {
    if (n_handlers_ >= MAX_HANDLERS) return false;
    handlers_[n_handlers_++] = {can_id, h, context};
    return true;
  }

  // Public so the application can static_assert MAX_ACTUATORS <= kMaxHandlers: every
  // DmActuator self-registers one handler, and a bus that hosts more actuators than it
  // has handler slots drops the surplus SILENTLY (registerHandler returns false and the
  // constructor cannot report it). Sized to MAX_ACTUATORS so one bus can host them all.
  static constexpr size_t kMaxHandlers = 32;

private:
  static constexpr size_t MAX_BUSES    = 4;
  static constexpr size_t MAX_HANDLERS = kMaxHandlers;

  struct Entry { uint32_t can_id; Handler h; void *context; };

  void dispatch(const CAN_message_t &msg) {
    for (size_t i = 0; i < n_handlers_; i++) {
      if (handlers_[i].can_id == msg.id) {
        handlers_[i].h(msg, handlers_[i].context);
      }
    }
  }

  // Per-slot static trampolines so each FlexCAN instance has its own free
  // function pointer to register, while we still demux to the right CanBus.
  template <uint8_t N>
  static void trampoline(const CAN_message_t &msg) {
    if (instances_[N]) instances_[N]->dispatch(msg);
  }

  inline static CanBus *instances_[MAX_BUSES] = {};
  inline static uint8_t next_slot_ = 0;
  inline static const _MB_ptr trampolines_[MAX_BUSES] = {
    &CanBus::trampoline<0>,
    &CanBus::trampoline<1>,
    &CanBus::trampoline<2>,
    &CanBus::trampoline<3>,
  };

  FlexCAN_T4_Base *bus_ = nullptr;
  Entry            handlers_[MAX_HANDLERS]{};
  size_t           n_handlers_ = 0;
};
