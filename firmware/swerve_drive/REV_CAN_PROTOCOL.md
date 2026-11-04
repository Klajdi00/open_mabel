# REV SPARK CAN Protocol Reference (V2.5)

Protocol reference derived from the vendored `sparkcan` library (`sparkcan_py/`, `SparkBase.hpp` / `SparkBase.cpp`). It has enough detail to write a raw Python (Linux SocketCAN) or C++ (Teensy / FlexCAN_T4) implementation without the `sparkcan_py` bindings; `scripts/set_drive_limits.py` is one such implementation.

## 1. Arbitration ID Construction

All SPARK frames use 29-bit extended IDs (`CAN_EFF_FLAG` must be set when sending on Linux). The ID encodes identity and intent natively.

**Constants:**

* `DEVICE_TYPE = 0x02` (Motor Controller)
* `MANUFACTURER = 0x05` (REV Robotics)

**Formula:**
`arbId = (0x02 << 24) | (0x05 << 16) | (apiClass << 10) | (apiIndex << 6) | deviceId`

**Important Distinction:** The source code's `APICommand` enum stores values compactly as `(apiClass << 4) | apiIndex`. When building the actual CAN ID, the code extracts `apiClass` with `cmd >> 4` and `apiIndex` with `cmd & 0x0F`, placing them at bit positions 10 and 6.

### Common Frame Examples

| Command | apiClass | apiIndex | Device 1 Arb ID |
| --- | --- | --- | --- |
| Velocity | 0 | 0 | `0x02050001` |
| Position | 0 | 4 | `0x02050101` |
| Heartbeat | 11 | 2 | `0x02052C81` |
| Period2 RX | 46 | 2 | `0x0205B881` |
| Period5 RX | 46 | 5 | `0x0205B941` |

---

## 2. Linux SocketCAN Frame Format

The standard Linux kernel struct, utilized by the Python `struct` module format string `"=IB3x8s"`:

* `I` — uint32 can_id (must include `0x80000000` for EFF)
* `B` — uint8 dlc
* `3x` — 3 padding bytes
* `8s` — 8 bytes data payload

---

## 3. API Commands (V2.5)

The V2.5 mappings differ significantly from V2.4.

### Motor Control (apiClass 0)

* **Velocity:** Index 0 (RPM, IEEE float)
* **DutyCycle:** Index 2 ([-1.0, 1.0], IEEE float)
* **Position:** Index 4 (Rotations, IEEE float)
* **Voltage:** Index 5 (Volts, IEEE float)

### System Commands

* **Heartbeat:** Class 11, Index 2. Payload: `[0xFF]*8`
* **ClearFaults:** Class 7, Index 14. Payload: `[0x00]*8`
* **BurnFlash:** Class 63, Index 15. Payload: `[0xA3, 0x3A]`

### Control Frame Payload Format

All motor setpoint commands (`Velocity`, `Position`, `DutyCycle`) use an identical 8-byte payload format:

* **Bytes 0–3:** IEEE 754 float, little-endian (Setpoint value)
* **Bytes 4–7:** `0x00 0x00 0x00 0x00` (Padding)

---

## 4. The Parameter System (API Class 48)

Parameters (like PID gains or Idle Mode) are configured separately from control frames. They intentionally overflow the standard `apiIndex` space to map across API classes 48-63.

**Parameter Arb ID Formula:**
`paramArbId = (0x02 << 24) | (0x05 << 16) | (48 << 10) | (paramId << 6) | deviceId`

**Parameter Write Payload (5 Bytes):**

* **Bytes 0–3:** Value encoded as little-endian float, uint32, or bool (uint8).
* **Byte 4 (Type Tag):** `0x01` UINT (uint32), `0x02` FLOAT (IEEE 754), `0x03` BOOL (uint8)

### Key Swerve Parameters

| Parameter | ID | Type | Notes |
| --- | --- | --- | --- |
| `kMotorType` | 2 | UINT | 0=Brushed, 1=Brushless |
| `kCtrlType` | 5 | UINT | 1=Velocity, 3=Position |
| `kIdleMode` | 6 | UINT | 0=Coast, 1=Brake |
| `kInverted` | 45 | BOOL | Flips motor direction |
| `kP_0` | 13 | FLOAT | PID slot 0 proportional gain |
| `kDutyCycleZeroOffset` | 154 | FLOAT | Absolute encoder zero point |
| `kForceEnableStatus5` | 191 | BOOL | Required for absolute steer telemetry |

---

## 5. Periodic Status Frames (Telemetry)

Motors autonomously broadcast telemetry at configurable rates. The Arb ID uses `apiClass=46` and `apiIndex` matching the frame period (0–5).

### Period 0 (Power / Temp)

**Arb ID:** `0x0205B800 | deviceId`

* **Bytes 0-1:** Duty Cycle Raw (int16) -> Divide by `32442.0f`
* **Bytes 2-3 (part):** Voltage Raw (12-bit) -> Divide by `128.0f`
* **Bytes 3-4 (part):** Current Raw (12-bit) -> Divide by `32.0f`
* **Byte 5:** Temperature (°C)

### Period 2 (Primary Encoder)

**Arb ID:** `0x0205B880 | deviceId`

* **Bytes 0–3:** IEEE 754 float -> Velocity (RPM)
* **Bytes 4–7:** IEEE 754 float -> Position (Rotations)

### Period 5 (Absolute Duty-Cycle Encoder)

**Arb ID:** `0x0205B940 | deviceId`

* **Bytes 0–3:** IEEE 754 float -> Absolute Encoder Velocity
* **Bytes 4–7:** IEEE 754 float -> Absolute Encoder Position [0.0, 1.0)
* *Note: Must set `kForceEnableStatus5 = true` (Parameter 191) to receive this frame.*

---

## 6. Shared Thread Architecture

The `sparkcan` library (`SparkBase.cpp`) uses a specific multi-threading architecture to prevent socket blocking and ensure fresh telemetry.

1. **Single Socket:** Only one `AF_CAN / SOCK_RAW / CAN_RAW` socket is opened, regardless of the number of motors.
2. **Global Registry:** A mutex-protected `std::unordered_map<uint8_t, SparkBase*>` maps CAN IDs to specific motor objects.
3. **RX Thread:** A single background thread (`RxLoop`) continually reads the socket in a non-blocking loop. It extracts the `deviceId` from the arbitration ID (`frame.can_id & 0x3F`), looks up the correct motor object in the registry, and pushes the new data into the object's cache.
4. **Mutex Caching:** User-facing calls like `GetVelocity()` do not request data; they simply return the latest cached value protected by a local mutex.
