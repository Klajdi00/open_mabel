// main_armlink.cpp — ARMS Teensy 4.1: ethernet MIT gateway for both DaMiao arms.
//
// ⚠ PLACEHOLDER-GRADE (2026-07-11): implements the full ArmLink contract
//   (hardware_bridge/docs/ARMLINK.md, include/armlink_protocol.h) but has NOT been
//   compiled/flashed against real motors yet. Build env: `pio run -e mabel_armlink`
//   (QNEthernet + FlexCAN_T4). Verify motor ids/buses against
//   hardware_bridge/calibration_data/mabel.json (slice arm/openarm-dual) before energising.
//
// Role: transparent gateway. The host (a JointGroupDriver in mabel_hw/drivers/) owns
// calibration and targets; this firmware forwards MIT commands onto the right CAN
// bus at a 500 Hz loop and streams every motor's feedback (pos/vel/tau/temps/err)
// back over UDP. Safety rules (boot limp, hold-don't-drop watchdog) below.

#include <Arduino.h>
#include <FlexCAN_T4.h>
#include <QNEthernet.h>

#include "armlink_protocol.h"
#include "config.h"

using namespace qindesign::network;
using namespace armlink;

// ---------------------------------------------------------------------------
// Network — static server on the arms QSFP28 lane (see install/setup_arms_lane.sh)
// ---------------------------------------------------------------------------
static const IPAddress kIp(192, 168, 125, 10);
static const IPAddress kMask(255, 255, 255, 0);
static const IPAddress kGw(192, 168, 125, 2);      // the host; never routed beyond
EthernetUDP udp;

// ---------------------------------------------------------------------------
// CAN — CAN1 = LEFT arm, CAN2 = RIGHT arm (1 Mbps, DaMiao MIT mode)
// ---------------------------------------------------------------------------
FlexCAN_T4<CAN1, RX_SIZE_256, TX_SIZE_16> LeftBus;
FlexCAN_T4<CAN2, RX_SIZE_256, TX_SIZE_16> RightBus;

// Motor table: EVERY motor this gateway owns. Uncommanded motors are held at
// their measured position while torque is on (never snapped to a stale target).
struct Motor {
    uint8_t  bus;        // 1 = left/CAN1, 2 = right/CAN2
    uint8_t  id;
    float    p_max, v_max, t_max;   // MUST match the motor's configured limits
    // live command (from the host) — NAN until first commanded => hold measured
    float    cmd_pos = NAN, kp = 0.0f, kd = 0.0f;
    // live feedback
    float    pos = 0.0f, vel = 0.0f, tau = 0.0f;
    uint8_t  t_mos = 0, t_rotor = 0, err = 0;
    bool     seen = false;          // any feedback yet?
};

// ids 1..7 proximal→distal per side; motor-model split mirrors the calibration
// placeholders (1-2 J8009 shoulders, 3-4 J4340 elbows, 5-7 J3507 wrists).
#define J8009 MABEL_CONFIG::J8009_PMAX, MABEL_CONFIG::J8009_VMAX, MABEL_CONFIG::J8009_TMAX
#define J4340 MABEL_CONFIG::J4340_PMAX, MABEL_CONFIG::J4340_VMAX, MABEL_CONFIG::J4340_TMAX
#define J3507 MABEL_CONFIG::J3507_PMAX, MABEL_CONFIG::J3507_VMAX, MABEL_CONFIG::J3507_TMAX
static Motor motors[] = {
    {1, 1, J8009}, {1, 2, J8009}, {1, 3, J4340}, {1, 4, J4340},
    {1, 5, J3507}, {1, 6, J3507}, {1, 7, J3507},
    {2, 1, J8009}, {2, 2, J8009}, {2, 3, J4340}, {2, 4, J4340},
    {2, 5, J3507}, {2, 6, J3507}, {2, 7, J3507},
};
static const size_t kNumMotors = sizeof(motors) / sizeof(motors[0]);

// ---------------------------------------------------------------------------
// Link state
// ---------------------------------------------------------------------------
static bool     torque_on = false;        // boot LIMP; only a Command enables
static uint32_t last_seq = 0;
static uint32_t last_cmd_ms = 0;          // command watchdog
static uint32_t last_rx_ms = 0;           // connection watchdog (any datagram)
static IPAddress host_ip;                 // learned from inbound packets
static uint16_t host_port = 0;
static uint16_t stream_hz = DEFAULT_STREAM_HZ;
static uint32_t next_state_us = 0;
static uint32_t next_ctrl_us = 0;
static const uint32_t CTRL_PERIOD_US = 2000;   // 500 Hz MIT re-send

// ---------------------------------------------------------------------------
// DaMiao MIT framing (same math as DamiaoMotor.cpp, kept local so this file is
// self-contained and bus-generic)
// ---------------------------------------------------------------------------
static int f2u(float x, float lo, float hi, int bits) {
    if (x < lo) x = lo;
    if (x > hi) x = hi;
    return (int)((x - lo) * ((float)((1 << bits) - 1)) / (hi - lo));
}
static float u2f(int v, float lo, float hi, int bits) {
    return (float)v * (hi - lo) / (float)((1 << bits) - 1) + lo;
}

static bool can_write(uint8_t bus, const CAN_message_t &msg) {
    return (bus == 1) ? LeftBus.write(msg) : RightBus.write(msg);
}

static void ctrl_frame(Motor &m, uint8_t byte7) {
    CAN_message_t msg;
    msg.id = m.id;
    msg.len = 8;
    for (int i = 0; i < 7; i++) msg.buf[i] = 0xFF;
    msg.buf[7] = byte7;
    can_write(m.bus, msg);
}
static void motor_enable(Motor &m)  { ctrl_frame(m, 0xFC); }
static void motor_disable(Motor &m) { ctrl_frame(m, 0xFD); }

static void send_mit(Motor &m, float pos, float kp, float kd) {
    int pi = f2u(pos, -m.p_max, m.p_max, 16);
    int vi = f2u(0.0f, -m.v_max, m.v_max, 12);
    int ki = f2u(kp, 0.0f, 500.0f, 12);
    int di = f2u(kd, 0.0f, 5.0f, 12);
    int ti = f2u(0.0f, -m.t_max, m.t_max, 12);
    CAN_message_t msg;
    msg.id = m.id;
    msg.len = 8;
    msg.buf[0] = (pi >> 8) & 0xFF;
    msg.buf[1] = pi & 0xFF;
    msg.buf[2] = (vi >> 4) & 0xFF;
    msg.buf[3] = ((vi & 0xF) << 4) | ((ki >> 8) & 0xF);
    msg.buf[4] = ki & 0xFF;
    msg.buf[5] = (di >> 4) & 0xFF;
    msg.buf[6] = ((di & 0xF) << 4) | ((ti >> 8) & 0xF);
    msg.buf[7] = ti & 0xFF;
    can_write(m.bus, msg);
}

static void parse_feedback(uint8_t bus, const CAN_message_t &msg) {
    if (msg.len < 8) return;
    uint8_t mid = msg.buf[0] & 0x0F;
    for (size_t i = 0; i < kNumMotors; i++) {
        Motor &m = motors[i];
        if (m.bus != bus || m.id != mid) continue;
        int p = ((int)msg.buf[1] << 8) | msg.buf[2];
        int v = ((int)msg.buf[3] << 4) | (msg.buf[4] >> 4);
        int t = (((int)msg.buf[4] & 0xF) << 8) | msg.buf[5];
        m.pos = u2f(p, -m.p_max, m.p_max, 16);
        m.vel = u2f(v, -m.v_max, m.v_max, 12);
        m.tau = u2f(t, -m.t_max, m.t_max, 12);
        m.t_mos = msg.buf[6];
        m.t_rotor = msg.buf[7];
        m.err = msg.buf[1] >> 4;      // status nibble (DaMiaoMotor.cpp convention)
        m.seen = true;
        return;
    }
}

// ---------------------------------------------------------------------------
// UDP handlers
// ---------------------------------------------------------------------------
static void set_torque_all(bool on) {
    if (on == torque_on) return;
    torque_on = on;
    for (size_t i = 0; i < kNumMotors; i++) {
        if (on) motor_enable(motors[i]); else motor_disable(motors[i]);
        // hold-in-place until the host overwrites: target = measured
        if (on) { motors[i].cmd_pos = NAN; }
    }
}

static void on_command(const uint8_t *body, size_t len) {
    if (len < sizeof(CommandHead)) return;
    CommandHead head;
    memcpy(&head, body, sizeof(head));
    // drop stale/duplicate commands (host guarantees monotonic seq per socket)
    if (head.seq != 0 && head.seq <= last_seq && last_seq - head.seq < 0x7FFFFFFF)
        return;
    if (len < sizeof(CommandHead) + head.count * sizeof(CommandMotor)) return;
    last_seq = head.seq;
    last_cmd_ms = millis();
    set_torque_all(head.flags & FLAG_TORQUE);
    const uint8_t *p = body + sizeof(CommandHead);
    for (uint8_t i = 0; i < head.count; i++) {
        CommandMotor cm;
        memcpy(&cm, p, sizeof(cm));
        p += sizeof(cm);
        for (size_t k = 0; k < kNumMotors; k++) {
            Motor &m = motors[k];
            if (m.bus == cm.bus && m.id == cm.id) {
                m.cmd_pos = cm.pos;
                m.kp = cm.kp;
                m.kd = cm.kd;
                break;
            }
        }
    }
}

static void stream_state() {
    if (host_port == 0) return;
    static uint8_t buf[1 + sizeof(StateHead) + 64 * sizeof(StateMotor)];
    buf[0] = TAG_STATE;
    StateHead head;
    head.seq = last_seq;
    head.t_us = micros();
    head.flags = torque_on ? FLAG_TORQUE : 0;
    head.count = 0;
    size_t off = 1 + sizeof(StateHead);
    for (size_t i = 0; i < kNumMotors; i++) {
        Motor &m = motors[i];
        if (!m.seen) continue;              // honest absence: no feedback, no row
        StateMotor sm = {m.bus, m.id, m.pos, m.vel, m.tau, m.t_mos, m.t_rotor, m.err};
        memcpy(buf + off, &sm, sizeof(sm));
        off += sizeof(sm);
        head.count++;
    }
    memcpy(buf + 1, &head, sizeof(head));
    udp.beginPacket(host_ip, host_port);
    udp.write(buf, off);
    udp.endPacket();
}

// ---------------------------------------------------------------------------
void setup() {
    Serial.begin(115200);
    LeftBus.begin();
    LeftBus.setBaudRate(MABEL_CONFIG::CAN_BAUD_RATE);
    RightBus.begin();
    RightBus.setBaudRate(MABEL_CONFIG::CAN_BAUD_RATE);

    Ethernet.begin(kIp, kMask, kGw);
    udp.begin(SERVER_PORT);
    Serial.println("ArmLink gateway up: 192.168.125.10:5800 (torque OFF / limp)");
}

void loop() {
    // 1) drain CAN feedback
    CAN_message_t msg;
    while (LeftBus.read(msg))  parse_feedback(1, msg);
    while (RightBus.read(msg)) parse_feedback(2, msg);

    // 2) drain UDP
    int sz;
    while ((sz = udp.parsePacket()) > 0) {
        static uint8_t rx[512];
        int n = udp.read(rx, sizeof(rx));
        if (n < 1) continue;
        host_ip = udp.remoteIP();
        host_port = udp.remotePort();
        last_rx_ms = millis();
        if (rx[0] == TAG_COMMAND) {
            on_command(rx + 1, n - 1);
        } else if (rx[0] == TAG_HELLO && n >= (int)(1 + sizeof(Hello))) {
            Hello h;
            memcpy(&h, rx + 1, sizeof(h));
            if (h.stream_hz > 0 && h.stream_hz <= 500) stream_hz = h.stream_hz;
        }
    }

    uint32_t now_us = micros();
    uint32_t now_ms = millis();

    // 3) 500 Hz MIT re-send while torque is on. Command watchdog: with no fresh
    //    Command, HOLD the measured position (never limp — arms must not fall).
    if ((int32_t)(now_us - next_ctrl_us) >= 0) {
        next_ctrl_us = now_us + CTRL_PERIOD_US;
        if (torque_on) {
            bool cmd_fresh = (now_ms - last_cmd_ms) <= CMD_WATCHDOG_MS;
            for (size_t i = 0; i < kNumMotors; i++) {
                Motor &m = motors[i];
                if (!m.seen) continue;
                if (cmd_fresh && !isnan(m.cmd_pos)) {
                    send_mit(m, m.cmd_pos, m.kp, m.kd);
                } else {
                    // hold-in-place: uncommanded joint OR dead commander
                    m.cmd_pos = NAN;               // re-latch on next fresh command
                    send_mit(m, m.pos, m.kp > 0 ? m.kp : 20.0f,
                             m.kd > 0 ? m.kd : 1.0f);
                }
            }
        }
    }

    // 4) STATE stream at stream_hz while the connection watchdog is fed
    if (host_port != 0 && (now_ms - last_rx_ms) <= CONN_WATCHDOG_MS &&
        (int32_t)(now_us - next_state_us) >= 0) {
        next_state_us = now_us + (1000000UL / stream_hz);
        stream_state();
    }
}
