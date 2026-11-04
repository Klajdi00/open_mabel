import can
import time
import threading
import sys

# ==========================================
# 1. MOTOR CONFIGURATIONS
# ==========================================
# Map your motor IDs to their specific limits (PMAX, VMAX, TMAX)
# Update these based on your Damiao config tool settings!
MOTOR_LIMITS = {
    1: {"p_max": 12.566, "v_max": 30.0, "t_max": 40.0},  # Example: J8009P
    2: {"p_max": 12.566, "v_max": 30.0, "t_max": 40.0},  # Example: J8009P
    3: {"p_max": 12.566, "v_max": 10.4, "t_max": 27.0},  # Example: J4340
    # Add Torso (J10422P) and others here as needed
}

# Global dictionary to store the latest feedback from motors
motor_states = {}

# ==========================================
# 2. MATH UTILITIES
# ==========================================
def float_to_uint(x, x_min, x_max, bits):
    span = x_max - x_min
    offset = x_min
    x = max(min(x, x_max), x_min) # Clamp
    return int((x - offset) * ((1 << bits) - 1) / span)

def uint_to_float(x_int, x_min, x_max, bits):
    span = x_max - x_min
    offset = x_min
    return (x_int * span / ((1 << bits) - 1)) + offset

# ==========================================
# 3. CAN COMMAND FUNCTIONS
# ==========================================
def send_enable(bus, motor_id):
    msg = can.Message(arbitration_id=motor_id, data=[0xFF]*7 + [0xFC], is_extended_id=False)
    bus.send(msg)

def send_disable(bus, motor_id):
    msg = can.Message(arbitration_id=motor_id, data=[0xFF]*7 + [0xFD], is_extended_id=False)
    bus.send(msg)

def send_clear_faults(bus, motor_id):
    msg = can.Message(arbitration_id=motor_id, data=[0xFF]*7 + [0xFB], is_extended_id=False)
    bus.send(msg)

def send_mit_command(bus, motor_id, p_des, v_des, kp, kd, t_ff):
    if motor_id not in MOTOR_LIMITS:
        print(f"Error: Limits for Motor {motor_id} not defined.")
        return

    limits = MOTOR_LIMITS[motor_id]
    p_int = float_to_uint(p_des, -limits["p_max"], limits["p_max"], 16)
    v_int = float_to_uint(v_des, -limits["v_max"], limits["v_max"], 12)
    kp_int = float_to_uint(kp, 0.0, 500.0, 12)
    kd_int = float_to_uint(kd, 0.0, 5.0, 12)
    t_int = float_to_uint(t_ff, -limits["t_max"], limits["t_max"], 12)

    data = [
        (p_int >> 8) & 0xFF,
        p_int & 0xFF,
        (v_int >> 4) & 0xFF,
        ((v_int & 0xF) << 4) | ((kp_int >> 8) & 0xF),
        kp_int & 0xFF,
        (kd_int >> 4) & 0xFF,
        ((kd_int & 0xF) << 4) | ((t_int >> 8) & 0xF),
        t_int & 0xFF
    ]

    msg = can.Message(arbitration_id=motor_id, data=data, is_extended_id=False)
    bus.send(msg)

def scan_bus(bus):
    print("\n--- Sweeping CAN Bus ---")
    for motor_id in range(1, 16):
        # Send read command for register 0x00 (UV_Value)
        msg = can.Message(
            arbitration_id=0x7FF, 
            data=[motor_id & 0xFF, (motor_id >> 8) & 0xFF, 0x33, 0x00], 
            is_extended_id=False
        )
        bus.send(msg)
        time.sleep(0.01)
    print("--- Sweep Complete ---\n")

# ==========================================
# 4. BACKGROUND RECEIVE THREAD
# ==========================================
def receive_loop(bus):
    while True:
        msg = bus.recv(timeout=1.0)
        if msg is None: continue

        # Check for Scanner Response
        if msg.arbitration_id == 0x7FF and len(msg.data) >= 3 and msg.data[2] == 0x33:
            responding_id = msg.data[0] | (msg.data[1] << 8)
            print(f"\n[SCANNER] Success: Motor ID {responding_id} detected!")
            continue

        # Standard Feedback (assuming Master ID is 0)
        if msg.arbitration_id == 0 and len(msg.data) == 8:
            sender_id = msg.data[0]
            if sender_id in MOTOR_LIMITS:
                limits = MOTOR_LIMITS[sender_id]
                err_code = msg.data[1] >> 4
                
                p_int = (msg.data[1] & 0x0F) << 8 | msg.data[2] # Adjust parsing based on manual exact bit masking
                p_int = (msg.data[1] << 8) | msg.data[2] # Standard Damiao parsing
                v_int = (msg.data[3] << 4) | (msg.data[4] >> 4)
                t_int = ((msg.data[4] & 0xF) << 8) | msg.data[5]

                motor_states[sender_id] = {
                    "pos": uint_to_float(p_int, -limits["p_max"], limits["p_max"], 16),
                    "vel": uint_to_float(v_int, -limits["v_max"], limits["v_max"], 12),
                    "torq": uint_to_float(t_int, -limits["t_max"], limits["t_max"], 12),
                    "temp": msg.data[6],
                    "err": err_code
                }

# ==========================================
# 5. MAIN CLI LOOP
# ==========================================
def main():
    try:
        bus = can.interface.Bus(channel='can0', bustype='socketcan')
    except Exception as e:
        print(f"Failed to open CAN bus: {e}")
        print("Did you run 'sudo ip link set can0 up type can bitrate 1000000'?")
        sys.exit(1)

    print("Connected to can0. Starting background listener...")
    listener_thread = threading.Thread(target=receive_loop, args=(bus,), daemon=True)
    listener_thread.start()

    print("\nMABEL Ubuntu Teleop CLI")
    print("Commands: SCAN, EN <id>, DIS <id>, CLEAR <id>, M <id> <pos>, STAT <id>, EXIT")

    while True:
        try:
            cmd_input = input(">> ").strip().upper()
            if not cmd_input: continue
            
            parts = cmd_input.split()
            cmd = parts[0]

            if cmd == "EXIT":
                break
            elif cmd == "SCAN":
                scan_bus(bus)
            elif cmd == "EN" and len(parts) == 2:
                send_enable(bus, int(parts[1]))
                print(f"Enabled Motor {parts[1]}")
            elif cmd == "DIS" and len(parts) == 2:
                send_disable(bus, int(parts[1]))
                print(f"Disabled Motor {parts[1]}")
            elif cmd == "CLEAR" and len(parts) == 2:
                send_clear_faults(bus, int(parts[1]))
                print(f"Cleared faults for Motor {parts[1]}")
            elif cmd == "STAT" and len(parts) == 2:
                id_val = int(parts[1])
                if id_val in motor_states:
                    st = motor_states[id_val]
                    print(f"Motor {id_val} | Pos: {st['pos']:.2f} | Vel: {st['vel']:.2f} | Torq: {st['torq']:.2f} | Temp: {st['temp']}C | Err: {hex(st['err'])}")
                else:
                    print(f"No feedback received yet from Motor {id_val}")
            elif cmd == "M" and len(parts) == 3:
                id_val = int(parts[1])
                pos_val = float(parts[2])
                # Impedance control parameters: Kp=30, Kd=1.5
                send_mit_command(bus, id_val, pos_val, 0.0, 30.0, 1.5, 0.0)
                print(f"Commanded Motor {id_val} to {pos_val} rad")
            else:
                print("Invalid command.")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error executing command: {e}")

if __name__ == "__main__":
    main()