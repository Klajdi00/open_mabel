import can
import time
import threading
import sys
import struct
import os

# ==========================================
# 1. GLOBAL STATE & CONFIGURATIONS
# ==========================================
DEFAULT_LIMITS = {"p_max": 12.566, "v_max": 30.0, "t_max": 40.0}
MOTOR_LIMITS = {}
motor_states = {}

# Global bus variable so auto-recovery can hot-swap the socket
bus = None  

# ==========================================
# 2. AUTO-RECOVERY & INITIALIZATION
# ==========================================
def init_bus():
    global bus
    if bus is not None:
        try:
            bus.shutdown()
        except:
            pass
    
    # Force the OS network down, then back up
    os.system("sudo ip link set can0 down 2>/dev/null")
    os.system("sudo ip link set can0 up type can bitrate 1000000")
    
    # Bind the new socket
    bus = can.interface.Bus(interface='socketcan', channel='can0', bitrate=1000000)
    return bus

# ==========================================
# 3. MATH UTILITIES
# ==========================================
def float_to_uint(x, x_min, x_max, bits):
    span = x_max - x_min
    x = max(min(x, x_max), x_min) # Clamp to safe limits
    return int((x - x_min) * ((1 << bits) - 1) / span)

def uint_to_float(x_int, x_min, x_max, bits):
    span = x_max - x_min
    return (x_int * span / ((1 << bits) - 1)) + x_min

# ==========================================
# 4. CAN COMMAND FUNCTIONS
# ==========================================
def send_enable(motor_id):
    msg = can.Message(arbitration_id=motor_id, data=[0xFF]*7 + [0xFC], is_extended_id=False)
    try: bus.send(msg)
    except: pass # Let receive_loop handle the crash

def send_disable(motor_id):
    msg = can.Message(arbitration_id=motor_id, data=[0xFF]*7 + [0xFD], is_extended_id=False)
    try: bus.send(msg)
    except: pass

def send_clear_faults(motor_id):
    msg = can.Message(arbitration_id=motor_id, data=[0xFF]*7 + [0xFB], is_extended_id=False)
    try: bus.send(msg)
    except: pass

def send_zero_position(motor_id):
    msg = can.Message(arbitration_id=motor_id, data=[0xFF]*7 + [0xFE], is_extended_id=False)
    try: bus.send(msg)
    except: pass

def send_change_id(old_id, new_id):
    master_id = new_id + 0x10 
    print(f"\nChanging Motor {old_id} -> CAN_ID: {new_id}, Master_ID: {hex(master_id)}...")
    send_disable(old_id) 
    time.sleep(0.1)

    val_bytes_mst = struct.pack('<I', master_id)
    data_write_mst = [old_id & 0xFF, (old_id >> 8) & 0xFF, 0x55, 0x07, val_bytes_mst[0], val_bytes_mst[1], val_bytes_mst[2], val_bytes_mst[3]]
    bus.send(can.Message(arbitration_id=0x7FF, data=data_write_mst, is_extended_id=False))
    time.sleep(0.1)

    val_bytes_can = struct.pack('<I', new_id)
    data_write_can = [old_id & 0xFF, (old_id >> 8) & 0xFF, 0x55, 0x08, val_bytes_can[0], val_bytes_can[1], val_bytes_can[2], val_bytes_can[3]]
    bus.send(can.Message(arbitration_id=0x7FF, data=data_write_can, is_extended_id=False))
    time.sleep(0.1)

    data_save = [new_id & 0xFF, (new_id >> 8) & 0xFF, 0xAA, 0x01, 0x00, 0x00, 0x00, 0x00]
    bus.send(can.Message(arbitration_id=0x7FF, data=data_save, is_extended_id=False))
    time.sleep(0.2)
    print("Success. Power cycle the motor NOW to apply changes.")

def send_mit_command(motor_id, p_des, v_des, kp, kd, t_ff):
    if motor_id not in MOTOR_LIMITS:
        print(f"Error: Motor {motor_id} not registered. Run SCAN first.")
        return

    limits = MOTOR_LIMITS[motor_id]
    p_int = float_to_uint(p_des, -limits["p_max"], limits["p_max"], 16)
    v_int = float_to_uint(v_des, -limits["v_max"], limits["v_max"], 12)
    kp_int = float_to_uint(kp, 0.0, 500.0, 12)
    kd_int = float_to_uint(kd, 0.0, 5.0, 12)
    t_int = float_to_uint(t_ff, -limits["t_max"], limits["t_max"], 12)

    data = [
        (p_int >> 8) & 0xFF, p_int & 0xFF,
        (v_int >> 4) & 0xFF,
        ((v_int & 0xF) << 4) | ((kp_int >> 8) & 0xF), kp_int & 0xFF,
        (kd_int >> 4) & 0xFF,
        ((kd_int & 0xF) << 4) | ((t_int >> 8) & 0xF), t_int & 0xFF
    ]

    msg = can.Message(arbitration_id=motor_id, data=data, is_extended_id=False)
    try: bus.send(msg)
    except: pass

def scan_bus():
    print("\n--- Sweeping CAN Bus ---")
    MOTOR_LIMITS.clear()
    motor_states.clear()

    for motor_id in range(1, 16):
        print(f"\rPinging Motor ID {motor_id:02d}...", end="", flush=True)
        msg = can.Message(
            arbitration_id=0x7FF, 
            data=[motor_id & 0xFF, (motor_id >> 8) & 0xFF, 0x33, 0x00], 
            is_extended_id=False
        )
        try: bus.send(msg)
        except: pass
        time.sleep(0.02)
        
    print("\rPing sequence complete. Waiting for responses...   ")
    time.sleep(0.5) 
    
    if MOTOR_LIMITS:
        print("\nActive Motors Registered:")
        for m_id in sorted(MOTOR_LIMITS.keys()):
            print(f"  -> Motor ID: {m_id}")
    else:
        print("\nNo motors responded. Check wiring and power.")
    print("--- Sweep Complete ---\n")

# ==========================================
# 5. BACKGROUND THREAD WITH AUTO-RECOVERY
# ==========================================
def receive_loop():
    global bus
    while True:
        try:
            if bus is None:
                time.sleep(0.5)
                continue
                
            msg = bus.recv(timeout=0.5)
            if msg is None: continue

            if msg.arbitration_id != 0x7FF and len(msg.data) == 8:
                # 1. Scanner Responses
                if msg.data[2] == 0x33 and msg.data[3] == 0x00:
                    resp_id = msg.data[0] | (msg.data[1] << 8)
                    if resp_id not in MOTOR_LIMITS:
                        MOTOR_LIMITS[resp_id] = DEFAULT_LIMITS
                    continue

                # 2. Standard Feedback
                sender_id = msg.data[0]
                if sender_id in MOTOR_LIMITS:
                    limits = MOTOR_LIMITS[sender_id]
                    err_code = msg.data[1] >> 4
                    p_int = (msg.data[1] << 8) | msg.data[2]
                    v_int = (msg.data[3] << 4) | (msg.data[4] >> 4)
                    t_int = ((msg.data[4] & 0xF) << 8) | msg.data[5]

                    motor_states[sender_id] = {
                        "pos": uint_to_float(p_int, -limits["p_max"], limits["p_max"], 16),
                        "vel": uint_to_float(v_int, -limits["v_max"], limits["v_max"], 12),
                        "torq": uint_to_float(t_int, -limits["t_max"], limits["t_max"], 12),
                        "temp": msg.data[6],
                        "err": err_code
                    }

        except can.exceptions.CanOperationError as e:
            if "Network is down" in str(e):
                print("\n\n[FAULT] EMI Spike Detected. CAN Bus crashed.")
                print("[RECOVERY] Automatically resetting network interfaces...")
                try:
                    init_bus()
                    print("[RECOVERY] Success. CAN interface restored. Re-type your command.\n>> ", end="", flush=True)
                except Exception as ex:
                    print(f"Recovery failed: {ex}")
                    time.sleep(2)
        except Exception:
            pass

# ==========================================
# 6. MAIN CLI LOOP
# ==========================================
def main():
    try:
        init_bus()
        print("Connected to can0. Starting background listener...")
    except Exception as e:
        print("Did you run 'sudo ip link set can0 up type can bitrate 1000000'?")
        sys.exit(1)

    listener_thread = threading.Thread(target=receive_loop, daemon=True)
    listener_thread.start()

    print("\nMABEL Professional FOC Teleop CLI")
    print("Commands:")
    print("  SCAN                     : Find connected motors")
    print("  EN <id>                  : Enable motor")
    print("  DIS <id>                 : Disable motor")
    print("  ZERO <id>                : Set current position to 0")
    print("  CLEAR <id>               : Clear faults")
    print("  ID <old> <new>           : Change Motor ID via CAN")
    print("  M <id> <pos> [kp] [kd]   : MIT move (e.g., M 1 1.5 30.0 0.5)")
    print("  STAT <id>                : Print motor state")
    print("  EXIT                     : Safely disable and quit")

    while True:
        try:
            cmd_input = input("\n>> ").strip().upper()
            if not cmd_input: continue
            
            parts = cmd_input.split()
            cmd = parts[0]

            if cmd == "EXIT":
                print("Disabling all motors...")
                for m_id in MOTOR_LIMITS.keys():
                    send_disable(m_id)
                break
            elif cmd == "SCAN":
                scan_bus()
            elif cmd == "EN" and len(parts) == 2:
                send_enable(int(parts[1]))
                print(f"Enabled Motor {parts[1]}")
            elif cmd == "DIS" and len(parts) == 2:
                send_disable(int(parts[1]))
                print(f"Disabled Motor {parts[1]}")
            elif cmd == "ZERO" and len(parts) == 2:
                send_zero_position(int(parts[1]))
                print(f"Zeroed Motor {parts[1]}")
            elif cmd == "CLEAR" and len(parts) == 2:
                send_clear_faults(int(parts[1]))
                print(f"Cleared faults for Motor {parts[1]}")
            elif cmd == "ID" and len(parts) == 3:
                send_change_id(int(parts[1]), int(parts[2]))
            elif cmd == "STAT" and len(parts) == 2:
                id_val = int(parts[1])
                if id_val in motor_states:
                    st = motor_states[id_val]
                    print(f"Motor {id_val} | Pos: {st['pos']:.2f} | Vel: {st['vel']:.2f} | Torq: {st['torq']:.2f} | Temp: {st['temp']}C | Err: {hex(st['err'])}")
                else:
                    print(f"No feedback received yet from Motor {id_val}. Try SCAN.")
            elif cmd == "M" and len(parts) >= 3:
                id_val = int(parts[1])
                pos_val = float(parts[2])
                kp_val = float(parts[3]) if len(parts) > 3 else 30.0
                kd_val = float(parts[4]) if len(parts) > 4 else 0.5
                
                send_mit_command(id_val, pos_val, 0.0, kp_val, kd_val, 0.0)
                print(f"Commanded Motor {id_val} to {pos_val} rad (Kp:{kp_val} Kd:{kd_val})")
            else:
                print("Invalid syntax. Review the command list.")
                
        except KeyboardInterrupt:
            print("\nEmergency Stop. Disabling motors...")
            for m_id in MOTOR_LIMITS.keys():
                send_disable(m_id)
            break
        except ValueError:
            print("Error: Invalid numerical argument provided.")
        except Exception as e:
            print(f"Error executing command: {e}")

if __name__ == "__main__":
    main()