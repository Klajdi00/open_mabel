import os
import sys
import time
import select
import tty
import termios
import serial
import serial.tools.list_ports

# --- System Configuration ---
BAUDRATE            = 1000000         
BROADCAST_ID        = 254 

# --- Trajectory Generator Settings ---
MIN_POS             = 0    
MAX_POS             = 4095 
MAX_SPEED           = 3000  # Maximum units per second
MIN_SPEED           = 100   # Minimum units per second
SPEED_STEP          = 200   # How much speed changes per W/S press

# --- Feetech Addresses ---
ADDR_TORQUE_ENABLE    = 0x28
ADDR_GOAL_ACC         = 0x29
ADDR_GOAL_POSITION    = 0x2A
ADDR_GOAL_SPEED       = 0x2E
ADDR_PRESENT_POSITION = 0x38

def auto_detect_port():
    ports = serial.tools.list_ports.comports()
    keywords = ['usb', 'ch340', 'cp210', 'ftdi', 'uart', 'serial', 'usbmodem']
    for port in ports:
        if any(kw in port.description.lower() or kw in port.device.lower() for kw in keywords):
            return port.device
    for port in ports:
        if 'bluetooth' not in port.description.lower():
            return port.device
    sys.exit("Error: Could not find a suitable port.")

def write_feetech_register(ser_conn, servo_id, address, data_bytes):
    """Raw byte writer - minimized for max speed"""
    instruction = 0x03 
    length = len(data_bytes) + 3 
    checksum_sum = servo_id + length + instruction + address + sum(data_bytes)
    packet = [0xFF, 0xFF, servo_id, length, instruction, address] + data_bytes + [(~checksum_sum) & 0xFF]
    ser_conn.write(bytearray(packet))

def ping_feetech(ser_conn, servo_id):
    ser_conn.reset_input_buffer()
    packet = [0xFF, 0xFF, servo_id, 2, 0x01, (~(servo_id + 2 + 0x01)) & 0xFF]
    ser_conn.write(bytearray(packet))
    start_time = time.time()
    while time.time() - start_time < 0.02: 
        if ser_conn.in_waiting >= 6:
            resp = ser_conn.read(6)
            if resp[0] == 0xFF and resp[1] == 0xFF and resp[2] == servo_id: return True
    return False

def read_feetech_position(ser_conn, servo_id):
    ser_conn.reset_input_buffer()
    packet = [0xFF, 0xFF, servo_id, 4, 0x02, ADDR_PRESENT_POSITION, 2, (~(servo_id + 4 + 0x02 + ADDR_PRESENT_POSITION + 2)) & 0xFF]
    ser_conn.write(bytearray(packet))
    start_time = time.time()
    while time.time() - start_time < 0.05:
        if ser_conn.in_waiting >= 8:
            resp = ser_conn.read(8)
            if resp[0] == 0xFF and resp[1] == 0xFF: return resp[5] + (resp[6] << 8)
    return None

def get_key():
    if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
        c = sys.stdin.read(1)
        if c == '\x1b':
            c2 = sys.stdin.read(1)
            c3 = sys.stdin.read(1)
            if c2 == '[':
                if c3 == 'A': return 'UP'
                elif c3 == 'B': return 'DOWN'
                elif c3 == 'C': return 'RIGHT'
                elif c3 == 'D': return 'LEFT'
        elif c == '\n' or c == '\r': return 'ENTER'
        return c.lower()
    return None

def main():
    com_port = auto_detect_port()
    try:
        ser = serial.Serial(com_port, BAUDRATE, timeout=0) 
    except Exception as e:
        sys.exit(f"Failed to open port: {e}")

    print("Scanning for Feetech motors (IDs 1-20)...")
    active_ids = [i for i in range(1, 21) if ping_feetech(ser, i)]
    if not active_ids: sys.exit("No motors detected.")
    print(f"Active Motor Array: {active_ids}")

    # To calculate exact velocity mathematically, we store float positions
    float_targets = {}
    for dxl_id in active_ids:
        pos = read_feetech_position(ser, dxl_id)
        float_targets[dxl_id] = float(pos) if pos is not None else 2047.0

    print("Configuring 200Hz Mathematical Velocity Stream...")
    write_feetech_register(ser, BROADCAST_ID, ADDR_GOAL_ACC, [0])
    write_feetech_register(ser, BROADCAST_ID, ADDR_GOAL_SPEED, [0, 0])
    write_feetech_register(ser, BROADCAST_ID, ADDR_TORQUE_ENABLE, [1])

    selected_mode = "ALL"
    global_speed = 1000 # Starting speed (Units per second)
    
    print("\n--- 200Hz VELOCITY-CONTROLLED TELEOP ---")
    print("[ RIGHT / LEFT ] : Move Forward / Backward")
    print("[ W / S ]        : Increase / Decrease Speed")
    print("[ ENTER ]        : Change Motor ID")
    print("[ Q ]            : Quit\n")

    old_term_settings = termios.tcgetattr(sys.stdin)
    
    try:
        tty.setcbreak(sys.stdin.fileno())
        
        direction = 0           
        last_key_time = time.time()
        last_print_time = time.time()
        loop_dt = 0.005 # 200Hz target time
        
        while True:
            loop_start = time.perf_counter()
            
            # 1. Poll the Keyboard
            key = get_key()
            if key:
                if key == 'RIGHT':
                    direction = 1
                    last_key_time = time.time()
                elif key == 'LEFT':
                    direction = -1
                    last_key_time = time.time()
                elif key in ['w', 'up']:
                    global_speed = min(global_speed + SPEED_STEP, MAX_SPEED)
                elif key in ['s', 'down']:
                    global_speed = max(global_speed - SPEED_STEP, MIN_SPEED)
                elif key == 'ENTER':
                    direction = 0 
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_term_settings)
                    user_input = input("\nEnter Motor ID (or press Enter for ALL): ").strip()
                    selected_mode = int(user_input) if user_input.isdigit() and int(user_input) in active_ids else "ALL"
                    print(f"-> Mode set to: {selected_mode}\n")
                    tty.setcbreak(sys.stdin.fileno())
                elif key == 'q' or key == '\x03':
                    break

            # 2. Key Release Detection
            if time.time() - last_key_time > 0.1:
                direction = 0

            # 3. Calculate New Position (Velocity * Time)
            if direction != 0:
                # Math: Distance = Velocity * Time 
                delta = direction * (global_speed * loop_dt)
                
                if selected_mode == "ALL":
                    for dxl_id in active_ids:
                        float_targets[dxl_id] = max(MIN_POS, min(MAX_POS, float_targets[dxl_id] + delta))
                        int_pos = int(float_targets[dxl_id])
                        p_L, p_H = int_pos & 0xFF, (int_pos >> 8) & 0xFF
                        write_feetech_register(ser, dxl_id, ADDR_GOAL_POSITION, [p_L, p_H])
                else:
                    float_targets[selected_mode] = max(MIN_POS, min(MAX_POS, float_targets[selected_mode] + delta))
                    int_pos = int(float_targets[selected_mode])
                    p_L, p_H = int_pos & 0xFF, (int_pos >> 8) & 0xFF
                    write_feetech_register(ser, selected_mode, ADDR_GOAL_POSITION, [p_L, p_H])

            # 4. Decoupled UI Printing (10Hz)
            if time.time() - last_print_time > 0.1:
                if selected_mode == "ALL":
                    sys.stdout.write(f"\r[ALL] Mode | Speed: {global_speed:04d} u/s          ")
                else:
                    int_pos = int(float_targets[selected_mode])
                    sys.stdout.write(f"\r[ID {selected_mode:02d}] Pos: {int_pos:04d} | Speed: {global_speed:04d} u/s      ")
                sys.stdout.flush()
                last_print_time = time.time()

            # 5. Lock loop to 200Hz
            elapsed = time.perf_counter() - loop_start
            time.sleep(max(0, loop_dt - elapsed))

    except Exception as e:
        print(f"\nError: {e}")
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_term_settings)
        write_feetech_register(ser, BROADCAST_ID, ADDR_TORQUE_ENABLE, [0])
        ser.close()

if __name__ == '__main__':
    main()