from pathlib import Path
import os
import sys
import time
import select
import tty
import termios
import serial
import serial.tools.list_ports
import threading
import json

# Enable ANSI escape codes for Windows Terminal
if os.name == 'nt':
    os.system('')

# ============================================================================
# --- 0. UI COLORS ---
# ============================================================================
class C:
    HEADER    = '\033[95m'
    BLUE      = '\033[94m'
    CYAN      = '\033[96m'
    GREEN     = '\033[92m'
    WARNING   = '\033[93m'
    FAIL      = '\033[91m'
    END       = '\033[0m'
    BOLD      = '\033[1m'

# ============================================================================
# --- 1. KINEMATIC MAPPING CONFIGURATION ---
# ============================================================================

CALIB_FILE = str(Path(__file__).resolve().parent.parent / "data" / "gestures" / "quick_grasp_limits.json")

FINGER_GROUPS = {
    "1": {"name": "THUMB",  "ids": [13, 14, 15, 16]},
    "2": {"name": "INDEX",  "ids": [1, 2, 3]},
    "3": {"name": "MIDDLE", "ids": [4, 8, 9]},
    "4": {"name": "RING",   "ids": [10, 11, 12]},
    "5": {"name": "PINKY",  "ids": [5, 6, 7]}
}

FINGER_MAPPING = {
    1: "Index_PIP", 2: "Index_MCP_Flex", 3: "Index_MCP_Abd",
    4: "Middle_MCP_Abd", 5: "Pinky_MCP_Abd", 6: "Pinky_PIP",
    7: "Pinky_MCP_Flex", 8: "Middle_PIP", 9: "Middle_MCP_Flex",
    10: "Ring_MCP_Flex", 11: "Ring_PIP", 12: "Ring_MCP_Abd",
    13: "Thumb_CMC_Flex", 14: "Thumb_CMC_Abd", 15: "Thumb_MP",
    16: "Thumb_IP", 17: "Wrist"
}

# --- System Configuration ---
BAUDRATE = 1000000
BROADCAST_ID = 254
STEP_SPEED = 0.5  

ADDR_TORQUE_ENABLE    = 0x28
ADDR_GOAL_ACC         = 0x29
ADDR_GOAL_POSITION    = 0x2A
ADDR_GOAL_SPEED       = 0x2E
ADDR_TORQUE_LIMIT     = 0x30
ADDR_PRESENT_POSITION = 0x38

# --- THE COMPLIANT GRASP FIX ---
# Lowered from 1000 to prevent Overload Protection from tripping when grabbing objects
GRASP_TORQUE = 500  

# ============================================================================
# --- 2. HARDWARE & UTILITIES ---
# ============================================================================

def auto_detect_port():
    ports = serial.tools.list_ports.comports()
    keywords = ['usb', 'ch340', 'cp210', 'ftdi', 'uart', 'serial', 'usbmodem']
    for port in ports:
        if any(kw in port.description.lower() or kw in port.device.lower() for kw in keywords):
            return port.device
    sys.exit(f"{C.FAIL}Error: Could not find a suitable port.{C.END}")

def load_limits():
    if not os.path.exists(CALIB_FILE):
        return None
    with open(CALIB_FILE, 'r') as f:
        raw_data = json.load(f)
    limits = {}
    for str_id, data in raw_data.items():
        limits[int(str_id)] = data
    return limits

def save_limits(limits):
    with open(CALIB_FILE, 'w') as f:
        json.dump(limits, f, indent=4)

def write_feetech_register(ser_conn, servo_id, address, data_bytes):
    try:
        instruction = 0x03 
        length = len(data_bytes) + 3 
        checksum_sum = servo_id + length + instruction + address + sum(data_bytes)
        packet = [0xFF, 0xFF, servo_id, length, instruction, address] + data_bytes + [(~checksum_sum) & 0xFF]
        ser_conn.write(bytearray(packet))
    except serial.SerialException:
        pass

def sync_write_positions(ser_conn, positions_dict):
    address = ADDR_GOAL_POSITION
    data_len = 2
    instruction = 0x83 
    
    params = []
    for dxl_id, pos in positions_dict.items():
        params.extend([dxl_id, pos & 0xFF, (pos >> 8) & 0xFF])
            
    if not params: return
    
    length = len(params) + 4
    checksum_sum = BROADCAST_ID + length + instruction + address + data_len + sum(params)
    packet = [0xFF, 0xFF, BROADCAST_ID, length, instruction, address, data_len] + params + [(~checksum_sum) & 0xFF]
    
    try:
        ser_conn.write(bytearray(packet))
        ser_conn.reset_input_buffer()
    except serial.SerialException:
        pass

def read_feetech_position(ser_conn, servo_id, retries=2):
    for _ in range(retries):
        ser_conn.reset_input_buffer()
        packet = [0xFF, 0xFF, servo_id, 4, 0x02, ADDR_PRESENT_POSITION, 2, (~(servo_id + 4 + 0x02 + ADDR_PRESENT_POSITION + 2)) & 0xFF]
        ser_conn.write(bytearray(packet))
        
        t0 = time.perf_counter()
        buffer = bytearray()
        
        while time.perf_counter() - t0 < 0.015: 
            if ser_conn.in_waiting > 0:
                buffer.extend(ser_conn.read(ser_conn.in_waiting))
            while len(buffer) >= 6: 
                if buffer[0] == 0xFF and buffer[1] == 0xFF and buffer[2] == servo_id:
                    pkt_len = buffer[3]
                    total_pkt_size = pkt_len + 4 
                    if len(buffer) >= total_pkt_size:
                        if pkt_len == 4: 
                            raw_pos = buffer[5] + (buffer[6] << 8)
                            return -(raw_pos & 0x7FFF) if (raw_pos & 0x8000) else raw_pos
                        else:
                            buffer = buffer[total_pkt_size:]
                    else:
                        break
                else:
                    buffer.pop(0)
    return None

def ping_feetech(ser_conn, servo_id):
    ser_conn.reset_input_buffer()
    packet = [0xFF, 0xFF, servo_id, 2, 0x01, (~(servo_id + 2 + 0x01)) & 0xFF]
    ser_conn.write(bytearray(packet))
    t0 = time.time()
    while time.time() - t0 < 0.02: 
        if ser_conn.in_waiting >= 6:
            resp = ser_conn.read(6)
            if resp[0] == 0xFF and resp[1] == 0xFF and resp[2] == servo_id: return True
    return False

def get_terminal_key():
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
        elif c in ['\n', '\r']: return 'ENTER'
        return c.lower()
    return None

def clear_line(msg):
    sys.stdout.write("\033[K") 
    sys.stdout.write(f"\r{msg}")
    sys.stdout.flush()

# ============================================================================
# --- 3. QUICK CALIBRATION (OPEN -> FIST) ---
# ============================================================================

def quick_calibrate(ser, active_ids):
    print(f"\n{C.HEADER}{C.BOLD}--- QUICK CALIBRATION ---{C.END}")
    print(f"{C.GREEN}[+] Disabling torque... Hand is limp.{C.END}")
    write_feetech_register(ser, BROADCAST_ID, ADDR_TORQUE_ENABLE, [0])
    time.sleep(0.5)
    
    input(f"\n{C.CYAN}1. Move hand to the FULLY OPEN position and press [ENTER]...{C.END}")
    open_pos = {}
    for dxl_id in active_ids:
        pos = read_feetech_position(ser, dxl_id)
        if pos is not None:
            open_pos[dxl_id] = pos
            sys.stdout.write(f"\r  -> Read ID {dxl_id:02d}: {pos:04d}   ")
            sys.stdout.flush()
    print(f"\n{C.GREEN}  [*] Saved {len(open_pos)} Open joints.{C.END}")
    
    input(f"\n{C.CYAN}2. Move hand to the FULLY CLOSED (FIST) position and press [ENTER]...{C.END}")
    fist_pos = {}
    for dxl_id in active_ids:
        pos = read_feetech_position(ser, dxl_id)
        if pos is not None:
            fist_pos[dxl_id] = pos
            sys.stdout.write(f"\r  -> Read ID {dxl_id:02d}: {pos:04d}   ")
            sys.stdout.flush()
    print(f"\n{C.GREEN}  [*] Saved {len(fist_pos)} Fist joints.{C.END}")
    
    limits = {}
    for dxl_id in active_ids:
        if dxl_id in open_pos and dxl_id in fist_pos:
            limits[str(dxl_id)] = {
                "name": FINGER_MAPPING.get(dxl_id, "Unknown"),
                "open": open_pos[dxl_id],
                "fist": fist_pos[dxl_id]
            }
            
    save_limits(limits)
    print(f"\n{C.GREEN}{C.BOLD}[*] Calibration saved to {CALIB_FILE}!{C.END}")
    time.sleep(1.5)
    return load_limits()

# ============================================================================
# --- 4. 200Hz KINEMATIC ENGINE ---
# ============================================================================

running = True

def teleop_engine(ser, limits):
    global running
    
    print(f"\n{C.CYAN}[+] Initializing Safe Start Sequence...{C.END}")
    
    current_positions = {}
    for dxl_id in limits.keys():
        pos = read_feetech_position(ser, dxl_id)
        if pos is not None:
            current_positions[dxl_id] = pos
            write_feetech_register(ser, dxl_id, ADDR_GOAL_POSITION, [pos & 0xFF, (pos >> 8) & 0xFF])
            
    finger_states = {"THUMB": 0.5, "INDEX": 0.5, "MIDDLE": 0.5, "RING": 0.5, "PINKY": 0.5}
    for f_key, group in FINGER_GROUPS.items():
        f_name = group["name"]
        raw_states = []
        for dxl_id in group["ids"]:
            if dxl_id in limits and dxl_id in current_positions:
                s_open = limits[dxl_id]["open"]
                s_fist = limits[dxl_id]["fist"]
                curr = current_positions[dxl_id]
                
                if s_fist != s_open:
                    st = (curr - s_open) / (s_fist - s_open)
                    raw_states.append(max(0.0, min(1.0, st)))
                    
        if raw_states:
            finger_states[f_name] = sum(raw_states) / len(raw_states)

    print(f"{C.CYAN}[+] Applying COMPLIANT GRASP Limits ({GRASP_TORQUE})...{C.END}")
    write_feetech_register(ser, BROADCAST_ID, ADDR_TORQUE_LIMIT, [GRASP_TORQUE & 0xFF, (GRASP_TORQUE >> 8) & 0xFF])
    write_feetech_register(ser, BROADCAST_ID, ADDR_GOAL_ACC, [40])
    write_feetech_register(ser, BROADCAST_ID, ADDR_GOAL_SPEED, [0, 0]) 
    
    print(f"{C.GREEN}[+] Engaging Torque...{C.END}")
    write_feetech_register(ser, BROADCAST_ID, ADDR_TORQUE_ENABLE, [1])

    if 17 in limits:
        wrist_hold = limits[17]["open"]
        write_feetech_register(ser, 17, ADDR_GOAL_POSITION, [wrist_hold & 0xFF, (wrist_hold >> 8) & 0xFF])

    old_term_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())
    
    ui_state = "TELEOP"
    selected_mode = "ALL"
    direction = 0           
    last_key_time = time.time()
    last_print_time = time.time()
    loop_dt = 0.005 # 200Hz
    
    current_targets = {}
    
    print(f"\n{C.BOLD}--- TELEOP ACTIVE ---{C.END}")
    print(" [UP/W] Close  |  [DOWN/S] Open  |  [ENTER] Select Finger  |  [Q] Quit\n")
    
    try:
        while running:
            loop_start = time.perf_counter()
            key = get_terminal_key()
            
            if ui_state == "MENU":
                if key:
                    if key in ['a', 'A', 'enter']:
                        selected_mode = "ALL"
                        ui_state = "TELEOP"
                    elif key in FINGER_GROUPS:
                        selected_mode = FINGER_GROUPS[key]["name"]
                        ui_state = "TELEOP"
                    elif key == 'q' or key == '\x03':
                        running = False
                        break

            elif ui_state == "TELEOP":
                if key:
                    if key in ['UP', 'RIGHT', 'w']:
                        direction = 1 
                        last_key_time = time.time()
                    elif key in ['DOWN', 'LEFT', 's']:
                        direction = -1 
                        last_key_time = time.time()
                    elif key == 'ENTER':
                        direction = 0 
                        ui_state = "MENU"
                    elif key == 'q' or key == '\x03':
                        running = False
                        break

                if time.time() - last_key_time > 0.1:
                    direction = 0

                if direction != 0:
                    delta = direction * (STEP_SPEED * loop_dt)
                    if selected_mode == "ALL":
                        for f_name in finger_states:
                            finger_states[f_name] = max(0.0, min(1.0, finger_states[f_name] + delta))
                    else:
                        finger_states[selected_mode] = max(0.0, min(1.0, finger_states[selected_mode] + delta))

                sync_packet_data = {}
                for f_key, group in FINGER_GROUPS.items():
                    f_name = group["name"]
                    state = finger_states[f_name]
                    
                    for dxl_id in group["ids"]:
                        if dxl_id in limits:
                            s_open = limits[dxl_id]["open"]
                            s_fist = limits[dxl_id]["fist"]
                            
                            target_pos = s_open + (s_fist - s_open) * state
                            int_pos = int(target_pos)
                            
                            if current_targets.get(dxl_id) != int_pos:
                                sync_packet_data[dxl_id] = int_pos
                                current_targets[dxl_id] = int_pos
                
                if sync_packet_data:
                    sync_write_positions(ser, sync_packet_data)

            if time.time() - last_print_time > 0.1:
                if ui_state == "MENU":
                    clear_line(f"{C.WARNING}Select Mode: [A]ll | [1]Thumb | [2]Index | [3]Middle | [4]Ring | [5]Pinky: {C.END}")
                elif selected_mode == "ALL":
                    clear_line(f"{C.GREEN}Mode: [ALL] | Target: {finger_states['INDEX']*100:02.0f}% FIST{C.END}")
                else:
                    clear_line(f"{C.BLUE}Mode: [{selected_mode}] | Target: {finger_states[selected_mode]*100:02.0f}% FIST{C.END}")
                last_print_time = time.time()

            elapsed = time.perf_counter() - loop_start
            time.sleep(max(0, loop_dt - elapsed))

    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_term_settings)
        write_feetech_register(ser, BROADCAST_ID, ADDR_TORQUE_ENABLE, [0]) 
        print(f"\n\n{C.GREEN}Hardware offline.{C.END}")

def main():
    os.system('cls' if os.name == 'nt' else 'clear')
    print(f"{C.HEADER}{C.BOLD}==========================================={C.END}")
    print(f"{C.HEADER}{C.BOLD}   ORCA HAND KINEMATIC MAPPING ENGINE      {C.END}")
    print(f"{C.HEADER}{C.BOLD}==========================================={C.END}")
    
    com_port = auto_detect_port()
    try:
        ser = serial.Serial(com_port, BAUDRATE, timeout=0) 
    except Exception as e:
        sys.exit(f"{C.FAIL}Failed to open port: {e}{C.END}")

    print(f"{C.CYAN}Scanning for active servos...{C.END}")
    active_ids = [i for i in range(1, 21) if ping_feetech(ser, i)]
    if not active_ids: 
        sys.exit(f"{C.FAIL}Error: No motors detected on the bus.{C.END}")

    limits = load_limits()
    
    if limits is None:
        print(f"\n{C.WARNING}No saved limits found.{C.END}")
        limits = quick_calibrate(ser, active_ids)
    else:
        print(f"\nFound saved calibration data with {len(limits)} joints.")
        choice = input(f"{C.CYAN}Press [1] to Recalibrate (Open -> Fist), or [ENTER] to use saved data: {C.END}").strip()
        if choice == '1':
            limits = quick_calibrate(ser, active_ids)

    t = threading.Thread(target=teleop_engine, args=(ser, limits))
    t.start()
    t.join()

if __name__ == '__main__':
    main()