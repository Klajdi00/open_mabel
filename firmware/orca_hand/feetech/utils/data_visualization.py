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
import csv
from datetime import datetime
from collections import deque

try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
except ImportError:
    sys.exit("Error: matplotlib is required for plotting. Run: pip install matplotlib")

# --- System Configuration ---
BAUDRATE            = 1000000         
BROADCAST_ID        = 254 

# --- Trajectory Settings ---
MIN_POS             = 0    
MAX_POS             = 4095 
MAX_SPEED           = 3000  
MIN_SPEED           = 100   
SPEED_STEP          = 200   

# --- Feetech Addresses ---
ADDR_TORQUE_ENABLE    = 0x28
ADDR_GOAL_ACC         = 0x29
ADDR_GOAL_POSITION    = 0x2A
ADDR_GOAL_SPEED       = 0x2E
ADDR_PRESENT_POSITION = 0x38 

# --- Global State ---
running = True
selected_mode = "ALL"
gui_key = None  
global_speed = 1000

# Telemetry Buffers
time_data = deque(maxlen=400)
pos_data = deque(maxlen=400)
vel_data = deque(maxlen=400)
load_data = deque(maxlen=400)
start_time = time.time()

# CSV Logging Setup
_LOGS = Path(__file__).resolve().parent.parent / "data" / "logs"
_LOGS.mkdir(parents=True, exist_ok=True)
log_filename = str(_LOGS / f"feetech_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
csv_file = None
csv_writer = None

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
    instruction = 0x03 
    length = len(data_bytes) + 3 
    checksum_sum = servo_id + length + instruction + address + sum(data_bytes)
    packet = [0xFF, 0xFF, servo_id, length, instruction, address] + data_bytes + [(~checksum_sum) & 0xFF]
    ser_conn.write(bytearray(packet))

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

# =========================================================================
# THE FIX: ROBUST PACKET PARSER
# =========================================================================
def read_feetech_feedback(ser_conn, servo_id, retries=3):
    """Reads telemetry and automatically discards leftover Write ACKs."""
    for _ in range(retries):
        ser_conn.reset_input_buffer()
        packet = [0xFF, 0xFF, servo_id, 4, 0x02, ADDR_PRESENT_POSITION, 6, (~(servo_id + 4 + 0x02 + ADDR_PRESENT_POSITION + 6)) & 0xFF]
        ser_conn.write(bytearray(packet))
        
        t0 = time.perf_counter()
        buffer = bytearray()
        
        while time.perf_counter() - t0 < 0.015: # 15ms timeout
            if ser_conn.in_waiting > 0:
                buffer.extend(ser_conn.read(ser_conn.in_waiting))
                
            # Minimum size of any Feetech packet is 6 bytes
            while len(buffer) >= 6: 
                # Check for the header and matching ID
                if buffer[0] == 0xFF and buffer[1] == 0xFF and buffer[2] == servo_id:
                    pkt_len = buffer[3]
                    total_pkt_size = pkt_len + 4 # 4 bytes for Header(2) + ID(1) + Length(1)
                    
                    if len(buffer) >= total_pkt_size:
                        # We received a full packet!
                        if pkt_len == 8: 
                            # 8 length = 6 data bytes + 1 Error + 1 Checksum. This is our Telemetry!
                            pos = buffer[5] + (buffer[6] << 8)
                            v_raw = buffer[7] + (buffer[8] << 8)
                            vel = (v_raw & 0x7FFF) * (-1 if (v_raw & (1 << 15)) else 1)
                            t_raw = buffer[9] + (buffer[10] << 8)
                            load = (t_raw & 0x3FF) * (-1 if (t_raw & (1 << 10)) else 1)
                            return pos, vel, load
                        else:
                            # It is a Write ACK (pkt_len=2) leftover in the stream. Discard it.
                            buffer = buffer[total_pkt_size:]
                    else:
                        # Packet is valid but hasn't fully arrived yet. Wait for more bytes.
                        break
                else:
                    # Misaligned junk byte (from the crash). Pop it off and shift the window.
                    buffer.pop(0)
                    
    return None, None, None
# =========================================================================


def get_terminal_key():
    global gui_key
    if gui_key:
        k = gui_key
        gui_key = None
        return k
        
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
        elif c in ['\x7f', '\b']: return 'BACKSPACE'
        return c.lower()
    return None

def clear_line(msg):
    sys.stdout.write("\033[K") 
    sys.stdout.write(f"\r{msg}")
    sys.stdout.flush()

# --- BACKGROUND THREAD: 200Hz Control Loop ---
def teleop_engine(ser, active_ids):
    global running, selected_mode, global_speed, time_data, pos_data, vel_data, load_data
    global csv_writer

    float_targets = {}
    print("\nReading safe physical startup positions...")
    for dxl_id in active_ids:
        pos, _, _ = read_feetech_feedback(ser, dxl_id, retries=10)
        if pos is not None:
            float_targets[dxl_id] = float(pos)
            print(f" -> ID {dxl_id} locked at position: {pos}")
        else:
            float_targets[dxl_id] = 2047.0

    print("\nConfiguring Robot Hand Safety Profile...")
    write_feetech_register(ser, BROADCAST_ID, ADDR_GOAL_ACC, [30]) 
    write_feetech_register(ser, BROADCAST_ID, ADDR_GOAL_SPEED, [0, 0])
    write_feetech_register(ser, BROADCAST_ID, ADDR_TORQUE_ENABLE, [1])

    old_term_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())
    
    ui_state = "TELEOP"
    id_buffer = ""
    direction = 0           
    last_key_time = time.time()
    last_print_time = time.time()
    loop_dt = 0.005 # 200Hz
    
    try:
        while running:
            loop_start = time.perf_counter()
            key = get_terminal_key()
            
            # --- STATE: TYPING ID ---
            if ui_state == "INPUTTING_ID":
                if key:
                    if key == 'ENTER':
                        if id_buffer == "":
                            selected_mode = "ALL"
                        elif id_buffer.isdigit() and int(id_buffer) in active_ids:
                            selected_mode = int(id_buffer)
                        else:
                            selected_mode = "ALL"
                            
                        time_data.clear(); pos_data.clear(); vel_data.clear(); load_data.clear()
                        ui_state = "TELEOP"
                        id_buffer = ""
                    elif key.isdigit():
                        id_buffer += key
                    elif key == 'BACKSPACE':
                        id_buffer = id_buffer[:-1]
                    elif key == 'q' or key == '\x03':
                        running = False
                        break

            # --- STATE: DRIVING ---
            elif ui_state == "TELEOP":
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
                        ui_state = "INPUTTING_ID"
                        id_buffer = ""
                    elif key == 'q' or key == '\x03':
                        running = False
                        break

                if time.time() - last_key_time > 0.1:
                    direction = 0

                if direction != 0:
                    delta = direction * (global_speed * loop_dt)
                    if selected_mode == "ALL":
                        for dxl_id in active_ids:
                            float_targets[dxl_id] = max(MIN_POS, min(MAX_POS, float_targets[dxl_id] + delta))
                            int_pos = int(float_targets[dxl_id])
                            write_feetech_register(ser, dxl_id, ADDR_GOAL_POSITION, [int_pos & 0xFF, (int_pos >> 8) & 0xFF])
                    else:
                        float_targets[selected_mode] = max(MIN_POS, min(MAX_POS, float_targets[selected_mode] + delta))
                        int_pos = int(float_targets[selected_mode])
                        write_feetech_register(ser, selected_mode, ADDR_GOAL_POSITION, [int_pos & 0xFF, (int_pos >> 8) & 0xFF])

                # Read Feedback for Live Plots & CSV Logging
                if selected_mode != "ALL":
                    p, v, t = read_feetech_feedback(ser, selected_mode)
                    if p is not None:
                        current_t = time.time() - start_time
                        
                        # Update Graph Buffers
                        time_data.append(current_t)
                        pos_data.append(p)
                        vel_data.append(v)
                        load_data.append(t)
                        
                        # Write to CSV
                        if csv_writer is not None:
                            csv_writer.writerow([f"{current_t:.4f}", selected_mode, p, v, t])

            # --- Decoupled UI Terminal Printing ---
            if time.time() - last_print_time > 0.1:
                if ui_state == "INPUTTING_ID":
                    clear_line(f"Enter Motor ID (or press Enter for ALL): {id_buffer}")
                elif selected_mode == "ALL":
                    clear_line(f"[ALL] Mode Active | Speed: {global_speed:04d} u/s")
                else:
                    clear_line(f"[ID {selected_mode:02d}] Live Telemetry Active | Target: {int(float_targets[selected_mode]):04d} | Speed: {global_speed:04d}")
                last_print_time = time.time()

            # Precise 200Hz Timing
            elapsed = time.perf_counter() - loop_start
            time.sleep(max(0, loop_dt - elapsed))

    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_term_settings)
        write_feetech_register(ser, BROADCAST_ID, ADDR_TORQUE_ENABLE, [0])
        ser.close()
        print("\n\nHardware offline. Close the plot window to fully exit.")


# --- MATPLOTLIB GUI EVENT HANDLER ---
def on_gui_keypress(event):
    global gui_key
    key_map = {'up': 'UP', 'down': 'DOWN', 'right': 'RIGHT', 'left': 'LEFT', 'enter': 'ENTER', 'q': 'q', 'w': 'w', 's': 's', 'backspace': 'BACKSPACE'}
    if event.key in key_map:
        gui_key = key_map[event.key]
    elif event.key and event.key.isdigit():
        gui_key = event.key

# --- MAIN EXECUTABLE ---
def main():
    global running, start_time, csv_file, csv_writer
    
    com_port = auto_detect_port()
    try:
        ser = serial.Serial(com_port, BAUDRATE, timeout=0) 
    except Exception as e:
        sys.exit(f"Failed to open port: {e}")

    active_ids = [i for i in range(1, 21) if ping_feetech(ser, i)]
    if not active_ids: sys.exit("No motors detected.")

    # Initialize CSV Logging
    try:
        csv_file = open(log_filename, mode='w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Time (s)", "Motor ID", "Position", "Velocity", "Torque Load"])
        print(f"-> Logging Telemetry to: {log_filename}")
    except Exception as e:
        print(f"Warning: Could not open CSV for logging: {e}")

    print("\n--- 200Hz SAFE TELEOP & LIVE TELEMETRY ---")
    print("[ RIGHT / LEFT ] : Move Forward / Backward")
    print("[ W / S ]        : Adjust Speed")
    print("[ ENTER ]        : Select Motor ID to activate plotting & logging")
    print("[ Q ]            : Quit\n")

    start_time = time.time()
    t = threading.Thread(target=teleop_engine, args=(ser, active_ids))
    t.daemon = True
    t.start()

    plt.style.use('dark_background')
    fig, (ax_pos, ax_vel, ax_trq) = plt.subplots(3, 1, figsize=(8, 7))
    fig.canvas.manager.set_window_title("Feetech Live Telemetry")
    fig.canvas.mpl_connect('key_press_event', on_gui_keypress)

    line_pos, = ax_pos.plot([], [], 'c-', lw=2)
    line_vel, = ax_vel.plot([], [], 'm-', lw=2)
    line_trq, = ax_trq.plot([], [], 'y-', lw=2)

    ax_pos.set_ylabel("Position (0-4095)")
    ax_vel.set_ylabel("Velocity (u/s)")
    ax_trq.set_ylabel("Torque Load (%)")
    ax_trq.set_xlabel("Time (s)")

    def animate(frame):
        if not running:
            plt.close(fig)
            return
            
        if selected_mode == "ALL" or len(time_data) < 2:
            ax_pos.set_title("Waiting for Single ID Selection...", color='gray')
            return line_pos, line_vel, line_trq
            
        ax_pos.set_title(f"Live Telemetry: Motor ID {selected_mode} (Logging Active)", color='white')
        t_list = list(time_data)
        
        line_pos.set_data(t_list, list(pos_data))
        line_vel.set_data(t_list, list(vel_data))
        line_trq.set_data(t_list, list(load_data))

        current_time = t_list[-1]
        ax_pos.set_xlim(current_time - 2, current_time + 0.1)
        ax_vel.set_xlim(current_time - 2, current_time + 0.1)
        ax_trq.set_xlim(current_time - 2, current_time + 0.1)

        ax_pos.set_ylim(0, 4095)
        ax_vel.set_ylim(-3500, 3500)
        ax_trq.set_ylim(-1050, 1050)

        return line_pos, line_vel, line_trq

    # Run the plot
    try:
        ani = animation.FuncAnimation(fig, animate, interval=30, blit=False, cache_frame_data=False)
        plt.tight_layout()
        plt.show()
    finally:
        # Guarantee the CSV file saves correctly when closing the app
        if csv_file is not None:
            csv_file.close()
            print(f"Data successfully saved to {log_filename}")

if __name__ == '__main__':
    main()