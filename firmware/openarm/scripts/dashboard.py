import can
import time
import threading
import struct
import sys
import cmd
import os

# ==========================================
# 1. CONSTANTS & UI COLORS
# ==========================================
GLOBAL_ID = 0x7FF

# ANSI Colors for UI
class C:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    CYAN = '\033[96m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

# Damiao Hardware Status Map
STATUS_MAP = {
    0x0: f"{C.YELLOW}DISABLED{C.RESET}",
    0x1: f"{C.GREEN}ENABLED {C.RESET}",
    0x2: f"{C.RED}ERR: ROTOR ENC{C.RESET}",
    0x3: f"{C.RED}ERR: OUT ENC{C.RESET}",
    0x8: f"{C.RED}ERR: OVER_VOLT{C.RESET}",
    0x9: f"{C.RED}ERR: UNDER_VOLT{C.RESET}",
    0xA: f"{C.RED}ERR: OVER_CURR{C.RESET}",
    0xB: f"{C.RED}ERR: MOS_TEMP{C.RESET}",
    0xC: f"{C.RED}ERR: COIL_TEMP{C.RESET}",
    0xD: f"{C.RED}ERR: COMM_LOSS{C.RESET}",
    0xE: f"{C.RED}ERR: OVERLOAD{C.RESET}"
}

# Standard MABEL limits (Adjust per motor via Damiao App if needed)
DEFAULT_LIMITS = {"p_max": 12.566, "v_max": 30.0, "t_max": 40.0}

# ==========================================
# 2. MOTOR DATA CLASS
# ==========================================
class DamiaoMotor:
    def __init__(self, motor_id, limits=DEFAULT_LIMITS):
        self.motor_id = motor_id
        self.limits = limits
        # Default impedance tuning parameters for this session
        self.kp = 30.0 
        self.kd = 1.5  
        self.state = {
            "pos": 0.0, "vel": 0.0, "torq": 0.0, 
            "t_mos": 0.0, "t_rotor": 0.0, "status_code": 0
        }

    def update_state(self, msg_data):
        self.state["status_code"] = msg_data[1] >> 4
        
        p_int = (msg_data[1] << 8) | msg_data[2]  
        v_int = (msg_data[3] << 4) | (msg_data[4] >> 4)
        t_int = ((msg_data[4] & 0xF) << 8) | msg_data[5]

        self.state["pos"] = self._uint_to_float(p_int, -self.limits["p_max"], self.limits["p_max"], 16)
        self.state["vel"] = self._uint_to_float(v_int, -self.limits["v_max"], self.limits["v_max"], 12)
        self.state["torq"] = self._uint_to_float(t_int, -self.limits["t_max"], self.limits["t_max"], 12)
        self.state["t_mos"] = msg_data[6]
        self.state["t_rotor"] = msg_data[7]

    def _uint_to_float(self, x_int, x_min, x_max, bits):
        span = x_max - x_min
        return (x_int * span / ((1 << bits) - 1)) + x_min

# ==========================================
# 3. CAN NETWORK CONTROLLER
# ==========================================
class DamiaoNetwork:
    def __init__(self, interface='can0'):
        try:
            self.bus = can.interface.Bus(channel=interface, bustype='socketcan', bitrate=1000000)
        except Exception as e:
            print(f"{C.RED}FATAL: Could not open {interface}. Did you run 'sudo ip link set can0 up'?{C.RESET}")
            sys.exit(1)
            
        self.motors = {}
        self.running = True
        self.listener_thread = threading.Thread(target=self._rx_loop, daemon=True)
        self.listener_thread.start()

    def _rx_loop(self):
        while self.running:
            msg = self.bus.recv(timeout=0.5)
            if msg is None: continue

            # FIX: Damiao motors reply to 0x7FF reads using their Master ID (default 0x000)
            if msg.arbitration_id == 0 and len(msg.data) == 8:
                
                # 1. Intercept Scanner Responses (Register Read Echo)
                if msg.data[2] == 0x33:
                    resp_id = msg.data[0] | (msg.data[1] << 8)
                    # Dynamically register the motor so commands work immediately
                    if resp_id not in self.motors:
                        self.motors[resp_id] = DamiaoMotor(resp_id)
                    continue

                # 2. Intercept Standard Feedback Frames
                sender_id = msg.data[0]
                if sender_id in self.motors:
                    self.motors[sender_id].update_state(msg.data)

    def _send_command(self, motor_id, cmd_byte):
        data = [0xFF]*7 + [cmd_byte]
        self.bus.send(can.Message(arbitration_id=motor_id, data=data, is_extended_id=False))

    def enable(self, motor_id): self._send_command(motor_id, 0xFC)
    def disable(self, motor_id): self._send_command(motor_id, 0xFD)
    def clear_faults(self, motor_id): self._send_command(motor_id, 0xFB)
    def zero_position(self, motor_id): self._send_command(motor_id, 0xFE)

    def change_motor_id(self, old_id, new_id):
        """Changes ID in RAM, then writes to Flash"""
        print(f"Changing ID from {old_id} to {new_id}...")
        self.disable(old_id) # Must be disabled to save
        time.sleep(0.1)

        # 1. Write new ID to ESC_ID Register (0x08)
        val_bytes = struct.pack('<I', new_id) # 32-bit unsigned int, little endian
        data_write = [
            old_id & 0xFF, (old_id >> 8) & 0xFF, 
            0x55, 0x08, 
            val_bytes[0], val_bytes[1], val_bytes[2], val_bytes[3]
        ]
        self.bus.send(can.Message(arbitration_id=GLOBAL_ID, data=data_write, is_extended_id=False))
        time.sleep(0.1)

        # 2. Save Parameters to Flash
        data_save = [old_id & 0xFF, (old_id >> 8) & 0xFF, 0xAA, 0x01]
        self.bus.send(can.Message(arbitration_id=GLOBAL_ID, data=data_save, is_extended_id=False))
        time.sleep(0.1) # Wait for flash write
        
        # 3. Update internal state
        if old_id in self.motors:
            self.motors[new_id] = self.motors.pop(old_id)
            self.motors[new_id].motor_id = new_id
        
        print(f"{C.GREEN}Success. Please power cycle the motor to verify.{C.RESET}")

    def send_mit(self, motor_id, p_des, v_des, kp, kd, t_ff):
        if motor_id not in self.motors: return
        lim = self.motors[motor_id].limits
        
        def float_to_uint(x, x_min, x_max, bits):
            x = max(min(x, x_max), x_min)
            return int((x - x_min) * ((1 << bits) - 1) / (x_max - x_min))

        p_int = float_to_uint(p_des, -lim["p_max"], lim["p_max"], 16)
        v_int = float_to_uint(v_des, -lim["v_max"], lim["v_max"], 12)
        kp_int = float_to_uint(kp, 0.0, 500.0, 12)
        kd_int = float_to_uint(kd, 0.0, 5.0, 12)
        t_int = float_to_uint(t_ff, -lim["t_max"], lim["t_max"], 12)

        data = [
            (p_int >> 8) & 0xFF, p_int & 0xFF,
            (v_int >> 4) & 0xFF,
            ((v_int & 0xF) << 4) | ((kp_int >> 8) & 0xF), kp_int & 0xFF,
            (kd_int >> 4) & 0xFF,
            ((kd_int & 0xF) << 4) | ((t_int >> 8) & 0xF), t_int & 0xFF
        ]
        self.bus.send(can.Message(arbitration_id=motor_id, data=data, is_extended_id=False))

# ==========================================
# 4. INTUITIVE CLI DASHBOARD
# ==========================================
class MabelCLI(cmd.Cmd):
    prompt = f"{C.CYAN}mabel-cli > {C.RESET}"

    def __init__(self, network):
        super().__init__()
        self.net = network
        self.do_clear(None) 
        self.print_header()

    def print_header(self):
        print(f"{C.BOLD}{C.BLUE}===================================================={C.RESET}")
        print(f"{C.BOLD}{C.BLUE}             MABEL MOTOR CONTROL CENTER             {C.RESET}")
        print(f"{C.BOLD}{C.BLUE}===================================================={C.RESET}")
        print(" Type 'help' for commands. Type 'monitor' for live dashboard.")

    def do_scan(self, arg):
        """Sweep bus to find connected motors."""
        print(f"{C.YELLOW}Sweeping CAN bus for motors...{C.RESET}")
        for i in range(1, 16):
            print(f"\r{C.CYAN}Pinging Motor ID {i:02d}...{C.RESET}", end="", flush=True)
            data = [i & 0xFF, (i >> 8) & 0xFF, 0x33, 0x00]
            self.net.bus.send(can.Message(arbitration_id=GLOBAL_ID, data=data, is_extended_id=False))
            time.sleep(0.02)
        
        print(f"\r{C.GREEN}Ping sequence complete. Waiting for responses...{C.RESET}")
        time.sleep(0.5) 
        self.do_stat(None)

    def do_stat(self, arg):
        """Print state of all known motors."""
        if not self.net.motors:
            print(f"{C.RED}No motors found. Run 'scan' or check wiring.{C.RESET}")
            return
        
        print(f"\n{C.BOLD}{'ID':<4} | {'STATUS':<15} | {'POS (rad)':<10} | {'VEL (rad/s)':<12} | {'TORQ (Nm)':<10} | {'TEMP (C)'}{C.RESET}")
        print("-" * 75)
        for m_id, m in sorted(self.net.motors.items()):
            st = m.state
            status_str = STATUS_MAP.get(st['status_code'], f"{C.RED}UNKNOWN ({hex(st['status_code'])}){C.RESET}")
            print(f"{m_id:<4} | {status_str:<24} | {st['pos']:>9.2f} | {st['vel']:>11.2f} | {st['torq']:>9.2f} | {st['t_rotor']:>6.0f}°C")
        print()

    def do_monitor(self, arg):
        """Open a live-updating dashboard. Press Ctrl+C to exit."""
        if not self.net.motors:
            print(f"{C.RED}No motors found. Run 'scan' first.{C.RESET}")
            return
            
        print(f"{C.YELLOW}Entering live monitor mode. Press Ctrl+C to exit...{C.RESET}")
        time.sleep(1)
        try:
            while True:
                os.system('clear' if os.name != 'nt' else 'cls') 
                self.do_stat(None)
                print(f"{C.YELLOW}[Live Monitoring Active - Press Ctrl+C to return to prompt]{C.RESET}")
                time.sleep(0.1) 
        except KeyboardInterrupt:
            os.system('clear' if os.name != 'nt' else 'cls')
            self.print_header()

    # --- SEMANTIC CONTROL COMMANDS ---

    def do_en(self, arg):
        """Enable motor: en <id>"""
        try: 
            self.net.enable(int(arg))
            print(f"{C.GREEN}>> Motor {arg} ENABLED{C.RESET}")
        except ValueError:
            print("Usage: en <id>")

    def do_dis(self, arg):
        """Disable motor: dis <id>"""
        try: 
            self.net.disable(int(arg))
            print(f"{C.YELLOW}>> Motor {arg} DISABLED{C.RESET}")
        except ValueError:
            print("Usage: dis <id>")

    def do_zero(self, arg):
        """Set current position to 0 (Motor MUST be disabled): zero <id>"""
        try: 
            self.net.zero_position(int(arg))
            print(f"{C.GREEN}>> Motor {arg} Zeroed.{C.RESET}")
        except ValueError:
            print("Usage: zero <id>")

    def do_reset(self, arg):
        """Clear faults (over-temp, over-voltage): reset <id>"""
        try: 
            self.net.clear_faults(int(arg))
            print(f"{C.GREEN}>> Motor {arg} faults cleared.{C.RESET}")
        except ValueError:
            print("Usage: reset <id>")

    def do_change_id(self, arg):
        """Change motor ID via CAN: change_id <old_id> <new_id>"""
        args = arg.split()
        if len(args) != 2:
            print("Usage: change_id <old_id> <new_id>")
            return
        try:
            self.net.change_motor_id(int(args[0]), int(args[1]))
        except ValueError:
            print(f"{C.RED}Invalid arguments. IDs must be integers.{C.RESET}")

    def do_tune(self, arg):
        """Change impedance gains for a motor: tune <id> <kp> <kd>"""
        args = arg.split()
        if len(args) != 3:
            print("Usage: tune <id> <kp> <kd>  (e.g., tune 1 30.0 1.5)")
            return
        try:
            m_id = int(args[0])
            if m_id in self.net.motors:
                self.net.motors[m_id].kp = float(args[1])
                self.net.motors[m_id].kd = float(args[2])
                print(f"{C.GREEN}>> Motor {m_id} tuned: Kp={args[1]}, Kd={args[2]}{C.RESET}")
            else:
                print(f"{C.RED}Motor {m_id} not found.{C.RESET}")
        except ValueError:
            print("Invalid numerical arguments.")

    def do_pos(self, arg):
        """Move to position using current impedance gains: pos <id> <radians>"""
        args = arg.split()
        if len(args) != 2:
            print("Usage: pos <id> <radians> (e.g., pos 1 1.5)")
            return
        try:
            m_id, pos = int(args[0]), float(args[1])
            if m_id in self.net.motors:
                m = self.net.motors[m_id]
                self.net.send_mit(m_id, pos, 0.0, m.kp, m.kd, 0.0)
                print(f">> Sent Pos: {pos} rad to Motor {m_id}")
            else:
                print(f"{C.RED}Motor {m_id} not found.{C.RESET}")
        except ValueError:
            print("Invalid numerical arguments.")

    def do_vel(self, arg):
        """Command constant velocity: vel <id> <rad/s>"""
        args = arg.split()
        if len(args) != 2:
            print("Usage: vel <id> <rad/s> (e.g., vel 1 5.0)")
            return
        try:
            m_id, vel = int(args[0]), float(args[1])
            if m_id in self.net.motors:
                m = self.net.motors[m_id]
                self.net.send_mit(m_id, 0.0, vel, 0.0, m.kd, 0.0)
                print(f">> Sent Vel: {vel} rad/s to Motor {m_id}")
            else:
                print(f"{C.RED}Motor {m_id} not found.{C.RESET}")
        except ValueError:
            print("Invalid numerical arguments.")

    def do_torq(self, arg):
        """Command feedforward torque: torq <id> <Nm>"""
        args = arg.split()
        if len(args) != 2:
            print("Usage: torq <id> <Nm> (e.g., torq 1 2.5)")
            return
        try:
            m_id, torq = int(args[0]), float(args[1])
            if m_id in self.net.motors:
                self.net.send_mit(m_id, 0.0, 0.0, 0.0, 0.0, torq)
                print(f">> Sent Torque: {torq} Nm to Motor {m_id}")
            else:
                print(f"{C.RED}Motor {m_id} not found.{C.RESET}")
        except ValueError:
            print("Invalid numerical arguments.")

    def do_exit(self, arg):
        """Exit the application safely."""
        print(f"{C.YELLOW}Disabling all motors and shutting down...{C.RESET}")
        for m_id in self.net.motors.keys():
            self.net.disable(m_id)
        self.net.running = False
        self.net.listener_thread.join()
        return True

if __name__ == '__main__':
    net = DamiaoNetwork('can0')
    try:
        MabelCLI(net).cmdloop()
    except KeyboardInterrupt:
        print(f"\n{C.YELLOW}Emergency Stop Detected via Ctrl+C. Disabling motors...{C.RESET}")
        for m_id in net.motors.keys():
            net.disable(m_id)
        net.running = False
        sys.exit(0)