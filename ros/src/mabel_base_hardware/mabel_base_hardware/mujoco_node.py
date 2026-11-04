import rclpy
from rclpy.node import Node
import mujoco
import mujoco.viewer
import numpy as np
import math
import threading
import time
import os

from mabel_interfaces.msg import MabelSwerveCmd
from sensor_msgs.msg import JointState

class MabelMujocoSim(Node):
    def __init__(self):
        super().__init__('mabel_mujoco_sim')
        
        # 1. Load the Simulation
        xml_path = os.path.expanduser('~/open_mabel/simulation/mabel_mujoco/componments/test_scene.xml')
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.physics_lock = threading.Lock() # Protects data during multithreading
        
        # 2. ROS Network Setup
        self.sub = self.create_subscription(MabelSwerveCmd, '/mabel_cmd', self.cmd_cb, 10)
        self.joint_pub = self.create_publisher(JointState, '/joint_states', 10)
        
        # 3. Timers (Separating AI, Network, and Human speeds)
        self.state_timer = self.create_timer(0.02, self.publish_joint_states)  # 50 Hz Data Logging
        self.log_timer = self.create_timer(0.5, self.print_telemetry)          # 2 Hz Human Dashboard
        
        # 4. Physical Limits & Targets
        self.WHEEL_RADIUS = 0.05
        self.MAX_LIN_ACCEL, self.MAX_ANG_ACCEL = 1.5, 4.0  
        self.target_vx, self.target_vy, self.target_omega, self.target_lift = 0.0, 0.0, 0.0, 0.0
        self.current_vx, self.current_vy, self.current_omega = 0.0, 0.0, 0.0
        
        self.MODULE_POSITIONS = {
            'fl': np.array([0.2, 0.2]),
            'fr': np.array([0.2, -0.2]),
            'bc': np.array([-0.2, 0.0])
        }

        # 5. Cache IDs for Lightning-Fast Lookups
        self.actuators = {}
        self.joints = {}
        
        for key in self.MODULE_POSITIONS.keys():
            self.actuators[f'steer_{key}'] = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_steer_{key}")
            self.actuators[f'drive_{key}'] = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_drive_{key}")
            
            # Save memory addresses for the Joint State Publisher
            j_steer = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"steer_{key}")
            j_drive = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"drive_{key}")
            self.joints[f'steer_{key}'] = {'qpos': self.model.jnt_qposadr[j_steer], 'qvel': self.model.jnt_dofadr[j_steer]}
            self.joints[f'drive_{key}'] = {'qpos': self.model.jnt_qposadr[j_drive], 'qvel': self.model.jnt_dofadr[j_drive]}
            
        self.actuators['lift'] = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_lift")
        j_lift = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "slide_middle")
        self.joints['lift'] = {'qpos': self.model.jnt_qposadr[j_lift], 'qvel': self.model.jnt_dofadr[j_lift]}

        # 6. Boot Physics Thread
        self.running = True
        self.thread = threading.Thread(target=self.physics_loop)
        self.thread.start()
        self.get_logger().info("🌍 MuJoCo Data Bridge Online. Broadcasting state at 50Hz.")

    def cmd_cb(self, msg: MabelSwerveCmd):
        if msg.hardware_state == 1:
            self.target_vx, self.target_vy, self.target_omega = 0.0, 0.0, 0.0
            return
        self.target_vx, self.target_vy, self.target_omega = msg.vx, msg.vy, msg.omega
        self.target_lift = max(0.0, min(0.4, msg.lift_pos))

    def publish_joint_states(self):
        """ Broadcasts the exact physical state of MuJoCo to the ROS Network """
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        
        with self.physics_lock: # Pause physics for a microsecond to read safe data
            for name, adr in self.joints.items():
                msg.name.append(name)
                msg.position.append(float(self.data.qpos[adr['qpos']]))
                msg.velocity.append(float(self.data.qvel[adr['qvel']]))
                
        self.joint_pub.publish(msg)

    def print_telemetry(self):
        """ Clean, non-spammy terminal dashboard updated 2x a second """
        with self.physics_lock:
            lift_pos = self.data.qpos[self.joints['lift']['qpos']]
            fl_speed = self.data.qvel[self.joints['drive_fl']['qvel']]
            
        print(f"\r[TELEMETRY] Cmd Vel: {self.current_vx:.2f} m/s | Lift Pos: {lift_pos:.3f} m | FL Wheel Spd: {fl_speed:.2f} rad/s    ", end="")

    def optimize_steering(self, target_angle, current_angle):
        diff = (target_angle - current_angle + math.pi) % (2 * math.pi) - math.pi
        dir_mult = 1.0
        if diff > math.pi / 2: target_angle -= math.pi; dir_mult = -1.0
        elif diff < -math.pi / 2: target_angle += math.pi; dir_mult = -1.0
        return current_angle + ((target_angle - current_angle + math.pi) % (2 * math.pi) - math.pi), dir_mult

    def swerve_ik(self):
        for key, r in self.MODULE_POSITIONS.items():
            v_ix = self.current_vx - self.current_omega * r[1]
            v_iy = self.current_vy + self.current_omega * r[0]
            v_mag = math.sqrt(v_ix**2 + v_iy**2)
            
            current_theta = self.data.qpos[self.joints[f'steer_{key}']['qpos']]
            
            if v_mag > 1e-3:
                raw_target_theta = math.atan2(v_iy, v_ix)
                optimal_theta, dir_mult = self.optimize_steering(raw_target_theta, current_theta)
                
                angle_error = abs(optimal_theta - current_theta)
                drive_scale = max(0.0, math.cos(angle_error)) if angle_error <= 0.45 else 0.0
                
                self.data.ctrl[self.actuators[f'steer_{key}']] = optimal_theta
                self.data.ctrl[self.actuators[f'drive_{key}']] = (v_mag / self.WHEEL_RADIUS) * dir_mult * drive_scale
            else:
                self.data.ctrl[self.actuators[f'steer_{key}']] = current_theta
                self.data.ctrl[self.actuators[f'drive_{key}']] = 0.0

    def physics_loop(self):
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while viewer.is_running() and self.running and rclpy.ok():
                step_start = time.time()
                dt = self.model.opt.timestep
                
                # Kinematic Ramping
                if self.current_vx < self.target_vx: self.current_vx = min(self.target_vx, self.current_vx + self.MAX_LIN_ACCEL * dt)
                elif self.current_vx > self.target_vx: self.current_vx = max(self.target_vx, self.current_vx - self.MAX_LIN_ACCEL * dt)
                if self.current_vy < self.target_vy: self.current_vy = min(self.target_vy, self.current_vy + self.MAX_LIN_ACCEL * dt)
                elif self.current_vy > self.target_vy: self.current_vy = max(self.target_vy, self.current_vy - self.MAX_LIN_ACCEL * dt)
                if self.current_omega < self.target_omega: self.current_omega = min(self.target_omega, self.current_omega + self.MAX_ANG_ACCEL * dt)
                elif self.current_omega > self.target_omega: self.current_omega = max(self.target_omega, self.current_omega - self.MAX_ANG_ACCEL * dt)

                with self.physics_lock:
                    self.swerve_ik()
                    self.data.ctrl[self.actuators['lift']] = self.target_lift
                    mujoco.mj_step(self.model, self.data)
                
                viewer.sync()
                
                time_until_next_step = dt - (time.time() - step_start)
                if time_until_next_step > 0: time.sleep(time_until_next_step)
                    
    def destroy_node(self):
        self.running = False
        self.thread.join()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = MabelMujocoSim()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__': main()